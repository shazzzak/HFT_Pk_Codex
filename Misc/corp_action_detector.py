"""
Corporate-action detector for multi-day PSX runs.

PURPOSE
-------
Each trading day, PSX disseminates a `prev_close` per symbol (in the snapshot
rows). Compare it against the ACTUAL close YOU recorded the previous session:

    * match            -> normal day, no action
    * ratio near a
      round factor      -> SPLIT (ratio<1) / REVERSE-SPLIT (ratio>1);
                           `factor` = price multiplier (old_px * factor = new scale)
    * small residual
      after ratio ~1    -> DIVIDEND               (price drops by ~cash amount)
    * large unexplained -> FLAG for manual review (rights issue, bonus, error)

WHY IT MATTERS (the contamination it prevents)
----------------------------------------------
Intraday replay uses RAW prices and needs NO adjustment (actions take effect
overnight). But any CROSS-DAY feature/label -- rolling vol, overnight returns,
multi-day VPIN buckets, fair-value training targets -- sees a phantom jump on
the action date unless corrected. This detector writes one row per
(symbol, date) into a registry so every downstream consumer knows which
symbol-days need adjusted handling, WITHOUT needing an external calendar.

SELF-BOOTSTRAPPING
------------------
Maintains its own close-history CSV. First time it sees a symbol, there is no
baseline -> status 'no_baseline' (not an error). From then on it compares.

USAGE
-----
    det = CorpActionDetector("corp_action_registry.csv", "close_history.csv")
    # for each day, in chronological order:
    det.process_day(date_str, snapshot_df, trades_df)
    # -> appends detections to the registry; updates close history
    det.save()

Then read `corp_action_registry.csv` in your feature pipeline and apply the
factor (splits: price*=factor, volume/=factor; dividends: additive) to any
window crossing an action date.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# round split factors to test against (both directions)
_SPLIT_FACTORS = [10, 5, 4, 3, 2, 1.5]                 # forward: 1-for-N style
_SPLIT_FACTORS += [1/f for f in _SPLIT_FACTORS]        # reverse
_RATIO_TOL = 0.02          # +/-2% to call a ratio a "round" split factor
_DIV_MAX_PCT = 0.15        # a same-day drop up to 15% via prev_close reset -> dividend
_FLAG_PCT = 0.15           # unexplained move beyond this -> manual review


def _ms(series):
    return (pd.to_datetime(series, utc=True, format="ISO8601")
            .dt.as_unit("ns").astype("int64") // 1_000_000)


def actual_close(trades_df):
    """Actual session close per symbol = last non-auction trade price by time.
    (If your feed has an official closing-price field, prefer that instead.)"""
    t = trades_df.copy()
    t["ts"] = _ms(t["transact_time"])
    t = t[t["initiator"] != "AUCTION"].sort_values("ts")
    if t.empty:
        return {}
    last = t.groupby("symbol").tail(1)
    return dict(zip(last["symbol"], last["price"].astype(float)))


def feed_prev_close(snapshot_df):
    """prev_close the feed reported this morning, per symbol (already action-adjusted
    by the exchange on ex-dates)."""
    s = snapshot_df[["symbol", "prev_close"]].dropna(subset=["prev_close"])
    return dict(zip(s["symbol"], s.groupby("symbol")["prev_close"].transform("first")
                    .groupby(s["symbol"]).first().astype(float)))


def _classify(my_close, feed_prev):
    """Return (status, factor, note) comparing my recorded close to the feed's
    prev_close for the SAME symbol on the next day."""
    if my_close is None or feed_prev is None or my_close <= 0 or feed_prev <= 0:
        return "no_data", np.nan, ""
    ratio = feed_prev / my_close           # = price_factor: today_ref = yesterday_close * ratio
    # 1) round split factor? Label by PRICE DIRECTION:
    #    ratio < 1 -> price fell overnight -> FORWARD split (e.g. 2-for-1: price halves)
    #    ratio > 1 -> price rose overnight -> REVERSE split (e.g. 1-for-5: price 5x)
    #    `factor` returned is the PRICE factor: multiply old prices by it to put
    #    them on the new scale. Volume is divided by it. Use this number, not the word.
    for f in _SPLIT_FACTORS:
        if abs(ratio - f) / f <= _RATIO_TOL:
            kind = "forward_split" if f < 1 else "reverse_split"
            return kind, ratio, f"price_factor={ratio:.4f}~{f}"
    # 2) near 1 -> dividend or nothing
    drop = (my_close - feed_prev) / my_close     # positive = price marked down
    if abs(ratio - 1.0) <= 0.001:
        return "normal", 1.0, ""
    if 0 < drop <= _DIV_MAX_PCT:
        return "dividend", 1.0, f"markdown {drop:.4%} of close (~cash div)"
    # 3) unexplained
    if abs(1 - ratio) > _FLAG_PCT:
        return "review", np.nan, f"unexplained ratio {ratio:.4f}"
    return "minor_diff", 1.0, f"ratio {ratio:.4f} (rounding/thin trading)"


class CorpActionDetector:
    def __init__(self, registry_path, history_path):
        self.registry_path = Path(registry_path)
        self.history_path = Path(history_path)
        self.registry = []                                   # new detections this run
        # load prior close history {symbol: {date: close}} -> we only need latest
        self.last_close = {}       # symbol -> (date, close) most recent seen
        if self.history_path.exists():
            h = pd.read_csv(self.history_path)
            for r in h.sort_values("date").itertuples():
                self.last_close[r.symbol] = (r.date, float(r.close))

    def process_day(self, date_str, snapshot_df, trades_df):
        """Detect actions for `date_str`, then record today's closes as the new
        baseline. Call in CHRONOLOGICAL order across days."""
        prev = feed_prev_close(snapshot_df)
        closes = actual_close(trades_df)

        for sym, fp in prev.items():
            base = self.last_close.get(sym)
            if base is None:
                status, factor, note = "no_baseline", np.nan, "first day seen"
            else:
                status, factor, note = _classify(base[1], fp)
            if status not in ("normal", "no_baseline", "minor_diff"):
                self.registry.append({
                    "symbol": sym, "date": date_str, "status": status,
                    "factor": factor, "my_prev_close": None if base is None else base[1],
                    "feed_prev_close": fp, "note": note,
                })

        # update baseline with TODAY's actual closes
        for sym, c in closes.items():
            self.last_close[sym] = (date_str, c)

    def save(self):
        if self.registry:
            df = pd.DataFrame(self.registry)
            if self.registry_path.exists():
                df = pd.concat([pd.read_csv(self.registry_path), df], ignore_index=True)
            df.drop_duplicates(subset=["symbol", "date", "status"]).to_csv(
                self.registry_path, index=False)
        # persist full close history for next run
        rows = [{"symbol": s, "date": d, "close": c}
                for s, (d, c) in self.last_close.items()]
        pd.DataFrame(rows).to_csv(self.history_path, index=False)
