# ============================================================================
# throttle_sweep_recover.py -- rebuild a PERNAME CSV from a checkpoint journal
# if a run DIED before its final write (OOM, PyCharm quit, power loss). Works
# for ANY of the three sweeps' journals:
#   throttle_sweep_CKPT.jsonl     (throttle: OFF/ON via 'thr')
#   skew_sweep_2d_CKPT.jsonl      (Run A lambda: cell via ofw/oth/mlam)
#   skew_sweep_2d_RUNB_CKPT.jsonl (Run B OFI:   cell via ofw/oth/mlam)
# Each journal line holds one completed cell's full per-bucket decomposition, so
# per-name daily results are fully recoverable even when the polished CSV was
# never written.
#
# Usage:  python throttle_sweep_recover.py <path_to_journal>
#   (also accepts the .done marker). Auto-detects the cell-identity schema.
# ============================================================================

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# results dir
RESULTS = Path("/Users/shazzak/Capital Stake - Results")


def main():
    # journal path (arg or auto-detect among the three known journals)
    if len(sys.argv) >= 2:
        jp = Path(sys.argv[1])
    else:
        # look for any live journal; prefer the throttle one, then A, then B
        candidates = ["throttle_sweep_CKPT.jsonl", "skew_sweep_2d_CKPT.jsonl",
                      "skew_sweep_2d_RUNB_CKPT.jsonl"]
        jp = None
        for c in candidates:
            p = RESULTS / c
            pd_ = RESULTS / (c + ".done")
            if p.exists():
                jp = p
                break
            if pd_.exists():
                jp = pd_
                break
        if jp is None:
            print("no journal found. Pass one explicitly:")
            print("  python throttle_sweep_recover.py <path_to_CKPT.jsonl>")
            sys.exit(1)
    # guard: no journal
    if not jp.exists():
        print(f"no journal found at {jp}")
        sys.exit(1)
    print(f"reading journal: {jp}")
    # accumulate rows per cell. The cell identity differs by sweep:
    #   throttle journal -> 'thr'
    #   Run A / Run B     -> ('ofw','oth','mlam')
    # Detect from the first data line and dedupe on whichever is present (last
    # line for a cell wins, in case a resume re-ran an in-flight cell).
    cells = {}
    n_lines = 0
    n_torn = 0
    schema = None
    with open(jp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                r = json.loads(line)
            except ValueError:
                # torn last line from a hard kill -> skip
                n_torn += 1
                continue
            # detect the cell-identity schema once
            if schema is None:
                schema = "thr" if "thr" in r else "cfg"
            # build the dedupe key for this cell
            if schema == "thr":
                ckey = (r["date"], r["sym"], r["thr"])
            else:
                ckey = (r["date"], r["sym"], str(r.get("ofw")),
                        r.get("oth"), str(r.get("mlam")))
            cells[ckey] = r
    print(f"  {n_lines} lines, {len(cells)} unique cells, schema={schema}"
          + (f", {n_torn} torn lines skipped" if n_torn else ""))
    # build the PERNAME rows: one per (cell, bucket) with the full decomposition
    # + the net identity (capture+markout+liq-fee).
    rows = []
    for ckey, r in cells.items():
        for b, per in r["per"].items():
            # net PKR identity, same as the sweep's bucket_net
            net = (per["capture"] + per["markout"]
                   + per["liq_cap"] + per["liq_mko"]
                   - per["fee"] - per["liq_fee"])
            row = {
                "date": r["date"], "symbol": r["sym"],
                "bucket": b,
                "net_pkr": net,
                "capture_pkr": per["capture"],
                "markout_pkr": per["markout"],
                "liq_pkr": per["liq_cap"] + per["liq_mko"],
                "fee_pkr": per["fee"] + per["liq_fee"],
                "opened_notional": per["opened_notional"],
                "fills": per["fills"],
                "net_bps": (1e4 * net / per["opened_notional"]
                            if per["opened_notional"] > 0 else float("nan")),
            }
            # cell-identity column(s) per schema
            if schema == "thr":
                row["throttle"] = "ON" if r["thr"] == 1 else "OFF"
            else:
                row["ofi_window"] = ("OFF" if r.get("ofw") is None
                                     else str(r.get("ofw")))
                row["ofi_thresh"] = r.get("oth")
                row["micro_lambda"] = r.get("mlam")
            rows.append(row)
    # guard: nothing recovered
    if not rows:
        print("no cells recovered from the journal")
        sys.exit(1)
    # write the recovered PERNAME CSV
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    # name the output after the source journal so multiple recoveries don't clash
    tag = jp.name.replace("_CKPT.jsonl", "").replace(".done", "")
    out = RESULTS / f"{tag}_PERNAME_RECOVERED_{stamp}.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"\n  wrote {out}  ({len(df)} rows)")
    # fast headline: portfolio net per cell-identity group
    if schema == "thr":
        gcol = "throttle"
    else:
        gcol = "ofi_window"
    for gval, sub in df.groupby(gcol):
        net = sub["net_pkr"].sum()
        on = sub["opened_notional"].sum()
        bps = 1e4 * net / on if on > 0 else float("nan")
        ndays = sub["date"].nunique()
        nnames = sub["symbol"].nunique()
        print(f"  {gcol}={str(gval):>16s}: net {net:>14,.0f} PKR   {bps:+.3f} bps"
              f"   ({nnames} names x {ndays} days)")
    print("\n  (full paired day-as-unit analysis: load this CSV; same schema as "
          "the sweep's PERNAME_*.csv)")


if __name__ == "__main__":
    main()
