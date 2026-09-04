"""
Final ready-leg MM watchlist from the 9-month persistence screen.

Reads persistence_REG_2p00.csv (Stage-2 output) and produces the definitive
two-tier watchlist: real ceiling opportunity AND persistent positive surviving
edge AND a quotable (not near-empty) book. Ranks WITHIN each tier by the axis
that defines it, rather than by a single blended score -- because "best" depends
on deployable capital, which the screen cannot know.
"""
import pandas as pd

# ------------------------------- gates --------------------------------------
MIN_DAYS_TRADED   = 100     # enough history to trust the medians
MIN_OPPORTUNITY   = 0.15    # pct_days_top20: real ceiling-material opportunity
MIN_EDGE_PERSIST  = 0.60    # pct_days_edge5_pos: edge is reliable, not fat-tailed
MAX_SPREAD_BPS    = 50.0    # excludes near-empty books (ZUMA 144, WTL 60)
# Tier split on spread: wide = margin names, tight = volume/ballast names.
WIDE_SPREAD_BPS   = 15.0

COLS = ["symbol", "net5_trec_median", "pct_days_edge5_pos", "pct_days_top20",
        "spread_bps_median", "notional_m_median", "ceiling_median", "days_traded"]

def main():
    p = pd.read_csv("persistence_REG_2p00.csv")

    # Apply all four gates -- opportunity, persistent edge, tradeable book, history.
    wl = p[(p["days_traded"]        >= MIN_DAYS_TRADED)
           & (p["pct_days_top20"]     >= MIN_OPPORTUNITY)
           & (p["pct_days_edge5_pos"] >= MIN_EDGE_PERSIST)
           & (p["spread_bps_median"]  <  MAX_SPREAD_BPS)].copy()

    # Deployable-edge proxy: per-fill edge scaled by how much volume carries it.
    # This is the capital-aware view the raw 'score' lacked -- bps x daily notional.
    wl["edge_x_notional"] = wl["net5_trec_median"] * wl["notional_m_median"]

    # Two tiers by spread regime.
    fat  = wl[wl["spread_bps_median"] >= WIDE_SPREAD_BPS].sort_values(
        "net5_trec_median", ascending=False)
    tight = wl[wl["spread_bps_median"] < WIDE_SPREAD_BPS].sort_values(
        "edge_x_notional", ascending=False)

    show = COLS + ["edge_x_notional"]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"WATCHLIST: {len(wl)} names "
              f"(gates: days>={MIN_DAYS_TRADED}, opp>={MIN_OPPORTUNITY}, "
              f"edge_persist>={MIN_EDGE_PERSIST}, spread<{MAX_SPREAD_BPS}bps)\n")
        print("=" * 100)
        print(f"TIER A -- FAT-EDGE / WIDE-SPREAD (>= {WIDE_SPREAD_BPS}bps)  "
              f"[margin names, ranked by per-fill edge]")
        print("=" * 100)
        print(fat[show].round(2).to_string(index=False))
        print("\n" + "=" * 100)
        print(f"TIER B -- TIGHT-SPREAD (< {WIDE_SPREAD_BPS}bps)  "
              f"[volume/ballast names, ranked by edge x notional]")
        print("=" * 100)
        print(tight[show].round(2).to_string(index=False))

    # One combined CSV for downstream use, tier-tagged.
    fat["tier"] = "A_fat_edge"; tight["tier"] = "B_volume"
    out = pd.concat([fat, tight])[["tier"] + show]
    out.to_csv("mm_watchlist_final.csv", index=False)
    print(f"\nwrote mm_watchlist_final.csv ({len(out)} names)")

if __name__ == "__main__":
    main()
