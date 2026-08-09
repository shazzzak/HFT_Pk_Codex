# run_micro_ubl.py — run as: python run_micro_ubl.py
import mm_backtest
assert mm_backtest.USE_TREC_FEE is True and abs(mm_backtest.FEE_TOTAL_PCT - 7.77e-05) < 1e-6
print("fee confirmed:", mm_backtest.FEE_TOTAL_PCT)

# import the micro strategy + the same loader/runner you used for naive,
# then run UBL 2026-06-30 through micro_mm instead of NaiveSymmetricMM,
# and print the result dict.