cd /Users/shazzak/PycharmProjects/HFT && \
echo "=== run_one return dict (what it hands back -- does it expose fills?) ===" && \
sed -n '233,275p' existing_mm_live/run_legacy_mm.py && \
echo "" && \
echo "=== fill_attribution build_fills_for_partition (the function to replace) ===" && \
sed -n '212,278p' existing_mm/fill_attribution.py