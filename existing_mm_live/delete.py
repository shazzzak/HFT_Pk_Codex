cd /Users/shazzak/PycharmProjects/HFT && \
echo "=== line counts ===" && \
wc -l existing_mm_live/micro_mm.py existing_mm_live/*PreChange*.py 2>/dev/null && \
echo "=== diff (empty = identical) ===" && \
diff existing_mm_live/micro_mm.py existing_mm_live/*PreChange*.py && echo "IDENTICAL" || echo "^ differences shown above"