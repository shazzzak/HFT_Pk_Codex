# Corrected P&L package — 26 September 2026

Unpack into `/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/Production` and run `run_corrected_pnl.py` with the existing `HFT_Pk_Codex/backtest/bin/python` interpreter. The entry point works when the terminal is in `existing_mm_live`: it explicitly runs verification, tests and replay from the corrected research directory.

The prior shell launcher let Python's module-based test discovery put the caller's historical directory ahead of `PYTHONPATH`. This loaded historical `clean_window_book.py` and caused the reported 6 failures and 22 errors. That failed attempt stopped during tests before launching P&L. The package corrects the shell launcher and provides a Python launcher with explicit child working directories.

This is an update for the existing machine, not a standalone data distribution. It reuses the installed backtest environment, read-only legacy dependencies, parsed source data, frozen manifest/assignment and existing validation evidence at `Capital Stake - Results Codex/book_reconstruction_validation_20260926_v3/validation.json`. None of those data inputs is replaced by extraction. The validated book Python files are unchanged by this packaging repair.

The Python launcher first verifies the book code and original inputs, then runs all regression tests. Add `--check-only` to stop after these checks. Without that flag, it runs all 113 stocks across 185 October 2025–June 2026 dates and all twelve arms with eight workers. A unique results directory is created and printed; earlier P&L results are preserved. Sleep prevention and limited library threads are configured automatically for child processes.

This remains retrospective clean-window research. Ambiguous snapshot timing and unreconstructible periods remain excluded. The full-history profit impact has not yet been computed by this repair.
