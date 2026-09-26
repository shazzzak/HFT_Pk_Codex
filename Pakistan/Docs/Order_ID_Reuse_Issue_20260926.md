# Issue for independent review: reused order ID falsely invalidates earlier snapshots

Please independently verify this issue against the files and source records below. Keep the active P&L run's code and inputs unchanged. This report distinguishes verified behavior, related defects and remaining uncertainty.

## Precise correction to the description

The verified example is **the same exchange order ID, the same quantity, and a changed price**. Both additions offer **5,000 shares**. The price changes from **120.70 to 120.50**. It is not evidence of a quantity-changing amendment. The recorded sequence is an add, a cancellation referencing that add, then a new add reusing the exchange ID; it is not an observed order-entry Cancel/Replace message.

## Exact faulty files, still present in the historical installation

Base directory: `/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/existing_mm_live`.

| File and lines | Actual behavior |
|---|---|
| `clean_window_data.py`, 118–130 | Builds `adds` by application sequence, preserving the exchange ID in `source['oid']`. This retains both additions initially. |
| `clean_window_data.py`, **191** | `by_oid={value['oid']:value for value in adds.values()}` collapses those separate additions into one entry per exchange ID. The last encountered addition wins. In the verified example, this is the later replacement. |
| `clean_window_data.py`, **219–223** | Tests `by_oid[oid]['ts'] > start` for every disclosed snapshot ID. If any such test is true, increments `FUTURE_ORDER_IN_SNAPSHOT` and rejects the checkpoint. The lookup has already discarded the earlier generation, so a genuinely earlier order can fail this test. |
| `clean_window_data.py`, **197–199**, and **271** | Separate defect: skips a checkpoint when `start < next_start`; the previous attempted reconstruction advanced `next_start` to its end plus one millisecond. Later snapshots inside that interval do not refresh the active book. |
| `clean_window_engine.py`, **76–85** | Separate refresh defect: constructs the replay stream from incremental events and timers, without adding subsequent snapshot-replacement events. |
| `clean_window_book.py`, **197–210** | Its `snapshot()` method updates phase and circuit limits only; it does not replace depth. |
| `clean_window_book.py`, **54–64** | Stores disclosed snapshot orders under their reusable exchange IDs. This is relevant to the identity model, but is not itself proof that every valid cancel/re-add produces wrong quantities. |

These are the assistant-authored **clean-window research modules inside `existing_mm_live`**. This is not a claim that all code in that folder has the same defect. The general `mm_backtest.py` still has snapshot replacement: its run loop calls `self.book.snapshot(...)` and `_on_snapshot_queue_reset()` at lines **2534–2535**.

## Actual exchange evidence

Stock: **SNGP**. Date: **21 January 2026**. Regular-market application channel: **2011**. Reused exchange OrderID/tag 37: **`0010T96R3F0055LY`**.

Times below are exchange transaction times from tag 60, converted from UTC to Pakistan time (UTC+05:00). Capture receipt times are distinct and are retained in the evidence file.

| Pakistan time | Message | Application sequence/tag 1181 | Relevant fields |
|---|---|---:|---|
| 09:40:56.400 | UA201 / add | 180376 | SELL; tag 37=`0010T96R3F0055LY`; tag 44=`120.7000`; tag 38=`5000.00` |
| 10:22:25.780 | UA202 / cancellation | 454949 | tag 150=`4`; tag 10117=`180376` (sell-side add reference); tag 32=`5000.00`; tag 31=`0.0000` |
| 10:22:25.780 | UA201 / replacement add | 454950 | SELL; the same tag 37; tag 44=`120.5000`; tag 38=`5000.00` |

The cancellation does **not** itself carry tag 37 in these recovered bytes. Its application reference ties it to the first add. The identical exchange ID is observed on the two add messages. Same-time cancellation and replacement are ordered by application sequence, 454949 before 454950; do not conflate this with FIX session sequence/tag 34.

Original raw archive inspected read-only: `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/Raw_Capture_Recovery/2026-01-21.tar.gz`.

Full recovered raw messages: `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/book_reconstruction_validation_20260926_v3/raw_generation_records.json`.

## Reproduction against an actual earlier snapshot

A fresh read of the parsed source day on 26 September found a valid-layout, continuous-auction snapshot with:

- Snapshot origin **09:40:58.000 Pakistan time** / 04:40:58 UTC.
- Conservative planner availability **09:40:59.000 Pakistan time** / 04:40:59 UTC.
- SELL level **120.70**, level quantity **5,000**, disclosed ID `0010T96R3F0055LY`, disclosed quantity `5000.00`.
- No symbol mutation in the planner's snapshot-origin-to-availability uncertainty interval.
- The original order already exists: add 180376 occurred at 09:40:56.400.

Nevertheless, the faulty dictionary contains the replacement's timestamp **10:22:25.780**. Its predicate becomes:

`10:22:25.780 > 09:40:59.000` → **true** → `FUTURE_ORDER_IN_SNAPSHOT`.

The snapshot is not reporting an order from the future. The code selected the wrong generation of an ID that existed earlier. This proves the false future predicate; the reproduction does not assert that this checkpoint survives every other planner eligibility rule.

The independent reproduction and exact decoded values are saved at `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/Engine_Gap_Assessment_20260926/order_id_reproduction.json`.

## Why it happened

Reading a whole day's additions to construct an index is not inherently the error. The error is treating a reusable exchange ID as a unique lifetime identity and selecting its final record without a cutoff at the snapshot's time. The add-reference index had both versions; the secondary `by_oid` dictionary lost that distinction.

A routine partial fill or cancellation that reduces the remaining quantity of one order is not the same issue. The evidence here is two distinct add references sharing one exchange ID.

## Impact and limits of the finding

The direct demonstrated impact is false rejection of earlier source checkpoints. That changes which intervals the backtest can reconstruct and trade. Reusable-ID addressing also needs protection against reducing a replacement or anonymous pool using a superseded add reference; that is a separately regression-tested correction.

The snapshot-skipping defect is separate. It prevents refresh from later exchange pictures and can leave reconstruction without known bid or offer depth. Do not attribute every lost minute to ID reuse or combine both defects into one cause without interval-level evidence.

The sign and amount of the P&L impact are not established by this example. Corrected and historical runs can differ both in included periods and executions within those periods. Their total-profit difference alone is not a pure execution-model comparison. Historical exclusions are not zero-profit observations.

## Where the correction is installed

Corrected directory: `/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/Production/clean_window_research`.

- `clean_window_data.py`, **132–134**: each add gets an immutable `ref:2011:<application_sequence>` key within its loaded stock-day.
- `clean_window_identity.py`, **5–34**: `AddIndex` keeps all ID generations; `latest()` resolves the last generation at or before the time/sequence cutoff.
- `clean_window_book.py`, **80–82**: snapshot linkage uses that cutoff plus matching side and price; unlinked quantity remains anonymous at its actual reported price.
- `clean_window_book.py`, **201–203**: a superseded reference cannot consume a later generation's anonymous pool.
- `clean_window_book.py`, **286 onward**: replaces market depth from a validated new snapshot and preserves relevant historical identity evidence.
- `clean_window_data.py`, **299–313**, and `clean_window_engine.py`, **103–131**: plan and apply eligible interior snapshot replacements; simulated cash, inventory and outstanding orders are preserved across refreshes.

This is stock-day/channel-scoped historical identity, not a claim that application sequence alone is globally unique across sessions or venues.

The old `existing_mm_live` copies are unchanged. The ZIP installs corrected research copies under `Production`; `/Users/shazzak/PycharmProjects/HFT_Pk_Codex/Pakistan/Production/run_corrected_pnl.py` launches them. A subsequent launcher fix prevents module-based tests from importing the historical directory merely because the terminal happens to be there.

## Validation already available

The corrected release passed 37 regression tests and 7,393 exact next-snapshot comparisons across five diagnostic stock-days, restricted to shared reported depth. A short TELE interval ran through all twelve strategy arms with seven refreshes each and closed flat. Evidence: `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/book_reconstruction_validation_20260926_v3/validation.json`.

Those are bounded checks, not a full-history certification. The corrected planner still excludes ambiguous snapshot cutovers, source gaps and unreconstructible periods. Anonymous queue priority remains a conservative assumption; exact book totals do not prove simulated queue realism or achievable profit.

## Authorship

The defect exists in the original staged clean-window package produced by the assistant in the **HFT Coding** task during **25 September 2026, 03:41:29–04:15:22 Pakistan time**. These are recorded task-session bounds, not the exact timestamp of this line's creation or the user's installation. The original staged `clean_window_data.py` has the same offending lines. The provenance record is `/Users/shazzak/HFT Data/Pakistan/Capital Stake - Results Codex/snapshot_refresh_validation_20260925_v1/PROVENANCE.md`.

Please review the original and corrected implementations independently, verify the raw-message chain and snapshot cutoff, and keep the separate snapshot-refresh defect distinct from this reused-ID lookup defect. Do not modify the active run's source or input data during review.
