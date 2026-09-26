# Review record — issue report and engine gap assessment

Scope: the complete issue explanation and gap-assessment conclusions were reconsidered in twenty successive review rounds, rotating the primary focus below. These are reasoning/source-review rounds, not twenty independent test runs. The test executions and limits are reported separately. Three misleading starting premises were corrected during investigation and review: the example changed quantity; recovery/account replay integration still needed to be built from scratch; the README could serve as a current inventory. Code and raw evidence contradict those premises.

| Round | Main focus while reconsidering the complete deliverable | Outcome |
|---|---|---|
| 1 | Quantity versus price | Both adds are 5,000 shares; state the price change explicitly. |
| 2 | Dictionary behavior and causal lookup | Describe last encountered addition; prove the actual later-add case rather than generalizing every file's ordering. |
| 3 | Time conversion | Add five hours to tag 60 UTC; distinguish session bounds from exact authorship time. |
| 4 | Snapshot reproduction | Actual origin 09:40:58, availability 09:40:59; false predicate proved, all other eligibility not asserted. |
| 5 | Message identities | Same ID on two adds; cancellation references 180376 and does not carry tag 37 in these bytes. |
| 6 | Price interpretation | Tag 31 zero on cancellation is not used to claim that the original order had no known price. |
| 7 | Historical source locations | Confirm lines 191, 197–199, 219–223, 271 and the exact sibling directory. |
| 8 | Separate refresh mechanism | Keep skipped-checkpoint selection, missing replay snapshots and phase-only snapshot handling distinct from ID reuse. |
| 9 | Scope of fault | General mm_backtest still replaces snapshots; do not label every original-folder module faulty. |
| 10 | Correction scope | Immutable references are stock-day/channel scoped, not global lifetime identities. |
| 11 | Impact logic | False exclusion demonstrated; P&L sign and full impact remain unknown; distinguish selection from execution effects. |
| 12 | Authorship evidence | Use recorded creation-session bounds and staged file evidence, not filesystem times alone. |
| 13 | Source versus receipt clocks | Label transaction, snapshot-origin and conservative availability times separately. |
| 14 | Prior validation scope | 7,393 comparisons in shared reported depth across five diagnostic days; no full-market or exact-queue claim. |
| 15 | Current test evidence | Nine unittest methods passed; eleven scenarios are included within one method. Missing pytest is not a passing suite. |
| 16 | Recovery limits | Six subprocess cases passed; live working-order reconciliation, nonzero bootstrap and physical power failure remain distinct. |
| 17 | Existing integration | AccountReplay and its gate already integrate durable/account code; the corrected research replay is a different path. |
| 18 | Data roots and documentation | No current loader mismatch demonstrated; distinguish configured roots from stale comments and README inventory claims. |
| 19 | Frozen run and artifacts | Compare source fingerprints, preserve inputs, put evidence outside active sources and avoid broad replay. |
| 20 | Whole-deliverable consistency and completeness | No new issue found; retain explicit unverified items and distinguish assessment from implementation work. |

Limitations: the full pytest suite was unavailable in the configured interpreter; archived broad gate evidence was not re-certified; live transport, broker/exchange reconciliation and actual execution realism were not verified; the ongoing full-history P&L was not evaluated here. The assessment deliberately did not install dependencies or edit frozen code.
