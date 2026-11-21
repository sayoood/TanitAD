# A-14 detector pin (Master Mind, 2026-11-09)

**`tactical_detector.py` sha256 `77d0bed2432e4e5120f21bb1f3a3e0c3481cdc600d5eba2c173aaeede47039d3`**, pinned by the T2 session at 07:33:35 CET on 2026-11-09, before the first REFe tactical table was run. The scorer it calls is `behaviour_score.py` sha256 `69d13d71...`; the detector refuses any other version.

## Why it changed
MM ruling, 2026-11-09: one definition of "lead" in the instrument. `lead_series` and `free_road` now call the pinned scorer's `build_run` and `derive`. The earlier centre and half-width path is reachable only through `P["MUT_LEAD_CENTRE"]`, as a deliberate-regression arm.

## Checks (INHERITED from the T2 session, thor3 `analysis/lane_offset/a14_compare2.json`)
* Example plus the 10 Gate H files, 1,911 ticks: lead_id, has-lead, gap and free-road each 0 differences.
* Regression arm (old path): lead_id differs on 18 (07), 4 (06), 3 (12), 3 (20), 1 (16) ticks, 0 elsewhere, the differences reported before the change.
* Human-vs-human Gate H: 22 events, 0 off-diagonal, 20 of 20 resolved agree, 2 AMBIG_UNRESOLVED on both sides. Same counts per class as before.
* 37 of 37 tests pass. One test changed, because the ruling requires what it forbade: `test_E1_no_simulator_or_scorer_import` is now `test_E1_no_simulator_import_and_scorer_is_pinned`.

Master Mind checked directly: this file's sha256, and that the diff against the earlier landed detector and test contains only the changes above.

## First REFe tactical table (T2-class, u400, 10 scenes, one seed; INHERITED)
Estimator: episode-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, only with at least 20 resolved events per class and at least 10 clusters. **No class reaches 20, so no interval exists: status INSUFFICIENT_N, counts only. No rate is read from this table.**

| class | n | resolved | agree | selected -> executed |
|---|---|---|---|---|
| FREE | 7 | 7 | 2 | go->go 2, go->stop 4, go->change-lane 1 |
| LEAD | 8 | 7 | 5 | follow->follow 5, follow->change-lane 2, follow->AMBIG_LEAD_LOST 1 |
| MERGE | 2 | 0 | n/a | yield->NOT_ENGAGED 2 |
| LANE | 2 | 1 | 1 | change-lane->change-lane 1, change-lane->AMBIG_WRONG_SIDE 1 |
| CONFLICT | 1 | 1 | 1 | yield->yield 1 |
| LINE_G | 2 | 0 | n/a | AMBIG_UNRESOLVED->stop 1, AMBIG_UNRESOLVED->NOT_ENGAGED 1 |

LINE_R has no event. Pooled, descriptive only: n 22, resolved 16, agree 9. Full rows per scene: thor3 `analysis/lane_offset/a14_tactical_table_tier1.json`.
