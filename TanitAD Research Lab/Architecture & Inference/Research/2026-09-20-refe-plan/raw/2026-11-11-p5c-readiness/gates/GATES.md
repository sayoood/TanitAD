# P5-C label-table gates TB-0..TB-7 (instrument + pod runner), 2026-11-12
Evidence class: MEASURED (own, CPU, dev box) unless marked. Pre-reg `PREREG_P5C.md` untouched. HOLD = this directory.

## Files
| file | role |
|---|---|
| `p5c_table_gates.py` | the gates, the merged-labels writer (4.1 raw-fallback), the table-meta fields, the 4.3 rank policy. stdlib + numpy. Last line `ZZTB_ALL_PASS` / `ZZTB_FAIL <ids>` |
| `test_p5c_table_gates.py` | real-subset run + a deliberate-regression arm per gate. `ZZTBGATES_PASS 94/94` (`TEST_RUN.log`) |
| `p5c_pod_label.sh` | fail-closed pod runner, `bash -n` OK, NOT run. `POD_SHIP_LIST.txt` = what to ship |
| `TB1_control_1018.json/.log` | TB-1 result |
| `subset_run/` | `gates.json` + `table_meta_fields.json` from the CLI on the 4,393-row subset |

## What each gate measures (and what the instrument can and cannot see)
| id | measured | bar (frozen) | note |
|---|---|---|---|
| TB-0 | max abs diff of main vs control labels on rows ok in both; also a control-ok row missing/errored in the main build, or a traj sha mismatch, fails | 0.0 | control rows are restricted to the bank given |
| TB-1 | reads the driver's `control` output | `bit_identical == n == 1018`, diff 0.0, 0 errors | needs the file; else NOT_RUN |
| TB-2 | count of rows whose driver knot control failed (error text `control/finite/shape check failed`), plus any persisted `knot_maxdev` > 1e-9 | 0 failures | **INDIRECT**: the driver does not persist the per-row knot deviation (it raises when dev > 1e-9, giving an error record). Value reads `None`. To make it direct, the driver would write `knot_maxdev` per record (the gate already reads that field) |
| TB-3 | identity table (label := traj through the table's JSON encoding, or a given `--identity-table`) equals the bank traj bytes | 100 % | the driver's own `ident` row (sign 0.0) is not persisted, so the table-level control is what is computable |
| TB-4 | table key set == bank key set (log, token, rank, sha), none extra, no duplicate record, no traj-sha mismatch, `--expect-rows` (pod: 150603), raw_fallback share | exact; fallback <= 1.0 % | absent record = key missing = FAIL; errored record = raw_fallback (label := traj) |
| TB-5 | every label finite, max abs dyaw vs traj < pi, `sim_hz == 10` on the bank row | 100 % | |
| TB-6 | per rank, on tc20s rows: median over rows of (max acc, max heading rate) of label / of traj (P5 `learnability()`, vectorised, verified equal per row) | <= 2x both. Rank-0 FAIL = STOP; rank-1 FAIL = rank-0 only | the bar reads the RATIO OF MEDIANS; median-of-row-ratios is logged as a diagnostic only |
| TB-7 | `n_differs` per rank (+ n rows, n fallback) recorded | recorded | a zero is a warning, not a fail |

4.3 policy: r_k recomputed from the per-row records of `repair_rank{0,1}.json` with 4.3's estimator (P5 `boot_weights` + ratio, 10,000 draws, seed 20260927). The reports used seed 20261111, so their CIs differ slightly (below). RK6 not required (sim8 only). Final decision also becomes STOP when TB-6 rank-0 fails.

## Results on the 4,393-row subset (30 logs, dev-box labels, `subset_run/gates.json`)
| gate | status | n | value |
|---|---|---|---|
| TB-0 | PASS | 4,393 | max abs diff 0.0 (control = the full dev-box shards) |
| TB-1 | PASS | 1,018 | 1018/1018, diff 0.0 |
| TB-2 | PASS (indirect) | 4,393 | 0 knot-control failures |
| TB-3 | PASS | 4,393 | 4393/4393 |
| TB-4 | PASS | 4,393 | 4393/4393 keys, 0 fallback (subset used as the bank) |
| TB-5 | PASS | 4,393 | 4393/4393 |
| **TB-6** | **FAIL (rank 0)** | 2,713 / 1,680 | rank-0 acc **1.706x**, heading rate **2.018x** (bar 2.0); rank-1 1.645x / 1.745x PASS. Diagnostic median-of-row-ratios: rank-0 1.614 / 1.904 |
| TB-7 | PASS | 4,393 | n_differs rank-0 2,713 of 2,713, rank-1 1,680 of 1,680 |

Final line: `ZZTB_FAIL TB-6 TB-6-rank0-STOP`. **This is a real measurement, not an instrument fault**: the clean control (labels := traj) reads exactly 1.0 and passes everything, and the 1e-3-order margin is on one of four numbers at SUBSET scale (30 logs, not the bank). I did not move the bar. Per section 4.2 a rank-0 FAIL at bank scale means STOP (next lever P5-R, card section 9); the full bank on the pod decides. Not computable here: the labels of the other 593 logs.

4.3 inputs (`rank_policy`, 300 rows per rank, estimator seed 20260927): r_0 +3.00 pp [1.04, 5.36] (report: [1.06, 5.28]); r_1 +4.67 pp [2.37, 7.29] (report: [2.33, 7.21]). RK1..RK5 all hold, so the 4.3 rules alone give `all` (`decision_from_4_3_rules_alone`), overridden to STOP by TB-6 rank-0 on this subset. Disclosure of section 17 item 9 applies.

## Tests (`ZZTBGATES_PASS 94/94`, ~50 s, CPU)
Every arm turns its gate FAIL by name and keeps its neighbours PASS where specific: TB-0 (1e-6 perturbation; a table row fallen back; no control = NOT_RUN), TB-1 (1017/1018; diff 1e-12; n=100), TB-2 (control-failure record; `knot_maxdev` 1e-6), TB-3 (identity label 1 ulp off), TB-4 (key dropped; extra key; sha mismatch; fallback 1.37 % FAIL vs 0.91 % PASS; duplicate; expected 150,603 vs 4,393), TB-5 (yaw + 2pi; NaN; sim_hz 20), TB-6 (jerky sawtooth on rank-0 -> STOP; on rank-1 -> rank-0 policy; clean control exactly 1.0), TB-7 (no rank-1 label), policy (UNDECIDED without records, RK0 STOP, RK1 both ways, RK2, determinism). Also: real `traj_label_table.py build` + `verify` on both merged variants (4393/4393; `rank0only` differs_from_raw = rank-0 n_differs only), CLI exit codes, traj_sha == the table module's == the labeller's.
Key agreement checked: `p5c_labels.traj_sha` (float64 LE bytes), `traj_label_table.traj_sha256` (`struct '<d'`) and this module agree on every subset row. No mismatch. (The driver docstring's "json.dumps compact" wording is stale; the code is the bytes form.)

## TB-1 (MEASURED)
`p5c_labels.py control --n 1018`, one process, CPU, `REFE_SIM_HZ=10`, maps root set: **1018/1018 bit-identical, max abs diff 0.0, 0 errors, 173.7 s.** (`TB1_control_1018.json`)

## Open items / MM decisions
1. **TB-6 rank-0 on the bank**: the subset reads 2.018x on heading rate. The full-bank pod run decides; if it holds, STOP per 4.2. The MM may want to state whether the bar is the ratio of medians (implemented, as worded) or the median of ratios (1.904, diagnostic).
2. **Meta fields**: `traj_label_table.py build` stores only label_name, require_coverage, bank_rows, covered, partial. The 4.1 meta (n_raw_fallback, n_differs per rank, code blob shas, bank sha, rank_policy, host) is in `table_meta_fields.json` only. Putting it into the table needs a `--meta-json` flag in the wiring hold's module (its sha is pinned in the runner; changing it moves the pin).
3. **Rank policy mask**: the table module has no `rank_policy` handling. The gates write `merged_labels_rank0only.jsonl` (rank-1 rows label := traj, `source: raw_rank1`) and the runner builds from it when the decision is `rank0`. G-DVB's `rank_policy` pin therefore has to read the table's own sha, not a meta field, unless item 2 is done.
4. **TB-2 is indirect** (see table). A driver change to persist `knot_maxdev` changes its pinned sha256.
5. The pod `refe/` (from `refe-plan-vdac`) is not hash-pinned; TB-0 is the guard against drift.
6. `p4_tracker_comp.py` (imported by `p5_target_labels.py`) must ship too; it is in the list and pinned.
7. Runner never executed: the first real run may need a path fix. It runs `nice -n 19`, 16 shards, resumable, refuses on a trainer process, a wrong sha, a missing ZZLABEL_DONE, or a wrong row count.
8. Nothing is left running (no HANDOFF.md needed).
