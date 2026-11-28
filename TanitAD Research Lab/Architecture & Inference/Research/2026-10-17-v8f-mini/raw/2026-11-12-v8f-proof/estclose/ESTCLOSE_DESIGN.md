# ESTCLOSE: form RRe, the closing speed of the lead rule estimated from the reader's own reads (design note, card ADD79)

Generator `make_estclose_wrapper.py` writes `v8f_t1_eval_reader_rule_rzgfhqe.py` (sha256 prefix a747abeb51ef041f, 1,517 lines) from the wrapper `v8f_t1_eval_reader_rule_rzgfhqc.py` (sha256 prefix 8732329b4476347d, 1,326 lines; the generator refuses any other parent). Parent line numbers below are those of the parent; "new" line numbers are those of the generated file. 21 asserted edits, each old text exactly once. The generator is `build()` plus a main; the test `test_generator_changes_only_the_asserted_places` diffs parent and output line by line and requires that every changed parent line lies inside an edit.

Tier: T1-kin (kinematic rollout on the model's own actions). Nothing here reads a label at inference: the estimate uses the logged tokens through the reader and the estimator file only.

## Edits (parent line -> new line)
| id | parent | new | what |
|---|---|---|---|
| X1 | 2 | 2 | header docstring (form, flag, looks, logs); the parent text follows it |
| X2 | 159 | 174 | `FORMS` gets `RRe` |
| X3 | 211 | 226 | `feed`: form RRe takes the RR branch (`used = min(g, Q(g))`, presence `p >= 0.5` and the full-history flag): presence and gap of RRe are those of RR |
| X4 | before 258 | 273-437 | helper block: `MID_EST`, `load_estimator`, `_head3`, `_est_fill`, `_est_at`, `est_fed`, `_est_tick`, `est_tick_arrays`, `_count_ok`, `est_line`, `write_est_line` |
| X5, X6 | 268, 283 | 439, 454 | `head_values_reader`: at the tick-level look form RRe takes `c_estv = _est_tick(n)` as closing; every other form returns `c` / `c_fed` as before |
| X7 | 712 | 887 | `ReaderPlanner.__call__` hands its `states` to the estimate (form RRe only) |
| X8, X9, X10 | 612, 616, 656 | 783, 787, 829-831 | `mid_looks`: the reader branch takes RRe; its closing is `est_fed(_est_at(st, rd, q))`; the row carries `c_est`, `est_ok` |
| X11 | 673 | 847 | `save_mid` writes the two keys (form RRe only) |
| X12 | 1116 | 1292-1299 | `finish_preroll`: the pre-roll rule log gets its own `c_est` / `est_ok` (the main-phase arrays are dropped from its meta) |
| X13, X14 | 1176, 1177 | 1360-1365 | flag `--closing-est`; refused for every form but RRe, required for RRe |
| X15 | 1207 | 1395 | later looks (`--rule-every` below the replan interval) admit RRe |
| X16, X17 | 1262, 1265 | 1450, 1453 | RRe needs `--reader` and `--qmap` |
| X18 | 1274 | 1462-1463 | `load_estimator(mine.closing_est, ...)` after the reader is loaded |
| X19 | 1279 | 1469 | mix of RRe: presence head, gap head, closing head (the value `head_values_reader` / `mid_looks` return, which is the estimate) |
| X20 | 1305 | 1495 | `log.save` gets `c_est`, `est_ok` as arrays of the length of the rule log (length asserted in `est_tick_arrays`) |
| X21 | 1321 | 1511-1513 | `main` writes the run line after `finish_preroll()` |

## The estimate at a look at frame e (`_est_at`, new 344)
* Look frame e = `s.j + e_off` per window: `e_off = 0` at the replan tick, `q` at a later look (the same frame the parent's `_reader_at` / `_reader_closing_at` use at a later look, parent 536-568).
* Needs frames `e - need + 1 .. e` of the clip, `need = est.need_frames(5) = H + 4`: condition `e - need + 1 >= 0 and e < len(s.d["feats"])` (new 352). Otherwise `c_est = NaN`, `est_ok = False`, and the rule gets exactly 0.0 (`est_fed`, new 369), the closing input of form RR.
* X [H, 3], oldest first: for the windows ending at frames `e-H+1 .. e`, the columns (gap read before any map in m, presence p, closing output times 10 m/s) of the five-frame window (`_est_fill`, `_head3`). `_head3` is the head of the reader on frame embeddings with the arithmetic of `TokReader.head_from_embeds` (p, gap) and `reader_closing` (closing, `CLOS_SCALE_MS` = 10, parent 161 and 218-228), i.e. the three quantities of `closeread_dump.py` 111-152. Estimate = `ClosingEstimator.apply(X)` of `closeread/closest.py` (imported, unchanged), float64 m/s, not clipped.
* Cache: frame embeddings per (clip, frame) and window reads per (clip, frame) in `STATE["est"]`; they are not part of the stores that the pre-roll snapshots (`_tracked`, parent 1035), because they depend on the clip and the frame only. Test `test_cache_returns_the_same_reads_as_no_cache` shows equal estimates with and without cache and that every frame is embedded once.
* A window ending at frame f embeds frames f-4..f of the clip's cache array. Read at frame e uses frames at or before e only.

## Pre-roll: how the rule is applied and which looks are logged
* The pre-roll runs P/30 segments through the ORIGINAL rollout with the same planner (`PrerollRun.__call__`, parent 950-994): each segment is a full `ReaderPlanner.__call__`, so the rule acts at every tick and, with `--rule-every` 5, at every later look, and form RRe estimates at both. The segment states start at `t0 - P`, so early pre-roll looks usually have too short a history (estimate 0.0).
* Logged: tick rows of the pre-roll go to `rule_log_pre_<run>.npz` (parent 1102-1120; new 1285-1300), now with `c_est` and `est_ok` built from the segment snapshots of `STATE["cols"]` (X12). Later-look rows of the pre-roll are snapshotted in `post["mid"]` and counted only (parent 1131-1132), not written to a file.
* Counts: `est_line()` (new 405) counts `est_ok` over the main-phase tick rows plus main-phase later-look rows (`scored`) and over the pre-roll segments of the model planner (`preroll`), and appends `est looks scored full <n> short <n> preroll full <n> short <n> file <sha256>` to `run_<run>.txt` beside the rule log (new 416), after `finish_preroll()` (new 1511-1513). `full` = est_ok True, `short` = False.

## How RRh feeds its closing (for the controls)
RRh: closing = third output of the reader times `CLOS_SCALE_MS` = 10 (parent 218-228, `reader_closing`), float32, sign lead speed minus ego speed, no clip. At the tick it is `c` in `head_values_reader` (parent 266, returned at 283); at a later look it is `_reader_closing_at(st, rd, q)` (parent 554-568; used at parent 616). The rule then forms `v_L = max(0, v_l + c_src)` (parent 640 for later looks; the same line in `MixPlanner.__call__` of the distrule file at the tick). A planted estimator "reader_closing" (weight 1 on the closing column of the last frame) therefore returns the RRh value on a full-history look; the test shows it within 1e-4 m/s (float32 batch against one-row evaluation).

## Logs and interface
`rule_log_<run>.npz`: `c_est` float64 (NaN where no full history), `est_ok` bool, row for row with the other arrays. `rule_log_mid_<run>.npz`: the same two keys. `rule_log_pre_<run>.npz`: the same two keys for its rows. The side file `rule_log_<run>_rd.npz` carries them too. Other forms: no new key anywhere (test `test_mid_look_rows_of_other_forms_are_the_parent` compares all mid-look arrays of RR, RRc, RRh, RLc, RLb with the parent run on the same stubs, and `test_other_forms_tick_look_is_the_parent` the tick look).

## Tests and mutations (all CPU)
`test_estclose_wrapper.py`: 25 tests (real `TokReader` with small random weights, random float16 clips; the rule is a recorder, everything else of `mid_looks` / `head_values_reader` / `finish_preroll` / `main` refusals runs as written). `run_mutations_ec.py`: 26 mutations of the new code, each must turn a test red; result in `MUTATIONS_EC.json` (control before 25 passed, 26 of 26 red, control after 25 passed).

## What I could not verify
* The real pod run: the wrapper was imported and exercised with stubs only (no pinned tool, no GPU, no real reader, no real clip). `finish_preroll` and the refusals were run through stubs; the full `main()` path of form RRe (X16-X20) is covered by text tests and the pod check only.
* That `s.d["feats"]` of a window state is the array the batch tokens are gathered from (the premise of the parent's own later look, parent 524-527). At the tick the batch tokens are NOT used by the estimate: the last row of X comes from the cache array, so a difference would show as a difference between `c_read` (batch) and `c_est` (cache) in check R2.
* The tick file (`ticks_<tag>.npz`) schema: the pod check compares it only between two runs of the same wrapper (zero estimator against RR), not against the banked run.
* That the banked subset comparison (check Z2) is exact: the subset run has other batches than the banked run; the check uses a tolerance (`--tol-banked`) and reports counts.
* The estimator file `meta_json` / `seed` match to the reader seed is not checked by the wrapper (the queue pairs seed k with model seed k by file name).
