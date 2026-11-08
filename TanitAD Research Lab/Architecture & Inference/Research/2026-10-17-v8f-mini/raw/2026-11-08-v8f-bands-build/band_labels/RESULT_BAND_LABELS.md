# RESULT: band labels (stream A), 2026-11-08

All numbers MEASURED from the files named. Labels are offline privileged-ego labels (not eval numbers, no tier). STRATEGIC family: PAUSED (PI item 25); nothing here is a strategic pass.
Schema and every changed line: `SCHEMA.md`. Outputs: `F:/strat_out/band_labels/` (single copy, see the manifest in `HANDOFF.md`).

## 1. Time-indexed strategic checkpoints (done, tested)

`strategic_geom_{train,eval}.npz`: train 4,572 clips, 461,539 rows; eval 147 clips (141 cache clips among them), 14,843 rows; 101 steps per clip on the 0.2 s grid; 9 horizons 6..30 s; `ck_xyh` [N, 9, 3], `ck_valid` [N, 9].

**Clock and join (re-verified on real clips).** For the 14 cache-eval clips tested, the cache `pose_grid[j]` equals the ego log read at `grid_start_s + 2 j dt_s`: mean speed error at most 0.006 m/s, position error below 0.05 m (`test_clock_and_join_on_real_clips`). The mutation (+0.2 s) gives a speed error 10x or more larger and goes red. Join key: step j reads v9 `k = 2 j`, and `now_s` is bit-equal to the v9 rows on all 139 eval clips (`test_join_key_k_equals_2j`).

**Coverage per horizon (percent of entries valid).**

| train, 4,572 clips | 6 s | 9 | 12 | 15 | 18 | 21 | 24 | 27 | 30 |
|---|---|---|---|---|---|---|---|---|---|
| strict mask, all 461,539 rows | 94.3 | 91.5 | 88.3 | 84.8 | 80.9 | 76.9 | 74.0 | 71.2 | 68.6 |
| strict mask, step 0 (4,572 rows) | 98.7 | 98.4 | 98.1 | 97.7 | 96.3 | 85.9 | 83.2 | 80.2 | 77.6 |
| `ck_valid_span` (gap rule off), all rows | 97.4 | 96.6 | 95.8 | 94.9 | 93.9 | 92.7 | 91.6 | 90.5 | 89.4 |
| `ck_valid_span`, step 0 | 98.7 | 98.4 | 98.1 | 97.7 | 96.3 | 94.9 | 94.4 | 93.5 | 92.9 |
| `ck_valid_span`, steps 0..39 (key frames up to 7.8 s) | 99.0 | 98.6 | 97.8 | 96.7 | 95.6 | 94.8 | 94.1 | 93.3 | 92.2 |

Eval (147 clips): strict step 0 at 30 s 81.0 %; `ck_valid_span` step 0 at 30 s 92.5 %, all rows 91.4 %.

Against the expectation (AUDIT_DATA 1.3, about 95 % at 30 s, span only, 4,512 clips, INHERITED): the span-only figure here is 92.9 % at step 0. The difference is the extra rules: the arc must lie inside the complete smoothing kernel (24 m before the log end), and 25 clips that never move have no path (`ck_why` 6, no valid entry at all).
**A second finding: the strict mask is lower still (77.6 % at 30 s, step 0).** 138 of 300 randomly drawn train logs hold at least one native-sample gap above 1.0 s (413 gaps, median 1.94 s, maximum 72.7 s), and the v9 gap literal (1.0 s) makes 21.6 % of the 30 s entries invalid (`ck_why` 3). `ck_gap_s` lets a consumer choose its own threshold; `ck_valid_span` is the mask without the rule. I did not decide which mask training uses: that is the Master Mind's call, and the span-only mask interpolates linearly across gaps of unknown content (UNVERIFIED how well).
Rows with NOW before the log start: 324 train, 18 eval (invalid, `ck_why` 1).

## 2. v10 tactical labels, band (2, 6) s (eval139 and train done and tested)

`v10/v10_labels_eval139.npz`, md5 in its manifest. Control: the unedited builder re-run on eval139 reproduces the banked `v9_labels_eval139.npz` bit for bit (md5 `6b5c7f207cffc3b7eb3cd527fd433599`), so the differences below are caused by the edited lines only.

Label-change table against v9, eval139 (27,664 rows; rows with v9 full band `h_obs >= 8 s`: 25,934). Full per-cell table: `F:/strat_out/band_labels/v10/compare_v9_v10_eval139.md`.

| field | defined v9 | defined v10 | changed, all rows | changed among v9-defined (full-band rows) | defined -> IGNORE | IGNORE -> defined |
|---|---|---|---|---|---|---|
| lat_cls_a | 83.8 % | 91.5 % | 13.6 % | **5.2 %** (5.2 %) | 437 | 2,559 |
| lat_cls_b | 83.8 % | 91.5 % | 15.2 % | **7.2 %** (7.2 %) | 437 | 2,553 |
| lon_cls | 70.7 % | 71.2 % | 11.7 % | **15.2 %** (15.0 %) | 134 | 251 |
| goal_y / goal_w (22-bit masks) | | | 15.0 % / 10.9 % | | | |

Label-change table against v9, train, 4,572 clips (909,651 rows; rows with v9 full band `h_obs >= 8 s`: 853,583). The v9 side is the file the ladder trained on, md5 `83e30bd3791f0767707eade5f4671759`. Full per-cell table: `F:/strat_out/band_labels/v10/compare_v9_v10_train.json`.

| field | defined v9 | defined v10 | changed, all rows | changed among v9-defined (full-band rows) | defined -> IGNORE | IGNORE -> defined |
|---|---|---|---|---|---|---|
| lat_cls_a | 83.7 % | 91.9 % | 14.1 % | **5.2 %** (5.2 %) | 14,415 | 88,287 |
| lat_cls_b | 83.8 % | 91.9 % | 15.4 % | **6.8 %** (6.8 %) | 14,389 | 87,730 |
| lon_cls | 69.1 % | 69.7 % | 10.3 % | **13.8 %** (13.7 %) | 2,322 | 6,939 |
| goal_y / goal_w (22-bit masks) | | | 15.3 % / 11.4 % | | | |

The share changed among v9-defined labels reproduces the earlier truncation numbers (AUDIT_MODEL F8: 5.3 / 7.3 / 15.4 %, INHERITED) within 0.2 points. Against the earlier 5 % bar, lat-a sits at it and lat-b and lon are above. **The PI's band ruling overrides that bar; I report it and did not act on it.**
Other changes worth knowing: the lateral labels gain coverage (83.8 to 91.5 % of rows hold a label, because the absence rule needs 6 s observed instead of 8 s); TURN_L and TURN_R positives fall (2.3 to 1.6 % and 6.3 to 3.8 % of rows) since turns that start after NOW + 6 s are not tactical; the longitudinal reach cells (`lon_t_reach_s`, `lon_d_reach_m`) are defined on 31.8 % of rows against 43.1 %; `v_end_ms` is the speed at NOW + 6 s (98.3 % of defined values differ, by construction); `lat_theta_deg` changes on 72 % of defined rows because it is the heading change over the band.
**The 6-8 s span** is covered by no tactical label in v10. It belongs to the strategic head: section 3 re-emits `g_str` / `a_str` over (6, 30).

## 3. Strategic goal tokens per frame over (6, 30) s (done, tested)

`strategic_tok_{train,eval}.npz`, built with the v7 emitter's `strategic()` on a 10 Hz track starting at NOW (two module constants overridden, listed in `SCHEMA.md`). `g_valid` is True on 367,877 of 461,539 train rows (positive tokens always; an absence claim only when 29.95 s are observed; 69.8 % of rows have the full 30 s observed, 1.9 % have under 3 s and claim nothing).

| train, valid rows | rows | clips with at least one valid row |
|---|---|---|
| FOLLOW_ROUTE | 226,510 | 3,051 |
| TURN_LEFT_FOLLOW_ROUTE | 53,040 | 1,029 |
| TURN_RIGHT_FOLLOW_ROUTE | 60,561 | 1,134 |
| STOP_AT_FOLLOW_ROUTE | 27,766 | 570 |
| EXIT_LEFT, EXIT_RIGHT, LANE_CHANGE_L, LANE_CHANGE_R `_FOLLOW_ROUTE` | **0, 0, 0, 0** | 0 |

**Tokens that never occur: 4 of the 8 goals (EXIT_LEFT, EXIT_RIGHT, LANE_CHANGE_L, LANE_CHANGE_R) and 2 of the 7 actions (PREPARE_EXIT, PREPARE_LANE_CHANGE).** They are untrainable from this geometry; they are reported, not padded. The same four goals are empty on eval (147 clips). Actions that occur (train rows): HOLD_MAIN_ROAD 194,184; PREPARE_TURN_L 53,040; PREPARE_TURN_R 60,561; PREPARE_STOP 27,766; RESUME_CRUISE 42,130.
Positive rows by `by_time_s`: 141,367 in total; 19,005 strictly inside 6 to 8 s, 87,004 at 8 s or later, 35,358 at exactly 6.0 s. The 6.0 s rows are turns or stops already under way at NOW + 6 s, reported at the band edge: the tactical-to-strategic continuation, not a new event. Percentiles of `by_time_s` over positive rows (5/25/50/75/95): 6.0 / 6.0 / 10.0 / 16.4 / 25.3 s; `within_m`: 6.7 / 31.5 / 63.3 / 129.3 / 252.1 m. Step-0 view (one frame per clip, comparable in kind to the v7.2 record in F9): FOLLOW_ROUTE 2,257, TURN_LEFT 687, TURN_RIGHT 785, STOP_AT 467; the v7.2 per-clip record (2,948 / 574 / 596 / 454) uses the (8, 30) band and the 8 s anchor, so it is not comparable row for row.
Census: `F:/strat_out/band_labels/strategic_tok_census.json` (script `token_census.py`).

## Tests

`test_band_labels.py`: 14 passed. `post_v10_train.cmd` reran both files after the build: 24 passed, 1 failed, the failure being the expected diff count of the builder test (26 lines once the 3-line one-point-grid guard is counted, the test expected 23). With the count set to 26, `test_band_labels_v10.py`: 11 passed with the train file present (2026-11-08 22:17). The control test reads a scratch rebuild that was deleted after its md5 was recorded, so it skips on a later run unless `run_control4572.cmd` is run again. Required items: schema pin; clock and join on real clips; straight constant-speed closed form (x = v h, y = 0, heading 0, tolerance 1e-3 m) and a ramp-speed closed form; mask behaviour (log end, kernel edge, gap, NOW before the log, static clip); mutation of the time offset (NOW read 0.5 s late, cache clock +0.2 s) goes red; gap and sagitta vectorisation equal the originals on 6 real clips; the v10 builder differs from v9 in exactly the listed lines; a synthetic turn inside 6-8 s.
Not applicable to this stream: the PI's goal-conditioning edge tests and the zero-gradient test (no model edge is built here).

## Deliverable manifest

See `HANDOFF.md`.
