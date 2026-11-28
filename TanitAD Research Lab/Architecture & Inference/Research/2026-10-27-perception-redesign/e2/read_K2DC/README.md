# refcv9 E2 read: K2DC vs K0 — verdict NO-GO (2026-11-16)

Banked for the Master Mind (session 4a3d05) by a banking agent. All numbers are MEASURED (ours, the file named in each cell) unless marked.
Windows: **DEV** (eval139 even half, `DEV_HOLD_SPLIT.json`, 70 clips / 2,397 windows); HOLD is never read. Rule: REFCV9_DESIGN 4.4.
Caveat (H-ESTIM-SEED-1): K2DC and K0 are one training run each; the paired CIs answer "another draw of episodes" only.

## Verdict (`read/E2_VERDICT.json`, written by the read, unedited)

| clause | status | tier | numbers (file) |
|---|---|---|---|
| (a) map primary `nf_swedge_prec_lv_5_20`, K2DC > K0 | **FAIL** s0, s1 | T0 | K2DC 0.4082 vs K0 0.7505, delta -0.3423 (E2_VERDICT.json clause a) |
| (b) box AP@2 m, no NMS | PASS | T0 | 0.3201 vs 0.3195, delta +0.00057 (clause b) |
| (c) planner S6c-DEV sidewalk rate, K2DC < K0 | **FAIL** s1 only | T1 | s0 0.1846 vs 0.1883 (PASS); s1 0.1864 vs 0.1852 (FAIL) (clause c, `read/s6c/`) |
| (d) no four-family row separated worse | **FAIL** s0, s1 | T1 | below (`read/families_s{0,1}.json`) |
| (e) W4 preview, K2DC separated better than camera-blind | PASS | T1 | ade_m + LON, LAT, TAC primaries (clause e, `read/camblind_s*.json`) |

(d), paired episode-cluster bootstrap B 2000 seed 0, 70 episodes, K2DC - K0, lower is better. dt 0.5 s (dt_check PASS, offset 0.0067).
* seed 0 separated worse: LAT_heading_mae +0.0464 deg [0.0021, 0.0986]; LAT_curvature_mae +0.0005 1/m [0.0002, 0.0011];
  LAT_yaw_rate_mae +0.0012 rad/s [0.0003, 0.0026]. LON_accel +0.0079 [-0.0002, 0.0187] is not separated.
* seed 1 separated worse: LON_accel_mae +0.0075 m/s2 [0.0017, 0.0134]. Every LAT row is not separated (heading +0.0112 [-0.0174, 0.0429]).
* LAT_cross +0.0032 / +0.0011 m and LAT_cross_abs_final +0.0074 / +0.0040 m (s0 / s1) are not separated. STRATEGIC: PAUSED (PI 2026-09-19, ITEM 25).

## Against K2M1N20 (`../read_K2M1N20/reread_dtfix/read/E2_VERDICT.json`, dt-corrected, same K0, same windows)

| | K2M1N20 | K2DC |
|---|---|---|
| (a) map primary | 0.4137 (delta -0.3368) FAIL | 0.4082 (delta -0.3423) FAIL |
| (b) box AP@2 m | 0.3281 PASS | 0.3201 PASS |
| (c) S6c sidewalk s0 / s1 | 0.1963 / 0.1938 FAIL, FAIL | 0.1846 / 0.1864 PASS, FAIL (K0 0.1883 / 0.1852) |
| (d) separated-worse rows s0 | 7 (2 LON, 5 LAT) | 3 (LAT heading, curvature, yaw rate) |
| (d) separated-worse rows s1 | 6 (LON accel, 5 LAT) | 1 (LON accel) |
| (e) | PASS | PASS |

Same verdict, NO-GO, with fewer separated rows. LON_accel s0 shrank from +0.0098 (separated) to +0.0079 (not); LAT_cross and
LAT_cross_abs_final are no longer separated. The map primary did not move (0.4137 to 0.4082): clause (a) is untouched by the controller.

## Dose rule (`dose/`, addendum 4 `dose/ADDENDUM4_DOSEDIAG.md`, instrument `dose/dose_diag.py`)

24 controller rows (`dose/dc_rows.jsonl`, `grep dc_share_read` on Thor `runs/r9e2-K2DC/metrics.jsonl`, steps 250 to 6,000).
* **Pre-registered rule, reported as written:** share in [0.05, 0.15] at >= 2 of the 13 firings at steps >= 3,000. In band at 3,750
  (0.0991) and 5,500 (0.0687): 2 of 13, **DOSE-VALID** (`dose/dose_diag_output.txt`). The 5,000 reading is negative (-0.0031).
* **Diagnostic (decides nothing):** over steps > 250, geometric mean 0.1090, firing-bootstrap 95% CI [0.0644, 0.1822] (n 22, log-sd 1.273);
  over steps >= 3,000, 0.1236 [0.0608, 0.2450] (n 12; `dose/dose_late.py`, `dose/dose_late_output.txt`). Both straddle 0.15 ⇒ "not
  resolved" under addendum 4. Only 6 of 22 readings sit in band; w swings 2.03 to 15.49. ESTIMATED: the rule passes with probability 0.906
  at 2x the target and 0.673 at 4x, so the pass is weak evidence of a held dose.

## Provenance

* Thor run `/home/nvidia/r8ns/r9e2/runs/E2_READ_K2DC/` (`logs/ROLL_STATUS.txt`, read-only cat; K0 seeded from E2_READ_K2M1N20), read commit
  9697fa4df004, read tree `read_tree_9697fa4df004_k2dc`. Rolls, camera-blind, dumps and evals ran on Thor 2026-11-16 12:20Z to 15:07Z.
* The S6c scoring, INPUTS.json and the verdict ran on the dev box (`logs/s6c_resume.log`); verdict written 2026-11-16T15:30:44Z, rc 1.
  Those 7 files (verdict, INPUTS, 4 S6c) are not on Thor; INPUTS.json carries dev-box paths.
* Pull `C:/Users/Admin/r9e2_read_K2DC/pull/read/` sha256-equal to Thor for all 14,404 files; the 8 JSONs here match `thor.sha256`.
* Not banked: `dump_K0`, `dump_K2`, `s6c/score_*` bulk (as for K2M1N20). K2M1N20's `ROLL_ABORTED.distkeep_1510Z` has no K2DC counterpart.
* `SHA256SUMS` covers every file except itself.
