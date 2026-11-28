# RESULT_STEP0 (2026-11-18): NAVRBD held-out floor F_HO

Evidence class MEASURED (ours, zero training, CPU). Tier T1-kin (self-action open loop, m 10, causal bucket), route supplied along the driven path (optimistic by construction).
Grid: the T1-feasible held-out grid, 5,736 windows / 497 episodes, sha256 ca289d0f40a8f71043796fe626b26ecaa141853f843acdb1eecbf5648b9ab148; dump (clip, t) pairs equal the grid exactly.
Arms: N1R_s0 and N1R_s1 (private HF Sayood/tanitad-v8f-strat1-runs), dumps pulled from the pod, 1,000 files sha256-verified against the pod copy.
Result file: `out/HO_FLOOR.json` (+ `.sha256`); code `code/ho_floor.py`.

TAC-LAT path (same as the NAVRB2 D1 cell): `strat1_read.load_dump_n` (strat1_read.py:124) -> plan track `cl` and GT track `g` -> `obd_geom.build().labels` = `read_otband.labels` (read_otband.py:112-118, `factor_from_kinematics`) -> accuracy = (label(cl) == label(g)), as in `obd_read.to_cells` (obd_read.py:347-351); delta via `obd_read.cell_delta` (obd_read.py:114), the estimator the NAVRB2 read calls (navrb2_read.py:65-66).

## D1-HO (TAC-LAT accuracy)
| arm | D1-HO |
|---|---|
| N1R_s0 | 0.5905 |
| N1R_s1 | 0.6328 |

## F_HO
**F_HO = |D1-HO(N1R_s1) - D1-HO(N1R_s0)| = 0.0424** (0.042364).
Paired episode-cluster bootstrap (B 2,000, seed 0) of s1 - s0: +0.0424, interval [+0.0260, +0.0582], separated. This interval answers only "would another draw of EPISODES say this?". The 0.0424 itself is a same-recipe seed-pair difference, so it is the rig's seed-to-seed spread on this grid (one pair, n = 1; no spread estimate from one pair). It is not a training-variance interval.

## Class shares and decision split (descriptive)
Ground-truth classes on this grid: lane_keep 0.337, turn_left 0.325, turn_right 0.338.
| | N1R_s0 | N1R_s1 |
|---|---|---|
| pred share lane_keep / left / right | 0.259 / 0.461 / 0.280 | 0.296 / 0.362 / 0.342 |
| lane-keep recall | 0.544 | 0.575 |
| turn recall (any side) | 0.886 | 0.846 |
| false-turn share on GT lane_keep | 0.456 | 0.425 |
| side given turn (both turn) | 0.693 (n 3,367) | 0.783 (n 3,216) |

Confusion (rows GT lane_keep/left/right, cols pred):
s0 [[1052,622,260],[149,1352,361],[286,671,983]]; s1 [[1112,440,382],[233,1285,344],[353,354,1233]].
The seed gap sits mostly in the side decision (0.693 vs 0.783) and in the left/right pred share; the lane-keep versus turn decision moves less. "Timing" is read here as the lane-keep versus turn decision rate per window (no per-event onset timing was computed).

## Controls
* K1 (arm against itself): +0.0 [0.0, 0.0], PASS.
* K-CONST: majority class (turn_right) share 0.338215; a constant predictor's accuracy 0.338215, equal exactly, PASS. Both arms sit well above it.
* K-RE: the same code recomputes the banked NAVRB2 cell N1R_s0 - N1_s0, TAC-LAT, all 141 episodes / 2,399 windows, from the NAVRB2 and STRAT1 dumps held locally: delta -0.004168403501 [-0.0163, +0.0067], identical to the banked `RESULT_NAVRB2.json` cell (pairs N1R-N1, seed 0) to 1e-12. PASS.

## Deletions
See `DELETIONS_STEP0.log`.
