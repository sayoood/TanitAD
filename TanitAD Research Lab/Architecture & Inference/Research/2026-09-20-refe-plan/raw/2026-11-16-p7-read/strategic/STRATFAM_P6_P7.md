# STRATEGIC family, first read: P7 vs P6 on navtest (REFe, 2026-11-17)

Instrument: `land/eval/instruments/strategic_family.py` (new, staged). Result JSON: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/strat_family/strat_family_P6_P7.json`.
Inputs (all MEASURED, ours): P7 seam `E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.npz`; P6 seam `.../seams/refe_p6final.npz` (md5 4733c3f8..., equals the BASELINES pin); logged future = `human_future_poses` of `F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz`.
n = 12,146 tokens, 136 logs (LEFT 2,501 / STRAIGHT 8,070 / RIGHT 1,575 / UNKNOWN 0; turn-active 4,076, 113 logs). Tier T1 (self-action open loop; the command is an input).
Estimator for every interval: paired log-cluster bootstrap, B 10,000, seed 20260927, proxy_eval.boot stream (answers "another draw of logs" only; training and inference variance unmeasured; one seed per arm). Delta = P7 - P6. Authority: none, reported family, no bar.

## Reproduction of the banked P7 adherence: EXACT
Turn-active adherence P7 = **0.8805201177625123**, CI [0.85551, 0.90254], identical bit for bit to `p7_full_read/command_adherence_p7_full.json` (accuracy, CI, kappa 0.77387, 3x3 confusion all equal). Same for all-token P7 (0.86226), P6 (turn-active 0.88690, all 0.86004), and the paired turn-active delta -0.00638 [-0.01549, +0.00326], not separated. The floor 0.8841 (INHERITED from the banked JSON, `random_proposal_floor.turn_active`, not recomputed here) is above P7's 0.8805: selection adds no adherence over a random proposal.

## S-DEC, per command class (adherence = recall, n)
| class | n | P6 | P7 | P7-P6 [95 %] | separated |
|---|---|---|---|---|---|
| LEFT | 2,501 | 0.9104 | 0.9128 | +0.0024 [-0.0074, +0.0136] | no |
| STRAIGHT | 8,070 | 0.8465 | 0.8530 | +0.0066 [-0.0033, +0.0160] | no |
| RIGHT | 1,575 | 0.8495 | 0.8292 | **-0.0203 [-0.0375, -0.0021]** | yes (adverse) |

P7 confusion (rows command, cols executed LEFT/STRAIGHT/RIGHT): LEFT 2283/206/12, STRAIGHT 617/6884/569, RIGHT 20/249/1306.

## S-GOAL at the 4 s endpoint (point P6 -> P7; delta CI)
| class | endpoint err m | heading err rad | route-progress ratio | signed lateral offset m (+ toward commanded side) |
|---|---|---|---|---|
| LEFT | 3.814 -> 3.705; **-0.109 [-0.208,-0.003]** | 0.139 -> 0.132; **-0.0075 [-0.0139,-0.0007]** | 1.014 -> 1.018; +0.003 ns | -0.195 -> -0.111; +0.083 ns |
| STRAIGHT | 3.208 -> 3.193; -0.016 ns | 0.0405 -> 0.0413; ns | 1.069 -> 1.055; **-0.0138 [-0.0171,-0.0107]** | +0.064 -> +0.068; ns |
| RIGHT | 3.358 -> 3.434; +0.075 ns | 0.172 -> 0.171; ns | 1.058 -> 1.059; ns | +0.208 -> +0.266; +0.057 ns |
| TURN_ACTIVE | 3.638 -> 3.600; -0.038 ns | 0.152 -> 0.147; -0.005 ns | 1.030 -> 1.033; ns | -0.039 -> +0.034; +0.073 [-0.0004,+0.139] ns |

Reading (descriptive, no bar): both arms overshoot the logged route (ratio > 1, 1.01-1.07). P7 improves LEFT endpoint and heading, and its STRAIGHT overshoot shrinks; P7 loses RIGHT adherence. Four of the 19 cells shown separate; each is a necessary, not sufficient, condition for a lever claim (one seed per arm, H-ESTIM-SEED-1; with 19 cells, about 1 false separation is expected at 5 %).

## Controls on this read (all ok)
Logged future as an arm: endpoint, heading, lateral errors exactly 0, ratio exactly 1.0, CI degenerate. Mirror: turn-active adherence P7 0.8805 -> 0.0079, P6 0.8869 -> 0.0081. Self-vs-self: every delta exactly 0.0, CI [0,0]. Tests: `test_strategic_family.py` 9/9 (scipy blocked), each with a RED arm; the mirror control goes RED when mirroring is patched to a no-op.

## Caveats
Route-progress is a ratio of sums (headline); the per-token mean ratio (logged arc >= 1 m, a validity mask, not a bar) is in the JSON as descriptive. Heading error uses the A7-repaired last heading. STRAIGHT lateral offset is raw left-positive y.
