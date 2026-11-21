# P6 on full navtest: result (Master Mind, 2026-11-09)

Checkpoint `model_final.pt` sha256 `968a469d...`. T1-class: NAVSIM v1.1 PDMS, self-action open loop against logged agents, not closed loop. Harness PASS 12,146 of 12,146. One checkpoint, deterministic planner, one training seed. Estimator: paired log-cluster bootstrap over 136 logs, 10,000 resamples, seed 20260927; it answers "another draw of logs" only.

## PREREG_P6, read as written: NOT PROVEN against u400 and against P2. BELOW GOAL.

| | PDMS |
|---|---|
| P6 level | **91.20 [90.41, 91.98]** (`level_ci.json`) |
| against C1, u400 88.25 | **+2.95 [+2.19, +3.77]**, better 6,861, worse 2,165, tied 3,120 |
| against C2, P2 88.82 | **+2.38 [+1.70, +3.05]** |
| against C1 without the 36 declared tokens (n 12,110) | +2.95 [+2.19, +3.77], level 91.17 |
| goal 94.55 (Table A13) | **BELOW GOAL**: upper bound 91.98 < 94.55 |

**Why not ADOPT.** The lower bound is above 0 on both comparators, so the outcome is not REFUTED. The rule also requires that no longitudinal or lateral family component separates adversely, and some do:
* Against u400, 9 of 21 components: along-track MAE, speed MAE and RMSE, progress error mean, target-speed accuracy within 0.5, 1.0 and 2.0 m/s, cross-track MAE and final cross-track MAE.
* Against P2, 1 of 21: progress ratio mean.

These are unpaired interval non-overlap, which the instrument calls a conservative reading; the card asks for a paired CI on each component. All of these are distances from the human driver, while PDMS rose. `MM_RULING_P6_READ_GAPS.md` loosely describes the rule as "PDMS against C1/C2 and the goal bar"; the card's clause is stricter and is applied here.

The tool's "NO-GO (adverse sub-score: DDC)" is P2's rule, not P6's. It is reported: DDC -1.18 [-1.72, -0.69], not part of the PDMS rule and not a listed family component.

## Sub-scores

| | P6 [95 % CI] | Table A13 | against u400 |
|---|---|---|---|
| NC | 98.85 [98.50, 99.18] | 98.80 | -0.09 [-0.43, +0.25] |
| DAC | 97.95 [97.50, 98.39] | 99.25 | +1.37 [+0.66, +2.17] |
| EP | 87.03 [85.76, 88.28] | 92.72 | +5.84 [+5.14, +6.60] |
| TTC | 95.08 [94.18, 95.93] | 95.74 | -0.54 [-1.23, +0.10] |
| C | 100.00 | | +0.03 |
| DDC | 96.48 [95.65, 97.25] | | -1.18 [-1.72, -0.69] |

The largest remaining gap to the paper is EP (-5.69), then DAC (-1.30), then TTC (-0.66).

## Four families
* **LONGITUDINAL:** EP up, TTC not separated; speed and along-track error to the human worse (above). Slow picks on cruise 0.224 [0.117, 0.348] against u400's 0.412; plan v4 minus GT v4 +0.84 m/s [-0.15, +1.68]; overspeed share 0.535. u400 undershot speed; P6 overshoots on cruise. `r2_result.json`, `P6_SPEED_RESULT.md`.
* **LATERAL:** DAC up; cross-track error to the human worse.
* **TACTICAL:** decision accuracy (no interval) lateral 0.913 against u400's 0.925, longitudinal 0.648 against 0.679, 5-way 0.754 against 0.783. **Command adherence** (MEASURED 2026-11-09, full navtest, executed plan, paired log-cluster bootstrap 10,000 draws): turn-active (n 4,076) 0.887 [0.863, 0.908], kappa 0.785, against u400 +0.004 [-0.014, +0.020], not separated; all tokens 0.860 [0.840, 0.879], kappa 0.737. The same labeller on the human's logged future reads 0.881 turn-active and 0.876 on all tokens; a random proposal from P6's own fan reads 0.885 and 0.835. On turns P6 follows the command as often as the human and as often as its own fan, so adherence comes from the proposals, not the scorer. **Selection quality** (MEASURED 2026-11-10, all 64 proposals scored on all 12,146 tokens): pick 91.20 [90.41, 91.98], random proposal 67.37 [65.92, 68.80], best of 64 97.90 [97.46, 98.33]; regret 6.70 [6.14, 7.30] = zero picks with a non-zero proposal in the fan 2.81 [2.34, 3.30] (355 tokens) + TTC-0 picks 1.58 [1.29, 1.89] (449) + the rest 2.32 [2.12, 2.52]; 29 tokens have no non-zero proposal. Best of P6's own top 5 by its rule: 94.47 (top 8: 95.28). `selection/` in this folder.
* **STRATEGIC:** PAUSED (PI 2026-09-19), not a pass.

## H-P6-SPEED-1 (`PREREG_P6_SCORER_SPEED`): REFUTED as written, all gates green
G_TTC -0.190 [-0.272, -0.102] and G_NC -0.157 [-0.239, -0.079] (bar: within 0.15). (C) fails: slow-pick share 0.224 against the bar 0.20, though its deltas clear (against u400 -0.189 [-0.313, -0.084], against u400 without lane -0.143 [-0.267, -0.037]). The pessimism shrank to about 40 % of u400's and did not go away.

## Departures, stated
1. Frame control: exit 1 at 6.18 m, the declared 36-token C1 set, max and median bit-identical to u400 and P2-s1 (`MM_NOTE_FRAME_CONTROL.md`). Not re-run.
2. The seam was made through `chain_p6.py` (Amendment 3), then read by the unchanged `stage3_full.py` with its `--arm-*` paths. No `--dump-proposals` in that pass; the second pass adds it, and also serves as the card's bit-identical inference check.

## In-run series, as the NOT PROVEN branch requires (MEASURED, pod `metrics.jsonl`, `train.log`)
Label coverage 100 % on all 3,809 logged lines of the final log; label lag 0 on all 10,105 on-policy records; on-policy skill rose from about 0.2 at the start to 0.77 to 0.85 in the last 5 records. The held-out ranking statistic is the Stage 0 instrument, on u400 only so far. The run looks healthy.

## Next, as committed
NOT PROVEN and REFUTED name the same lever: a scorer-only PDM fine-tune on P6 (`PREREG_P6`; SFT-4's recipe took SFT-4 +3.79) and the scorer retrain (`PREREG_P6_SCORER_SPEED`), the arms Stage 0 unlocked. They were designed on u400. Building them on P6 needs its own pre-registration; EP is not in the current loss.
