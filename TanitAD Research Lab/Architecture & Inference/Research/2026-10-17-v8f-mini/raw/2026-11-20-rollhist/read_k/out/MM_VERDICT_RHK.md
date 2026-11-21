# ROLLHIST next arm RHK (a DAgger bank generated under readout K): verdict of record ROLLHIST-HARMS (MM, 2026-11-21)

## Card, arms and method
* **Card:** `PREREG_V8F_ROLLHIST.md` (781eea87) with Amendments 1 to 8. Amendment 8 (sha256 d9976e80) was frozen at 14:42:04Z, before any RHK bank, training step or number.
* **Arms:** RHK_s0 and RHK_s1 against the existing B0_s0 and B0_s1. RHK keeps RH's dose (p = 0.1) and flags and trains on a bank generated under readout K (token `causal_g0x`). Code of record 97fe0a56.
* **Evidence class:** MEASURED (ours). The pod trained both arms. The dev box ran 12 rollouts and the reads with zero training.
* **Tier:** T1-kin/hb0.2 (kinematic rollout on the model's own actions); G-T0 at T0.
* **Estimator:** paired episode-cluster bootstrap, B 2,000, seed 0. It answers "another draw of EPISODES"; the two seeds answer "another TRAINING run"; inference is deterministic.
* **Floor:** F = the larger of the two arms' seed spreads per cell.
* **Files:** result `RESULT_ROLLHIST_K.json`; guard `GT0_ROLLHIST_K.json`; STRATEGIC `READ_STRAT_ROLLHIST.json` (controls all pass).

## P1: RHK - B0 under the form of record (`R0C10XHb20`), both seeds
| family | cell | RHK - B0 [interval] | F | verdict |
|---|---|---|---|---|
| LONGITUDINAL | speed_mae | +0.0049 m/s [-0.0122, +0.0237] | 0.126 | NOT-SEPARATED |
| | along_mae | +0.0232 m [-0.0087, +0.0574] | 0.122 | NOT-SEPARATED |
| | headway U | -0.0018 [-0.0087, +0.0044] | 0.023 | NOT-SEPARATED |
| LATERAL | heading_mae | -0.300 deg [-0.458, -0.135] | 0.70 | NOT-SEPARATED (the interval straddles the floor) |
| | yaw_rate_mae | -1.05 deg/s [-1.92, -0.12] | 3.06 | NOT-SEPARATED (the interval straddles the floor) |
| TACTICAL | ade_tac | -0.0065 m [-0.0572, +0.0469] | 0.164 | NOT-SEPARATED |
| STRATEGIC | position at tick 2 | -1.138 m [-1.475, -0.840] | 0.71 | BETTER |

* **Families:** STRATEGIC BETTER; LONGITUDINAL, LATERAL and TACTICAL NOT-SEPARATED; no family WORSE.
* Under the card's straddle-by-sign view, LATERAL also reads BETTER. The label is the same.

## Beside, no bar
* **RHK - RH under the form of record** (bank K against bank C\*, same dose):
  * LONGITUDINAL MIXED: speed +0.269 m/s [0.149, 0.400] and along +0.248 m [0.080, 0.424], both worse.
  * LATERAL MIXED: yaw rate +4.50 deg/s [3.61, 5.54] worse.
  * TACTICAL WORSE: ade_tac +0.356 m [0.105, 0.618].
  * STRATEGIC WORSE: +0.476 m [0.266, 0.710].
  * So the K bank gives LESS of the own-action gain than the C\* bank did.
* **RHK - B0 under readout K** (`R0KHb20`): LATERAL BETTER (heading -0.53 deg, yaw rate -0.82 deg/s); the other families are not separated.

## The guard and the other conditions
* **G-T0** (RHK - B0 at T0 ADE6, at most +0.0705 m; F_T0 INHERITED, from one seed pair): **fails at both seeds**.
  * s0: +0.0641 m [0.0341, 0.0962].
  * s1: +0.0474 m [0.0194, 0.0753].
  * By comparison, RH (C\* bank) failed it at s0 only, and RHp3 (p = 0.3) failed at both.
* P2 False (RHK is WORSE on every repair pair; the B0 attribution control holds).
* P3 False (RH_R0Hb20 and RH_R0C10XHb20 DEGRADE).
* P4 False.
* The reader's answer: "K bank repairs NEITHER P2 nor P3 and G-T0 FAILS at all seeds".
* The beside UE.B localisation is NOT-COMPUTABLE, because STAGEB never ran.

## Label
By the card's precedence (BLOCKED, then HARMS), G-T0 failing at both seeds is **ROLLHIST-HARMS**, whatever P1 shows. Amendment 8 committed this in advance. It is reported as written.

## What it says
* Two bank readouts (C\* and K) and two doses (0.1 and 0.3) have now been tried. Every arm that trains on its own rolled history costs the teacher-forced T0 forecast on logged history:
  * the C\* bank failed G-T0 at one seed (p = 0.1) or both (p = 0.3);
  * the K bank fails at both seeds while giving less own-action gain.
* So the cost is not set by the bank's readout. The levers on the bank's readout and on the dose are spent.
* What stays untested is the LABEL on the own states. The DAgger bank keeps the human future as the target for on-route own states and drops the off-route ones. `H-ST-RECOVERYLABEL-1` (2026-10-28, MEASURED) built a feasible recovery label for those off-route states. It kept 59.5 % / 62.0 % of tick-2 states, against a 75 to 81 % prediction that was not met. Its named refinements are a curvature-bounded (Dubins or clothoid) recovery and loss-masked partial-horizon labels.
* By the eval doctrine, T0 is a world-model diagnostic, not driving performance, and T1 is the tier for capability claims. The card made G-T0 a binding guard, so the label stands; this reading sits beside it.

## Next lever (card section 3.4 HARMS row, Amendment 8 outcome table)
The recovery label for off-route states (`H-ST-RECOVERYLABEL-1`). It is a new arm, which needs its own card, a build and a PI go.
