# RESULT_WMACT_SHRINK: level 1 aims right and steps about twice too far (`AMENDMENT_WMACT_1.md`, frozen prefix 2,249 bytes `503ba1b8`)

MEASURED 2026-11-07, dev-box CPU, 8.4 s. Tier T0: a WM diagnostic, never a driving claim.
* The shrink factor c_k is fitted on 1,833 TRAIN windows (141 TRAIN clips, `RESULT_WMACT_TRAIN.md`) and applied to 1,976 eval-141 windows (141 episodes).
* Estimator: episode-cluster bootstrap B 2,000, seed 0, with c_k held fixed. Raw: `RESULT_WMACT_SHRINK.json`; code `run_shrink.py`.

## Controls (both read their known values)
* c := 1 reproduces the read of record's eval SKILL: the difference is 0.0 in all six rows.
* c := 0 reads exactly 0.0 in all six rows.

## Level 1 (* = separated)
| row | c fitted on TRAIN | SKILL as trained (c = 1) | SHRUNK SKILL on eval | ceiling on eval (cos²) | best c on eval |
|---|---|---|---|---|---|
| s0, k1 (0.2 s) | 0.496 | +0.0010 [−0.0114, +0.0118] | **+0.0313 [+0.0267, +0.0357]*** | 0.0313 | 0.504 |
| s1, k1 | 0.526 | −0.0010 [−0.0141, +0.0103] | **+0.0310 [+0.0255, +0.0359]*** | 0.0311 | 0.496 |
| s0, k5 (1.0 s) | 0.501 | −0.0689* | +0.0902 [+0.0741, +0.1062]* | 0.0927 | 0.431 |
| s1, k5 | 0.509 | −0.0577* | +0.0925 [+0.0761, +0.1085]* | 0.0948 | 0.441 |
| s0, k10 (2.0 s) | 0.461 | −0.1500* | +0.1310 [+0.1054, +0.1558]* | 0.1333 | 0.407 |
| s1, k10 | 0.457 | −0.1724* | +0.1249 [+0.0983, +0.1516]* | 0.1281 | 0.395 |

## Outcome by the amendment's rule: **L1-OVERSHOOTS**
* SHRUNK SKILL at k1 is above 0 and separated at both seeds.
* The level-1 predictor aims right and takes about twice the step it should. Halving its predicted change, with the factor fitted on TRAIN only, turns "no better than nothing changes" into positive skill at every horizon read. That holds at 2 s too, where the unshrunk rollout was worse than "nothing changes".
* The TRAIN-fitted factor reaches the eval ceiling at k1: 0.0313 against cos² 0.0313. A scalar cannot do better.
* Committed by the amendment: a TRAIN-fitted scalar is a zero-training fix, and any reliance arm uses the shrunk L1 rollout.

## What it does not say
* Even shrunk, level 1 explains little of the 0.2 s change (3 %; 13 % at 2 s). Most of the bottom level's frame-to-frame change is not predictable from the latent and the action at this model size. The upper levels carry the larger skill (L3 k3 +0.33, `RESULT_WMACT.md`).
* Why a predictor trained under MSE is miscalibrated by about 2× on its own TRAIN windows is UNVERIFIED. Candidates: the rollout uses context 1 while training conditions on longer context, and the WM loss is a sum with SIGReg and IDM. Each is a separate read.
* The planner reads none of this; nothing here is a driving result.
