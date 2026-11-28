# VERDICT: PREREAD_V8F_STOPDEC = DECISION-MISSES at both seeds (Master Mind, 2026-11-10)

Card frozen 16:29:05 (card 063a1ba4, script 5491fc9a; pre-registered in e8600570), full read 16:29:52-16:34:51, `out/STOPDEC.json` and the per-window `out/stopdec_PARITY_DAGOACC_s{0,1}.npz`. Tier: **T1-kin first tick**, eval-141, an oracle diagnostic, **never driving performance**. Estimator: episode-cluster bootstrap, B 2,000, seed 0 ("another draw of episodes" only). STRATEGIC: **PAUSED** (PI item 25). All numbers MEASURED.

Controls (both arms): TOP1 = dump `ol` 0.0, GT future = dump 0.0, `v0` = dump 0.0, IDENT = TOP1 0.0, OWN-ALL = IDENT 0.0, goal probabilities in [0, 1], lon rows sum to 1 within 2.0e-7; both frozen sha files checked before the run; 2,351 windows / 139 episodes; STOP_POINT 117, regime STOP 132, regime DECEL 487 windows.

## The committed rule (primary cell STOP_POINT x goal_stop), applied verbatim
| seed | hit rate (T says STOP_POINT) | false-positive rate | speed MSE hit / miss (m/s)^2 | hit MSE / complement | miss mass share | class |
|---|---|---|---|---|---|---|
| s0 | **0.103** [0.016, 0.211] | 0.006 | 16.9 / 30.9 | 2.79 [0.50, 6.87] | **0.940** [0.876, 0.998] | DECISION-MISSES |
| s1 | **0.128** [0.040, 0.229] | 0.009 | 13.5 / 30.9 | 2.25 [0.81, 4.52] | **0.938** [0.876, 0.990] | DECISION-MISSES |

**Outcome: DECISION-MISSES.** T's own decision does not see the stop in about nine of ten stop-point windows, and that is where the speed error sits. Committed trigger: a stop-cue perception card.

## Beside, no bar
* T's 7-way longitudinal decision in STOP_POINT windows (argmax counts s0 / s1): FOLLOW 45 / 41, ACCELERATE 30 / 26, DECELERATE 21 / 21, STOP 12 / 19, KEEP 8 / 7. In GT regime STOP windows (the car is stopped by 5.2-6.0 s): **ACCELERATE 63 / 56 of 132**, STOP 19 / 22, HOLD 15 / 19.
* Where the decision does see a slowdown, the speed cells follow it: regime DECEL x lon_slow, speed MSE 5.51 (hit) against 16.71 (miss) at s0, 6.16 against 16.66 at s1; lon_slow hit rate 0.40 / 0.41.
* Mean p(STOP_POINT) in STOP_POINT windows 0.154 / 0.185; in regime DECEL 0.032 / 0.048.

## What follows (Rule Zero)
The perception card's first question is whether the stop cue is in T's features but under the 0.5 operating point of a ~5 % class, or absent. `PREREAD_V8F_STOPCUE` (frozen before its AUROC was computed) reads the AUROC of T's own p(STOP_POINT) from these npz files: CUE-PRESENT (>= 0.85, lower bound >= 0.80) -> a class-rebalancing training card; CUE-ABSENT (< 0.70) -> a perception card (stop-region resolution, LoRA arm L's read, a stop-line auxiliary); PARTIAL -> both.
