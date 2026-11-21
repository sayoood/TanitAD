# PREREG_V8F_PROOF addendum 2: WMSTACK and JUMPCURVE (frozen 2026-11-13 ~02:00, before their data)

Trigger: WMCEIL read WM-BELOW-LINEAR and JUMP-DOMINATES (`wm/RESULT_WMCEIL.json`). The card committed the next lever: closed-form weights written beside the learned predictor. Same latents file, same 1,976 windows / 141 episodes, same 5 episode folds, both seeds, same SKILL definition (ratio of sums). Tier T0: a world-model diagnostic, never driving performance.

## WMSTACK: a programmed predictor (no gradient step)
* Cross-fitted ridge predicting the CHANGE from [z_t, actions, (learned prediction - z_t)], at 0.2, 0.4, 0.8 and 2.0 s. At 2.0 s the learned input is the composed ten-step rollout and the linear part is the direct jump.
* Controls that must read known values: (k1) shuffled change: SKILL <= 0.005; (k2) a stack whose only input is the learned prediction with its coefficient fixed at 1 reproduces the learned SKILL exactly.
* Outcomes: STACK-ADDS if SKILL_stack exceeds max(learned, linear) by more than 0.01 at two or more horizons at both seeds, with the paired episode-cluster bootstrap (B 2,000) of stack minus the better single excluding 0; STACK-EQUALS-BEST otherwise.
* Either way the per-horizon stack is the reference world model for later cards, quoted with its skill table.

## JUMPCURVE: theorem T1 along the horizon
* Direct ridge at k = 1, 2, 4, 5, 10 steps of 0.2 s against the k-fold composition of the one-step ridge, on (z, actions) inputs, the same error definition as WMCEIL part (d): ratio(k) = eps_k / (k * eps_1), and composed error / direct error, per fold.
* Prediction: ratio(k) < 1 for every k >= 2, and non-increasing in k. Outcome JUMP-MONOTONE if that holds at both seeds on the pooled folds; JUMP-MIXED otherwise.
