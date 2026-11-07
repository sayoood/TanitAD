# RESULT_WMACT: v8F-mini's world model, level by level (card `PREREG_V8F_WMACT.md`, frozen prefix 5,478 bytes `d21b14cc`)

MEASURED 2026-11-07 on the dev-box CPU. Tier T0: teacher-forced, logged actions, true future frames. This is a WM diagnostic, never a driving claim.
* Data: eval-141, 1,976 of 2,399 windows (those whose 16 frames are all real; start at window frame 0), 141 episodes.
* Estimator: episode-cluster bootstrap B 2,000, seed 0, the draws of `taniteval/ci.py`. The interval answers another draw of episodes; the two seeds answer the training run; inference is deterministic.
* Raw: `RESULT_WMACT.json` (M2b), `RESULT_WMACT_C0.json` (C0 beside), `weights_check.json`.

## Controls (all four read their known values, so the read is valid)
| control | M2b s0 | M2b s1 |
|---|---|---|
| 1. identity donor: ACTION-USE and sens, max abs | 0.0, 0.0 | 0.0, 0.0 |
| 2. zeroed action embedding: ACTION-USE at k 1 / 5 / 10, sens at level 1 | 0.0 / 0.0 / 0.0, 0.0 | 0.0 / 0.0 / 0.0, 0.0 |
| 3. rollout from another window's start: SKILL at L1 k1 (must be below 0, separated) | −39.2162 [−42.8020, −36.1577] | −40.5909 [−44.2353, −37.3847] |
| 4. WM loss through `HJEPAStack.loss` against `WMAux.forward`, max abs difference | 0.0 | 0.0 |

Control 3 is below 0 and separated at every level and horizon read (`RESULT_WMACT.json`, `CTRL3_SKILL_startdonor`).

## The statistics at k = 1 (95 % interval; * = separated)
| level (cycle) | SKILL s0 | SKILL s1 | ACTION-USE s0 | ACTION-USE s1 | ACTION-USE-MEAN s0 | ACTION-USE-MEAN s1 |
|---|---|---|---|---|---|---|
| L1 (0.2 s) | +0.0010 [−0.0114, +0.0118] | −0.0010 [−0.0141, +0.0103] | +0.0054 [+0.0025, +0.0085]* | +0.0049 [+0.0015, +0.0084]* | +0.0029 [+0.0005, +0.0052]* | +0.0030 [+0.0002, +0.0061]* |
| L2 (0.4 s) | +0.0295 [+0.0095, +0.0475]* | +0.0264 [+0.0061, +0.0436]* | +0.0424 [+0.0286, +0.0576]* | +0.0462 [+0.0284, +0.0660]* | +0.0317 [+0.0193, +0.0451]* | +0.0254 [+0.0130, +0.0379]* |
| L3 (0.8 s) | +0.1316 [+0.0953, +0.1694]* | +0.1351 [+0.0925, +0.1790]* | +0.1850 [+0.1364, +0.2343]* | +0.2000 [+0.1389, +0.2643]* | +0.1041 [+0.0725, +0.1372]* | +0.1074 [+0.0700, +0.1461]* |

Longer horizons:
| row | s0 | s1 |
|---|---|---|
| L1 k5 SKILL | −0.0689 [−0.1145, −0.0284]* | −0.0577 [−0.1024, −0.0183]* |
| L1 k10 SKILL | −0.1500 [−0.2324, −0.0777]* | −0.1724 [−0.2609, −0.0938]* |
| L1 k10 ACTION-USE | +0.0897 [+0.0620, +0.1166]* | +0.0900 [+0.0624, +0.1192]* |
| L2 k3 SKILL | +0.1084 [+0.0588, +0.1576]* | +0.1185 [+0.0622, +0.1740]* |
| L3 k3 SKILL | +0.3283 [+0.2624, +0.3873]* | +0.3175 [+0.2459, +0.3823]* |
| L3 k3 ACTION-USE | +0.3423 [+0.2764, +0.4037]* | +0.3238 [+0.2505, +0.3957]* |

* Level 1, one channel at a time, at k1: acceleration alone and yaw rate alone are each separated at one seed only. At k5 and k10 both channels are separated at both seeds (`RESULT_WMACT.json`, `ACTION-USE[acc swapped to donor]` and `[yaw ...]`).
* Beside: relative sensitivity `sens_rom/all` at level 1 is 0.189 (s0, k1). It is far above the card's inertness line of 0.01.

## Outcomes by the card's rules
* **L1: ACTION-REACTIVE; PREDICTS not met.** The true action predicts the real future better than another episode's action and better than the mean action, at both seeds. But at 0.2 s the rollout does not beat "nothing changes" at either seed, and at 1.0 and 2.0 s it is worse than "nothing changes", separated.
* **L2: PREDICTS and ACTION-REACTIVE.**
* **L3: PREDICTS and ACTION-REACTIVE.** This is the largest skill: +0.33 at k3 (2.4 s).
* **Committed next (card, level 1):** "PREDICTS fails" ⇒ the same statistics on TRAIN windows, to separate fit from generalisation. The upper levels' outcome is "PREDICTS and ACTION-REACTIVE", whose committed next is the reliance arm (O reads the WM's rollout under its own draft plan), under its own card.

## C0 beside (deciding nothing; its WM tensors differ from M2b's, 12 of 145 bit-equal)
The same pattern holds, and all four controls read their known values.
* L1 k1 SKILL is not separated: +0.0103 [−0.0008, +0.0199] / +0.0062 [−0.0047, +0.0157].
* L1 k10 SKILL is worse than "nothing changes": −0.0925 / −0.0914, separated.
* L2 k1 SKILL is +0.0290 / +0.0249, and L3 k1 SKILL is +0.1261 / +0.1160, all separated.
* ACTION-USE is separated at every level at both seeds. ACTION-USE-MEAN at L1 k1 is not separated at s1 (+0.0025 [−0.0003, +0.0054]).

## What this answers, and what it does not
* The world model's prediction reacts to the action at all three levels, and the reaction points the right way: the true action is closer to the real future than another episode's action or the mean action.
* The two upper (more abstract) levels predict the future better than "nothing changes". The bottom level does not at 0.2 s, and drifts worse than "nothing changes" by 1 to 2 s.
* None of S, T or O reads these predictions. That is the reason for the reliance arm, and it is not a driving result.
* Implementation choices are logged in the card before any statistic: a tip export of the code, start at window frame 0, the mean action over eval-141's real frames rather than TRAIN, and context 1.
