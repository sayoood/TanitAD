# RESULT_LATSEM: what each world-model level encodes, what it imagines, and whether the imagined future follows the action

MEASURED 2026-11-07, dev-box CPU, run 3 (69 s). **Tier T0: a world-model diagnostic, never a driving claim. The planner reads none of these predictions.**
* Card `PREREG_V8F_LATSEM.md`: frozen prefix 4,882 bytes, `0b0c2f73`, frozen 09:56:59 before any statistic. It is landed with this result, not before it, so the freeze time rests on the logged sha256.
* `AMENDMENT_LATSEM_1.md` (1,375 bytes, `ddb07ff8`) sets control 1's z1 tolerance to 1e-5. Run 1 read INVALID on float32 rounding alone, and no statistic was looked at.
* `AMENDMENT_LATSEM_2.md` (1,642 bytes, `83446cc6`) adds SPEED-SKILL, because R2's instrument has no headroom.
* Data: M2b s0 / s1, eval-141 restricted to the 139 episodes with a frame file. 9,460 frames carry the probes and the retrieval bank; 1,948 windows with 16 real frames carry R2, R3 and A2.
* Estimator: episode-cluster bootstrap over 139 episodes, B 2,000, seed 0. It answers "another draw of episodes"; the two seeds stand in for "another training run"; inference is deterministic.
* Raw data in `RESULT_LATSEM.json`; code in `run_latsem.py`; logs `run_latsem.run{1,2,3}.log`.

## Controls (all read their known values in runs 2 and 3)
* C1: z1 copies of one frame agree to 1.2e-6 (≤ 1e-5, Amendment 1); raw actions are exactly equal.
* C2: per-frame z2 / z3 equal `HJEPAStack.encode` to 1.4e-6.
* C3: the start-state query reads exactly 0.0 in R2 and A2; two identical action sequences give CF exactly 0.0.
* C4: no probe fitted on labels permuted across episodes is separated above 0.

## Outcomes
1. **R1, ENCODES speed at all three levels, at both seeds** (R² 0.65 / 0.66 at L1, 0.52 / 0.56 at L2, 0.42 / 0.52 at L3). **NOT-ENCODES acceleration, yaw rate, the 2 s speed change or the 2 s heading change, at any level.**
   * Each level's state is a pointwise function of ONE frame (window w = 1). A still frame shows how fast a scene usually is (highway, junction, queue), not how the car is moving now. Motion can live only in the transitions.
2. **R1, TD-INCONSISTENT** (DD +0.497 / +0.270, separated). H-JEPA's temporal decomposition, slow state kept up the stack and fast detail dropped, does not appear here. Speed is lost going up, and acceleration was never there.
3. **A2, DECODES-FUTURE-SPEED at L2 and L3, at both horizons and both seeds; level 1 does not.** L3's 2.4 s rollout beats "speed stays the same" by +0.233 / +0.223, L2's 1.2 s by +0.168 / +0.157. Level 1, even shrunk, is separated at s0 only (+0.2 and +1.0 s).
   * The rollout beats the oracle, the true future latent through the same probe (+0.038 / +0.062 at L3 k3). The rollout is driven by the true future actions, which carry the speed change; the one-frame future latent carries speed only through scene context. So this skill is the action pathway turning a manoeuvre into a future speed. It is not scene foresight.
4. **R3, ACTION-SEMANTIC-LON at all three levels, at both seeds.** "Accelerate" against "brake" moves the imagined speed by 27 % to 43 % of the kinematic reference.
   * **LAT:** L3 reads ACTION-SEMANTIC, but small (+0.012 / +0.062 rad/s against 0.5). L2 reads MIXED (the seeds disagree). L1 reads MIXED, and both seeds are separated in the WRONG direction (−0.028 / −0.019).
   * The yaw-rate probe has no skill on real frames (R1), so CF-LAT reads a direction in latent space, not a yaw rate.
5. **R2, NOT at every level and horizon.** The oracle query is not separated above 0 on most rows, so nearest-frame retrieval from other episodes cannot resolve a 0.2 to 2.4 s speed change, even with a perfect prediction. Retrieval is kept for the pictures, where it shows what kind of scene a level imagines.

## What this means, and the next levers (Rule Zero)
* **Answer to "does the prediction react to potential actions":** yes longitudinally, at every level, in the right direction, at about a third of the physical size. Steering is barely represented, and level 1 has its sign wrong.
* **Answer to "the semantics of each level":** every level is a scene-and-speed-regime code. Speed lives in the state; motion lives in the predictor; L2 and L3 turn an action sequence into a correct future speed.
* **Lever 1, a picture of the latent itself:** a post-hoc pixel decoder per level, as the paper uses (a lightweight transformer from one latent, for visualisation only). A box-2 training item, not run. **[H]** One 128-d latent from 4×10 pooled tokens will decode blurry.
* **Lever 2, the reliance arm:** O reads the L2 / L3 rollouts, the levels that carry action-conditioned future speed. It needs its own card and launch-gate entry.
* **Lever 3, motion in the state:** windowed upper encoders (w = s, so level 2 reads 2 frames and level 3 reads 4; the paper's Eq. 1 allows w > 1), then a TD re-read. A WM training change with its own card.
* **Exploratory, deciding nothing:** at cruise speed (≥ 8 m/s) the imagined future barely answers "accelerate" and answers "brake" by 1.3 to 3.3 m/s. From standstill (< 3 m/s), L3's "accelerate" response is +2.19 / +1.70 m/s against a kinematic +3.6. So the world model does represent a launch. **[H]** The cruise asymmetry is rarity in the data or probe saturation at high speed.

## Pictures (decide nothing; windows chosen by true labels only; M2b s0)
* `fig/pca_levels.png` (in the repo): per-level PCA maps of the 9,460 frames, coloured by speed and yaw rate.
* **Kept OFF the public repo**, because they show raw camera frames. They are at `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/latsem/fig/`, sha256 below:
  * `strip_{STOP,LAUNCH,TURN,CRUISE}.png`: per level, now, the true future and the imagined future (the nearest real frame from another episode).
  * `imagine_{LAUNCH,TURN}.png`: per level, the imagined future under brake, hold, accelerate, w+ and w−.

| file | sha256 |
|---|---|
| imagine_LAUNCH.png | `44291197790b2d3c6c48e569c4f38d8bcd66821426bc179f0c8718e30bae7f67` |
| imagine_TURN.png | `ef91c4d90e3c076b1b298dc4564831ed6a118ce88df2b50f77d17edc4cb4b044` |
| strip_CRUISE.png | `58446eded525e2ad3e5eeb0fefeaccfb8b006107be2c13e8240acbebd6ede560` |
| strip_LAUNCH.png | `524b35d8db07dd8d3bc8622c3deccc60a7b8b96dcda82d8f3180885201e0e0bf` |
| strip_STOP.png | `08027bb570d497c76a9726b9ec0f9cb63d63d382bf7e1fffbcb4f8d10569d3fb` |
| strip_TURN.png | `d7d9ba1c398c3beab07f15743159b8ce84e93467809ca6168485501e747166a9` |

## Tables (generated from RESULT_LATSEM.json)

## R1: what each level ENCODES (cross-fitted ridge probe, eval R², s0 / s1; * = separated)
| level | speed v | accel a | yaw rate w | Δv over 2 s | Δheading over 2 s |
|---|---|---|---|---|---|
| L1 | +0.651 [+0.542, +0.731]* / +0.656 [+0.562, +0.726]* | -0.338 [-0.531, -0.194]* / -0.327 [-0.509, -0.179]* | -0.095 [-0.273, +0.042] / -0.062 [-0.277, +0.092] | -0.356 [-0.558, -0.206]* / -0.339 [-0.544, -0.175]* | -0.100 [-0.297, +0.054] / -0.134 [-0.427, +0.056] |
| L2 | +0.516 [+0.293, +0.660]* / +0.563 [+0.392, +0.683]* | -0.285 [-0.470, -0.140]* / -0.321 [-0.488, -0.187]* | -0.151 [-0.450, +0.067] / +0.013 [-0.226, +0.175] | -0.327 [-0.562, -0.152]* / -0.462 [-0.690, -0.299]* | -0.354 [-0.719, -0.068]* / -0.114 [-0.387, +0.079] |
| L3 | +0.420 [+0.175, +0.586]* / +0.520 [+0.265, +0.688]* | -0.072 [-0.180, +0.022] / -0.193 [-0.328, -0.086]* | -0.022 [-0.361, +0.212] / +0.116 [-0.119, +0.287] | -0.117 [-0.249, -0.007]* / -0.262 [-0.440, -0.128]* | -0.072 [-0.456, +0.188] / +0.064 [-0.188, +0.252] |

Temporal decomposition DD = [R²_a(L3) − R²_a(L1)] − [R²_v(L3) − R²_v(L1)]: +0.497 [+0.307, +0.715]* / +0.270 [+0.073, +0.530]* → **TD-INCONSISTENT**.

## A2: predicted future speed through the probe, SPEED-SKILL vs "speed stays the same" (s0 / s1)
| level, horizon | deciding rollout | raw rollout (L1) | oracle: the true future latent | outcome |
|---|---|---|---|---|
| L1, +0.2 s | +0.045 [+0.011, +0.083]* / +0.004 [-0.053, +0.063] | +0.053 [-0.016, +0.123] / -0.048 [-0.159, +0.065] | +0.018 [-0.001, +0.039] / +0.012 [-0.009, +0.033] | **NOT** |
| L1, +1.0 s | +0.052 [+0.009, +0.097]* / +0.011 [-0.063, +0.089] | +0.023 [-0.070, +0.112] / -0.087 [-0.235, +0.061] | +0.012 [-0.019, +0.042] / +0.011 [-0.022, +0.044] | **NOT** |
| L1, +2.0 s | +0.042 [-0.013, +0.099] / +0.012 [-0.073, +0.097] | -0.075 [-0.224, +0.055] / -0.189 [-0.378, -0.008]* | +0.030 [-0.018, +0.080] / +0.024 [-0.029, +0.073] | **NOT** |
| L2, +0.4 s | +0.103 [+0.042, +0.187]* / +0.090 [+0.025, +0.159]* | same | +0.023 [+0.002, +0.045]* / +0.029 [+0.008, +0.052]* | **DECODES-FUTURE-SPEED** |
| L2, +1.2 s | +0.168 [+0.100, +0.259]* / +0.157 [+0.076, +0.232]* | same | +0.025 [-0.006, +0.052] / +0.042 [+0.009, +0.073]* | **DECODES-FUTURE-SPEED** |
| L3, +0.8 s | +0.147 [+0.071, +0.235]* / +0.145 [+0.054, +0.217]* | same | +0.004 [-0.014, +0.019] / +0.016 [-0.006, +0.039] | **DECODES-FUTURE-SPEED** |
| L3, +2.4 s | +0.233 [+0.146, +0.342]* / +0.223 [+0.121, +0.309]* | same | +0.038 [-0.006, +0.081] / +0.062 [+0.010, +0.115]* | **DECODES-FUTURE-SPEED** |

## R3: does the imagined future FOLLOW the action (decoded at the longest horizon, s0 / s1)
| level (horizon) | accel − brake, speed (m/s) | kinematic reference | share of reference | yaw-rate w+ − w− (rad/s) | LON | LAT |
|---|---|---|---|---|---|---|
| L1 (2.0 s) | +2.723 [+2.454, +2.983]* / +3.536 [+3.255, +3.812]* | 8.23 | 33 % / 43 % | -0.028 [-0.033, -0.024]* / -0.019 [-0.024, -0.013]* | **ACTION-SEMANTIC** | **MIXED** |
| L2 (1.2 s) | +1.349 [+1.224, +1.474]* / +1.674 [+1.558, +1.788]* | 5.09 | 27 % / 33 % | -0.004 [-0.007, -0.001]* / +0.008 [+0.005, +0.013]* | **ACTION-SEMANTIC** | **MIXED** |
| L3 (2.4 s) | +3.634 [+3.447, +3.809]* / +2.895 [+2.742, +3.047]* | 9.69 | 37 % / 30 % | +0.012 [+0.007, +0.018]* / +0.062 [+0.056, +0.068]* | **ACTION-SEMANTIC** | **ACTION-SEMANTIC** |

## R2: nearest-frame retrieval, SEM-SKILL for speed (s0 / s1); the oracle row is the instrument's ceiling
| level, horizon | deciding query | oracle query | outcome |
|---|---|---|---|
| L1, +0.2 s | +0.013 [-0.007, +0.030] / +0.008 [-0.010, +0.027] | +0.013 [-0.052, +0.071] / +0.015 [-0.023, +0.053] | NOT |
| L1, +1.0 s | +0.052 [+0.001, +0.104]* / +0.023 [-0.021, +0.066] | +0.015 [-0.048, +0.072] / +0.013 [-0.032, +0.056] | NOT |
| L1, +2.0 s | +0.073 [-0.011, +0.174] / +0.037 [-0.020, +0.094] | +0.066 [-0.017, +0.128] / -0.012 [-0.089, +0.053] | NOT |
| L2, +0.4 s | +0.036 [-0.009, +0.079] / -0.017 [-0.044, +0.012] | +0.027 [-0.017, +0.063] / -0.008 [-0.074, +0.060] | NOT |
| L2, +1.2 s | +0.016 [-0.071, +0.093] / -0.041 [-0.102, +0.018] | +0.014 [-0.033, +0.057] / +0.002 [-0.059, +0.068] | NOT |
| L3, +0.8 s | -0.024 [-0.127, +0.078] / +0.066 [-0.008, +0.140] | -0.027 [-0.049, -0.003]* / -0.002 [-0.021, +0.015] | NOT |
| L3, +2.4 s | -0.041 [-0.242, +0.127] / +0.101 [-0.044, +0.243] | -0.023 [-0.073, +0.030] / +0.043 [-0.023, +0.105] | NOT |

## Exploratory, post hoc, deciding nothing (AMENDMENT_LATSEM_2): decoded speed response split by start speed (m/s, s0 / s1)
| level | start speed | windows | accel − hold | hold − brake |
|---|---|---|---|---|
| L1 | < 3 m/s | 237 | +0.87 / +1.34 | +1.61 / +2.02 |
| L1 | 3-8 m/s | 433 | +0.48 / +0.95 | +2.44 / +2.81 |
| L1 | ≥ 8 m/s | 1278 | +0.01 / +0.57 | +2.69 / +2.92 |
| L2 | < 3 m/s | 237 | +1.02 / +1.20 | +0.89 / +0.62 |
| L2 | 3-8 m/s | 433 | +0.57 / +0.74 | +1.03 / +0.74 |
| L2 | ≥ 8 m/s | 1278 | -0.12 / +0.27 | +1.28 / +1.44 |
| L3 | < 3 m/s | 237 | +2.19 / +1.70 | +0.89 / +0.98 |
| L3 | 3-8 m/s | 433 | +1.56 / +0.61 | +2.47 / +2.55 |
| L3 | ≥ 8 m/s | 1278 | +0.30 / -0.45 | +3.30 / +3.29 |
