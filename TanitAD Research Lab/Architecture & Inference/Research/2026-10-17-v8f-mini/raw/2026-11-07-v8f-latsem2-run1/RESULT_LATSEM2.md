# RESULT_LATSEM2: does each world-model level encode and imagine the scene, not only ego motion?

MEASURED 2026-11-07, dev-box CPU, `run_latsem2.py` run 1 (210 s for the numbers incl. beside draws). **Tier T0: a world-model diagnostic, never a driving claim. The planner reads none of these predictions.**

**READ INVALID: control(s) 3, 4 failed.** Per the card the read is INVALID and no outcome below is a decision. The numbers are reported as measured, flagged `uncontrolled`, with the evidence for the Master Mind to amend a control that the data shows mis-specified. Nothing was re-drawn to pass.

* Card `PREREG_V8F_LATSEM2.md`: frozen prefix 3,769 bytes, `1325f21c`, frozen 12:57:27. `IMPL_CHOICES_LATSEM2.md` (14,840 bytes, `b17688dc`) froze 13:08:14, before any statistic; the run asserts both hashes.
* Data: M2b s0 / s1, eval-141 restricted to the 139 episodes with a frame file (LATSEM's bank): 9,460 frames carry the probes, 1,948 windows with 16 real frames carry S2, 505 of them (53 episodes) have a near lead at frame 0 and carry S3.
* Estimator: episode-cluster bootstrap over the 139 episodes, B 2,000, seed 0. It answers "another draw of EPISODES". It does not answer "another PROBE FIT": the control failures below are exactly that blind spot (`H-ESTIM-SEED-1` family). The two training seeds stand in for another training run. `*` marks a separated interval.
* Probes: ridge lambda 10 (continuous), logistic C 0.1 (binary, 7-class), cross-fitted by the same two episode folds; features standardised on the fit fold. 72 logistic fits, all converged. Code `run_latsem2.py`, `labels_latsem2.py`; raw numbers `RESULT_LATSEM2.json`; log `run_latsem2.run1.log`.

## Targets and n (labels only; frozen in IMPL_CHOICES)

| target | n frames | note |
|---|---|---|
| lead present (BA - majority) | 8,874 | trainer's `lead_near` (lead within 40 m) on `lead_eval139.npz`; 28 % positive |
| lead gap (R2) | 2,489 | m, clipped [0, 40]; 56 episodes |
| lead closing speed (R2) | 2,311 | m/s over 0.4 s, abs value above 20 m/s set NaN (lead-track switches) |
| agents within 30 m (R2) | 8,874 | obstacle join, centre distance <= 30 m, all 11 classes the join carries, any azimuth |
| nearest-agent distance (R2) | 8,636 | m, frames with at least one cuboid |
| 7-class longitudinal (BA - majority) | 8,834 | v9 `lon_cls`, 7 classes (HOLD 95 ... KEEP 2,891) |
| turn ahead (BA - majority) | 9,460 | abs heading change over 4 s above 0.3 rad; 15 % positive |
| beside: closing speed, no 20 m/s cap (R2) | 2,364 | beside |
| beside: agents within 30 m inside the frustum (R2) | 8,874 | beside |

All seven targets are buildable from the data: the obstacle join (`b1eval_agents_3d.jsonl.xz`, 139 clips) and the lead file (`lead_eval139.npz`) both exist for eval-141's 139 episodes. The card says 10 dynamic classes; the join carries 11 class strings (adds `protruding_object`) and none was dropped.

## Controls

| control | committed value | observed | status |
|---|---|---|---|
| C1 start-state query SKILL | exactly 0.0 on every S2 row | max abs point 0.0, max abs interval end 0.0 (126 rows) | PASS |
| C2 two identical action sequences, S3 | exactly 0.0 | max abs 0.0 (6 rows, intervals [0, 0]) | PASS |
| C3 permuted-label probes (S1, 54 rows) | none separated above 0 | rows separated above 0: M2b_s0/L2/turn | **FAIL** |
| C4 label-shuffled S3 | not separated below 0 | rows separated below 0: M2b_s0/L2, M2b_s0/L3, M2b_s1/L3 | **FAIL** |
| LATSEM's copy / level checks, re-run | z1 copies <= 1e-5, actions exactly equal, z2/z3 equal `encode` to 1e-5 | 1.19e-06, 0.0, 1.43e-06 | PASS |
| metric self-test (synthetic) | perfect predictor BA 1.0, baseline 1/3, paired self-difference exactly 0 | 1.0, 0.3333, 0.0 | PASS |

**What the two failures show (evidence, not a decision).**
* **C4.** Shuffled gap labels give a separated S3 of EITHER sign. Primary draw: M2b_s0/L1 +0.504 [+0.373, +0.630]*, M2b_s0/L2 -0.353 [-0.590, -0.122]*, M2b_s0/L3 -0.796 [-1.339, -0.313]*, M2b_s1/L1 +0.283 [+0.072, +0.492]*, M2b_s1/L2 -0.185 [-0.523, +0.132], M2b_s1/L3 -0.518 [-1.049, -0.016]*. Over 20 further draws (120 rows) 25 are separated below 0 and 50 above 0. The reason is structural: accelerate and brake move the imagined latent along a nearly fixed direction, so ANY probe direction reads a consistent nonzero difference, and the bootstrap (episodes resampled, probe held fixed) does not see the probe's own variance. A separated interval is necessary, not sufficient. Permutation reference (beside): the real S3 point against the 20 shuffled draws, per seed and level, in the S3 table below.
* **C3.** Twenty further permutation draws per row (beside; rows x draws separated above 0): classification (lead_present, lon7, turn): 18 of 360; continuous (the other 6): 0 of 720. A classification row is separated above 0 at a rate of 5.0 % per draw, so one draw over the 18 classification rows fails with probability 60 %: the control as committed cannot pass reliably. The primary failure is row `M2b_s0/L2/turn`, balanced accuracy minus majority +0.056 [+0.006, +0.110]*. Balanced accuracy minus a majority baseline has a null centred on 0 (a junk classifier predicts the minority class at random), so with 18 classification rows a single separated draw is expected at some rate; R2 rows have a strongly negative junk value and never fail.

## S1: what a level ENCODES (R2 for continuous, balanced accuracy minus the fit-fold majority's for binary and 7-class)

Cell = s0 / s1, point [95 % interval]; `*` separated. ENCODES = point > 0 and separated at BOTH seeds (**bold**). Permuted-probe rows beside in the json.

| target | L1 | L2 | L3 |
|---|---|---|---|
| lead present (BA - majority) | **+0.20 [+0.13, +0.26]* / +0.19 [+0.12, +0.25]*** | **+0.12 [+0.06, +0.18]* / +0.19 [+0.13, +0.25]*** | **+0.14 [+0.08, +0.19]* / +0.24 [+0.18, +0.30]*** |
| lead gap (R2) | +0.24 [-0.00, +0.46] / +0.03 [-0.33, +0.31] | -0.47 [-1.23, +0.05] / -0.05 [-0.44, +0.26] | -0.06 [-0.47, +0.25] / -0.12 [-0.55, +0.24] |
| lead closing speed (R2) | -1.17 [-2.04, -0.51]* / -0.54 [-1.03, -0.18]* | -0.81 [-1.38, -0.40]* / -0.92 [-1.76, -0.33]* | -0.56 [-0.92, -0.28]* / -0.51 [-1.10, -0.16]* |
| agents within 30 m (R2) | +0.31 [+0.04, +0.43]* / +0.23 [-0.16, +0.39] | -0.25 [-1.27, +0.20] / +0.03 [-0.68, +0.32] | -0.23 [-1.02, +0.09] / +0.00 [-0.83, +0.32] |
| nearest-agent distance (R2) | +0.11 [-0.14, +0.27] / +0.14 [-0.12, +0.29] | +0.13 [-0.15, +0.31] / +0.12 [-0.16, +0.30] | +0.07 [-0.22, +0.24] / +0.15 [-0.12, +0.31] |
| 7-class longitudinal (BA - majority) | **+0.08 [+0.05, +0.11]* / +0.09 [+0.06, +0.12]*** | **+0.08 [+0.04, +0.11]* / +0.12 [+0.05, +0.19]*** | **+0.10 [+0.07, +0.14]* / +0.13 [+0.05, +0.20]*** |
| turn ahead (BA - majority) | **+0.16 [+0.11, +0.21]* / +0.18 [+0.12, +0.23]*** | **+0.20 [+0.14, +0.27]* / +0.17 [+0.11, +0.24]*** | **+0.21 [+0.16, +0.26]* / +0.21 [+0.16, +0.27]*** |
| beside: closing speed, no 20 m/s cap (R2) | -0.66 [-1.88, -0.26]* / -0.39 [-1.18, -0.14]* | -0.91 [-2.42, -0.38]* / -0.66 [-1.90, -0.26]* | -0.48 [-1.25, -0.22]* / -0.27 [-0.79, -0.10]* |
| beside: agents within 30 m inside the frustum (R2) | +0.23 [-0.18, +0.36] / +0.14 [-0.45, +0.31] | -0.20 [-1.47, +0.21] / -0.06 [-1.11, +0.25] | -0.30 [-1.48, +0.06] / -0.11 [-1.28, +0.23] |

**SCENE-AWARE at level l** (card: ENCODES lead gap OR agent count at both seeds):

| level | by lead gap | by agents within 30 m | verdict (uncontrolled) |
|---|---|---|---|
| L1 | no | no | not SCENE-AWARE |
| L2 | no | no | not SCENE-AWARE |
| L3 | no | no | not SCENE-AWARE |

Beside the card's definition: **lead present** ENCODES at all three levels at both seeds, the **7-class label** at L1, L2, L3 and **turn ahead** at L1, L2, L3. The card's SCENE-AWARE counts only the gap and the agent count, which are continuous targets read by ridge lambda 10; none of them is separated positive at both seeds at any level (the only separated cells are agents within 30 m at L1 s0, +0.31, and lead closing speed, which is separated NEGATIVE).

## S2: what a level IMAGINES (SKILL of the rollout against the start state through the same probe)

Decision query: L1 shrunk (TRAIN c_k), L2 and L3 raw. IMAGINES = SKILL > 0 separated at both seeds. **Lead gap, every level and horizon** (s0 / s1; `perm` = the permuted-label probe through the same query, beside):

| level, k | n (s0) | decision SKILL s0 | decision SKILL s1 | IMAGINES | oracle s0 / s1 | perm s0 / s1 |
|---|---|---|---|---|---|---|
| L1 k1 | 529 | -0.01 [-0.06, +0.03] | +0.01 [-0.03, +0.04] | no | -0.02 [-0.06, +0.02] / +0.01 [-0.02, +0.03] | +0.03 [-0.01, +0.06] / +0.03 [-0.00, +0.06] |
| L1 k5 | 528 | -0.01 [-0.10, +0.06] | +0.01 [-0.06, +0.07] | no | -0.02 [-0.09, +0.05] / +0.04 [-0.01, +0.08] | +0.08 [+0.01, +0.13]* / +0.03 [-0.02, +0.08] |
| L1 k10 (longest) | 521 | -0.02 [-0.14, +0.08] | -0.02 [-0.12, +0.08] | no | +0.03 [-0.09, +0.14] / +0.08 [-0.02, +0.16] | +0.12 [+0.03, +0.19]* / +0.01 [-0.06, +0.08] |
| L2 k1 | 530 | +0.07 [+0.02, +0.11]* | +0.01 [-0.07, +0.08] | no | +0.02 [-0.01, +0.05] / -0.02 [-0.06, +0.02] | +0.06 [+0.00, +0.12]* / +0.03 [-0.04, +0.13] |
| L2 k3 (longest) | 526 | +0.09 [+0.01, +0.15]* | +0.03 [-0.12, +0.14] | no | +0.02 [-0.03, +0.06] / -0.06 [-0.13, +0.00] | +0.12 [+0.02, +0.20]* / +0.08 [-0.00, +0.20] |
| L3 k1 | 534 | +0.05 [-0.09, +0.16] | +0.02 [-0.12, +0.12] | no | -0.03 [-0.07, +0.01] / -0.03 [-0.07, +0.00] | +0.01 [-0.09, +0.11] / +0.06 [-0.04, +0.14] |
| L3 k3 (longest) | 524 | +0.08 [-0.07, +0.22] | +0.12 [-0.07, +0.26] | no | -0.08 [-0.18, +0.02] / +0.01 [-0.07, +0.08] | -0.03 [-0.16, +0.10] / +0.11 [-0.02, +0.22] |

**All targets at each level's longest horizon** (decision query; s0 / s1 point, `*` separated, `P` = a permuted-label probe is separated above 0 at that cell at either seed; **bold** = IMAGINES):

| target | L1 k10 | L2 k3 | L3 k3 |
|---|---|---|---|
| lead present (BA - majority) | -0.023 / -0.032 | +0.130* / +0.014 | +0.079 / -0.016 P |
| lead gap (R2) | -0.017 / -0.018 P | +0.090* / +0.029 P | +0.076 / +0.125 |
| lead closing speed (R2) | +0.066 / +0.021 P | **+0.137* / +0.118* P** | +0.072 / +0.077 P |
| agents within 30 m (R2) | +0.037 / +0.099* P | +0.144 / +0.090 P | +0.233* / +0.113 P |
| nearest-agent distance (R2) | +0.043 / -0.003 P | +0.029 / +0.138* P | +0.090 / +0.153* P |
| 7-class longitudinal (BA - majority) | +0.020 / +0.010 | +0.033 / +0.021 | +0.021 / +0.074* |
| turn ahead (BA - majority) | +0.015 / +0.068 P | +0.050 / +0.078 P | +0.187* / +0.105 P |
| beside: closing speed, no 20 m/s cap (R2) | +0.062* / +0.029 P | **+0.171* / +0.111* P** | **+0.086* / +0.073* P** |
| beside: agents within 30 m inside the frustum (R2) | **+0.082* / +0.098* P** | +0.152 / +0.103 P | +0.152 / +0.094 P |

IMAGINES holds at both seeds for 2 of the card's 49 (level, k, target) cells: L2/k1/lead_closing, L2/k3/lead_closing; 2 of them also read a separated permuted-label probe at one seed. Beside targets: L1/k1/agents30_view, L1/k5/agents30_view, L1/k10/agents30_view, L2/k1/lead_closing_uncapped, L2/k3/lead_closing_uncapped, L3/k3/lead_closing_uncapped; 6 of 6 carry the same flag. **SCENE-IMAGINING (lead gap at the longest horizon): L1 no, L2 no, L3 no.** The permuted-label probe is itself separated above 0 on 70 of 126 S2 cells: a probe fitted on junk reads a positive SKILL when the rollout state sits nearer the label mean than the start state does (a variance effect), so a positive SKILL on a target the probe cannot encode (closing speed has S1 R2 below 0 at every level) is not foresight. The oracle query (the true future latent) is separated above 0 on the lead gap at no cell, so this instrument has no headroom on the gap.

## S3: does the imagined lead gap follow the action? (windows with a near lead at frame 0)

Imagined gap under accelerate (+1.5) minus brake (-3.0) at the longest horizon, m. The card's ACTION-SEMANTIC-LEAD needs this **below 0** (accelerating closes the gap). Beside: the 20 shuffled-label draws.

| seed, level | accel - brake (decision) | identical sequences | decoded mean gap accel / brake | shuffled draws (20): point min .. max | shuffled primary | ACTION-SEMANTIC-LEAD |
|---|---|---|---|---|---|---|
| s0 L1 | +2.42 [+2.11, +2.73]* | +0.0 | 22.9 / 20.4 | -0.90 .. +0.47 | +0.50 [+0.37, +0.63]* | no |
| s0 L2 | +2.90 [+2.49, +3.28]* | +0.0 | 21.8 / 18.9 | -0.48 .. +1.09 | -0.35 [-0.59, -0.12]* | no |
| s0 L3 | +7.64 [+6.58, +8.61]* | +0.0 | 24.3 / 16.7 | -0.86 .. +1.54 | -0.80 [-1.34, -0.31]* | no |
| s1 L1 | +2.92 [+2.46, +3.37]* | +0.0 | 22.8 / 19.9 | -0.89 .. +0.96 | +0.28 [+0.07, +0.49]* | no |
| s1 L2 | +2.78 [+2.26, +3.29]* | +0.0 | 22.6 / 19.9 | -1.05 .. +1.21 | -0.18 [-0.52, +0.13] | no |
| s1 L3 | +2.60 [+1.68, +3.39]* | +0.0 | 21.7 / 19.1 | -1.37 .. +0.93 | -0.52 [-1.05, -0.02]* | no |

The decoded gap is LARGER under accelerate than under brake, at every level and both seeds, by 2.4 to 7.6 m: the wrong direction for the card, and larger than the largest shuffled draw (|point| at most 1.6 m) in all six rows. Read with LATSEM (CF-LON: accelerate raises the imagined SPEED) and the label relation below, the lead-gap probe most likely reads a speed correlate, not the physical constraint (HYPOTHESIS: not tested by an intervention here). L1 raw (unshrunk) is beside in the json and has the same sign (+5.2 / +6.4 m).

Beside (labels only, `beside_gap_speed.py`): over the 2,489 bank frames with a near lead, the lead gap and the ego speed correlate at r = +0.52 (folds +0.55 / +0.51), slope 0.74 m per m/s; the median gap is 9.8 / 19.1 / 26.8 m in the slow / middle / fast speed tercile. A latent that encodes speed (S1 and LATSEM R1) gives a gap probe the sign of this correlation, which is the sign S3 reads.

## Outcomes per level (as the card defines them; uncontrolled, NOT decisions while controls 3 and 4 stand failed)

| level | SCENE-AWARE | SCENE-IMAGINING | ACTION-SEMANTIC-LEAD |
|---|---|---|---|
| L1 | no | no | no (wrong sign) |
| L2 | no | no | no (wrong sign) |
| L3 | no | no | no (wrong sign) |

Reported outcome of the read: **INVALID (control 3, 4 failed)**. The table is the card's definitions applied to uncontrolled numbers.

## What this does not say

* Not a driving claim (T0). The planner reads none of these latents.
* Not that the scene is absent from the levels: the card's probe is a 128-d linear map with ridge lambda 10 fitted on 56 episodes' lead frames; the continuous gap and count targets are read at R2 near or below 0, while the binary and class targets are read with positive balanced accuracy.
* Each level's state is a pointwise function of ONE frame (w = 1), so closing speed and any rate are out of reach by construction (all six closing-speed rows are separated NEGATIVE, worse than the mean).
* A rollout that moves toward the label mean scores positive SKILL with a junk probe (the S2 permuted rows), so SKILL > 0 is not foresight by itself.
* MRATE's W0 / W1 do not exist yet (`theory/mrate/` is an empty tree): no beside rows on them.

## Next lever (the card's rule, then the evidence)

* The read is INVALID, so the card's rule does not fire. If the Master Mind amends controls 3 and 4 (a permutation reference on the point estimate instead of a not-separated interval, and R draws with a rate bar instead of one draw), the uncontrolled tables above give: no level SCENE-AWARE by the card's definition, so the card's next lever is MRATE's W1 and then a scene-target auxiliary (own card); S2 would need a reference that removes the variance effect before any SCENE-IMAGINING reading; S3 reads the wrong sign at all levels. An amendment written after these numbers were seen is post hoc and should say so; run 2 would recompute everything from the same inputs.
* The positive finding to carry: presence of a lead, the manoeuvre class and a turn ahead are linearly decodable from every level (balanced accuracy 0.08 to 0.24 above the majority baseline) while their magnitudes and rates are not. A scene-target auxiliary should therefore ask for the gap and the agent count, which the frozen levels do not hold.

## Pictures (decide nothing; seed s0)

| file | sha256 | content |
|---|---|---|
| `fig/pca_levels_scene.png` | `dc0ef99d1098fb73fd0a10b794e180229cb78ae056aaae53ed5b1f35ba5ea719` | per-level PCA maps coloured by lead gap and by agents within 30 m; no camera frame, safe to land |
| `fig/strip_CLOSE.png` | `9385359c599c1611abb19b527efe3a24a5bb454070416e86f42ba5feda926bfb` | retrieval strip, window 1539 (lead gap 0.0 m at frame 0); embeds downscaled camera frames: local only, not for the public repo |
| `fig/strip_FAR.png` | `bb61570c83394eb309e3b898eefd44d4939b7fb78741cba823a5de937c6d88ec` | retrieval strip, window 929 (lead gap 18.4 m at frame 0); embeds downscaled camera frames: local only, not for the public repo |

Strips show, per level, the frame now, the real future frame, and the nearest real frame (from ANOTHER episode) to the imagined state under accelerate and under brake, each titled with that frame's own lead gap and agent count. CLOSE is the S3 window with the smallest frame-0 gap: 0.0 m, a clipped negative gap with no lead visible in the frame (3 of the 2,489 gap labels are 0.0, 5 are at most 1 m: a label-quality flag at the extreme, not a finding); FAR is the one nearest the median gap.

