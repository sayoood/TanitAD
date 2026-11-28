## VERDICT (analyst-authored from `E1_DECOMP.json` of this run; every number is MEASURED from that file unless marked)

**Stamp for every number.**
* Tier: **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).** EARLY READ of the banked EMA step 30,000; the bar is decided at step 48,852.
* Set: S2 0-2 s, ceiling-compliant set (A-R8-CLAMP-2, mask sha `4ebc6b86…`), **4,159 windows / 138 episodes**. On this set capped == uncapped (control K).
* Estimator: paired episode-cluster bootstrap, B 2000. It answers "another draw of EPISODES" only. Not another training run: there is a single training seed. Not another inference run: seed 1 is shown beside seed 0.
* Delta: d = ADE(refcv8 `os`) − ADE(echo `ha0_ext`). The echo holds the measured (a0, κ0) at t0.
* Controls: M, R, K, Z, S, P and F all PASS.
  * R: the all-bin total reproduces the E1 cell exactly at both seeds. Seed 0 is −0.0085 [−0.0229, +0.0065] and seed 1 is −0.0076 [−0.0216, +0.0073], both n 4,159 / 138. It matches the banked tree json (capped and uncapped) and the brief's literals.
  * Z: the identity contrasts read exactly 0 in every bin.
  * P: 200 random partitions give 98.9 % / 98.6 % coverage.
  * F: force_lon's FB_all = 460 and CT pool = 1,279 are re-derived exactly.

### 1. E1 is a small net of two large, opposite parts

The total's CI spans 0, so a share of the net is not readable. The gross split is the reading.

| regime (the echo's own 2-s error; model-free) | n win / ep | contribution, seed 0 | seed 1 | echo ADE | refcv8 ADE |
|---|---|---|---|---|---|
| **echo-good** (\|along 2 s\| < 0.5 m and \|cross 2 s\| < 0.3 m) | 2,320 / 136 (55.8 %) | **+0.0264 [+0.0171, +0.0365] SEP** | +0.0280 [+0.0188, +0.0375] SEP | 0.127 | 0.174 |
| echo-lon-wrong only | 918 / 132 | −0.0123 [−0.0171, −0.0076] SEP | −0.0125 SEP | 0.466 | 0.411 |
| echo-lat-wrong only | 494 / 95 | −0.0066 [−0.0108, −0.0026] SEP | −0.0063 SEP | 0.333 | 0.278 |
| echo-both-wrong | 427 / 84 | −0.0161 [−0.0235, −0.0096] SEP | −0.0167 SEP | 0.657 | 0.499 |

* **Wherever the hold-action prior is wrong, refcv8 already beats it, separated, in every regime (−0.0350 in total).**
* All of E1's loss sits where the prior is right.
* The gap to separation is small: hi must move by −0.0065 m (seed 0) or −0.0073 m (seed 1). That is about 2.5 % of the echo's 0.281 m.

**"Cannot beat a constant-velocity prior" is the wrong frame.**
* Against the true constant-velocity arm `ha0` (v0 only, which is exactly the ego information refcv8 is fed), refcv8 wins by **−0.3817 [−0.4509, −0.3203] SEP**.
* refcv8 recovers from vision about all of what the measured (a0, κ0) buys over v0 alone. The capture ratio is about 1.0 overall: 0.654 → 0.272 m, against 0.654 → 0.281 for `ha0_ext`.
* The bar it fails is the hold-(a0, κ0) prior, and only where that prior is right.
* For reference, refcv7-50,400 on the same set, minus the echo, reads −0.0036 [−0.0209, +0.0145] (seed 0) and −0.0038 [−0.0203, +0.0139] (seed 1). It would fail E1 too.

### 2. Where refcv8 loses (the +0.0264): about 29 % is steady cruise, about 72 % is a near-term false go / under-brake

| cell | n win / ep | contribution, seed 0 [CI] | seed 1 | LON / LAT part | echo / refcv8 / refcv7 ADE | signed along-track at 2 s, refcv8 vs echo |
|---|---|---|---|---|---|---|
| echo-good, **dv2 cruising** (steady cruise) | 1,764 / 120 | +0.0076 [+0.0014, +0.0148] SEP | +0.0089 [+0.0028, +0.0156] SEP | +0.0024 / **+0.0052 SEP** | 0.139 / 0.156 / 0.149 | −0.053 vs −0.105 m |
| echo-good, **dv2 NOT cruising** (an ongoing speed change the echo holds correctly) | 556 / 71 | **+0.0189 [+0.0127, +0.0260] SEP** | +0.0191 [+0.0128, +0.0262] SEP | **+0.0171 SEP** / +0.0017 | **0.089 / 0.231** / 0.270 | **+0.252 vs +0.006 m (too far)** |

**Steady cruise.** In steady cruise refcv8 is within 0.017 m of the echo. Its excess there is mostly LATERAL.

**The speed-changing cell is the dominant loss.** Read by GT speed change over 2 s (dv2, `s6_lon_decomp`'s rule):

| dv2 class | n | contribution | refcv8 vs echo signed along at 2 s | ADE refcv8 / echo |
|---|---|---|---|---|
| slow_both | 376 | +0.0093 [+0.0049, +0.0148] SEP | +0.415 vs +0.126 m | 0.321 / 0.218 |
| stopping | 58 | +0.0042 [+0.0021, +0.0069] SEP | +0.845 vs +0.076 m; 91 % of windows too far | — |
| decelerating | 424 | +0.0066 [+0.0021, +0.0120] SEP | +0.667 vs +0.385 m | — |

By initial speed, **v0 < 2 m/s: +0.0082 [+0.0037, +0.0140] SEP** (ADE refcv8 0.326 / echo 0.249). Seed 1 gives the same picture in every row.

**refcv8 keeps moving, or moves too early, where the human stays slow or keeps decelerating.**

**The tactical head is associated with it. This is an association, NOT an intervention.**
* The lon head argmax is **ACCELERATE in 270 / 376 slow_both windows (72 %)** and in 311 / 447 windows with v0 < 2.
* But the GT `lon_v7` label, which covers the [NOW+2 s, NOW+8 s] window, **is also ACCELERATE in 111 / 161 labelled slow_both windows (69 %)**. The head agrees with the label in 60 % of labelled slow_both windows.
* So the head is mostly *right about 2-8 s*, and the plan executes the acceleration *already inside 0-2 s*.
* HYPOTHESIS (UNVERIFIED): a **timing** defect. The tactical class conditions the near-term plan without its start time.
* refcv7-50,400 is even worse in these windows (v0 < 2: 0.385; slow_both: 0.381). This is an inherited weakness that refcv8 has reduced but not closed.

### 3. Longitudinal vs lateral (exact two-factor Shapley split of d, ego frame at t0)

* **Total.** LON −0.0031 [−0.0142, +0.0090], LAT −0.0055 [−0.0134, +0.0024] (seed 1: −0.0027 / −0.0049). In the GT-tangent frame: LON −0.0019, LAT −0.0066.
* **Loss.** Of the echo-good loss, **LON is +0.0195 SEP (74 %)** and LAT +0.0070 SEP (26 %).
* **Win.** refcv8's \|cross-track\| at 2 s beats the echo: −0.0468 [−0.0746, −0.0220] SEP.
* **Not separated.** \|along\| at 2 s +0.0069 and speed MAE +0.0044.

### 4. The queued force_lon re-roll will NOT move E1 by itself

* **Its windows carry almost none of E1.** FB windows on the compliant S2 set are 235 selected + 196 not selected = 431 of FB_all's 460. They contribute **+0.0022 (seed 0) / +0.0024 (seed 1), not separated**. The whole head-BRAKE_TO set contributes +0.0005 [−0.0035, +0.0046] (seed 1 +0.0012).
* **Even tying the echo on every FB window does not pass E1.** The tie gives E1 −0.0108 [−0.0244, +0.0037] (seed 0) and −0.0100 [−0.0233, +0.0047] (seed 1). Neither is separated.
* **What passing through FB alone would take (ESTIMATED arithmetic).** refcv8 would have to beat the echo by about 0.04 m per FB window. Today it trails the echo there by 0.021 m (0.233 vs 0.210).
* **Free S2 read.** Its forced plans carry all 8 slots, so the S2 slots can be scored against `ha0_ext` at no roll cost. This is an addition to the analysis only.
* **Do not widen the FB/CT design.** Widen to the windows where E1 is actually lost, as a NEW S2 roll. The sets are defined on the compliant S2 set by the dv2 rule and the echo-regime rule of the module docstring. They are listed sha12-keyed in `E1_DECOMP.json` → `seeds.0.s2_widen_sets` once the gated re-run lands (see `HANDOFF.md`; regenerating them takes about 25 s of CPU):
  * **S2A**: dv2 ∈ {slow_both, stopping}, 434 windows, +0.0135.
  * **S2B**: dv2 decelerating, 424 windows, +0.0066.
  * **S2C**: echo-good and dv2 not cruising, 556 windows, +0.0189.
* **Arms (pre-register before any forced number):**
  * N (native);
  * F1 (oracle `lon_v7` → LON6);
  * **F_slow**: force HOLD on S2A and BRAKE_TO on S2B, via `set_r8_force(lon6=…)`. LON6 index 4 is HOLD and index 2 is BRAKE_TO.
  * Fown;
  * N2 (sampler-noise floor).
* **Prediction from the labels.** F1 ≈ N on S2A, because the oracle class there is mostly ACCELERATE. If F_slow closes ≥ 50 % of S2A's contribution while F1 does not, the near-term plan is controllable and the defect is the class's timing, not its identity.

### 5. Levers RUN in this turn (zero GPU, banked plans; Rule Zero)

| lever | E1 cell, seed 0 | seed 1 | vs raw refcv8 | verdict |
|---|---|---|---|---|
| **ENS2**: per-window mean of the seed-0 and seed-1 `os` plans (test-time, model outputs only) | **−0.0138 [−0.0279, +0.0012]** | (one read: it consumes both seeds) | −0.0052 [−0.0068, −0.0038] SEP and −0.0061 [−0.0075, −0.0048] SEP | **closest of all; NOT separated** |
| RT_XFIT: speed offset per (v0 bin × head argmax), 2 episode folds, DIAGNOSTIC | −0.0107 [−0.0248, +0.0037] | −0.0098 [−0.0237, +0.0046] | −0.0022 [−0.0055, +0.0009] (n.s.) | **FAILED** |
| MILESTONE: EMA 5k / 10.5k / 15k / 20k / 30k on the same windows (seed 0) | −0.0119 / **−0.0137 [−0.0273, +0.0003]** / −0.0108 / −0.0127 / −0.0085 | — | — | **no banked checkpoint passes** |

**ENS2.** The sampler noise is large next to the gap. The per-window \|ADE(seed 0) − ADE(seed 1)\| has mean 0.057 m and p90 0.131 m, and 39 % of windows differ by more than 0.05 m. The echo is deterministic.

**RT_XFIT.** On the fit folds the offset chooses −0.15 to −0.20 m/s for "v0 < 2 \| head ACCELERATE", which is the false go seen from the other side. A uniform offset cannot remove it.

**MILESTONE.** E1 regressed from 20k to 30k (−0.0127 → −0.0085). The regression is in the same bins:
* echo-good +0.0221 → +0.0264;
* steady cruise +0.0053 → +0.0076;
* S2C +0.0169 → +0.0189;
* slow_both +0.0073 → +0.0093.

These are seed-0 reads on a single training seed.

**ELIMINATED as share-carrying levers this turn (MEASURED):**
* **Missing measured a0.** The \|a0\| bins carry no positive contribution (gross +0.0000 / −0.0085).
* **Curves.** The yaw-rate bins carry no positive contribution (gross +0.0000).
* **The tactical BRAKE_TO false brake** (section 4).
* **A post-hoc speed offset** (RT_XFIT).
* **Checkpoint choice among the banked EMA milestones.**

### 6. Ranked levers (by measured contribution to E1, seed 0 / seed 1)

| rank | lever | measured share | counterfactual bound: E1 if the bin's mean excess is removed (SHIFT, spread kept; arithmetic, not a result) | next experiment | compute | status |
|---|---|---|---|---|---|---|
| 1 | **Near-term false go / under-brake in slow and decelerating windows** (S2C; the dv2 slow_both + stopping + decelerating rows) | +0.0189 / +0.0191 SEP (72 % of the gross loss) | S2C: −0.0274 [−0.0407, −0.0126] / −0.0267 [−0.0395, −0.0124], **SEP at both seeds**. slow_both alone: −0.0178 / −0.0170, SEP at both. v0 < 2 alone: SEP at both | the S2 force roll of section 4 (N, F1, F_slow, Fown, N2) on S2A/S2B. Then, if F_slow closes it, a time-aware tactical conditioning (the start-time constraint of R8-1) | CPU about 20 s per window for 3 forced forwards, so about 2.5 h for S2A and about 5 h for S2A + S2B; or one GPU roll | **BLOCKED**: the CPU and the commit headroom are held by the running force_lon roll (ETA about 22:45-23:15). It needs a NEW roll script, because `force_lon_reroll_r8.py` is a live-chain file (no edits) and has no HOLD arm. The fix itself is a PI retraining decision |
| 2 | **Steady-cruise precision plus sampler noise** (mostly LAT) | +0.0076 / +0.0089 SEP (29 %) | −0.0161 [−0.0304, −0.0010] / −0.0165 [−0.0305, −0.0011], SEP at both | K-draw ensemble (K = 4, 8) with an independent replicate (seeds 2-9). ENS2 already moves E1 by −0.005 to −0.006 SEP | GPU forward only (one battery roll per K-set); no training | **BLOCKED** on the GPU (the battery holds the lock). Inference-compute cost ×K, which is a deployment decision |
| 3 | 20k→30k drift in the same bins | E1 +0.0042 worse from 20k to 30k (seed 0) | — | the same decomposition on 40k and on 48,852 the moment they are banked (`e1_decomp_r8.py compute` with TREE switched; about 25 s). A schedule change would need a non-eval val split | CPU seconds | waits on the checkpoints |
| 4 | Tactical BRAKE_TO false brake (force_lon FB) | +0.0022 / +0.0024 (n.s.) | TIE: −0.0108 / −0.0100, **not separated** | keep it for S6/NR; add the free S2 read of its forced plans | none extra | queued roll; cannot pass E1 alone |

**Done-state under Rule Zero.**
* **BAR-R8-E1 is FAILED at this early read** (not separated at either seed).
* Three zero-GPU levers were run this turn. The closest, ENS2, reaches −0.0138 [−0.0279, +0.0012]: not separated, one read.
* Every remaining lever is blocked, and each blocker is named:
  * rank 1 by CPU/commit headroom plus a new roll script (then a PI retraining decision for the fix);
  * rank 2 by the GPU lock (then a deployment-cost decision);
  * rank 3 by checkpoints not yet banked.


---

# E1 decomposition -- refcv8 EMA-30k vs the hold-action echo on the ceiling-compliant S2 set

Tool `code/e1_decomp_r8.py` (sha256 `0d721c8967f7e930`), generated 2026-10-18T21:13:45.  **T1 self-action OPEN LOOP on logged eval139 frames (not driving performance).**  EARLY READ of a banked milestone (EMA step 30,000); the bars are decided at step 48,852.
Estimator: paired episode-cluster bootstrap over the episodes the compliant set contains, models FIXED, B 2000, seed 0 (draws identical to taniteval.ci._draws): answers 'another draw of EPISODES' only; NOT another training run (single training seed), NOT another inference run (seed 1 shown beside seed 0 as the inference replicate).
Bar: BAR-R8-E1: refcv8 os - ha0_ext, S2 ADE 0-2 s, < 0 and separated at both inference seeds, ceiling-compliant windows (A-R8-CLAMP-2, b6f7fac).

Controls: R_PASS PASS, Z_PASS PASS, S_PASS PASS, P_PASS PASS, K_PASS PASS, M_PASS PASS, F_PASS PASS, ALL_PASS PASS

## Inference seed 0

E1 cell (reproduced exactly): **-0.0085 [-0.0229, +0.0065]**, separated False, n 4,159 windows / 138 episodes.  Gap to separation (hi -> < 0): -0.0065 m.

| split | total [95 % CI] |
|---|---|
| d | -0.0085 [-0.0229, +0.0065] |
| d_lon | -0.0031 [-0.0142, +0.0090] |
| d_lat | -0.0055 [-0.0134, +0.0024] |
| tangent_frame_d_lon | -0.0019 [-0.0132, +0.0102] |
| tangent_frame_d_lat | -0.0066 [-0.0143, +0.0008] |

| component (os - echo) | delta [CI] | n |
|---|---|---|
| LON_along_mae_m | -0.0067 [-0.0188, +0.0062] | 4159 |
| LAT_cross_mae_m | -0.0032 [-0.0133, +0.0066] | 4159 |
| LON_speed_mae_mps | +0.0044 [-0.0091, +0.0185] | 4159 |
| LON_along_abs_final_m | +0.0069 [-0.0218, +0.0352] | 4159 |
| LAT_cross_abs_final_m | -0.0468 [-0.0746, -0.0220] SEP | 4159 |
| LAT_heading_mae_deg | -0.0465 [-0.1563, +0.0588] | 3913 |
| fde_m | -0.0280 [-0.0646, +0.0075] | 4159 |

### dv2 (seed 0)  -- gross: losses +0.0201, wins -0.0287 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| start_from_stop | 27 / 13 | 0.6 % | -0.0007 [-0.0018, +0.0001] | 9 % | -0.0007 [-0.0018, +0.0001] | -0.0000 [-0.0001, +0.0000] | -0.1153 [-0.2526, +0.0188] | 0.5404 | 0.4251 | 0.44 |
| stopping | 58 / 24 | 1.4 % | +0.0042 [+0.0021, +0.0069] SEP | -49 % | +0.0040 [+0.0019, +0.0067] SEP | +0.0002 [+0.0000, +0.0005] SEP | +0.3007 [+0.1889, +0.4389] | 0.2108 | 0.5115 | 0.84 |
| slow_both | 376 / 36 | 9.0 % | +0.0093 [+0.0049, +0.0148] SEP | -109 % | +0.0091 [+0.0048, +0.0145] SEP | +0.0003 [+0.0001, +0.0005] SEP | +0.1030 [+0.0687, +0.1550] | 0.2185 | 0.3214 | 0.54 |
| accelerating | 274 / 51 | 6.6 % | -0.0084 [-0.0140, -0.0041] SEP | 98 % | -0.0044 [-0.0071, -0.0021] SEP | -0.0040 [-0.0074, -0.0015] SEP | -0.1274 [-0.1974, -0.0701] | 0.5031 | 0.3758 | 0.36 |
| decelerating | 424 / 73 | 10.2 % | +0.0066 [+0.0021, +0.0120] SEP | -78 % | +0.0042 [+0.0009, +0.0075] SEP | +0.0024 [-0.0000, +0.0055] | +0.0652 [+0.0216, +0.1121] | 0.3895 | 0.4546 | 0.60 |
| cruising | 3000 / 126 | 72.1 % | -0.0195 [-0.0299, -0.0088] SEP | 229 % | -0.0152 [-0.0231, -0.0069] SEP | -0.0044 [-0.0106, +0.0022] | -0.0271 [-0.0412, -0.0121] | 0.2518 | 0.2247 | 0.40 |

### dv6 (seed 0)  -- gross: losses +0.0108, wins -0.0194 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| start_from_stop | 94 / 15 | 2.3 % | +0.0004 [-0.0012, +0.0020] | -4 % | +0.0004 [-0.0012, +0.0020] | -0.0000 [-0.0001, +0.0000] | +0.0162 [-0.0557, +0.0924] | 0.2335 | 0.2497 | 0.46 |
| stopping | 148 / 23 | 3.6 % | +0.0044 [+0.0016, +0.0078] SEP | -51 % | +0.0032 [+0.0009, +0.0062] SEP | +0.0011 [-0.0000, +0.0028] | +0.1226 [+0.0534, +0.2086] | 0.2903 | 0.4129 | 0.62 |
| slow_both | 165 / 21 | 4.0 % | +0.0044 [+0.0020, +0.0076] SEP | -52 % | +0.0043 [+0.0019, +0.0075] SEP | +0.0001 [+0.0000, +0.0003] SEP | +0.1119 [+0.0773, +0.1551] | 0.3721 | 0.4840 | 0.64 |
| accelerating | 661 / 73 | 15.9 % | -0.0088 [-0.0157, -0.0021] SEP | 103 % | -0.0074 [-0.0107, -0.0038] SEP | -0.0014 [-0.0063, +0.0039] | -0.0551 [-0.0984, -0.0134] | 0.4022 | 0.3471 | 0.39 |
| decelerating | 529 / 77 | 12.7 % | +0.0017 [-0.0010, +0.0045] | -19 % | +0.0018 [-0.0002, +0.0041] | -0.0001 [-0.0020, +0.0017] | +0.0130 [-0.0085, +0.0354] | 0.2911 | 0.3042 | 0.50 |
| cruising | 1654 / 119 | 39.8 % | -0.0097 [-0.0163, -0.0033] SEP | 114 % | -0.0062 [-0.0119, -0.0006] SEP | -0.0035 [-0.0069, -0.0002] SEP | -0.0244 [-0.0406, -0.0083] | 0.2351 | 0.2107 | 0.41 |
| no full 6-s future | 908 / 117 | 21.8 % | -0.0009 [-0.0067, +0.0060] | 10 % | +0.0008 [-0.0039, +0.0067] | -0.0017 [-0.0047, +0.0008] | -0.0041 [-0.0307, +0.0271] | 0.2559 | 0.2518 | 0.42 |

### v0bin (seed 0)  -- gross: losses +0.0119, wins -0.0205 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| v0<2 | 447 / 39 | 10.7 % | +0.0082 [+0.0037, +0.0140] SEP | -97 % | +0.0080 [+0.0036, +0.0137] SEP | +0.0002 [+0.0000, +0.0005] SEP | +0.0767 [+0.0409, +0.1222] | 0.2489 | 0.3256 | 0.53 |
| v0 2-8 | 1150 / 75 | 27.7 % | +0.0037 [-0.0042, +0.0119] | -43 % | +0.0040 [-0.0020, +0.0102] | -0.0003 [-0.0048, +0.0051] | +0.0133 [-0.0154, +0.0424] | 0.3257 | 0.3390 | 0.52 |
| v0 8-15 | 1468 / 81 | 35.3 % | -0.0048 [-0.0112, +0.0019] | 56 % | -0.0018 [-0.0057, +0.0023] | -0.0030 [-0.0076, +0.0017] | -0.0135 [-0.0321, +0.0057] | 0.2638 | 0.2504 | 0.46 |
| v0 15-25 | 827 / 31 | 19.9 % | -0.0082 [-0.0134, -0.0033] SEP | 96 % | -0.0059 [-0.0099, -0.0024] SEP | -0.0023 [-0.0057, +0.0009] | -0.0413 [-0.0605, -0.0195] | 0.2653 | 0.2240 | 0.33 |
| v0>=25 | 267 / 11 | 6.4 % | -0.0075 [-0.0130, -0.0029] SEP | 88 % | -0.0074 [-0.0127, -0.0030] SEP | -0.0001 [-0.0009, +0.0006] | -0.1166 [-0.1491, -0.0825] | 0.2807 | 0.1641 | 0.14 |

### yawbin (seed 0)  -- gross: losses +0.0000, wins -0.0085 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| straight <0.02 | 2800 / 130 | 67.3 % | -0.0030 [-0.0134, +0.0083] | 35 % | -0.0023 [-0.0123, +0.0083] | -0.0006 [-0.0029, +0.0017] | -0.0044 [-0.0198, +0.0125] | 0.2285 | 0.2241 | 0.44 |
| mild 0.02-0.08 | 763 / 96 | 18.3 % | -0.0023 [-0.0067, +0.0019] | 28 % | +0.0012 [-0.0020, +0.0042] | -0.0035 [-0.0067, -0.0006] SEP | -0.0128 [-0.0369, +0.0101] | 0.3305 | 0.3177 | 0.45 |
| curve 0.08-0.20 | 373 / 58 | 9.0 % | -0.0028 [-0.0081, +0.0028] | 33 % | -0.0011 [-0.0036, +0.0016] | -0.0018 [-0.0057, +0.0026] | -0.0314 [-0.0908, +0.0312] | 0.4627 | 0.4313 | 0.44 |
| sharp >=0.20 | 223 / 33 | 5.4 % | -0.0004 [-0.0053, +0.0054] | 5 % | -0.0008 [-0.0033, +0.0015] | +0.0004 [-0.0029, +0.0054] | -0.0076 [-0.1058, +0.1058] | 0.4615 | 0.4540 | 0.42 |

### tac_lon (seed 0)  -- gross: losses +0.0071, wins -0.0156 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| FOLLOW | 1079 / 59 | 25.9 % | -0.0035 [-0.0106, +0.0042] | 41 % | -0.0025 [-0.0087, +0.0038] | -0.0009 [-0.0052, +0.0029] | -0.0134 [-0.0407, +0.0172] | 0.2729 | 0.2595 | 0.42 |
| CRUISE | 1250 / 66 | 30.1 % | -0.0122 [-0.0190, -0.0055] SEP | 143 % | -0.0093 [-0.0150, -0.0038] SEP | -0.0029 [-0.0064, +0.0004] | -0.0405 [-0.0588, -0.0204] | 0.2534 | 0.2129 | 0.35 |
| BRAKE_TO | 1015 / 75 | 24.4 % | +0.0005 [-0.0035, +0.0046] | -6 % | +0.0026 [-0.0009, +0.0061] | -0.0021 [-0.0040, -0.0003] SEP | +0.0021 [-0.0145, +0.0181] | 0.2523 | 0.2544 | 0.53 |
| CREEP | 7 / 3 | 0.2 % | +0.0002 [+0.0000, +0.0005] | -3 % | +0.0002 [+0.0000, +0.0005] | +0.0000 [+0.0000, +0.0000] | +0.1287 [+0.0702, +0.3831] | 0.0799 | 0.2086 | 0.71 |
| HOLD | 60 / 6 | 1.4 % | +0.0022 [+0.0003, +0.0050] SEP | -26 % | +0.0022 [+0.0003, +0.0050] SEP | +0.0000 [+0.0000, +0.0001] SEP | +0.1511 [+0.0790, +0.2247] | 0.0872 | 0.2383 | 0.55 |
| ACCELERATE | 748 / 80 | 18.0 % | +0.0042 [-0.0039, +0.0133] | -49 % | +0.0038 [-0.0016, +0.0104] | +0.0004 [-0.0043, +0.0060] | +0.0234 [-0.0227, +0.0745] | 0.3935 | 0.4169 | 0.47 |

### tac_brake (seed 0)  -- gross: losses +0.0005, wins -0.0090 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| head BRAKE_TO | 1015 / 75 | 24.4 % | +0.0005 [-0.0035, +0.0046] | -6 % | +0.0026 [-0.0009, +0.0061] | -0.0021 [-0.0040, -0.0003] SEP | +0.0021 [-0.0145, +0.0181] | 0.2523 | 0.2544 | 0.53 |
| head not BRAKE_TO | 3144 / 128 | 75.6 % | -0.0090 [-0.0226, +0.0058] | 106 % | -0.0056 [-0.0162, +0.0054] | -0.0034 [-0.0110, +0.0041] | -0.0119 [-0.0299, +0.0079] | 0.2899 | 0.2779 | 0.41 |

### tac_correct (seed 0)  -- gross: losses +0.0070, wins -0.0155 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| head == lon_v7 | 1826 / 123 | 43.9 % | -0.0155 [-0.0236, -0.0075] SEP | 182 % | -0.0135 [-0.0192, -0.0078] SEP | -0.0020 [-0.0072, +0.0036] | -0.0353 [-0.0520, -0.0176] | 0.2795 | 0.2442 | 0.37 |
| head != lon_v7 | 1391 / 112 | 33.4 % | +0.0046 [-0.0028, +0.0128] | -54 % | +0.0054 [-0.0001, +0.0113] | -0.0008 [-0.0045, +0.0039] | +0.0138 [-0.0086, +0.0381] | 0.2791 | 0.2929 | 0.51 |
| unlabelled | 942 / 118 | 22.6 % | +0.0023 [-0.0043, +0.0094] | -28 % | +0.0050 [-0.0002, +0.0115] | -0.0027 [-0.0057, -0.0000] SEP | +0.0104 [-0.0190, +0.0410] | 0.2854 | 0.2958 | 0.46 |

### echo_regime (seed 0)  -- gross: losses +0.0264, wins -0.0350 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| echo-good | 2320 / 136 | 55.8 % | +0.0264 [+0.0171, +0.0365] SEP | -310 % | +0.0195 [+0.0105, +0.0289] SEP | +0.0070 [+0.0051, +0.0091] SEP | +0.0474 [+0.0305, +0.0660] | 0.1268 | 0.1742 | 0.53 |
| echo-lon-wrong only | 918 / 132 | 22.1 % | -0.0123 [-0.0171, -0.0076] SEP | 144 % | -0.0165 [-0.0211, -0.0120] SEP | +0.0042 [+0.0028, +0.0061] SEP | -0.0556 [-0.0742, -0.0354] | 0.4664 | 0.4108 | 0.35 |
| echo-lat-wrong only | 494 / 95 | 11.9 % | -0.0066 [-0.0108, -0.0026] SEP | 77 % | +0.0028 [+0.0014, +0.0042] SEP | -0.0094 [-0.0140, -0.0054] SEP | -0.0552 [-0.0845, -0.0237] | 0.3335 | 0.2783 | 0.31 |
| echo-both-wrong | 427 / 84 | 10.3 % | -0.0161 [-0.0235, -0.0096] SEP | 189 % | -0.0089 [-0.0124, -0.0058] SEP | -0.0073 [-0.0120, -0.0026] SEP | -0.1572 [-0.2143, -0.1006] | 0.6565 | 0.4993 | 0.25 |

### echo_ade (seed 0)  -- gross: losses +0.0333, wins -0.0418 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| echo ADE <0.1 | 889 / 122 | 21.4 % | +0.0232 [+0.0180, +0.0292] SEP | -272 % | +0.0202 [+0.0152, +0.0262] SEP | +0.0030 [+0.0022, +0.0041] SEP | +0.1085 [+0.0906, +0.1322] | 0.0551 | 0.1636 | 0.72 |
| echo ADE 0.1-0.2 | 1160 / 136 | 27.9 % | +0.0101 [+0.0061, +0.0146] SEP | -118 % | +0.0069 [+0.0034, +0.0110] SEP | +0.0032 [+0.0019, +0.0046] SEP | +0.0361 [+0.0211, +0.0537] | 0.1491 | 0.1852 | 0.49 |
| echo ADE 0.2-0.4 | 1308 / 137 | 31.4 % | -0.0118 [-0.0181, -0.0053] SEP | 139 % | -0.0113 [-0.0170, -0.0056] SEP | -0.0005 [-0.0028, +0.0021] | -0.0376 [-0.0550, -0.0180] | 0.2832 | 0.2456 | 0.31 |
| echo ADE 0.4-0.8 | 594 / 120 | 14.3 % | -0.0127 [-0.0177, -0.0079] SEP | 149 % | -0.0106 [-0.0137, -0.0079] SEP | -0.0021 [-0.0059, +0.0020] | -0.0892 [-0.1203, -0.0563] | 0.5407 | 0.4514 | 0.29 |
| echo ADE >=0.8 | 208 / 66 | 5.0 % | -0.0172 [-0.0238, -0.0116] SEP | 202 % | -0.0082 [-0.0111, -0.0055] SEP | -0.0090 [-0.0135, -0.0054] SEP | -0.3442 [-0.4249, -0.2614] | 1.2208 | 0.8766 | 0.20 |

### steady_x_dv (seed 0)  -- gross: losses +0.0271, wins -0.0356 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| echo-good | dv2 cruising | 1764 / 120 | 42.4 % | +0.0076 [+0.0014, +0.0148] SEP | -89 % | +0.0024 [-0.0032, +0.0088] | +0.0052 [+0.0035, +0.0072] SEP | +0.0179 [+0.0031, +0.0351] | 0.1386 | 0.1564 | 0.50 |
| echo-good | dv2 not cruising | 556 / 71 | 13.4 % | +0.0189 [+0.0127, +0.0260] SEP | -221 % | +0.0171 [+0.0112, +0.0240] SEP | +0.0017 [+0.0010, +0.0026] SEP | +0.1411 [+0.1128, +0.1773] | 0.0895 | 0.2305 | 0.65 |
| echo-lon-wrong only | dv2 cruising | 559 / 117 | 13.4 % | -0.0107 [-0.0142, -0.0073] SEP | 126 % | -0.0131 [-0.0164, -0.0099] SEP | +0.0024 [+0.0013, +0.0040] SEP | -0.0799 [-0.1005, -0.0561] | 0.3910 | 0.3111 | 0.29 |
| echo-lon-wrong only | dv2 not cruising | 359 / 76 | 8.6 % | -0.0015 [-0.0046, +0.0017] | 18 % | -0.0033 [-0.0063, -0.0003] SEP | +0.0018 [+0.0011, +0.0026] SEP | -0.0177 [-0.0529, +0.0205] | 0.5838 | 0.5661 | 0.45 |
| echo-lat-wrong only | dv2 cruising | 409 / 81 | 9.8 % | -0.0072 [-0.0112, -0.0035] SEP | 84 % | +0.0013 [+0.0001, +0.0025] SEP | -0.0085 [-0.0126, -0.0047] SEP | -0.0733 [-0.1058, -0.0382] | 0.3296 | 0.2563 | 0.27 |
| echo-lat-wrong only | dv2 not cruising | 85 / 38 | 2.0 % | +0.0006 [-0.0007, +0.0022] | -8 % | +0.0015 [+0.0008, +0.0023] SEP | -0.0009 [-0.0022, +0.0006] | +0.0318 [-0.0337, +0.1065] | 0.3523 | 0.3841 | 0.52 |
| echo-both-wrong | dv2 cruising | 268 / 74 | 6.4 % | -0.0092 [-0.0134, -0.0052] SEP | 108 % | -0.0056 [-0.0077, -0.0039] SEP | -0.0035 [-0.0065, -0.0003] SEP | -0.1422 [-0.1954, -0.0839] | 0.5882 | 0.4460 | 0.20 |
| echo-both-wrong | dv2 not cruising | 159 / 48 | 3.8 % | -0.0070 [-0.0125, -0.0024] SEP | 82 % | -0.0032 [-0.0056, -0.0011] SEP | -0.0038 [-0.0071, -0.0007] SEP | -0.1825 [-0.2835, -0.0720] | 0.7716 | 0.5891 | 0.32 |

### fl_set (seed 0)  -- gross: losses +0.0032, wins -0.0117 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| FB selected (force_lon) | 235 / 45 | 5.7 % | +0.0013 [-0.0001, +0.0028] | -15 % | +0.0018 [+0.0005, +0.0031] SEP | -0.0005 [-0.0010, -0.0001] SEP | +0.0232 [-0.0021, +0.0464] | 0.2096 | 0.2328 | 0.61 |
| FB not selected | 196 / 44 | 4.7 % | +0.0009 [-0.0003, +0.0021] | -11 % | +0.0013 [+0.0004, +0.0022] SEP | -0.0003 [-0.0012, +0.0003] | +0.0195 [-0.0060, +0.0426] | 0.2066 | 0.2260 | 0.58 |
| CT selected (force_lon) | 239 / 87 | 5.7 % | -0.0023 [-0.0041, -0.0005] SEP | 27 % | -0.0017 [-0.0028, -0.0005] SEP | -0.0006 [-0.0018, +0.0004] | -0.0398 [-0.0706, -0.0093] | 0.2545 | 0.2147 | 0.37 |
| head BRAKE_TO, S6, not GT-cruising | 374 / 57 | 9.0 % | -0.0027 [-0.0053, -0.0003] SEP | 32 % | -0.0015 [-0.0035, +0.0003] | -0.0012 [-0.0028, +0.0001] | -0.0300 [-0.0565, -0.0035] | 0.3145 | 0.2846 | 0.43 |
| head BRAKE_TO, no full 6-s future | 210 / 42 | 5.0 % | +0.0010 [-0.0005, +0.0025] | -11 % | +0.0010 [-0.0003, +0.0024] | -0.0000 [-0.0008, +0.0007] | +0.0192 [-0.0095, +0.0479] | 0.2321 | 0.2513 | 0.58 |
| other | 2905 / 128 | 69.8 % | -0.0067 [-0.0197, +0.0073] | 79 % | -0.0040 [-0.0142, +0.0067] | -0.0028 [-0.0100, +0.0043] | -0.0096 [-0.0287, +0.0107] | 0.2928 | 0.2831 | 0.41 |

### plan2_vs_gt2 (seed 0)  -- gross: losses +0.0180, wins -0.0265 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| GT decel / refcv8 decel | 240 / 52 | 5.8 % | +0.0043 [+0.0010, +0.0084] SEP | -50 % | +0.0027 [-0.0001, +0.0059] | +0.0015 [+0.0001, +0.0037] SEP | +0.0741 [+0.0171, +0.1348] | 0.2930 | 0.3671 | 0.57 |
| GT decel / refcv8 cruise | 160 / 54 | 3.8 % | +0.0064 [+0.0033, +0.0102] SEP | -75 % | +0.0057 [+0.0031, +0.0088] SEP | +0.0007 [-0.0004, +0.0019] | +0.1671 [+0.1023, +0.2326] | 0.4795 | 0.6467 | 0.71 |
| GT decel / refcv8 accel | 2 / 2 | 0.0 % | -0.0002 [-0.0005, +0.0000] | 2 % | -0.0001 [-0.0003, +0.0000] | -0.0001 [-0.0002, +0.0000] | -0.3278 [-0.6776, +0.0219] | 1.7277 | 1.3999 | 0.50 |
| GT cruise / refcv8 decel | 80 / 40 | 1.9 % | +0.0023 [+0.0002, +0.0049] SEP | -27 % | +0.0022 [+0.0002, +0.0045] SEP | +0.0001 [-0.0005, +0.0007] | +0.1199 [+0.0095, +0.2338] | 0.5006 | 0.6205 | 0.66 |
| GT cruise / refcv8 cruise | 3308 / 138 | 79.5 % | -0.0170 [-0.0270, -0.0065] SEP | 200 % | -0.0133 [-0.0212, -0.0052] SEP | -0.0037 [-0.0091, +0.0017] | -0.0214 [-0.0336, -0.0082] | 0.2426 | 0.2212 | 0.41 |
| GT cruise / refcv8 accel | 94 / 38 | 2.3 % | +0.0048 [+0.0012, +0.0095] SEP | -57 % | +0.0050 [+0.0018, +0.0096] SEP | -0.0001 [-0.0017, +0.0017] | +0.2144 [+0.0631, +0.3755] | 0.3031 | 0.5175 | 0.74 |
| GT accel / refcv8 decel | 1 / 1 | 0.0 % | +0.0001 [+0.0000, +0.0003] | -1 % | +0.0001 [+0.0000, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.4379 [+0.4379, +0.4379] | 0.8385 | 1.2764 | 1.00 |
| GT accel / refcv8 cruise | 101 / 39 | 2.4 % | -0.0025 [-0.0040, -0.0011] SEP | 29 % | -0.0019 [-0.0030, -0.0010] SEP | -0.0005 [-0.0014, +0.0002] | -0.1014 [-0.1569, -0.0523] | 0.6201 | 0.5187 | 0.33 |
| GT accel / refcv8 accel | 173 / 44 | 4.2 % | -0.0068 [-0.0120, -0.0032] SEP | 80 % | -0.0034 [-0.0057, -0.0015] SEP | -0.0034 [-0.0068, -0.0011] SEP | -0.1646 [-0.2606, -0.0838] | 0.4767 | 0.3121 | 0.37 |

### a0bin (seed 0)  -- gross: losses +0.0000, wins -0.0085 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| |a0| <0.3 | 2061 / 136 | 49.6 % | -0.0043 [-0.0119, +0.0044] | 50 % | -0.0020 [-0.0090, +0.0057] | -0.0023 [-0.0049, +0.0004] | -0.0086 [-0.0234, +0.0093] | 0.2261 | 0.2175 | 0.41 |
| |a0| 0.3-1.0 | 1417 / 128 | 34.1 % | -0.0019 [-0.0089, +0.0053] | 22 % | -0.0001 [-0.0048, +0.0048] | -0.0018 [-0.0062, +0.0028] | -0.0054 [-0.0262, +0.0158] | 0.3030 | 0.2976 | 0.46 |
| |a0| 1.0-2.0 | 522 / 92 | 12.6 % | -0.0024 [-0.0080, +0.0030] | 28 % | -0.0009 [-0.0048, +0.0031] | -0.0015 [-0.0044, +0.0015] | -0.0192 [-0.0652, +0.0236] | 0.3943 | 0.3751 | 0.47 |
| |a0| >=2.0 | 159 / 39 | 3.8 % | +0.0000 [-0.0023, +0.0023] | -0 % | -0.0001 [-0.0021, +0.0022] | +0.0001 [-0.0009, +0.0009] | +0.0000 [-0.0655, +0.0626] | 0.4159 | 0.4159 | 0.55 |

### Counterfactual lever arithmetic (seed 0) -- NOT results: what E1 would read if refcv8's excess in the bin were removed

| bin | n | contrib | SHIFT (excess removed, spread kept) | TIE (== echo on the bin) |
|---|---|---|---|---|
| echo_regime :: echo-good | 2320 | +0.0264 | -0.0350 [-0.0487, -0.0198] SEP<0 | -0.0350 [-0.0451, -0.0251] SEP<0 |
| UNION :: echo-good (steady) all | 2320 | +0.0264 | -0.0350 [-0.0487, -0.0198] SEP<0 | -0.0350 [-0.0451, -0.0251] SEP<0 |
| echo_ade :: echo ADE <0.1 | 889 | +0.0232 | -0.0317 [-0.0445, -0.0179] SEP<0 | -0.0317 [-0.0435, -0.0194] SEP<0 |
| steady_x_dv :: echo-good | dv2 not cruising | 556 | +0.0189 | -0.0274 [-0.0407, -0.0126] SEP<0 | -0.0274 [-0.0396, -0.0145] SEP<0 |
| echo_ade :: echo ADE 0.1-0.2 | 1160 | +0.0101 | -0.0186 [-0.0328, -0.0033] SEP<0 | -0.0186 [-0.0308, -0.0053] SEP<0 |
| dv2 :: slow_both | 376 | +0.0093 | -0.0178 [-0.0315, -0.0032] SEP<0 | -0.0178 [-0.0306, -0.0039] SEP<0 |
| v0bin :: v0<2 | 447 | +0.0082 | -0.0168 [-0.0305, -0.0020] SEP<0 | -0.0168 [-0.0294, -0.0031] SEP<0 |
| steady_x_dv :: echo-good | dv2 cruising | 1764 | +0.0076 | -0.0161 [-0.0304, -0.0010] SEP<0 | -0.0161 [-0.0285, -0.0037] SEP<0 |
| UNION :: echo-good AND dv2 cruising | 1764 | +0.0076 | -0.0161 [-0.0304, -0.0010] SEP<0 | -0.0161 [-0.0285, -0.0037] SEP<0 |
| dv2 :: decelerating | 424 | +0.0066 | -0.0152 [-0.0293, -0.0002] SEP<0 | -0.0152 [-0.0290, -0.0005] SEP<0 |
| tac_correct :: head != lon_v7 | 1391 | +0.0046 | -0.0131 [-0.0273, +0.0016] | -0.0131 [-0.0243, -0.0015] SEP<0 |
| dv6 :: slow_both | 165 | +0.0044 | -0.0130 [-0.0270, +0.0020] | -0.0130 [-0.0269, +0.0017] |
| dv6 :: stopping | 148 | +0.0044 | -0.0129 [-0.0266, +0.0019] | -0.0129 [-0.0264, +0.0012] |
| tac_lon :: ACCELERATE | 748 | +0.0042 | -0.0127 [-0.0269, +0.0022] | -0.0127 [-0.0238, -0.0008] SEP<0 |
| dv2 :: stopping | 58 | +0.0042 | -0.0127 [-0.0266, +0.0020] | -0.0127 [-0.0265, +0.0017] |
| v0bin :: v0 2-8 | 1150 | +0.0037 | -0.0122 [-0.0263, +0.0027] | -0.0122 [-0.0232, -0.0007] SEP<0 |
| tac_correct :: unlabelled | 942 | +0.0023 | -0.0109 [-0.0251, +0.0042] | -0.0109 [-0.0228, +0.0018] |
| UNION :: force_lon FB_all (selected + not selected) | 431 | +0.0022 | -0.0108 [-0.0252, +0.0041] | -0.0108 [-0.0244, +0.0037] |
| tac_lon :: HOLD | 60 | +0.0022 | -0.0107 [-0.0248, +0.0042] | -0.0107 [-0.0250, +0.0041] |
| dv6 :: decelerating | 529 | +0.0017 | -0.0102 [-0.0245, +0.0049] | -0.0102 [-0.0238, +0.0041] |
| fl_set :: FB selected (force_lon) | 235 | +0.0013 | -0.0098 [-0.0242, +0.0052] | -0.0098 [-0.0239, +0.0048] |
| fl_set :: head BRAKE_TO, no full 6-s future | 210 | +0.0010 | -0.0095 [-0.0239, +0.0056] | -0.0095 [-0.0237, +0.0057] |
| fl_set :: FB not selected | 196 | +0.0009 | -0.0094 [-0.0238, +0.0057] | -0.0094 [-0.0235, +0.0055] |
| steady_x_dv :: echo-lat-wrong only | dv2 not cruising | 85 | +0.0006 | -0.0092 [-0.0236, +0.0060] | -0.0092 [-0.0235, +0.0057] |
| tac_lon :: BRAKE_TO | 1015 | +0.0005 | -0.0090 [-0.0234, +0.0060] | -0.0090 [-0.0226, +0.0058] |
| tac_brake :: head BRAKE_TO | 1015 | +0.0005 | -0.0090 [-0.0234, +0.0060] | -0.0090 [-0.0226, +0.0058] |
| UNION :: head BRAKE_TO (all compliant) | 1015 | +0.0005 | -0.0090 [-0.0234, +0.0060] | -0.0090 [-0.0226, +0.0058] |
| dv6 :: start_from_stop | 94 | +0.0004 | -0.0089 [-0.0232, +0.0062] | -0.0089 [-0.0230, +0.0062] |
| tac_lon :: CREEP | 7 | +0.0002 | -0.0087 [-0.0230, +0.0064] | -0.0087 [-0.0230, +0.0064] |
| a0bin :: |a0| >=2.0 | 159 | +0.0000 | -0.0085 [-0.0229, +0.0065] | -0.0085 [-0.0222, +0.0066] |
| fl_set :: CT selected (force_lon) | 239 | -0.0023 | -0.0062 [-0.0205, +0.0087] | -0.0062 [-0.0199, +0.0082] |
| fl_set :: head BRAKE_TO, S6, not GT-cruising | 374 | -0.0027 | -0.0058 [-0.0202, +0.0094] | -0.0058 [-0.0200, +0.0090] |
| fl_set :: other | 2905 | -0.0067 | -0.0018 [-0.0162, +0.0132] | -0.0018 [-0.0061, +0.0025] |

## Inference seed 1

E1 cell (reproduced exactly): **-0.0076 [-0.0216, +0.0073]**, separated False, n 4,159 windows / 138 episodes.  Gap to separation (hi -> < 0): -0.0073 m.

| split | total [95 % CI] |
|---|---|
| d | -0.0076 [-0.0216, +0.0073] |
| d_lon | -0.0027 [-0.0133, +0.0090] |
| d_lat | -0.0049 [-0.0128, +0.0030] |
| tangent_frame_d_lon | -0.0017 [-0.0126, +0.0103] |
| tangent_frame_d_lat | -0.0059 [-0.0135, +0.0016] |

| component (os - echo) | delta [CI] | n |
|---|---|---|
| LON_along_mae_m | -0.0058 [-0.0176, +0.0068] | 4159 |
| LAT_cross_mae_m | -0.0029 [-0.0130, +0.0071] | 4159 |
| LON_speed_mae_mps | +0.0055 [-0.0077, +0.0191] | 4159 |
| LON_along_abs_final_m | +0.0095 [-0.0183, +0.0377] | 4159 |
| LAT_cross_abs_final_m | -0.0461 [-0.0739, -0.0207] SEP | 4159 |
| LAT_heading_mae_deg | -0.0420 [-0.1506, +0.0650] | 3912 |
| fde_m | -0.0255 [-0.0609, +0.0095] | 4159 |

### dv2 (seed 1)  -- gross: losses +0.0201, wins -0.0277 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| start_from_stop | 27 / 13 | 0.6 % | -0.0008 [-0.0018, -0.0000] SEP | 10 % | -0.0007 [-0.0017, +0.0000] | -0.0000 [-0.0001, +0.0000] | -0.1199 [-0.2416, -0.0028] | 0.5404 | 0.4205 | 0.37 |
| stopping | 58 / 24 | 1.4 % | +0.0043 [+0.0022, +0.0069] SEP | -56 % | +0.0041 [+0.0021, +0.0066] SEP | +0.0002 [+0.0000, +0.0004] SEP | +0.3053 [+0.1983, +0.4372] | 0.2108 | 0.5160 | 0.83 |
| slow_both | 376 / 36 | 9.0 % | +0.0094 [+0.0050, +0.0149] SEP | -123 % | +0.0091 [+0.0048, +0.0146] SEP | +0.0002 [+0.0001, +0.0005] SEP | +0.1037 [+0.0679, +0.1562] | 0.2185 | 0.3222 | 0.55 |
| accelerating | 274 / 51 | 6.6 % | -0.0086 [-0.0142, -0.0043] SEP | 113 % | -0.0051 [-0.0076, -0.0028] SEP | -0.0035 [-0.0068, -0.0010] SEP | -0.1303 [-0.1976, -0.0718] | 0.5031 | 0.3729 | 0.32 |
| decelerating | 424 / 73 | 10.2 % | +0.0065 [+0.0022, +0.0114] SEP | -85 % | +0.0041 [+0.0010, +0.0072] SEP | +0.0024 [-0.0001, +0.0055] | +0.0634 [+0.0219, +0.1067] | 0.3895 | 0.4529 | 0.58 |
| cruising | 3000 / 126 | 72.1 % | -0.0184 [-0.0286, -0.0078] SEP | 241 % | -0.0141 [-0.0219, -0.0061] SEP | -0.0042 [-0.0103, +0.0023] | -0.0254 [-0.0392, -0.0108] | 0.2518 | 0.2264 | 0.41 |

### dv6 (seed 1)  -- gross: losses +0.0112, wins -0.0188 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| start_from_stop | 94 / 15 | 2.3 % | +0.0007 [-0.0009, +0.0024] | -9 % | +0.0007 [-0.0009, +0.0024] | -0.0000 [-0.0001, +0.0000] | +0.0294 [-0.0419, +0.1042] | 0.2335 | 0.2629 | 0.47 |
| stopping | 148 / 23 | 3.6 % | +0.0043 [+0.0016, +0.0076] SEP | -57 % | +0.0031 [+0.0009, +0.0058] SEP | +0.0012 [-0.0000, +0.0030] | +0.1215 [+0.0548, +0.2010] | 0.2903 | 0.4118 | 0.61 |
| slow_both | 165 / 21 | 4.0 % | +0.0043 [+0.0019, +0.0073] SEP | -56 % | +0.0042 [+0.0018, +0.0071] SEP | +0.0001 [+0.0000, +0.0003] SEP | +0.1085 [+0.0665, +0.1604] | 0.3721 | 0.4806 | 0.64 |
| accelerating | 661 / 73 | 15.9 % | -0.0092 [-0.0162, -0.0025] SEP | 121 % | -0.0079 [-0.0115, -0.0043] SEP | -0.0013 [-0.0063, +0.0041] | -0.0581 [-0.1006, -0.0157] | 0.4022 | 0.3440 | 0.36 |
| decelerating | 529 / 77 | 12.7 % | +0.0019 [-0.0007, +0.0045] | -24 % | +0.0018 [-0.0002, +0.0039] | +0.0001 [-0.0018, +0.0020] | +0.0147 [-0.0055, +0.0352] | 0.2911 | 0.3058 | 0.52 |
| cruising | 1654 / 119 | 39.8 % | -0.0086 [-0.0148, -0.0022] SEP | 113 % | -0.0051 [-0.0107, +0.0004] | -0.0035 [-0.0068, -0.0001] SEP | -0.0216 [-0.0374, -0.0055] | 0.2351 | 0.2135 | 0.41 |
| no full 6-s future | 908 / 117 | 21.8 % | -0.0009 [-0.0066, +0.0056] | 12 % | +0.0007 [-0.0037, +0.0064] | -0.0016 [-0.0046, +0.0010] | -0.0043 [-0.0302, +0.0251] | 0.2559 | 0.2516 | 0.43 |

### v0bin (seed 1)  -- gross: losses +0.0116, wins -0.0192 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| v0<2 | 447 / 39 | 10.7 % | +0.0081 [+0.0038, +0.0138] SEP | -107 % | +0.0079 [+0.0037, +0.0135] SEP | +0.0002 [+0.0000, +0.0004] SEP | +0.0758 [+0.0425, +0.1198] | 0.2489 | 0.3247 | 0.53 |
| v0 2-8 | 1150 / 75 | 27.7 % | +0.0034 [-0.0045, +0.0117] | -45 % | +0.0033 [-0.0029, +0.0095] | +0.0001 [-0.0046, +0.0054] | +0.0124 [-0.0167, +0.0415] | 0.3257 | 0.3380 | 0.50 |
| v0 8-15 | 1468 / 81 | 35.3 % | -0.0039 [-0.0104, +0.0025] | 52 % | -0.0013 [-0.0051, +0.0025] | -0.0026 [-0.0072, +0.0021] | -0.0112 [-0.0297, +0.0076] | 0.2638 | 0.2527 | 0.47 |
| v0 15-25 | 827 / 31 | 19.9 % | -0.0078 [-0.0126, -0.0032] SEP | 103 % | -0.0054 [-0.0090, -0.0020] SEP | -0.0025 [-0.0058, +0.0006] | -0.0394 [-0.0572, -0.0190] | 0.2653 | 0.2260 | 0.33 |
| v0>=25 | 267 / 11 | 6.4 % | -0.0074 [-0.0128, -0.0030] SEP | 97 % | -0.0072 [-0.0124, -0.0030] SEP | -0.0002 [-0.0009, +0.0004] | -0.1156 [-0.1409, -0.0895] | 0.2807 | 0.1651 | 0.16 |

### yawbin (seed 1)  -- gross: losses +0.0000, wins -0.0076 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| straight <0.02 | 2800 / 130 | 67.3 % | -0.0027 [-0.0132, +0.0083] | 36 % | -0.0023 [-0.0121, +0.0084] | -0.0005 [-0.0029, +0.0019] | -0.0041 [-0.0193, +0.0125] | 0.2285 | 0.2244 | 0.43 |
| mild 0.02-0.08 | 763 / 96 | 18.3 % | -0.0013 [-0.0058, +0.0031] | 17 % | +0.0020 [-0.0011, +0.0050] | -0.0033 [-0.0063, -0.0004] SEP | -0.0071 [-0.0325, +0.0166] | 0.3305 | 0.3234 | 0.47 |
| curve 0.08-0.20 | 373 / 58 | 9.0 % | -0.0033 [-0.0085, +0.0021] | 43 % | -0.0013 [-0.0039, +0.0011] | -0.0020 [-0.0060, +0.0024] | -0.0369 [-0.0944, +0.0226] | 0.4627 | 0.4259 | 0.43 |
| sharp >=0.20 | 223 / 33 | 5.4 % | -0.0003 [-0.0051, +0.0056] | 4 % | -0.0011 [-0.0034, +0.0010] | +0.0008 [-0.0027, +0.0059] | -0.0052 [-0.1029, +0.1104] | 0.4615 | 0.4563 | 0.39 |

### tac_lon (seed 1)  -- gross: losses +0.0076, wins -0.0152 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| FOLLOW | 1079 / 59 | 25.9 % | -0.0043 [-0.0113, +0.0033] | 56 % | -0.0035 [-0.0094, +0.0024] | -0.0008 [-0.0048, +0.0028] | -0.0164 [-0.0426, +0.0125] | 0.2729 | 0.2564 | 0.41 |
| CRUISE | 1250 / 66 | 30.1 % | -0.0110 [-0.0178, -0.0046] SEP | 144 % | -0.0080 [-0.0132, -0.0029] SEP | -0.0030 [-0.0065, +0.0003] | -0.0365 [-0.0541, -0.0167] | 0.2534 | 0.2169 | 0.36 |
| BRAKE_TO | 1015 / 75 | 24.4 % | +0.0012 [-0.0026, +0.0050] | -16 % | +0.0032 [-0.0003, +0.0066] | -0.0020 [-0.0040, -0.0002] SEP | +0.0049 [-0.0108, +0.0207] | 0.2523 | 0.2572 | 0.54 |
| CREEP | 7 / 3 | 0.2 % | +0.0002 [+0.0000, +0.0006] | -3 % | +0.0002 [+0.0000, +0.0005] | +0.0000 [+0.0000, +0.0000] | +0.1428 [+0.0852, +0.4047] | 0.0799 | 0.2227 | 0.71 |
| HOLD | 60 / 6 | 1.4 % | +0.0022 [+0.0002, +0.0050] SEP | -28 % | +0.0021 [+0.0002, +0.0049] SEP | +0.0000 [+0.0000, +0.0001] SEP | +0.1504 [+0.0654, +0.2129] | 0.0872 | 0.2376 | 0.52 |
| ACCELERATE | 748 / 80 | 18.0 % | +0.0040 [-0.0045, +0.0133] | -52 % | +0.0031 [-0.0022, +0.0099] | +0.0009 [-0.0040, +0.0065] | +0.0222 [-0.0244, +0.0736] | 0.3935 | 0.4157 | 0.46 |

### tac_brake (seed 1)  -- gross: losses +0.0012, wins -0.0088 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| head BRAKE_TO | 1015 / 75 | 24.4 % | +0.0012 [-0.0026, +0.0050] | -16 % | +0.0032 [-0.0003, +0.0066] | -0.0020 [-0.0040, -0.0002] SEP | +0.0049 [-0.0108, +0.0207] | 0.2523 | 0.2572 | 0.54 |
| head not BRAKE_TO | 3144 / 128 | 75.6 % | -0.0088 [-0.0220, +0.0061] | 116 % | -0.0059 [-0.0158, +0.0049] | -0.0029 [-0.0105, +0.0047] | -0.0117 [-0.0294, +0.0083] | 0.2899 | 0.2782 | 0.40 |

### tac_correct (seed 1)  -- gross: losses +0.0070, wins -0.0146 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| head == lon_v7 | 1826 / 123 | 43.9 % | -0.0146 [-0.0226, -0.0064] SEP | 191 % | -0.0132 [-0.0189, -0.0075] SEP | -0.0014 [-0.0067, +0.0043] | -0.0332 [-0.0502, -0.0151] | 0.2795 | 0.2463 | 0.38 |
| head != lon_v7 | 1391 / 112 | 33.4 % | +0.0048 [-0.0026, +0.0130] | -63 % | +0.0058 [+0.0001, +0.0112] SEP | -0.0010 [-0.0046, +0.0038] | +0.0144 [-0.0079, +0.0379] | 0.2791 | 0.2936 | 0.51 |
| unlabelled | 942 / 118 | 22.6 % | +0.0021 [-0.0044, +0.0092] | -28 % | +0.0047 [-0.0004, +0.0112] | -0.0026 [-0.0057, +0.0001] | +0.0094 [-0.0200, +0.0411] | 0.2854 | 0.2948 | 0.45 |

### echo_regime (seed 1)  -- gross: losses +0.0280, wins -0.0356 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| echo-good | 2320 / 136 | 55.8 % | +0.0280 [+0.0188, +0.0375] SEP | -367 % | +0.0206 [+0.0118, +0.0298] SEP | +0.0074 [+0.0055, +0.0097] SEP | +0.0501 [+0.0338, +0.0685] | 0.1268 | 0.1770 | 0.54 |
| echo-lon-wrong only | 918 / 132 | 22.1 % | -0.0125 [-0.0173, -0.0081] SEP | 164 % | -0.0168 [-0.0213, -0.0125] SEP | +0.0043 [+0.0029, +0.0060] SEP | -0.0567 [-0.0746, -0.0373] | 0.4664 | 0.4097 | 0.35 |
| echo-lat-wrong only | 494 / 95 | 11.9 % | -0.0063 [-0.0105, -0.0025] SEP | 83 % | +0.0030 [+0.0016, +0.0044] SEP | -0.0093 [-0.0138, -0.0054] SEP | -0.0534 [-0.0835, -0.0228] | 0.3335 | 0.2801 | 0.31 |
| echo-both-wrong | 427 / 84 | 10.3 % | -0.0167 [-0.0242, -0.0100] SEP | 220 % | -0.0094 [-0.0129, -0.0064] SEP | -0.0073 [-0.0120, -0.0026] SEP | -0.1631 [-0.2198, -0.1068] | 0.6565 | 0.4935 | 0.24 |

### echo_ade (seed 1)  -- gross: losses +0.0343, wins -0.0420 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| echo ADE <0.1 | 889 / 122 | 21.4 % | +0.0235 [+0.0182, +0.0296] SEP | -308 % | +0.0205 [+0.0154, +0.0266] SEP | +0.0030 [+0.0022, +0.0040] SEP | +0.1098 [+0.0909, +0.1340] | 0.0551 | 0.1649 | 0.72 |
| echo ADE 0.1-0.2 | 1160 / 136 | 27.9 % | +0.0109 [+0.0068, +0.0154] SEP | -143 % | +0.0073 [+0.0037, +0.0111] SEP | +0.0036 [+0.0022, +0.0052] SEP | +0.0390 [+0.0237, +0.0567] | 0.1491 | 0.1881 | 0.49 |
| echo ADE 0.2-0.4 | 1308 / 137 | 31.4 % | -0.0107 [-0.0167, -0.0047] SEP | 141 % | -0.0104 [-0.0160, -0.0048] SEP | -0.0003 [-0.0026, +0.0022] | -0.0342 [-0.0504, -0.0153] | 0.2832 | 0.2490 | 0.31 |
| echo ADE 0.4-0.8 | 594 / 120 | 14.3 % | -0.0138 [-0.0188, -0.0088] SEP | 181 % | -0.0117 [-0.0148, -0.0090] SEP | -0.0022 [-0.0060, +0.0020] | -0.0967 [-0.1273, -0.0632] | 0.5407 | 0.4439 | 0.29 |
| echo ADE >=0.8 | 208 / 66 | 5.0 % | -0.0174 [-0.0241, -0.0117] SEP | 228 % | -0.0083 [-0.0112, -0.0056] SEP | -0.0091 [-0.0136, -0.0054] SEP | -0.3477 [-0.4242, -0.2716] | 1.2208 | 0.8731 | 0.19 |

### steady_x_dv (seed 1)  -- gross: losses +0.0285, wins -0.0361 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| echo-good | dv2 cruising | 1764 / 120 | 42.4 % | +0.0089 [+0.0028, +0.0156] SEP | -116 % | +0.0037 [-0.0020, +0.0099] | +0.0052 [+0.0035, +0.0072] SEP | +0.0209 [+0.0063, +0.0381] | 0.1386 | 0.1595 | 0.51 |
| echo-good | dv2 not cruising | 556 / 71 | 13.4 % | +0.0191 [+0.0128, +0.0262] SEP | -251 % | +0.0169 [+0.0110, +0.0236] SEP | +0.0022 [+0.0013, +0.0033] SEP | +0.1429 [+0.1144, +0.1791] | 0.0895 | 0.2324 | 0.63 |
| echo-lon-wrong only | dv2 cruising | 559 / 117 | 13.4 % | -0.0107 [-0.0142, -0.0074] SEP | 140 % | -0.0131 [-0.0164, -0.0099] SEP | +0.0024 [+0.0013, +0.0039] SEP | -0.0795 [-0.1002, -0.0575] | 0.3910 | 0.3114 | 0.29 |
| echo-lon-wrong only | dv2 not cruising | 359 / 76 | 8.6 % | -0.0018 [-0.0048, +0.0014] | 24 % | -0.0037 [-0.0065, -0.0008] SEP | +0.0019 [+0.0012, +0.0028] SEP | -0.0211 [-0.0539, +0.0158] | 0.5838 | 0.5628 | 0.44 |
| echo-lat-wrong only | dv2 cruising | 409 / 81 | 9.8 % | -0.0069 [-0.0110, -0.0032] SEP | 90 % | +0.0014 [+0.0003, +0.0026] SEP | -0.0083 [-0.0124, -0.0045] SEP | -0.0700 [-0.1023, -0.0356] | 0.3296 | 0.2596 | 0.26 |
| echo-lat-wrong only | dv2 not cruising | 85 / 38 | 2.0 % | +0.0005 [-0.0007, +0.0021] | -7 % | +0.0016 [+0.0008, +0.0024] SEP | -0.0010 [-0.0024, +0.0005] | +0.0264 [-0.0375, +0.1029] | 0.3523 | 0.3787 | 0.52 |
| echo-both-wrong | dv2 cruising | 268 / 74 | 6.4 % | -0.0096 [-0.0141, -0.0057] SEP | 127 % | -0.0061 [-0.0082, -0.0044] SEP | -0.0035 [-0.0065, -0.0003] SEP | -0.1497 [-0.2040, -0.0912] | 0.5882 | 0.4385 | 0.21 |
| echo-both-wrong | dv2 not cruising | 159 / 48 | 3.8 % | -0.0071 [-0.0125, -0.0025] SEP | 93 % | -0.0033 [-0.0057, -0.0013] SEP | -0.0038 [-0.0070, -0.0008] SEP | -0.1855 [-0.2855, -0.0756] | 0.7716 | 0.5861 | 0.29 |

### fl_set (seed 1)  -- gross: losses +0.0034, wins -0.0110 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| FB selected (force_lon) | 235 / 45 | 5.7 % | +0.0015 [+0.0000, +0.0030] SEP | -19 % | +0.0018 [+0.0005, +0.0032] SEP | -0.0004 [-0.0009, +0.0000] | +0.0258 [+0.0002, +0.0500] | 0.2096 | 0.2354 | 0.61 |
| FB not selected | 196 / 44 | 4.7 % | +0.0009 [-0.0004, +0.0021] | -12 % | +0.0013 [+0.0004, +0.0022] SEP | -0.0004 [-0.0013, +0.0003] | +0.0191 [-0.0083, +0.0424] | 0.2066 | 0.2257 | 0.60 |
| CT selected (force_lon) | 239 / 87 | 5.7 % | -0.0020 [-0.0038, -0.0003] SEP | 27 % | -0.0013 [-0.0026, -0.0002] SEP | -0.0007 [-0.0019, +0.0004] | -0.0352 [-0.0658, -0.0062] | 0.2545 | 0.2194 | 0.37 |
| head BRAKE_TO, S6, not GT-cruising | 374 / 57 | 9.0 % | -0.0022 [-0.0048, +0.0003] | 29 % | -0.0010 [-0.0032, +0.0010] | -0.0012 [-0.0028, +0.0001] | -0.0247 [-0.0521, +0.0038] | 0.3145 | 0.2898 | 0.45 |
| head BRAKE_TO, no full 6-s future | 210 / 42 | 5.0 % | +0.0011 [-0.0004, +0.0026] | -14 % | +0.0011 [-0.0002, +0.0025] | -0.0000 [-0.0009, +0.0007] | +0.0209 [-0.0080, +0.0503] | 0.2321 | 0.2530 | 0.57 |
| other | 2905 / 128 | 69.8 % | -0.0068 [-0.0193, +0.0073] | 89 % | -0.0046 [-0.0138, +0.0056] | -0.0022 [-0.0093, +0.0050] | -0.0097 [-0.0278, +0.0107] | 0.2928 | 0.2830 | 0.41 |

### plan2_vs_gt2 (seed 1)  -- gross: losses +0.0168, wins -0.0244 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| GT decel / refcv8 decel | 243 / 54 | 5.8 % | +0.0046 [+0.0013, +0.0087] SEP | -61 % | +0.0031 [+0.0004, +0.0062] SEP | +0.0015 [+0.0000, +0.0036] SEP | +0.0791 [+0.0241, +0.1391] | 0.2913 | 0.3703 | 0.56 |
| GT decel / refcv8 cruise | 158 / 54 | 3.8 % | +0.0060 [+0.0030, +0.0096] SEP | -79 % | +0.0052 [+0.0028, +0.0080] SEP | +0.0008 [-0.0003, +0.0023] | +0.1589 [+0.0968, +0.2212] | 0.4909 | 0.6498 | 0.69 |
| GT decel / refcv8 accel | 1 / 1 | 0.0 % | -0.0001 [-0.0005, +0.0000] | 2 % | -0.0001 [-0.0003, +0.0000] | -0.0001 [-0.0002, +0.0000] | -0.6177 [-0.6177, -0.6177] | 2.1511 | 1.5334 | 0.00 |
| GT cruise / refcv8 decel | 78 / 43 | 1.9 % | +0.0018 [-0.0003, +0.0044] | -24 % | +0.0019 [+0.0001, +0.0041] SEP | -0.0001 [-0.0008, +0.0006] | +0.0965 [-0.0155, +0.2144] | 0.5198 | 0.6163 | 0.59 |
| GT cruise / refcv8 cruise | 3309 / 138 | 79.6 % | -0.0150 [-0.0248, -0.0048] SEP | 196 % | -0.0117 [-0.0199, -0.0037] SEP | -0.0033 [-0.0085, +0.0022] | -0.0188 [-0.0310, -0.0060] | 0.2416 | 0.2228 | 0.41 |
| GT cruise / refcv8 accel | 95 / 37 | 2.3 % | +0.0042 [+0.0005, +0.0091] SEP | -56 % | +0.0046 [+0.0014, +0.0093] SEP | -0.0003 [-0.0018, +0.0014] | +0.1858 [+0.0246, +0.3575] | 0.3250 | 0.5108 | 0.73 |
| GT accel / refcv8 decel | 1 / 1 | 0.0 % | +0.0001 [+0.0000, +0.0003] | -1 % | +0.0001 [+0.0000, +0.0003] | +0.0000 [+0.0000, +0.0000] | +0.3548 [+0.3548, +0.3548] | 0.8385 | 1.1934 | 1.00 |
| GT accel / refcv8 cruise | 107 / 40 | 2.6 % | -0.0028 [-0.0044, -0.0013] SEP | 36 % | -0.0023 [-0.0034, -0.0012] SEP | -0.0005 [-0.0013, +0.0002] | -0.1074 [-0.1621, -0.0571] | 0.6107 | 0.5033 | 0.30 |
| GT accel / refcv8 accel | 167 / 44 | 4.0 % | -0.0066 [-0.0118, -0.0029] SEP | 86 % | -0.0035 [-0.0056, -0.0018] SEP | -0.0030 [-0.0063, -0.0006] SEP | -0.1631 [-0.2642, -0.0801] | 0.4776 | 0.3145 | 0.32 |

### a0bin (seed 1)  -- gross: losses +0.0000, wins -0.0076 m

| bin | n win / ep | frac | contrib ADE [CI] | share | contrib LON | contrib LAT | bin mean d [CI] | echo ADE | refcv8 ADE | P(refcv8 worse) |
|---|---|---|---|---|---|---|---|---|---|---|
| |a0| <0.3 | 2061 / 136 | 49.6 % | -0.0025 [-0.0099, +0.0058] | 33 % | -0.0005 [-0.0074, +0.0072] | -0.0020 [-0.0046, +0.0006] | -0.0050 [-0.0196, +0.0121] | 0.2261 | 0.2211 | 0.41 |
| |a0| 0.3-1.0 | 1417 / 128 | 34.1 % | -0.0027 [-0.0098, +0.0045] | 36 % | -0.0004 [-0.0052, +0.0046] | -0.0023 [-0.0068, +0.0023] | -0.0079 [-0.0284, +0.0133] | 0.3030 | 0.2951 | 0.46 |
| |a0| 1.0-2.0 | 522 / 92 | 12.6 % | -0.0022 [-0.0077, +0.0032] | 29 % | -0.0015 [-0.0054, +0.0025] | -0.0007 [-0.0036, +0.0024] | -0.0178 [-0.0621, +0.0256] | 0.3943 | 0.3766 | 0.44 |
| |a0| >=2.0 | 159 / 39 | 3.8 % | -0.0002 [-0.0024, +0.0021] | 3 % | -0.0002 [-0.0022, +0.0018] | +0.0000 [-0.0009, +0.0009] | -0.0051 [-0.0681, +0.0556] | 0.4159 | 0.4108 | 0.54 |

### Counterfactual lever arithmetic (seed 1) -- NOT results: what E1 would read if refcv8's excess in the bin were removed

| bin | n | contrib | SHIFT (excess removed, spread kept) | TIE (== echo on the bin) |
|---|---|---|---|---|
| echo_regime :: echo-good | 2320 | +0.0280 | -0.0356 [-0.0490, -0.0206] SEP<0 | -0.0356 [-0.0458, -0.0259] SEP<0 |
| UNION :: echo-good (steady) all | 2320 | +0.0280 | -0.0356 [-0.0490, -0.0206] SEP<0 | -0.0356 [-0.0458, -0.0259] SEP<0 |
| echo_ade :: echo ADE <0.1 | 889 | +0.0235 | -0.0311 [-0.0437, -0.0174] SEP<0 | -0.0311 [-0.0427, -0.0189] SEP<0 |
| steady_x_dv :: echo-good | dv2 not cruising | 556 | +0.0191 | -0.0267 [-0.0395, -0.0124] SEP<0 | -0.0267 [-0.0389, -0.0141] SEP<0 |
| echo_ade :: echo ADE 0.1-0.2 | 1160 | +0.0109 | -0.0185 [-0.0322, -0.0032] SEP<0 | -0.0185 [-0.0309, -0.0048] SEP<0 |
| dv2 :: slow_both | 376 | +0.0094 | -0.0170 [-0.0301, -0.0022] SEP<0 | -0.0170 [-0.0296, -0.0035] SEP<0 |
| steady_x_dv :: echo-good | dv2 cruising | 1764 | +0.0089 | -0.0165 [-0.0305, -0.0011] SEP<0 | -0.0165 [-0.0287, -0.0042] SEP<0 |
| UNION :: echo-good AND dv2 cruising | 1764 | +0.0089 | -0.0165 [-0.0305, -0.0011] SEP<0 | -0.0165 [-0.0287, -0.0042] SEP<0 |
| v0bin :: v0<2 | 447 | +0.0081 | -0.0158 [-0.0290, -0.0010] SEP<0 | -0.0158 [-0.0282, -0.0021] SEP<0 |
| dv2 :: decelerating | 424 | +0.0065 | -0.0141 [-0.0279, +0.0004] | -0.0141 [-0.0276, +0.0004] |
| tac_correct :: head != lon_v7 | 1391 | +0.0048 | -0.0124 [-0.0262, +0.0021] | -0.0124 [-0.0233, -0.0013] SEP<0 |
| dv6 :: stopping | 148 | +0.0043 | -0.0119 [-0.0255, +0.0024] | -0.0119 [-0.0250, +0.0022] |
| dv6 :: slow_both | 165 | +0.0043 | -0.0119 [-0.0255, +0.0031] | -0.0119 [-0.0255, +0.0028] |
| dv2 :: stopping | 58 | +0.0043 | -0.0119 [-0.0254, +0.0024] | -0.0119 [-0.0251, +0.0022] |
| tac_lon :: ACCELERATE | 748 | +0.0040 | -0.0116 [-0.0254, +0.0032] | -0.0116 [-0.0222, -0.0002] SEP<0 |
| v0bin :: v0 2-8 | 1150 | +0.0034 | -0.0110 [-0.0250, +0.0035] | -0.0110 [-0.0217, +0.0000] |
| UNION :: force_lon FB_all (selected + not selected) | 431 | +0.0024 | -0.0100 [-0.0238, +0.0049] | -0.0100 [-0.0233, +0.0047] |
| tac_lon :: HOLD | 60 | +0.0022 | -0.0098 [-0.0236, +0.0050] | -0.0098 [-0.0235, +0.0049] |
| tac_correct :: unlabelled | 942 | +0.0021 | -0.0097 [-0.0237, +0.0053] | -0.0097 [-0.0215, +0.0027] |
| dv6 :: decelerating | 529 | +0.0019 | -0.0095 [-0.0233, +0.0054] | -0.0095 [-0.0230, +0.0046] |
| fl_set :: FB selected (force_lon) | 235 | +0.0015 | -0.0091 [-0.0230, +0.0059] | -0.0091 [-0.0226, +0.0059] |
| tac_lon :: BRAKE_TO | 1015 | +0.0012 | -0.0088 [-0.0229, +0.0061] | -0.0088 [-0.0220, +0.0061] |
| tac_brake :: head BRAKE_TO | 1015 | +0.0012 | -0.0088 [-0.0229, +0.0061] | -0.0088 [-0.0220, +0.0061] |
| UNION :: head BRAKE_TO (all compliant) | 1015 | +0.0012 | -0.0088 [-0.0229, +0.0061] | -0.0088 [-0.0220, +0.0061] |
| fl_set :: head BRAKE_TO, no full 6-s future | 210 | +0.0011 | -0.0087 [-0.0227, +0.0063] | -0.0087 [-0.0227, +0.0065] |
| fl_set :: FB not selected | 196 | +0.0009 | -0.0085 [-0.0225, +0.0064] | -0.0085 [-0.0222, +0.0064] |
| dv6 :: start_from_stop | 94 | +0.0007 | -0.0083 [-0.0222, +0.0067] | -0.0083 [-0.0225, +0.0064] |
| steady_x_dv :: echo-lat-wrong only | dv2 not cruising | 85 | +0.0005 | -0.0082 [-0.0222, +0.0068] | -0.0082 [-0.0218, +0.0065] |
| tac_lon :: CREEP | 7 | +0.0002 | -0.0079 [-0.0217, +0.0069] | -0.0079 [-0.0218, +0.0070] |
| fl_set :: CT selected (force_lon) | 239 | -0.0020 | -0.0056 [-0.0196, +0.0092] | -0.0056 [-0.0190, +0.0084] |
| fl_set :: head BRAKE_TO, S6, not GT-cruising | 374 | -0.0022 | -0.0054 [-0.0192, +0.0098] | -0.0054 [-0.0190, +0.0095] |
| fl_set :: other | 2905 | -0.0068 | -0.0008 [-0.0149, +0.0141] | -0.0008 [-0.0051, +0.0033] |

## Per-bin levels (seed 0): echo / refcv8 / constant-velocity `ha0` / refcv7-50,400 ADE, signed along-track at 2 s, head argmax

| partition :: bin | n | echo | refcv8 | CV ha0 | refcv7 | refcv8 signed along 2 s | echo signed along 2 s | CV-capture | head argmax | GT lon_v7 |
|---|---|---|---|---|---|---|---|---|---|---|
| dv2 :: start_from_stop | 27 | 0.540 | 0.425 | 1.067 | 0.440 | -0.075 | -1.043 | 1.22 | {'FOLLOW': 9, 'HOLD': 1, 'ACCELERATE': 17} | {'unlabelled': 6, 'FOLLOW': 3, 'CRUISE': 1, 'ACCELERATE': 17} |
| dv2 :: stopping | 58 | 0.211 | 0.511 | 1.569 | 0.501 | +0.845 | +0.076 | 0.78 | {'FOLLOW': 8, 'CRUISE': 5, 'BRAKE_TO': 11, 'HOLD': 9, 'ACCELERATE': 25} | {'unlabelled': 18, 'BRAKE_TO': 12, 'CREEP': 8, 'HOLD': 7, 'ACCELERATE': 13} |
| dv2 :: slow_both | 376 | 0.218 | 0.321 | 0.359 | 0.381 | +0.415 | +0.126 | 0.27 | {'FOLLOW': 58, 'CRUISE': 1, 'BRAKE_TO': 1, 'CREEP': 2, 'HOLD': 44, 'ACCELERATE': 270} | {'unlabelled': 215, 'FOLLOW': 1, 'CRUISE': 17, 'CREEP': 8, 'HOLD': 24, 'ACCELERATE': 111} |
| dv2 :: accelerating | 274 | 0.503 | 0.376 | 1.344 | 0.384 | -0.312 | -0.629 | 1.15 | {'FOLLOW': 80, 'CRUISE': 40, 'BRAKE_TO': 51, 'ACCELERATE': 103} | {'unlabelled': 31, 'FOLLOW': 73, 'CRUISE': 41, 'BRAKE_TO': 29, 'ACCELERATE': 100} |
| dv2 :: decelerating | 424 | 0.389 | 0.455 | 1.307 | 0.481 | +0.667 | +0.385 | 0.93 | {'FOLLOW': 89, 'CRUISE': 82, 'BRAKE_TO': 164, 'HOLD': 5, 'ACCELERATE': 84} | {'unlabelled': 54, 'FOLLOW': 32, 'CRUISE': 31, 'BRAKE_TO': 208, 'CREEP': 18, 'ACCELERATE': 81} |
| dv2 :: cruising | 3000 | 0.252 | 0.225 | 0.514 | 0.220 | -0.102 | -0.180 | 1.10 | {'FOLLOW': 835, 'CRUISE': 1122, 'BRAKE_TO': 788, 'CREEP': 5, 'HOLD': 1, 'ACCELERATE': 249} | {'unlabelled': 618, 'FOLLOW': 541, 'CRUISE': 1068, 'BRAKE_TO': 427, 'CREEP': 1, 'ACCELERATE': 345} |
| v0bin :: v0<2 | 447 | 0.249 | 0.326 | 0.436 | 0.385 | +0.359 | +0.007 | 0.59 | {'FOLLOW': 77, 'CRUISE': 8, 'BRAKE_TO': 2, 'CREEP': 3, 'HOLD': 46, 'ACCELERATE': 311} | {'unlabelled': 233, 'FOLLOW': 10, 'CRUISE': 25, 'CREEP': 8, 'HOLD': 24, 'ACCELERATE': 147} |
| v0bin :: v0 2-8 | 1150 | 0.326 | 0.339 | 0.800 | 0.342 | +0.128 | -0.131 | 0.97 | {'FOLLOW': 230, 'CRUISE': 123, 'BRAKE_TO': 451, 'CREEP': 4, 'HOLD': 14, 'ACCELERATE': 328} | {'unlabelled': 170, 'FOLLOW': 147, 'CRUISE': 228, 'BRAKE_TO': 281, 'CREEP': 27, 'HOLD': 7, 'ACCELERATE': 290} |
| v0bin :: v0 8-15 | 1468 | 0.264 | 0.250 | 0.701 | 0.247 | -0.049 | -0.066 | 1.03 | {'FOLLOW': 343, 'CRUISE': 553, 'BRAKE_TO': 477, 'ACCELERATE': 95} | {'unlabelled': 245, 'FOLLOW': 236, 'CRUISE': 553, 'BRAKE_TO': 301, 'ACCELERATE': 133} |
| v0bin :: v0 15-25 | 827 | 0.265 | 0.224 | 0.577 | 0.216 | -0.145 | -0.218 | 1.13 | {'FOLLOW': 281, 'CRUISE': 448, 'BRAKE_TO': 85, 'ACCELERATE': 13} | {'unlabelled': 237, 'FOLLOW': 158, 'CRUISE': 245, 'BRAKE_TO': 94, 'ACCELERATE': 93} |
| v0bin :: v0>=25 | 267 | 0.281 | 0.164 | 0.367 | 0.174 | -0.083 | -0.374 | 2.35 | {'FOLLOW': 148, 'CRUISE': 118, 'ACCELERATE': 1} | {'unlabelled': 57, 'FOLLOW': 99, 'CRUISE': 107, 'ACCELERATE': 4} |
| echo_regime :: echo-good | 2320 | 0.127 | 0.174 | 0.462 | 0.178 | +0.020 | -0.079 | 0.86 | {'FOLLOW': 601, 'CRUISE': 760, 'BRAKE_TO': 570, 'CREEP': 6, 'HOLD': 52, 'ACCELERATE': 331} | {'unlabelled': 555, 'FOLLOW': 339, 'CRUISE': 765, 'BRAKE_TO': 344, 'CREEP': 19, 'HOLD': 29, 'ACCELERATE': 269} |
| echo_regime :: echo-lon-wrong only | 918 | 0.466 | 0.411 | 0.732 | 0.417 | +0.046 | -0.163 | 1.21 | {'FOLLOW': 261, 'CRUISE': 203, 'BRAKE_TO': 268, 'CREEP': 1, 'HOLD': 8, 'ACCELERATE': 177} | {'unlabelled': 198, 'FOLLOW': 178, 'CRUISE': 181, 'BRAKE_TO': 166, 'CREEP': 16, 'HOLD': 2, 'ACCELERATE': 177} |
| echo_regime :: echo-lat-wrong only | 494 | 0.333 | 0.278 | 0.915 | 0.291 | +0.010 | -0.074 | 1.09 | {'FOLLOW': 125, 'CRUISE': 176, 'BRAKE_TO': 111, 'ACCELERATE': 82} | {'unlabelled': 103, 'FOLLOW': 74, 'CRUISE': 139, 'BRAKE_TO': 94, 'ACCELERATE': 84} |
| echo_regime :: echo-both-wrong | 427 | 0.657 | 0.499 | 1.224 | 0.501 | -0.001 | -0.366 | 1.28 | {'FOLLOW': 92, 'CRUISE': 111, 'BRAKE_TO': 66, 'ACCELERATE': 158} | {'unlabelled': 86, 'FOLLOW': 59, 'CRUISE': 73, 'BRAKE_TO': 72, 'ACCELERATE': 137} |
| steady_x_dv :: echo-good | dv2 cruising | 1764 | 0.139 | 0.156 | 0.389 | 0.149 | -0.053 | -0.105 | 0.93 | {'FOLLOW': 480, 'CRUISE': 732, 'BRAKE_TO': 487, 'CREEP': 4, 'HOLD': 1, 'ACCELERATE': 60} | {'unlabelled': 354, 'FOLLOW': 295, 'CRUISE': 733, 'BRAKE_TO': 258, 'ACCELERATE': 124} |
| steady_x_dv :: echo-good | dv2 not cruising | 556 | 0.089 | 0.231 | 0.696 | 0.270 | +0.252 | +0.006 | 0.77 | {'FOLLOW': 121, 'CRUISE': 28, 'BRAKE_TO': 83, 'CREEP': 2, 'HOLD': 51, 'ACCELERATE': 271} | {'unlabelled': 201, 'FOLLOW': 44, 'CRUISE': 32, 'BRAKE_TO': 86, 'CREEP': 19, 'HOLD': 29, 'ACCELERATE': 145} |
| steady_x_dv :: echo-lon-wrong only | dv2 cruising | 559 | 0.391 | 0.311 | 0.450 | 0.299 | -0.259 | -0.371 | 2.35 | {'FOLLOW': 182, 'CRUISE': 166, 'BRAKE_TO': 166, 'CREEP': 1, 'ACCELERATE': 44} | {'unlabelled': 117, 'FOLLOW': 144, 'CRUISE': 156, 'BRAKE_TO': 67, 'CREEP': 1, 'ACCELERATE': 74} |
| steady_x_dv :: echo-lon-wrong only | dv2 not cruising | 359 | 0.584 | 0.566 | 1.171 | 0.600 | +0.519 | +0.160 | 1.03 | {'FOLLOW': 79, 'CRUISE': 37, 'BRAKE_TO': 102, 'HOLD': 8, 'ACCELERATE': 133} | {'unlabelled': 81, 'FOLLOW': 34, 'CRUISE': 25, 'BRAKE_TO': 99, 'CREEP': 15, 'HOLD': 2, 'ACCELERATE': 103} |
| steady_x_dv :: echo-lat-wrong only | dv2 cruising | 409 | 0.330 | 0.256 | 0.806 | 0.266 | -0.027 | -0.084 | 1.15 | {'FOLLOW': 110, 'CRUISE': 150, 'BRAKE_TO': 88, 'ACCELERATE': 61} | {'unlabelled': 85, 'FOLLOW': 65, 'CRUISE': 123, 'BRAKE_TO': 69, 'ACCELERATE': 67} |
| steady_x_dv :: echo-lat-wrong only | dv2 not cruising | 85 | 0.352 | 0.384 | 1.443 | 0.412 | +0.189 | -0.023 | 0.97 | {'FOLLOW': 15, 'CRUISE': 26, 'BRAKE_TO': 23, 'ACCELERATE': 21} | {'unlabelled': 18, 'FOLLOW': 9, 'CRUISE': 16, 'BRAKE_TO': 25, 'ACCELERATE': 17} |
| steady_x_dv :: echo-both-wrong | dv2 cruising | 268 | 0.588 | 0.446 | 1.027 | 0.452 | -0.217 | -0.419 | 1.32 | {'FOLLOW': 63, 'CRUISE': 74, 'BRAKE_TO': 47, 'ACCELERATE': 84} | {'unlabelled': 62, 'FOLLOW': 37, 'CRUISE': 56, 'BRAKE_TO': 33, 'ACCELERATE': 80} |
| steady_x_dv :: echo-both-wrong | dv2 not cruising | 159 | 0.772 | 0.589 | 1.556 | 0.584 | +0.363 | -0.276 | 1.23 | {'FOLLOW': 29, 'CRUISE': 37, 'BRAKE_TO': 19, 'ACCELERATE': 74} | {'unlabelled': 24, 'FOLLOW': 22, 'CRUISE': 17, 'BRAKE_TO': 39, 'ACCELERATE': 57} |
| a0bin :: |a0| <0.3 | 2061 | 0.226 | 0.218 | 0.346 | 0.220 | -0.024 | -0.114 | 1.07 | {'FOLLOW': 506, 'CRUISE': 756, 'BRAKE_TO': 504, 'CREEP': 5, 'HOLD': 24, 'ACCELERATE': 266} | {'unlabelled': 514, 'FOLLOW': 311, 'CRUISE': 742, 'BRAKE_TO': 264, 'CREEP': 5, 'HOLD': 11, 'ACCELERATE': 214} |
| a0bin :: |a0| 0.3-1.0 | 1417 | 0.303 | 0.298 | 0.688 | 0.303 | +0.014 | -0.109 | 1.01 | {'FOLLOW': 417, 'CRUISE': 367, 'BRAKE_TO': 361, 'CREEP': 1, 'HOLD': 25, 'ACCELERATE': 246} | {'unlabelled': 317, 'FOLLOW': 253, 'CRUISE': 341, 'BRAKE_TO': 234, 'CREEP': 7, 'HOLD': 9, 'ACCELERATE': 256} |
| a0bin :: |a0| 1.0-2.0 | 522 | 0.394 | 0.375 | 1.317 | 0.388 | +0.193 | -0.138 | 1.02 | {'FOLLOW': 119, 'CRUISE': 96, 'BRAKE_TO': 116, 'HOLD': 11, 'ACCELERATE': 180} | {'unlabelled': 76, 'FOLLOW': 71, 'CRUISE': 63, 'BRAKE_TO': 132, 'CREEP': 18, 'HOLD': 8, 'ACCELERATE': 154} |
| a0bin :: |a0| >=2.0 | 159 | 0.416 | 0.416 | 2.157 | 0.426 | +0.144 | -0.403 | 1.00 | {'FOLLOW': 37, 'CRUISE': 31, 'BRAKE_TO': 34, 'CREEP': 1, 'ACCELERATE': 56} | {'unlabelled': 35, 'FOLLOW': 15, 'CRUISE': 12, 'BRAKE_TO': 46, 'CREEP': 5, 'HOLD': 3, 'ACCELERATE': 43} |
| tac_brake :: head BRAKE_TO | 1015 | 0.252 | 0.254 | 0.544 | 0.241 | -0.076 | +0.018 | 0.99 | {'BRAKE_TO': 1015} | {'unlabelled': 184, 'FOLLOW': 28, 'CRUISE': 364, 'BRAKE_TO': 333, 'CREEP': 7, 'HOLD': 1, 'ACCELERATE': 98} |
| tac_brake :: head not BRAKE_TO | 3144 | 0.290 | 0.278 | 0.689 | 0.289 | +0.055 | -0.173 | 1.03 | {'FOLLOW': 1079, 'CRUISE': 1250, 'CREEP': 7, 'HOLD': 60, 'ACCELERATE': 748} | {'unlabelled': 758, 'FOLLOW': 622, 'CRUISE': 794, 'BRAKE_TO': 343, 'CREEP': 28, 'HOLD': 30, 'ACCELERATE': 569} |
| fl_set :: FB selected (force_lon) | 235 | 0.210 | 0.233 | 0.386 | 0.198 | -0.231 | -0.082 | 0.87 | {'BRAKE_TO': 235} | {'unlabelled': 14, 'FOLLOW': 6, 'CRUISE': 155, 'BRAKE_TO': 39, 'ACCELERATE': 21} |
| fl_set :: FB not selected | 196 | 0.207 | 0.226 | 0.398 | 0.190 | -0.262 | -0.132 | 0.90 | {'BRAKE_TO': 196} | {'unlabelled': 11, 'FOLLOW': 7, 'CRUISE': 125, 'BRAKE_TO': 30, 'ACCELERATE': 23} |
| fl_set :: CT selected (force_lon) | 239 | 0.255 | 0.215 | 0.526 | 0.210 | +0.017 | -0.146 | 1.15 | {'FOLLOW': 89, 'CRUISE': 133, 'CREEP': 1, 'ACCELERATE': 16} | {'unlabelled': 10, 'FOLLOW': 67, 'CRUISE': 128, 'BRAKE_TO': 15, 'ACCELERATE': 19} |
| fl_set :: head BRAKE_TO, S6, not GT-cruising | 374 | 0.315 | 0.285 | 0.764 | 0.291 | +0.096 | +0.120 | 1.07 | {'BRAKE_TO': 374} | {'unlabelled': 8, 'FOLLOW': 13, 'CRUISE': 61, 'BRAKE_TO': 234, 'CREEP': 5, 'HOLD': 1, 'ACCELERATE': 52} |
| fl_set :: head BRAKE_TO, no full 6-s future | 210 | 0.232 | 0.251 | 0.466 | 0.251 | -0.037 | +0.089 | 0.92 | {'BRAKE_TO': 210} | {'unlabelled': 151, 'FOLLOW': 2, 'CRUISE': 23, 'BRAKE_TO': 30, 'CREEP': 2, 'ACCELERATE': 2} |
| fl_set :: other | 2905 | 0.293 | 0.283 | 0.703 | 0.295 | +0.058 | -0.175 | 1.02 | {'FOLLOW': 990, 'CRUISE': 1117, 'CREEP': 6, 'HOLD': 60, 'ACCELERATE': 732} | {'unlabelled': 748, 'FOLLOW': 555, 'CRUISE': 666, 'BRAKE_TO': 328, 'CREEP': 28, 'HOLD': 30, 'ACCELERATE': 550} |
| echo_ade :: echo ADE <0.1 | 889 | 0.055 | 0.164 | 0.382 | 0.173 | +0.089 | +0.005 | 0.67 | {'FOLLOW': 181, 'CRUISE': 220, 'BRAKE_TO': 231, 'CREEP': 5, 'HOLD': 43, 'ACCELERATE': 209} | {'unlabelled': 246, 'FOLLOW': 93, 'CRUISE': 255, 'BRAKE_TO': 133, 'CREEP': 8, 'HOLD': 23, 'ACCELERATE': 131} |
| echo_ade :: echo ADE 0.1-0.2 | 1160 | 0.149 | 0.185 | 0.513 | 0.182 | +0.012 | -0.060 | 0.90 | {'FOLLOW': 297, 'CRUISE': 422, 'BRAKE_TO': 331, 'CREEP': 1, 'HOLD': 7, 'ACCELERATE': 102} | {'unlabelled': 248, 'FOLLOW': 185, 'CRUISE': 399, 'BRAKE_TO': 199, 'CREEP': 11, 'HOLD': 2, 'ACCELERATE': 116} |
| echo_ade :: echo ADE 0.2-0.4 | 1308 | 0.283 | 0.246 | 0.676 | 0.253 | -0.007 | -0.175 | 1.10 | {'FOLLOW': 416, 'CRUISE': 426, 'BRAKE_TO': 283, 'CREEP': 1, 'HOLD': 8, 'ACCELERATE': 174} | {'unlabelled': 289, 'FOLLOW': 238, 'CRUISE': 386, 'BRAKE_TO': 187, 'CREEP': 9, 'HOLD': 4, 'ACCELERATE': 195} |
| echo_ade :: echo ADE 0.4-0.8 | 594 | 0.541 | 0.451 | 1.019 | 0.456 | -0.073 | -0.339 | 1.19 | {'FOLLOW': 141, 'CRUISE': 140, 'BRAKE_TO': 131, 'HOLD': 2, 'ACCELERATE': 180} | {'unlabelled': 102, 'FOLLOW': 102, 'CRUISE': 98, 'BRAKE_TO': 124, 'CREEP': 7, 'HOLD': 2, 'ACCELERATE': 159} |
| echo_ade :: echo ADE >=0.8 | 208 | 1.221 | 0.877 | 1.421 | 0.892 | +0.257 | -0.146 | 2.72 | {'FOLLOW': 44, 'CRUISE': 42, 'BRAKE_TO': 39, 'ACCELERATE': 83} | {'unlabelled': 57, 'FOLLOW': 32, 'CRUISE': 20, 'BRAKE_TO': 33, 'ACCELERATE': 66} |

## Lever arms run in this turn (zero GPU, banked plans)

**ENS2** -- per-window mean of the seed-0 and seed-1 refcv8 os plans (EMA-30k), scored on the compliant S2 set vs ha0_ext; numpy ADE.  EARLY single read, zero GPU; consumes BOTH inference seeds, so the bar's inference replicate needs two more seeds (GPU).

seed-to-seed per-window |ADE diff| mean 0.0570 m (p90 0.1306); 39.4 % of windows differ by > 0.05 m; plan L2 seed0-vs-seed1 0.0896 m; triangle inequality holds on every window: True.

* E1_ens2_minus_echo: -0.0138 [-0.0279, +0.0012] separated False (n 4159 / 138)
* E1_seed0_numpy: -0.0085 [-0.0229, +0.0065] separated False (n 4159 / 138)
* E1_seed1_numpy: -0.0076 [-0.0216, +0.0073] separated False (n 4159 / 138)
* ens2_minus_seed0: -0.0052 [-0.0068, -0.0038] separated True (n 4159 / 138)
* ens2_minus_seed1: -0.0061 [-0.0075, -0.0048] separated True (n 4159 / 138)

**RT_XFIT** -- plan moved along its own path by beta m/s, beta per (v0bin x head argmax) group, 2 episode folds (eid % 2), grid -2..2 step 0.05.  DIAGNOSTIC EARLY read (fit on eval episodes, scored only on the other fold); not deployable.

* seed 0 E1_calibrated_minus_echo: -0.0107 [-0.0248, +0.0037] separated False
* seed 0 calibrated_minus_raw: -0.0022 [-0.0055, +0.0009] separated False
* seed 1 E1_calibrated_minus_echo: -0.0098 [-0.0237, +0.0046] separated False
* seed 1 calibrated_minus_raw: -0.0022 [-0.0057, +0.0012] separated False

**MILESTONE** -- EMA milestones, inference seed 0, panel `os`, numpy ADE on the compliant S2 windows of step30000ema (same keys).

| tree | E1 (seed 0) | echo-good | echo-good & dv2 cruising | echo-good & dv2 not cruising | slow_both | stopping | decelerating | v0<2 |
|---|---|---|---|---|---|---|---|---|
| step5000ema_unc | -0.0119 [-0.0269, +0.0037] | +0.0239 (0.170) | +0.0049 (0.150) | +0.0190 (0.232) | +0.0118 (0.349) | +0.0044 (0.529) | +0.0050 (0.439) | +0.0115 (0.356) |
| step10500ema_unc | -0.0137 [-0.0273, +0.0003] | +0.0202 (0.163) | +0.0041 (0.148) | +0.0161 (0.210) | +0.0088 (0.316) | +0.0034 (0.453) | +0.0049 (0.437) | +0.0086 (0.329) |
| step15000ema_unc | -0.0108 [-0.0248, +0.0049] | +0.0220 (0.166) | +0.0047 (0.150) | +0.0173 (0.219) | +0.0082 (0.309) | +0.0039 (0.492) | +0.0057 (0.445) | +0.0081 (0.324) |
| step20000ema_unc | -0.0127 [-0.0265, +0.0021] | +0.0221 (0.166) | +0.0053 (0.151) | +0.0169 (0.216) | +0.0073 (0.299) | +0.0040 (0.495) | +0.0048 (0.437) | +0.0065 (0.310) |
| step30000ema_unc | -0.0085 [-0.0229, +0.0065] | +0.0264 (0.174) | +0.0076 (0.156) | +0.0189 (0.231) | +0.0093 (0.321) | +0.0042 (0.511) | +0.0066 (0.455) | +0.0082 (0.326) |

## S2 widening sets for the next force roll (seed-0 head; the window lists are in the JSON, sha12-keyed)

| set | n win / ep | contrib (seed 0) | head argmax | GT lon_v7 |
|---|---|---|---|---|
| S2A_slow_or_stopping (dv2 slow_both | stopping) | 434 / 37 | +0.0135 | {'FOLLOW': 66, 'CRUISE': 6, 'BRAKE_TO': 12, 'CREEP': 2, 'HOLD': 53, 'ACCELERATE': 295} | {'unlabelled': 233, 'FOLLOW': 1, 'CRUISE': 17, 'BRAKE_TO': 12, 'CREEP': 16, 'HOLD': 31, 'ACCELERATE': 124} |
| S2B_decelerating (dv2 decelerating) | 424 / 73 | +0.0066 | {'FOLLOW': 89, 'CRUISE': 82, 'BRAKE_TO': 164, 'HOLD': 5, 'ACCELERATE': 84} | {'unlabelled': 54, 'FOLLOW': 32, 'CRUISE': 31, 'BRAKE_TO': 208, 'CREEP': 18, 'ACCELERATE': 81} |
| S2C_echo_good_speed_changing (echo-good & dv2 not cruising) | 556 / 71 | +0.0189 | {'FOLLOW': 121, 'CRUISE': 28, 'BRAKE_TO': 83, 'CREEP': 2, 'HOLD': 51, 'ACCELERATE': 271} | {'unlabelled': 201, 'FOLLOW': 44, 'CRUISE': 32, 'BRAKE_TO': 86, 'CREEP': 19, 'HOLD': 29, 'ACCELERATE': 145} |

## Controls (detail)

```
{
 "F_force_lon_sets": {
  "FB_all_rederived": 460,
  "FB_all_select_stats": 460,
  "CT_pool_rederived": 1279,
  "CT_pool_select_stats": 1279,
  "cruising6_rederived": 1739,
  "cruising6_select_stats": 1739,
  "n_plan": 500,
  "n_plan_missing": 0,
  "n_plan_clip_sha12_mismatch": 0,
  "n_plan_not_in_its_set": 0,
  "n_FB_selected": 250,
  "n_CT_selected": 250,
  "PASS": true
 },
 "F_head_argmax_differs_seed1_vs_seed0": 0,
 "R_PASS": true,
 "Z_PASS": true,
 "S_PASS": true,
 "P_PASS": true,
 "K_PASS": true,
 "M_PASS": true,
 "F_PASS": true,
 "ALL_PASS": true
}
```

seed 0: {"R": {"reread": {"delta": -0.0085, "lo": -0.0229, "hi": 0.0065, "n_windows": 4159, "n_episodes": 138, "separated": false, "n_dropped_nonfinite": 0}, "banked_capped": {"delta": -0.0085, "lo": -0.0229, "hi": 0.0065, "n_windows": 4159, "n_episodes": 138, "separated": false, "n_dropped_nonfinite": 0}, "exact_vs_banked_capped": true, "banked_uncapped": {"delta": -0.0085, "lo": -0.0229, "hi": 0.0065, "n_windows": 4159, "n_episodes": 138, "separated": false, "n_dropped_nonfinite": 0}, "exact_vs_banked_uncapped": true, "exact_vs_brief_literal": true, "joint_equals_ci_at_4dp": true}, "R_PASS": true, "K": {"n_changed_plans_on_compliant_windows": 0, "n_binding_on_compliant_windows": 0, "PASS": true}, "S": {"max_abs_lon_plus_lat_minus_d": 4.440892098500626e-16, "max_abs_tangent_lon_plus_lat_minus_d": 8.881784197001252e-16, "max_abs_numpy_vs_components_ade_os": 2.797144422750364e-07, "max_abs_numpy_vs_components_ade_ha": 2.758176718131722e-07, "PASS": true}, "Z_PASS": true, "P": {"n_perm": 200, "n_bin_reads": 1000, "coverage_of_frac_x_total": 0.989, "rate_contrib_separated_from_0": 0.064, "PASS": true}, "M": {"sha": "4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129", "sha_equal_expected": true, "n_compliant_dump": 4159, "n_dump": 4754, "n_compliant_panel": 4159, "n_panel": 4754, "n_episodes_compliant": 138, "PASS": true}}

yaw source: {'yaw_source': 'world yaw (X-yaw PASS)', 'x_yaw_max_err_m': 7.699265890961455e-06, 'x_yaw_p99_err_m': 3.6806216870480407e-06}

seed 1: {"R": {"reread": {"delta": -0.0076, "lo": -0.0216, "hi": 0.0073, "n_windows": 4159, "n_episodes": 138, "separated": false, "n_dropped_nonfinite": 0}, "banked_capped": {"delta": -0.0076, "lo": -0.0216, "hi": 0.0073, "n_windows": 4159, "n_episodes": 138, "separated": false, "n_dropped_nonfinite": 0}, "exact_vs_banked_capped": true, "banked_uncapped": {"delta": -0.0076, "lo": -0.0216, "hi": 0.0073, "n_windows": 4159, "n_episodes": 138, "separated": false, "n_dropped_nonfinite": 0}, "exact_vs_banked_uncapped": true, "exact_vs_brief_literal": true, "joint_equals_ci_at_4dp": true}, "R_PASS": true, "K": {"n_changed_plans_on_compliant_windows": 0, "n_binding_on_compliant_windows": 0, "PASS": true}, "S": {"max_abs_lon_plus_lat_minus_d": 2.220446049250313e-16, "max_abs_tangent_lon_plus_lat_minus_d": 4.440892098500626e-16, "max_abs_numpy_vs_components_ade_os": 3.1041335679660165e-07, "max_abs_numpy_vs_components_ade_ha": 2.758176718131722e-07, "PASS": true}, "Z_PASS": true, "P": {"n_perm": 200, "n_bin_reads": 1000, "coverage_of_frac_x_total": 0.986, "rate_contrib_separated_from_0": 0.05, "PASS": true}, "M": {"sha": "4ebc6b8604f064f69e9e94c15e099062f68339a8ef38d666e8a29d29cb1c4129", "sha_equal_expected": true, "n_compliant_dump": 4159, "n_dump": 4754, "n_compliant_panel": 4159, "n_panel": 4754, "n_episodes_compliant": 138, "PASS": true}}

yaw source: {'yaw_source': 'world yaw (X-yaw PASS)', 'x_yaw_max_err_m': 7.699265890961455e-06, 'x_yaw_p99_err_m': 3.6806216870480407e-06}
