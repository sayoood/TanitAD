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

## CORRECTION (Master Mind, 2026-10-19 ~00:10 Berlin; RETRACTION_LOG R50)
§1 says `ha0` (v0 only) is "exactly the ego information refcv8 is fed". **That is wrong.** refcv8's argv
(`E:/refcv8_eval_kit/ckpt/config.json['argv']`) carries `--ego-history` (past speed, longitudinal accel, yaw
rate) and `--residual-prior ha0_ext_pose`. That prior holds a0 and a yaw-rate-derived kappa0
(`stack/tanitad/models/kinematic_prior.py:53-58`). The E1 echo `ha0_ext` differs from that prior by reading
kappa0 from the STEER channel, the curvature the PI ruled inadmissible on 2026-10-18 (`D-EGO-STATE-T0`).
Every number in this verdict stands; only the sentence about refcv8's inputs and the "capture ratio" framing
built on it are withdrawn.
