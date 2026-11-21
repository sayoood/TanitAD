# IDM diagnosis — which ceiling is each channel actually hitting?

**Date:** 2026-07-26 · **Agent:** `idm-v2` · **Pod:** `tanitad-eval` (A40, idle; pod1/pod2/pod3 untouched)
**Question:** for `speed`, `yaw_rate`, `steer`, `long_accel` — is the ceiling
**(a)** the frozen encoder's representation, **(b)** the target's own noise/definition,
**(c)** monocular scale ambiguity, or **(d)** head capacity / training budget?

**Answer, one line per channel:**

| channel | verdict | the number that decides it |
|---|---|---|
| **speed** | **(c) monocular scale, then (a)** — **not (d)** | a *linear* probe on frozen z reaches **R² 0.772**, the trained 2.9 M head reaches 0.775; and **56.6 % of the deployed head's speed MSE is a clip-level bias proportional to the clip's speed — a systematic 17 % shrinkage toward the training mean (gain 0.830, r = −0.653 over 36 clips, replicated inside both corpora)** |
| **yaw_rate** | **(b) the label — on comma2k19 only** | comma yaw's own smooth-fit ceiling is **R² 0.352**; 0.30 % of frames are physically impossible (|ω| up to 15.5 rad/s, all at v ≈ 0). On PhysicalAI the same linear probe gets **R² 0.746** |
| **steer** | **(b) the target is redundant and not SI-comparable** | corr(steer, yaw/v) = **0.9865** on PhysicalAI — the label *is* `atan(2.9·κ)` by construction, carrying zero information beyond (yaw_rate, speed) |
| **long_accel** | **(b) the label is not the quantity the video shows, then (a)** — **not (d)** | PhysicalAI `long_accel` correlates only **r = 0.434** with the pose-derived dv/dt; a *perfect* kinematic estimator caps at **R² 0.188** against it. On comma, where the label *is* dv/dt (r = 0.925), the frozen latent still yields **R² −0.095** |

**(d) — "more training" — is REJECTED on all four channels.** Measured below: the
data curve is flat past 34 clips, the step curve is flat past 25 epochs, capacity
does nothing, and `long_accel` gets **monotonically worse** with more of any of them.

---

## 1. Substrate — what these numbers are measured on

| | |
|---|---|
| encoder | flagship-v1 `flagship4b-speedjerk-30k`, **FROZEN**; ckpt md5 `b5f07d9e3dd2ca643949bc86832e6585` **asserted at encode time**, step 29999, state_dim 2048 |
| corpora | `physicalai-val-0c5f7dac3b11` (40 eps, T=199) + `comma2k19-val-76b6e94a97a1` (64 eps, T=300) = **104 episodes**, encoded once with `run_idm_proof.encode_frames` (the shipped path, not a re-implementation) |
| split | episode-disjoint, domain-stratified, deterministic: **68 train / 36 val episodes** |
| windows | **15,875 train** (stride 1) / **4,195 val** (stride 2), built at k=8 once so 9- and 17-frame arms share identical centres |
| estimator | `taniteval.ci.episode_cluster_bootstrap`, resampling the **36 val episodes**. `overlapping_holdout_se` is not used anywhere in this work |

**Evidence class of everything in this document: MEASURED (this run)** unless a row
says otherwise. Artifacts: `labels.json`, `labels2.json`, `probe.json`, `curve.json`
in this directory; code `idm2_*.py` (same directory), run on `tanitad-eval:/root/idm2/`.

⚠️ **This is not the card's val set.** `idm_head_v1`'s card was measured on pod3's
`lat_flagshipv1` latents; pod3 is running E1c and is off-limits. These 104 episodes
were *never seen by the head or its encoder*, so A0's numbers here are a fresh
held-out measurement of the same artifact — larger n (4,195 vs 9,420 windows but
36 vs 90 episodes, and episodes are the unit) on a different clip mix. Both are reported.

---

## 2. The metric problem, first — because R² is lying about two channels

Measured on this val set: comma2k19 `yaw_rate` has **std 0.4290 rad/s but MAD
0.0112 rad/s** — a ratio of **38** where a Gaussian gives 1.48. R²'s denominator is
therefore set by a handful of corrupt frames, and the estimator says so out loud:

> the linear probe's yaw_rate R² is **0.0526 with an episode-cluster bootstrap CI
> of [−0.024, +0.778]** — the interval spans essentially the entire range, because
> the answer depends on whether a corrupt comma episode is drawn.

**So this work reports, for every channel: R² · Spearman ρ · MAE · medAE ·
nMedAE**, where **nMedAE = medAE / MAD(label)** is scale-free and outlier-proof:
below 1.0 the model beats "always predict the label's median", above 1.0 it does not.
This is the number to quote for `yaw_rate` and `long_accel`.

---

## 3. (b) The targets — three of the four are defective, in three different ways

### 3.1 The noise floor: what any smooth predictor can reach

The encoder sees images, which are near-noiseless observations of the physical
state. Whatever part of a label is **not a smooth function of time over the head's
own receptive field** is unreachable by construction. So the ceiling is the R² of
the label against the best degree-2 polynomial fitted over that same window
(`idm2_diag_labels.py`, exact Savitzky-Golay centre tap):

| channel | domain | std | MAD | ρ₁ | **R² ceiling w9** | R² ceiling w17 | residual RMS w9 |
|---|---|---:|---:|---:|---:|---:|---:|
| speed | pai | 9.3225 | 5.1990 | 0.987 | **1.0000** | 1.0000 | 0.0088 m/s |
| speed | cm | 9.2541 | 3.1935 | 0.994 | **1.0000** | 1.0000 | 0.0103 m/s |
| yaw_rate | pai | 0.1268 | 0.0088 | 0.976 | **0.9998** | 0.9990 | 0.0018 rad/s |
| **yaw_rate** | **cm** | **0.4290** | **0.0112** | 0.889 | **0.3521** | **0.2245** | **0.3487 rad/s** |
| steer | pai | 0.0895 | 0.0027 | 0.972 | 0.9999 | 0.9995 | 0.0007 |
| steer | cm | 0.0157 | 0.0022 | 0.985 | 0.9995 | 0.9975 | 0.0004 |
| long_accel | pai | 0.9465 | 0.2543 | 0.790 | 0.9808 | 0.9721 | 0.1327 m/s² |
| long_accel | cm | 0.5012 | 0.1788 | 0.698 | 0.9394 | 0.9196 | 0.1224 m/s² |

> 🔴 **RE-ISSUED 2026-07-27 (C29) — the `cm yaw_rate` ceiling of 0.3521 is a property of a BROKEN
> LABEL, not of comma2k19.** It was computed with **`heading_repair` OFF**; comma's heading is
> `arctan2` of the ENU velocity, undefined at standstill (**26.27 %** of comma frames below 0.5 m/s
> physically impossible, **0.000 %** above — PhysicalAI zero in every bin, so its `0.9998` row is
> untouched). The row's own `std 0.4290` vs `MAD 0.0112` is the defect's signature. **The ceiling was
> the load-bearing premise for "no model can score well against a label that noisy" — that inference
> is RETRACTED.** With the repair on, the *deployed* head already reads comma yaw **R² +0.3308**
> (≈ the old "ceiling"), and a retrained head reads **+0.679** — i.e. **above** it.
> ⚠️ **STALE-PENDING, not corrected:** the ceiling statistic itself has NOT been recomputed on
> repaired labels. Do not quote **0.3521** as comma's yaw ceiling; there is currently **no measured
> repaired ceiling**. Inventory: `…/incoming/2026-07-27-comma-yaw-reissue/COMMA_YAW_REISSUE.md`.
>
> 🔴 **AMENDED 2026-07-27 (`anchor-settlement`, class C43).** *(Block above keeps its text and date.)*
> **`+0.3308` is WITHDRAWN** — by content (sha256 of raw pose bytes *and* raw sensor bytes), **2 of
> the 22 comma val episodes it was scored on are bit-identical to 2 of that head's own comma TRAINING
> clips**; without them it reads **−0.746**. So the *"the deployed head already reads ≈ the old
> ceiling"* half of the retraction above **no longer holds**. `+0.679` (= `R0`, no leak) **stands**,
> and is **+0.3038 [+0.054, +0.479]** on the 20 content-clean episodes — **still above 0.3521? NO,
> below it.**
> ⛔ **But that comparison is inadmissible and must not be made:** `0.3521` is a *smooth-fit ceiling
> on the broken label*, `+0.3038` is a *model R² on the repaired label over a different episode set*.
> They are not on the same axis. **The RETRACTION of the "no model can score well" inference still
> stands** — a retrained head reads the channel with ρ ≈ 0.60 and a separated positive R² — but it
> stands on the *mechanism*, not on a number-vs-number comparison.
> ⛔ **The repaired ceiling remains NOT recomputed and must stay unknown until measured.** Record:
> `…/Benchmarks & Eval/Implementation/incoming/2026-07-27-anchor-settlement/ANCHOR_SETTLEMENT.md`.

**Only one cell is a noise floor: comma2k19 `yaw_rate`, where 65 % of the label's
variance is not a smooth function of time at all.** Everything else is smooth —
including `long_accel`, whose problem turns out to be different (§3.3).

### 3.2 comma2k19 `yaw_rate` — found the frames, found the mechanism

`labels2.json`:

- **3 of 64** comma clips carry |ω| > 5 rad/s; **0.299 %** of comma frames exceed
  1.5 rad/s (86 °/s); worst per-episode yaw-rate std **2.365 rad/s**.
- PhysicalAI: **zero** frames beyond 1.5 rad/s, in 40 clips.
- The worst clip, `cm_00052` (episode 1346272467) at frame 113 — raw yaw jumps
  `+1.788 → −1.389 rad` in one step, giving yaw_rate **−15.12 then +15.53 rad/s**,
  **while the speed is 0.00–0.01 m/s**. The vehicle is stationary.

The mechanism is in the loader, not in the physics: comma2k19 heading is
`arctan2(enu_v[:,1], enu_v[:,0])` (`stack/tanitad/data/comma2k19.py:172`) — a
heading estimated from the **ENU velocity vector, which is undefined at zero
speed**. PhysicalAI takes heading from the orientation quaternion
(`stack/tanitad/data/physicalai.py:379-431`), which is standstill-robust.
*(Loader lines: INHERITED from a primary-source read of the shipped loaders;
the frame-level consequence above is MEASURED here.)*

**Winsorising at a physical limit costs almost nothing and fixes the statistic:**

| clip limit | comma frames touched | comma yaw std |
|---|---:|---:|
| none | — | 0.4290 |
| 1.5 rad/s | **0.299 %** | **0.1088** (−3.9×) |
| 1.0 rad/s | 0.404 % | 0.0864 |
| 0.6 rad/s | 0.582 % | 0.0662 |

PhysicalAI is untouched at every limit.

### 3.3 `long_accel` — the label is not the quantity the video shows

This is the finding that changes what should be built. Pooled over all frames
(`labels2.json`):

| | PhysicalAI | comma2k19 |
|---|---:|---:|
| corr( label , centred dv/dt ) | **0.4335** | **0.9245** |
| **R² of the BEST affine predictor in dv/dt** | **0.1879** | **0.8547** |
| residual std / label std | 0.918 / 0.949 | 0.195 / 0.501 |
| residual autocorr @ lags 1,2,5,10,20 | **0.973, 0.958, 0.901, 0.758, 0.450** | 0.285, −0.371, 0.087, 0.023, −0.011 |
| best lag (frames) | +1 | −2 |

Read it in this order:

1. On **PhysicalAI**, `actions[:,1]` is **only r = 0.434 correlated with the
   acceleration of the vehicle's own pose**. A *perfect* kinematic accelerometer —
   which is exactly what "differentiate the predicted trajectory" would give you —
   **caps at R² 0.188 against this label**.
2. The 81 % residual is **not noise**: its autocorrelation is 0.973 at 0.1 s and
   still 0.45 at 2 s. It is a **slowly varying non-kinematic component** — the
   signature of a longitudinal accelerometer that includes the gravity projection
   of road grade, or a slow sensor bias. It is smooth (hence the 0.98 ceiling in
   §3.1) but it is a different physical quantity.
3. On **comma2k19** the label *is* dv/dt (r = 0.925, ceiling 0.855, white residual).
   And there the frozen latent still yields **R² −0.095** — so on the corpus where
   the target is clean, the *representation* is the limit.

**Therefore `long_accel`'s ceiling is (b) on PhysicalAI and (a) on comma — and on
neither is it (d).** The right move is to change the target, and I commit to that
recommendation independent of how any arm below performs.

### 3.4 `steer` — redundant by construction, and not the same quantity twice

| | PhysicalAI | comma2k19 |
|---|---:|---:|
| corr( steer , yaw_rate / v ) | **0.9865** | 0.8754 |
| corr( steer , yaw_rate ) | 0.9690 | 0.8897 |

PhysicalAI's steer label is literally `atan(WHEELBASE · κ)` with `WHEELBASE = 2.9`
(`stack/tanitad/data/physicalai.py:51`); comma2k19's uses `STEER_RATIO = 15.3`
(`stack/tanitad/data/comma2k19.py:45`). So (i) on PhysicalAI the channel carries
**zero information beyond (yaw_rate, speed)** and is spending head capacity and
gradient to re-derive an arctangent, and (ii) the two corpora's `steer` are **not
the same physical quantity**, which makes any pooled steer R² units-confounded.

> ⚠️ **Correction to the record.** `idm_head_v1_card.json` states *"steer and
> long_accel are DROPPED as unusable (pilot NOTE 2026-07-24)"*. The cited NOTE
> does not mention steer at all, and no artifact in the repo measures steer as
> unusable — its measured R² is 0.57–0.86 across arms and **0.742 for A0 on this
> val set**. `long_accel`'s drop is fully MEASURED; **steer's is inherited prose
> (class C4)**. The defensible statement is *"steer is redundant with (yaw_rate,
> speed) and is not SI-comparable across corpora"* — which is a reason to
> **derive** it, not a reason to call it unusable.

### 3.5 Where the variance lives (needed to read §5 honestly)

Between-clip fraction of each label's total variance:

| channel | pai | cm |
|---|---:|---:|
| **speed** | **0.939** | **0.956** |
| yaw_rate | 0.196 | 0.007 |
| steer | 0.260 | 0.221 |
| long_accel | 0.161 | 0.114 |

**94–96 % of speed variance is BETWEEN clips.** This is why a per-clip oracle
recalibration flatters speed so much (§5) and why the honest scale evidence has to
come from cross-domain transfer, not from a pooled per-clip fit.

---

## 4. (a) The frozen representation — the first per-channel linear probe ever run here

Closed-form ridge on the frozen latents; λ chosen on an **episode-disjoint inner
split of TRAIN only**; three receptive fields. *(Prior art: the repo has a speed-only
ridge probe — `taniteval/diag_v2mech.py`, flagship-v1 `probe_speed_r2 = 0.861`. No
probe for yaw_rate / steer / long_accel has ever been run. This is it.)*

**Pooled (both corpora), n = 4,195 windows / 36 episodes:**

| target | probe R² (w9) | 95 % CI (ep-cluster boot) | ρ | MAE | nMedAE | **trained 2.9 M head (B0)** |
|---|---:|---|---:|---:|---:|---:|
| speed | **0.7715** | [0.6440, 0.8356] | 0.802 | 4.305 m/s | 0.723 | **0.775** |
| **log speed** | **0.8352** | [0.7249, 0.8891] | 0.727 | — | — | — |
| yaw_rate | 0.0526 | [−0.024, **+0.778**] | 0.476 | 0.0438 rad/s | 1.844 | 0.086 |
| steer | 0.6294 | [0.3944, 0.7039] | 0.529 | 0.0174 | 3.407 | 0.358 |
| long_accel | 0.0796 | [−0.024, 0.1448] | 0.073 | 0.394 m/s² | 1.156 | −0.250 |
| curvature κ = ω/v | −0.0142 | — | 0.347 | — | — | — |

**Per-domain, same probe:**

| target | pai (w9) | cm (w9) | pai (centre frame) | pai (w17) |
|---|---:|---:|---:|---:|
| speed | 0.716 | 0.662 | 0.707 | 0.708 |
| **yaw_rate** | **0.746** | **−0.030** | 0.638 | 0.733 |
| steer | 0.630 | 0.128 | 0.510 | 0.641 |
| long_accel | 0.113 | −0.095 | 0.065 | **0.156** |
| κ | 0.500 | −0.043 | 0.456 | 0.464 |

**What this settles:**

1. **A linear readout of the frozen latent matches the trained 2.9 M-param
   non-causal transformer on every channel** (speed 0.772 vs 0.775; yaw 0.053 vs
   0.086; long_accel 0.080 vs −0.250, i.e. the probe is *better*). **The head is not
   the bottleneck.** Precedent held: this is the same instrument E2a used to show
   lateral offset was linearly recoverable at R² 0.72 with the loss downstream.
2. **The representation contains yaw.** On PhysicalAI a *linear* probe reads
   **R² 0.746** (0.800 when trained in-domain, §5). The "yaw_rate 0.010 at scale"
   headline is not a representation failure — it is the comma third of that val mix.
3. **The representation does not contain acceleration.** w9 pooled 0.080; even on
   comma, where the label is clean, **−0.095**. Going from 1 → 9 → 17 frames moves
   PhysicalAI 0.065 → 0.113 → **0.156**: a real but small receptive-field effect
   that does not change the verdict.
4. **`log v` is decodable at R² 0.835** (pai 0.816, cm 0.752). ⚠️ This is R² against
   `log v`, **not** against `v` — it is *not* comparable to the 0.772 row above and
   must not be read as an improvement (see the retraction in §5.3). It is reported
   because it shows the latent carries speed information in a relative as well as an
   absolute sense; the end-to-end test of whether that helps is arm B2, and it does
   not.

---

## 5. (c) Monocular scale — measured three ways, and the first answer was too coarse

⚠️ **A pooled per-clip oracle recalibration is NOT admissible evidence here.** 94 %
of speed variance is between clips (§3.5), so a "just know the clip's mean speed"
control already reaches **R² 0.975** — above the deployed head. Any argument built
on per-clip recalibration must be reported against that control. Three sharper
instruments, in increasing order of relevance to the deployed model:

### 5.1 Cross-domain transfer — the unseen-camera case (this is the YouTube case)

`probe.json → cross_domain_w9`. `f_eff` is already canonicalised to ≈266 px on all
corpora (rig-A 266.13 / rig-B 266.10 / comma 266.50, `results_regate.json`), so any
residual gain is **camera height/pitch**, not focal length:

| probe trained on | evaluated on | speed R² | speed MAE | + corpus affine | + **per-clip** affine |
|---|---|---:|---:|---:|---:|
| PhysicalAI | PhysicalAI | +0.8171 | 3.55 m/s | +0.8630 | +0.9890 |
| **PhysicalAI** | **comma2k19** | **−0.2177** | **8.73 m/s** | **+0.3141** | **+0.9787** |
| comma2k19 | PhysicalAI | +0.2466 | 7.62 m/s | +0.4103 | +0.9809 |
| comma2k19 | comma2k19 | +0.7178 | 3.83 m/s | +0.7208 | +0.9795 |

A PhysicalAI-trained readout on comma is wrong in absolute m/s (R² −0.218, MAE
8.7 m/s) and nearly right in shape (0.979 after a 2-parameter per-clip affine).
**But note the middle column:** one constant per corpus recovers only −0.218 →
+0.314. **The gain is not a per-corpus constant.**

### 5.2 The deployed head has NO per-corpus gain left — a measured no-op

`scale.json`. Fit an affine on `idm_head_v1`'s speed output using the **68 TRAIN
episodes** (which it has also never seen), apply to the **36 VAL episodes**:

| | R² | MAE |
|---|---:|---:|
| A0 raw | 0.8651 | 2.994 m/s |
| A0 + global affine (TRAIN-fit) | 0.8650 | 3.040 m/s |
| **A0 + per-corpus affine (TRAIN-fit)** | **0.8672** | **2.985 m/s** |

Paired episode-cluster bootstrap on Δ MAE: **−0.0095 [−0.2133, +0.2357], NOT
separated.** Fitted coefficients are already near-identity (pai 0.933·p − 0.077;
cm 0.908·p + 3.096). **A multi-domain-trained head has already absorbed the
corpus-level gain — so shipping a per-camera calibration constant would buy
nothing.** This retires the obvious "metric grounding = one constant per corpus"
proposal before anyone spends a GPU-day on it.

### 5.3 Where A0's speed error actually is — and it *is* scale, per clip

`speed_decomp.json`. Split A0's speed MSE into a per-clip level term and a
within-clip term, then ask whether the level term is proportional to speed:

| | pooled | PhysicalAI | comma2k19 |
|---|---:|---:|---:|
| speed MSE | 17.400 | 10.784 | 20.060 |
| **fraction that is a CLIP-LEVEL bias** | **56.6 %** | **59.3 %** | **56.0 %** |
| bias_c vs clip mean speed — slope | **−0.1697** | −0.1772 | −0.1992 |
| — Pearson r (over clips) | **−0.653** (36) | −0.762 (14) | −0.566 (22) |
| — **implied gain g** | **0.830** | 0.823 | 0.801 |
| intercept | +2.930 m/s | +2.677 | +3.894 |
| R² after removing the clip-level term (ORACLE) | **0.9415** | 0.9621 | 0.8940 |
| MAE after removing it (ORACLE) | **1.873** | 1.587 | 1.988 |
| "know each clip's mean speed" oracle R² | 0.9754 | 0.9700 | 0.9634 |
| mean WITHIN-clip corr(pred, gt) | +0.340 | +0.583 | +0.184 |

**Read:** more than half of the deployed head's speed error is *not knowing how
fast this clip is going*, and that error is **proportional to the clip's speed —
a systematic ~17 % shrinkage toward the training mean (g ≈ 0.83), replicated
independently inside BOTH corpora.** Predictions collapse toward
2.93/(1 − 0.83) ≈ 17 m/s, i.e. the training prior.

**And that has a consequence for what can fix it.** Shrinkage of exactly this form
is the **MMSE-optimal** response of *any* regressor to an under-determined input:
if the image does not pin the metre, the least-squares answer is to hedge toward
the prior, in proportion to the distance from it. So **no loss re-weighting and no
output reparametrisation can remove it — only additional information can.**
Pre-registered prediction, to be checked against arms B2/B3 in `IDM_V2_RESULTS.md`:
log-speed and clip-context will help *only in so far as they carry information
about the clip's metric scale*.

Supporting, consistent with a gain rather than a bias: per-clip **scale-only**
recalibration recovers as much as the full affine (0.924 vs 0.986; offset-only
0.919 — all against the 0.975 mean-only control).

> ⚠️ **Retraction, same document, before it was used for anything.** An earlier
> draft of this section read *"log-space parametrisation alone buys +0.064 R²
> pooled on the probe (0.7715 → 0.8352)"*. **That is a units mismatch and is
> withdrawn.** 0.8352 is R²(log v̂, log v) and 0.7715 is R²(v̂, v) — different
> targets with different variances; they are not comparable and the difference is
> not an improvement. **Root-cause class: C6 (confounded comparison — the contrast
> varied the target as well as the parametrisation).** The end-to-end measurement
> is in `IDM_V2_RESULTS.md` arm **B2**, and it goes the *other* way: predicting
> log-speed makes m/s error **worse** (R² 0.771 → 0.695, MAE 3.60 → 4.11 m/s),
> because a squared loss in log space optimises *relative* error while the metric
> is *absolute*. The multiplicative structure in §5.3 stands — it is measured
> directly on the per-clip biases, not through this comparison — but it cannot be
> harvested by changing the output space. That is precisely what §5.3 predicts:
> the shrinkage is an information deficit, not a parametrisation artefact.

### 5.4 The cross-check: yaw does NOT behave this way

Rotation is observable from a single camera **without** metric scale, so if the yaw
failure were a scale failure it would not exist. It doesn't on PhysicalAI (in-domain
probe 0.800). And on comma, per-clip recalibration **cannot** rescue yaw
(0.043 for pai→cm, 0.067 for cm→cm) — because you cannot recalibrate your way out
of a label that is undefined. **Yaw is a label problem; speed is a scale problem.**
The instrument separates them, which is the main reason to trust both verdicts.

---

## 6. (d) Head capacity / training budget — REJECTED on all four channels

v1's exact recipe, everything held fixed but one axis (`curve.json`):

**vs DATA** (nested, domain-stratified train subsets):

| clips | windows | speed R² | yaw R² | steer R² | long_accel R² | ADE@2s |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1,873 | −1.663 | 0.058 | 0.177 | −0.050 | 21.58 |
| 17 | 4,018 | −0.060 | 0.077 | 0.370 | −0.004 | 14.78 |
| **34** | 7,940 | **+0.830** | 0.084 | 0.373 | −0.358 | 4.675 |
| **68** | 15,875 | **+0.767** | 0.086 | 0.340 | −0.295 | 4.773 |

**vs STEPS** (full data):

| epochs | speed R² | yaw R² | steer R² | long_accel R² | ADE@2s |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.511 | 0.085 | 0.450 | **+0.038** | 12.42 |
| **25** | **0.826** | 0.091 | 0.367 | −0.078 | 4.784 |
| 50 | 0.767 | 0.086 | 0.340 | −0.295 | 4.773 |
| 100 | 0.780 | 0.083 | 0.386 | **−0.420** | 4.921 |
| 200 | 0.766 | 0.060 | 0.408 | −0.280 | 4.961 |

**vs CAPACITY** (50 epochs, full data — a **23× parameter range**):

| d_model | depth | params | speed R² | yaw R² | steer R² | long_accel R² | ADE@2s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **128** | 3 | **0.86 M** | **+0.803** | 0.090 | **0.518** | **−0.132** | 4.707 |
| 256 | 2 | 2.11 M | 0.788 | 0.089 | 0.396 | −0.234 | 4.722 |
| **256** | **3** | **2.90 M** *(shipped)* | 0.767 | 0.086 | 0.340 | −0.295 | 4.773 |
| 256 | 6 | 5.27 M | 0.787 | 0.088 | 0.394 | −0.363 | 4.656 |
| 512 | 3 | 10.52 M | **0.710** | 0.090 | 0.452 | −0.243 | 5.191 |
| 512 | 6 | 19.98 M | 0.746 | 0.089 | 0.394 | −0.356 | 5.184 |

**The shipped 2.90 M head is already too big.** The 0.86 M head beats it on speed
(+0.803 vs +0.767), on steer (+0.518 vs +0.340) and on long_accel (−0.132 vs
−0.295); 20 M is worse than 0.86 M on every channel. `yaw_rate` is **flat at
0.086–0.090 across the whole 23× range** — capacity is simply not the variable.

**Reading (MEASURED):**
- **speed** saturates between 34 and 68 clips and between 25 and 50 epochs; doubling
  either does nothing. The linear probe sits at the same place (0.772), which is the
  independent confirmation.
- **yaw_rate** is flat at **0.060–0.091 across an 8.5× data range, a 20× step range
  and a 23× parameter range.** Nothing on any of the three axes touches it — exactly
  as §3.2 predicts, because the target is contaminated.
- **long_accel** is the diagnostic case: it is **best at 10 epochs (+0.038) and gets
  monotonically worse with more training (−0.078 → −0.295 → −0.420)**. That is a
  model memorising a target its input cannot explain. More training is not neutral
  here — **it is actively harmful.**
- **capacity is negative-return**: the shipped 2.90 M head is beaten on 3 of 4
  channels by a **0.86 M** head, and 19.98 M is worse than 0.86 M on all four.

**Verdict (d): rejected on all three axes — data, steps and parameters.** "Train
longer / on more data / with a bigger head" is not the fix for any channel; for
`long_accel` it is the wrong direction, and for `speed` a **smaller** head is better.

---

## 7. Consolidated verdict, and what it implies for the build

| channel | (a) representation | (b) target | (c) scale | (d) budget | **primary** |
|---|---|---|---|---|---|
| **speed** | secondary — in-domain probe 0.817, ~18 % unexplained | clean (ceiling 1.000) | **PRIMARY** — 56.6 % of MSE is a clip-level bias, proportional to clip speed (**gain 0.830**, r −0.653 over 36 clips, replicated in both corpora) | **rejected** — flat past 34 clips / 25 epochs; a 0.86 M head is *better* | **(c)** |
| **yaw_rate** | not limiting on pai (probe 0.746) | **PRIMARY on comma** — ceiling 0.352, 0.30 % impossible frames | n/a (rotation is scale-free) | **rejected** — flat 0.060–0.091 over 8.5×/20×/23× | **(b)** |
| **steer** | probe 0.630 | **PRIMARY** — redundant (r = 0.9865 with ω/v), and cross-corpus units differ | — | **rejected** | **(b)** |
| **long_accel** | **PRIMARY on comma** — clean label, probe −0.095 | **PRIMARY on pai** — r = 0.434 vs dv/dt, ceiling **0.188** | — | **rejected — harmful** | **(b) + (a)** |

**Which means the build order is: better targets first, better substrate second,
better architecture third, and more training never.** That is what
`PRE_REGISTRATION_IDMV2.md` commits to and what `IDM_V2_RESULTS.md` measures.

> **Forward reference, added after the arms ran (do not read this section without
> it).** The build reordered one item: the largest *measured* single lever turned
> out to be an **architectural** change — a clip-context token (+0.106 speed R²
> off the recipe control) — not a target change. It belongs in the (c) bucket
> rather than the architecture bucket, because it works by supplying the head with
> **information about the clip's metric scale**, which is exactly what §5.3 says
> is missing; it is not extra capacity (a *smaller* head is better). The target
> fixes remain the cheapest wins and the only 0-GPU ones. Two candidate ideas this
> diagnosis pointed at — log-space speed and deriving `long_accel` by
> differentiating a predicted trajectory — were **measured and refuted**
> (`IDM_V2_RESULTS.md` §4.2, §4.3), and one of the refutations forced the
> retraction inside §5.3 above.

## 8. Honest limits of this diagnosis

- **n = 104 episodes, 2 corpora.** The comma-yaw mechanism is a loader-level fact
  and will replicate; the *magnitude* of the speed scale gap is a 2-corpus reading.
- **A0 is measured here, not reproduced from its card** (pod3 off-limits). The two
  sets differ; both are quoted, neither is substituted for the other.
- **The `long_accel` grade hypothesis is a MECHANISM, not a measurement.** What is
  measured is: r = 0.434 to dv/dt, a 0.188 affine ceiling, and a residual with
  autocorrelation 0.45 at 2 s. That the slow residual *is* road grade is the most
  parsimonious reading, **not** something I have instrumented. (Class C3 risk —
  flagged deliberately.)
- Per-clip oracle recalibration numbers are reported **with** their mean-only
  control, because without it they are dominated by the 94 % between-clip variance
  and would be a misleading argument for scale.
