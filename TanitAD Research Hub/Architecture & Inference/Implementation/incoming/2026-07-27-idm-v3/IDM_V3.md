# IDM v3 — the detailed approach

**Date:** 2026-07-27 · **Agent:** `idm-v3` · **Pod:** `tanitad-eval` (A40).
pod1 / pod2 / pod3 untouched.
**Reads with:** `PRE_REGISTRATION_IDMV3.md` (bars frozen before the arms ran),
`CITATIONS.md` (the literature), and the v2 predecessors `IDM_DIAGNOSIS.md` /
`IDM_V2_RESULTS.md`.

**Evidence class of every number here: MEASURED (this run)** unless marked
otherwise. Estimator everywhere:
`taniteval.ci.(paired_)episode_cluster_bootstrap`, unit = the **36 val
episodes**, B = 2000. `overlapping_holdout_se` is not called anywhere in this
work. Tier is stated per claim.

---

## 0. The headline

> **The IDM's worst channel was never a model failure. It was a broken label,
> and repairing it moves pooled `yaw_rate` R² from 0.105 to 0.83 — on the
> corpus where it was 0.011, comma2k19 goes to ~0.65.**
>
> **And the PI's geometry hypothesis is now MEASURED rather than assumed. The
> camera geometry the IDM has never been told is real and large — camera height
> varies per clip from 1.245 m to 1.607 m, a 29 % spread, and all three `cam_h`
> constants circulating in this repo are wrong. But the ground-plane correction
> that this geometry prescribes makes speed significantly WORSE, not better.**

---

## 1. What was wrong, and what the literature said to do about it

`CITATIONS.md` reviews ~50 papers. The one sentence that reorganised this work
is classical, not recent:

> A calibrated two-view monocular measurement fixes **rotation completely** and
> **translation only up to a positive scalar** — the magnitude ‖t‖ is not in the
> essential matrix at all. *(Nistér, TPAMI 2004, five-point relative pose;
> Longuet-Higgins & Prazdny, Proc. R. Soc. B 1980, decompose optic flow into a
> rotational component independent of scene depth and a translational component
> scaled by inverse depth.)* `PUBLISHED (cited)`

**Our four channels therefore do not share a failure mode, and must not share a
fix:**

| channel | physical status | what the literature prescribes |
|---|---|---|
| `speed` | translation — **unobservable up to scale** | known camera height / ground plane (Song & Chandraker CVPR 2014; Wang ICRA 2021), canonical focal normalisation (Metric3D), camera-parameter conditioning (CAM-Convs; TartanVO's intrinsics layer), a velocity prior |
| `yaw_rate` | rotation — **fully observable, scale-free** | nothing from the scale toolkit applies. Fix the labels; give rotation its own loss (TartanVO's up-to-scale loss; Lee & Civera Rotation-Only BA) |
| `steer` | rotation, geometrically redundant with `yaw/v` | treat with the rotation toolkit, not the scale toolkit |
| `long_accel` | second-order motion | the literature has **no positive result**; human vision does not compute acceleration directly (Werkhoven 1992) and CV needs a dedicated method just to *amplify* it |

⭐ **The single most actionable citation was a protocol one:** comma.ai's own
`calib_challenge` **discards every frame below 4 m/s**, because direction of
travel is undefined at standstill. We were scoring comma2k19 yaw-rate over
exactly the frames comma excludes by design.

⚠️ **Two levers the literature recommends are already pulled in our pipeline,
and saying so prevented two wasted GPU-days:**
- **Canonical focal normalisation (Metric3D's `ω = f_c/f`)** — our f-theta crop
  already lands `f_eff ≈ 266 px` on every corpus (rig-A 266.13 / rig-B 266.10 /
  comma 266.50). There is no focal mismatch left to remove.
- **TartanVO's intrinsics layer** appends per-pixel `((u−cx)/fx, (v−cy)/fy)`
  channels — it operates on **pixels**, and our IDM reads a **frozen latent**.
  Not applicable without unfreezing the encoder.

---

## 2. The geometry substrate — MEASURED, and it settles a standing conflict

The PI asked whether we had considered extrinsics. **We had not: across all 15
IDM scripts there is not one reference to intrinsics, extrinsics, camera height,
focal length or FOV.** So the first thing v3 did was go and get them.

Pulled from PhysicalAI-AV's **own** gated `calibration/camera_intrinsics` and
`calibration/sensor_extrinsics`, joined to our 40 val episodes through the
build's ordered clip list. **The join is verified, not assumed: `ep_id =
int.from_bytes(clip_id[:4],"big")` reproduces 40/40 eval-pod `episode_id`s.**

| quantity | measured over the 40 val clips |
|---|---|
| **camera height** | **1.2450 – 1.6066 m** · mean 1.3417 · std 0.0991 · CV **7.4 %** · full range **29.0 %** · **37 distinct values in 40 clips** |
| principal point `cy` | 535.0 – 764.5 px — two clusters = the two rigs (14 rig-A / 26 rig-B) |
| `f_paraxial` | 920.1 – 938.5 px @ native 1920×1080 (2.0 % spread) |
| pitch (optical axis) | −1.15° to +2.34° |
| camera forward offset | 1.797 – 2.140 m |

### 2.1 The `cam_h` conflict is resolved — by refuting all three candidates

The repo carries **three mutually inconsistent camera heights: 1.5 / 1.43 /
1.22 m**. The brief asked to reconcile them from the data. The answer is that
**none of them is right, because the quantity is not a constant**:

- **1.22 m is below the observed minimum** (1.245 m) — it is outside the data.
- **1.43 m sits at the 93rd percentile.**
- **1.5 m is exceeded by only 5 of 40 clips.**
- The measured **median is 1.306 m**, and it varies **per clip**.

**Recommendation (escalated, needs an owner): stop conditioning on a constant.**
`tanitad/replay/rr_log.py:93-94`, `taniteval/cam_overlay.py:29`,
`taniteval/clhorizon.py:87` and `scripts/viz_trajectory_fan.py:43` each hard-code
one of the three. Every horizon/fan overlay drawn with them is wrong by up to
**29 %** in the ground-plane projection.

### 2.2 ⭐ Rig identity is NOT a proxy for camera height — which retires a prior result

- rig-A median height **1.3127 m** · rig-B median **1.2931 m** — **1.5 % apart**.
- The **within-rig** spread is **29 %**.

**So a rig label carries almost no information about the metric scale factor.**
The sibling result that tested rig-conditioned scale on world-model latents and
found within-cluster fits 2.4–3.2 % worse used a **row-profile rig PROXY**; this
measurement shows that even a *perfect* rig label would have been the wrong
variable. The two findings are consistent and neither settles the height
question. `MEASURED · CONFIRMED`

---

## 3. The physics, stated so the predictions are falsifiable

For focal `f`, camera height `h`, a ground point at forward distance `X`
projects `Δv = f·h/X` px below the horizon. Differentiating along the motion:

> **(1)  v = (f·h) · Φ(image motion)** — metric speed is **linear in `f·h`**
> **(2)  ω = (du/dt) / f** — yaw rate depends on **`f` alone**, never on `h`

Because our pipeline already canonicalises `f_eff` to ≈266 px everywhere,
**(2) is already satisfied and (1) is not.** That asymmetry is the discriminator
the whole Phase-3 design turns on, and it was **pre-registered**:

> Geometry conditioning **must help SPEED and must NOT help YAW**. An arm that
> improves both by a CI-separated amount is reading corpus identity, whatever
> its input is named.

---

### 3.1 Two independent routes reached the same decomposition

The latent-action research stream asked a different question — *does a learned
latent action sidestep the metric-scale ambiguity?* — and answered **no**, for
the same structural reason: a latent action **separates** ego motion into
*(scale-free structure, which monocular video determines)* × *(one per-clip
scalar, which it does not)*. It relocates the scalar into the grounding head; it
does not remove it.

**That is the same split as (1)/(2) above, derived from a different direction**,
and it predicts the channel ordering we measure without being fitted to it:

| channel | order of the motion | measured (A0) |
|---|---|---|
| `yaw_rate` on PhysicalAI | rotation — scale-free | **+0.9035** |
| `speed` | translation — up to one unknown scalar | **+0.8651** (oracle per-clip ceiling 0.942) |
| `long_accel` | translation, **twice** differentiated | **−0.2398** |

⇒ **`long_accel`'s difficulty is structural, not a tuning failure.** A modest
result there is the expected result, not a defect in the head.

⚠️ **The latent-action stack itself is NOT adopted, and its own pre-registered
refutation criteria are why** (2 of 4 fired). The sharpest is that **the sign is
inverted for us**: UniVLA / MVP-LAM / "Why Latent Actions Fail" all derive their
gains from *suppressing* ego motion — but for driving **ego motion IS the
action**. Reading their numbers as support would be reading a control's
improvement as the treatment's. Meanwhile four driving systems exploit exactly
our large-unlabelled/small-labelled asymmetry — LFG, Vista, **VPT (a
*supervised* IDM on 1,962 h)**, DriveVA — and **none uses a latent-action
bottleneck.** `INHERITED (sibling stream), not re-verified here`

---

---

## 4. Phase 2 — the label fixes (the largest measured lever, and it needed no GPU)

### 4.1 The defect, located by speed bin

Derived `yaw_rate` on comma2k19, all 64 segments, by ground-truth speed:

| speed bin (m/s) | n frames | share | yaw std | **% physically impossible (\|ω\|>1.5)** | max \|ω\| |
|---|---:|---:|---:|---:|---:|
| **[0.0, 0.5)** | 217 | 1.1 % | **4.0008** | **26.27 %** | **15.53 rad/s** |
| [0.5, 1.0) | 62 | 0.3 % | 0.0424 | **0.00 %** | 0.12 |
| [1.0, 2.0) | 184 | 1.0 % | 0.0481 | 0.00 % | 0.14 |
| [2.0, 5.0) | 575 | 3.0 % | 0.0562 | 0.00 % | 0.30 |
| [5.0, 10.0) | 1 540 | 8.1 % | 0.1198 | 0.00 % | 0.51 |
| [10.0, 20.0) | 2 028 | 10.6 % | 0.0183 | 0.00 % | 0.11 |
| [20.0, 100) | 14 466 | 75.8 % | 0.0264 | 0.00 % | 0.18 |

PhysicalAI: **zero** impossible frames in any bin — its heading comes from an
orientation quaternion, which is standstill-robust.
**The defect is razor-sharp and confined to v < 0.5 m/s.** That is what sets the
threshold; it was measured, not assumed. `MEASURED · DECISION-GRADE`

### 4.2 The repair, and E1's verdict

Hold the last *observable* heading direction through the standstill (operating on
the unit vector, so no 2π wrap can be introduced), then re-derive `yaw_rate`.
**The DEPLOYED head, nothing retrained:**

| protocol | n windows | pooled R² | PhysicalAI | comma2k19 |
|---|---:|---:|---:|---:|
| legacy — **what we published** | 4 195 | **+0.1046** | +0.9035 | +0.0114 |
| legacy, 9 impossible windows **deleted** (v2's fix) | 4 186 | +0.4967 | +0.9035 | +0.0719 |
| **repaired (hold)** | **4 195** | **+0.8108** | +0.9035 | **+0.3308** |
| comma.ai's own 4 m/s gate — *either* label set | 3 924 | +0.7612 | +0.8792 | +0.3383 |

> ### ⭐ E1 — **PASS.** The repair beats deletion (0.8108 vs 0.4967) **and
> discards nothing** (4 195 windows kept vs 4 186).

Two independent confirmations fall out of that table:
- **PhysicalAI is bit-identical at +0.9035 in every row.** The pre-registered
  falsifier — *"if the repair moves PhysicalAI yaw, it is touching something it
  must not"* — **did not fire**. `n_pai_changed = 0`, asserted in the artifact.
- **comma.ai's own protocol reaches the same place by a different route.** Their
  `calib_challenge` discards frames below 4 m/s; doing that gives the *identical*
  number (0.7612) for the legacy and repaired labels, because the 4 m/s gate
  removes exactly the frames the repair touches. **Our repair is strictly better:
  it keeps 271 more windows and scores higher.**

### 4.3 The audit — is this a real correction or an R² artifact?

R² carries the label's own variance in its denominator, so shrinking a wild label
can flatter it without the model improving. Three checks, all reported:

**(a) Isolation.** 4 145 of 4 195 windows are **bit-identical** before and after
(`max_abs_label_delta = 0.0`). 50 windows (1.19 %) changed. Zero are PhysicalAI.

**(b) What actually happened in those 50 windows** — this is the decisive panel:

| | value |
|---|---:|
| ground-truth **speed** there | max **0.528 m/s**, mean **0.042 m/s** — the vehicle is stationary |
| **legacy** label \|ω\| | max **9.47 rad/s** (543 °/s), mean 1.146 |
| **repaired** label \|ω\| | max **0.0436**, mean 0.00087 |
| **what A0 predicted** \|ω\| | mean **0.0233** |
| A0 MAE vs legacy label | **1.1419 rad/s** |
| A0 MAE vs repaired label | **0.0242 rad/s** |

> **The model was right and the label was wrong.** A road vehicle at v ≈ 0 cannot
> rotate in place; A0 said ≈ 0.02 rad/s, and the label said up to 543 °/s.

**(c) ⚠️ The honest caveat — outlier-proof statistics barely move:**

| statistic (pooled) | legacy | repaired | change |
|---|---:|---:|---|
| R² | 0.1046 | **0.8108** | +0.706 |
| MAE | 0.0405 | **0.0272** | **−33 %** |
| medAE | 0.02118 | 0.02107 | **−0.5 %** |
| **nMedAE** | 1.798 | **1.904** | **slightly WORSE** |
| Spearman ρ | 0.4143 | 0.4157 | +0.001 |

🔴 **So the claim must be stated precisely, and I state it against my own
interest:** the repair fixes the **tail** and the **mean-square summary
statistic**, not the typical prediction. nMedAE even worsens, purely because the
repaired label's MAD is smaller. **What was broken was our measurement and our
training signal — not, as the headline R² would suggest, the model's everyday
yaw accuracy on comma2k19.**

### 4.4 The training signal was broken too — and fixing it is a real, separated gain

`R0` (v1 recipe on repaired labels) vs `R0LEG` (identical recipe on legacy
labels), **both scored against the repaired labels**, paired over 36 episodes:

| | pooled R² | comma R² | paired Δ MAE vs R0LEG |
|---|---:|---:|---|
| R0LEG | 0.092 | 0.003 | — |
| **R0** | **0.841** | **0.679** | **−0.00320 [−0.00520, −0.00150] SEPARATED ✓** |

That is a **12 % MAE reduction from the label fix alone**, and it is a genuine
model improvement rather than a re-measurement. It also reproduces v2's positive
control from the other side: v2 injected 1 % bad yaw labels and measured
CI-separated degradation; v3 removed 0.7 % of real ones and measured
CI-separated improvement.

---

## 5. Phase 3 — geometry conditioning. Tested properly. **REFUTED.**

### 5.1 The 0-GPU closed-form test first (n = 40 PhysicalAI clips, all held out from A0)

| test | result |
|---|---|
| apply `v̂ · h/h̄`, **what eq. (1) prescribes** | MAE **2.960 → 3.236**, Δ CI [+0.051, +0.551] — **significantly WORSE** |
| apply the opposite sign `v̂ · h̄/h` | 2.960 → 2.826, not separated |
| **SHUFFLED heights (negative control)** | 2.960 → 2.862, not separated — **as good as the real ones** |
| oracle per-clip scale (the headroom that exists) | 2.960 → **1.607**, Δ CI [−1.869, −0.881] |
| partial corr(rel-bias, h \| clip mean speed) | **+0.469 [+0.052, +0.728]** — separated, **but the sign is backwards** |
| oracle scale factor `k` vs camera height | **r = −0.466**; partial given v̄ **−0.352** — **opposite** of the ground-plane sign |

### 5.2 The learned arms and their three controls — paired Δ MAE on speed, 36 episodes

Negative = the first arm is better. `*` = CI excludes 0.

| contrast | Δ MAE speed [95 % CI] | verdict |
|---|---|---|
| `Sctxn` clip-context vs `R0` | **−0.6253 [−1.3447, −0.0411] \*** | **clip context HELPS** |
| `V2R` v2 recipe vs `R0` | **−0.9401 [−1.5296, −0.4401] \*** | **the v2 recipe HELPS** |
| **`G1n` geometry vs `R0`** | **+0.4944 [+0.0001, +1.1021] \*** | **significantly WORSE than nothing** |
| `G1h` camera-height-only vs `R0` | +0.3794 [−0.0627, +0.9403] | worse, not separated |
| `Ccorpn` **corpus one-hot** vs `R0` | +0.1628 [−0.1425, +0.4847] | no effect |
| `Crign` **rig one-hot** vs `R0` | +0.2386 [−0.0785, +0.5591] | no effect |
| **`Cshufn` SHUFFLED geometry vs `R0`** | **+0.6424 [+0.1791, +1.2103] \*** | significantly worse |
| 🔴 **`G1n` vs `Cshufn` (real vs shuffled)** | **−0.1479 [−0.4830, +0.1821]** | **INDISTINGUISHABLE** |
| `G1n` vs `Ccorpn` | +0.3316 [−0.1488, +0.9377] | no better than the control |
| `G1` vs `V2R` (geometry on top of clip-context) | +0.1336 [−0.0716, +0.3614] | no gain |
| `G1` vs `Cshuf` (real vs shuffled, on clip-context) | −0.0360 [−0.1939, +0.1202] | **INDISTINGUISHABLE again** |
| **`G2` physics `v=(f·h)·Φ` vs `V2R`** | **+0.6712 [+0.2385, +1.1531] \*** | **significantly WORSE** |

> ### E2 — **FAIL, and cleanly.**
> Geometry conditioning does not beat its corpus-embedding control, does not beat
> its rig control, **is statistically indistinguishable from SHUFFLED geometry at
> both layers**, and is *significantly worse than no conditioning at all*.
>
> ### E4 — **the pre-registered prediction is CONFIRMED.** I wrote before the run
> that the physics arm `G2` would fail because the closed form already had. It
> did, with a CI-separated margin.
>
> ### E3 — **moot.** The discriminator interprets a positive speed effect; there
> is none to interpret.

### 5.3 What the negative control actually tells us — the mechanism

Real geometry ≡ shuffled geometry ⇒ the token carries **no information**; the
measured degradation is the **cost of the token itself**. And the *corpus* and
*rig* one-hots behave the same way. So the finding is not "camera geometry is
irrelevant to monocular scale" — physics says it is not. It is narrower and more
useful:

1. **Our pipeline has already consumed the geometry.** The f-theta crop
   canonicalises `f_eff` to ≈266 px **and** centres on the **per-clip principal
   point**. The frames handed to the frozen encoder are already
   geometry-normalised, so there is little left for a conditioning token to add.
2. **The residual — camera height — does not act the way the ground plane
   predicts.** Both the closed-form and the learned test agree, and the sign is
   backwards in the closed form. `cam_h` is plausibly acting as a *vehicle-class*
   proxy (taller mounts correlate with slower clips, r = −0.335) rather than as a
   scale knob.
3. **A low-dimensional clip-constant token appears to invite the head to lean on
   a prior instead of the image** — which is exactly the failure the shrinkage
   already represents. The 4 096-number clip-context token helps because it
   carries *per-clip scene* information; a 6-number geometry token does not.

### 5.4 ⭐ The decisive within-corpus test — and the trap it caught

Everything above pools two corpora, so a sceptic can always say the controls are
confounded by domain. So the same ladder was **trained AND evaluated on
PhysicalAI alone**: 26 train / 14 val episodes, 1 203 val windows. **Corpus
identity is constant there, while camera height still varies 1.245–1.607 m and
the principal point still splits into two rig clusters.** A gain here cannot be
corpus memorisation, because there is only one corpus.
*(Assertion checked in the artifact: the label repair changes **0** PhysicalAI
windows, so this arm is unaffected by Phase 2.)*

| contrast (speed, paired Δ MAE over 14 episodes) | result |
|---|---|
| `G1n` geometry **vs `R0` nothing** | +0.2049 [−0.2997, +0.7632] **not separated** |
| `G1h` camera-height-only **vs `R0` nothing** | +0.0713 [−0.3777, +0.5787] **not separated** |
| `Sctxn` clip-context vs `R0` | +0.1015 [−0.3122, +0.5537] **not separated** |
| `Cshufn` **SHUFFLED vs `R0`** | **+0.7843 [+0.1798, +1.4316] SEPARATED — worse** |
| **`G1n` real vs `Cshufn` shuffled** | **−0.5794 [−1.0133, −0.1680] SEPARATED** |
| **`G1h` real vs `Cshufn` shuffled** | **−0.7131 [−1.2318, −0.1986] SEPARATED** |

> 🔴 **The trap, stated plainly because I nearly walked into it.** Real geometry
> beats shuffled geometry with a **CI-separated** margin. Reported on its own,
> that reads as *"geometry conditioning works, p < 0.05"*. **It does not.** Real
> geometry is **statistically indistinguishable from no geometry at all**; it is
> only *shuffled* geometry that is significantly worse.
>
> **The real-vs-shuffled separation measures HARM AVOIDED, not BENEFIT GAINED.**
> True geometry is consistent with the pixels, so the head can ignore it at no
> cost; false geometry contradicts them, so it actively misleads. Only the
> **three-way** comparison — real / shuffled / **none** — recovers the truth.
> A two-arm design would have produced a confident false positive here.

**A second, unlooked-for result in the same table:** the clip-context token —
the one lever that *does* work in the pooled setting (−0.6253 separated) —
**buys nothing within a single corpus** (+0.1015, not separated). ⇒ **its win is
cross-corpus scale disambiguation, not per-clip scene information.** That
reframes v2's headline finding and is stated here as a correction to it.

---

*(§6 long_accel · §7 the shipped model · §8 what still fails — below.)*
