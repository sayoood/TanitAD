---
license: other
license_name: tanitad-research
tags:
  - autonomous-driving
  - inverse-dynamics
  - ego-motion
  - video-pretraining
  - pseudo-labelling
extra_gated_prompt: >-
  Research use. These are WEIGHTS ONLY. They were trained on frozen latents
  derived in part from NVIDIA PhysicalAI-Autonomous-Vehicles, which is a GATED
  dataset; no dataset content, clip identifier, frame or pose is redistributed
  here. You must obtain the underlying datasets yourself from their own sources.
---

# TanitAD `idm_head_v3` — inverse-dynamics head on a frozen driving world-model encoder

A small non-causal transformer that reads a window of **frozen** latents from the
TanitAD flagship-v1 encoder and regresses the ego-vehicle's state at the window
centre. It is an **offline pseudo-labeller** — non-causality is intended, and it
is never run as a policy.

## Headline

> **The predecessor's published `yaw_rate` R² of 0.105 was measuring a broken
> label, not a broken model.** comma2k19 heading is `arctan2` of the ENU
> velocity, which is undefined at standstill. Repairing it moves the *same
> already-deployed head* from **0.105 to 0.811** with nothing retrained, and
> retraining on the repaired labels reaches **0.841**.

> **Camera-geometry conditioning was tested with three controls and REFUTED.**
> It is reported here because a refuted idea and an untried idea are worth very
> different amounts.

## Results — per channel, PER CORPUS, never pooled alone

`n` = 4,195 held-out windows over **36 episodes**, episode-disjoint from
training. Intervals are a **paired episode-cluster bootstrap** (B = 2000, unit =
episode). Point estimates are the mean over 3 seeds.

| channel | pooled R² | **PhysicalAI R²** | **comma2k19 R²** | MAE | unit |
|---|---:|---:|---:|---:|---|
| **speed** | **+0.907** | +0.856 | +0.878 | 2.662 | m/s |
| **yaw_rate** | **+0.841** | +0.868 | +0.679 | 0.0212 | rad/s |
| **steer** | +0.408 | +0.360 | +0.583 | 0.0155 | road-wheel rad |
| ~~long_accel~~ | −0.243 | −0.089 | −0.714 | 0.471 | **NOT SHIPPED** |

### Against the previously deployed head (`idm_head_v1`), both scored on the repaired labels

| channel | paired Δ MAE [95 % CI] | verdict |
|---|---|---|
| **yaw_rate** | **−0.0060 [−0.0090, −0.0029]** | **CI-separated improvement (−22 % MAE)** |
| speed | −0.3327 [−0.9536, +0.3022] | better, not separated |
| steer | +0.0035 [−0.0005, +0.0093] | **worse**, not separated |
| long_accel | +0.0320 [−0.0367, +0.1132] | not separated |

## ⚠️ Known failures — read these before using it

1. **`long_accel` is NOT a supported output. Its R² is NEGATIVE on both corpora**
   (−0.089 PhysicalAI, −0.714 comma2k19) — worse than predicting the mean. It is
   excluded from `scalar_names`. Two independent reasons: on PhysicalAI the CAN
   label correlates only r = 0.434 with the vehicle's own dv/dt, so a *perfect*
   kinematic estimator caps at R² 0.188 against it; and on comma2k19, where the
   label is clean, a linear probe on the frozen latent reads −0.095, i.e. the
   representation does not contain acceleration. A 101-bin HL-Gauss ordinal
   classification head was tried and **also failed** (−0.26 to −1.17) at a
   measured discretisation ceiling of 0.96–1.00, so this is not a binning
   artifact.
2. **`steer` is WORSE than the previous head (0.408 vs 0.742).** This model was
   trained on 68 clips; the previous one had 160. Do not use v3 for `steer` —
   this is a data-budget regression, not a recipe improvement.
3. **`steer` is not the same physical quantity across corpora.** PhysicalAI's is
   `atan(L·κ)` from a per-clip wheelbase; comma2k19's uses `STEER_RATIO = 15.3`.
   **Never pool steer across corpora.**
4. **comma2k19 `yaw_rate` (0.679) remains well below PhysicalAI (0.868)** even
   after the repair, because comma's heading is a *derived* quantity throughout.
   The dataset ships its own fused INS/GNSS/Vision orientation, which would be a
   better primary source; that fix is not in this model.
5. **Speed carries a residual per-clip scale bias.** An oracle per-clip rescale
   would take PhysicalAI speed MAE from 2.960 to 1.607 m/s. That headroom is
   real and unclaimed — see the geometry section.

## Architecture — a two-expert composite

v3's measurement is that **rotation and translation want different recipes**, so
the artifact holds two heads and routes channels between them:

| expert | channels | window | d_model | clip-context |
|---|---|---|---|---|
| `rotation` | `yaw_rate`, `steer` | 9 frames (k=4) | 256 | no |
| `translation` | `speed`, trajectory | 17 frames (k=8) | 128 | yes |

Total **4,301,848** parameters. Both read the same frozen 2048-d latents at 10 Hz.

⚠️ Decoupling by **loss weights** (training one head with the other channels
masked) was also tested and was **worse** than the joint head — it is the
*recipe*, not the gradient isolation, that differs.

## Camera geometry — tested and refuted, with all three controls

The obvious hypothesis was that this model fails at metric speed because it is
never told the camera geometry. That was tested, not assumed.

**Measured first (from PhysicalAI-AV's own calibration features, 40 clips):
camera height is per-clip and varies 1.245–1.607 m — a 29 % spread. All three
camera-height constants circulating in our codebase (1.5 / 1.43 / 1.22 m) are
wrong; 1.22 is below the observed minimum. Rig identity is NOT a proxy for
height (rig medians differ by 1.5 % while the within-rig spread is 29 %).**

Then, on speed (paired Δ MAE, negative = better):

| arm | vs no conditioning |
|---|---|
| geometry token (focal, height, f·h, pitch, principal point) | **+0.494 [+0.000, +1.102] — significantly WORSE** |
| camera height alone | +0.379, not separated |
| **corpus one-hot (CONTROL)** | +0.163, not separated |
| **rig one-hot (CONTROL)** | +0.239, not separated |
| **shuffled geometry (NEGATIVE CONTROL)** | +0.642 [+0.179, +1.210] — worse |
| **real geometry vs shuffled geometry** | **−0.148 [−0.483, +0.182] — INDISTINGUISHABLE** |
| physics parametrisation `v = (f·h)·Φ` | **+0.671 [+0.239, +1.153] — significantly WORSE** |

The closed-form version agrees: applying the ground-plane correction `v̂·h/h̄`
made MAE **worse** (2.960 → 3.236, CI-separated), and the oracle per-clip scale
factor tracks camera height at **r = −0.466**, the *opposite* of the
ground-plane sign.

**Why (best current reading):** the preprocessing already canonicalises the
effective focal to ≈266 px on every corpus **and** crops around the per-clip
principal point, so the frames reaching the frozen encoder are already
geometry-normalised. The residual — mount height — behaves like a vehicle-class
proxy, not a scale knob.

## Provenance

| | |
|---|---|
| encoder | TanitAD flagship-v1 `flagship4b-speedjerk-30k`, **FROZEN**, md5 `b5f07d9e3dd2ca643949bc86832e6585`, step 29999, state_dim 2048 |
| corpora | NVIDIA PhysicalAI-Autonomous-Vehicles (**gated**) + comma2k19 (`commaai/comma2k19`) |
| substrate | 104 episodes → 68 train / 36 val, episode-disjoint, domain-stratified |
| windows | 15,875 train / 4,195 val |
| estimator | `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2000, unit = episode. `overlapping_holdout_se` is **not** used |
| seeds | 3 per arm; the shipped weights are **seed 0** while the tabulated metrics are the 3-seed mean |
| contents of this repo | **weights + scalar config only** — no frames, no poses, no clip identifiers, no per-clip geometry table |

## Intended use and limits

Offline pseudo-labelling of driving video for which the TanitAD flagship-v1
encoder is appropriate. **Not** a controller, **not** a safety component, and not
validated outside the two corpora above. Out-of-corpus cameras differ in field
of view (a survey of driving video measured HFOV 32–77°, median 66.6°) and this
model has no mechanism to adapt to that — as the geometry section shows, giving
it one did not help.

## Licence

Weights only. PhysicalAI-AV is gated and none of its content is redistributed
here; comma2k19 is MIT at source. Obtain both datasets from their own providers
under their own terms.
