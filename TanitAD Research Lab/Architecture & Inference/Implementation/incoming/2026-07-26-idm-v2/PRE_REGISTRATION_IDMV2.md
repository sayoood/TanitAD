# PRE-REGISTRATION — IDM v2

**Written 2026-07-26 01:52 UTC, BEFORE any v2 arm was trained.** The diagnosis
(a)–(d) that motivates it was complete and is frozen in `IDM_DIAGNOSIS.md`; the
candidate arms below had not been run when this file was written.
Agent: `idm-v2` · Pod: `tanitad-eval` (A40, 0 MiB used, not training anything).
pod1 / pod2 / pod3 were not touched.

---

## 0. Why a pre-registration at all

Operating standard rule 5: *settle conflicts with the cheapest discriminating
experiment, pre-registered with BOTH outcomes committed in advance.* Every arm
below has a stated outcome for **fail** as well as for **pass**, and the
falsifiers in §5 are the ones that would make me withdraw the diagnosis rather
than re-explain it.

## 1. Substrate (MEASURED, fixed before the arms)

| | |
|---|---|
| encoder | flagship-v1 `flagship4b-speedjerk-30k`, **FROZEN**, ckpt md5 `b5f07d9e3dd2ca643949bc86832e6585` (asserted in `idm2_encode.py`), step 29999, state_dim 2048 |
| corpora | PhysicalAI-val `physicalai-val-0c5f7dac3b11` (40 eps, T=199) + comma2k19-val `comma2k19-val-76b6e94a97a1` (64 eps, T=300) — **104 episodes**, encoded once |
| split | episode-disjoint, domain-stratified, deterministic (`idm2_lib.split_tags`, every 3rd episode of each domain → val): **68 train / 36 val episodes** |
| windows | built at k=8 (17 frames) ONCE; 9-frame arms read the inner slice, so **every arm scores the identical val windows** and every contrast is paired |
| n | train **15,875** windows (stride 1) · val **4,195** windows (stride 2), 36 episodes |
| held-out from A0 | `idm_head_v1` was trained on pod3 `tr_a/tr_b/cm` tags — it has seen **none** of these 104 episodes |

**Estimator, fixed:** `taniteval.ci.episode_cluster_bootstrap` and
`paired_episode_cluster_bootstrap`, resampling the **36 val episodes**,
n_boot = 2000. `overlapping_holdout_se` is not called anywhere in this work.

## 2. Metrics — and the ones that are INADMISSIBLE here

R² is the wrong summary for `yaw_rate` and `long_accel` on this corpus and I am
not going to headline it. Measured reason: comma2k19 `yaw_rate` has
std 0.4290 rad/s but MAD 0.0112 rad/s (std/MAD = 38 vs 1.48 for a Gaussian) —
a tiny core with catastrophic outliers, so R²'s denominator is set by the
outliers and R² can be driven anywhere by 0.3 % of frames.

Every channel is therefore reported as **R² · Spearman ρ · MAE · medAE ·
nMedAE**, where

> **nMedAE = medAE / MAD(label)** — scale-free. nMedAE < 1 means the model
> beats "always predict the label's median"; it cannot be inflated by a
> low-variance channel and it cannot be destroyed by 0.3 % of corrupt labels.

Trajectory is reported with the **lat/lon decomposition and lat p90 @ 2 s**
(the M1 recommendation from `LATERAL_VS_LONGITUDINAL_ANALYSIS.md`), never as a
bare ADE.

**Not admissible in this work:** any single-clip number as a headline; any R² on
`yaw_rate`/`long_accel` without its nMedAE beside it; any interval from
`overlapping_holdout_se`.

## 3. Arms (each adds exactly ONE thing to its predecessor)

| arm | change | tests |
|---|---|---|
| **A0** | `idm_head_v1.pt` (the persisted artifact, md5 `fa4462f0…`) evaluated as-is | the deployed **before** |
| **B0** | v1 *recipe* retrained on this split | substrate control for A0 |
| **B1** | + **robust targets**: winsorise each channel at a physical limit (yaw 1.5 rad/s, accel 12 m/s², speed 60 m/s, steer 1.0) and standardise by **median / 1.4826·MAD** instead of mean/std | (b) label contamination |
| **B2** | + **log-speed** parametrisation (regress log v, exponentiate) | (c) multiplicative error |
| **B3** | + **clip-context token** (mean & std of z over the WHOLE clip → one extra token). Legitimate: the IDM is an offline, already-non-causal labeler | (c) per-clip/camera gain |
| **B4** | + **derived targets**: predict a per-frame **speed sequence**, obtain `long_accel` as its Savitzky-Golay derivative; obtain `steer` from bicycle geometry `a·(ŷaw/v̂)+b`; both drop out of the loss | targets |
| **B5** | + **17-frame window** | receptive field |
| **S2…S5** | each single change applied to **B0 alone** (log-speed / ctx / derived / w17) | de-confounds the ladder |
| **P1** | B0 with **1 % of train yaw labels replaced by ±8 rad/s** | POSITIVE CONTROL for the contamination mechanism |

3 seeds (0,1,2) for B0–B5, 2 seeds for S/P arms. 50 epochs, AdamW lr 3e-4,
wd 0.01, batch 256 — v1's recipe held fixed everywhere except the named change.

## 4. What counts as an improvement — declared now

Primary endpoints, on **n = 4,195 val windows / 36 episodes**, paired
episode-cluster bootstrap vs **A0**:

1. **yaw_rate — PASS** iff the paired CI on Δ medAE excludes 0 in v2's favour
   **and** pooled **nMedAE drops below 1.0** (i.e. the channel finally beats
   "predict the median"). A0's pooled yaw R² on the card is 0.0104; I will
   *also* report R², but R² is not the gate.
2. **speed — PASS** iff the paired CI on Δ MAE excludes 0 in v2's favour **and**
   the pooled effect is **≥ 0.5 m/s**.
3. **long_accel — PASS** iff the derived-accel arm reaches **R² ≥ 0.30 against
   the kinematic target dv/dt** *and* is **not worse than A0 against the
   original CAN label**.
   ⚠️ **Committed in advance, independent of the outcome:** the measured
   best-affine-in-dv/dt ceiling against the PhysicalAI CAN `long_accel` label is
   **R² = 0.188** (§(b) of the diagnosis). If that number stands, then *no*
   architecture can exceed 0.188 on that label and I will recommend **changing
   the published target to the kinematic one**, whether or not arm B4 passes.
4. **steer — PASS (for dropping the channel)** iff the *derived* steer is **not
   worse** than the *regressed* steer (paired CI on Δ medAE does **not** exclude
   0 against it, or excludes it in the derived channel's favour). Passing
   licenses removing steer from the loss.

Secondary (reported, not gating): ADE@2s with lat/lon split and lat p90 @ 2 s;
per-domain everything; cross-domain transfer.

## 5. Falsifiers — what would make me withdraw the diagnosis

| if | then the verdict is WRONG and I report |
|---|---|
| **B1** does not move comma yaw_rate | the yaw ceiling is **not** the label → verdict flips from (b) to (a) |
| **P1** does not collapse yaw R² toward 0 | the contamination mechanism is **refuted**; the card's 0.79 rad/s train std has another cause |
| the learning curve is still rising at 68 clips / 200 epochs | **(d) is not rejected** — "more training" is live and I say so |
| **B2** (log-speed) does not improve speed | the multiplicative-error reading of the cross-domain gap is wrong |
| **B3** (clip context) does not close the cross-domain speed gap | the camera gain is **not** inferable from the latents; metric grounding then needs the actual `cam_h`/`f_eff` per corpus, and I will say that is the required next step rather than claim a fix |
| the linear probe were to fall far *below* the trained head | the head **is** doing real work and (d) cannot be dismissed |

## 6. Known confounds I am NOT resolving here (stated, not hidden)

- The val corpora are the *val* builds available on the eval pod, not the pod3
  `lat_flagshipv1` latents the card was measured on (pod3 is running E1c and is
  off-limits). A0's numbers here are therefore **not** the card's numbers; they
  are A0 measured on a new, larger, still-fully-held-out set. Both are reported.
- comma2k19's `steer` is derived with `STEER_RATIO = 15.3` while PhysicalAI uses
  `atan(WHEELBASE·κ)` with `WHEELBASE = 2.9` — **the two corpora's steer is not
  the same physical quantity**, so pooled steer numbers are units-confounded by
  construction and are reported per-domain.
- `f_eff` is already canonicalised to ≈266 px on all three corpora
  (rig-A 266.13 / rig-B 266.10 / comma 266.50, `results_regate.json`), so any
  residual camera gain is **height/pitch**, not focal length. The repo carries
  three mutually inconsistent `cam_h` values (1.5 / 1.43 / 1.22); I do not
  resolve that here and no claim below depends on which is right.
