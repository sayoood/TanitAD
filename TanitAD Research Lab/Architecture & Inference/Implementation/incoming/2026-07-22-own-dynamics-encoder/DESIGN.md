# OUR OWN rig-robust dynamics-estimation encoder — design

> 🟥 **OUTCOME 2026-07-24 — the load-bearing mechanism (§3, GAIA-2 per-block camera conditioning) was BUILT,
> trained to 40k as `dynenc-branchB`, and REFUTED on held-out-rig transfer (`MEASURED`).** Best cross-rig
> speed R² **−0.667** (gate +0.9), yaw R² negative everywhere; **weaker** than the plain frozen flagship-v1
> encoder (+0.657) paired on 3/4 arms; own 40k head reads rig-B speed R² only **0.24**. The §0-row-3→§3
> thesis ("rig-invariance must be *engineered in* via explicit conditioning") is **not supported at this
> scale** — the deficit is upstream, representation *quality*. Full:
> `../../2026-07-24-branchb-transfer-eval/RESULTS_branchB.md`; `Project Steering/MODEL_REGISTRY.md §10`.
> Surviving lever (HYPOTHESIS, **Sayed-gated new arm, not auto-launch**): a flagship-warm-started,
> longer-trained, augmentation-matched encoder — the frozen flagship-v1 latent is the stronger (though not
> uniformly rig-robust) cross-rig substrate.

**Author:** dynamics-encoder stream (Sayed directive 2026-07-22: *"train our own encoder for the
IDM; we need our own model to estimate dynamics"*). **Status:** DESIGN + smoke-validated scaffolding,
staged, NOT launched (a multi-GPU-day run needs Sayed's go — see `LAUNCH_PLAN.md`).

**Evidence-class legend (CLAUDE.md operating standard):** `MEASURED` (ours, artifact-cited) ·
`PUBLISHED` (external, URL-cited) · `INHERITED` (another doc, not re-verified) · `HYPOTHESIS`
(extrapolation, not measured). **A claim that decides a GPU-day is MEASURED or PUBLISHED, never
INHERITED.**

**Reads:** `AGENT_OPERATING_STANDARD`, `RETRACTION_LOG`, `IDM_VIDEO_PRETRAIN_DESIGN` §3 (updated),
`Research/2026-07-22-encoder-strategy-and-vjepa2ac.md`, the landed IDM proof / re-gate / **multi-rig**
JSON, and the WAM deep-research output (`tasks/wgmi9zg09.output`, 11 verified claims — synthesis died on
a session limit, so this doc IS the synthesis).

---

## 0. The problem, stated as measurement

We need an encoder whose latent is **action/dynamics-predictive AND rig-robust**: it must let a small
head read ego **action** (speed, yaw-rate, steer, long-accel) and the metric ego-**trajectory** from
monocular driving video, and it must **keep doing so on a camera rig it never saw in training** (so we
can pseudo-label heterogeneous video, ultimately YouTube, and pretrain the WM at data scales the parity
corpus cannot reach — `IDM_VIDEO_PRETRAIN_DESIGN` §0).

Three converging results say the naive paths are dead, and **fix the exact target this design must hit:**

| # | finding | class | artifact |
|---|---|---|---|
| 1 | **Trained ≫ frozen-external in-distribution.** flagship-v1 ADE@2s **0.4522** vs REF-A frozen-DINOv2 **2.13–2.92** (strict parity, 4.7–6.5×). So the substrate is a **trained** encoder, not a frozen foundation model. | `MEASURED` | `MODEL_REGISTRY §1.2/§2` |
| 2 | **Even our trained encoder's latent is RIG-SPECIFIC.** Frozen IDM: in-dist speed R² **0.930** → cross-rig **−2.465**; f-theta canon a **no-op** (f_eff matched 266); light-FT **inert** cross-domain (0.406→0.411). | `MEASURED` | `…/idm-proof/results.json`, `results_regate.json` |
| 3 | **⭐ Multi-domain co-training does NOT fix it.** Co-train {rig-A + comma} → held-out **rig-B**: light-FT speed R² **−1.61** (vs −1.65 single-domain — no recovery). Symmetric {rig-A+rig-B} → held-out comma **0.452** (vs 0.411). **Data-diversity REFUTED; the collapse is REPRESENTATIONAL.** | `MEASURED` | `…/idm-proof/results_multirig.json` |
| 4 | **Scale doesn't buy it either.** V-JEPA2-AC freezes a **1M+ h** SSL encoder and still reports camera-pose sensitivity ("manually tried different camera positions"). | `PUBLISHED` | arXiv:2506.09985 §limitations |

**The conclusion that fixes the design:** rig-invariance is neither a data-diversity nor a scale
property — it must be **engineered into the encoder explicitly**. That single sentence makes the
camera-conditioning mechanism (§3) the load-bearing part, not an afterthought, and retires the "cheap
multi-domain-cotrain suffices" branch that the earlier plan carried.

---

## 1. What the field actually does — ranked shortlist to combine (WAM research, PUBLISHED)

The deep-research pass (24 primary sources, 11 verified 3-0/2-1 claims) gives us the mechanisms. **No
published model solves rig-robust *monocular* dynamics estimation for driving** (§7 verdict); we combine
the four best pieces:

| rank | borrow from | the specific idea | evidence |
|---|---|---|---|
| **1 ⭐** | **GAIA-2** (Wayve, 2025) | **Explicit camera-parameter conditioning**: separate learned embeddings for **intrinsics, extrinsics, distortion**, summed into one camera encoding, **injected at every transformer block**. Rig-generalization credited to *explicit conditioning + multi-rig training, NOT scale*. | `PUBLISHED` arXiv:2503.20523 (**verified 3-0**, verbatim §2.2.3) |
| **2** | **V-JEPA 2** + "Prediction over Reconstruction" | **Masked-latent (predictive) SSL** as the backbone objective — predictive temporal SSL beats pixel-reconstruction (MAE/VideoMAE) for **action-recoverability** and is **more robust** across corruption/occlusion axes; and "train on temporal video" is a bigger lever than the specific loss. | `PUBLISHED` arXiv:2506.09985; 2606.07687; 2606.31232 |
| **3** | **Cosmos-3** (NVIDIA) inverse-dynamics | **Ground metric scale by SUPERVISED in-domain trajectories, not monocular geometry.** Cosmos-3 recovers metric ego-pose from mono video and beats VGGT/DepthAnything3 by a huge margin (ATE **0.98 m** vs **23.46 / 9.29 m**) precisely because scale comes from supervised in-domain driving logs; general-domain VO **drifts on absolute scale**. | `PUBLISHED` cosmos3 tech report (**verified 3-0 / 2-0**) |
| **4** | **OmniNWM / PRoPE / Plücker-raymaps** (complementary) | **Geometry-as-input** alternatives to global conditioning: normalized Plücker ray-maps decouple motion from intrinsics (OmniNWM, verified 3-0); per-pixel Plücker concat-to-RGB or projective relative positional encoding improve novel-viewpoint / OOD-intrinsics generalization (single-source, lower confidence). Held as **design options** to add if §3 global conditioning underperforms. | `PUBLISHED` 2510.18313 (3-0); 2510.02268, 2507.10496 (single-source) |

Label-efficiency (why a small CAN-labeled set suffices to ground the head): the **non-causal IDM** is a
lower-complexity target than a policy (VPT 2206.11795; IDM-vs-BC 2602.02762); latent-action methods reach
control with **~5 %** action labels (LAWM 2512.10016) / **150 traj** (LAPA 2410.11758) — we keep
continuous regression, not a codebook (metric precision; `IDM_VIDEO_PRETRAIN_DESIGN` §1).

**Contrast that rules OUT the easy reads** (C6 discipline): Vista (2405.17398) is a *forward* WM with **no
camera conditioning** — not an IDM, does not solve rig-robustness. Doe-1 (2412.09627) and DINO-WM
(`D5RNACOZEI`) use **frozen** general tokenizers — REF-A's closed ceiling. TRIG (2607.05801) decouples
pose on a **frozen DINOv2** but is **multi-cam surround, not monocular**, and reports **no cross-rig
eval**. GAIA-2 is **multi-view generative** and needs **known calibration + multi-rig data** — we adapt
its *conditioning*, not its task.

---

## 2. The encoder — backbone + objective

### 2.1 Backbone (video ViT, sub-300M) — reuse the flagship v1 arch, add geometry conditioning

`MEASURED` justification for reusing the flagship encoder arch: it is the best in-program in-dist arm
(row 1), and its input contract already makes short-horizon motion visible to the encoder.

- **Input:** 9-channel = 3 RGB frames channel-stacked at 100 ms spacing, 256 px (D-015 contract), patch
  16 → **16×16 = 256 tokens**, `d_model 768`, **depth 12** (flagship4b encoder). Spatial-grid readout
  4×4 × d_readout 128 → **state_dim 2048** (flagship parity; A7 never global-pool).
- **Multi-frame-in-one-encode** already gives per-encode motion; the window (2k+1=9 non-causal frames)
  adds ~0.8 s temporal context to the IDM head. The masked-latent SSL (§2.2) makes the *encoder itself*
  temporally predictive — the "video ViT" property — without a heavier tubelet-attention stack in the
  modest branch. The FAIL/scale branch (`LAUNCH_PLAN` Branch B) swaps in tubelet + temporal attention.
- **Measured budget** (launch config, instantiated): ViT **87.0 M** + GAIA-2 conditioning **7.4 M** +
  readout 0.1 M + IDM head 2.9 M = **DEPLOYABLE 97.4 M** — comfortably sub-300 M, with headroom to widen
  or deepen for Branch B. (`MEASURED` — `stack/tests/test_dynamics_encoder.py::test_launch_config_*`,
  and `smoke_report.json`.)

### 2.2 SSL / training objective — ranked and chosen (a multi-task loss)

Each objective addresses a distinct requirement; they are co-trained. Ranking rationale is `PUBLISHED`
+ `MEASURED`, not taste.

| objective | what it buys | why (cited) | our module |
|---|---|---|---|
| **Masked-latent prediction (V-JEPA2)** | a general, temporally-predictive, **scalable-to-unlabeled** latent | predictive SSL > reconstruction for action-recoverability + robustness (2606.07687, 2606.31232); scales to YouTube (2506.09985) | `MaskedLatentPredictor` (new) |
| **Action-conditioned forward prediction** | makes the latent **action/dynamics-predictive** by construction | our flagship's proven recipe (row 1); DINO-WM causal-ViT predictor (`D5RNACOZEI`) | `OperativePredictor` (reuse) |
| **SIGReg anti-collapse** | prevents representation collapse without EMA/stop-grad | LeJEPA 2511.08544 (our validated `sigreg.py`, λ=0.1) | `SigReg` (reuse) |
| **Supervised metric IDM** | the **direct dynamics readout** + label-efficient grounding | VPT non-causal IDM 2206.11795; DriveWAM forecast-then-decode 2605.28544 | `IDMHead` (reuse `idm_head.py`) |
| **Metric-scale grounding** | metre-scale ego-motion in the latent | Cosmos-3: supervised in-domain grounding ≫ mono-geometry (ATE 0.98 vs 23.46 m) | `MetricInverseDynamics` (reuse `metric_dynamics.py`) |

**Rejected as primary:** discrete latent-action codebooks (Genie/LAPA quantization) — a mismatch for
continuous metric steer/accel/yaw (`IDM_VIDEO_PRETRAIN_DESIGN` §1, `PUBLISHED`); kept only as an optional
auxiliary CE over the v3 factorised LAT×LON vocabulary. Pure pixel-reconstruction (VideoMAE) — dominated
for action-recoverability (2606.07687).

The action/dynamics **head** is the existing **non-causal `IDMHead`** (bidirectional over the 9-frame
window, center readout of speed/yaw/steer/accel + 2 s ego trajectory; VPT's offline-labeler trick), plus
the **forecast-latent auxiliary** (decode the action from the predictor's *imagined* future latent — the
Seer/DriveWAM path) which reuses the shared predictor. This is `IDM_VIDEO_PRETRAIN_DESIGN` §2 unchanged;
what changes vs that doc is the **substrate is no longer frozen** — the encoder trains, and it is
camera-conditioned.

---

## 3. ⭐ The rig-robustness mechanism (the load-bearing part) — GAIA-2 per-block camera conditioning

This is the disruptive core. The measured failure (§0 rows 2–4) is that a bare encoder **has no explicit
geometry**, so it *implicitly* binds the mapping "image-space optical flow → metric ego-motion" to one
camera pose. rig-A (cy≈543) and rig-B (cy≈755) differ in the **extrinsics** (camera pitch/height vs the
road), so the same ego speed produces different image flow; the encoder learned rig-A's mapping and it is
**wrong** for rig-B — and neither more data (row 3) nor more scale (row 4) repairs a mapping the encoder
was never told is rig-dependent.

**The fix: make the geometry an INPUT.** Port GAIA-2's mechanism (arXiv:2503.20523, verified 3-0):

```
camera params ─► [intrinsics MLP]  ┐
(f_eff,cx,cy)    (extrinsics MLP)   ├─ sum ─► unified camera encoding  z_cam ∈ R^768
(pitch,h,roll)   (distortion MLP)   ┘                                    │
(k1,is_fisheye)                                            per-block:  t ← t + inject_i(z_cam)
                                                           then block_i(t)     (i = 1..depth)
```

- **Separate embeddings for intrinsics / extrinsics / distortion, summed** — GAIA-2 verbatim ("We compute
  separate embeddings for intrinsics, extrinsics, and distortion, which are then summed to form a unified
  camera encoding"). Intrinsics = focal + principal point (GAIA-2 §2.2.3, verified). Distortion carries
  the f-theta-vs-rectilinear model flag (PhysicalAI is fisheye, comma rectilinear).
- **Injected at EVERY transformer block** (GAIA-2 "added to the input latents at each transformer block"),
  via a **zero-init** per-block projection so the encoder is a **plain ViT at init** → flagship-v1 weights
  **warm-start byte-identically** (the whole point of the modest branch).
- **A per-parameter known/unknown MASK** so **"unknown intrinsics" is itself in-distribution** — L2D ships
  no intrinsics (`l2d.py`), and YouTube won't either; the mask lets the encoder fall back gracefully
  rather than being fed a false f_eff as truth. This is our extension beyond GAIA-2 (which assumes known
  calibration) and is exactly what an uncalibrated-video labeler needs.

**Why this is expected to work where diversity/scale failed** (`HYPOTHESIS`, pre-registered test in §6):
GAIA-2 *credits its cross-rig generalization to this conditioning* (verified 3-0), and it is the one lever
neither our multi-rig cotrain (row 3) nor V-JEPA2-AC (row 4) had. Rig-robustness comes from **explicit
conditioning ⊗ multi-rig training**, both required — GAIA-2's own attribution.

**Three orthogonal supports around the mechanism (all engineered in the scaffolding):**
1. **Multi-domain / multi-rig co-training** (§4) — necessary but not sufficient alone (row 3); it is the
   *training-time coverage* half of GAIA-2's "conditioning + diversity".
2. **Geometry (extrinsics) domain-randomisation** — jitter camera pitch/height (a vertical-shift ≈ small
   pitch homography) **and feed the matching camera params**, so the encoder is forced to *use* the
   conditioning to explain the perturbation rather than memorise one rig. Implemented consistent
   (`geom_augment`): a vertical image shift updates cy and pitch = atan(dv/f_eff) together.
3. **Complementary geometry-as-input options** (design fallbacks if the global conditioning underperforms
   the §6 gate): per-pixel **Plücker ray-maps** concatenated to the 9-ch input (2510.02268) or **PRoPE**
   projective relative positional encoding (2507.10496). These are input-/attention-level rig signals that
   compose with the GAIA-2 global conditioning; pre-registered as the FAIL-branch escalation, not the
   first bet.

---

## 4. Metric-scale grounding + multi-domain data

### 4.1 Metric scale (Cosmos-3-validated: supervised in-domain, not monocular geometry)

- **Primary — supervised odometry grounding.** `MetricInverseDynamics` regresses the metric relative
  ego-pose (Δx, Δy, Δyaw) from a latent pair against CAN/IMU odometry (our `metric_dynamics.py`, the
  `_ego` convention). Cosmos-3 is the `PUBLISHED` evidence this beats monocular geometry on absolute scale
  (ATE 0.98 vs 23.46 m); general-domain VO **drifts**.
- **Speed-prior scale head.** Predict a **scale-normalized** trajectory + a **speed magnitude**, recover
  metric scale from the speed prior (CAN `v0`; dashcam speedometer OCR; or a per-clip speed distribution).
  Speed R² is invariant to a global scale, so this fixes **MAE/ADE**, not R² (noted; the §6 gate reads R²
  on speed/yaw which the scale head cannot game).
- **Optional weak prior for label-free video** (YouTube, no CAN): a monocular **metric-depth** teacher —
  UniDepthV2 (2502.20110, calibration-free metric depth at inference) or Metric3D-v2 (2404.15506,
  canonical-camera-space transform). Used as a *weak* scale prior only; Cosmos-3 says general-domain
  geometry drifts, so it never becomes the anchor.

### 4.2 Multi-domain data mix (parity-firewalled SIDE model)

| domain | role | camera params fed | licence/notes |
|---|---|---|---|
| **PhysicalAI-AV rig-A / rig-B** | the measured cross-rig pair; f-theta fisheye | intrinsics + extrinsics (per-clip cy→pitch/height); `is_fisheye=1` | our corpus (both rigs; the built-in intrinsics testbed) |
| **comma2k19** | different vehicle/rectilinear rig | f_eff 266 rectilinear; `is_fisheye=0` | ungated HF mirror (`comma2k19.py`) |
| **L2D (yaak)** | 735 h, extrinsics-only | **intrinsics UNKNOWN (mask 0)** + extrinsics_RDF | Apache-2.0 tier `ship` (`l2d.py`) |
| **YouTube (later)** | the scale lever | estimated + flagged (horizon/VP or a learned intrinsics head), mask-marked | tier-gated (`YOUTUBE_DASHCAM_STRATEGY.md`); IDM-labeled, pretrain-prefix only |

**Parity firewall (sacred):** this is a SIDE model. It never reads the WM parity key `e438721ae894` /
skip-hash `f09e44db` as truth and never re-selects parity episodes; its splits are **by rig / by corpus**,
orthogonal to the WM selection. YouTube-pseudo-labeled video is a **pretraining prefix** for the WM,
strictly separated from the parity fine-tune (`IDM_VIDEO_PRETRAIN_DESIGN` §3/§6).

---

## 5. The scaffolding (code) — what is built and smoke-validated

All additive (no edits to existing modules); full suite green (778 passed, 2 skipped).

| file | what |
|---|---|
| `stack/tanitad/models/dynamics_encoder.py` | `CameraConditionedEncoder` (GAIA-2 per-block conditioning), `CameraEncoding`, `MaskedLatentPredictor`, `DynamicsEncoderModel` (the combined objective), `DynEncConfig` (+smoke). Reuses `ViTEncoder`/`SpatialGridReadout`/`OperativePredictor`/`SigReg`/`MetricInverseDynamics`/`IDMHead`. |
| `stack/scripts/train_dynamics_encoder.py` | `MultiDomainWindowDataset` (domain mixing + `geom_augment` geometry randomisation), the training loop, `build_domains_from_caches` (launch data path), `maybe_warm_start` (flagship-v1), and `--smoke` (self-contained CPU proof). |
| `stack/tests/test_dynamics_encoder.py` | 5 CPU tests (pipeline finite/differentiable/fits/mixes-domains; zero-init conditioning == identity; launch-config sub-300M + state_dim 2048; geometry aug consistency; dataset balancing). |

**Smoke result** (`MEASURED`, `smoke_report.json`, CPU, tiny config): pipeline **finite + differentiable**
(grad-norm 338), batch **mixes 4 domains** (rig-A/rig-B/comma/L2D-unknown-intrinsics), **camera
conditioning live** (mean|Δz| = 2.7e-2 on a pitch-param change with the injection enabled), all five
sub-losses finite, and the combined loss **falls 4.80 → 1.88** with supervised **IDM 2.76 → 0.98** over 40
steps. This proves *encoder + GAIA-2 conditioning + masked-latent SSL + forward-pred + SIGReg + IDM head +
metric grounding + multi-domain dataloader* is a working, learnable pipeline — the pre-launch gate.

---

## 6. The pre-registered decisive experiment (see `PRE_REGISTRATION.md`)

The multi-rig cotrain already answered "does diversity alone fix it?" — **no**. The **cheapest next
question that decides the whole line** is: **does GAIA-2 explicit camera-conditioning recover cross-rig
transfer?** Pre-registered ablation, both outcomes committed, on data we already have (~hours on pod3),
reusing the re-gate harness: **conditioning ON vs OFF**, held-out rig, same gate
(cross speed R² > 0.9 AND yaw R² > 0.9 AND ADE@2s < 1.5× in-domain). PASS/material-recovery → the mechanism
works, scale it; FAIL → the conditioning must be learned from-scratch jointly with the SSL encoder (the
pre-registered justification for the expensive Branch B). Full config + compute in `LAUNCH_PLAN.md`.

---

## 7. Verdict — is this open territory?

**Yes.** `PUBLISHED` (WAM research, adversarially verified): **no published world-action model solves
rig-robust *monocular* dynamics estimation for driving.** GAIA-2 has the conditioning but is a multi-view
*generative* model needing known calibration + multi-rig data (not an IDM). Cosmos-3 does monocular IDM
and metric scale but grounds scale by supervised in-domain data and makes **no cross-rig claim**.
V-JEPA2-AC has scale but the *documented* rig-sensitivity. Our contribution — **a trained, GAIA-2-camera-
conditioned, V-JEPA2-masked-latent driving encoder that recovers metric ego-dynamics and stays valid
across an unseen rig, with unknown-intrinsics handled by a mask** — occupies genuinely open ground, and it
is built to attack the one failure (§0 rows 2–4) that our own MEASUREMENTS and the closest PUBLISHED
precedent both exhibit.

**[UPDATE 2026-07-24] The territory is open; our first occupancy of it FAILED.** The design was built
(`dynenc-branchB`, 40k) and measured: it did **not** recover cross-rig transfer (best speed R² −0.667) and
was a weaker substrate than the plain flagship-v1 encoder (`MODEL_REGISTRY §10`). "Open territory" was
correct — *no one has solved rig-robust monocular driving dynamics* — but this particular attack (explicit
GAIA-2 conditioning learned from-scratch at 40k) is now `MEASURED`-refuted. The ground is still open; the
next attempt should be a flagship-warm-started variant (Sayed-gated), not more from-scratch conditioning.

---

## Sources

GAIA-2 arXiv:2503.20523 (verified 3-0) · V-JEPA2/-AC 2506.09985 · Cosmos-3 (research.nvidia.com/labs/
cosmos-lab/cosmos3, verified 3-0/2-0) · OmniNWM 2510.18313 (3-0) · Prediction-over-Reconstruction
2606.07687 · JEPA-robustness 2606.31232 · Plücker-raymap 2510.02268 · PRoPE 2507.10496 · UniDrive
2410.13864 · TRIG 2607.05801 · Metric3D-v2 2404.15506 · UniDepthV2 2502.20110 · VPT 2206.11795 · DriveWAM
2605.28544 · LAPA 2410.11758 · LAWM 2512.10016 · Vista 2405.17398 · DINO-WM openreview D5RNACOZEI · LeJEPA
2511.08544. Full claim ledger + votes: `tasks/wgmi9zg09.output`. Our MEASURED inputs: `…/idm-proof/*.json`,
`MODEL_REGISTRY §1.2/§2`.
