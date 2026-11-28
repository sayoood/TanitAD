# P3 — TanitAD_DataReconstruction — SPEC

`Product spec. Owner: TanitAD_DataFlyWheel. Status: SPEC v1, 2026-08-23.`
`Governed by Project Steering/TANITAD_PROGRAMME.md (§1 P3 row, §3 schema, §6 quality).`
`Where this doc and MODEL_REGISTRY.md disagree, the registry wins and this doc gets fixed.`

---

## 0. One line

**Turn video that carries no labels — dashcam, smartphone, YouTube — into episodes that obey
the TanitAD episode contract**, by *reconstructing* the three things such video is missing:
**camera calibration**, **metric ego-motion**, and **actions**.

The action half is the **IDM** (Inverse Dynamics Model), assigned to the DataFlyWheel by the
PI on 2026-08-22.

> ⚠️ **Read §3.7 before quoting any number from this document.** The reconstruction line has
> a long measured history, it contains several results that were later **retracted**, and at
> least one currently-banked JSON headline is **unsafe to quote**.

---

## 1. Purpose & scope

### 1.1 Why this product exists

Every corpus TanitAD trains on today is *already instrumented*: PhysicalAI-AV ships poses and
per-clip intrinsics, comma2k19 ships CAN and global pose. That is a few thousand episodes.
Action-free driving video on the open web is orders of magnitude larger and is the only
source that scales the way the programme's thesis needs
(`TanitAD Research Hub/Data Engineering/Research/YOUTUBE_DASHCAM_STRATEGY.md:8-13`).

P3 is the machine that converts the second class into the first. If it works, it is the data
moat. If it does not, the programme is bounded by what it can buy or download pre-labelled.

### 1.2 In scope

| | |
|---|---|
| **Ingest** | video acquisition, decode, shot/clip segmentation, dedup, anonymisation |
| **Calibration estimation** | intrinsics (focal, principal point, distortion) and mounting (pitch/roll/height) from the video itself — no calibration target, no dataset metadata |
| **Ego-motion recovery** | per-frame pose track from monocular video, **including metric scale** |
| **Action recovery (IDM)** | the control signal the driver applied, recovered from observation |
| **Quality gating** | per-clip admissibility: is this clip's calibration / scale / action track good enough to train on, and at what loss weight |
| **Export** | the TanitAD episode contract + a P3 provenance sidecar |
| **Provenance & licence** | per-source licence class, propagated to every derived artifact |

### 1.3 Out of scope (explicitly)

- **Training any deployed driving model.** P3 makes data; P4 trains.
- **Curating, filtering, mixing or generating datasets** from already-instrumented corpora —
  that is P2.
- **Scenario semantics, descriptions and search** — that is P5 (TanitScena).
- **The world-model encoder itself.** P3 *reads* an encoder owned by P1.

---

## 2. Boundaries

### 2.1 P3 vs P2 (Data pipelines)

**The dividing line is whether the signal has to be manufactured.**

| the source already carries… | owner |
|---|---|
| calibration + pose + actions (PhysicalAI-AV, comma2k19, nuScenes, Argoverse2, L2D) | **P2** — curate, filter, mix, generate |
| none of them (YouTube, a phone on a windscreen, a dashcam SD card) | **P3** — reconstruct, then hand to P2 |

P3's output is an *input* to P2. P3 does not do corpus mixing, curriculum design or dataset
configuration; it hands P2 episodes plus the quality/provenance record P2 needs to decide
inclusion and loss weight.

⚠️ **The seam is real, not a formality.** `stack/tanitad/data/comma2k19.py` and
`argoverse2.py` are P2 adapters — they *read* published calibration and pose. P3 must never be
implemented by editing them, because that would put an estimator behind an interface whose
contract is "this corpus is ground truth".

### 2.2 P3 vs P5 (TanitScena)

P3 emits per-clip **descriptors**; P5 indexes, describes and searches them. P3 does not own
embeddings, semantic search, or the scenario database. The handoff is one direction:
P3 → P5, never P5 → P3 at reconstruction time.

### 2.3 P3 vs P1 (models)

The IDM is a model but it is a **data-manufacturing instrument**, not a driving policy. It
never deploys to a vehicle, it is always run offline, and it is allowed to be non-causal. P1
owns the encoder the IDM reads; P3 owns the head, the label contract and the validation
ladder.

---

## 3. Current state — MEASURED from this repo

`Every row cites file:line or a JSON key path in this worktree. Evidence class marked where`
`it is not MEASURED.`

### 3.1 Capability matrix

| # | capability | status | evidence |
|---|---|---|---|
| C1 | **Episode contract** (frames / actions[T,2] / poses[T,4] / episode_id) | **EXISTS** | `stack/tanitad/data/_contract.py:11-16`, `assert_contract:59` |
| C2 | **Focal canonicalisation** given KNOWN intrinsics | **EXISTS** | `stack/tanitad/data/calib.py:9-15`, `F_REF=266.0` `:41`, `focal_crop_resize:267`, `ftheta_crop_resize:496` |
| C3 | **Intrinsics ESTIMATION from video** | **EXISTS — PARTIAL** (rectilinear only; **does not track focal**, r=0.41; fisheye ⛔) | §3.3 |
| C4 | **Extrinsics / mounting normalisation** (height, pitch, roll) | ⛔ **MISSING** | `calib.py:17-18` states extrinsics are *"NOT yet fully normalized"*; the one fix that exists uses per-clip `(cx,cy)` **read from PhysicalAI's `calibration/`** (`calib.py:19-23`) — supplied, not estimated |
| C5 | **Monocular ego-motion / visual odometry** | **PARTIAL — rotation only** | §3.4 |
| C6 | **Metric scale resolution** | ⛔ **MISSING** (and one candidate route already REFUTED) | §3.4, §4.1 |
| C7 | **IDM — action recovery from observation** | **EXISTS**, 3 of 4 channels admissible | §3.2 |
| C8 | **IDM four-metric-family instrument** | **EXISTS + RUN ONCE, but UNWIRED in `stack/`** | §3.2.4 |
| C9 | **Parity firewall for non-parity corpora** | **EXISTS** | `stack/tanitad/data/parity.py:78-80`, non-parity escape hatch in the module header |
| C10 | **Harvest: ingest, gating, anonymisation, canonicalisation** | **EXISTS — ran at 343 clips**, with measured defects | §3.3b |
| C11 | **Provenance / licence sidecar on an episode** | ⛔ **MISSING** | `_contract.py:11-16` — the contract is frames/actions/poses/episode_id only |
| C12 | **Quality gate assigning a per-clip loss weight** | ⛔ **MISSING as code** | design only (`YOUTUBE_DASHCAM_STRATEGY.md:45-48`) |

### 3.2 The IDM — what exists, measured

**Module** `stack/scripts/idm_head.py` (569 lines). **Tests** `stack/tests/test_idm_head.py`
(14), `stack/tests/test_idm_families.py`. **Orchestration** `stack/scripts/run_idm_proof.py`,
`run_idm_ft.py`, `run_idm_parity_validation.py`, `run_idm_downstream_ablation.py`,
`run_idm_pipeline_derisk.py`.

**What it is** (`idm_head.py:1-24`): a small **bidirectional (non-causal)** temporal
transformer over a window of *frozen* encoder latents `z_{t-k..t+k}` (k=4 → 9 frames), reading
out **at the window centre**. Defaults `state_dim=2048, d_model=256, depth=3, n_heads=4,
window=9` (`:216-219`); **2,899,724 params**. Outputs 4 continuous scalars
`(speed, yaw_rate, steer, long_accel)` (`SCALAR_NAMES:37`) **and** a 2 s ego-frame trajectory
at horizons `{5,10,15,20}` steps (`:38`, `DT=0.1` `:39`). Loss: Huber on standardised scalars
+ smooth-L1 on the trajectory scaled by `traj_scale=10.0` m (`:299-315`) — the scale exists
because without it *"the speed/yaw/steer heads never train"* (`:306-308`).

**Design invariant, already in code** (`:12-15`): *"The encoder is FROZEN and PURELY VISUAL:
`encode_window` uses only `encoder` + `readout` and takes NO action/speed channel."* This is
the leak guard — see §5.1.

#### 3.2.1 Absolute per-channel accuracy — the shipped head

`…/Benchmarks & Eval/Implementation/incoming/2026-07-27-fleet-sync-idm-steer/raw/idm5_ensemble.json`,
key `ensemble_3seed.channels.<ch>.r2`. Rung 757 (141,628 train windows), 3 seeds,
**n = 4,195 windows / 36 episodes**, repaired labels, `leak_check.residual_content_overlap = 0`.

| channel | pooled R² | PhysicalAI | comma2k19 | MAE |
|---|---:|---:|---:|---:|
| `speed` | **+0.86504** | +0.93117 | +0.74527 | 3.2231 m/s |
| `yaw_rate` | **+0.91879** | +0.96238 | +0.69482 | 0.016971 rad/s |
| `steer` | **+0.79926** | +0.78576 | +0.80713 | 0.008647 rad |
| **`long_accel`** | ⛔ **−0.05906** | −0.03691 | −0.22580 | 0.43781 m/s² |

⚠️ **Two quoting traps in this one file.**
1. `idm4_steer.json.rungs.757.seed_mean.steer.r2 = 0.79926` is a **mean of PREDICTIONS**
   (ensemble), not a mean of per-seed metrics; the arithmetic mean of the three per-seed steer
   R² is **0.78716**.
2. `seed0_only.note` states verbatim: *"what the currently staged `idm_head_v4_steer.pt`
   actually scores — the ensemble number may NOT be quoted for it."* ⇒ **the staged single
   checkpoint is not the ensemble.**

#### 3.2.2 Separation from a shuffled-latent control — a *different* quantity

`…/2026-08-03-idm-derived-accel/results_idm_derived_accel.json`. 50 content-clean comma2k19
episodes, **episode-disjoint 33/17**, 3 seeds, 100 epochs, **paired episode-cluster bootstrap
B=2000**.

| channel | ΔR² vs shuffled-latent control | separated? |
|---|---|---|
| `speed` | **+0.7187** | ✅ |
| `yaw_rate` | **+0.2252** | ✅ |
| `steer` | **+0.34** (the detected reference) | ✅ |
| `long_accel` | **−0.0984 [−0.3087, +0.0179]** | ⛔ **spans zero** |

⚠️ **§3.2.1 and §3.2.2 are NOT the same measurement** and must never be blended: the first is
absolute R² on 36 episodes at rung 757; the second is a paired delta against a control on 17
held-out comma2k19 episodes. Quote each with its protocol.

#### 3.2.3 `long_accel` — the settled negative, and its *two* causes

`…/2026-08-03-idm-accel-recoverability/results_accel_recoverability.json`
(`split.unit = "EPISODE-DISJOINT"`, 33/17 eps, 4,554/2,346 windows, 3 seeds, B=2000).
**30 arms = 17 real + 13 `__CTRL`.**

- **Empirical null**: `arms.NULL_train_mean.r2.long_accel.point = −0.0626 [−0.213, −0.002]`.
  Predicting the training mean scores **−0.063, not 0**, on held-out episodes — the honest floor.
- **Best real arm on frozen latents**: `NN_transformer_d512_L6` = **−0.1076 [−0.249, −0.015]**
  — *worse than the null*. **All 17 arms ≤ −0.0529.** Every ridge arm lands exactly on the
  null: the regulariser drove the head to a constant.
- **Oracle / capacity control** (`raw/oracle_input_capacity.json`, input = true CAN speed at
  the 9 window positions): `ORACLEIN_ridge_linear_window.r2.long_accel.point = **+0.92623**
  [0.88764, 0.95074]` — closed-form, 9 features. ⇒ **the head, regulariser and protocol are
  not the limit.**
- **Longer context does not help**: k=12 (2.5 s, 51,200 features) → −0.06647.
- **Deriving accel from the predicted speed track is REFUTED, not merely unhelpful**:
  ΔR² **−0.25298 [−0.48314, −0.09967] separated-worse** (−0.20614 → −0.45913), and it also
  costs `yaw_rate` (−0.11417, separated-worse).
- **Mechanism** (`raw/speed_error_mechanism.json`): the held-out speed error is
  **autocorrelated 0.9265 at 0.2 s** — differencing *cancels* ~93 % of its variance rather
  than amplifying it. The route still fails because the surviving volatility is
  **4.787 m/s² against a target whose own std is 0.568**. It is a **DYNAMIC-RANGE** problem.
  Budget: 0.1 m/s of speed noise already drops derived accel to +0.549; 0.25 m/s → −1.41.

⭐⭐ **AND A SECOND CAUSE THE INPUT-SIDE STORY ALONE MISSES.** `…/2026-07-26-idm-v2/labels2.json`
key `long_accel.pai`: `corr_label_vs_dv_dt = 0.4335`,
`r2_ceiling_of_best_affine_in_dv_dt = **0.18794**` — on **PhysicalAI the `long_accel` LABEL
itself is only 0.188-explainable by a perfect kinematic estimator.** (comma2k19: r 0.9245,
ceiling 0.8547.) ⇒ **On PhysicalAI, part of the `long_accel` failure is a bad LABEL, not a
missing signal.** Any fix must separate the two; treating it purely as an input problem would
chase a target that is partly noise.

⚠️ **Quote the floor with the claim** (`idm_head.py:145-151`): the null is bounded by power.
On a random carrier at SNR ≈ 7 the protocol detects a planted true R² of 0.3 and **misses
0.1**. What is excluded is that `long_accel` is carried *as strongly as `steer` is* — **not**
that the latents contain nothing.

#### 3.2.4 The four metric families — computed once, and STRATEGIC is absent

`…/Benchmarks & Eval/Implementation/incoming/2026-08-03-idm-four-families/results_idm_four_families.json`.
Head `idm_head_v4_steer_ens3`, rung 757, 3 seeds. `meta.split` = comma2k19 only
(`comma2k19-val-61c46fca8f7f`, `ep_00040..ep_00089`), **n = 6,900 windows / 50 episodes**.

| family | status | values |
|---|---|---|
| **LONGITUDINAL** | computed; one sub-metric absent | `traj_speed_mae` 0.3135 m/s · `along_mae` 0.3499 m · `along_final_bias` 0.042 m · `accel_mae` 0.2488 m/s² · ⛔ `distance_keeping.status = "UNAVAILABLE"` (**n=0**, no lead-agent track) |
| **LATERAL** | computed in full | `heading_mae` 0.01756 rad (n 25,204) · `curvature_mae` 0.0065 1/m (n 18,842) · `yaw_rate_mae` 0.0197 rad/s · `cross_track_mae` 0.1952 m · `cross_track_final_mae` 0.4152 m |
| **TACTICAL** | computed; goal-setting partial | lateral BA **0.7722** [0.7216, 0.8553] (chance .3333) · longitudinal BA **0.68654** [0.6123, 0.7377] · mixed 5-class BA 0.6653 (chance .2) · `goal_setting.status = "PARTIAL"` |
| **STRATEGIC** | ⛔ **NOT COMPUTED** | `STRATEGIC.status = "UNAVAILABLE"` — **no route/goal/map label on either substrate** |

`ADE_2s_m = 0.4482 [0.4121, 0.4866]`. The file's own `_contract` says it best: ***"A family
marked UNAVAILABLE is a WORK ITEM, not a pass."***

⛔⛔ **DO NOT CITE THIS FILE'S `scalar_r2`.** It reports `long_accel = +0.9421`, contradicted by
**five** other artifacts. Three reasons from its own `meta`: `leak_provenance` ends
*"VERIFIED-AGAINST-42, UNVERIFIED-AGAINST-79"* (v4 trained on 121 comma episodes; fingerprints
exist for only 42); the `LEAK_SENSITIVITY_known_leaked_episodes` arm scores **lower** than the
"clean" split — backwards for a leak contrast; and `encode_fidelity_caveat` says latents were
re-encoded on the dev box so *"every contrast here is internal to this run."*

#### 3.2.5 Cross-domain transfer — the pre-registered gate has NEVER passed

`…/2026-07-22-idm-proof/results.json`. Gate (`meta.pass_rule`): cross speed R² > 0.9 **AND**
yaw R² > 0.9 **AND** ADE@2s < 1.5× in-domain. `go_no_go.PASS = **false**`.

| slice | n | speed | yaw_rate | steer | ADE@2s |
|---|---:|---:|---:|---:|---:|
| PhysicalAI in-corpus heldout | 7,028 | +0.92973 | +0.92442 | +0.85789 | 2.733 |
| **→ comma2k19 cross-domain** | 12,420 | **+0.65724** | **+0.00047** | +0.57433 | **6.556** (2.40×) |
| rig-A in-rig heldout | 3,957 | +0.78632 | +0.80847 | +0.76869 | 4.357 |
| **→ rig-B cross-rig** | 26,392 | ⛔ **−2.46537** | −0.10944 | +0.52349 | **17.471** (4.01×) |

Neither remedy closes it: light fine-tune pai→comma 0.4058 → 0.4111 (no recovery); multi-domain
co-train → held-out rig-B −1.6076 vs single-domain −1.6468 (Δ+0.0392).
`results_multirig.json.overall_verdict.data_diversity_hypothesis = **"REFUTED on existing
assets"**` — the collapse is *representational*, not a data-diversity shortfall.
`results_regate.json.overall_verdict.fix1_frontend = "NO-OP (already applied); rig-B collapse
NOT intrinsics-driven"` (measured f_eff 266.13 / 266.10 / 266.5 vs `F_REF` 266.0), and
`NEITHER_FIX_REACHES_GATE = true`.

`Project Steering/MODEL_REGISTRY.md` — the only quotable source for model facts — records the
own-encoder line built to fix this as **🟥 FAIL — REFUTED** (`:3093`), stating *"The
own-encoder / YouTube-IDM thesis resting on it is not supported."*

#### 3.2.6 Scaling helps three channels and not the fourth

`…/2026-07-27-fleet-refill/raw/idm4_steer.json`, `rungs.<R>.seed_mean`, repaired labels,
3 seeds, n_val 4,195:

| rung (episodes) | train windows | speed | yaw_rate | steer | long_accel |
|---:|---:|---:|---:|---:|---:|
| 68 | 15,875 | +0.809 | +0.829 | +0.417 | −0.158 |
| 200 | 37,444 | +0.833 | +0.912 | +0.726 | −0.287 |
| 400 | 74,854 | +0.855 | +0.916 | +0.750 | −0.185 |
| **757** | **141,628** | **+0.865** | **+0.919** | **+0.799** | ⛔ **−0.059** |

⇒ **11× the data buys steer +0.38 and yaw +0.09, and leaves `long_accel` negative.**
`rungs.757.paired_vs_a0`: yaw and steer separated-better; **speed Δ+0.2287 [−0.377, +0.835]
not separated and nominally worse**.

### 3.3 Calibration estimation — **EXISTS, QUALIFIED PASS, with a false-positive trap**

**Module** (real, not a stub):
`…/Architecture & Inference/Implementation/incoming/2026-07-25-geocalib/geocalib_intrinsics.py`
— per-video intrinsics with **GeoCalib** (ECCV 2024, `cvg/GeoCalib`), designed as a **pure
geometry-front-end swap**: `estimate_from_video(mp4)` → `focal_px` → the existing
`calib.focal_crop_resize` → `f_eff ≈ 266`. *"The ONLY change vs the pilot is the FOCAL
SOURCE"* (`:12-21`). Per-frame `calibrate()` (`:158-175`), median vFoV + **MAD outlier
rejection** over N frames (`:178-226`), confidence tiers, fallback to fixed HFOV above
`MAD_FALLBACK_DEG=9.0` (`:214-215`).

**Validation** — MEASURED 2026-07-25 on an A40; artifacts `VALIDATION_REPORT.md` +
`geocalib_validation_results.json`. Verdict recorded as **QUALIFIED PASS**.

| arm (model = `distorted`) | n | focal err mean | focal \|err\| **median** | vFoV err |
|---|---:|---:|---:|---:|
| comma_native (GT always 910) | 12 | −6.4 % | **6.8 %** | +3.6° |
| comma_480p (GT always 910) | 12 | −5.9 % | **7.1 %** | +3.3° |
| **comma_focalsweep (GT VARIES)** | 4 | **−26.7 %** | **25.0 %** | +12.7° |
| physicalai_native (120° fisheye) | 8 | — | — | vFoV est ≈ 37° |

⭐⭐ **THE TRAP, AND IT IS THE MOST IMPORTANT THING IN THIS SECTION.** The two passing arms sit
inside ±10 % — **but in both, the true focal is always 910.** An estimator that ignored the
image and always emitted ~850 px would score the same. The **only** arm where the true focal
*varies* lands at **25.0 % median error with Pearson r = 0.41**, and the module records that
GeoCalib *"regresses toward a ~50-55 deg vFoV prior"* (`:26-28`).

⇒ **A ±10 % pass read off the constant-focal arms is exactly the false-positive class
TANITAD_PROGRAMME.md §6.2 exists to prevent** — a headline a constant-only control would also
achieve. §8.2's E-P3-CALIB-1 is therefore scored on a **varying-focal** arm with the
constant-predictor control published alongside. To the report's credit it names this itself as
*"the main limitation"*; the failure mode is in the **re-quote**, not the report.

**Other measured limits:** systematic ~6–11 % focal under-estimate; per-frame outliers to
−34 % / +20 % ⇒ *"single-frame use is unsafe"*; **resolution-robust** (480p ≈ native, material
because the harvest decodes ≤480p); ⛔ **cannot handle true wide fisheye** (PhysicalAI 120°
f-theta reads ≈25–55°, `confidence="low"` is *"expected and correct"*); `pinhole` weights
uniformly worse (median |err| 13.9 %).

**Two further real estimators exist:**
- `stack/scripts/pod_ops/horizon_probe.py:51-83` `vp_row()` — road vanishing point via pure
  numpy Hough + pairwise intersection.
- `…/2026-07-22-youtube-idm-pipeline/ftheta_frontend_prototype.py` → `ftheta_frontend_result.json:116-127`
  — two-orthogonal-VP focal round-trip, **910.0 recovered, 0.00 % error** (analytic).

⛔ **`geocalib_shim.py` (in `…/2026-07-25-youtube-idm-scaleup/`) is a STUB** and says so at
`:24`. 59 lines: `poll()` = `os.path.exists`, `focal_to_hfov()` one-liner, `validate()` =
an hfov-in-[40,140] check. **No estimation.**

**Status by capability:**

| capability | status |
|---|---|
| intrinsics (focal), rectilinear | **EXISTS — PARTIAL**; usable with robust aggregation + fallback, but **does not track focal** (r=0.41) |
| intrinsics, fisheye | ⛔ **MISSING** — outside the method's competence by design |
| principal point / distortion | **UNVERIFIED** — GeoCalib returns `.c` and uses `distorted` weights, but no recovery number for either appears in the validation table. Probe: read `geocalib_validation_results.json` for a `c`/distortion error field |
| **extrinsics (pitch / roll / height)** | ⛔ **MISSING**. `calib.py:17-18`; **height is not observable from a single image at all** without a metric anchor (§4.1) |

### 3.3b Harvest and ingest — **EXISTS, ran at 343 clips, with measured defects**

**Pipeline** (`…/2026-07-25-youtube-idm-scaleup/harvest_scaleup.py`): ytsearch/channel
discovery → **GATE1** licence (`:309-316`) → **GATE2** `bad_title` (`:318-321`) → **GATE3**
duration 60–1500 s (`:322-325`) → yt-dlp → **full-res Haar face/plate/body blur** → GeoCalib
per-video focal → `focal_crop_resize` to `f_eff≈266` → 250-frame clips → **GATE4** drop if
`shotcut_score > 9.0` (`:386-388`) → `stack_frames(3)` → frozen encoder → 2048-d latents.
**Raw mp4 deleted; pointers + latents only.**

| run | tried | accepted | clips | rejects | artifact |
|---|---:|---:|---:|---|---|
| pilot 07-24 | 63 | 31 | **80** | bad_title 3, duration 15, dl_fail 2 | `…/2026-07-24-youtube-idm-pilot/pod_artifacts/manifest.json:4-14` |
| scaleup 07-26 | 24 | — | **200** | (stale round-4 snapshot) | `…/2026-07-26-idm-youtube-db-retry/pod_artifacts/harvest_manifest.json:42-47` |
| **final 07-27** | — | — | **343** | `merged_latents: 343` | `…/2026-07-27-yt-dB-retry/pod_artifacts/harvest_manifest.json:42-48` |
| geocalib 07-28 | 28 | 3 | **20** | duration 25, not_cc_kept 27 | `…/2026-07-28-youtube-geocalib-idm/harvest_manifest.json:3-15` |

`yield_verification.json:15-16`: `"yield_vs_target": "200/400"`, `"yield_fraction": 0.5`.

⭐ **The load-bearing correction — the corpus is far smaller than 343.**
`corpus_census_n343.json:2-4`: **343 clips from only 55 videos / 43 channels**;
`kish_neff: 35.298` (`:62`), `design_effect: 9.717` (`:63`) ⇒ **≈35 independent units.**
Any interval computed with the **clip** as the resampling unit is ~3× too narrow. See §8.1.

**Four further measured defects:**
1. **Contamination 12.83 %** with **zero filter efficacy**: `:79-81`
   `contamination_frac: 0.1283`, **`caught_by_shipped: []`**. `BAD_TITLE`
   (`harvest_scaleup.py:104-106`) lists `"5x","10x","4x speed","2x speed"` but **`"3x"` is
   absent**, and there is **no game/sim filter** — BeamNG footage passed.
2. ⛔ **The 343-clip corpus was canonicalised with the FIXED 100° HFOV fallback, not GeoCalib.**
   `geocalib_shim.py:5-7` records that at scale-up launch the real GeoCalib deliverable *"had
   NOT landed"*. Measured: fixed 100° was wrong for **11 of 12** real clips; hfov median
   **66.56°** (`youtube_geocalib_measurement.json:2-33`). Consequence: **63.3 % of clips are
   `fully_canonical == false`**, median achieved `f_eff` **337.6** vs target **266**
   (`YT_DB_RETRY.md:177-190`).
3. **Harvest ended in a bot-block, not at target**: round 9, **650/650 videos refused**,
   mislabelled *"pool exhausted at 343 — proceeding"* (`YT_DB_RETRY.md:111-140`).
4. **Licence: `None` on 343/343** (`YT_DB_RETRY.md:167`) because `harvest_scaleup.py:217`
   sets `--allow-noncc default=True` — **the opposite of the licence analysis's own
   recommendation** (§6.4).

**Where the data is now — 🔴 stranded, probably gone.** 343 clip latents only at
`pod3:/workspace/tmp/yt_scaleup/latents/`, described as *"(regenerable; transient by design)"*
(`YT_DB_RETRY.md:376`); 20 clips / 2.8 GB at `pod3:/workspace/tmp/yt_geo/clips/`. **Nothing on
HF.** Locally only JSON, logs and scripts. Grepping the entire `stack/experiments/pod-rescue-20260802/`
tree for `yt_scaleup|yt_geo|youtube` returns **zero hits** ⇒ the latents were not rescued.
"Regenerable" is optimistic: re-harvest needs an egress **bot-blocked on 2026-07-26**, and
`Project Steering/LOOP_STATE.md:693-703` records the authorization as **SPENT** — *"Any further
YouTube harvesting is a NEW decision for the PI."*

### 3.4 Ego-motion and metric scale — **PARTIAL (rotation only); scale MISSING**

⚠️ **A methodological correction I must record, because it nearly produced a false absence
claim in this very document.** My first probe for VO code used the ripgrep-backed search and
returned essentially nothing. **On this Google-Drive-backed worktree, ripgrep silently SKIPS
un-hydrated files** — the result was a false negative. Re-running after hydration found real
implementations. ⇒ **On this tree, a "no matches" grep is not evidence of absence.** Same
class as the `df`/Thor/cgroup traps in CLAUDE.md: a probe that reports the wrong scope.

| capability | status | evidence |
|---|---|---|
| **rotational ego-motion, GT-free** | **EXISTS** | `…/2026-07-25-idm-youtube-validation/yt_idm_reconstruct.py:105-125` — optical-flow yaw rate + focus-of-expansion via `cv2.calcOpticalFlowFarneback` (`:125`), taken from the far field above the horizon *"where translation-induced flow vanishes and flow is pure rotation"* (`:17-19`). This is the IDM↔VO cross-check `PIPELINE_DESIGN.md:103-105` specified |
| **translational scale** | ⛔ **MISSING** | `stack/scripts/geom_sanity.py:214-298` looks like it: Farneback + FOE horizon by least squares (`:253-256`), then `_fit_scale()` (`:281-290`) recovers `f·h` from `du/dd = (r-r0)²/(f·h)`. **But `dd` comes from GT poses (`:231`)** — it solves for scale *given* metric ego-motion, i.e. **the inverse of scale recovery** |
| **full VO / SLAM / SfM** | ⛔ **MISSING** | no DROID-SLAM, COLMAP, SfM, `solvePnP`, `recoverPose`, `findEssentialMat` anywhere |
| **metric monocular depth** | ⛔ **MISSING** | no UniDepth, Metric3D, MoGe or DepthAnything implementation; proposals only (`Ressources/Deep Think Analysis/Deep Think 8.md:29,41,69`) |
| **how metric scale is obtained today** | **read from known calibration** | `stack/experiments/pod-rescue-20260802/pod3/workspace/idm3_geom.py:95-102` takes `cam_h_m` (*"sets the metric scale"*) from PhysicalAI's `calibration/`; comma2k19's **1.22 m is hard-coded and flagged "INHERITED and UNVERIFIED"** (`:73-77`) |

⛔⛔ **One candidate route is ALREADY REFUTED on our own corpus.**
`…/incoming/2026-07-27-latent-action-models/LATENT_ACTION_RESEARCH.md:364` records the
**ground-plane metric-scale route** as *"ALREADY REFUTED on our corpus"*. It must not be
re-proposed as if open — see §4.1.

### 3.5 Label hygiene already earned — and what it cost

`stack/tanitad/data/comma2k19.py` carries a measured label defect and its fix, and it is the
template for how P3 must treat every reconstructed channel.

- comma2k19 heading is `arctan2(enu_v_north, enu_v_east)` — the direction of the ENU
  **velocity** vector — and is therefore **undefined at standstill**. MEASURED on the
  64-segment val build: in the `v < 0.5 m/s` bin, **26.27 % of frames carry a physically
  impossible |yaw_rate|**, up to **15.53 rad/s (890 °/s) at 0.00–0.01 m/s** (`:48-56`).
- The remedy shipped as **typed refusals + an admissibility mask**, never a silent clamp:
  `LegacyHeadingRefused:136`, `resolve_heading_mode:141`,
  `hold_heading_through_standstill:217`, `InadmissibleYawLabel:303`, `admissible_from_poses:308`,
  `yaw_rate_from_heading:339`, `assert_yaw_rate_admissible:385`.

⭐⭐ **What that defect cost, measured: an entire model generation's yaw channel.** Same recipe,
matched seeds — `arms_v3.R0LEG` (legacy labels) yaw **+0.0877 ± .003** vs `arms_v3.R0`
(repaired) yaw **+0.8292 ± .004** (`…/2026-07-27-idm-v3/results/arms_v3.json`). **All of v2's
yaw sits at ≈0.08–0.10 for this reason alone.** Nothing was retrained; only the label changed.

⚠️ But the repair is not free and not complete: on the comma rescore
(`…/2026-07-27-heading-default/raw/idm_head_v1_comma_rescore.json`, n 4,140 / 30 eps)
yaw R² OFF +0.000048 → ON −0.000421, **Δ not separated**; MAE does improve
(0.2288 → 0.1527, separated) but nMedAE is **10.1 % worse**, and
`residual_defect.n_impossible_after_repair = **84**`, max |label| **15.275 rad/s**.

⇒ **P3's rule: every reconstructed channel ships with an admissibility mask and a typed
refusal, never a clamped value.** A reconstructed channel is *more* prone to this class than a
CAN channel, not less.

### 3.6 ⚠️ Leakage — the failure that has already inverted a headline here

`…/2026-07-25-idm-youtube-validation/idm_head_v1_card.json`, key
`anchor_settlement_2026_07_27` (retraction class **C43**): **2 of 22** comma val episodes were
**bit-identical** to 2 of the head's own training clips — verified by sha256 of the raw poses
*and* `frames_u8`.

| slice | comma yaw R² |
|---|---:|
| ALL22 (as originally published) | **+0.330822** |
| **CLEAN20, content-disjoint** | ⛔ **−0.745999 [−1.574, −0.177]** |
| the 2 leaked episodes alone | +0.856185 |

⇒ **Two leaked episodes out of twenty-two moved a headline from −0.75 to +0.33 and the claim
was WITHDRAWN.** For P3 this is not a cautionary tale, it is a design requirement:
**content-hash disjointness (frames *and* poses), not id disjointness**, is the only
acceptable split check — and reconstructed corpora, where the same source video can yield many
near-duplicate clips, are structurally more exposed than curated ones.

### 3.7 ⚠️ Retracted and unsafe-to-quote results (read before citing anything)

| item | status |
|---|---|
| `…/2026-07-26-idm-v2/accel_budget.json` | **`_RETRACTED` — "FALSIFIED BY DIRECT MEASUREMENT".** Predicted derived-accel ceilings +0.657 / +0.898; measured **−8.81 / −4.13**. Root cause class **C3, "mechanism instead of measurement"**. Kept unmodified *"as the record of a wrong prediction that was caught by measuring it."* |
| v3 comma yaw anchor **+0.3308** | **WITHDRAWN** (C43 leak) — §3.6 |
| `results_idm_four_families.json.…scalar_r2.long_accel = +0.9421` | ⛔ **NOT SAFE TO QUOTE** — §3.2.4 |
| `idm4_steer.json.rungs.*.seed_mean` | ensemble-of-predictions, **not** a mean of per-seed metrics |
| IDM v2 | scored **4 of 4 pre-registered endpoints FAIL** vs A0; the doc self-reports that 2 of its own 4 bars were mis-set (class C10) |
| v3 geometry conditioning | **REFUTED** — real geometry `G1n` *"SIGNIFICANTLY WORSE than no conditioning"*; and `G1n_vs_Cshufn` (real vs **shuffled** geometry) is **indistinguishable**, −0.1479 [−0.4830, +0.1821] |
| v3 steer | **worse than its predecessor**: 0.408 vs 0.742 |
| "channel ordering matches the physics" | **INVERTED** — on every matched slice A0's speed beats its yaw (`…/2026-08-02-parallel-streams-verified/VERIFIED_idm_validation.md`, an adversarial re-derivation: 47 claims, 32 confirmed exact, **15 refutations**) |

### 3.8 Prior work inventory — and where it lives

⚠️ **Only `idm_head.py`, `idm_families.py` and the `run_idm_*.py` orchestrators are in
`stack/`.** Everything else lives under `TanitAD Research Hub/…/Implementation/incoming/`
across **13 work-package directories** — the programme's measured stranding pattern
(AGENT_OPERATING_STANDARD rule 3). Full list in BACKLOG **P3-1**.

---

## 4. Architecture — the reconstruction chain

```
 ①  video ingest            ②  shot / clip segmentation      ③  calibration estimation
    URL → frames               scene cuts, driving detect,       intrinsics (f, cx, cy, dist)
    + licence class            dedup, anonymisation              + mounting (pitch, roll, h)
        │                            │                                   │
        └────────────────────────────┴───────────────────────────────────┘
                                     ▼
 ④  canonicalisation  ──►  ⑤  ego-motion recovery  ──►  ⑥  metric scale resolution
    calib.py, F_REF=266        relative pose track           the ambiguity — §4.1
        │                            │                                   │
        └────────────────────────────┴───────────────────────────────────┘
                                     ▼
 ⑦  action recovery (IDM)  ──►  ⑧  quality gating  ──►  ⑨  export
    idm_head.py                    per-clip admissibility     episode contract
                                   + loss weight              + P3 provenance sidecar
```

| # | stage | module | status |
|---|---|---|---|
| ① | video ingest | `…/2026-07-25-youtube-idm-scaleup/harvest_scaleup.py` (+ pilot `harvest.py`, `yt_pilot_common.py`) | **EXISTS** — ran to 343 clips; ⛔ **licence gate mis-set** (§3.3b) |
| ② | segmentation / dedup / anonymisation | same module: `shotcut_score` GATE4 `:386-388`, Haar face/plate/body blur, `bad_title` `:104-106` | **PARTIAL** — contamination **12.83 %** with **zero** filter efficacy (§3.3b) |
| ③ | calibration estimation | `…/2026-07-25-geocalib/geocalib_intrinsics.py`; `pod_ops/horizon_probe.py:51-83`; `ftheta_frontend_prototype.py` | **PARTIAL** — rectilinear focal only, **r=0.41**; ⛔ fisheye out of competence; ⛔ **extrinsics MISSING** |
| ④ | canonicalisation | `stack/tanitad/data/calib.py` — `focal_crop_resize:267`, `ftheta_crop_resize:496`, `pinhole_rectify:796`, `cylindrical_rectify:960` | **EXISTS** (consumes calibration, does not produce it) |
| ⑤ | ego-motion recovery | `yt_idm_reconstruct.py:105-125` (flow yaw + FOE) | **PARTIAL — rotation only** |
| ⑥ | metric scale resolution | — | ⛔ **MISSING**; one route already **REFUTED** (§3.4, §4.1) |
| ⑦ | action recovery (IDM) | `stack/scripts/idm_head.py` | **EXISTS** — `speed`/`yaw_rate`/`steer` admissible, ⛔ `long_accel` at the null |
| ⑧ | quality gating | — | ⛔ **MISSING as code**; design only (`YOUTUBE_DASHCAM_STRATEGY.md:45-48`) |
| ⑨ | export + sidecar | `stack/tanitad/data/_contract.py:80` `assemble_episode` | **PARTIAL** — contract yes, **sidecar MISSING** |

### 4.1 ⚠️ The scale-ambiguity problem — stated explicitly

**The physics.** Monocular ego-motion is scale-ambiguous. From images alone one recovers the
*direction* of translation and the full rotation, but the *magnitude* of translation is free:
the scene can be scaled by any λ > 0 with identical pixels. Every metric quantity P3 must emit
— speed, waypoints in metres, longitudinal accel, headway — is downstream of that λ.

**Why this is the highest-risk item in P3, specifically for TanitAD.**

- The deployed flagship's longitudinal family is a **systematic over-speed prior, not a
  tail**: `speed_bias` **+0.484 m/s**, `along_final_bias` **+0.943 m**, `ego_progress`
  **1.0795×**, **71.95 %** of windows ahead at 2 s, **75.51 %** faster than the human
  (`Project Steering/MODEL_REGISTRY.md:1148`). The registry localises the fault as *"the
  PLANNER (speed/selection)"* (`:867`).
- **88.7 % of the programme's oracle gap is longitudinal** (CLAUDE.md, the four-families
  rule's stated rationale).
- The IDM's own `long_accel` is at/below its empirical null, and the budget to fix it by
  differentiating a speed track is **~47×** (§3.2.3).
- Today, **metric scale is not estimated at all** — it is read from known calibration, and on
  comma2k19 the camera height is a **hard-coded 1.22 m flagged "INHERITED and UNVERIFIED"**
  (§3.4).

⇒ **A monocular pipeline that resolves scale badly reproduces the programme's worst known
failure mode, at scale, inside the training data itself.** This is not a tuning risk; it is the
risk that P3 manufactures a corpus that actively teaches the wrong speeds.

**Three couplings that must not be forgotten:**

1. **Calibration error propagates into scale.** An error in focal biases recovered translation
   magnitude, so scale accuracy is bounded by ③ — and ③ currently **does not track focal**
   (r=0.41). Any scale claim must state the calibration error it assumed.
2. **Scale error is multiplicative ⇒ measure it in log space.** Report `log(λ̂/λ)`; a mean
   absolute *metre* error hides that ×2 and ×0.5 are equally wrong.
3. **Scale is per-clip; drift is per-frame.** A single global λ per clip is a much easier
   object than a λ that drifts along the track. Report **both**.

⭐ **A genuine design tension P3 must own, not paper over.** `Deep Think 8.md:35` argues for
predicting **path curvature `κ = ω/v`** precisely *because* yaw rate and velocity scale
identically with depth, so **their ratio cancels the scale ambiguity**. That is correct — and
it collides head-on with the operative layer's MEASURED finding that κ is the **ill-conditioned**
target (kurtosis 38.9, a 9.5× low/high-speed tail ratio, explodes as v→0), which is exactly why
`UnicycleStepReadout` emits **yaw rate** instead (§5.2). ⇒ **The scale-invariant quantity and
the learnable quantity are not the same quantity.** P3 resolves this by keeping them separate:
learn `yaw_rate` (well-conditioned), report κ only as a *derived diagnostic*, and treat λ as an
**explicit, separately-estimated factor** rather than something to cancel away.

**How P3 proposes to resolve metric scale** — a **ranked ladder with an explicit fallback**,
because every single mechanism has a known failure class:

| rank | mechanism | metric anchor | fails when | status |
|---|---|---|---|---|
| **S-a** | in-frame **speedometer OCR** | the vehicle's own instrument | no speedometer visible; km/h vs mph ambiguity; blur/occlusion | HYPOTHESIS — named repeatedly in prior design (`IDM_VIDEO_PRETRAIN_DESIGN.md:120`), **never implemented** |
| **S-b** | ~~ground-plane homography from camera height~~ | ~~mount height h~~ | — | ⛔ **REFUTED on our corpus** (`LATENT_ACTION_RESEARCH.md:364`). **Do not re-propose.** |
| **S-c** | **lane-marking pitch** as a ruler | standardised dash/gap length per jurisdiction | jurisdiction unknown; markings worn/absent | HYPOTHESIS — not implemented |
| **S-d** | learned **metric monocular depth** (UniDepthV2, Metric3D-v2) | a published metric-depth prior | domain shift; the prior's own scale bias unmeasured on our frames | HYPOTHESIS — proposed in `…/2026-07-22-own-dynamics-encoder/DESIGN.md:191`, not implemented |
| **S-e** | **supervised in-domain grounding** | metric ego-pose learned from labelled driving logs | needs labelled in-domain data — which is what P3 lacks for new sources | **PUBLISHED support**: Cosmos-3 reports ATE **0.98 m** vs VGGT/DepthAnything3 **23.46 / 9.29 m** *"precisely because scale comes from supervised in-domain driving logs; general-domain VO drifts on absolute scale"* (`DESIGN.md:64`) |
| **S-f** | video metadata / telemetry (GPS, GoPro GPMF, phone IMU) | direct | almost never present on re-hosted video | HYPOTHESIS |

⚠️ **S-a, S-c, S-d, S-f are HYPOTHESIS class — none implemented, none measured on our frames.
They must not be quoted as capabilities.** S-b is **refuted**. S-e is the only route with
published in-domain support, and it is also the one that most constrains P3's reach.

**The design commitment that makes the ladder safe** — more important than which mechanism wins:

> **P3 predicts a SCALE-NORMALISED trajectory and a SEPARATE, EXPLICIT scale factor λ, each
> with its own confidence. They are never fused into one metric output inside the model.**

Then scale error is *attributable*; it can be gated independently; a clip with good shape and
bad λ is still usable (video-only, or at reduced longitudinal weight); and a better λ estimator
later is a drop-in that does not require retraining the shape path.

### 4.2 Pre-registered validation of scale — both outcomes committed

**E-P3-SCALE-1. The spoof test** — the falsifier already named in
`YOUTUBE_DASHCAM_STRATEGY.md:28-30`, generalised from focal to scale.

- **Substrate**: held-out comma2k19 **route-disjoint** segments (`split_by_route`,
  `comma2k19.py:480`) — CAN speed gives ground-truth metric scale for free — plus PhysicalAI-AV
  held-out clips as the second domain.
- **Blinding**: strip CAN, strip published intrinsics, strip pose. Feed the pipeline decoded
  frames only, exactly as a YouTube clip arrives. The corpus is an **oracle**, never an input.
- **Splits**: route-disjoint / episode-disjoint, and ⛔ **content-hash disjoint** (frames *and*
  poses) per §3.6's C43 lesson — never id-only.
- **Estimator**: **paired episode-cluster bootstrap** (`taniteval/ci.py`), B=2000.
  ⛔ Never `overlapping_holdout_se`. ⛔ And on reconstructed corpora the cluster unit is the
  **source video**, not the clip (§3.3b: design effect **9.717**).
- **Primary metric**: distribution of per-clip **`log(λ̂/λ_true)`** — median, IQR, fraction
  within ±10 %.
- **Secondary**: within-clip scale drift; recovered speed MAE and R²; and the
  **calibration-conditioned** version, so the ③→⑥ coupling is visible.

**Controls that must run alongside** (programme §6.2):

- **Constant-only control** — a single λ (the training median) for every clip. The
  no-information value the estimator must beat. *(The IDM stream's `NULL_train_mean` is the
  model to copy: it reads **−0.0626, not 0**, on held-out episodes.)*
- **Raw-input floor** — λ from a trivial signal (mean optical-flow magnitude → linear fit). If
  the pipeline does not beat this, it is decoration.
- **Deliberate-regression arm** — inject a known ×1.5 scale bias and confirm the gate **turns
  red**.

**Committed outcomes:**

- ✅ **SUPPORTED** if median `|log(λ̂/λ)| ≤ log(1.10)` **on both domains**, both controls are
  separated-worse, and the regression arm goes red ⇒ metric scale is good enough to mint
  longitudinal labels; proceed to E-P3-DOWNSTREAM-1.
- ❌ **REFUTED** if either domain misses the bound or a control is not separated ⇒ **P3 does not
  mint metric longitudinal labels.** The product falls back to the **scale-free deliverable**:
  canonicalised frames + *shape-only* (scale-normalised) trajectories + lateral actions,
  entering training as representation learning and lateral supervision only, with the
  longitudinal channel **explicitly masked out, not down-weighted.**

⚠️ **The asymmetry is deliberate: REFUTED does not kill P3, it removes one channel.** Written
down now so a disappointing result is not argued away later.

---

## 5. The IDM sub-spec

### 5.1 Inputs — vision-only, by necessity as well as by rule

| stage | what may be read |
|---|---|
| **label derivation on instrumented corpora** (training the IDM) | CAN, poses, future frames, published calibration — anything. Labels are built offline; privileged signals are FINE (CLAUDE.md binding, PI 2026-08-03) |
| **IDM inference** (labelling uninstrumented video) | ⛔ **the video only.** Non-causal windows (past **and** future) are permitted — the IDM is always offline. Auxiliaries *derived from the video itself* (speedometer OCR, overlays) are permitted and are not a leak, because they exist in the target domain too |
| **the downstream driving model's inference** | ⛔ **VISION ONLY**, unchanged. P3 changes nothing here |

⚠️ **The leak test, applied to the IDM.** *"Ask whether an input at inference contains
something the label was derived from."* The IDM's labels **are** CAN speed and steering, so
feeding it ego speed would make its speed R² a leak, not a capability. The code already forbids
it (`idm_head.py:12-15`) — but that is a **comment, not a guard**, and needs a regression test
(BACKLOG **P3-6**).

There is also a hard practical reason it can never be relaxed: **on YouTube video there is no
ego channel.** An IDM that needs one cannot do the job it exists for.

### 5.2 Outputs — and the gap to the operative action parameterisation

**What the IDM emits today** (`idm_head.py:37-38`): `(speed, yaw_rate, steer, long_accel)` at
the window centre, plus 4 ego-frame waypoints at `{5,10,15,20}` steps (**0.5 s** spacing).

**What the operative layer consumes — MEASURED from source, and NOT what "`(a, κ)`" suggests.**
`stack/tanitad/models/metric_dynamics.py:472` `class UnicycleStepReadout`, documented at `:473`
as *"Latent transition -> per-step `(accel, yaw_rate)`, integrated as a unicycle"*, with
`unicycle_step_dpose(controls, v0, dt=0.1)` at `:624` ⇒ **`dt = 0.1` s = 10 Hz**.

⭐ **The controls are `(accel, yaw_rate)`, deliberately NOT `(accel, curvature)`**, and the
programme MEASURED why (39 OOD-val clips × 20 steps, 2026-08-06,
`…/2026-08-06-v1-defect-triage/OPTION2_UNICYCLE_READOUT.md:131-175`; `metric_dynamics.py:488`):

| target | std | kurtosis | \|·\| p99 v<3 | \|·\| p99 v>8 | low/high tail ratio |
|---|---:|---:|---:|---:|---:|
| accel (m/s²) | 0.80438 | — | — | — | — |
| **curvature (1/m)** | 0.02091 | **38.9** | 0.1748 | 0.0184 | **9.5×** |
| **yaw rate (rad/s)** | 0.06930 | 10.4 | 0.3452 | 0.1521 | **2.3×** |

*"`kappa = yaw_rate / v` **explodes as v → 0**"* — **the division is the defect, not the
interface.** Two design facts travel with it: each channel is emitted **in units of its own
target std** (a **38.5×** imbalance otherwise), and yaw rate is **clamped to `±|v|·kappa_max`**
so turn-in-place stays unrepresentable at `v = 0` (pinned by
`test_yaw_rate_parameterisation_still_cannot_turn_in_place`).

**The channel-by-channel gap — the crux of the product:**

| operative control | can the IDM mint it? | evidence |
|---|---|---|
| **`yaw_rate`** — the lateral control *as the operative layer actually wants it* | ⭐ **YES — a DIRECT channel match, no division required** | emitted natively (`idm_head.py:37`); absolute R² **+0.91879** pooled (§3.2.1); separated-positive ΔR² **+0.2252** vs shuffled-latent (§3.2.2) |
| **`accel`** — the longitudinal control | ⛔ **NO — not at present** | absolute R² **−0.05906** (§3.2.1); ΔR² **−0.0984 [−0.3087, +0.0179]** spans zero; 17-arm sweep ≤ the **−0.0626** null; derivation **separated-worse** (§3.2.3) |
| `speed` — an **input** to the readout | **YES** | absolute R² **+0.86504**; ΔR² **+0.7187** |

⚠️ **Correcting the brief's framing on the record.** This work package was briefed as *"the
operative layer uses continuous unicycle `(a, κ)` controls @10 Hz"*. The **rate is right**
(`dt=0.1`), but the second channel is **yaw rate, not curvature** — and a P3 exporter built to
emit κ would divide by `v`, reintroducing by hand the exact ill-conditioning the operative layer
was redesigned to remove. ⇒ **P3 exports `(accel, yaw_rate)` + `speed`, never κ.**

⭐ **Stated plainly: P3 can reconstruct the LATERAL control and cannot reconstruct the
LONGITUDINAL one — 1 of the 2 operative control channels.** That is the opposite of convenient,
since 88.7 % of the oracle gap is longitudinal, and it must not be softened. Three admissible
responses, in priority order:

1. **Change the input, not the head.** The capacity control settles that the head, regulariser
   and protocol are not the limit (**+0.92623** from the *true* speed window vs ≤ null from
   latents). Candidate levers: higher temporal resolution than the 9-frame window; an encoder
   whose readout is not geometry-bottlenecked; features that are *differential* by construction.
   ⚠️ **But first separate the two causes** — on PhysicalAI the **label** caps a perfect
   kinematic estimator at **R² 0.188** (§3.2.3). Fixing the input cannot beat a broken target.
2. **Change the target.** Instantaneous `accel` may be the wrong ask. A **target-speed profile**
   over the horizon, or **headway/time-gap** to the lead agent, is what the longitudinal family
   actually scores — and note that family's `distance_keeping` is currently **UNAVAILABLE at
   n=0** for want of a lead-agent track (§3.2.4), so this route needs that instrument first.
3. **Ship `yaw_rate`, mask `accel`.** Emit the lateral control with its admissibility mask and
   mark the longitudinal channel inadmissible. Honest, immediately available, compatible with
   §4.2's REFUTED fallback.

⛔ **NOT admissible**: shipping `accel` anyway at low loss weight and hoping. The measured value
is *at the null*; a null channel at low weight is noise injected into the one axis the programme
is already worst at.

⚠️ **CADENCE — a 5× error already documented as having bitten.** IDM: **4** waypoints, 0.5 s
apart, over 2 s (`idm_head.py:38`, `idm_families.py:57-58`). Operative: **10 Hz**
(`metric_dynamics.py:624`) ⇒ **20** waypoints over the same 2 s. `taniteval.four_families`
hard-codes `DT_S = 0.1`, so *"feeding IDM trajectories to that module unchanged reads every
speed and yaw-rate 5× too large"* (`idm_families.py:30-38`).

⇒ **Every P3 → P4 export carries `cadence_hz` explicitly and is tested against it.** The pattern
exists: `idm_families.geometry()` takes cadence as a parameter and `tests/test_idm_families.py`
pins it against `four_families._seq_geometry` at `dt=0.1`. ⛔ Resampling 0.5 s → 10 Hz is **not**
free interpolation — it invents 4 intermediate steps per emitted one, and whatever it invents
becomes training signal. Either the IDM emits at 10 Hz natively, or the export declares the
upsampling method **and its measured error**. **Neither is done today.**

### 5.3 Training supervision

- **Corpora**: comma2k19 (CAN speed + steering-wheel angle + global pose, `comma2k19.py:5-19`)
  and PhysicalAI-AV.
- **Targets**: from the episode contract — `poses[T,4] = (x,y,yaw,v)`,
  `actions[T,2] = (steer_road_rad, accel_mps2)` (`idm_head.py:17-19`). `yaw_rate` is a
  **centred** finite difference of wrapped heading (`scalar_targets_at:60-71`) — non-causal,
  matching the labeller's regime.
- **Standardiser fit on TRAIN only** — already enforced (`train_head:483-496`).
- **Label admissibility mandatory** (§3.5): standstill yaw is inadmissible, not clamped.
- ⚠️ **`STEER_RATIO = 15.3` is a constant and an approximation** — `comma2k19.py:46`, annotated
  *"steering wheel -> road wheel, v0 constant"* (Civic/Corolla-class). Every steer claim
  inherits it; state it.
- ⚠️ **PhysicalAI's `long_accel` label is measurably poor** (kinematic ceiling R² **0.188**).
  Do not train or score that channel on PhysicalAI without saying so.

### 5.4 The validation ladder — every rung mandatory

A number is admissible only if **every** rung below it ran on the same windows.

| rung | what it is | status in the IDM stream |
|---|---|---|
| **L0 — constant-only control** | predict the train mean of each channel | ✅ **EXISTS** — `arms.NULL_train_mean` (**−0.0626**, not 0), `NEG2_blind_mean_predictor`; v3 `Ccorp`/`Crign` corpus/rig one-hots |
| **L1 — raw-input floor** | ridge on a trivial raw-pixel feature, **no learned encoder** | ⛔ **MISSING.** The `ORACLEIN_*` arms are an *oracle ceiling* (true CAN speed in), **not** a raw-pixel floor. Distinct thing, still absent |
| **L2 — shuffled-latent control** | matched protocol, latent↔target link destroyed | ✅ **EXISTS** — 13 `__CTRL` arms; `NEG_shuffled_latents`; `NEG1_latents_shuffled_across_windows` |
| **L3 — guard shown ABLE TO FAIL** | inject a known defect, confirm the gate goes red | ✅ **EXISTS, and impressively so** — see below |
| **L4 — episode/route-disjoint generalisation** | ≥3 seeds, paired episode-cluster bootstrap B=2000 | ✅ **EXISTS** (33/17 episode-disjoint; 36-episode val) |
| **L5 — cross-domain transfer** | train one domain, score the other | ⛔ **FAILS** — `go_no_go.PASS = false`, never passed (§3.2.5) |
| **L6 — four metric families** | per-family, with its own `n` | 🟡 **RUN ONCE** (§3.2.4) but ⛔ **STRATEGIC UNAVAILABLE**, `distance_keeping` n=0, and **no `stack/` script imports the instrument** |
| **L7 — downstream utility** | does data labelled by this IDM improve a fixed model? | 🟡 **RUN, POSITIVE, CONTESTED** — §8.1 |

⭐ **L3 deserves emphasis, because it is what makes the `long_accel` null credible.** Four guards
have been demonstrated able to move:

| guard | demonstrated sensitivity |
|---|---|
| **planted-signal detector** | injecting a rank-1 copy of `long_accel` at **1e-5 of latent RMS** takes held-out R² from −0.06259 to **+0.94335 [0.90475, 0.96046]**; 3e-5 → 0.99231; 1e-4 → 0.99818 |
| **lateral-family sign check** | `NEG3_gt_with_lateral_sign_flipped` feeds *ground truth* with the lateral sign flipped: **scalar R² = 1.0 by construction**, yet lateral BA drops to **0.3333 (chance)** and ADE 0.5538 vs 0.4482 (Δ−0.1055 [−0.2067, −0.0124] separated). ⭐ **The lateral family catches an error the scalars structurally cannot** — the single best argument for L6 |
| **shuffled-geometry** | `Cshufn_vs_R0.speed` ΔMAE +0.7843 [+0.1798, +1.4316] separated-worse |
| **shuffled-latent** | `NEG1` ADE **16.3159** vs shipped **0.4482**; lateral BA 0.3297 ≈ chance |

⇒ **`long_accel` is *unrecoverable* from these latents, not merely unrecovered** — a probe that
detects a planted signal at 1e-5 of latent RMS, and reaches +0.926 given true speed, reports
≤ null on the real thing.

⚠️ **Correction to an earlier draft of this SPEC.** I initially wrote that L0 and L3 were "not
evidenced anywhere I probed". **That was wrong** — both exist, and L3 is unusually strong. The
error came from probing `stack/tests/` only and not the `incoming/` work packages. Recorded here
as an instance of CLAUDE.md's *"absence found at ONE location is not absence"*. **The genuine
gaps are L1, L5, and L6's two missing families.**

---

## 6. Interfaces

### 6.1 P3 → P2

Per accepted clip:

1. **An episode** in the standard contract (`_contract.py:11-16`) so every existing consumer
   works unchanged.
2. **A provenance sidecar** (⛔ **MISSING today**, BACKLOG P3-4). Mandatory fields:

| field | why |
|---|---|
| `source_id`, `source_url`, `retrieved_at`, `content_hash` | traceability, dedup, and §3.6's content-hash disjointness |
| **`source_video_id`** | ⛔ **the CLUSTER UNIT for every interval** — §3.3b measured design effect **9.717** |
| `licence_class` (`ship` / `ship-sa` / `nc` / `refuse`) | §6.4 |
| `corpus_key` (P3-namespaced) | §7.1 |
| `calib_estimate` + `calib_confidence` + `calib_method` + **`fully_canonical`** + `achieved_f_eff` | §3.3b measured 63.3 % not fully canonical |
| `scale_lambda` + `scale_confidence` + `scale_method` + `scale_drift` | §4.1 — scale explicit, never folded in |
| `idm_version`, `idm_commit` | reproducibility; and §3.2.1's ensemble-vs-single-checkpoint trap |
| **`channel_admissibility`** per channel (`speed`, `yaw_rate`, `steer`, `accel`, waypoints) | §3.5, §5.2 — inadmissible ⇒ **masked**, not down-weighted. ⛔ no `κ` channel exists |
| `cadence_hz` **and** `upsample_method` + its measured error if not native 10 Hz | §5.2's 5× trap |
| `quality_gate` verdicts and the resulting loss weight | stage ⑧ |
| `is_parity: false` (constant) | §7.1 |

3. **A corpus manifest**: counts, per-gate yields, licence composition, **and the
   video/clip/channel census** (§3.3b) so the effective n is never mistaken for the clip count.

### 6.2 P3 → P4

P4 receives a **corpus key + manifest**, never loose files.

- The key is **P3-namespaced** and **not** registered as a parity split, so `parity.py` prints
  its loud `[parity] NON-PARITY` line and returns `parity=False`.
- **Channel masks are honoured, not advisory.** If `accel` is inadmissible, the export contains
  no `accel` and the loss cannot compute one. A mask a trainer can ignore is not a mask.

### 6.3 P3 → P5 (TanitScena)

One clip record per accepted clip: identifiers, licence class, situation tags, quality scores,
pointer to the episode. P5 owns embeddings and search.

### 6.4 Provenance and licence obligations of internet-sourced video

**The tier system already exists** —
`…/Data Engineering/Implementation/incoming/2026-07-22-youtube-idm-pipeline/LICENSING_TIER_ANALYSIS.md:17-19`
defines four tiers: **`ship` · `ship-sa` · `nc` · `refuse`**, with the augmentation rule
(`:21-22`): **`tier(derivative) = strictest(source, generator, conditioning_labels)`**.

Its conclusions:

| artifact | tier |
|---|---|
| raw YouTube frames | ⛔ **`refuse`** to re-host; `nc`-with-caveats internal only (`:31-48`) |
| pseudo-labels | inherit source; a standalone URL+timestamp layer is `nc` pending review — the OpenDV model (`:50-58`) |
| a YouTube-pretrained world model | internal research OK with a provenance stamp; **public/commercial release gated on legal review** (`:65-80`) |

⛔⛔ **AND THE PIPELINE DID THE OPPOSITE OF ITS OWN RECOMMENDATION.** The analysis recommends a
clean first slice of **CC-BY channels only** (`:136`). But `harvest_scaleup.py:217` sets
`--allow-noncc default=True`, and the resulting corpus carries **licence `None` on 343/343
clips** (`YT_DB_RETRY.md:167`). The result note flags it as *"a posture for the PI to confirm,
not one to adopt silently"* (`YOUTUBE_GEOCALIB_IDM_RESULT.md:66-69`).

Three `HYPOTHESIS`/`gated` rows are explicitly routed *"to Sayed + legal before any public or
commercial step"* (`:142-143`). ⇒ §9 **Q1**.

**Binding regardless of the above:** every source carries a licence class **at ingest**;
derived artifacts inherit the **strictest** input tier; augmented datasets are published
**PRIVATE** on the PI's paid HF account (TANITAD_PROGRAMME.md §0); research-licensed material
**is** usable — the programme forbids reflexive refusals — but *usable* is not *re-hostable*.

---

## 7. Invariants

### 7.1 ⛔ Reconstructed data is NON-PARITY BY CONSTRUCTION

> The canonical train corpus is **`physicalai-train-e438721ae894`** (2376 episodes) with
> skip-hash **`f09e44db`** (`stack/tanitad/data/parity.py:78-80`). **Anything that re-selects
> episodes breaks cross-arm comparability and must be refused.**

Every episode P3 produces is **outside** that corpus, permanently:

1. **P3 never writes into a parity cache directory** and never adds, removes or re-selects an
   episode in `e438721ae894`.
2. **P3 corpus keys are namespaced** (`p3-<source>-<hash>`) and **never registered in
   `parity_manifest.json` as a parity split** ⇒ they hit the documented non-parity path: one
   loud `[parity] NON-PARITY` line, `parity=False`.
3. ⚠️ **Namespacing is load-bearing.** `parity.corpus_key_of` (`:211`) resolves keys by **path
   substring**, and the module header records that the pre-existing check *"passes for a
   correctly named directory holding the wrong number of episodes"*. A P3 key containing a
   parity key as a substring would be a silent catastrophe. **A test must assert disjointness**
   (BACKLOG P3-5).
4. **P3 data enters training only as a PRETRAINING PREFIX**, never mixed into the parity
   fine-tune (`IDM_VIDEO_PRETRAIN_DESIGN.md:102-107`). The parity fine-tune stays bit-identical
   whether or not P3 data exists — which is what makes §8.1's A/B interpretable.
5. **The known-leaky val split stays refused**: `physicalai-val-f1b378f295ae` has 78.5 % of its
   populated episodes in the parity train set.

### 7.2 Labels may use privileged data; inference is VISION-ONLY

Binding (CLAUDE.md, PI 2026-08-03). Applied in §5.1. Offline label derivation may use ego, CAN,
future frames, calibration. Any model that deploys is vision-only. The IDM sits in the first row
— and its own inputs must still exclude anything its labels were derived from.

### 7.3 Licence class stated per source

No source is ingested without a licence class recorded **at ingest**; derived artifacts inherit
the strictest input tier; nothing is re-hosted publicly without the PI's decision (§9).
⚠️ **Currently violated**: 343/343 clips carry licence `None` (§6.4).

### 7.4 Every reconstructed channel carries an admissibility mask

§3.5. Typed refusal, never a clamp. A reconstructed channel is more failure-prone than a
measured one.

### 7.5 Estimator discipline

- Paired **episode-cluster bootstrap** for every interval; ⛔ never `overlapping_holdout_se` —
  it biases the **point estimate** as well as the interval.
- ⛔ **On reconstructed corpora the cluster unit is the SOURCE VIDEO, not the clip.** MEASURED:
  343 clips ← 55 videos, design effect **9.717**, `kish_neff` **35.3**; and the clip unit has
  already **manufactured a false separation** on `long_accel` (clip [1.018, 1.337] excludes 1;
  video [0.976, 1.372] does not).
- Splits are route-, episode-, **and content-hash**-disjoint (§3.6).

---

## 8. Success criteria — pre-registered, both outcomes committed

### 8.1 ⭐ E-P3-DOWNSTREAM-1 — the headline criterion

> **Does reconstructed data measurably improve a fixed model versus not having it?**

⚠️ **A version of this has ALREADY been run, and its result is POSITIVE but CONTESTED. Read
this before designing the next one.**

`…/2026-07-27-yt-dB-retry/pod_artifacts/results_scaleup_downstream.json` — 343 clips /
38,416 windows / 4 seeds:

| metric | floor | pseudo-YT |
|---|---:|---:|
| `speed_r2` | −0.4387 | **+0.7264** |
| `yaw_r2` | 0.5505 | **0.7285** |
| `ade_2s` | 12.6098 | **4.9776** |

`pseudo_yt_beats_floor_all_seeds: true`, `ci_excludes_0_all_seeds: true`, `frac_boot_gt0: 1.0`
on all four seeds. `YT_DB_RETRY.md:322` scores the pre-registration ***"① HOLDS — DECISION-GRADE
WIN"***.

⛔ **Three reasons it cannot be quoted as settled** — all from the same documents:

1. **The estimator's cluster unit is wrong.** `bootstrap_gap_ci()`
   (`run_youtube_pilot_downstream.py:147-163`) resamples **clips**. The corpus is 343 clips from
   **55 videos**, design effect **9.717** ⇒ intervals ~3× too narrow. The same unit error
   **already manufactured a false separation** on `long_accel` elsewhere in this corpus.
2. **The "ceiling" is not a ceiling.** Pre-registered criterion (c) fraction-of-ceiling =
   **1.0695 > 1** — the pseudo arm beats the real-label reference. `YT_DB_RETRY.md:328-343`
   lists five violated preconditions and its own recommendation (`:345-351`) is to record ① by
   the letter and treat the substantive claim as **② PARTIAL/BOUND**.
3. **A separate, NEGATIVE out-of-corpus result**: yaw-rate spread ratio YT/PhysicalAI =
   **0.547 [0.419, 0.931]**, excluding 1 under the **corrected video-cluster** bootstrap and
   replicated at n = 100 / 200 / 343 — the reconstructed corpus is **measurably less dynamic**
   than the real one.

**The re-run design:**

- Fix one architecture, recipe, parity fine-tune, seed set (≥3).
  **Arm A** parity fine-tune only. **Arm B** pretrain on N hours of P3 data, then the
  **bit-identical** parity fine-tune.
- **Scoring**: the deployed val, **all four families, per-family, never pooled**, plus ADE.
- **Estimator**: paired episode-cluster bootstrap, B=2000, ⛔ **clustered on SOURCE VIDEO**.
- **Controls**: Arm B repeated with the P3 prefix **label-shuffled** (same frames, scrambled
  actions). If shuffled-label pretraining helps as much as real, the gain is from **pixels, not
  from the IDM** — and the *product* claim is not supported even if ADE improves.

**Committed outcomes:**

- ✅ **SUPPORTED** — paired Δ separated-better on ADE **and** ≥1 family, **none separated-worse**,
  **and** the label-shuffled control separated-worse than Arm B ⇒ the reconstruction line is the
  data moat; scale it.
- ❌ **REFUTED** — Δ not separated, or separated-worse, or the label-shuffled control matches
  Arm B ⇒ **P3's action-reconstruction claim is not supported at this scale.** Value falls back
  to (i) calibration estimation as a service to P2, (ii) the IDM as an analysis instrument,
  (iii) frames-only representation pretraining — each independently checkable, none requiring
  the action claim.

⚠️ **N is a parameter; a null at small N is not a null at large N.** Fix N **before** the run
and scope the verdict to it. ⚠️ And note the effective N here is **videos, not clips**.

### 8.2 Supporting criteria

| id | criterion | SUPPORTED | REFUTED |
|---|---|---|---|
| **E-P3-CALIB-1** | calibration on **spoofed-unknown** frames, ⛔ scored on a **VARYING-focal** arm with the constant-predictor control published alongside (§3.3) | focal within **±10 %** median on the varying arm, **separated from the constant predictor**, and focal-tracking `r` reported | ⇒ restrict to clips with detectable lane geometry, report the surviving fraction, **and treat the fixed-HFOV fallback as the operating point** |
| **E-P3-CALIB-2** | **extrinsics** (pitch / roll); height unobservable without a metric anchor | pitch/roll within a stated bound on clips with known extrinsics | ⇒ extrinsics are supplied-or-assumed and §4.1's scale coupling inherits an unbounded term — say so in every scale claim |
| **E-P3-SCALE-1** | §4.2 | median \|log(λ̂/λ)\| ≤ log(1.10) both domains, controls separated, regression arm red | ⇒ scale-free deliverable; longitudinal channel **masked** |
| **E-P3-IDM-1** | per-channel admissibility, full L0–L4 ladder | channel separated-positive vs L2 **and** beats L1 | ⇒ that channel is **masked** in export |
| **E-P3-XDOM-1** | cross-domain transfer (L5) — currently `PASS = false` | separated-positive out-of-domain | ⇒ P3 is **per-domain**: an IDM per camera family, and the cost model changes |
| **E-P3-AGREE-1** | IDM ↔ ego-motion cross-agreement as the per-clip quality score (`YOUTUBE_DASHCAM_STRATEGY.md:45-48`) | agreement predicts held-out label error (rank correlation, with CI) | ⇒ no usable per-clip gate; gating must come from elsewhere |
| **E-P3-FAM-1** | four families computed for every IDM eval (L6), incl. the two currently absent | `idm_families` wired into every `run_idm_*.py`; **STRATEGIC** and `distance_keeping` computed or their absence justified per family with `n` | implementation obligation, not a hypothesis |
| **E-P3-CONTAM-1** | harvest filters actually catch contamination | contamination-catch rate separated above zero | ⇒ **currently REFUTED**: `contamination_frac 0.1283`, `caught_by_shipped: []` (§3.3b) |

### 8.3 Production readiness

Spec'd (this document), tested (`stack/tests/`), documented, demoable. `pytest -q` green before
any commit. Instrument changes ship with their regression test **in the same commit**. A **video
demo** in the standing viz format — camera overlay + metric BEV inset + decoded tactical
manoeuvre + strategic route/goal — on a *reconstructed* clip, with the reconstructed track drawn
**against CAN** where CAN exists.

---

## 9. Open questions for the PI

**Q1 — ⚖️ LEGAL / LICENCE POSTURE ON YOUTUBE-SOURCED VIDEO. A PI decision, explicitly not mine.**
The tier framework exists (§6.4) and routes three rows *"to Sayed + legal"*. What is needed:
  a. May we **download and train on** YouTube-sourced video for research?
  b. May we **re-host** frames or derived episodes — even PRIVATE on the paid HF account? *(The
     analysis says raw frames are `refuse` to re-host.)*
  c. **Do we restrict harvesting to CC-BY?** The analysis recommends exactly that as the clean
     first slice; the shipped pipeline defaulted to `--allow-noncc=True` and produced **343/343
     clips with licence `None`**. ⚠️ **This is a live discrepancy between a written
     recommendation and shipped behaviour**, not a hypothetical.
  d. Do **derived pseudo-labels** inherit the source tier (the current assumption), or are they
     a separate work?
  e. Is there a preferred **lower-risk substitute corpus** (OpenDV-YouTube pointers, licensed
     dashcam data, purchased footage) the PI would rather fund?
⛔ **Until (a)–(c) are answered, P3 must not harvest at scale.** Note the prior authorization is
recorded as **SPENT** — *"Any further YouTube harvesting is a NEW decision for the PI"*
(`LOOP_STATE.md:693-703`).

**Q2 — the longitudinal channel.** Given `accel` is at the null (§5.2), does the PI want P3 to
(i) ship `yaw_rate` and mask `accel`, (ii) fund an input-side change, or (iii) re-target to a
speed profile / headway? ⚠️ Note (ii) and (iii) both require first separating the **input**
cause from the **label** cause — on PhysicalAI the label caps a perfect kinematic estimator at
**R² 0.188**.

**Q3 — cross-domain transfer.** `go_no_go.PASS = false` has never passed, the data-diversity
hypothesis is **REFUTED**, and the registry calls the own-encoder thesis **not supported**. Does
the PI want to (i) re-open the substrate question, (ii) accept a **per-domain IDM** cost model,
or (iii) gate P3 on a P1 encoder change?

**Q4 — the stranded corpus.** The 343 clip latents exist only on `pod3:/workspace/tmp/`, are
absent from the 2026-08-02 pod rescue, and are almost certainly **gone**; re-harvest is
bot-blocked and unauthorised. Does the PI want a recovery attempt, a re-harvest decision, or
acceptance of the loss?

**Q5 — compute posture.** All P3 de-risking is scoped to local/CPU and the dev-box GPU. Confirm
any HF Pro GPU use is pre-authorised per job, given the hard quota ceiling.

**Q6 — scope of "reconstruction".** Does P3 also own *partially* instrumented corpora (pose but
no calibration, or calibration but no actions)? A genuine grey zone with P2 that changes the
backlog ordering.

---

## 10. Provenance of this document

**Read in full**: `Project Steering/TANITAD_PROGRAMME.md` §1/§3/§6.

**`Project Steering/AGENT_CHARTERS.md` — DOES NOT EXIST in this worktree.** Two probes:
`find . -iname "*CHARTER*"` (no hits) and `git ls-files | grep -i charter` (no hits). The brief
directed me to read its §0 + §2. **Reported as an integration item, not worked around silently.**

**Primary sources read directly**: `stack/scripts/idm_head.py`,
`stack/tanitad/eval/idm_families.py`, `stack/tanitad/data/{calib,_contract,comma2k19,parity}.py`,
`stack/tanitad/models/metric_dynamics.py`, `Project Steering/MODEL_REGISTRY.md` (IDM sections),
`…/IDM_VIDEO_PRETRAIN_DESIGN.md`, `…/YOUTUBE_DASHCAM_STRATEGY.md`,
`…/2026-07-25-geocalib/{geocalib_intrinsics.py,VALIDATION_REPORT.md}`,
`…/2026-08-06-v1-defect-triage/OPTION2_UNICYCLE_READOUT.md`.

**Raw-JSON verification** of §3.2, §3.3b, §6.4 and §8.1 was performed by two delegated
read-only surveys reading the artifacts directly; their numbers are cited to artifact + JSON key
path throughout. Evidence class **MEASURED**, with the re-verification obligation in BACKLOG
**P3-2** (a number quoted from a docstring is quoted from prose until it is read from the JSON).

⚠️ **Two methodological cautions earned while writing this document, both recorded because they
are reusable:**
1. **On this Drive-backed worktree, ripgrep silently skips un-hydrated files.** A "no matches"
   result is **not** evidence of absence — it produced a false negative on VO code here (§3.4).
   Re-run greps after hydration, or use a retry loop on direct reads.
2. **I published a false absence claim in an earlier draft** ("L0 and L3 have never been run")
   because I probed `stack/tests/` and not the `incoming/` work packages. Corrected in §5.4.
   CLAUDE.md's *"absence found at ONE location is not absence"* applies to this document too.
