# YouTube-dashcam → IDM data pipeline — DESIGN

**Status:** design + one prototype (the intrinsics front-end, §3, is BUILT and MEASURED).
**Scope discipline:** the full ingest/labeling pipeline is **gated on the IDM cross-rig proof**
(`stack/scripts/run_idm_proof.py`, running separately). This doc builds only the parts that are
**valuable regardless of that verdict**: the intrinsics front-end (de-risks the dominant
`FAIL`-branch failure mode), the licensing tier verdict, and the pipeline skeleton the proof plugs
into. Nothing here scrapes or downloads YouTube — ingest is design-only (Sayed decides scraping under
the ToS verdict in `LICENSING_TIER_ANALYSIS.md`).

**Feeds:** `Architecture & Inference/IDM_VIDEO_PRETRAIN_DESIGN.md` (the IDM this pipeline labels for).
**Amends/operationalises:** `Data Engineering/Research/YOUTUBE_DASHCAM_STRATEGY.md` (H7, stages Y0–Y3).

**Evidence classes** (CLAUDE.md operating standard): every external number is `PUBLISHED (cited)`;
the front-end round-trip is `MEASURED (this task, ftheta_frontend_result.json)`; forward-looking
design choices are `DESIGN`/`HYPOTHESIS`. No number here decides a GPU-day.

---

## 0. Where this sits — and what is gated

The IDM/YouTube thesis is a **three-lever chain**, and only two levers are proof-independent:

| lever | what | status |
|---|---|---|
| **Domain/intrinsics transfer** (does an IDM trained on our CAN corpora recover action on a *different camera*?) | the pre-registered rig-A→rig-B then PhysicalAI→comma proof | **RUNNING SEPARATELY** — `run_idm_proof.py`. Its PASS/FAIL is the go/no-go for the whole line |
| ⭐ **Intrinsics normalisation** (can uncalibrated YouTube frames be brought to our F_REF=266 encoder distribution?) | the f-theta front-end, §3 | **BUILT + MEASURED this task** — de-risks the proof's `FAIL` branch *before* the verdict |
| **Licensing** (can we train on / re-host YouTube-derived data?) | the tier verdict | **RESOLVED this task** — `LICENSING_TIER_ANALYSIS.md` |

The front-end and the licensing verdict are worth landing **now** because they are exactly the two
"evidence gaps — zero claims" the IDM research flagged as least-de-risked (`IDM_VIDEO_PRETRAIN_DESIGN.md`
§4), and neither depends on how the proof lands. If the proof PASSES, the front-end is on the critical
path immediately; if it FAILS, the front-end **is** the first item of the §5 `FAIL` mitigation. Either
way it is not wasted — which is the test for doing it pre-verdict.

---

## 1. End-to-end flow

```
                                   ┌─────────────────────── GATED ON THE IDM PROOF ───────────────────────┐
 S0 discovery/ingest → S1 dedup → S2 quality gate → S3 intrinsics front-end → S4 IDM labeler → S5 WM prefix
   URL list (not bytes)   drive-  resolution /         ⭐ f-theta canon to        (stub now;      pretrain, then
   yt-dlp/OpenDV method   level   fwd-dashcam /         F_REF=266 (BUILT §3)      filled after     FIRE-WALLED from
   DESIGN ONLY            hashes  ego-motion / day                                the proof)       the parity corpus
                                   └── buildable now ──┘ └─ BUILT ─┘              └──── gated ────┘
```

The break point is deliberate: **S0–S3 are corpus-preparation and are safe to specify/build now; S4–S5
are where the (unproven) IDM action-recovery enters, so they stay stubs until the proof lands.**

---

## 2. Stages (each with its falsifier, mirroring the H7 strategy discipline)

### S0 — Source discovery + ingest  *(DESIGN ONLY — do not run pre-ToS-verdict)*
- **Discovery:** curated channel lists + search seeds (dashcam / driving-tour channels), as a **CSV of
  YouTube video IDs + time ranges** — never a bytes dump. This is the exact model OpenDV-YouTube uses:
  it ships a video-ID sheet + a download script and has the user fetch frames themselves "rather than
  providing pre-hosted versions, respecting YouTube's licensing requirements" [OpenDV/GenAD,
  arXiv:2403.09630]. **We adopt that model verbatim** — see `LICENSING_TIER_ANALYSIS.md`.
- **Ingest mechanism:** `yt-dlp` at a fixed format/resolution ladder, frames extracted at a target
  fps, immediately handed to S1–S3. **Gated:** the *decision to run* S0 is Sayed's, under the ToS
  verdict. The tooling is designed; it is not invoked here.
- **Falsifier S0:** if a channel's videos carry heavy overlays / non-driving segments > a threshold
  (measured in S2), the channel is dropped at discovery — cheaper than filtering frame-by-frame.

### S1 — Drive / clip dedup
- **Two levels, both cheap, both pre-decode where possible:**
  1. **Video-level:** exact dup by YouTube ID; near-dup re-uploads by title/duration + a perceptual
     hash (pHash) of N sampled thumbnails.
  2. **Drive-level (within a long video):** a single upload is often one continuous drive; slice into
     windows on scene-cut boundaries and dedup **overlapping windows** with a pHash `groupby` — the
     same double-counting trap the tier work flagged for L2D ("sliding-window overlap double-counts
     ~50 % unless split on drives", `TANITDATASET_TIER_INTEGRATION_2026-07-21.md` §6). Reuse that
     de-dup primitive (augmentation **A2**), not a new one.
- **Falsifier S1:** two windows with pHash distance < τ **and** ego-motion (S2) correlation > ρ are the
  same drive segment → keep one. Tune τ,ρ on a labelled dup/non-dup set before scaling.

### S2 — Quality filtering (the "is this usable dashcam" gate)
Ordered cheapest-first so most rejects die early:
| filter | signal | reject if |
|---|---|---|
| resolution / bitrate | container metadata | < 720p or heavy compression blocking |
| **forward-facing dashcam** | horizon present in mid-frame; road-VP detectable (reuse `vp_row`, §3) | no stable road VP over the clip → not a forward dashcam (interior/rear/handheld) |
| **ego-motion present** | optical-flow radial-expansion signature (driving-segment detection, H7 Y1) | flow ≈ static → parked / non-driving segment |
| overlay / watermark | static-pixel map over the clip (comma overlay precedent, H7 Y1) | large static text/logo region → tag + optionally crop |
| day / clear (v0) | luma histogram + simple sky test | night/rain only if we choose to include (tag, don't hard-drop — H7 treats it as embodiment/condition diversity) |
- **Falsifier S2:** hand-label 200 clips accept/reject; the gate must reach > 0.9 agreement with the
  human accept set, else loosen to "video-only" (representation-learning) use rather than dropping.
- **Reuse:** the road-VP detector (`vp_row`) is dual-purpose — a *filter* here and a *calibration cue*
  in S3. One primitive, two jobs.

### S3 — ⭐ f-theta intrinsics front-end  *(BUILT + MEASURED — §3 below)*
Estimate per-video intrinsics, canonicalize every frame to our **F_REF=266 pinhole** (9-ch, 256px), so
the frames land in the exact distribution the encoder trained on. **This is the de-risk. It round-trips
cleanly (§3).**

### S4 — IDM labeler hook  *(STUB — filled after the proof)*
- **Interface, defined now:** `label_clip(frames_canon[T,9,256,256]) -> {steer, yaw_rate, long_accel,
  target_speed, ego_traj_2s, action_quality}` — the non-causal IDM head from `IDM_VIDEO_PRETRAIN_DESIGN.md`
  §2, applied offline (past+future frames available). The front-end guarantees its input is in-distribution.
- **Cross-check (H7 Y2):** monocular VO on the same canonical frames → an independent action estimate;
  (IDM ↔ VO) agreement is the per-clip `action_quality`. Low agreement ⇒ **video-only** use (encoder
  representation learning, no action conditioning) — still valuable.
- **Left a stub on purpose:** the IDM's action-recovery accuracy on a foreign camera is precisely what
  the proof measures. Wiring the labeler before the verdict would be building on an unproven premise
  (operating standard rule 5 / §0 above).

### S5 — WM pretraining prefix  *(STUB — parity-fire-walled, §4)*
Pseudo-labeled YouTube video becomes a **pretraining prefix** for the flagship trunk + predictor;
the parity corpus fine-tune is untouched. See §4.

---

## 3. The intrinsics front-end — design + PROTOTYPE RESULT

**Problem.** YouTube frames have UNKNOWN intrinsics; our encoder trained on F_REF=266-canonicalized
frames (`calib.py`, D-016). Feed a differently-scaled frame and the action→pixel-motion geometry is
wrong — corrupting exactly the dynamics the WM must learn. The IDM research lists this as an
"evidence gap — zero claims" (`IDM_VIDEO_PRETRAIN_DESIGN.md` §4). This front-end closes it.

### 3a. Intrinsics ESTIMATION (research + method choice)
YouTube gives no calibration, so intrinsics must be estimated from image content. Three method families:

| family | method | fit for our case | evidence |
|---|---|---|---|
| **learned single-image** ⭐ | **GeoCalib** — end-to-end net + geometric optimisation; predicts **focal + gravity (⇒ horizon/pitch)**; supports **shared-intrinsics across many frames of one camera** and shared gravity for a rigid mount | **Best fit.** A whole YouTube video = many frames from ONE camera → GeoCalib's shared-intrinsics mode is exactly our setting; gravity gives the pitch that D-016's principal-point crop needs | `PUBLISHED` — Veicht, Sarlin, Lindenberger, Pollefeys, **ECCV 2024**, arXiv:2409.06704, `cvg/GeoCalib` (verified this task) |
| **classical geometric** | vanishing-point + horizon self-calibration; **two orthogonal VPs ⇒ f = √(−(vp₁−pp)·(vp₂−pp))** (Caprile–Torre-class) | Works on the self-selecting subset with detectable Manhattan/lane+vertical geometry; **implemented + round-trips exactly** in the prototype (§3c) | `MEASURED` (formula round-trip, this task) + classical result |
| **learned focal regressor** | DeepCalib / CTRL-C-class CNN focal(+distortion) predictor | H7's "learned cross-check"; agreement with the geometric estimate = the acceptance test | `PUBLISHED — from-memory, verify arXiv before finalizing` |

**Chosen production path:** **GeoCalib in shared-intrinsics (per-video) mode**, cross-checked against the
classical road-VP horizon and a learned regressor; accept a video's intrinsics only when the estimators
agree (H7 Y0 acceptance test). **Falsifier Y0 (from the strategy):** on held-out comma frames with
spoofed unknown focal (random crop+resize), estimation must recover f within ±10 %; worse ⇒ restrict to
videos with detectable lane/Manhattan geometry (still a huge subset).

> **Honest limitation (MEASURED, §3c):** a **single** road VP under-determines absolute focal — it fixes
> pitch *given* focal, not focal itself. Absolute focal needs a **second orthogonal VP**, **metric
> structure** (lane width + flat-ground + camera height), or a **learned prior** (GeoCalib). This is
> *why* the strategy pairs the geometric method with a learned cross-check, and why GeoCalib (which
> carries a learned focal prior) is the production anchor rather than pure VP geometry.

### 3b. CANONICALIZATION (prototype — reuses the AlpaSim/D-016-validated primitives)
Once intrinsics are estimated, canonicalization **reuses code already validated end-to-end**, no new
geometry:
- **Rectilinear dashcam** (most dashcams) → `calib.focal_crop_resize(vid, f_est, 256)`.
- **Wide / fisheye dashcam** → `calib.ftheta_crop_resize(intr, center="principal")` — the exact
  function just validated in AlpaSim Option A; `per_clip=True` intrinsics drive the principal-point crop.
- **9-channel encoder contract** → `comma2k19.stack_frames(n_stack=3)` → `[T-2, 9, 256, 256]`.

**PROTOTYPE RESULT** (`ftheta_frontend_prototype.py` → `ftheta_frontend_result.json`; MEASURED on a
real comma2k19 night-highway frame from our lake, native 874×1164):

| branch | function | achieved `f_eff` | dev vs 266 | pass |
|---|---|---|---|---|
| pinhole / rectilinear | `focal_crop_resize` | **266.545 px** | +0.205 % | ✅ |
| wide / fisheye | `ftheta_crop_resize(center="principal")`, per_clip cy=548 | **266.02 px** | +0.007 % | ✅ |
| 9-ch contract | `stack_frames(n_stack=3)` | shape `[2, 9, 256, 256]` | — | ✅ |

**VERDICT: the canonicalization round-trips cleanly to f_eff ≈ 266 on both camera branches.** The
achieved focals reproduce the calib.py test bounds (< 1 %). `canonical_sample.png` shows the resulting
256×256 frame — a coherent road scene whose lane markings converge to the VP the estimator detected
(7 887 intersection votes on the real frame; VP row 119.1 vs the model's assumed 128).

### 3c. What the estimate-ERROR costs (MEASURED — the tolerance argument)
The output's TRUE effective focal, if we canonicalise with a wrong estimate, obeys
**`f_eff_true = F_REF · (f_true / f_est)`** — a focal error maps *linearly* to an f_eff error:

| f_est/f_true | 0.8 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|
| f_eff_true (px) | 332.3 | 295.6 | 280.0 | 266.5 |
| dev from 266 | **+24.9 %** | **+11.1 %** | +5.3 % | +0.2 % |

So Falsifier Y0's ±10 % focal tolerance ≈ ±10 % f_eff — an encoder seeing a scene "11 % more zoomed"
than training. This **quantifies the estimator bar**: intrinsics good to a few-% keep f_eff within a
few-% of 266. (For over-estimates on comma the table saturates because comma is the reference camera —
its crop is already at the sensor edge, so it cannot over-crop; a genuinely wide dashcam frame has pixel
headroom and the law is symmetric both ways. Documented so the saturation is not mistaken for the law
breaking.)

**Two-orthogonal-VP focal round-trip** (analytic, exact): project two orthogonal directions through a
known K(f=910), recover via the formula → **910.0, 0.00 % error**. The estimator *math* is correct; the
real-world difficulty is *detecting* the two VPs — hence GeoCalib as the production estimator.

---

## 4. Parity firewall (references `IDM_VIDEO_PRETRAIN_DESIGN.md` §3 / §6)

**YouTube data is a PRETRAINING PREFIX, never mixed into the parity corpus.** The invariant
(`CLAUDE.md`): the canonical train corpus is `physicalai-train-e438721ae894` (2376 eps, skip-hash
`f09e44db`); anything that re-selects episodes breaks cross-arm comparability and **must be refused**.

- The IDM is a **side model** — it does not touch the WM train corpus or its parity key
  (`IDM_VIDEO_PRETRAIN_DESIGN.md` §3).
- The GenAD/Vista integration pattern (§6): **(1)** train IDM on CAN corpora → **(2)** pass the proof →
  **(3)** canonicalize + IDM-label a YouTube slice → **(4) pretrain** the flagship trunk + predictor on
  pseudo-labeled YouTube → **(5) fine-tune** on the parity corpus (sacred, unchanged).
- **Mechanical separation:** the YouTube-pretrained prefix is a **distinct checkpoint artifact** with
  its own provenance stamp, so the contamination is traceable and the parity fine-tune can always be
  reproduced without it. This mirrors the tier firewall (`assemble_lake_record` raises on a gated source
  reaching the lake) — the firewall is code, not a promise.

---

## 5. Buildable-now vs gated — the manifest of intent

| stage | build now? | why |
|---|---|---|
| S0 ingest | ❌ design only | gated on Sayed's ToS decision (`LICENSING_TIER_ANALYSIS.md`) |
| S1 dedup | ✅ specifiable now | reuses A2 drive-level de-dup; no YouTube bytes needed to build/test on L2D-style data |
| S2 quality gate | ✅ partially | the VP/ego-motion/overlay primitives are corpus-agnostic; testable on comma/L2D |
| **S3 front-end** | ✅ **BUILT** | the intrinsics de-risk; proof-independent |
| S4 IDM labeler | ❌ stub | the proof measures whether the IDM transfers at all |
| S5 WM prefix | ❌ stub | downstream of S4 + a parity-firewalled training run |

**Escalation (operating standard rule 3):** S3's canonicalization primitives are already in HEAD
(`stack/tanitad/data/calib.py`, `comma2k19.py`) — the front-end needs **no new geometry merged**, only
a per-video intrinsics-estimation module (GeoCalib wrapper + the VP cross-check) when S0 is greenlit.
That module is the single integration task this line will raise post-proof; it is scoped here so it is
not re-discovered.

---

## 6. Deliverables — see `MANIFEST.md`
The prototype (`ftheta_frontend_prototype.py`, `ftheta_frontend_result.json`, `canonical_sample.png`),
this design, and `LICENSING_TIER_ANALYSIS.md`, all staged under this incoming dir.
