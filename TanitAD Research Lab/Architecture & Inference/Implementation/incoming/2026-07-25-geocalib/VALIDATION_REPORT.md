# GeoCalib per-video intrinsics — validation report

**Agent:** geocalib subagent · **Pod:** `tanitad-eval` (A40, idle) · **Date:** 2026-07-25
**Goal:** replace the YouTube-IDM pilot's FIXED 100° HFOV geometry approximation with a
per-video camera-intrinsics estimate (GeoCalib, ECCV 2024, `cvg/GeoCalib`), so arbitrary
YouTube dashcam video can be canonicalized to our `f_eff≈266` with correct geometry.

Evidence classes: **MEASURED** (ours + artifact path) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

---

## TL;DR verdict — **QUALIFIED PASS** (use GeoCalib with robust aggregation + fallback)

GeoCalib recovers known intrinsics on **rectilinear** dashcam frames well enough to
**substantially reduce** the geometry error vs the fixed-HFOV approximation, and it is
resolution-robust (native ≈ 480p). It is **not** a precise calibration oracle: it carries a
systematic ~6-11 % focal under-estimate, has per-frame outliers up to ±30 %, weakly tracks
focal changes on identical content (regresses toward a ~50-55° vFoV prior), and **cannot**
handle true wide fisheye (fits a much narrower pinhole). The deliverable therefore uses
**multi-frame MAD-robust aggregation + a confidence-gated fall-back to the fixed HFOV** — so
the swap is a strict improvement on rectilinear video and *never does worse* than the pilot.

---

## P1 — GeoCalib running on the eval pod  [MEASURED]

- Installed into a `--system-site-packages` venv (`/workspace/geocalib_work/gcvenv`) that
  inherits the pod's `torch 2.8.0+cu128` (CUDA A40); GeoCalib built editable from a shallow
  clone of `github.com/cvg/GeoCalib` (not on PyPI). Deps: opencv/kornia/matplotlib. Weights
  auto-download from the repo's GitHub v1.0 release (`geocalib-{pinhole,distorted}.tar`).
- Smoke (`comma_00`, native 1164×874): `result["camera"].f = 791.6 px`, `vfov = 57.8°`
  (GT f=910, vfov=51.3°) — runs, estimates focal + vFoV from a single frame. **P1 met.**
- API used: `GeoCalib(weights=…).calibrate(img_chw_[0,1], camera_model="pinhole")` →
  `camera.f (fx,fy px)`, `.vfov/.hfov (rad)`, `.c`, plus `focal_uncertainty`.

## P2 — Intrinsics recovery on KNOWN-GT data  [MEASURED · `geocalib_validation_results.json`]

Known-GT set (`prep_frames.py`, 36 frames): comma2k19 (rectilinear, GT focal **910 px** @
1164×874, our `COMMA2K19_FOCAL_PX`) at native + ~480p; a controlled digital focal-sweep on one
comma frame (exact GT focal 910·s, s∈{1,1.25,1.5,2}); PhysicalAI front-wide (120° f-theta
fisheye, per-clip `fw_poly` from `calibration/camera_intrinsics`).

| set (model=**distorted**) | n | focal err mean | focal \|err\| median | vFoV err mean | note |
|---|---|---|---|---|---|
| comma_native (GT 910) | 12 | **−6.4 %** | **6.8 %** | +3.6° | PRIMARY absolute test |
| comma_480p (YouTube res) | 12 | −5.9 % | 7.1 % | +3.3° | **resolution-robust** (≈native) |
| comma_focalsweep (exact GT) | 4 | −26.7 % | 25.0 % | +12.7° | **weak** (Pearson r=0.41) |
| physicalai_native (120° fisheye) | 8 | — | — | — | vFoV est ~37° (mean); f/paraxial 1.8 |

*(`pinhole` weights were uniformly worse: comma_native \|err\| median 13.9 %, vFoV err +5.9°.
`distorted` is the chosen default.)*

**Reading:**
1. **Rectilinear absolute recovery is usable, not exact.** On comma the distorted model lands
   focal to ~7 % median error with a systematic ~6 % under-estimate (⇒ vFoV over-estimate ~+3.6°).
   Per-frame it is noisy: most frames ±3-8 %, but outliers reach −34 % (`comma_01`) and +20 %
   (`comma_10`). ⇒ **single-frame use is unsafe; aggregate over frames.**
2. **Resolution-robust.** 480p ≈ native (the estimate rides on vFoV, which is scale-free) —
   important because the YouTube pilot decodes ≤480p.
3. **Weak absolute-focal tracking (the main limitation).** Digitally zooming ONE comma frame by
   up to 2× (true focal ×2) barely moves GeoCalib's estimate (est ratio 0.93→0.74→0.76→0.50;
   r=0.41). GeoCalib leans on a learned ~50-55° vFoV prior and resists narrow fields. So it
   discriminates cameras near the training FoV prior far better than extreme ones.
4. **Fisheye is out of model.** On the 120° f-theta front-wide, GeoCalib's pinhole fit reports
   vFoV ~22-55° (fitting the central quasi-rectilinear region), f ≈ 1.8× the paraxial focal —
   an expected, honest model mismatch. For such cameras the estimate must be rejected (it is;
   see P3 fallback).

**Honest caveat on generalization:** the *absolute* rectilinear claim rests on **one** real
camera (comma2k19); the sweep adds controlled relative GT but on the same content. A stronger
validation would add more published-intrinsics rectilinear cameras (KITTI/nuScenes/Waymo).
This is the top follow-up (see NOTE.md).

## P3 — Integration `geocalib_intrinsics.py`  [MEASURED — importable + runs on real video]

Clean importable interface (the parallel non-CC-scale agent consumes this):
- `GeoCalibEstimator(weights="distorted", camera_model="pinhole", device=None, hfov_fallback_deg=100)`
  - `.estimate_from_frames(frames)` and `.estimate_from_video(mp4, n_frames=16, anonymizer=…)`
    → **`EstimatedIntrinsics`** (median vFoV over frames, **MAD outlier rejection**, per-frame
    spread → `confidence` ∈ {high,medium,low}, `fallback_used`).
  - `EstimatedIntrinsics.focal_px(width, height)` → focal for any decode resolution, ready for
    `tanitad.data.calib.focal_crop_resize(vid, f_px, size)` → canonical `f_eff≈266`.
- `decode_canonical_geocalib(mp4, anonymizer, estimator|estimated, …)` — a **drop-in** for
  `yt_pilot_common.decode_canonical`; the ONLY change is the focal source (GeoCalib per-video
  instead of `nominal_focal_px(W, hfov_deg)`). Everything downstream is byte-identical.
- **Safety by construction:** low confidence (vFoV MAD > 9°) or < 4 valid frames ⇒
  `fallback_used=True` and the geometry is exactly the pilot's fixed-HFOV crop.

MEASURED on real video (pod):
- comma `video.hevc` (16 frames): vFoV **59.5°**, confidence **high**, MAD 2.3°, 13/16 frames used.
- PhysicalAI `.mp4` fisheye (16 frames): vFoV 38°, confidence **low** (MAD 6.3°) — correctly flagged.
- 8-frame comma PNG set → high confidence; 8-frame fisheye PNG set → **fallback** (MAD>9°).

**The geometry the swap actually changes** (crop side that lands `f_eff≈266`), comma real video:

| geometry | focal (px) | crop side | retained width | vs true (~910px / 75 %) |
|---|---|---|---|---|
| **GeoCalib** | 765 | 736 | **63 %** | −12 pp |
| fixed-100° HFOV (pilot) | 488 | 470 | **40 %** | **−35 pp** |

For a ~66°-HFOV camera like comma, the fixed 100° assumption over-crops (~1.6× tighter than
GeoCalib) — inflating apparent motion ⇒ biasing pseudo-speed. GeoCalib recovers ~2/3 of that
error. *(`decode_canonical_geocalib` drop-in end-to-end result appended below.)*

## P4 — GeoCalib vs fixed-100° HFOV on the pilot's REAL YouTube clips
[MEASURED · `youtube_geocalib_measurement.json` · 12 of the pilot's 32 CC videos re-downloaded
via the staged `pointers.jsonl`; **no imagery persisted** — each mp4 deleted right after estimation.]

14 tried → 12 measured (2× HTTP-403), 11 GeoCalib-confident, 1 fell back. All arrived at **640×360**
(the pilot's ≤480p filter can yield 360p — see escalation). Per-video estimated HFOV:

| stat (est. HFOV, deg) | all 12 | GeoCalib-confident (11) |
|---|---|---|
| mean | 63.4 | 60.1 |
| median | 66.6 | 60.5 |
| min–max | 32.4 – 100.0¹ | 32.4 – 77.3 |

¹ the 100.0 is the single **fallback** clip (self-inconsistent → fixed HFOV, as designed).

**Decision-grade reads:**
1. **A single fixed HFOV cannot fit these clips.** GeoCalib's per-clip HFOV spans **32–77°** — and
   **only 1 of 12** clips is within 10° of the assumed **100°**. Whatever the exact truth, the clips
   *differ from each other*, so per-video estimation is warranted. This is the core motivation,
   confirmed on real data.
2. **The fixed 100° was too WIDE for ~every clip** (11/12 estimated narrower), the same direction as
   the comma finding. Consequence in-pipeline: the fixed assumption crops to ~72 % of the frame and
   applies a **~1.4× excess linear zoom** vs GeoCalib (median `linear_zoom_fixed_over_gc` = 1.40),
   inflating apparent motion ⇒ biasing pseudo-speed high.
3. GeoCalib **self-flags** the clips it cannot trust: 3/12 `confidence="low"` + 1 fallback — exactly
   the clips a safe pipeline should down-weight or drop.

**Honest limits of P4 (do not overclaim):**
- **Unknown GT** — these YouTube intrinsics are the very thing we cannot measure, so per-clip
  *absolute* accuracy is unverified. Trust is bounded by the P2 known-GT result (~7 % median) and the
  confidence gating, not by these numbers themselves.
- GeoCalib's **prior-regression bias + 360p compression** likely make the estimates *optimistically
  narrow*; several fall below the canonical 51° vFoV, which is why the 480p crop metric **saturates**
  at 1.40 (the canonical `f_eff=266` square exceeds the 360-px height → clamps to full frame). So
  1.40 is a floor on the geometry change, measured coarsely, not a precise correction.
- **Downstream ADE closure not done here** — re-pretraining on GeoCalib-cropped vs fixed-cropped
  YouTube and comparing parity-val `speed_r2` needs the IDM encoder+labeler on pod3 (out of this
  eval-pod agent's lane). P4 quantifies the *geometry* change; the ADE delta is the pre-registered
  follow-up (NOTE.md escalation 3).

## Decode drop-in — end-to-end confirmation [MEASURED]
`decode_canonical_geocalib` verified across a **repeated multi-clip estimate(CUDA)+decode loop**
(the parallel agent's exact per-clip flow), output `[T,3,256,256]` uint8, **no hang**:
- comma `video.hevc` (pts-less → index resample): achieved **`f_eff` = 266.1**, `fully_canonical=True`
  (hFoV 74.7°) — the canonical target is hit for a normal rectilinear clip.
- fisheye `.mp4` (pts path): `f_eff` = 372, `fully_canonical=**False**` — correctly flags a clip whose
  estimated field is narrower than the canonical 51° so the crop clamps to the frame (a provenance
  signal the pilot's fixed path lacked).

**Decode-path bug found + fixed here (MEASURED):** the estimate(CUDA)→decode-per-clip flow that the
GeoCalib integration introduces hit an **intermittent PyAV deadlock** — a threaded decoder
(`thread_type="AUTO"`) torn down while a CUDA context is live hangs on container close. The pilot
never saw it (its decode had no inline CUDA). Fixed by decoding **single-threaded**
(`DECODE_THREAD_TYPE="NONE"`) in both decode paths; decode of small frames is not the bottleneck. The
consuming pipeline MUST use this module's decode path (or `thread_type != "AUTO"`) when GeoCalib runs
inline. *(Both resample paths — pts mp4 + pts-less hevc index-fallback — verified.)*

