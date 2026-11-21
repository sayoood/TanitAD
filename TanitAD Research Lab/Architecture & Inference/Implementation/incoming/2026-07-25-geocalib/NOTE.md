# GeoCalib per-video intrinsics — running NOTE (bank-as-you-go)

**Agent:** geocalib subagent · **Pod:** `tanitad-eval` (A40, idle; shares box with pre-existing
`alpa-invest/alpasim` procs at 0 % GPU) · **Started/Date:** 2026-07-25
**Goal (Sayed, 2026-07-25):** implement + validate a per-video camera-INTRINSICS estimator
(GeoCalib) for the YouTube-IDM pipeline, removing the fixed-HFOV approximation so arbitrary
YouTube dashcam video is pseudo-labeled with accurate ego-motion geometry.

Evidence classes: **MEASURED** (ours + path) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

---

## VERDICT — **QUALIFIED PASS**: adopt GeoCalib as the YouTube geometry front-end, with the
robust-aggregate + confidence-gated fallback that `geocalib_intrinsics.py` implements.

GeoCalib measurably beats the fixed-100° HFOV approximation on **rectilinear** dashcam frames
(the YouTube case), is resolution-robust, and degrades *safely* (falls back to the fixed HFOV)
where it is unreliable (true fisheye, self-inconsistent clips). It is **not** an exact oracle —
~7 % median focal error, systematic ~6 % under-estimate, per-frame outliers, weak absolute-focal
tracking — so it is used with multi-frame aggregation, never single-frame. See VALIDATION_REPORT.md.

## What was done (P1-P3 MEASURED; details in VALIDATION_REPORT.md)
- **P1** GeoCalib installed + running on the eval pod (`/workspace/geocalib_work/gcvenv`,
  system-site torch cu128; built from `cvg/GeoCalib`, weights from its GitHub release). Estimates
  focal + vFoV from one frame.
- **P2** Recovery on KNOWN GT (`geocalib_validation_results.json`, 36 frames): comma2k19 (GT focal
  910) **distorted** model → focal \|err\| median **6.8 %**, vFoV err +3.6°, **resolution-robust**
  (480p≈native). Weak on a controlled digital focal-sweep (r=0.41 — regresses to a ~50-55° prior).
  120° f-theta fisheye is out-of-model (pinhole fit ~37° vFoV) → correctly rejected downstream.
- **P3** `geocalib_intrinsics.py` — importable estimator + a **drop-in** for
  `yt_pilot_common.decode_canonical`; the only change is the per-video GeoCalib focal. MAD-robust
  over N frames, `confidence`/`fallback_used`, `EstimatedIntrinsics.focal_px(W,H)` feeds
  `focal_crop_resize` → `f_eff≈266`. MEASURED on real comma (vFoV 59.5°, high conf) + fisheye
  (low conf) video.

## P4 — GeoCalib vs fixed-100° HFOV on the pilot's REAL YouTube clips  [MEASURED]
`youtube_geocalib_measurement.json` — 12 of the pilot's 32 CC videos re-downloaded via the staged
pointers (no imagery persisted; mp4 deleted after estimation).
- Estimated HFOV **median 66.6° / mean 63.4°, range 32-77°** (confident-only median 60.5°). **Only
  1 of 12** clips is within 10° of the assumed **100°** → a single fixed HFOV cannot fit them;
  per-video estimation is warranted (core motivation confirmed on real data).
- Direction matches comma: real clips are NARROWER than 100° → the fixed assumption over-crops,
  applying a **~1.4× excess zoom** (in-pipeline, at 360p) → pseudo-speed inflated. GeoCalib removes it.
- GeoCalib self-flags the untrustworthy clips (3/12 low-confidence + 1 fallback).
- HONEST: unknown-GT (can't verify per-clip absolute accuracy; trust bounded by P2's ~7 % + the
  confidence gate). Estimates likely optimistically-narrow (prior-regression + 360p compression); the
  crop metric saturates at 360p (see VALIDATION_REPORT). Downstream ADE closure needs pod3 (escalation 3).

## Decode-path fix (found + fixed here) [MEASURED]
The estimate(CUDA)→decode-per-clip flow the GeoCalib integration introduces hit an **intermittent
PyAV deadlock**: a threaded decoder (`thread_type="AUTO"`) torn down while a CUDA context is live
hangs on container close. The pilot never saw it (its decode had no inline CUDA). **Fix:** decode
single-threaded (`DECODE_THREAD_TYPE="NONE"`, module constant) in both `estimate_from_video` and
`decode_canonical_geocalib` — decode of small frames is not the bottleneck. Verified across a repeated
multi-clip estimate+decode loop (no hang). **The parallel agent MUST use this module's decode path (or
`thread_type="NONE"`)** — calling GeoCalib inline with an "AUTO" decoder will deadlock.

## DELIVERABLE MANIFEST
| artifact | repo path (staged) | pod path |
|---|---|---|
| **`geocalib_intrinsics.py`** (the importable estimator + drop-in) | `repo:.../2026-07-25-geocalib/geocalib_intrinsics.py` | `/workspace/geocalib_work/geocalib_intrinsics.py` |
| VALIDATION_REPORT.md + this NOTE.md | `repo:.../2026-07-25-geocalib/{VALIDATION_REPORT,NOTE}.md` | — |
| P2 known-GT results | `repo:.../geocalib_validation_results.json` | `/workspace/geocalib_work/geocalib_validation_results.json` |
| P4 YouTube-vs-fixed results | `repo:.../youtube_geocalib_measurement.json` | `/workspace/geocalib_work/youtube_geocalib_measurement.json` |
| known-GT frame builder | `repo:.../prep_frames.py` | (frames: `/workspace/geocalib_work/gc_frames/`) |
| P2 validation runner | `repo:.../run_geocalib_validation.py` | `/workspace/geocalib_work/run_geocalib_validation.py` |
| P4 YouTube measurement | `repo:.../measure_youtube_geocalib.py` | `/workspace/geocalib_work/measure_youtube_geocalib.py` |
| module test + multi-clip decode verify | `repo:.../{test_geocalib_module,verify_fix}.py` | `/workspace/geocalib_work/` |

Nothing of value lives ONLY on the pod: the module, both result JSONs, the report and all scripts
are staged in the repo working tree. GeoCalib weights (cached on pod) + the frame set are
reproducible from `prep_frames.py` + the staged install steps.

## ESCALATIONS
1. **READY FOR THE PARALLEL NON-CC-SCALE AGENT.** `geocalib_intrinsics.py` is a clean importable
   drop-in for the pilot's geometry step. Integration recipe at its pseudo-label stage:
   ```python
   import sys; sys.path.insert(0, "<.../incoming/2026-07-25-geocalib>")
   from geocalib_intrinsics import GeoCalibEstimator, decode_canonical_geocalib
   est = GeoCalibEstimator()                      # distorted, cuda; reuse across clips
   frames_u8, meta = decode_canonical_geocalib(mp4, anonymizer, estimator=est)  # replaces decode_canonical
   ```
   Needs `pip install -e git+https://github.com/cvg/GeoCalib` in the pipeline venv + the tanitad
   stack on PYTHONPATH (for `focal_crop_resize`). `meta["intrinsics"]` records the per-clip estimate
   + `fallback_used` for the provenance record.
2. **Top validation gap:** the absolute rectilinear-focal claim rests on ONE real camera
   (comma2k19). Add ≥2 more published-intrinsics rectilinear cameras (KITTI/nuScenes/Waymo) before
   treating GeoCalib as decision-grade for a large harvest. The systematic ~6 % under-estimate is a
   candidate bias-correction only after more cameras confirm its sign/magnitude.
3. **Closing the downstream loop** (re-pretrain on GeoCalib-cropped YouTube vs fixed-HFOV, compare
   parity-val speed_r2) needs the IDM encoder + labeler that live on pod3 — a pod3-class job, out of
   this eval-pod agent's lane. P4 here quantifies the GEOMETRY change (decision-grade); the ADE
   closure is the pre-registered follow-up.
