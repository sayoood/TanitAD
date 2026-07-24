# YouTube-IDM SCALE-UP — non-CC, decision-grade

Upgrade the directional 80-clip CC pilot (`…/2026-07-24-youtube-idm-pilot/`) to a
**decision-grade** result at scale. Sayed committed to **non-CC** licensing (2026-07-25),
removing the CC-scarcity ceiling that capped the pilot at 80 clips (from ~339 CC candidates).
Runs on **tanitad-pod3** (idle A40, fully provisioned — v1 encoder ckpt, parity caches,
cached parity latents; pod1 lacks this env and the ckpt is HF-403-blocked to move).

## What this delivers
- **P1** non-CC harvest extension (`harvest_scaleup.py`) — CC filter removed, discovery
  broadened to general forward-dashcam video; **all privacy preserved verbatim** from the pilot.
- **P2** a footprint-bounded, resumable **driver** (`run_scaleup.sh`) targeting ~500–1000 clips.
- **P3** pseudo-label with the frozen v1 encoder + multi-domain IDM head (reuses `pseudo_label.py`).
- **P4** decision-grade downstream lift, ≥4 seeds (reuses `run_youtube_pilot_downstream.py`),
  against the pre-registered bar in `PRE_REGISTRATION.md`.

## Layout
```
harvest_scaleup.py   P1/P2 — non-CC harvest (broadened discovery; privacy verbatim; GeoCalib-ready)
queries_noncc.txt    broadened forward-dashcam search queries (no CC restriction)
channels.txt         optional channel-uploads discovery (long continuous drives = high yield)
run_scaleup.sh       DRIVER — batched harvest -> dd-check -> encode+delete -> repeat -> P4 (>=4 seeds)
PRE_REGISTRATION.md  P4 bar, both outcomes committed BEFORE the read
geocalib_shim.py     GeoCalib integration contract + fixed-HFOV fallback documentation
NOTE.md              bank-as-you-go running log (yield, geometry, verdict)
pod_artifacts/       harvest manifest + pointers + pseudo-labels + downstream results (banked from pod3)
```
The pilot's `pseudo_label.py`, `run_youtube_pilot_downstream.py`, `yt_pilot_common.py` are reused
UNMODIFIED (copied into the pod scripts dir for self-containment; the privacy/geometry lib
`yt_pilot_common.py` is byte-identical to the pilot's).

## Privacy contract (MANDATORY — unchanged from the pilot, this is the responsible non-CC impl)
- face + license-plate + body Haar blur on the **FULL-RES** frame **BEFORE** the 256 downscale.
- source mp4 **DELETED** immediately after decode; clip frames deleted after they are encoded to
  latents. **No raw video, no full-res frame is ever persisted.** Persistent artifacts = latents
  (2048-d, non-imagery) + pseudo-labels (numbers) + URL/timestamp **pointers** to public videos
  ("ship pointers, never bytes" / OpenDV model). License recorded per pointer for auditability.
- if the privacy detector cannot load, harvest **RAISES** (refuse-to-store) → STOP + escalate.

## Geometry
Fixed-HFOV (100°) fallback unless the GeoCalib agent's `geocalib_intrinsics.json`
(`{video_id: {hfov_deg|focal_px}}`) is dropped at `/workspace/tmp/yt_scaleup/geocalib_intrinsics.json`;
the harvest auto-detects it. HFOV used is recorded per pointer, so a fixed-HFOV run is **re-runnable
with GeoCalib later by re-decoding from the pointers** (no re-harvest of new videos needed).

## Run (pod3, detached)
```bash
ssh tanitad-pod3 'PYTHONPATH=/workspace/TanitAD/stack TARGET=800 SEEDS=4 \
  nohup bash /workspace/tmp/yt_scaleup/scripts/run_scaleup.sh \
  > /workspace/tmp/yt_scaleup/run.log 2>&1 &'
```
Progress banks incrementally to `/workspace/tmp/yt_scaleup/results/` (`harvest_manifest.json`,
`pseudo_labels.json`, then `results_scaleup_downstream.json` + a `DONE` sentinel).
```
```
