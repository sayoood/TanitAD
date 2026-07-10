# INTAKE — WorldModel-Synthetic pose probe + video-only loader (2026-07-10, Data Engineering)

**Verdict (orchestrator writes here):** _pending triage_

## What
Two standalone modules for `PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios`
(OpenMDW-1.1, ≈264 k clips / 8.3 TB):
1. `probe_worldmodel_synth.py` — network-only gating probe: navigates the HF tree (not a full
   `list_repo_files` walk, which hangs on ~3.7 M paths), builds a per-clip field census, hunts for
   pose/action files, inspects one description JSON, emits a JSON verdict.
2. `tanitad_worldmodel_synth.py` — a **video-only** loader (`discover_clips` + metadata filters,
   `parse_description`, `build_episode`, `build_manifest`, CLIP-level `split_clips`,
   `WMSVideoDataset`, pod-only `verify_real_clip`).

## Why
Backlog **P0.1**: settle whether this corpus ships ego pose/actions (decides loader path). **Measured
answer: it does NOT** (tree probe: fields = `{video, description}` only, exts = `.mp4`/`.json` only, 0
pose hits; HF card confirms "no pose/trajectory/actions/steering/CAN"). So the "near-zero cosmos-mirror"
plan is dead; the corpus is IDM/H7-gated or video-only. See
`Research/2026-07-10-worldmodel-synthetic-pose-probe-and-idm-path.md`.

## Evidence (measured, this run, RTX-4060 host + web, $0)
- Probe: 15 clips across all 5 families → per-clip fields `{description:15, video:15}`; ext census
  `{.json:105, .mp4:105}`; **0 pose/action file hits**; description keys `{framerate, nb_frames,
  t2w_windows, metadata{weather,time_of_day,surface_type,region}}`; **verdict NO-POSE**.
- One real `front_wide.mp4`: 4K (3840×2160), 24 fps, 462 frames (19.25 s), 14.1 MB, A8 0.0248/0.0137.

## Honesty design (P8) — no fabricated actions
- `build_episode` fills `actions`/`poses` with a **NaN sentinel** (`ACTION_SOURCE="idm_pending"`),
  never zeros → any action-conditioned trainer fails loud (NaN loss).
- `CORPUS_META["actions"] is None` → `i7_task_identity` mismatches comma2k19/Cosmos →
  `MixedWindowDataset`/probe-fit **mechanically exclude** this corpus from the action-conditioned mix
  until IDM labels exist. Frame geometry (channels/size/f_eff_px) still matches → shared encoder for
  video-only pretraining.

## Tests run
`pytest tests -q` → **10 passed in 1.64 s** (no real bytes, no `av`, no network — decode injected,
description JSONs written in the real schema). Covers: description parse (real schema + missing-key
safety), discovery + family/weather/time-of-day filters, incomplete-clip skip, **episode-contract
(9-ch frames + NaN-sentinel actions/poses)**, stride=12 Hz, **I7 exclusion from action mix**,
frame-geometry task match, manifest, CLIP-level split disjointness.

## Proposed target in `stack/`
- `stack/tanitad/data/worldmodel_synth.py` (the loader), mirroring `cosmos_drive.py` placement.
- `stack/scripts/probe_worldmodel_synth.py` (the probe tool), or keep in Implementation as an ops tool.
- Add `stack/tests/test_worldmodel_synth.py`.
- **Do NOT** register it in the default D-010 training mix — it is video-only until an IDM head lands.

## Risk
Low. New module, zero stack files touched, 0 new deps, excluded from the action mix by construction.
`front_wide` HFOV is assumed nominal 120° (no calib in corpus) → focal-canon is nominal; `verify_real_clip`
is the pod-side check before any trained claim. Public-claimability of OpenMDW-1.1 is the open **D-022**
question (default: firewall held to comma2k19 + Cosmos).

## Rollback
Delete the module + test; nothing else depends on it (not wired into the mix).
