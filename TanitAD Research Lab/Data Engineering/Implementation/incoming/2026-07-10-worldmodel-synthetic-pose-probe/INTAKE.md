# INTAKE — WorldModel-Synthetic pose probe + video-only loader (2026-07-10, Data Engineering)

**Verdict (orchestrator writes here):** _pending triage_

> ✅ **RE-CONFIRMED STILL PENDING 2026-08-16 — but it is now VISIBLE and QUEUED, not lost.**
> The package was **salvaged by path** out of the dead `data-engineering-20260710` branch by the
> 2026-08-03 daily orchestrator sweep (commit `3a27899`, whose body lists it verbatim under
> *"SALVAGED BY PATH (4 packages, 3 branches)"* and records that the salvage *"SURFACED 8 more INTAKE
> packages … 7 of them un-adjudicated"*). So the reason there is no verdict is the adjudication
> backlog, not a lost artifact.
> ⚠️ **The loader is STILL NOT integrated:** `stack/tanitad/data/worldmodel_synth.py` is absent from
> the tip working tree AND `git log --all -- stack/tanitad/data/worldmodel_synth.py` returns **nothing**
> — it was never committed on any branch. The only copies are this package's
> `tanitad_worldmodel_synth.py` and the stale worktree mirror. Swept by the 2026-08-16 stale-blocker sweep.

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

> ⏹ **CLOSED 2026-08-16 — "until an IDM head lands" CLEARED. THE IDM HEAD LANDED, and it has been
> iterated to v4.** Evidence (MEASURED, tip working tree):
> - `stack/scripts/idm_head.py` — *"Supervised predictive NON-CAUSAL Inverse-Dynamics (IDM) head on a
>   FROZEN encoder"*, reading out `speed / yaw_rate / steer / long_accel` + the 2 s ego-frame
>   trajectory at horizons {5,10,15,20}. This is exactly the module this line was waiting for.
> - `stack/tanitad/models/dynamics_encoder.py:44` — *"Deployable substrate = camera-conditioned
>   encoder + readout + IDM head (sub-300M …)"*; `:299` wires *"(a) supervised metric IDM head (the
>   deployed dynamics readout)"*.
> - `stack/tanitad/eval/idm_families.py:1` — a four-family instrument built for it, which states the
>   purpose in one line: *"the IDM's whole purpose is to mint pseudo-labels for action-free video."*
>   That is this corpus's use case verbatim.
> - **Trained weights exist across four generations**, banked in-repo: `idm_head_v1.pt`
>   (`…/Architecture & Inference/…/2026-07-25-idm-youtube-validation/`), `idm_head_v3.pt`
>   (`…/2026-07-27-idm-v3/`), `idm_head_v4_steer.pt` and `idm_head_v4_steer_ens3.pt`
>   (`…/Benchmarks & Eval/…/2026-07-27-fleet-refill/` and `…/2026-07-27-fleet-sync-idm-steer/`).
>
> ⚠️ **CLEARED ≠ "just switch the mix on" — carry these two honesty conditions with it.**
> 1. **Label quality is channel-dependent and was over-claimed once already.** MEASURED
>    (`stack/tanitad/data/comma2k19.py:49-92`): the deployed head's `yaw_rate` R² is **+0.9035 on
>    PhysicalAI (bit-identical on re-measure, n_changed = 0)** but the comma2k19 figure **+0.3308 is
>    WITHDRAWN** — 2 of its 22 val episodes were bit-identical to its own training clips, and on the
>    20 content-clean episodes the deployed head reads **R² −0.746 (CI [−1.574, −0.177])**. A
>    *retrained* head (R0) reaches **+0.3038 (CI [+0.054, +0.479])**. ⇒ IDM-minted yaw is corpus-specific;
>    do not assume it transfers to this synthetic corpus without measuring on it.
> 2. **The mechanical exclusion described above still holds and is still correct.** `CORPUS_META["actions"]
>    is None` → `i7_task_identity` mismatch, so the corpus stays out of the action-conditioned mix
>    **until IDM labels are actually minted for it** — the head existing is necessary, not sufficient.
> Swept by the 2026-08-16 stale-blocker sweep.

## Risk
Low. New module, zero stack files touched, 0 new deps, excluded from the action mix by construction.
`front_wide` HFOV is assumed nominal 120° (no calib in corpus) → focal-canon is nominal; `verify_real_clip`
is the pod-side check before any trained claim. Public-claimability of OpenMDW-1.1 is the open **D-022**
question (default: firewall held to comma2k19 + Cosmos).

## Rollback
Delete the module + test; nothing else depends on it (not wired into the mix).
