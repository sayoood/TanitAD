# STATE — Data Engineering

LAST_RUN: 2026-07-10 (Tuesday agent) — worktree branch `agent/data-engineering-20260710` (D-026).
QUALITY: full (G-A…G-C, G-E, G-H, G-D1, G-D2 met; measured experiment with numbers; intake pkg
  **10✓ standalone**, no stack files touched). Shared-file rows (KB, DATASET_LANDSCAPE, HYPOTHESIS_LEDGER,
  SCENARIO_DATABASE, PROJECT_STATE §5, DECISIONS D-022) **applied in-worktree** — D-026 worktree isolation
  removes the concurrent-write race that forced last run's deferral; the orchestrator merges the branch.
  Last run's deferred HANDOFF rows (R1 selection + D-022) were **not** merged to main → re-applied this run.

## This run (2026-07-10)
- **WorldModel-Synthetic-Scenarios pose probe = NO-POSE (backlog P0.1 CLOSED, measured, negative).**
  Tree probe (`probe_worldmodel_synth.py`, 15 clips × 5 families, 18.9 s): each clip = **only**
  `video/` + `description/`; exts **only `.mp4`/`.json`**; **0 pose/action/calib files**; description
  keys `{framerate, nb_frames, t2w_windows(caption), metadata{weather,time_of_day,surface_type,region}}`.
  **HF card confirms** "no ego pose/trajectory/actions/steering/CAN", OpenMDW-1.1, no companion action set.
  → the "cosmos-mirror" assumption is **falsified**; corpus is **IDM/H7-gated or video-only**, EXCLUDED
  from the action-conditioned D-010 mix.
- **Real bytes:** one `front_wide.mp4` = **4K (3840×2160), 24 fps (real, not mux artifact), 462 fr /
  19.25 s, 14 MB**, A8 0.0248/0.0137 (emergency/night stop clip). Frames real, clean-decoding.
- **Increment (G-D2/G-E):** WMS **video-only loader** (intake `2026-07-10-worldmodel-synthetic-pose-probe/`,
  **10 tests ✓**): front_wide → D-016 focal-canon → D-015 9-ch; **actions/poses = NaN sentinel**
  (`ACTION_SOURCE="idm_pending"`, no fabrication P8); `CORPUS_META["actions"]=None` → I7 mismatch →
  mechanical exclusion from the action mix; + `build_manifest` (scene index for the scenario duty).
- **Lit (D-013):** IDM errors accumulate at distribution edges (VPT/survey) → highway-IDM least reliable
  on WMS's long-tail; DriveWAM (2605.28544) = video-generative-prior WAM (graduation trigger);
  Latent-WAM (2603.24581); `nvidia/omni-dreams-models` now public (Phase-1 sim watch).
- **Joint duty:** SC-02/05/06 rows updated — WMS = perception/OOD/video-only VALIDATION scenes, NOT
  closed-loop telemetry (no ego actions).
- Note: `2026-07-10-worldmodel-synthetic-pose-probe-and-idm-path.md`.

## Next (backlog, priority order)
1. **R1 top-up (2 chunks) on pod** → clear 2,000; then camera-fetch (≤32 chunks, ~64 GB, ~1 h) extracting
   ALL gate-passing clips per chunk; build epcache AFTER the 30k trainer finishes (cgroup). *(pod-gated:
   training has priority; check `nvidia-smi` before touching the pod.)*
2. **WMS video-only pilot** — front_wide epcache on ~200 `weather_degradation` clips as a **never-trained
   D8 OOD visual probe** (complements Cosmos weather pairs for SC-05). ~2–3 h, front_wide only (~3 GB).
3. **IDM edge-reliability guard (H7)** — before any WMS/BDD pseudo-labeling, measure the trained IDM's
   steer/accel error on **real long-tail actions (Zenseact ZOD, EU/night CAN)** vs comma highway.
   Needs the post-30k IDM head; design note now, run when it lands.
4. **Loader `selection_path` param** for `physicalai.py` (load `r1_selection.parquet` w/o R0 provenance).
5. Zenseact ZOD pilot loader (real-CAN #2, EU/night — also the IDM validator for #3); steering-ratio
   calibration log on real Chunk_1; A8 harness on real Chunk_1.

## Open decisions (Sayed)
- **D-022 (proposed):** widen public firewall to WMS (OpenMDW-1.1). Default HOLD (firewall stays
  comma2k19 + Cosmos-DD). Nothing blocked by holding.
