# STATE — Data Engineering

LAST_RUN: 2026-07-11 (Tuesday agent, TWO work items) · branch `agent/data-engineering-20260711`
  (worktree-isolated, D-026; **rebased onto current main `a7563e0` this run** so the delta is a clean
  fast-forward — the focal commit + the pm semantic-survey commit sit directly on main's tip).
QUALITY: full (G-A…G-C, G-E, G-H, G-D1 met; G-D2 n/a — survey/spec, no new loader shipped this run).
  pm run: intake `2026-07-11-semantic-label-survey/` 10✓ standalone; L2D probed on **real HF bytes**;
  no stack file touched. am run: focal-invariance 8✓ (below). Ledger/DATASET_LANDSCAPE/KB/BACKLOG updated
  in-worktree on the rebased base (no shared-file race this run — quiescent at commit time).

## This run — pm (2026-07-11): semantic/strategic-label dataset survey (Sayed directive, REF-B review)
- **Sayed directive P1 #2d DONE** — comma2k19's `follow`-starved strategic supervision is **sourceable**.
  Ranked survey (DATASET_LANDSCAPE Tier 1.5): **L2D #1 (train)**, nuPlan #2, CoVLA #3, Bench2Drive #4,
  DriveLM #5, Talk2Car #6; AUTOPILOT-VQA → Benchmarks (D-028 seam); Intention-Drive (2512.12302) = WATCH.
- **MEASURED (G-H) — L2D taxonomy probe on real HF bytes** (`probe_l2d_taxonomy.py`, truststore, $0, ~3 min):
  `yaak-ai/L2D` **Apache-2.0 (public-claimable)**, 100k eps / **26.5 M frames** @10 fps, 6 surround cams +
  map; **4,219 distinct compositional nav commands** (96 % distance / 74 % speed-limit / 61 % road-class
  tokens) **co-registered with real ego actions** (`action.continuous`-3, `action.discrete`-2) + waypoints-10
  + rich scene state. vs comma's ~1 effective strategic class. → **recommended Phase-1 strategic-supervision
  ingest** (filtered slice). Caveats (P8): instructions map-templated (L1/L2 not L3); `front_left` needs
  D-016; `action[3]→[steer,accel]` sign/unit decode pending; 90 TB → stream, don't clone.
- **Increment (G-E):** `l2d_contract_map.py` (front-cam key, 3→2 action algebra, nav-command→class parser,
  label-entropy metric, contract-row assembler) + `tests/test_l2d_contract_map.py` **10✓ offline** on the
  measured schema + verbatim instructions. Note `2026-07-11-semantic-strategic-label-dataset-survey.md`.
- **Falsifier pre-registered** for the L2D recommendation (backlog P1 #2e): a 200-ep decode must show a
  sane `[steer,accel]` and `front_left` f_eff within ±25 % post-D-016, else L2D→EVAL-only, nuPlan→train pick.

## This run — am (2026-07-11): focal-invariance validation (see `2026-07-11-focal-invariance-*` note)
- **D-016 focal canonicalization VALIDATED on the trained encoder (G-H)** — first end-to-end evidence,
  not just arithmetic. Intake `incoming/2026-07-11-focal-invariance-validation/` (8✓). Controlled
  single-scene test on real Cosmos 120° clips + `ckpt_full.pt` (48 scenes, RTX 4060, 7.9 s, $0): the
  SAME scene at focal z·f0 canonicalized with **correct** per-camera intrinsics vs **wrong** (=f0) ones —
  relative latent drift correct/wrong: z1.25 **9.8×** (cos .999/.919), z1.5 **14.5×** (.999/.753), z2.0
  **14.9×** (.997/.600). Falsifier fails at every zoom ⇒ validated. **Wrong intrinsics cost ~10–15×
  encoder drift** → per-clip intrinsics is high-value; Y-pilot-50 Y1 focal-recovery bar set (±25 %→cos≥.92).
- **Hardening shipped:** `assert_effective_focal()` / `achieved_focal()` — fail-loud guard for cameras
  with no FOV headroom (`focal_crop_resize` silently clamps → f_eff drifts above F_REF, breaking the I7
  `f_eff_px=266` claim). Proposed target `stack/tanitad/data/calib.py` (additive; no signature change).
- **Joint SCENARIO duty (D-020 §5): SC-13 sourced + measured.** CAN-speed mine of all 188 local
  comma2k19 Chunk_1 segments → only **13 stop-go events / 12 segments (6 %)**, all low-speed (0 from
  >8 m/s). comma = abundant car-following but sparse *stopped-lead* → SC-13 **catalogued → data-sourced
  (partial)**; needs full 10-chunk mine or CARLA/synthetic braking recipes. Row updated + HANDOFF.
- Lit (2 searches): VLM3 backs the F_REF principle; DeltaCam / Latent-WAM / IDOL / DriveWAM external
  H7 support, no status change (P8). Note `2026-07-11-focal-invariance-validation-and-sc13-sourcing.md`.

## Run-start situational notes (for the next run / orchestrator)
- **Unmerged prior data-eng branch:** `agent/data-engineering-20260710` (`96d85eb`) holds the completed
  **P0.1 WorldModel-Synthetic pose probe = NO-POSE → video-only loader** intake — branched off an older
  main, not yet swept. **Not re-done this run.** Orchestrator: merge/triage it; then P0.1 is closed.
- **Pod trainer still running the 30k record run** (PID 200748) → pod off-limits; **P0.0 R1 top-up is
  blocked** until the run terminates. Do NOT touch the pod for agent work.

## Next (backlog, priority order)
0. **L2D loader (Phase-1 strategic supervision — the survey's recommendation, backlog P1 #2e):** cheap
   pre-loader experiment first (200-ep decode → action-sanity + front-cam f_eff + label-entropy gap),
   then `stack/tanitad/data/l2d.py` (Cosmos-mirror + D-016, filtered slice). HF stream + 4060, no pod.
1. **Land per-clip intrinsics** (promoted — this run measured that wrong focal = ~10–15× drift):
   PhysicalAI `calibration/` per-clip focal replaces the nominal 120° in `physicalai.py`; per-video
   focal recovery for YouTube (Y1). Wire `assert_effective_focal` into each loader's data-card path.
2. **R1 top-up to 2,000** (2 egomotion chunks on pod) — BLOCKED on the 30k run finishing; then camera
   fetch (extract-all-per-chunk rule) + epcache after the trainer frees the cgroup.
3. **SC-13 full-chunk mine** — extend the Chunk_1 stop-go mine to all 10 comma chunks (or CARLA
   lead-brake recipe) to reach the matched stopped-lead segment set the SC-13 falsifier needs.
4. **Y-pilot-50** (Sayed directive): now unblocked-in-principle by the validated focal path — 50 YouTube
   dashcams through Y0–Y2; Y1 acceptance bar = per-video focal recovery within ±25 % (cos≥.92 measured).
5. Zenseact ZOD pilot loader (real-CAN #2, H4 arm-B, EU/night).

## HANDOFF — shared-file rows (worktree-written; mirror verbatim if merge can't auto-resolve)

**SCENARIO_DATABASE.md — SC-13 Data sources + Status** (applied in worktree): stopped-lead subset
measured THIN in comma2k19 Chunk_1 (13 events/12 of 188 segments/6 %, all <5 m/s, 0 from >8 m/s) →
status **data-sourced (partial)**; next = full 10-chunk mine or CARLA/synthetic braking. (Excellence
scoreboard still groups SC-13 under `catalogued` — stale; split at next shared-file quiescence.)

**HYPOTHESIS_LEDGER.md change-log row:**
- 2026-07-11: Data Eng — H7 evidence (no status change, P8): **D-016 focal canonicalization validated
  end-to-end** on the trained encoder — correct per-camera intrinsics hold a scene's encoding cos≥0.997
  across a 25–100 % focal change vs cos 0.60 ignored (~10–15× drift). Grounds the H7 heterogeneous-video
  flywheel (VLM3 principle) and sets the Y-pilot-50 Y1 focal-recovery bar. See
  `Data Engineering/Research/2026-07-11-focal-invariance-validation-and-sc13-sourcing.md`.

**PROJECT_STATE.md §5 session-log row (newest):**
| 2026-07-11 (Tue) | Data Engineering agent | **D-016 focal canonicalization VALIDATED on the trained encoder** (G-H, intake `2026-07-11-focal-invariance-validation/` 8✓): controlled same-scene test on real Cosmos 120° clips + `ckpt_full.pt` (48 scenes, 4060, $0) — **correct** vs **wrong** intrinsics relative latent drift z1.25 **9.8×**/z1.5 **14.5×**/z2.0 **14.9×** (cos ≥0.997 correct vs 0.60 wrong) → first end-to-end evidence the focal transform delivers focal-invariance, not just arithmetic. **Wrong focal = ~10–15× drift** → per-clip intrinsics promoted; Y-pilot-50 Y1 bar = ±25 %→cos≥.92. Shipped `assert_effective_focal` fail-loud ingest guard. **Joint duty: SC-13 data-sourced (partial)** — comma2k19 Chunk_1 stopped-lead subset measured THIN (13 events/12 of 188 segs). Pod off-limits (30k run live) → P0.0 blocked; prior branch `…-20260710` (P0.1 pose-probe) unmerged, not re-done. | `.../Data Engineering/Research/2026-07-11-*.md`, `.../Implementation/incoming/2026-07-11-focal-invariance-validation/`, `DATASET_LANDSCAPE.md`, `KNOWLEDGE_BASE.md` |
