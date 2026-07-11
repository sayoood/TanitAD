# Data Engineering — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — next run

-2. **HF recipe-dataset `Sayood/tanitad-realmix` (Sayed 2026-07-11 night).** Versionable,
   improvable realmix WITHOUT shipping PhysicalAI-derived data (license doctrine: NEVER to HF,
   even privately). Contents: comma-by-reference (existing public dataset), r0_selection.parquet
   + build params + split seed + mix ratio, per-episode SHA256 of built episodes (verify-without-
   ship), data card + rebuild instruction (fetch-camera + build_pai_cache.py — users pass NVIDIA's
   license gate themselves; OpenDV/LAION pattern). v1 = 402-ep selection (tonight, PRIVATE until
   Sayed's explicit "public"); v2 = R1 2,000; future: Y-track, Cosmos. Owner: MVP loop tonight,
   then Data-Eng maintains versions.
-1. **Data-mix-as-recipe (Sayed 2026-07-11 night: "generate the same data mix without pod1").**
   Kill the single point of failure: (a) extract the episode MANIFEST (source clip IDs, config
   hash, counts, mix ratio) from pod1's comma+physicalai epcaches post-30k → commit as
   `stack/DATA_MANIFEST.json` (IDs only — no license/size issue); (b) `scripts/rebuild_cache.py`:
   any pod self-provisions the exact corpus from origin (comma via public HF tars — already
   works; physicalai via HF chunks + deterministic epcache build, verified by sampled episode
   checksums vs pod1). Feeds the paper's reproducibility statement. Trigger incident: throttled
   rsync off the training pod stalled the record run (2026-07-11 ~21:30) — caches must be
   rebuildable, not only copyable.

0. **R1 top-up to 2,000 (measured 2026-07-09: 1,926 reachable from cache, 74 short).** Fetch **2 more
   egomotion chunks** on the pod (~4 GB), re-run `physicalai_r1.py --target 2000` → clears 2,000. Then
   camera-fetch (≤32 chunks, ~64 GB, ~1 h pod) **extracting ALL gate-passing clips per chunk** (per-chunk
   bandwidth → 3.85× clips free); epcache AFTER the 30k trainer finishes (62 GB cgroup). Tool + R1 report
   already landed (intake `2026-07-09-physicalai-r1-selection/`). Expected: 2,000 urban clips, ~24 countries.
1. **WorldModel-Synthetic-Scenarios pose probe (was "license check" — license DONE 2026-07-09: OpenMDW-1.1
   ungated).** `huggingface_hub` file-listing on one clip's parquet set to confirm/deny an ego-pose field.
   Decides the loader path: near-zero cosmos-mirror (if poses) vs IDM/H7 or video-only (if none). Expected:
   yes/no pose verdict + one sample decoded. This is now the gating question, not the license.
2. **SCENARIO_DATABASE data sourcing (joint duty, D-020 §5)** — SC-02/05/06 rows advanced 2026-07-09 (see
   STATE HANDOFF; pending merge into the DB). Next: fill SC-04 (stop-arm) + SC-11 (wrong-side) with CARLA
   recipes / Cosmos obstruction clips; download+verify one public sample each where public.

## P1

2d. **Semantic/strategic-label dataset survey (Sayed directive 2026-07-11, from the REF-B review):**
   comma2k19 is highway-dominated — nav-command and target-behavior learning is signal-starved
   (REF-B's strategic layer trains on route-geometry pseudo-labels that are ~all `follow`). Survey +
   rank datasets with RICH semantic strategic/behavior labels for Phase-1 strategic/tactical
   training AND richer pseudo-label validation: **nuPlan** (route/mission goals, closed-loop sim),
   **DriveLM** (graph-QA on nuScenes/CARLA), **CoVLA** (language-annotated trajectories),
   **L2D / Learning-to-Drive** (nav-instruction driving), **Talk2Car / nuScenes annotations**,
   **AUTOPILOT-VQA** (behavior taxonomy, see Benchmarks P1.0), **Bench2Drive** (CARLA commands).
   Per dataset: license class (train / eval-only / no), label taxonomy depth (nav command? maneuver?
   intention? free text?), camera/calibration compatibility, size, ingest cost. Output: ranked-list
   rows + a recommendation for ONE Phase-1 ingest. Resource: web + HF API, no GPU.

2c. **Y-pilot-50 (Sayed directive 2026-07-09): YouTube dashcam pilot** — 50 diverse videos through
   the full Y0-Y2 pipeline (self-calibration -> canonicalize -> filter -> pseudo-actions); measure
   focal-recovery spread, action agreement r vs VO, A8, probe-fit vs comma. Strategy + falsifiers:
   `Research/YOUTUBE_DASHCAM_STRATEGY.md`. Go/no-go for the Phase-1 OpenDV-2K ingest.


2b. **NuRec feasibility probe (Sayed suggestion 2026-07-09, scenario-data doctrine):** what NuRec
   (neural-reconstruction) assets exist in the PhysicalAI-AV HF family (NuRec/NCore variants),
   their format (Gaussian-splat? USD?), license class, and whether reconstructed scenes can be
   perturbed (insert cones/occluders) with available tooling. Deliverable: yes/no adoption note +
   one downloaded sample inspected. Feeds the Phase-1 targeted-generation WP (Cosmos-conditioned
   scenario sweeps + NuRec real-geometry eval scenes).

3. **Zenseact ZOD pilot loader** — real-CAN corpus #2 (EU/night distribution comma lacks);
   ~3–4 h loader cost, contract tests on 5 drives; feeds H4 arm-B diversity. Expected: CORPUS_META
   fingerprint byte-compatible; A8 consequence stat reported. License: research/NC — tag like
   physicalai until reviewed.
4. **PhysicalAI-WorldModel-Synthetic-Scenarios license + pilot** — verify HF card; if claimable,
   mirror the cosmos loader (shares pose/contract code, near-zero cost). This is H6/H15/D9
   long-tail material (emergency/lanechange/pedestrian/weather_degradation).
5. **`data:physicalai` tag audit** — grep ledger/leaderboard/paper for untagged PhysicalAI-AV
   numbers; one-command audit script committed to Implementation/. Expected: 0 violations.
6. **comma2k19 Chunk_2–10 streaming plan** — measure per-chunk ingest cost (curl + sanitizing
   extractor + epcache) on one additional chunk; extrapolate to full 33 h for the Phase-0 close.

## P2

7. **BDD100K + IDM pseudo-labels** (H7 proof-of-concept, Phase 1) — needs trained inv-dyn head;
   design note + 100-clip pilot once 30k checkpoint lands.
8. **Monthly HF `datasets` sweep** for new 2026 AV video releases (standing duty, D-012).
9. **OpenDV-2K heterogeneous-focal canonicalization** (H7 flywheel, Phase 1) — depends on D-016
   robustness beyond two corpora.

## Done / retired
- (2026-07-08) Cosmos loader integrated; verify_real_clip PASSED on real bytes (A8=0.109,
  60/60 pose pairing after base-id fix). Layout finding documented.
- (2026-07-08 loop) **Cosmos T=39 RESOLVED + chunk-pairing bug fixed**: videos = 121-frame
  chunks @30 Hz (container fps is a mux artifact), clips = 10 s/300 poses; chunk-1 videos
  (~half the extract) were getting chunk-0 actions → `_chunk_of` + pose offset + episode-id
  fix, 141 tests green. Cosmos cleared for the D-010 mix. See layout-finding note UPDATE 2.
