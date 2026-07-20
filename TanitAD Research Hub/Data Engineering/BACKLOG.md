# Data Engineering — Experiment Backlog

Prioritized roadmap (D-020 §4). Each run: execute ≥1 item, report measured numbers, re-prioritize.

## P0 — FLEET DIRECTIVE 2026-07-17 (Sayed; supersedes prior P0 ordering; resource-mandated G-I)

Context: `Project Steering/FLEET_REVIEW_2026-07-17.md`. Review verdict on your work: D-016 R1
rectify (A), PandaSet loader (A), OWN_DATASET_PLAN (A), Lake Phase-A (A−, REAL not scaffolding —
byte-equivalence gate genuinely tested). All merged to the tip 2026-07-17. The strategic gap:
everything servable today is highway-heavy (74% straight — the exact enabling condition of the
ego-status shortcut pathology; nuScenes is 73.9%). The corpus-diversity gain is PENDING on you.

1. **ZOD ingest — the #1 unlock.** **LOADER DONE 2026-07-18** (intake `2026-07-18-zod-loader/`, 19✓):
   `kb_to_ftheta` (KB radius ≡ `FThetaIntrinsics.poly`, zero new geometry math) → CC-BY-SA SEPARATE shard;
   OxTS-heading yaw → `cosmos_drive.poses_to_signals`. **Falsifier ANSWERED — PASS** (grounded on the published
   120° spec, robust to real KB): **f_eff=266.0, observed_frac=1.00, drop_in=True** → geometrically UNBLOCKED,
   NO calib.py R1 needed (fisheye path suffices; narrow-40° witness falsifies at 0.34). **NOW P0 (remaining, both
   access-blocked → escalated):** (a) **ZOD ACCESS** — Sayed/orchestrator sign the CC-BY-SA agreement
   (`opendataset@zenseact.com`; HF is a code-loader, no plain download); (b) **real-bytes verify** —
   run `zod_pilot_jobcard.md` (pod3-idle/Colab T4, M-1.3): 5-drive `verify_real_clip` (drop-in on real per-drive
   KB, OxTS↔cam alignment, steer-ratio, A8 vs comma) + epcache precompute. Orchestrator: intake `zod.py` into
   `stack/` (additive, ready-loader).
2. **Run the lake FOR REAL: Cosmos + PandaSet ingestors at scale** (they're implemented, never
   run at scale) → publish **tanitad-own lake v0** (comma + cosmos + pandaset) to HF gated —
   kills the per-pod rebuild ritual. PandaSet real-bytes verification (`verify_real_clip`) rides
   pod3 when idle (M-1.4). Deliverable: lake catalog row counts + one pod pulling the lake
   instead of rebuilding.
3. **Curve-rebalance mix for flagship-v2:** **MEASURED 2026-07-18** (intake `2026-07-18-curve-rebalance/`,
   12✓): per-source D1-strata on 630 real eps — **comma 83.1% / PhysicalAI 56.0% / natural pool 63.9% straight**;
   the ~74% ≈ a comma 0.65-0.70 mix. Turn-weight recipe `β=s(1-t)/(t(1-s))` = 1.31→2.22 to hit 57.5%; primary
   lever is source-mix (shift comma→urban/ZOD/PandaSet, +10pp comma≈+2.7pp straight). **NOW (remaining):**
   (a) re-run on the ZOD/PandaSet epcaches once real-bytes land (quantify what each urban corpus contributes,
   as a delta vs comma); (b) **PROPOSE the sampler/mix change to Sayed** (D-018 tactics — ESCALATE, don't flip
   the live trainer): `WeightedRandomSampler` over `MixedWindowDataset` window strata OR a comma-downweight for
   flagship-v2. This is the data-side half of the REF-B curve failure (loss-side shipped in refbpatch).
4. **R1 rectify + rig-cy calib consolidation:** land `calib_r1.py` symbols into
   `stack/tanitad/data/calib.py`, flip PandaSet `_decode_frames` to `pinhole_rectify`, and run
   the THREE calib suites (test_calib, test_calib_r1, test_physicalai_rig) as ONE gate (they
   share symbols — the review flagged the consistency risk).

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
1. ~~**D-016 R1 pad-crop + undistort**~~ **DONE 2026-07-17** (intake `2026-07-17-d016-r1-pinhole-rectify/`, 9✓).
   Built `pinhole_rectify` (grid_sample rectify-to-canvas, Brown-Conrady undistort + pad) → PandaSet **467→266.0
   exact** (drop-in), 37.7% masked periphery, 109px k1 corrected; comma reference untouched (266.0/99.6%). New
   **`observed_frac ≥ ~0.5` ingest gate** (Udacity-like falsifies at 0.13). **NOW P0 (integration):** fold into
   `stack/tanitad/data/calib.py` + flip PandaSet `_canonicalize` (GeometryError → observed_frac guard) + carry
   `observed_frac` into the data card, then **verify ONE real PandaSet sequence on the HF mirror** (real-bytes
   drop-in — the last mile to `drop_in=True` on real bytes for G1). Fisheye (ZOD) already covered by `ftheta_*`.
1b. **ZOD pilot loader (owned real-urban #1, real-CAN #2).** Geometry unblocked (fisheye → existing
   `ftheta_undistort`; the rectify family is now proven end-to-end). Fetch 5 drives from the ZOD host, canonicalize
   via `ftheta_*`, contract-test, recover the camera-yaw offset (motion vs quaternion heading, the PandaSet method).
   License CC-BY-SA (owned public shard). Expected: `observed_frac`, A8 stat, drop-in PASS on real bytes.
1c. **`stats` uint8-safe (small fix, found 2026-07-17).** `frame_change_fraction`/`consequence_dominance_stats`
   silently mis-measure on uint8 epcache frames (a direct call gave ~0.74 vs the true 0.06). Auto-`to_float_frames`
   or assert dtype. Trivial intake; prevents future A8 mis-reads.
2. **WorldModel-Synth semantic-label index (NEW P0 — the usable-today value; pose gate CLOSED 2026-07-15 =
   pose-less).** Build a scenario/semantic table from the (tiny) `description/*.json` across all 5 families:
   per-clip `{family, weather, time_of_day, surface_type, region, qwen_caption}`. Serves BACKLOG P1 2d
   (semantic-label survey — comma is `follow`-starved) + SCENARIO_DATABASE mining (caption-searchable
   SC-02/05/06). Cheap (metadata only, no 8.3 TB pixels). Expected: a queryable index + per-family/weather/region
   counts. (Action-conditioned use of the pixels stays a Phase-1 IDM target, G2.)
3. **PandaSet real-bytes verification (`verify_real_clip`)** — after item 1 lands, fetch one sequence from the
   `georghess/pandaset` HF mirror (~44.5 GB zip; large-disk/pod job) → confirm world-frame planarity/axis order,
   camera↔pose timestamp alignment, and recover the constant camera-yaw offset (motion-heading vs `heading`
   quaternion). Expected: episode-contract PASS on real bytes + offset stability across sequences.
4. **SCENARIO_DATABASE data sourcing (joint duty, D-020 §5)** — SC-02/05/06 rows advanced 2026-07-15 (WorldModel
   pose-less video+caption sources merged into the DB). Next: fill SC-04 (stop-arm) + SC-11 (wrong-side) with
   CARLA recipes / Cosmos obstruction clips; download+verify one public sample each where public.

## P1

2e. **Probe "A global dataset of continuous urban dashcam driving" (arXiv 2604.01044, found 2026-07-18).**
   NEW urban continuous-dashcam corpus — directly serves the curve-rebalance duty (P0#3: off 74% straight).
   Probe: license class (CC / owned-tier vs YouTube-class copyright barrier), actions availability (ego-motion/CAN
   present, or video-only → IDM/H7), camera calibration/FOV (drop-in via `kb_to_ftheta`/`focal_crop_resize`?),
   scale, urban/night fraction. Web + HF, no GPU. Expected: landscape row + a go/no-go for a pilot loader.

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
- (2026-07-15) **WorldModel-Synth pose probe DONE** (was P0.1) → **POSE-LESS** on real bytes (captions+metadata,
  no pose/CAN). "Cosmos-mirror loader" for it is RETIRED; re-filed as semantic-index (new P0.2) + Phase-1 IDM.
- (2026-07-15) **PandaSet loader shipped** (intake `2026-07-15-pandaset-loader/`, 16✓) — CC-BY-4.0 owned core;
  pose/signal/contract path validated, geometry BLOCKED (→ new P0.1 D-016 R1).
- (2026-07-08) Cosmos loader integrated; verify_real_clip PASSED on real bytes (A8=0.109,
  60/60 pose pairing after base-id fix). Layout finding documented.
- (2026-07-08 loop) **Cosmos T=39 RESOLVED + chunk-pairing bug fixed**: videos = 121-frame
  chunks @30 Hz (container fps is a mux artifact), clips = 10 s/300 poses; chunk-1 videos
  (~half the extract) were getting chunk-0 actions → `_chunk_of` + pose offset + episode-id
  fix, 141 tests green. Cosmos cleared for the D-010 mix. See layout-finding note UPDATE 2.
