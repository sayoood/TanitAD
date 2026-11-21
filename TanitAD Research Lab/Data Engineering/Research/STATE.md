# STATE — Data Engineering

LAST_RUN: 2026-08-02 (Tuesday agent) — branch `agent/data-engineering-20260802` (worktree `C:/Users/Admin/wt-de-0802`, D-026). Executed `Project Steering/BACKLOG.md` **A3** (C64 option B) end-to-end: column semantics → feasibility re-measurement → selector → frozen candidate manifests.
QUALITY: full (G-A…G-E, G-H, G-D1 n/a — no new corpus row, G-D2 satisfied by the contract+corruption tests, G-I declared). One intake pkg, **27 ✓ standalone**, 3 measured experiments, no `stack/` file touched.
RESOURCE (G-I): dev-box **CPU only**, ~2.6 h session / **6.1 s** for the measurement itself, **$0**. No GPU, no pod, no network. Why not the eval pod: the question is a property of two parquet files — no model, checkpoint or GPU is in the path, and the eval pod's value this week is the arms it is scoring.

## This run (2026-08-02): A3 — the clean v2-line val, unblocked and re-scoped by what it measured

- **The blocker is gone.** `pool_columns.py` turns `v2_pool_scored.parquet`'s semantics into a
  machine-checked contract — **34/34 checks PASS on 18,988 real rows**, every identity corruption-tested.
  `lk…bs` are frame COUNTS out of `nlab` (`lk_rate` train **0.4500** vs remainder 0.6718 — reproduces the
  design's "lane_keep 45.0 %"); `stopped/city/hw` are FRACTIONS with `stopped+city+hw ≡ 1`. `nlab ≡ 179`
  everywhere ⇒ the `lk` misread was a pure scale error, rank-preserving.
- ⛔ **A3's "6.77× ⇒ FEASIBLE" was one axis wide.** Headroom at n=600: junction **6.77×** → +has_turn 4.05×
  → +speed **1.07×** → +has_brake **0.77× INFEASIBLE**. Headroom belongs to its axis set, like an exponent
  to its window. And the cell-quota design it justified leaves **max |d| 0.3997 (10/13 axes over bar)**;
  greedy balancing reaches **0.0094**, the shipped hybrid **0.0532** with the cell match intact (L1 0.0047).
- ⚠️ **No draw from this remainder is exchangeable with train** — within each matched cell the residue is
  still skewed (median |d| **0.359**, p90 0.915, max 2.351); max KS 0.12–0.19 vs a 0.057 critical value.
  Mean-balanced ≠ distributionally matched, and this limit travels with any v2-line number.
- ⛔ **THE FINDING THAT CHANGED THE DELIVERABLE: a clean v2 val is not clean for v1.** A v2-only-clean
  600-draw holds **62 clips of v1's TRAIN** (+24 of v1's val) ⇒ scoring v1 there scores it on its own
  training data. Manifests now exclude the whole parity selection; parity-free remainder **8,298** ⇒
  **600 is unavailable** (headroom 0.95). **Shipped: n=400** (max |d| 0.0409, census 69.4 %, sha256
  `abe041db72a045b3…`) + n=300 variant.
- **Partly answers backlog A5 with no pod:** at CLIP granularity **256 of v1's 600 parity-val clips
  (42.7 %)** sit inside v2corpus's training selection ⚠️ **not** the same statistic as C64's 21/40 EVAL
  EPISODES — compare within a granularity, never across.
- **Absence, four probes:** PhysicalAI-AV ships **no session/drive id and no absolute clock**
  (`egomotion.timestamp` is clip-local µs) ⇒ the L2D time-overlap dedup cannot be run and **clip
  granularity is the provable ceiling** for every PhysicalAI split.
- **Provenance correction:** the pool has **18,988 rows / 18,987 unique clips** — `32ad1a3a-…` is registered
  under chunks 1573 AND 3117 and is in the v2 selection. Both figures were in circulation, unlabelled.
- Intake: `Implementation/incoming/2026-08-02-v2-clean-val-selector/` (`pool_columns.py`,
  `clean_val_select.py`, `make_results.py`, `tests/` 27 ✓, `selector_comparison.json`, 2 manifests, INTAKE).
- Note: `2026-08-02-v2-clean-val-semantics-and-selector.md`.

### ESCALATIONS (this run)
1. ⭐ **PI (C64 option B):** the split is **400 clips, not 600**, and the reason is contamination, not taste.
   Freeze `v2_clean_val_manifest.json` / the n=300 variant — or reject option B on the exchangeability limit.
2. **Orchestrator:** intake `pool_columns.py` → `stack/tanitad/data/`, `clean_val_select.py` →
   `stack/scripts/` (additive, 27 standalone tests) so every future consumer of the pool validates first.
3. **`Project Steering/BACKLOG.md` A3** carried the 6.77× figure; corrected on this branch.

### D-026 session guard (G-F) — `RESULT: PASS`, and the debt it surfaces
- Run from the worktree root against tip `c6c1701`: **PASS** (nothing stranded uncommitted).
- ⚠️ **WARN: 11 unmerged `agent/*` branches** vs tip — `agent/opponent-20260802` (+5),
  `agent/benchmarks-eval-20260802` (+3), `agent/phase0-highway-dataset` (+3),
  `agent/data-engineering-20260711` (+2), `agent/opponent-20260721` (+2),
  `agent/pod-code-intake-20260720` (+2), `agent/prod-opt-20260711` (+2), plus four at +1
  (`data-engineering-20260710`, `opponent-20260715`, `opponent-20260720`, `tools-devenv-20260721`).
  **Two of my own discipline's branches are in there and predate this run by three weeks.**
- ⚠️ **WARN: 26 INTAKE packages carry no orchestrator verdict**, oldest 25 days — the same
  unfixed backlog PROJECT_STATE flagged as "19 of 23" on 07-20; it has grown, not shrunk.
  Four are mine (`2026-07-15-pandaset-loader`, `2026-07-17-d016-r1-pinhole-rectify`,
  `2026-07-18-curve-rebalance`, `2026-07-18-zod-loader`) plus the two VLM packages.
  ⇒ **orchestrator escalation**, not a note nobody re-reads.

### Fleet context
- **No Monday output exists this week** (last `tools-devenv` note 2026-07-21): the weekly-agent cadence has
  been superseded by the autonomous main loop since ~07-24. Consumed `LOOP_STATE.md`, C64/C65 and the
  DE-touching commits (`fe400f0`, `2b7fe3f`, `82692f2`, `af78e86`) instead.
- **This STATE was 15 days stale** (LAST_RUN 2026-07-18) while the DE work it should describe — TanitDataSet,
  the L2D adapter, the lead-state gate, the v2 corpus — landed through the main loop. Naming the gap rather
  than back-filling it: those runs are recorded in their own notes and commits.

## Prior run (2026-07-18, Tuesday agent) — branch `agent/data-engineering-20260718` (worktree `C:/Users/Admin/wt-de-0718`, D-026). TWO increments: (1) ZOD ingest [P0#1] AM; (2) curve-rebalance measured [P0#3] PM.
QUALITY: full (G-A…G-E, G-H, G-D1, G-D2, G-I met; 2 intake pkgs 19✓ + 12✓ standalone; 2 measured experiments [ZOD geometry falsifier + curve-rebalance on 630 real eps]; no `stack/` files touched).
RESOURCE (G-I): local RTX-4060 dev box only, ~2.4 h total, $0. Both experiments are pure-CPU (closed-form geometry + epcache pose reads) — no model/GPU/network needed to answer them; the GPU halves (ZOD 5-drive precompute) are ACCESS-blocked → shipped as a job card (M-3). Why not eval-pod/Colab: neither question needs a model or GPU; the compute-blocked real-bytes work is escalated, not skipped.

### 2026-07-18 increment 2 (PM): curve-rebalance MEASURED [FLEET P0#3]
- **The data-side attack on the #1 risk, on real bytes.** Analyzer `curve_rebalance.py` (intake
  `2026-07-18-curve-rebalance/`, 12✓) measures the D1 curvature strata (`|net yaw@2s|` <5°/5-20°/>20°, copied
  verbatim from `driving_diagnostic.py` + drift-guard test) per source and derives a turn-weighted sampling recipe.
- **MEASURED (630 eps / 125,247 windows, local epcache, $0, ~1.5 min):** **comma2k19 83.1% straight**,
  **PhysicalAI 56.0%**, natural pool **63.9%**. **KEY: the "74% straight" is a comma/HIGHWAY property** — urban
  (PhysicalAI, and incoming ZOD/PandaSet) is already at the 55-60% target. The fleet's ~74% ≈ a comma 0.65-0.70 mix.
- **Recipe:** two quantified levers to 57.5% — source-mix (+10pp comma ≈ +2.7pp straight → shift to urban = the
  ZOD rationale in numbers) and window turn-weight β = s(1-t)/(t(1-s)) = **1.31**(pool)→2.22(comma-0.70), verified
  by construction. Sampler/mix wiring is a training-recipe change → **D-018 ESCALATE** (proposal, not a live flip).
- **Lit delta (SEARCH):** IDM/latent-action surge — Sensorimotor-WM (2606.20104, IDM-regularizer = pose-less
  pseudo-label recipe), ACID (2607.02403, action-cycle-consistency = pseudo-label quality gate), X-Lens (2607.12993,
  intrinsic-canon → D-016/H17), "What Do LAMs Learn?" (2506.15691, caution); new datasets 2604.01044 / DrivingGen /
  ScenePilot-4K → Benchmarks seam. No status change (P8).
- Note: `2026-07-18-curve-rebalance-measured-and-idm-lit.md`.

### 2026-07-18 increment 1 (AM): ZOD ingest
- **ZOD loader SHIPPED — the FLEET_REVIEW P0#1 / OWN_DATASET_PLAN §7#1 unlock** (intake `2026-07-18-zod-loader/`,
  `zod.py` + 19✓ + job card + INTAKE). CC-BY-SA-4.0 owned real-urban: 14 EU countries, day/night/seasons/weather,
  **real CAN steer + OxTS RT3000 ego-motion** — the diversity the 74%-straight day-only mix lacks.
- **Pre-registered geometry falsifier ANSWERED — PASS.** ZOD front = KB fisheye (3848×2168, **120° HFOV**);
  measured (grounded on the published spec, robust to real KB coeffs): **f_eff=266.0, observed_frac=1.00,
  drop_in=True**. ZOD is **geometrically UNBLOCKED** (no calib.py R1 needed — the fisheye path suffices; contrast
  PandaSet height-bound at 467). Narrow-40° witness falsifies at 0.34 → gate not vacuous. **No escalation on
  geometry** (falsifier did not trip).
- **Key reuse result:** Kannala-Brandt radius ≡ `FThetaIntrinsics.poly` (odd powers) → `kb_to_ftheta` reuses the
  proven crop path with ZERO new geometry math (confirms OWN_DATASET_PLAN's "fisheye→ftheta_*" with numbers).
- **OxTS heading drives yaw** (offset-free, defined at standstill) — cleaner than PandaSet's motion-heading
  fallback; `zod_signals` reuses tested `cosmos_drive.poses_to_signals`; CAN steer is a cross-check via
  `can_steer_ratio` (recovered on real bytes).
- **ESCALATION (Sayed/orchestrator): request ZOD access** (`opendataset@zenseact.com`, CC-BY-SA-4.0 + privacy/
  no-military; HF repo is a code-loader, no plain download). The ONE blocker on the #1 owned ingest. Accept
  CC-BY-SA for a *separate public ZOD shard* = OWN_DATASET_PLAN §9 open-Q #1 (recommend: accept).
- **Lit/seam:** WorldLens (CVPR-26 Oral) + DrivingGen driving-WM benchmarks → Benchmarks&Eval seam; NEW urban
  dashcam corpus arXiv 2604.01044 → curve-rebalance probe queued (P1). Red-suite (Monday's #1) already resolved —
  suite collects 391 tests, `calib.py` ships the two-rig symbols.
- Note: `2026-07-18-zod-loader-and-geometry-falsifier.md`.

## Next (backlog, priority order)
1. **ZOD real-bytes verification** — on ZOD access (escalated), run `zod_pilot_jobcard.md` on pod3-idle/Colab:
   5-drive `verify_real_clip` → confirm drop-in on the REAL per-drive KB, OxTS↔camera timestamp alignment,
   steer-ratio recovery, A8 vs comma; epcache precompute → feeds the lake (P0#2).
2. **Run the lake at scale** (BACKLOG P0#2): Cosmos + PandaSet ingestors at scale → publish `tanitad-own` lake v0
   (comma + cosmos + pandaset) to HF gated; PandaSet real-bytes verify rides pod3 when idle.
3. **Curve-rebalance** (BACKLOG P0#3) **MEASURED this run** — remaining: (a) re-run on ZOD/PandaSet epcaches
   once real bytes land (per-urban-corpus contribution vs comma); (b) **ESCALATE the sampler/mix proposal to
   Sayed** (D-018 tactics) for flagship-v2. **Probe arXiv 2604.01044** (global urban dashcam: license + actions)
   still open (P1).
4. **calib_r1 consolidation** (BACKLOG P0#4) — PandaSet-only (ZOD needs none); land `pinhole_rectify` into
   `stack/tanitad/data/calib.py`, run the 3 calib suites as one gate. (MVP-orchestrator task — stack/ edit.)
5. **`stats` uint8-safe** (small fix, 2026-07-17) — auto-`to_float_frames`/assert dtype.

## Prior run (2026-07-17)
- **D-016 R1 pinhole rectify BUILT + validated — the owned real-urban BLOCKER (last run's #1) is RESOLVED for the
  pinhole family.** New primitive `pinhole_rectify` (grid_sample rectify-to-canvas, Brown-Conrady undistort + pad;
  mirrors the existing fisheye `ftheta_undistort`) lands `f_eff=266` **exactly by construction**. Intake pkg
  `2026-07-17-d016-r1-pinhole-rectify/` (`calib_r1.py` + tests + report + INTAKE), **9/9 tests ✓ standalone**.
  Measured on grounded real intrinsics ($0, CPU): **PandaSet front 467→266.0** (BLOCKED→drop-in), cost **37.7%
  masked periphery** (native VFOV 30.7° < canonical 51.4°; sky/hood unobserved, road band kept) + **109px k1
  distortion corrected**; comma2k19 reference untouched (266.0, 99.6% observed). **New ingest rule: gate every
  source on `observed_frac ≥ ~0.5`** — Udacity-like falsifies at 0.13 (narrow FOV = 87% mask). Undistort
  correctness: fwd↔iterative-inverse <1e-4 + checkerboard recovery corr>0.9; contract-drop-in (G-D2).
- **Secondary (real bytes): A8 on 12 comma-val eps (3,600 frames) = 0.0596@0.05 / 0.0240@0.10**, reproduces the
  2026-07-07 baseline. Found a harness pitfall: `stats` needs float, epcache is uint8 (direct uint8 call → bogus
  ~0.74). BACKLOG: make `stats` uint8-safe.
- **Coverage map now complete:** pinhole (PandaSet/Udacity/comma) → `pinhole_rectify` (this); fisheye (ZOD
  Kannala-Brandt / PhysicalAI / Cosmos f-theta) → existing `ftheta_*`. Every owned real-urban source has a rectify
  path → OWN_DATASET_PLAN §0 "one owned dataset, real episodes" is geometrically unblocked (GOAL G1 movement).
- **Housekeeping:** committed the untracked `OWN_DATASET_PLAN.md` (2026-07-13 plan v1, was sitting untracked in
  the shared main tree) into this branch.
- Note: `2026-07-17-d016-r1-pinhole-rectify-unblocks-owned-real-urban.md`.

## Next (backlog, priority order)
1. **MVP integration of the R1 rectify** — fold `calib_r1.py` symbols into `stack/tanitad/data/calib.py`; flip
   PandaSet `_canonicalize` to `pinhole_rectify` + carry `observed_frac` into the data card (GeometryError →
   `observed_frac<floor` guard). Then **verify one real PandaSet sequence on the HF mirror** (real-bytes drop-in).
2. **ZOD pilot loader** (real-CAN #2, owned candidate #1) — fisheye via existing `ftheta_undistort`; the rectify
   family is now proven, so ZOD is unblocked on geometry. Fetch+verify one real drive; recover camera-yaw offset.
3. **`stats` uint8-safe** (small fix) — auto-`to_float_frames` or assert dtype so no future A8 mis-measure.
4. **WorldModel-Synth semantic-label index** (pose-less; captions+metadata are the usable-today value).
5. **R1 top-up to 2,000** (1,926 reachable, 74 short) — pod job, BLOCKED (pod busy: 3-arm bake-off training).

## Prior run (2026-07-15)
- **WorldModel-Synthetic-Scenarios pose gate CLOSED (BACKLOG P0.1) — POSE-LESS, measured on real bytes.** HF
  tree walk + real-clip fetch: each clip = `<family>/<clip>/{description,video}`; `video/` = **7 camera mp4s**
  (front_wide/front_tele/3 fisheyes/rear L/R) @24 fps ~462 frames; `description/<cam>.json` = a **Qwen2.5-7B
  caption + `{weather,time_of_day,surface_type,region}`**, NOT a pose. No vehicle_pose/CAN/trajectory. →
  "near-zero cosmos-mirror" assumption DEAD; the pixels are a Phase-1 IDM (H7) target, the **captions+metadata
  are a usable-today semantic-label index** (BACKLOG P1 2d + SC-02/05/06 data rows, merged into SCENARIO_DATABASE
  this run).
- **PandaSet loader shipped (intake `2026-07-15-pandaset-loader/`, 16 tests ✓) + a grounded D-016 GEOMETRY
  BLOCKER.** CC-BY-4.0 owned-core real-urban adapter (OWN_DATASET_PLAN §7 #2); reuses cosmos geometry
  (motion-heading 4×4 → `poses_to_signals`), I7≡comma2k19, I3 seq-split; schema grounded from the pandaset-devkit
  source. **Blocked-by-design:** on the REAL front calib (arXiv 2112.12610: fx=1970.01, 1920×1080, k1=−0.589),
  the centered square-crop is **height-bound** (ideal crop 1896 px > 1080 frame height) → lands **f_eff=467 px vs
  canonical 266** (~1.75× scale mismatch) + ignores k1=−0.589 distortion. Loader **fails loud**
  (`GeometryError`) so it can't pollute the mix. **Rule: any fx>1122 px on a 1080-tall frame is not
  square-croppable to 266.**
- **Key strategic finding:** the D-016 R1 **pad-crop + undistort** (same mechanism as the in-flight two-rig cy
  fix) is now a **blocking prerequisite for the entire owned real-urban tier** (ZOD fisheye + Udacity narrow-FOV
  hit the same bound), promoted from "deferred R1 nicety" → P0 blocker with numbers.
- **Landscape/lit (D-012/D-013):** new `nvidia/…Cosmos-Synthetic` (card-only, watch); `Newsflare` AV = stock
  copyright barrier (excluded); no new ungated real-AV video → owned gap stays ZOD-shaped. IDM/latent-action
  literature dense (2601.05230, 2602.16229, LatentVLA, FLAM) → frozen-encoder IDM+WM is the recipe to make
  pose-less corpora trainable via the comma/ZOD real-CAN bridge. ZOD license **corrected** research/NC → CC-BY-SA.
- Note: `2026-07-15-worldmodel-pose-gate-and-pandaset-geometry.md`. GOALS.md created (G1 owned-tier unblock /
  G2 IDM loop / G3 D1 data-side).

## Next (backlog, priority order)
1. **D-016 R1 pad-crop + undistort intake** (BACKLOG P0.1) — the owned real-urban BLOCKER; unblocks PandaSet + ZOD.
2. **WorldModel-Synth semantic-label index** (BACKLOG P0.2) — build the queryable `{family,weather,tod,region,
   caption}` table (cheap, metadata-only) for P1 2d + SC mining.
3. **PandaSet real-bytes verification** (`verify_real_clip`) once R1 pad-crop lands — one real sequence from the
   HF mirror; recover the camera-yaw offset.
4. **R1 top-up to 2,000** (1,926 reachable, 74 short) — pod job, BLOCKED (pod busy: `refb-speed-30k` training).
5. ZOD pilot loader (fisheye→ftheta + R1 pad-crop) — real-CAN #2 / owned candidate #1 (H4 arm-B, EU/night).

## Notes
- Pod OFF-LIMITS this run (`refb-speed-30k` training active per PROJECT_STATE 2026-07-15). All work local/HF, $0.
- Worktree isolation (D-026): committed on `agent/data-engineering-20260715`; MVP orchestrator merges. Shared-file
  edits (KB, DATASET_LANDSCAPE, HYPOTHESIS_LEDGER, SCENARIO_DATABASE, PROJECT_STATE §5) made on the branch — no
  main-tree clobber risk (the reason the worktree exists).
