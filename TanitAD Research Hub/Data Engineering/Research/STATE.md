# STATE — Data Engineering

LAST_RUN: 2026-07-18 (Tuesday agent) — branch `agent/data-engineering-20260718` (worktree `C:/Users/Admin/wt-de-0718`, D-026). TWO increments: (1) ZOD ingest [P0#1] AM; (2) curve-rebalance measured [P0#3] PM.
QUALITY: full (G-A…G-E, G-H, G-D1, G-D2, G-I met; 2 intake pkgs 19✓ + 12✓ standalone; 2 measured experiments [ZOD geometry falsifier + curve-rebalance on 630 real eps]; no `stack/` files touched).
RESOURCE (G-I): local RTX-4060 dev box only, ~2.4 h total, $0. Both experiments are pure-CPU (closed-form geometry + epcache pose reads) — no model/GPU/network needed to answer them; the GPU halves (ZOD 5-drive precompute) are ACCESS-blocked → shipped as a job card (M-3). Why not eval-pod/Colab: neither question needs a model or GPU; the compute-blocked real-bytes work is escalated, not skipped.

## This run — increment 2 (2026-07-18 PM): curve-rebalance MEASURED [FLEET P0#3]
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

## This run — increment 1 (2026-07-18 AM): ZOD ingest
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
