# STATE — Data Engineering

LAST_RUN: 2026-07-17 (Tuesday agent) — branch `agent/data-engineering-20260717` (worktree `C:/Users/Admin/wt-de-0717`, D-026)
QUALITY: full (G-A…G-E, G-H, G-D1, G-D2 met; intake pkg 9✓ standalone; 2 measured experiments; no `stack/` files touched).

## This run (2026-07-17)
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
