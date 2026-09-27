# NEW-2 build record — THE map at 10 cm, one lift for everything (refcv7: SPEC_REFCV7 §6.2, A3 §8, A4 §9, A6 §11, A7 §12, A8 §13)

**Status (2026-09-27, ~04:15 Berlin = 02:15 UTC).**
- **REBASED onto fixes batch 3** (`4797ffb`): candidate tree `C:/Users/Admin/nb2_tree_4797ffb`.
  - Batch 3's 14 code files at `4797ffb` equal the `code/fix3` blobs of its LANDING_READY (0 differ). NEW-2 was pre-merged against those files (the preview tree `nb2_tree_b3prev`), and the official rebase output is byte-identical to that preview. The only difference is five renderer files the tip itself adds under `taniteval/tools` and `taniteval/tests`.
  - 0 conflicts. `refc.py`, `refc_v3.py`, `refc_v3_train.py`, `declared_vs_built.py` and `test_declared_vs_built.py` merge 3-way cleanly. Batch 3 adds no flags, so the registry stays at 213.
  - Earlier rebases:
    - onto fixes batch 2 (`b4a59b9`): 1 conflict, the registry count, resolved by union (204 + 9 = 213);
    - onto NEW-1 (`06380de`): 2 conflicts, both unions (`DECODER_PASSTHROUGH` carries NEW-1's three prior keys AND `fmap_s8`; 203 + 9 = 212).
  - Tooling: `code/rebase_onto_tip.py` + `code/make_fix_bundle.py` + `code/write_landing_ready.py`.
- **A8 applied:** the class weights are `sqrt_mf` (the script's default and the only id a JSON marks `pre_registered`); the per-band bar rule is A8 item 2 exactly.
- **D3 × NEW-2 (found on this rebase, FIXED).** Batch 2's first-row check (`declared_vs_built.check_logged_rows`) looks only for the `ga_*` keys that config.json DECLARES. The declaration listed only the parts from `refcv6_perception_branch.grad_reach_report`. So the 10 cm branch's `ga_mh_*` keys were written but never declared: a dead 10 cm reach row would pass the check. And a map-hires-only arm (no perception branch, no tac decoder) declared nothing and wrote nothing. The fix, in `refc_v3_train.py`: `_grad_reach_declared` now includes the 10 cm branch, and `_grad_reach_declaration` lists `ga_mh_{lift,encoder,refine,trunk_s8_stage}` plus each `_n`. An arm without the branch declares exactly what the tip did (same keys, same `source` string). Tests are `test_D3_*` in `test_map_hires_wiring.py`:
  - GREEN;
  - a RED arm: one dead `ga_mh_refine` is named at the first row;
  - the blind spot shown directly: under the tip's declaration, a row missing all eight `ga_mh_*` keys passes.
- **Fixes batch 3 pre-check** (the Master Mind's two consequences):
  - NEW-2 sets NO attribute on the six `@strict_fields` classes (`AgentSeamConfig`, `DiffusionFlags`, `EgoHistoryConfig`, `MaxSpeedConfig`, `Refcv7HeadConfig`, `TacticalDecoderConfig`). Its config lives in `MapHiresConfig` and in two DECLARED `PerceptionBranchConfig` fields (`bev_source`, `planner_crop_m`); `_pin_map_hires` only reads argv and refuses.
  - NEW-2 freezes NOTHING. It writes `requires_grad` nowhere. Under `bev_source = map_hires_pool`, the stride-16 lift and the 0.5 m map branch are never constructed, so there are no dead parameters.
  - Evidence: INSPECTED (grep over the shared-edit diff and the new files), not mutation-tested. Batch 3's own G-DVB is the test once it lands.
- **Thor-gate portability (MEASURED on Thor, in the gate's own environment: the `tanitad-train` venv plus its `--no-deps` pytest folder, CPU only). Two defects, both FIXED.**
  1. `taniteval/taniteval/map_hires_metrics.py` imported scipy at module level, and that venv has NO scipy. `test_map_hires_metrics.py` would have been an import error on the gate.
     - Now the 0.2 m tolerance is an exact offset footprint (`_tol_offsets` / `_near`), which is the distance transform's `< tol` by construction.
     - Tested three ways: against scipy's EDT on random masks at 9 tolerances (runs where scipy exists); by literal footprints (3 × 3 at 0.2 m, 21 cells at 0.25 m); and by an import with scipy BLOCKED.
     - Mutation-checked: re-adding the module-level import turns that guard RED.
     - Every other scipy use in the repo is already a lazy import.
  2. `test_map_head_hires.py`'s constant-logit control held float32 sums to 1e-6. On aarch64, `lc[1, 0]` reads 2.0e-6 relative off the exact value: float32 pairwise summation at N = 128,000.
     - Now the control uses a declared `F32_SUM_REL = 1e-5`, with the measurement in its docstring.
     - A real defect (for example the 255 fold) still moves `lc` by tens of percent.
  - After both fixes, the NEW-2 set of 11 files on Thor: **281 passed, 7 skipped, 0 failed**. The skips are the D:-path real-data tests (5), the scipy-equivalence test, and the pre-existing mount sidecar.
- **BUILT and unit-tested** to A6 (option (c): ONE stride-8 lift at 0.25 m + ONE BEV encoder feed the 10 cm map AND every BEV consumer) and A7 (extent 100 m × ±30 m, planner crop 60 m × ±16 m, `/3` reader with anchored coordinates, decoder gradient checkpointing).
- **WIRED** as a `code/fix/` proposal against the tip's blobs (`BASE_BLOBS.txt`, `LANDING_BLOCKS.txt`; the diff re-applies to the RAW tip blobs byte for byte, `code/make_fix_bundle.py`).
- **NOT integrated yet.** Landing order (Master Mind): NEW-1 (`06380de`) → fixes batch 2 (`b4a59b9`) → fixes batch 3 (`4797ffb`) → NEW-2, which is now based on `4797ffb`.
- **`train()` end to end on REAL files: a plumbing smoke, not a training run** (MEASURED 2026-09-27 ~04:12, dev-box CPU, 35 s).
  - Setup: the b4a59b9 candidate's own `train()`; `--smoke` model; resnet18 at 416 × 1024 with a 9-channel stack; 2 steps at batch 1. The data is the eval-139 v2 view (the only local clips with `/3` GT, via `--allow-eval-clips-in-train` BY NAME) plus the 137 real `/3` files; uniform scratch weights. The checkpoint was deleted; nothing is kept.
  - The 10 cm census reads `frac_ok 0.9856 of 24,328 windows over 139 clips (PASS)`.
  - Both metrics rows carry 443 `map_hires*` keys over the five A7 bands; `map_hires` = 2.0979576 against Σ `lc` = 2.0979575.
  - config.json DECLARES the eight `ga_mh_*` keys. D3's first-row check PASSED on the real row, and every 10 cm part took gradient (`ga_mh_lift` 500.8, `encoder` 4,422, `refine` 466, `trunk_s8_stage` 109.3).
  - **GPU s/step and memory are still UNMEASURED here.** The early G-MAP-OVERFIT on Thor records the branch's; G-LIVE is the launch-gate smoke.
- **`/3` GT: VERIFIED on the COMPLETE real eval export.** Source: `D:/refcv6_eval_kit/data/sam3_gt_v3_eval/`, 137 files, 636,747,415 B; the Master Mind's COPY_VERIFY passed. Results:
  - every file opens at 100 × ±30 through every guard;
  - every file is REFUSED under the `/2` extent;
  - inside the anchored old window, its `fine_codes` AND `cart_frac` equal `/2` byte for byte: **0 of 5,287,680,000 fine and 0 of 1,903,564,800 cart cells differ over 27,540 frames** (MEASURED).
  - Evidence: `raw/v3_reader_control_eval.json` (by `code/v3_reader_control.py`). The run is bracketed: the 137 files' (name, size, mtime) listing is identical before and after.
  - The first 23-file pass read 0 of 887,616,000.

`RESULT.md` in this package is not edited here (another agent's §5). The D7 correction to its §4.1 is in §8 below.

- Builder: the NEW-2 build agent, for the Master Mind (the single committer).
- Clean trees:
  - `C:/Users/Admin/nb2_tree_4797ffb`: the candidate (`4797ffb` + NEW-2; `tanitad.__file__` asserted inside it). The same tree is on Thor at `/home/nvidia/nb2_gate_4797ffb/code` (2,809 files md5-verified, plus the audit raw dir);
  - `C:/Users/Admin/nb2_tree_b4a5`: the previous candidate (`b4a59b9` + NEW-2);
  - `C:/Users/Admin/nb2_tip_b4a5`: the pristine raw export of `b4a59b9`;
  - `C:/Users/Admin/nb2_b1check_b4a5`: `b4a59b9` + batch 1 ONLY (the independence check);
  - `C:/Users/Admin/nb2_tree_0638`: the previous candidate (`06380de` + NEW-2);
  - `C:/Users/Admin/nb2_snap_preA6_0927`: the pre-A6 candidate (reference only, never land).

## 1. What A6/A7 changed, and where each piece is

| item | status | where |
|---|---|---|
| **A6 (c): ONE lift** — the 10 cm branch's stride-8 lift + 0.25 m encoder run INSIDE the forward (`refc_v3._bev_hook`), and its encoder output (`map_hires_bev`) is shared | DONE | `map_head_hires.MapHiresBranch`; `code/fix`: `refc.py` (hook gets `fmap_s8`), `refc_v3.py` (hook) |
| **(ii) the planner's pooled BEV** — crop 60 m × ±16 m, 2 × 2 average pool to 0.5 m, 1×1 + GN to the consumers' width (96): the SAME `bev_feats` / `bev_tokens` / box-memory BEV refcv6 consumers read | DONE | `code/fix`: `refcv6_perception_branch.PlannerBEVPool`, `PerceptionBranchConfig.bev_source = "map_hires_pool"` |
| **removed**: the stride-16 lift, the 0.5 m encoder and map head/loss; `--map-lowres`; `aux05_*` | DONE | `--map-hires on` REQUIRES `--w-map 0`; the lowres/aux05 machinery is gone from the trainer |
| **the seam, declared**: `--bev-source {s16_lift,map_hires_pool}` → `PerceptionBranchConfig.bev_source` (G-HYG) + G-DVB `bev_source`; red arm: a consumer still wired to a stride-16 lift | DONE | `map_head_hires.dvb_check_bev_source`; test `test_DELIBERATE_REGRESSION_a_consumer_still_wired_to_a_stride16_lift_goes_RED` |
| **A7 crop, declared**: `--bev-planner-crop-m 60 16` → `PerceptionBranchConfig.planner_crop_m` + G-DVB `bev_planner_crop_m`; red arm: an uncropped 200 × 120 planner grid | DONE | `map_head_hires.dvb_check_planner_crop`; test `..._an_uncropped_200x120_planner_grid_goes_RED` |
| **the EXTENT, declared end to end**: `--map-hires-x-max-m` / `--map-hires-y-half-m` (unset = 100 / 30, §12) → `MapHiresConfig.x_max_m/y_half_m`: reader, targets, lift grid (400 × 240), decoder output (1000 × 600), bands (every 20 m), class weights, pooling | DONE | `semantic_map_gt_fine.MapExtent`; G-DVB `map_hires_x_max_m`, `map_hires_y_half_m` |
| **the `/3` reader**: schema `tanitad.sam3_map_gt/3`, extent from meta, REFUSES a mismatch (never crops or pads); a `/2` file is refused outside 60 × ±16; anchored lateral centres; the `/2`-window control (fine AND cart) | DONE (synthetic; real-file test pending the export) | `semantic_map_gt_fine.open_path_fine`, `V3_CONTRACT`, `compare_v2_window` |
| **gradient checkpointing, declared**: `--map-hires-grad-ckpt {on,off}` (unset = ON) → `MapHiresConfig.grad_ckpt` + G-DVB | DONE | the encoder and the decoder recompute in backward |
| **ga\_\* groups**: `ga_mh_lift`, `ga_mh_encoder` (SHARED by map + planner), `ga_mh_refine` (map only), `ga_mh_trunk_s8_stage`, and `ga_bev_pool` (planner only); `ga_lift` / `ga_bev_encoder` / `ga_map_head` are absent under A6; the `ga_mh_*` keys are DECLARED in config.json `grad_reach_logging` and held at the first log row (D3) | DONE | `grad_reach_report_hires`, `refcv6_perception_branch.grad_reach_report` |
| **the Watch contract**: 40 literal keys `eval_map_hires_iou_{cls}_{band}` | DONE, pinned | §4 |
| **the audit's gmo_spec.json AS WRITTEN** (md5 `d3a41b92…`): `s8_zeros`, `must_fail_all`, the ≥ 1,000-cell floor, C1–C3 in-run, both rules, the 1 ms guard as a GATE condition; R3 `refcv6_head_05m` recorded NOT RUN with its reason | DONE | `stack/scripts/map_hires_overfit.py` |
| **logging per LOGGING_SPEC_MAP10** in every band: counts (never per-batch IoUs), `lc`, `gn`, `gno`, both rules, `map_hires` itself logged EXACT (check 2: Σ `lc` = `map_hires` to 1e-5) | DONE | trainer `_train_row_scalars`, `_eval_row_from_acc` |
| **D1** declared prior-corrected rule; ⭐ NEW defect fixed: a weight-0 class is now NEVER decided (the naive `z − log 0 = +∞` decided it on EVERY cell) | DONE, RED arms | `decide`, `logits_to_codes` |
| **A8**: the weights are `sqrt_mf` (default, `pre_registered` only for it); the loader stamps `definition_id` + `pre_registered` into config.json; the per-band bar rule | DONE | `compute_map_class_weights.py` (`REGISTERED = "sqrt_mf"`), `load_class_weights` |

## 2. The architecture under A6/A7

```
fmap_s8 [B, 512, 52, 128] (resnet101 layer2, newest frame; the trunk tap)
  lift @ 0.25 m (projection first)         -> [B, 64, 400, 240]   131,200 params
  encoder: stem + 5 dilated blocks          -> [B, 64, 400, 240]   406,912   <- SHARED
    (i)  refine: 1x1+GN -> x2.5 -> 2x(3x3+GN+GELU) -> 1x1  -> [B, 8, 1000, 600]   20,936   THE map
    (ii) PlannerBEVPool: crop rows 0..240, cols 56..184 -> avg 2x2 -> 1x1+GN
                                            -> [B, 96, 120, 64]      6,336   -> box3d, BEV tokens (30x16), coupling
```

- Parameters: branch 559,048 + pool 6,336 = **565,384**; REMOVED stride-16 path 1,310,697 (lift 524,544 + 0.5 m encoder 785,280 + head 873). Net **−745,313**. MEASURED.
- The consumers keep their shapes and widths (`bev_cfg.d_out` = 96 = `--tac-decoder-d-bev`, the coupling's `d_bev`): the planner side changes only in WHERE its BEV comes from.
- The crop is exact: both grids start at x = 0, the lateral offset is (30 − 16) / 0.25 = 56 cells; a test pins the pooled cells = the mean of their four 0.25 m cells.
- The forward's `perception_grid/perception_valid` under A6 are the 0.25 m bank's; the perception branch REFUSES a stride-16 geometry and a hook with a 10 cm branch AND a stride-16 perception branch refuses ("ONE lift").
- ⚠️ **This changes the planner's input** relative to refcv6 (§11.1 says so): the tokens/coupling now read a 0.25 m-encoded, pooled BEV shaped by the 10 cm map loss.

## 3. The extent and the `/3` reader

- `MapExtent(x_max_m, y_half_m)`: frozen, multiples of 0.5 m, never below 60 × ±16. `EXTENT_REFCV7 = (100, 30)`: fine 1000 × 600, lift 400 × 240, 0.5 m 200 × 120, bands `0_20 … 80_100`, anchor offsets 140 (0.1 m) / 28 (0.5 m) — all literal-tested.
- `/2`: every existing guard via `semantic_map_gt.open_path`; REFUSED under any extent but 60 × ±16 ("Re-export it as /3").
- `/3`: `V3_CONTRACT` is written down in the module (the `/2` contract minus the 0.5 m requirement; `cart_frac` tolerated, never read). The extent is read from `meta.fine` (or `meta.extent`; they must agree) and must EQUAL the declared one. Identity, legend, frame axis, world-map resolution, codes: the same refusals as `/2`, each tested.
- ⭐ **Anchoring, MEASURED here:** inside the old window the anchored lateral centre `−16 + (j_rel + 0.5)·0.1` equals the `/2` float bit for bit on all 320 columns; the naive `−30 + (j + 0.5)·0.1` differs on **127 of 320**. (At 0.5 m both agree.) This is why the export anchors.
- The control `compare_v2_window` (fine AND cart, the anchored window) passes on a synthetic anchored export and catches a 5-column shift, and reads 0 differing cells on the REAL `/3` eval export (23 files, 4,623 frames, MEASURED). The real `/3` meta matches `V3_CONTRACT` key for key (schema, rig frame, channels, `fine` / `extent` = 100 × 30, `world_map_res_m` 0.1, `clip_sha12`).

## 4. Logging, and the Watch contract (final spelling)

Train rows: `map_hires`, `n_map_hires_cells`, `n_map_hires_cells_seen`, and per class × band `map_hires_{n,lc,gn,gno,inter,union,interraw,unionraw}_<cls>_<band>` — unrounded — plus the derived `map_hires_{iou,iouraw,lshare}_*`. Eval rows: the same with `eval_`.

**The Watch/gate contract — 40 keys:** `eval_map_hires_iou_{cls}_{band}`
- `cls` ∈ `nocls, drivable, lane, crosswalk, arrow, edge, hatched, sidewalk`
- `band` ∈ `0_20, 20_40, 40_60, 60_80, 80_100`

(with `eval_map_hires_iouraw_*` and `eval_map_hires_lshare_*` beside them, 40 each). A test writes the 40 names literally; RED arms: a 60 m branch (16 missing) and a renamed class.

## 5. The loss, and D4

The loss: hard CE, `ignore_index` 255, lift-valid narrowing, **sqrt(median-frequency) weights (`sqrt_mf`, SPEC_REFCV7 §13 A8)** clipped at 25, counted **at the declared extent** (the loader refuses weights counted on another extent; a JSON without `extent` = the `/2` grid). Ids: `sqrt_mf` (registered, default), `mf` (Eigen & Fergus), `mf_global`; `--weight-floor` by name only.

**D4 — the big classes' share of the logit gradient at convergence (S2), ANALYTIC** (`code/d4_weighting_proposals.py`, `raw/d4_weighting_proposals.json`; the audit's own formula reproduces its published MF shares to 1e-17):

| weighting | S0 thin share (init) | S2 big class, each (min) | S2 big, total | S2 thin min | w max/min |
|---|---:|---:|---:|---:|---:|
| unweighted | 0.029 | 0.202 | 0.731 | 0.006 | 1 |
| **MF-present, clip 25 (PRE-REGISTERED)** | **0.474** | **0.027** | 0.081 | 0.051 | 167 |
| MF-global | 0.625 | 0.015 | 0.046 | 0.191 | 593 |
| sqrt(MF-present) | 0.135 | 0.104 | 0.340 | 0.024 | 12.9 |
| MF-present, floor 0.25 | 0.151 | 0.086 | 0.310 | 0.038 | 28.9 |
| MF-present, floor 0.5 | 0.082 | 0.131 | 0.473 | 0.029 | 14.4 |
| 0.5·MF-present + 0.5 | 0.066 | 0.148 | 0.532 | 0.020 | 7.9 |

- Under A6 there is no 0.5 m auxiliary: the audit's "drivable keeps ~35 % through aux05" (option a) no longer exists. Every planner-facing BEV feature now sits downstream of the 10 cm loss AND the consumers' losses (box3d, tactical, coupling), which still carry dense big-class-relevant signal.
- **Resolved by A8 (the Master Mind, 2026-09-27 ~02:20):** `sqrt_mf` — each big class keeps 10.4 % of the gradient at convergence (vs 2.7 % under MF-present) and every thin class is still up-weighted vs unweighted CE (S0 thin share 13.5 % vs 2.9 %). The decision rule is unaffected (prior-corrected with the SAME frozen weights).

**D5 — far-range markings.** Not changed (no range mask added). The per-band signal is logged from step 1 (`map_hires_{lc,gno}_<cls>_<band>` for 5 bands). **No range-aware weight is proposed:** the audit's numbers bound resolvability (3 and 2 stride-8 feature rows for 20–40 / 40–60 m; 20 and 7 image rows) but do not measure what far-band supervision does to near-band quality, and the label mass of the thin classes is nearly flat with range (lane 1.68 / 1.63 / 1.60 % of seen cells at 0–20 / 20–40 / 40–60 m, audit proxy) — the weights do not concentrate on far cells by frequency. The in-run per-band `lc` shares are the measurement a proposal would need.

**D9 — stride 8.** Adequate laterally to ~20 m; beyond ~15 m longitudinally the 416-row image is the limit, not the stride (audit). No action. At the A7 extent the 60–100 m bands sit further beyond it; they are reported with their n and are expected near 0 for both models.

## 6. `--w-map-hires`

Rule (A6): the 10 cm map inherits refcv6's map loss budget at init — `w_map_hires = w_map(refcv6) × median L_map_0.5m(init) / median L_hires(init)`, `w_map = 1.0`. MEASURED at 60 × ±16 (8 windows, 4 clips, ImageNet resnet101): L_map_0.5m 2.2864; L_hires 2.3237 unweighted / 2.1814 with the dry-run weights → **0.984 / 1.048**. **Proposal: 1.0.** At 100 × ±30 and with the TRAIN weights the ratio is ESTIMATED unchanged (near-uniform heads at init: ≈ ln 9 / ln 8 = 1.057); re-measure on Thor with the `/3` GT if the MM wants the number there.

## 7. Cost (ANALYTIC-EXACT: CPU saved-tensor accounting, fp32, batch 1 × 16; `raw/cost_analytic_a6_cpu.json`)

| extent | grad ckpt | saved at b16 (branch + pool) |
|---|---|---:|
| 60 × ±16 | off | 6.971 GiB |
| 60 × ±16 | on | 0.599 GiB |
| **100 × ±30 (A7)** | off | **21.304 GiB** |
| **100 × ±30 (A7)** | **on (default)** | **1.391 GiB** |
| removed stride-16 path (refcv6) | — | 3.218 GiB given back |

- The trunk tap (stem → layer2, one frame): 208 MiB at b16 with `--trunk-chunk-ckpt` (8.35 GiB unchunked). MEASURED earlier, extent-independent.
- Net activation change vs refcv6 at A7 with checkpointing: +1.39 + 0.20 − 3.22 ≈ **−1.6 GiB** (ESTIMATED from the three analytic numbers).
- CPU time at b1 is recorded but is not a GPU number. **s/step on Thor: UNMEASURED** (G-LIVE smoke; > +25 % over refcv6's 6.4 s goes to the PI). GPU `max_memory_allocated`: UNMEASURED (`cost_map_hires.py --mode gpu` is ready).

## 8. Corrections and findings

1. **D7 correction (to this package's `RESULT.md` §4.1, which is not edited here):** the claim that "the soft-target out-voting … disappears by construction" at 10 cm is wrong in general. With localisation error σ ≥ 0.2 m the calibrated posterior at a line is < 0.5 and the edge ceiling falls to 0.04 (audit §2c). The levers are localisation, then the decision rule — the prior-corrected rule is declared for that reason. The `map_head_hires` docstring says so.
2. **FIXED here (found by this build): a weight-0 class under the prior-corrected rule.** `argmax(z − log w)` with `w_c = 0` adds `+∞` and decides class c on EVERY cell (both `decide` and `logits_to_codes` had it). Now a zero-weight class is never decided; negative / non-finite weights are refused. RED arms in both test files. (No launch config has a zero weight; `lane_w0` in the harness decides with the launch weights.)
3. **Integration conflict, flagged not changed:** the DrivoR-T scorer (`--w-r7-scorer`, a different design) needs `--w-map > 0` for its DAC oracle's 0.5 m `map_frac`; A6 pins `--w-map 0` under `--map-hires on`. Not in refcv7's argv.
4. **`declared_vs_built._c_perception`** (a shared file owned by the G-DVB agent) gains one clause: under `--bev-source map_hires_pool` the perception branch is built for its consumers even at `--w-box3d 0`. Included in `code/fix`.
5. **The 1 ms time guard** needs camera timestamps (v2ep payloads carry none, MEASURED): `--cam-ts-dir` with the corpus's `<clip>[.camera_front_wide_120fov].timestamps.parquet`. Without it the harness runs the pose-content check and its verdict CANNOT be PASS.
6. **Bars in every band:** a band where a class is absent from GT and every prediction is UNDEFINED (n 0), reported, never dropped; a bar PASSES iff every band with GT cells passes (≥ 1 such band). Implementer's rule, made before any number — for the Master Mind.
7. **The refcv6 baseline on the larger extent** predicts only its 60 × ±16 window; every other cell is `NO_PREDICTION`, so its seen GT there is missed (IoU 0 in 60–100 m by construction). Reported with n.
8. Carried over: pooled (not mean-of-window) statistics; strict 0.2 m tolerance; `fmap_s8` is the newest frame, unfused; D3 (`ga_*` off-by-one) is fixed by batch 2 (`b4a59b9`). `ga_mh_*` and `ga_bev_pool` inherit its cadence fix, and `ga_mh_*` are now DECLARED, so its first-row check holds them (see Status); the eval loader's 0.5 m lift-bank equalisation finding is moot under A6 (no 0.5 m bank) but open for refcv6 records; the Training Watch builder is not changed here (40 panels + both rules + the alarm tile, LOGGING_SPEC §5).

## 9. Tests (this tree; heavy runs go to Thor)

- On the batch-3 rebase (`4797ffb` + NEW-2), the 33 files below plus batch 3's `test_grad_unreachable_declared.py`, 34 files:
  - Dev box: **629 passed, 2 skipped** (both pre-existing), 3 deselected by name, 123 s.
  - Thor, in the gate's environment: **623 passed, 8 skipped, 0 failed**, 136 s. The skips are D:-path data tests, the scipy-equivalence test, the mount sidecar, and NEW-1's platform-pinned digest.
  - Batch-1 independence on `4797ffb`: **134 passed**.
- On the REBASED tree (`b4a59b9` + NEW-2 + the two portability fixes), 33 files: the 31 below plus batch 2's own `test_grad_reach_logged.py` and `test_navc_tau_file.py`.
  - **600 passed, 2 skipped** (597 before the portability fixes added 3 tests). Both skips are pre-existing: a banked sidecar unreadable on the mount, and a rebuild case that needs a full model.
  - On Thor, in the gate's environment, the 11 NEW-2 and seam files: **281 passed, 7 skipped, 0 failed** (see Status).
  - 3 deselected by name, as on every earlier run (`ANALYTIC_CONTROL`, `codes_are_in_the_legend`, `frame_axis_follows`).
  - The `/3` real-file test now RUNS; it was skipped while the export was pending.
  - Batch-1 independence on `b4a59b9` (tip + batch 1 only): **134 passed**.
- On the REBASED tree (`06380de` + NEW-2): NEW-2 + every seam-touching shared test + NEW-1's own tests (`test_residual_prior.py`, `test_ddv2_refc_chain.py`, `test_refcv6_f3_cascade_reaches_loss.py`), 31 files: **571 passed, 3 skipped** (the `/3` real-file control: export pending; two pre-existing), 160 s; after A8: the NEW-2 set **226 passed, 1 skipped**.
- Before the rebase (tip `b3f7ea6` + NEW-2): **504 passed, 1 skipped**; the real-data subset **10 passed, 2 skipped**; the fine reader's real-corpus controls (137 eval files) passed.
- Batch-1 independence: tip + batch-1 files ONLY: **130 passed** (real-data tests deselected).
- The full suite was NOT re-run after A6 on the dev box (RAM, and the Master Mind runs it on Thor). The pre-A6 candidate's full run: 10 failures = the 10 that fail on pristine `b3f7ea6` (`test_openloop_suite` ×4, `test_refcv6_diffusion` ×3, `test_refcv6_no_session_paths` ×3).

**Deliberate-regression arms — every one RED as required:** fine reader: spec check removed; metrics: not-seen counted, inclusive tolerance, a correction ignoring the weights, the naive `log 0`; loss / per-class signal: 255 folded into a class; decision: a rule ignoring the weights, the naive `z − log 0`; `fmap_s8` dropped from the forward; `fmap_s8` removed from `DECODER_PASSTHROUGH`; a drivable-only row; a zero lane weight (G-LIVE); the 5-dp-rounded loss breaking the `lc`-sum check; a 60 m branch / a renamed class breaking the 40-key Watch contract; a consumer wired to a stride-16 lift; an uncropped 200 × 120 planner grid; per-lever G-DVB disagreements (extent, grad ckpt, decision rule, weights, bev source); a `/3` file of another extent; a `/2` file under A7; an un-anchored (shifted) `/3` window; a loader that skips the rebuild; the harness: a passing regression arm, `s8_zeros` failing 4 of 5, the presence floor, a control that does not reproduce, a run without the 1 ms guard.

## 10. Final flags (for the launch gate)

| flag | default | G-DVB kind | declared field |
|---|---|---|---|
| `--map-hires {off,on}` | `off` | built | the branch |
| `--w-map-hires <float>` | 0 | loss | `MapHiresConfig.w_map_hires` |
| `--map-hires-class-weights <json>` | — | built | `MapHiresConfig.class_weights_sha256` |
| `--map-hires-decision-rule {prior_corrected,raw}` | `prior_corrected` | built | `MapHiresConfig.decision_rule` |
| `--map-hires-x-max-m <m>` | unset = 100 | built | `MapHiresConfig.x_max_m` |
| `--map-hires-y-half-m <m>` | unset = 30 | built | `MapHiresConfig.y_half_m` |
| `--map-hires-grad-ckpt {on,off}` | unset = on | built | `MapHiresConfig.grad_ckpt` |
| `--bev-source {s16_lift,map_hires_pool}` | `s16_lift` | built | `PerceptionBranchConfig.bev_source` |
| `--bev-planner-crop-m X Y` | unset = 60 16 | built | `PerceptionBranchConfig.planner_crop_m` |

Removed since the last proposal: `--map-lowres`. Registry: 204 at `b4a59b9` (with NEW-1's `--residual-prior` and batch 2's `--nav-compliance-tau-file`) → **213** with NEW-2. `fmap_s8` rides `RefCModel.DECODER_PASSTHROUGH` (`stack/tanitad/refs/refc.py`), beside NEW-1's three prior keys.

**refcv7 launch argv, relative to refcv6's:** drop `--w-map 1.0`; add `--map-hires on --w-map-hires 1.0 --map-hires-class-weights <TRAIN json @ 100 × 30> --bev-source map_hires_pool`; `--map-gt-root` = the **`/3`** export; keep `--agent-rig-camera extrinsics --agent-rig-extrinsics …`, `--w-box3d`, `--tac-decoder-d-bev 96`, `--bev-coupling`, `--equalize-bottom-rows 43`, `--trunk-frozen-bn`, the chunk/bf16/NHWC levers.

**Launch prep on Thor** (`$CODE` = an integrated tree, `$D` = `/home/nvidia/data`, `$V3` = the `/3` corpus root):

```bash
export PYTHONPATH="$CODE/stack:$CODE/taniteval" PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1
# 1. launch class weights (sqrt_mf, A8 -- the default) + positional-prior FIT counts at
#    100 m x +-30 m, ONE pass over TRAIN
python $CODE/stack/scripts/compute_map_class_weights.py --v2-cache $D/refcv6-b1-416x1024-train \
  --gt-root $V3 --split train --out $OUT/map_hires_class_weights_train_100x30.json \
  --prior-out $OUT/map_hires_prior_train_100x30.npz
# 2. G-MAP-OVERFIT, the audit's prereg as written. The 1 ms guard reads the camera
#    timestamps: 4,719 `<clip>.timestamps.parquet` files under _b1stage416 on Thor, which
#    cover all 4 frame-set clips; MEASURED, time_1ms passes on all 16 frames (section 11)
python $CODE/stack/scripts/map_hires_overfit.py --spec "<audit>/raw/gmo_spec.json" \
  --frameset "<audit>/raw/gmo_frameset.json" --decision-rule prior_corrected \
  --class-weights $OUT/map_hires_class_weights_train_100x30.json \
  --v2-cache $D/refcv6-b1-416x1024-train --gt-root $V3 --cam-ts-dir $D/_b1stage416/r0/camera_front_wide \
  --extrinsics $D/refcv6_train_eval139_extrinsics.json \
  --launch-commit <sha> --launch-argv-sha256 <sha> --out $OUT/g_map_overfit
```

## 11. The TRAIN class weights (MEASURED) and the early, NON-BINDING G-MAP-OVERFIT (Thor)

**The sqrt_mf TRAIN weights at 100 × ±30** were computed on Thor, 2026-09-27 04:14 Berlin, by the candidate's own `compute_map_class_weights.py` (git blob `8012922f…`, the landed batch-1 blob).

- Coverage: 4,369 of 4,369 train clips (coverage 1.0), 878,016 frames, 382.3 G seen cells, 2,518 s.
- File: `/home/nvidia/gmo_early_0327/weights/map_hires_class_weights_train_100x30.json`, sha256 `d70dec80…`, copied to `raw/`. Its `PROVENANCE.json` (sha256, script blob, argv) sits beside it.
- The Master Mind copies it to the canonical `/home/nvidia/data/refcv7/` only if the landed blob and argv match.

| class | share of seen cells | f_c (presence-normalised) | **w (sqrt_mf)** |
|---|---:|---:|---:|
| seen, no class | 45.09 % | 0.4509 | **0.141** |
| drivable | 21.13 % | 0.2113 | **0.206** |
| lane / road line | 0.92 % | 0.0102 | **0.939** |
| crosswalk | 0.33 % | 0.0078 | **1.074** |
| arrow / text | 0.04 % | 0.0010 | **2.957** |
| non-drivable edge | 0.36 % | 0.0036 | **1.571** |
| hatched | 0.05 % | 0.0019 | **2.153** |
| sidewalk / verge | 32.09 % | 0.3295 | **0.165** |

median f = 0.00897; nothing clipped (the max is 2.96, far below the clip of 25). MEASURED.

**The early G-MAP-OVERFIT's INPUTS, checked against the prereg.** Read-only, through the harness's own `load_frames`; `code/gmo_inputs_check.py`, `raw/gmo_early_inputs_check.json`.

- The frame-set md5 `4eafa03c…` equals the spec's. The 16 (sha12, raw frame) pairs equal the prereg table.
- x is [16, 9, 416, 1024]; codes are [16, 1000, 600], read at A7.
- **The 1 ms time guard (`time_1ms`) passes on all 4 clips / 16 frames**, with `--cam-ts-dir /home/nvidia/data/_b1stage416/r0/camera_front_wide`.
- Label census, 0–20 m, under three masks:

| class | prereg (approx. cone) | old window, no cone | the harness's scored mask (full width, lift-valid) |
|---|---:|---:|---:|
| seen-no-class | 50,123 | 79,283 | 99,924 |
| drivable | 351,454 | 480,942 | 414,196 |
| lane | 9,003 | 11,718 | 12,034 |
| crosswalk | 44,528 | 56,325 | 46,488 |
| arrow / text | 3,475 | 3,500 | 3,475 |
| edge | 7,820 | 10,508 | 8,829 |
| hatched | 18,220 | 24,906 | 18,476 |
| sidewalk | 190,081 | 302,807 | 303,854 |

Every class clears the 1,000-cell floor under the scored mask. The prereg census was an APPROXIMATE cone; the harness recounts with the exact §4 mask, as the prereg says it would.

**The run:**
- Scripts: `code/gmo_early_runner.py` (the candidate's harness, unmodified, in-process for peak memory) and `code/gmo_early_launch.sh` / `code/gmo_early_launch2.sh`.
- The record is stamped `"binding": false`, with `launch_commit` = `NONBINDING-EARLY-b4a59b9+NEW2`, and saved as `g_map_overfit.EARLY_NONBINDING.json`, never the canonical name.
- Phase 1 REFUSED at 04:14: another agent's GPU job (the box builder's `g_box_overfit.py`) held the GPU. At 04:16 the Master Mind ruled the two runs may SHARE the GPU. The record is stamped `gpu_shared_with`, and its s/step and peak memory are informative only.

**MAIN (healthy) at step 1,000 — FAILS on lane and edge.** Declared rule `prior_corrected`, band 0–20 m, pooled over 16 frames. MEASURED, from the harness's step-1,000 line.

| class | bar | IoU @ 1,000 | verdict | IoU every 100 steps (100 → 1,000) |
|---|---:|---:|---|---|
| seen-no-class | 0.85 | 0.873 | pass | .35 .51 .70 .75 .80 .80 .84 .85 .87 .87 |
| drivable | 0.85 | 0.906 | pass | .59 .71 .81 .84 .85 .86 .88 .89 .90 .91 |
| sidewalk | 0.85 | 0.936 | pass | .52 .68 .82 .84 .87 .89 .91 .93 .93 .94 |
| crosswalk | 0.50 | 0.742 | pass | 0 .31 .62 .62 .67 .68 .72 .70 .72 .74 |
| arrow / text | 0.50 | 0.613 | pass | 0 0 .17 .47 .42 .52 .54 .61 .60 .61 |
| hatched | 0.50 | 0.782 | pass | .55 .67 .73 .75 .69 .75 .79 .76 .78 .78 |
| **lane / road line** | 0.50 | **0.417** | **FAIL** | 0 0 0 .13 .20 .20 .35 .39 .36 .42 |
| **non-drivable edge** | 0.50 | **0.076** | **FAIL** | 0 0 0 0 0 .009 .034 .026 .047 .076 |

- Train loss .217 at step 1,000, against .215 at 900.
- Lane is still rising at 1,000. Edge first moves at step 600.
- Presence holds: edge 8,829 cells, lane 12,034. So this is a FAIL, not INCONCLUSIVE.
- Reported to the Master Mind BEFORE any code was touched: a failing class is a finding.
- The end-of-run record, below, carries the raw rule, the CE ratios, C1–C3 and R1/R2.

**The end-of-run record** (`raw/gmo_early/g_map_overfit.EARLY_NONBINDING.json`, md5 `2e56f14c`). IoU at step 1,000, declared / raw rule:

| arm | nocls | drivable | sidewalk | lane | crosswalk | arrow | edge | hatched |
|---|---|---|---|---|---|---|---|---|
| **MAIN** | .873/.875 | .906/.870 | .936/.930 | **.417/.490** | .742/.685 | .613/.539 | **.076/.233** | .782/.713 |
| s8_zeros (R2) | .189/.170 | .514/.433 | .433/.350 | 0/0 | 0/.068 | 0/.047 | 0/0 | 0/.102 |
| lane_w0 (R1) | .889/.890 | .895/.868 | .932/.938 | 0/0 | .735/.715 | .613/.524 | .099/.205 | .788/.746 |
| s8_detached (informative) | .871/.868 | .901/.862 | .939/.920 | .339/.438 | .729/.675 | .560/.473 | .153/.219 | .797/.731 |

- §6.4, CE@1,000 / CE@0 for MAIN: nocls .080, drivable .135, sidewalk .055, lane .280, crosswalk .067, arrow .079, edge .290, hatched .026. All pass (≤ 0.5).
- The loss was finite at every step, and `time_1ms` passed on every clip.
- R1 failed lane, as required. R2 failed all five thin classes, as required.
- C1: IoU_drivable = 414,196 / 907,276 = 0.456527 exactly. C2: 1.0 for all 8 classes. C3: bit-identical.
- **G_MAP_OVERFIT = FAIL** (MAIN: lane and edge). Under the raw rule both still fail, so §9's lever 1 (the decision rule) is excluded.

**MAIN_long** (INFORMATIVE; `raw/gmo_early/g_map_overfit_MAIN_long.INFORMATIVE.json`): the MAIN config run for 3,000 steps, alone on the GPU (0.607 s/step at b4). Its init fingerprints are identical to MAIN's.
- **Replicate.** At step 1,000 it reads lane .480 (MAIN: .417) and edge .096 (.076); the other classes are within ±.003, except hatched (+.021). So the run-to-run spread on lane at 1,000 is about 0.06.
- **First step at or above the bar**, declared / raw rule; each class stays above once it crosses:
  - crosswalk 200 / 300; hatched 200 / 300; sidewalk 400 / 500;
  - drivable 500 / 800; arrow 600 / 900; nocls 800 / 800;
  - lane **1,200 / 1,300**;
  - **edge: NEVER.** It reads .430 / .386 at 3,000.
- Edge's slope per 100 steps: +0.020 over 1,100–2,000 (R² 0.84), then +0.010 over 2,100–3,000 (R² 0.40). It is decelerating.
- **Edge at the 0.2 m tolerance:**

| step | P | R | F1 | IoU |
|---|---:|---:|---:|---:|
| 1,000 | .943 | .246 | .391 | .096 |
| 2,000 | .967 | .663 | .786 | .305 |
| 3,000 | .979 | .843 | .906 | .430 |

  By 3,000 steps edge is FOUND and misplaced by about one cell.
- **Edge by range** at 3,000 (n 164 in the 0–5 m bin):

| 0–5 m | 5–10 m | 10–15 m | 15–20 m | 20–25 m | 25–37 m |
|---:|---:|---:|---:|---:|---:|
| .415 | .523 | .448 | .397 | .412 | .348 |

  This is flat, while the stride-8 lateral footprint grows 5.5× over 5–20 m. Lane by range: .799 / .770 / .742 / .734 / .654 / .457.

**The GT-only grid oracle** (`code/grid_oracle.py`, `raw/gmo_early/grid_oracle_16frames.json`): the gate's 16 frames and scored cells. It averages the 10 cm GT to class fractions on a coarse grid, decodes them the way `HiresRefine` does (bilinear), and takes the argmax.

| grid | edge | lane | arrow | crosswalk | hatched | big classes |
|---|---:|---:|---:|---:|---:|---|
| 0.1 m (control) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **0.25 m** | **0.328** | 0.770 | 0.823 | 0.936 | 0.916 | .97–.98 |
| 0.5 m | 0.022 | 0.402 | | | | |

So from 0.25 m class fractions, edge cannot reach the bar.

## 12. NEW-2 R2: the lift lever, SPEC_REFCV7 §17 (A12, landed ab1fb45)

**The choice: (b), a 0.1 m near-range lift, map-only.** Pre-registered before any number of the arm. The case for it, all measured in §11:
- The 0.25 m class-fraction ceiling for edge is 0.328.
- The residual edge error is exact-cell placement (tolerance F1 .906 against IoU .430).
- The error is flat over range, so the stride is not the limit. That argues against (c).
- Nothing in the record implicates height mixing. That argues against (a).
- (a) and (c) would change the SHARED encoder and, with it, the planner's BEV.

**What changes** (`code/fix_r2/`: 8 files EDITED against `cef9709`; the diff re-applies byte for byte):
- `--map-hires-near-lift-m` (default 0 = off; the arm uses 20) maps to `MapHiresConfig.near_lift_x_m`. It is stamped as `near_lift_x_m` and `near_lift_rows`. It must be a multiple of 0.5 m, at least 0, and at most the extent; it is refused under `--map-hires off`.
- G-DVB kind `map_hires_near_lift_m` (`dvb_check_near_lift`) takes the registry to **214**.
- `NearLiftSkip` samples the same `fmap_s8` with one 1×1 projection per height (d_image to `d_up`). It runs at the 0.1 m cell centres over x 0–20 m × y ±30 m (200 × 600) and the main lift's 4 heights, summed.
  - Its output is ADDED to `HiresRefine`'s bilinear-upsampled input on those rows, inside the decoder's checkpoint.
  - It is **zero-initialised** and built last, so the arm is the NEW-2 function at step 0.
- The geometry is **derived** from the batch's 0.25 m geometry (`derive_near_geometry`: bilinear, linear extrapolation at the borders, conservative validity). There is no new interface.
- `ga_mh_near` / `ga_mh_near_n` are declared for D3 automatically.
- The eval loader rebuilds `near_lift_x_m`.
- The harness gains `--near-lift-m`, the must-fail arm `near_zeros` (zeros into the near lift only), `_fingerprint_without` and the spec checks.

**Tests.**
- Derived vs exact 0.1 m geometry, nominal camera, with and without eq43:
  - p99 error 0.098 / 0.085 image px;
  - max 0.59 px from 3 m out (inside 3 m the camera-height sample reaches about 13 px; no road-plane sample is involved);
  - 0 cells valid only in the derived geometry, and 98.0 % of the exact valid cells kept.
- Zero-init: the shared parameters are identical and the logits equal at step 0.
- Map-only: `map_hires_bev` is byte-identical when the skip is perturbed.
- Reach: the gradient reaches the skip from its zero init. With a zero source, the projection's gradient is exactly 0.
- RED arms: a naive index, a majority validity rule, a non-zero init, 4 DVB disagreements, a loader that drops the field, and a declared-but-unbuilt skip.
- The A12 spec is md5-pinned and differs from the prereg only in `amends`, `must_fail`, `near_lift_m` and `registered`.
- **Results.** Dev box, 35 files: **666 passed, 2 skipped**. Thor, gate environment: **660 passed, 8 skipped, 0 failed**.

**Cost** (ANALYTIC, `code/lift_lever_cost.py`, `raw/gmo_early/lift_lever_cost.json`; b1 ×16, grad ckpt on):

| option | saved GiB at b16 | fwd GFLOPs per sample | CPU time | params |
|---|---:|---:|---:|---:|
| as landed | 1.391 | 102.5 | 1.00 | 559,048 |
| **(b)** | 1.391 (+0) | 103.4 (+0.9 %) | 1.057 | +65,600 |
| (a) | 1.610 | 106.3 | 1.050 | +34,880 |
| (c) | 1.692 (plus the tap output, 416 vs 208 MiB) | 104.3 | 0.966 | −65,536 |

- (b)'s transient peak is 0.92 GiB at b16.
- (b)'s step time is ESTIMATED at ≤ +0.1 s at b16. NEW-2's own step time against the 8.0 s line is for G-LIVE to measure.

**The arm** (`raw/gmo_spec_A12.json`, md5 `f0aae8f1…`):
- The prereg literals are unchanged. MAIN is the lever arm.
- Must-fail: `s8_zeros` (both lifts; all five thin classes) and `near_zeros` (edge). `lane_w0` (lane) is kept.
- Runner: `code/gmo_r2_runner.py`, stamped non-binding. Launcher: `code/gmo_r2_launch.sh`, one GPU job at a time, queued behind the box builder's diagnosis.
- **Result: MAIN FAILS** (early, non-binding). The run started 10:40 Berlin, alone on the GPU. The declared-rule IoU at step 1,000, read from the harness's step lines:

| arm | nocls | drivable | sidewalk | lane | crosswalk | arrow | edge | hatched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **MAIN (lever)** | .895 | .912 | .944 | **.496** | .780 | .742 | **.194** | .834 |
| s8_zeros | .195 | .513 | .434 | 0 | 0 | 0 | 0 | 0 |
| near_zeros | .866 | .901 | .932 | .466 | .730 | .573 | .072 | .779 |
| lane_w0 | .885 | .913 | .949 | 0 | .802 | .715 | .207 | .827 |

  - The must-fails hold. s8_zeros fails all five thin classes. near_zeros fails edge (it reproduces NEW-2 MAIN). lane_w0 fails lane.
  - The lever's effect, isolated by near_zeros: edge +.122 (.072 to .194), lane +.030, and every other class up.
  - Edge at the 0.2 m tolerance: P .902, R .429, F1 .582. The gap is RECALL.
- **The complete record** (`raw/gmo_early/g_map_overfit_A12.EARLY_NONBINDING.json`, md5 `92dd6e14`):
  - It was paused from 11:31:59 to 13:32:15 Berlin; `paused_s` 7,216 is stamped.
  - C1–C3 and `time_1ms` hold. **G_MAP_OVERFIT = FAIL.**
  - Raw rule for MAIN: lane .502 (it would pass), edge .198 (fails). So the decision rule alone cannot pass the gate.
  - CE@1000 / CE@0: lane .184, edge .238. The area classes sit at .018–.066.
  - MAIN IoU by range, 0–5 / 5–10 / 10–15 / 15–20 / 20–25 / 25–37 m:
    - lane: .403 / .661 / .621 / .534 / .373 / .175;
    - edge: .076 / .220 / .249 / .195 / .122 / .038.
  - The near lift helps most at 5–15 m.

## 13. NEW-2 R3: the decoder lever, SPEC_REFCV7 §20 (A15, landed c1ed8d9), stacked on the near lift

**The choice: (3) the decoder, ranked ahead of (4) the weights, from the numbers.** The classes that lag are the thin LINES, not the rare or the low-weight ones:
- lane (weight .94) against crosswalk (weight 1.07): CE ratio at 1,000 is .280 against .067, and lane crosses its bar at step 1,200 against 200.
- arrow and hatched are the rarest classes and are among the fastest.
- edge never crosses by 3,000.

The decoder's own 0.1 m receptive field is 0.5 m. Lane sits at the median frequency, so no median-frequency weighting can raise it.

**What changes** (`code/fix_r3/`: the same 8 files EDITED against `2374cd2`; the diff re-applies byte for byte):
- `--map-hires-near-refine-blocks` (default 0; the arm uses 1) maps to `MapHiresConfig.near_refine_blocks`, an integer in [0, 4] that needs `near_lift_x_m > 0`. It is stamped, and refused under `--map-hires off` or without the near lift.
- G-DVB `map_hires_near_refine_blocks` (`dvb_check_near_refine`: the count and the registered design) takes the registry to **215**.
- `NearRefineBlock` computes `x + conv_d4(GELU(GN(conv_d2(x))))` on the near rows after the near lift is added. It is map-only. The last conv is zero-initialised and the block is built last.
- `ga_mh_near_refine` is declared automatically, and the eval loader rebuilds the field.
- The harness gains `--near-refine-blocks`, the must-fail arm `near_block_zeros` (the block reads zeros), and `_fingerprint_without(*prefixes)`.

**Tests.**
- At step 0 the R3 branch is byte-identical to R2: the shared parameters and the logits.
- `map_hires_bev` is byte-identical when the block is perturbed.
- The gradient reaches the block from its zero init. Under `near_block_zeros`, the block's gradient is exactly 0: it IS the A12 function, and it stays so.
- RED arms: a non-zero last conv, 5 DVB disagreements (including a block of another design), a loader that drops the field, and a declared-but-unbuilt block.
- The A15 spec is md5-pinned (`20929e21`) and differs from the A12 spec only in `amends`, `must_fail`, `near_refine_blocks` and `registered`.
- **Results.** Dev box, 35 files: **685 passed, 2 skipped**. Thor, gate environment: **679 passed, 8 skipped, 0 failed**. A literal-pin scan of 644 test files finds 0 broken pins.

**Cost** (ANALYTIC; `code/decoder_lever_cost.py`, `raw/gmo_early/decoder_lever_cost.json`, which measures the REAL R3 branch; b1 ×16, grad ckpt on):

| option | saved GiB at b16 | fwd GFLOPs per sample | params |
|---|---:|---:|---:|
| R2 | 1.391 | 103.4 | 624,648 |
| **R3 (A15)** | 1.391 (+0) | 107.8 (+4.3 %) | 643,144 (+18,496) |
| a deeper full-map conv (rejected) | | 114.5 | |
| d_up 64 (rejected) | | 171.3 (2.29 GiB per 0.1 m activation) | |

Step time is ESTIMATED at ≤ +0.1 s/step at b16.

**The arm** (`raw/gmo_spec_A15.json`):
- The A12 spec plus `near_refine_blocks` 1. MAIN is R2 plus the block.
- Must-fail: `s8_zeros` (both lifts; all five thin classes) and `near_block_zeros` (edge). `lane_w0` is kept.
- Runner: `code/gmo_r3_runner.py`, stamped non-binding.
- Launcher: `code/gmo_r3_launch.sh`. It waits for the A12 run's `A12_DONE` AND for the box builder's chain (PID 3676134) to exit. It tolerates stopped processes and runs one GPU job at a time.
- **Result: MAIN FAILS on edge only** (early, non-binding). Record `raw/gmo_early/g_map_overfit_A15.EARLY_NONBINDING.json`, md5 `a4c70212`; alone on the GPU, 0.711 s/step at b4. IoU at 1,000, declared / raw rule:

| arm | nocls | drivable | sidewalk | lane | crosswalk | arrow | edge | hatched |
|---|---|---|---|---|---|---|---|---|
| **MAIN** | .885/.879 | .922/.886 | .946/.922 | **.522**/.522 | .795/.735 | .742/.594 | **.259**/.236 | .829/.748 |
| s8_zeros | .206/.198 | .515/.447 | .427/.386 | 0/0 | 0/.045 | 0/.010 | 0/0 | 0/.100 |
| near_block_zeros | .854/.861 | .916/.890 | .924/.922 | .459/.534 | .796/.761 | .700/.695 | .134/.233 | .826/.736 |
| lane_w0 | .883/.874 | .914/.896 | .942/.934 | 0/0 | .806/.779 | .729/.639 | .216/.259 | .811/.809 |

  - The must-fails hold, and C1–C3 and `time_1ms` hold. **G_MAP_OVERFIT = FAIL** (edge).
  - The shared init is proven: the fingerprint without the block equals A12's, and without the lift it equals NEW-2's.
  - Lane now passes (.522, inside the ~.06 replicate spread).
  - Edge: .259; 0.2 m tolerance P .937, R .581, F1 .717. CE ratios: lane .166, edge .157.
  - Within the run, the block adds edge +.125 over `near_block_zeros` (the A12 function). Across runs it adds +.065 over A12's MAIN.
  - Edge rose in every range bin from 5 m out: .052 / .285 / .312 / .237 / .240 / .164.
  - Next, per the Master Mind: §9 lever (4), the weights (mf), stacked on R3.

## 14. NEW-2 R4: the weights lever, SPEC_REFCV7 §21 (A16, landed 879673c), stacked on R3

**Why (4), from the A15 record.** Edge is still RECALL-limited at step 1,000: at the 0.2 m tolerance P is .937 and R is .581.
- Under weighted CE a class moves with its weight RELATIVE to the classes that fill most cells (RETR-2026-09-27-A15-WEIGHT-ARGUMENT). Read from the two TRAIN files, mf raises that ratio:
  - for edge, ×7.6 (w_edge / w_drivable 7.6 → 58.2);
  - for lane, ×4.6 (4.6 → 20.8).
- The prior-corrected decision, argmax(z − log w), divides the same weights back out at inference. So the lever changes what is LEARNED, not where the decision threshold sits.
- Its cost is the big classes' gradient share, about 10.4 % → 2.7 % each at convergence (SPEC §20.1). They pass their 0.85 bars today at .885–.946.

**What R4 changes.** Harness only; no model, trainer or registry change.
- **`edge_w0`, a new must-fail arm:** edge's LOSS weight is 0, and the decision keeps the run's weights, as in `lane_w0`. An edge pass must therefore come from edge's own weighted loss term.
- **`class_weights_definition` in a spec:** a weights file of another definition is refused before any trunk is built.
  - Tests: the refusal (a sqrt_mf file against an mf spec), and its red arm, the acceptance (a file of the registered definition reaches the trunk build).
  - Two mutations, a check that refuses every file and no check at all, each turn the matching test RED.
- **The A16 spec** (`raw/gmo_spec_A16.json`, md5 `a4ef45d0`): the A15 spec plus `class_weights_definition: "mf"`, with must-fail `edge_w0` in place of `near_block_zeros`. Every other literal is unchanged. A test pins its md5 and its field diff against A15, with a red arm (a one-byte edit fails it).

**The weights.** `raw/map_hires_class_weights_train_100x30_MF.json` (sha256 `8ff4fd6d`), with provenance in `…_MF.PROVENANCE.json`.
- Computed by the landed script (blob `8012922f`) with `--definition mf`; otherwise the argv of the sqrt_mf launch file.
- Same TRAIN inputs: inputs_sha256 `205cdadc`, 4,369 clips, 878,016 frames.
- Run on CPU at nice 19 with idle I/O; 2,737 s. `pre_registered` is false, as the script writes for every definition but sqrt_mf; the spec's definition check is what admits it.

| class | sqrt_mf (launch) | mf (lever 4) |
|---|---:|---:|
| nocls | 0.141 | 0.0199 |
| drivable | 0.206 | 0.0424 |
| sidewalk | 0.165 | 0.0272 |
| lane | 0.939 | 0.883 |
| crosswalk | 1.074 | 1.154 |
| arrow | 2.957 | 8.745 |
| edge | 1.571 | 2.469 |
| hatched | 2.153 | 4.635 |

**The arm.**
- Runner: `code/gmo_r4_runner.py`, generated by `code/make_r4_arm.py` from the A15 runner; stamped non-binding.
- Launcher: `code/gmo_r4_launch.sh`.
  - One GPU job at a time; stopped processes are tolerated.
  - Its optional box-PID wait defaults to none. The generator first defaulted it to PID 1, which always exists, so the launcher would have waited forever. That was caught in the dry run before anything was shipped.
- **Result: MAIN FAILS on 5 of 8 classes** (nocls, drivable, sidewalk, lane, edge; declared rule, early, non-binding). SPEC §22.1 records A16 as FAIL; the launch weights stay sqrt_mf (A8).
  - **Partial record:** `raw/gmo_early/g_map_overfit_A16.PARTIAL_NONBINDING.json` (md5 `63f20286`), parsed from the harness's own log by `code/a16_partial_record.py`.
    - The Master Mind stopped the run after MAIN (§22.1: the remaining arms cannot change a FAIL); the arm's python got SIGTERM by explicit PID and exited 143 (`a16_A16_DONE.json`).
    - So there is no harness verdict: no raw rule, no CE ratios, no C1–C3. The must-fail readings below are DERIVED from the logged step-1,000 IoU against the spec's bars.

| class | bar | A15 MAIN @1000 | A16 MAIN @800 | @900 | **@1000** |
|---|---|---|---|---|---|
| nocls | .850 | .885 | .777 | .743 | **.608** FAIL |
| drivable | .850 | .922 | .877 | .868 | **.813** FAIL |
| sidewalk | .850 | .946 | .891 | .882 | **.795** FAIL |
| lane | .500 | .522 | .392 | .426 | **.363** FAIL |
| crosswalk | .500 | .795 | .730 | .718 | .548 |
| arrow | .500 | .742 | .703 | .682 | .690 |
| edge | .500 | .259 | .092 | .086 | **.054** FAIL |
| hatched | .500 | .829 | .781 | .787 | .606 |

  - The read-out step caught a transient: train loss at steps 800 / 900 / 1,000 read 0.283 / 0.176 / 0.393, and every class dropped at 1,000.
  - The FAIL does not rest on the transient. At steps 800 and 900, MAIN also misses nocls, lane and edge.
  - Edge at the 0.2 m tolerance, step 1,000: P 0.869, R 0.119, F1 0.209 (A15: P .937, R .581, F1 .717).
  - Must-fails, derived, failed as required: {"s8_zeros": true, "edge_w0": true}. `lane_w0` was stopped at step 100; `s8_detached` never ran.
  - Reading (cross-run, one seed each; the same-seed spread is ~.06 on lane and edge):
    - mf gave no MEASURABLE edge gain at this horizon and constant lr. Steps 700 / 800 / 900 read .127 / .092 / .086, against A15's .056 / .121 / .086.
    - It cost the big classes BEYOND the spread, as its gradient share predicted: nocls at steps 800 / 900 read .777 / .743, against .883 / .879.
  - R4 never landed on its own. Its harness generalisation is folded into R5 (§15).

## 15. NEW-2 R5: G-MAP-OVERFIT's lr decays over the final 10 %, SPEC_REFCV7 §22.1 (A17.1, landed 2ac0bfb)

**PI, ~15:44 Berlin:** "Yes, same rule for the map". This is A17's end-of-run lr decay for G-BOX-OVERFIT, applied to the map. The builder proposed it at 13:32Z, **before any A16 number existed**.
- The measured reason is in `raw/gmo_early/readout_noise.json`. On every early MAIN, the constant-lr step-1,000 reading is jumpy.
  - A15's edge read .073 / .056 / .121 / .086 / .259 over steps 600–1,000.
  - A16's train loss doubled (.176 → .393) in its final 100 steps.
- The real launch decays its lr (cosine to step 50,400); a constant-lr snapshot does not.

**The change** (harness + its test only):
- A spec key `lr_decay: {"kind": "cosine_to_zero", "start_step": 900}`. The lr multiplier is 1.0 for steps ≤ 900, then ½(1 + cos(π·(step − 900)/100)): 0.5 at step 950 and 0 at step 1,000.
- It is applied to every optimiser group before each step's `opt.step()`.
- With no key, the constant-lr path never touches the lr.
- A malformed key (an unknown kind, or a start outside (0, steps)) is refused before any trunk is built.
- The record's `optimiser` stamps `lr_decay`, and each curve row carries the lr in use when the decay is on.
- **Tests, all with literal expectations:**
  - the multiplier at steps 1 / 899 / 900 / 925 / 950 / 1,000;
  - six malformed keys, each refused;
  - the lr each `opt.step()` applied on the tiny rig: [.01, .01, .005, 0], and [.01] × 4 with no key;
  - a decay that bites changes the result, while the constant path reproduces itself exactly.
- **Red arms:** three mutations each turn a test RED: the LambdaLR-style off-by-one (`step − 1`), the decay never applied, and a linear ramp.
- R5 folds in R4 (§14): `edge_w0` and the spec-registered weights definition.

**The arm** (`raw/gmo_spec_A171.json`, md5 `5abd5b90`), from `code/make_r5_arm.py A171 §22.1 2ac0bfb`:
- A15's spec plus `lr_decay`, and nothing else. A test pins the md5 and the field diff, `[amends, lr_decay, registered]`.
- A15's configuration: near lift 20 m, 1 near refine block, and the TRAIN sqrt_mf weights (`d70dec80`).
- Must fail: `s8_zeros` and `near_block_zeros`; `lane_w0` is kept. Every prereg literal is unchanged.
- **The candidate is the tip 2ac0bfb plus the R5 blobs.** Of its 2,953 files, 2,949 are blob-exact to 2ac0bfb; the other four are the harness and its test (R4+R5) and the A16 and A17.1 specs.
  - The tar (`87cecca4`) was shipped to Thor, and `md5sum -c` passed on all 2,953 files (exit 0), with positive checks on the spec and the harness.
- Runner `code/gmo_r5_runner.py` stamps the record non-binding, with launch_commit `NONBINDING-EARLY-2ac0bfb+NEW2R5+A171`. Launcher: `code/gmo_r5_launch.sh`.
- Started 2026-09-27 14:07:34Z on the Thor GPU, alone. Steps 1–900 are A15's procedure exactly, so its step-900 reading is also a same-seed replicate of A15's.
- **Result: MAIN FAILS on edge alone** (declared rule, early, non-binding). It is read from the harness's own step lines in `raw/gmo_early/a171_launch.PARTIAL_at_MAIN.log` (md5 `9e79827b`). The must-fail arms were still running; the complete record follows in an addendum.

| class | bar | A15 @1000 (no decay) | A17.1 @800 | @900 | **@1000** |
|---|---|---|---|---|---|
| nocls | .850 | .885 | .886 | .882 | **.921** |
| drivable | .850 | .922 | .915 | .905 | **.932** |
| sidewalk | .850 | .946 | .943 | .939 | **.962** |
| lane | .500 | .522 | .446 | .464 | **.538** |
| crosswalk | .500 | .795 | .774 | .739 | **.806** |
| arrow | .500 | .742 | .675 | .690 | **.745** |
| edge | .500 | .259 | .135 | .149 | **.254** FAIL |
| hatched | .500 | .829 | .822 | .790 | **.840** |

  - **The decay did what A17.1 registered it for.** Train loss at steps 800 / 900 / 1,000 read 0.269 / 0.221 / 0.167, and every class rose from step 900 to 1,000. A16 had the opposite (loss doubled, every class dropped).
    - The big classes clear 0.85 with margin, and lane passes.
  - **Edge's level at step 1,000 is about .25 with or without the decay** (.254 vs A15's .259). So A15's .259 was not a lucky draw, and the read-out is not what keeps edge from its bar.
  - Edge at the 0.2 m tolerance: P 0.974, R 0.509, F1 0.669. It is still RECALL-limited: the cells it does find are almost all right.
- **The complete record:** `raw/gmo_early/g_map_overfit_A171.EARLY_NONBINDING.json` (md5 `0186504c`) and `raw/gmo_early/a171_launch.log` (md5 `92bdbd76`). Alone on the GPU; MAIN 0.7024 s/step at b4; peak 5.808 GiB. IoU at 1,000, declared / raw rule:

| arm | nocls | drivable | sidewalk | lane | crosswalk | arrow | edge | hatched |
|---|---|---|---|---|---|---|---|---|
| **MAIN** | .921/.919 | .932/.909 | .962/.946 | .538/.557 | .806/.776 | .745/.661 | .254/.301 | .840/.797 |
| s8_zeros | .170/.146 | .517/.423 | .438/.402 | .000/.000 | .000/.055 | .000/.049 | .000/.000 | .000/.100 |
| near_block_zeros | .914/.912 | .929/.904 | .956/.941 | .518/.546 | .804/.775 | .724/.651 | .199/.282 | .835/.790 |
| lane_w0 | .924/.920 | .922/.904 | .962/.948 | .000/.000 | .810/.778 | .748/.648 | .251/.308 | .851/.788 |
| s8_detached | .904/.902 | .920/.890 | .952/.937 | .471/.507 | .782/.742 | .731/.616 | .159/.255 | .827/.761 |

  - **G_MAP_OVERFIT = FAIL.** MAIN fails the bar on: edge. CE-ratio criterion holds on all 8.
  - Must-fails, failed as required: {"lane_w0": true, "s8_zeros": true, "near_block_zeros": true}. Controls C1–C3 reproduced: True. Time guard: ['time_1ms'].
  - The init is A15's, which proves the shared init: every fingerprint equals A15's (True).
  - The lr each logged step used: 900: 1.00e-03, 1000: 0.00e+00.
  - CE@1000 / CE@0: lane 0.166, edge 0.156. The area classes sit at 0.016–0.054.
  - MAIN's edge IoU by range, 0–5 / 5–10 / 10–15 / 15–20 / 20–25 / 25–37 m: .049 / .274 / .312 / .249 / .229 / .080.

**Prepared, NOT run (the Master Mind: it goes to the PI): the second near refine block arm.**
- Spec `raw/gmo_spec_BLOCK2_DRAFT.json`. Its `registered` field says DRAFT / NOT REGISTERED. Field diff against A17.1: `[amends, near_refine_blocks, registered]` (1 → 2). The generator is `code/make_block2_arm.py`.
- **No code change:** the flag and config allow up to 4 blocks. `near_block_zeros` zeroes the input of every block, so that arm is the A12 function plus constants.
- **Cost, ANALYTIC** (`raw/gmo_early/block2_cost.json`, by `code/block2_cost.py`, the real branch):
  - +4.42 GFLOPs per sample (+4.1 %), +18,496 params, +0 GiB saved at b16;
  - near-row receptive field 1.2 → 2.4 m.
  - The one-block row reproduces the A15 cost exactly, which is the control.

**What went to the PI, and the ruling.** The records supported three options:
- (a) the second block, the largest within-run edge lever measured (+.125 in A15; under the decay A17.1's own reading is +.055: MAIN .254 against `near_block_zeros` .199);
- (b) the step budget: NEW-2 without levers reached edge .430 at step 3,000 and was still rising;
- (c) a thin-structure loss term, not built and with no evidence on this rig.

In every arm edge is RECALL-limited: the cells it predicts are right, but too few of them are predicted by step 1,000. **The PI chose (b), "Budget to 3,000 steps"** (SPEC_REFCV7 §23, A18; §16 below). Every 1,000-step FAIL stays on the record.

## 16. A18: G-MAP-OVERFIT at 3,000 steps (SPEC_REFCV7 §23, the PI's "Budget to 3,000 steps", landed 37086c3)

**The amendment.** A18 is a dated goalpost amendment; every 1,000-step FAIL stays on the record.
- N = 3,000 steps, read out at step 3,000. The A17.1 decay moves with it: lr 1e-3 for steps 0–2,699, then cosine to 0 over 2,700–3,000.
- The configuration is A15's: near lift 20 m, one near refine block, the TRAIN sqrt_mf weights.
- Every bar and must-fail is unchanged.
- The spec is `raw/gmo_spec_A18.json` (md5 `4eda0636`, sha256 `5cb4f6fc…`), built by `code/make_a18_arm.py A18 §23 37086c3` from the A17.1 spec. Its field diff is `[amends, lr_decay, registered, steps]`.

**The early MAIN** (MAIN only, the Master Mind; non-binding):
- It ran with `code/gmo_a18_main.py` and `code/gmo_a18_launch.sh` on the R5 candidate (2ac0bfb + R5 blobs + the A18 spec), code-identical to 37086c3 + R5 (37086c3 changed only the SPEC). The launch tag is `NONBINDING-EARLY-37086c3+NEW2R5+A18`.
- The fingerprints are asserted equal to A15's record.
- It uses the harness's own `load_spec`, `run_arm` (the spec's steps and decay), `controls` and `verdict`.
- The must-fail arms were not run; they run in the binding run.

- **Result: MAIN PASS** (record `raw/gmo_early/g_map_overfit_A18_MAIN.EARLY_NONBINDING.json`, md5 `a2716f2c`; 0.6946 s/step at b4 alone; peak 5.808 GiB). Declared-rule IoU by step:

| class | bar | @1000 | @1500 | @2000 | @2500 | @2700 | **@3000 (declared / raw)** |
|---|---|---|---|---|---|---|---|
| nocls | .850 | .895 | .916 | .939 | .942 | .938 | **.964**/.963 |
| drivable | .850 | .923 | .935 | .952 | .958 | .958 | **.969**/.957 |
| sidewalk | .850 | .950 | .959 | .970 | .974 | .972 | **.982**/.974 |
| lane | .500 | .543 | .621 | .703 | .746 | .709 | **.779**/.758 |
| crosswalk | .500 | .795 | .834 | .875 | .897 | .879 | **.919**/.895 |
| arrow | .500 | .707 | .762 | .805 | .815 | .815 | **.845**/.787 |
| edge | .500 | .213 | .332 | .432 | .460 | .447 | **.555**/.493 |
| hatched | .500 | .845 | .825 | .864 | .879 | .885 | **.895**/.853 |

  - The first reading at or above 0.50 comes at step 1000 for lane and step 2900 for edge.
  - Train loss at steps 2,600 / 2,700 / 2,800 / 2,900 / 3,000: 0.12 / 0.101 / 0.085 / 0.087 / 0.075.
  - CE@3000 / CE@0: lane 0.058, edge 0.063.
  - C1–C3 on MAIN's logits reproduced: True. Time guard: ['time_1ms'].
  - Edge at the 0.2 m tolerance, step 3,000: P 0.996, R 0.884, F1 0.937.

**The MAP-LIFT closure** (`LANDING_READY_MAPLIFT.txt`). It lands ONLY on an A18 MAIN PASS, together with R5 (`LANDING_READY.txt`).
- **The canonical argv carries the FINAL 154-token list.** That is the box A17 argv plus `--map-hires-near-lift-m 20 --map-hires-near-refine-blocks 1`, placed right after `--map-hires-grad-ckpt on`. Its gate sha256 is `6402d33d…`, pinned as a literal in a test with a red arm.
  - `todo_map_lift` is closed; both flags are in `changes_vs_refcv6` with their SPEC sources.
- **`launch_gate.py`:**
  - MAP-LIFT is closed, and both flags are `required_values`.
  - The G-MAP-OVERFIT judge now binds the A18 PROTOCOL: the spec sha256, steps 3,000, lr 1e-3, batch 4, seed 0, `lr_decay` from 2,700, near lift 20 with one block, and the must-fail `near_block_zeros`.
  - It checked no protocol literal before, so it would have taken a 1,000-step record: the box judge's trap, inverted. Seven mutations each turn a test RED, including the prereg's lr 2e-4 and the 1,000-step protocol.
  - Sources are now read utf-8-sig at two sites:
    - `import_closure`, where G-HYG crashed on the BOM of `tanitad/eval/__init__.py` (the Master Mind's dry run);
    - `static_eager_own_files`, which silently SKIPPED such a file, so its imports were never required of a closure record.
    - The fixture-tree tests have literal red arms. No repo BOM was stripped.
- **The A18 spec pin** is a NEW test file, `stack/tests/test_map_hires_a18_spec.py`, so R5's test file stays R5's.

**The MAP BINDING run** (`code/binding/run_gmo_binding.sh`, the analogue of the box builder's script) wraps the unmodified harness with `closure_run.py --binding`.
- All four A18 arms: healthy, s8_zeros, near_block_zeros, lane_w0.
- The A18 spec and the audit's frame set are staged md5-checked under the run dir as closure data.
- The launch sqrt_mf weights are sha256-checked.
- `--launch-argv-sha256 6402d33d…`.
- The harness must be blob `9bec9e88`.
- PREFLIGHT_ONLY mode, and a strict GPU-idle check.
- **Dry-run on Thor (PREFLIGHT_ONLY):** on a hard-linked tree with the final argv, every input check passed; it then refused, as it should, because the GPU was busy (exit 4, the A18 MAIN).

## 17. A19: the BINDING overfit runs are MAIN-only (SPEC_REFCV7 §24, the PI 17:46 / 17:51 Berlin, landed 36cc332)

**Found before the map binding started (MEASURED on the frozen harness).** `map_hires_overfit.py` (blob `9bec9e88`, `main()` lines 650–652) refuses a run whose spec names must-fail arms that `--arms` does not run.
- So `--arms healthy` on the A18 spec itself exits at once with "⛔ the gated arm 'lane_w0' is not in --arms".
- The early A18 MAIN never hit this: its runner called `run_arm` / `verdict` directly.
- The A11 closure judge requires the registered harness to be the wrapped script, so a side runner is not an option.

**The A19 map spec** `raw/gmo_spec_A19_MAP_MAIN.json` (md5 `2d11ba07`, sha256 `56dea067…`), from `code/make_a19_map_spec.py 36cc332`:
- It is the A18 spec with `must_fail` and `must_fail_all` removed; `registered`, `amends` and the inherited-record path are added. Every other literal is A18's.
- `code/a19_spec_accept_check.py`, on the frozen harness:
  - `main()` with `--arms healthy` passes every spec check;
  - the A18 spec is refused (the red arm);
  - the harness's `verdict()` gives PASS for a passing MAIN with no must-fail rows, and FAIL with edge 0.49.
- `code/binding/run_gmo_binding_a19.sh` (md5 `a317bfef`) is the binding script for it. The Master Mind's chain started the map binding with this spec at 18:05 Berlin.

**The gate** (`LANDING_READY_A19.txt`, stacked on MAPLIFT):
- **The policy is explicit in the refcv7 PROFILE:** `overfit_main_only`, which carries:
  - the policy id and its SPEC source;
  - the A19 map spec sha256;
  - the inherited must-fail evidence with its paths:
    - map: A17.1's record;
    - box: `presence_w0` by construction, and `memory_zeros` NOT measured at full scale on the launch head (the TOY test and the pre-A14 run). The PI accepted this.
  - `_REFC` defaults it to None (every arm must run), so refcv6 and refc are unchanged.
- **Both judges accept a record whose must-fail arms are ABSENT, and MAIN is re-judged from its OWN literals:**
  - **map:** the 8 bars on the declared rule, ≥ 1,000 cells, the CE ratio (literal 0.5, now `map_overfit.ce_ratio_max`) and the finite loss recomputed from the record's own numbers, C1–C3, the 1 ms guard, and the A18 or A19 spec;
  - **box:** the six criteria at step 2,000, 113/77, the A17 schedule and the A13 optimiser.
  - The harness's overall verdict is NOT read under A19: it reads FAIL without the arms.
- **A must-fail arm that RAN is judged from its own result against the gate's literal bars.** If it passed, that is still a VOID (FAIL).
- **When A19 applied, the PASS text carries the inherited-evidence statement**, so the token says what it does not cover.
- **`eval_loader`:** the refcv7 profile defaults to `stack/tanitad/eval/refcv7_loader.py` (the loader agent's integration diff, re-applied).
- **Tests** (9 new, with literal expectations):
  - MAIN-only + MAIN PASS → PASS, whatever the harness verdict says;
  - each missed MAIN literal → a named FAIL, while the harness says PASS;
  - an arm that RAN and PASSED → VOID FAIL (map and box);
  - with the profile's A19 field removed, a MAIN-only record FAILS as before;
  - the A19 spec equals the A18 spec minus its must-fail rows;
  - the PASS text;
  - the eval-loader default, with refcv6's as the red arm.
- **Nine mutations each turn a test RED:**
  - map requires the harness PASS; map trusts the harness PASS;
  - map excuses a VOID; map reads the harness flags under A19;
  - box requires RESULT PASS; box excuses a VOID;
  - eval loader back to refcv6's; no A19 policy; PASS text without A19.

**The map BINDING (A19, MAIN-only): MAIN PASSES.** The Master Mind's chain ran it on Thor, 18:05–18:39 Berlin (rc 0).
- Files: record `/home/nvidia/refcv7_bind/run_map/out/g_map_overfit.json` (md5 `75c4c8f2`); closure `…/run_map/gmo_closure.json` (md5 `e740fc67`). Both are the Master Mind's to bank with the gate.
- Spec 56dea067 (A19), argv 6402d33d, 3,000 steps with the decay from 2,700; `results` holds `healthy` only.
- Declared-rule IoU at step 3,000: nocls .963, drivable .968, sidewalk .982, lane .773, crosswalk .915, arrow .838, **edge .547**, hatched .893.
  - Edge's margin is +.047, against the early run's +.055: the fresh draw came in .008 lower.
- C1–C3 reproduced; the 1 ms guard held.
- **Pre-judged by the A19 `judge_map_overfit`** (`code/a19_prejudge_binding.py`; record half, the closure half needs Thor):
  - **PASS**, with A19 applied and the inherited statement carried;
  - under the pre-A19 profile, the same record FAILS (4 reasons: the spec, and the three arms not run).
- `code/a19_real_shape_check.py`: the judge passes a real harness-shaped record (the early A18 run's own `run_arm`/`verdict` output), and fails it with edge forced to 0.49.
