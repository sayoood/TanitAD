# INVENTORY: what exists for an aux-raster (geoDAC) veto read on navtest C3 and RL-val

**2026-10-16, Sonnet agent for the Master Mind.** Dev box, CPU only. Read-only on every shared artifact. NOT committed, NOT pushed.
Machine twin: `INVENTORY.json` (every sha256 and shape below was read by `inventory.py`, run 2026-10-16 07:3x). Evidence class: MEASURED unless marked.
No pick, no PDMS delta and no veto outcome was computed on any raster. The only table statistic quoted is the DAC = 0 base rate over ALL proposals.

**Keys.** `token` = 16-hex NAVSIM token (a string; never an int). The label-store key is `(log_name, token)`. C3 = **1,058 navtest tokens, 123 logs**
(529 with u400 NC x DAC = 0 plus 529 uniform non-zeroed, seed 20261011; `tokens.json`).
⚠️ **Row order = `table.npz['token']` order, NOT `tokens.json` order** (same set; `tokens_json_order_equals_table_order = false`). Index by token string.

## 0. Verdict

| item | exists for C3? | what it is |
|---|---|---|
| (a) 64 proposals per token | **YES, u400 only** | `proposals [1058,64,8,3] f32` |
| (b) NAVSIM v1.1 per-proposal sub-scores + PDMS | **YES, u400 only** | `sub [1058,64,6]`, `pdms [1058,64]` |
| (c) scorer per-proposal scores, B's pick reproducible | **YES** | `logits [1058,64,7]`; pick reproduced **1058 / 1058** |
| (d) GT drivable-area rasters | **YES, 1058 / 1058** | navtest label store, 12,146 keys |
| (e) PREDICTED rasters, held-out cross-fold | **YES for A2 (10k) and 3 other probe configs; NO head weights for them** | `logit/sdf [n,40,32]` per fold unit |
| (f) footprint + NAVSIM DAC definition | **YES** | section 6 |
| RL-val | **YES for sets / PDM labels / scores / GT rasters; NO held-out predicted rasters** | section 7 |

## 1. (a) The proposals
* **Scorer:** u400 = SFT-4 arm B, `D:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt` (`proposals.npz['ckpt']`). Rule `navsim_v1_lane`, `select = best`, `repair_last_heading = True` (Amendment 7).
* **Files** (`raw/2026-10-11-c3-selection-headroom/`): `proposals.npz` sha256 `4678233b8e70d185…`, 8,644,607 B; `table.npz` sha256 `cf30519051bea03b…`, 8,329,021 B. Both carry the SAME `proposals`, `logits`, `pick` (bit-equal, `INVENTORY.json`).
* **Shape / dtype:** `proposals [1058, 64, 8, 3] float32`. **Frame:** ego rear-axle frame at t0, x forward, y left, heading in radians. **Grid:** 8 poses at 0.5 … 4.0 s (`sampling = [8, 0.5]`); the t = 0 pose is the origin and is NOT stored. These are the EXECUTED poses (repair applied), exactly what the harness scored.
* **Extent (MEASURED):** x in [-0.21, 60.59] m, y in [-21.53, 23.90] m. Inside the GT raster (x in [-8, 72), |y| < 32), with a >= 11 m margin on x.
* **Fidelity gate:** the pick's poses equal the banked seam, max |d| = 0.0 m on 1058 tokens (`gate_fidelity.json`).
* **P2 is NOT this set.** `D:/Projects/TanitAD-artifacts/p2_zero_tail_table/{table,proposals}.npz` (sha256 `289f4b15…` / `50165a78…`) hold P2's 64 proposals for **509 tokens** (P2's zero tokens, 6-column logits, no lane head). **343 of the 509 are in C3, 166 are not.** The two tables have different proposal sets, so they cannot be pooled. A full-navtest P2 table does not exist yet: `p2_full_table/dumps/dump_main_c00.npz` holds 990 tokens, no `P2FULL_DONE` marker at read time.

## 2. (b) NAVSIM v1.1 sub-scores
* `table.npz['sub'] [1058, 64, 6] float64`, columns `[no_at_fault_collisions, drivable_area_compliance, ego_progress, time_to_collision_within_bound, comfort, driving_direction_compliance]` (`sub_names`). `pdms [1058, 64]`, `valid [1058, 64]` all True.
* Produced by the unchanged harness `eval/score_navtest_refe.py`, one single-proposal seam per index k (64 columns, each with all guards PASS: `STATUS.json`). G2 reproduces the banked full-navtest per-token score at the pick: **max |diff| 0.0 on 1058 tokens** (`gate_table.json`).
* DAC takes values {0, 1} only. **DAC = 0 on 22,443 of 67,712 proposals (33.14 %)**; 394 tokens have all 64 proposals DAC = 1, 68 tokens have all 64 DAC = 0. (Descriptive; C3 over-represents the failure stratum, so this is not a navtest rate.)
* ⚠️ **NAVSIM scores the SIMULATED trajectory (LQR + kinematic bicycle), not the proposal's own poses.** `navsim/evaluate/pdm_score.py:97-128` in the v1.1 source the harness used (`navsim-3e8291bfa89ff247231e0227778840cd0a036896`, `PYTHONPATH` in `refe_stage2_sft4_B_u400_navtest_full_manifest.json`). This matters for the instrument; see `AGREEMENT.json`.

## 3. (c) The scorer's per-proposal scores
* `table.npz['logits'] [1058, 64, 7] float32` = raw logits in the order (NC, DAC, EP, TTC, C, DDC, lane). `pick [1058] int64`.
* **B's pick is reproducible: 1058 / 1058** from the logits with `rl_val_eval.select(.., "navsim_v1_lane", space="logit")` (`eval/rl_val_eval.py:61-75`: `NC * DAC * (5 EP + 5 TTC + 2 C) / 12 * p_lane`). Scorer only: no raster and no sub-score lookup at the pick was involved.

## 4. (d) GT drivable-area rasters
* **Store:** `D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest/`, `manifest.json` sha256 `757788495c2a81bc…`, `store_sha256` `985a3bc77f3ab7a96b0dceace48e73b80e5b69b70710958635060939bce0d7c1`. 136 shards `shards/<log>.aux1` (one packed shard per log), **12,146 keys = all of navtest**. **C3: 1058 / 1058 found, 0 missing.** All from the nuPlan DB `ego_pose` (`s = "db"`); maps: LV 233, BOS 310, PIT 276, SG 239 tokens.
* **Read API:** `refe/aux_label_store.py:275` `read_label(store, log, token) -> int8 [160, 128]` (decode `:268`). Entry header carries the rear-axle origin `p = [x, y, heading]` in the map frame.
* **Geometry** (builder: `refe/aux_label_store.py`):
  * extent x in [-8, 72), y in [-32, 32), **0.5 m cells, 160 x 128**, array index `[ix, iy]` C-order, cell centre `(X0 + (ix + .5) * RES, Y0 + (iy + .5) * RES)` (`:59-60`, `:77-83`);
  * frame = rear axle at t0, **x forward, y LEFT** (`:3`, `:59`);
  * value = **signed distance, int8, unit 0.1 m, clipped +-12.7 m, POSITIVE = inside drivable**, never 0 (`:61`, `:106-129`); mask = `value > 0`;
  * ego-centre = rear axle + 1.461 m along the heading (`:64`, `:142`); polygons within 100 m of the ego CENTRE (`:63`).
* **Drivable classes:** ROADBLOCK + INTERSECTION + CARPARK_AREA (`:186-191`, `NuplanBackend.polys`). `ROADBLOCK_CONNECTOR` is queried and dropped. NAVSIM's own `PDMDrivableMap.from_simulation` (`pdm_occupancy_map.py:130-190`, v1.1 source) adds only the roadblock polygons, the interior lane edges (types LANE / LANE_CONNECTOR, which are NOT in `drivable_area_idcs`), INTERSECTION and CARPARK_AREA, so the sets are equal.
* **Fidelity of the label against NAVSIM's exact polygon rule** (`raw/2026-10-12-aux-label-store/s0_gate_pre.json`, 832 shifted proposals from 64 train frames): whole-proposal verdict agreement **97.6 %** (bilinear). The navtest build repeats the path-A/B gate: 0 cells differ over 136 samples (`raw/2026-10-13-aux-navtest-store/s0_gate_navtest.json`).
* **Provenance cross-check** (`raw/2026-10-16-aux-shift-locate/SHIFT_LOCATE.md` section 1): the train and navtest stores come from the same code and give the same label for the same pose (cell-sign agreement 0.987 cross-store vs 0.987 within-train).

## 5. (e) PREDICTED rasters
All are **held-out by log** (5 folds, `fold = int(sha256(log_id).hexdigest(), 16) % 5`, `folds.json` sha256 `65b652ba916b2af9…`; the probe's own folds hash is `20dec3431663de22…`). Each unit holds `held_idx` (row indices into the TABLE order), `logit [n,40,32] f32`, `sdf [n,40,32] f32`, `meta_json`, `loss_curve`. **Union of the 5 folds = 1058 / 1058 tokens, each exactly once.**

| config | dir | updates | held-out IoU x in [0,40) | note |
|---|---|---|---|---|
| **A2: V (visual tap), seed 0** | `D:/Projects/TanitAD-artifacts/scene_tap_probe_a2/units/V_s0_f{0..4}.npz` | **10,000** | **0.6752** (A2_RESULT) | sha256 `89b5ca53…`, `47c5d6a0…`, `2a3b55a3…`, `f2f5d5cf…`, `af015544…`; 2.1–2.6 MB each |
| probe V / S / C | `…/scene_tap_probe/units/{V,S,C}_s*_f*.npz` | 2,000 | V 0.635, S 0.609, C 0.326 | V has 3 seeds; C = label-shuffle control |
| A3R: V seed 0 | `…/scene_tap_probe_a3r/units/V_s0_f*.npz` | **400** | 0.428 | ⚠️ the reader's `A3R` folder prints "U10k" in `A2_RESULT.md`; the units are 400 updates (`meta_json.info.updates`) |
| S1 heads H1/H2 x 3 seeds | `…/s1_read_cache/arms/H*-s*/aux_drv.pt` (29.3 MB) | 400, trained on NAVTRAIN | 0.50 on C3 | weights exist; navtest predictions regenerable (scratch copies `Temp/shf_scratch/navtest_head_pred_H1-s*.npz`, sha256 in `INVENTORY.json`) |

* **Grid:** 40 x 32 cells of 2 m over the SAME frame as the GT store (x in [-8, 72), y in [-32, 32), `[ix, iy]`). `sdf` is in metres = `6 * tanh(.)`, so it saturates at +-6 m (GT clips at +-12.7 m). `logit > 0` and `sdf > 0` disagree on **2.0 %** of cells (sign agreement 0.9797 on A2): the mask head and the SDF head are not the same raster, and a rule must name which one it reads.
* **Frame validated against the GT store** (`pred_geometry_check.py` -> `PRED_GEOMETRY.json`): the identity convention reproduces A2's banked held-out IoU **0.675218** (computed 0.6752177) and the 2k run's 0.634891 (computed 0.6348907). A y-flip reads 0.321, an x-flip 0.412, a 1-cell x shift 0.663–0.664, a 1-cell y shift 0.621–0.625. So the predicted rasters sit in the GT frame.
* ⛔ **No head WEIGHTS were banked for the probe family** (A2 / A3R / 2k units hold predictions and loss curves only). Those heads cannot predict any token outside C3. Only the S1 heads (trained on navtrain) have weights.
* ⛔ `D:/Projects/TanitAD-artifacts/aux_navtest_forward/c3_full/head_pred_H2-s0.npz` is the **FIXTURE head, never trained** (plumbing only). Do not read it as a prediction.
* **Inputs for any NEW head on C3** (fp32, bit-checked against the planner): `aux_navtest_forward/c3_full_ctx/ctx_<token>.npy` `[7680, 256]` (7.86 MB each, 1058 files + 1058 `logits_*.npy`; the `visual_ctx` tap) and `c3_full_scene_ctx/` (`[64, 256]`, manifest sha256 `05ca8b2766cf3f6e…`). Both cover exactly the 1058 C3 tokens.

## 6. (f) Footprint and NAVSIM's DAC
* **Footprint:** nuPlan Pacifica, `get_pacifica_parameters()`: half-length **2.588**, half-width **1.1485**, rear-axle-to-centre **1.461** m (length 5.176, width 2.297). Verified by import in the `driverl-eval` venv; same constants at `refe/navsim_dac.py:35-37`.
* **NAVSIM DAC** (v1.1 source `navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py`): a step is non-drivable when **fewer than 4 of the 4 corners** (centre ignored) lie in at least one drivable polygon (`:284-286`); DAC = 0 if ANY step is non-drivable (`:355-358`). Steps = the 41-state 0.1 s grid over 4 s including t = 0. Polygons: ROADBLOCK, INTERSECTION, DRIVABLE_AREA (never populated), CARPARK_AREA within `map_radius = 100` of the ego centre (`navsim/planning/metric_caching/metric_cache_processor.py:46`; `refe/navsim_dac.py` docstring). **The states are the simulator's output** (`pdm_score.py:109`).
* Repo restatement: `refe/navsim_dac.py` (inherited 100.00 % agreement with the harness on 25,600 proposals, MODEL_REGISTRY 14.1 departure 6). Re-derived here from NAVSIM v1.1's OWN code (simulator + scorer on the metric cache, `tracked_dump.py`): **67,712 / 67,712** table DAC values reproduced (`AGREEMENT.json` B_decomposition.R0).

## 7. RL-val
* **Exists.** `C:/Users/Admin/AppData/Local/Temp/rls1dl/data/` (downloaded from `Sayood/tanitad-refe-rls1-c0`; Temp, so single-copy and perishable): `rlfan_sets_val.tar.gz` (sha256 `c6fa51f4…`, 29.4 MB), `rlfan_pdm_val.tar.gz` (`3dd6cba3…`, the SAME hash `FIT.json` records), `rlfan_terms_val.tar.gz` (`6f12b90f…`), and `final/scores_step0_A.npz` (`6332fcc6…`, the hash `FIT.json` records).
* **Shape:** 3,000 sets, 75 logs, 64 proposals per set = 192,000 proposals. Sets carry `traj [64,20,2]` + `yaw [64,20]` at 5 Hz (REFe 20-pose grid; `navsim_dac.to_navsim` maps it to the 8-pose grid). `scores [3000,64,7]`. Keys `(log, token)` match across sets / pdm / scores (3000 / 3000).
* **Proposals come from the pre-P2 base REFe checkpoint** (`fan_ckpt_sha256 4246a320…`, ckpt step 10075), scored by u400 (`SELRULE-DAC` A1). DAC labels are the repo's PDM labeller (`navsim_dac_and_comfort`), **not the NAVSIM harness**.
* **GT rasters:** these are NAVTRAIN keys; **3000 / 3000 are in the train store** (`label_store/`, `store_sha256 4a17db67…`; 103,139 keys).
* **Predicted rasters: none held-out.** The 75 RL-val logs are disjoint from S1's 24 held-out logs (0 in common), so every existing head either trained on them or never saw them for an unrecorded reason. Which of S1's 1,133–1,145 drawn logs contain the RL-val logs is not recorded (UNVERIFIED). A clean RL-val prediction needs a head trained without those 75 logs on navtrain `visual_ctx`, and the dev box holds 6,044 of 412,148 train images (INHERITED, `S1_DATA_AUDIT.md`). **Blocked on navtrain features.**

## 8. Alignment keys and mismatches (all MEASURED)
* Token strings: table = proposals = `tokens.json` = folds rows = label store (1058 / 1058) = visual-ctx cache = scene-ctx cache. Order differs ONLY in `tokens.json`.
* `folds.json` rows order == table order, so `held_idx` indexes table order directly.
* C3 vs P2-509: 343 shared, 166 P2-only. C3 vs P2-full: not comparable until `P2FULL_DONE`.
* GT raster vs NAVSIM: the store's polygons equal NAVSIM's drivable set, but NAVSIM judges the simulated states (section 2).
* The numbers 0.675 (A2), 0.635 (2k) and 0.50 (S1) are three different heads on the same 1058 tokens; the 0.50 head was trained out-of-domain.
