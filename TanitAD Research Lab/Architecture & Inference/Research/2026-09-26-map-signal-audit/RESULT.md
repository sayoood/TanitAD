# RESULT — Map-signal audit: was refcv6's map head trained to extract all 8 SAM3 classes, and does NEW-2 fix it?

**For:** the Master Mind (the single committer), who relays to the PI.
**Date:** 2026-09-26 ~21:50 to 2026-09-27 ~00:30 Berlin.
**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/`.

**Code:**
- tip `agent/arch-inf-20260803` at `2c510fb` (clean tree `C:/Users/Admin/msa_tree_2210`; on Thor `/home/nvidia/msa_2358/tree`; `tanitad.__file__` asserted inside it in both);
- the tip has since moved to `a0dc97b`.
- **Why 2c510fb and not the moving tip:** FIX-3 (`ab436ee`) makes the trunk honour `--equalize-bottom-rows`. A tip tree would therefore rebuild refcv6 with an equalising trunk, which is NOT the model as trained.
- The files this audit traces are unchanged between `284393c` (refcv6's launch) and `2c510fb`: `refcv6_perception_branch.py`, `bev_encoder.py`, `bev_lift.py`, `semantic_map_gt.py`, `perception_targets.py` (`git diff --stat` empty), and the map block of `refc_v3_train.py` (same lines, shifted).

**The PI's questions (verbatim, 2026-09-26):**
- *"our model is seldomly detecting other sematic classes, no roadmarkings, no roadmarking infrstructure in intersections etc... We need to analyze it"*
- *"Did we review the wiring, the architecture, the training and the training signal flow for the map heads to verfiy its trained to extact all sematic classes?"*

## HEADLINE

**Short answer: NO.** refcv6's map head was trained ON every class but not TO EXTRACT them, and nothing measured the difference.

**MEASURED at step 38,000** (Thor GPU, as trained; 16 fixed eval windows from 16 clips; 105,369 supervised cells; every control reproduced its analytic value to ≤ 8e-6).

- **Every class got signal.** The five thin classes (lane/road line, crosswalk, arrow/text, non-drivable edge, hatched) carry:

  | quantity | thin-class share |
  |---|---:|
  | label mass | **6.9 %** |
  | map loss | **30 %** |
  | gradient at the map logits | **16 %** |
  | gradient at the lift output | **15 %** |
  | gradient into the trunk's stride-16 feature | **11 %** |

- **The head learned some scene-specific evidence for each thin class.** Scored against a GT-shuffled control, its cross-entropy per unit of class mass is lower by 0.7–2.4 nats.
- **It never became confident.** Mean probability on a class's own cells, with the maximum anywhere in 16 windows in brackets:

  | class | mean probability on own cells | max anywhere |
  |---|---:|---:|
  | lane | 0.11 | 0.72 |
  | crosswalk | 0.058 | 0.53 |
  | hatched | 0.015 | 0.10 |
  | edge | 0.012 | 0.13 |
  | arrow | 0.005 | 0.11 |

- **So the argmax (the map) names lane on 5.6 % of lane cells, crosswalk on 1.2 %, and arrow/text, edge and hatched NEVER.**

**Why, ranked by measured size (§8):**

1. **Localisation + soft-target argmax.**
   - The stride-16, 0.5 m lift cannot place a 10–30 cm marking. EXACT geometry: 0.35–1.25 m of road per feature laterally over 0–40 m. The whole 20–40 m road plane falls in 2 feature rows, and 40–60 m in 1.
   - Under that blur the soft 0.5 m target plus argmax gives thin classes ~0. The ANALYTIC ceiling with PERFECT information at σ = 0.5 / 1.0 m is lane 0.26 / 0.05 and edge 0.00 / 0.00.
   - The measured lane probability on lane cells, 0.11 against a median label fraction of 0.40, is that regime.
2. **The 0.5 m target erases the edge class.** 0.9 % of its area survives (MEASURED census).
3. **Signal share in the shared path.**
   - Thin classes get 11 % of the map's trunk-feature gradient: arrow **0.14 %**, hatched 1.7 %.
   - The map is only **3.4 %** of the total loss (MEASURED).
   - At init its gradient into the shared lift, BEV encoder and trunk was **12× / 11× / 145×** smaller than box3d's (MEASURED).
   - This is the INFERRED reason arrow and hatched stay below even the σ = 1 m ceiling.
4. **Far-range labels are non-causal and sub-pixel.** A 0.4 m stop line is 0.75 px tall at 20 m (EXACT).
5. **Why nobody saw it.**
   - Only drivable IoU was ever logged: `with_metrics` was never passed, so there are 0 per-class keys in 4,621 rows.
   - The per-head gradient-reach instrument never wrote a row. It is an off-by-one: 0 `ga_*` keys.

**NEW-2 (10 cm) review (§4).** It is well built but carried three HIGH/MED defects:
- the raw argmax of median-frequency-weighted logits (D1);
- an invalid must-fail arm (D2);
- dead gradient-reach logging (D3).

D1, D2, D3 and D6 are **ACCEPTED as SPEC_REFCV7 §9 (A4, `b3f7ea6`) but NOT yet in the builder's code.**

**Stride 8** (D9): adequate laterally to ~20 m. Beyond ~15 m the 416-row image, not the stride, is the limit.

**Launch-gate checks (§5–§6):**
- **G-MAP-OVERFIT**, pre-registered before any run: `raw/PREREG_G_MAP_OVERFIT.md` + `raw/gmo_spec.json`.
  - 16 TRAIN frames from 4 clips.
  - Pass: big classes ≥ 0.85 IoU, thin classes ≥ 0.50 (0–20 m, pooled, declared rule).
  - Must-fail arms: `lane_w0` and `s8_zeros`.
- **G-DVB/map-logging**: `raw/LOGGING_SPEC_MAP10.md`; its regression arm is drivable-only logging.

**Architecture (§7):** I recommended option (a), keeping the 0.5 m head as a declared internal auxiliary. **The PI chose (c)** on 2026-09-27: one 0.25 m lift for everything, and a MAXIMAL range (SPEC_REFCV7 §11, A6, `546f34c`).

**Map extent (§9A, A6):** the pre-registered census on all 4,369 TRAIN clips (78,321 frames; byte-identity controls 24/24) selects **100 m ahead × ±30 m**.
- It is robust to every sensitivity variant.
- The limits come from the SAM3 renderer's 35 m camera range (`R_MAX`, sam3map_render_v5m.py:33) plus the drive remaining in each clip.
- A `/3` re-export takes ~1.5 h on 6 Thor workers and ~19.6 GB, and it is byte-identical to `/2` inside the old window only with ANCHORED coordinates.
- The decoder's saved activations grow to ~21 GB at b16 (~1.1 GB with `grad_ckpt`); the s/step cost is UNMEASURED.

**⛔ INTEGRATION NEEDED (Master Mind).**
- (i) The builder's `code/fix/` must implement D1, D2, D3 and D6 before any G-MAP run.
- (ii) `map_hires_overfit.py` must accept `raw/gmo_spec.json` (PREREG §9a).
- (iii) The gate agent wires G-MAP-OVERFIT and G-DVB/map-logging from the two `raw/*.md` specs.
- (iv) The PI decides option (a) / (b) / (c).

---

## 1. Task 1 — refcv6's map signal path, end to end (every hop cited)

**Evidence class.** PUBLISHED-CODE, read at the tip. A hop marked **MEASURED** was measured here, on `metrics.jsonl` or on real geometry. The refcv6 run is **refcv6-r101-s0** (quoting stamp from `MODEL_REGISTRY.md`):
- steps ≤ 34,500: "F3 detach-only, F4 on the last layer only; labels ~0.37 s early"; later steps: "hybrid: F3 + true label clock from 34,500";
- every step: "trunk C26 equalisation OFF (declared 43, dropped)".

| # | hop | what happens (file:line) | class-specific signal lost here? |
|---|---|---|---|
| 1 | **SAM3 GT file** | `semantic_maps/gt/<clip>.sam3mapgt.npz`, schema `tanitad.sam3_map_gt/2`. `cart_frac` `[T,9,120,64]` uint8 = cell fraction × 255 at 0.5 m; channel order at `semantic_map_gt.py:73-75`, `not seen` = channel 8 (`:77`). The file ALSO carries `fine_codes` `[T,600,320]` at 0.1 m, but *"(polar_frac, polar48_frac, fine_codes are not read here)"* (`:21`), and `_REQUIRED` (`:88-94`) does not list it. | ⚠️ **YES: the 0.5 m target.** MEASURED (census, `…/2026-09-26-refcv7-map-hires/raw/map_res_census.json`, 137 files, 2,867 frames): under the eval's ≥ 0.5 rule the 0.5 m grid keeps **0.9 %** of non-drivable-edge area, 61.8 % of lane lines, 64.7 % of arrows and 77.7 % of crosswalks. |
| 2 | **label semantics** | The labels are non-causal (clip lifetime, *"labels use every frame of the clip"*, quoted at `refcv6_perception_branch.py:525-533`). A far cell is labelled with what the ego saw when it got close. | ⚠️ **YES, beyond ~20 m, by physics.** ANALYTIC-EXACT (`raw/lift_resolution.json`, 137 real extrinsics): a 0.40 m stop line is **0.75 px** tall at 20 m and **0.32 px** at 30 m; a 3 m-deep crosswalk is 1.3 px at 40 m; a 0.15 m line is 1.9 px wide at 40 m. The label still says "marking". |
| 3 | **reader** | `ClipMapGT.read` (`semantic_map_gt.py:317-333`): `cart = u8/255` (`:328`); `seen = 2·(255 − not_seen) ≥ 255`, i.e. ≤ 50 % not-seen (`:330-331`); raw frame = stacked row + n_stack − 1 (`:128-135`). | no |
| 4 | **targets per window** | `MapGTStore.frames_for_windows` (`perception_targets.py:173-186`); `V3Dataset._map_item` (`refc_v3_train.py:2937-2960`) at the window's NOW `t + w − 1` (`:3181-3187`, with `map_ep` for the lift). Coverage refusal `enable_map_gt` (`:2867-2935`). MEASURED (`config.json`): train 746,946 windows over 4,369 clips, `frac_ok` **1.0**; eval 0.9856 (2 clips lack GT). | no |
| 5 | **trunk feature** | `fmap_s16` = the stride-16 map (resnet101 layer3, **1024 × 26 × 64**) of the LAST frame of the window only (`refc.py:4026-4032`: `s16_all.reshape(b, w, …)[:, -1]`). | see hop 7 |
| 6 | **BEV lift geometry** | `LiftGeometryBank` (`refcv6_perception_branch.py:219-306`) → `build_lift_geometry` (`bev_lift.py:146-184`), per clip, 4 heights (0, 0.5, 1.5, 2.5 m) (`:69`), 120 × 64 @ 0.5 m, stride 16. **The lift got `equalize_bottom_rows` 43** (`refc_v3_train.py:6922-6925`, `config.json` stamp), so the bottom 43 image rows count as unobserved. **The trunk got 0**: an undeclared attribute at `:381` is dropped by `dataclasses.replace` at `:459` (D-REFCV6-EQUALIZE-DROPPED). | minor. MEASURED (real geometry): the strip covers the road plane up to rig x ≈ **5.25 m** (p10–p90 5.25–6.45). **0.86 %** of 0–20 m cells lose their road-plane sample. `map_valid` is unchanged (0.669 with and without). |
| 7 | **BEV lift** | `BEVLift.forward` (`bev_lift.py:248-270`): bilinear `grid_sample` of `fmap_s16` at each height; invalid samples zeroed; heights concatenated (4 × 1024 = 4,096 ch); 1×1 conv → 128; cells with no valid height get the learned `unobserved` vector. | ⚠️⚠️ **YES, and it is the resolution floor.** ANALYTIC-EXACT (`raw/lift_resolution.json`): the road plane of the whole **20–40 m band falls in 2 distinct stride-16 feature rows** (p10–p90 2–3), and **40–60 m in 1** (1–2); 0–20 m gets 12. One feature is **0.60 m wide at 20 m** and 1.25 m at 40 m (lane-centre). **Longitudinally it spans 8.5 m at 20 m** and 36.9 m at 40 m. A 0.15 m line is one quarter of a feature at 20 m. |
| 8 | **BEV encoder** | `BEVEncoder` (`bev_encoder.py:154-216`): 3×3 stem + 4 dilated residual blocks (1, 2, 4, 8), GroupNorm, stride 1, ±31 cells = ±15.5 m receptive field; out 96 ch. **Shared**: its `bev_feats` also feed box3d (`Box3DMemory`) and the tactical decoder's 30 × 16 BEV tokens (`refcv6_perception_branch.py:461`, attached per PI R3). | ⚠️ **YES, dilution.** MEASURED pre-launch (`…/2026-09-17-refcv6-perception-training/raw/grad_reach_isolated.json`, at init): `grad_abs_sum` on the BEV encoder is **9,176 from the map loss vs 103,814 from box3d** (11.3×); on the lift **8,320 vs 98,586** (11.8×); on the trunk **12,511 vs 1,818,182** (145×). |
| 9 | **map head** | `MapHead` (`bev_encoder.py:219-239`): ONE 1×1 conv, 96 → 9, **873 parameters**. | no (a linear read-out: it can only separate what hop 8 encodes) |
| 10 | **loss** | `map_loss_row` (`refcv6_perception_branch.py:513-573`) → `map_soft_ce` (`bev_encoder.py:269-361`). The formula is `L = (1/n) Σ_cells Σ_{c=0..8} −p_c log softmax(z)_c`, with `p` = `cart_frac` renormalised (`:334`), n = seen ∧ lift-valid cells, **`class_weight=None`** (default, never set). **The `not seen` channel IS a predicted class** (9 logits; `p_8` ≤ 0.5 on a supervised cell; mass share 0.045 %). | ⚠️ **YES.** ANALYTIC (§2): at init the five thin classes carry **2.94 %** of the logit gradient. **Soft targets + argmax** give an ANALYTIC ceiling, with a model that has the right information but localises to σ = 0.5 m, of lane **0.26**, crosswalk 0.57, edge **0.00**. |
| 11 | **cell mask** | The trainer passes `lift_valid=map_valid` (`refc_v3_train.py:4472-4481`; flag default ON at `:6903`, `:9359-9368`; the same code at the launch `284393c:4377-4386`). | no. MEASURED (`metrics.jsonl`, 765 rows): **9,821,736 of 89,040,848** seen cells (**11.03 %**) were removed as unreachable, matching the design's 11.048 %. The removed cells are not thin-class-rich (lane mass share 1.635 % on seen → 1.694 % on seen ∧ valid). |
| 12 | **weight in the total loss** | `loss = loss + w_map · map` (`refc_v3_train.py:4482`), **`--w-map 1.0`** (argv). MEASURED (`metrics.jsonl`, median per band): map / total = **2.6 %** (steps ≤ 1k), 2.5 % (1–5k), 2.7 % (5–10k), 3.1 % (10–20k), 3.6 % (20–30k), **3.8 %** (30–34.5k), 3.4 % (34.5–38.25k). At 34.5–38.25k: map 0.656 of a total 19.0; box3d 5.20, cascade 2.59, goal_tac 4.20. | the map is a small term |
| 13 | **optimiser** | `build_optimizer` `--opt dd` (`refc_v3_train.py:6289-6333`), built at `:7720` after the branch (`:6908-6947`). So the branch is in the head group: AdamW lr 1e-4, wd 1e-4; the trunk at 0.5×. Global `clip_grad_norm_(…, 10.0)` at `:8248`. | ⚠️ partly. AdamW normalises per PARAMETER, so each class's row of the 1×1 head gets full-size updates. The shared lift, BEV encoder and trunk receive the SUM (hop 8). |
| 14 | **gradient reach** | `grad_reach_report` (`refcv6_perception_branch.py:614-646`), called at `refc_v3_train.py:8243-8247` only when `step % log_every == 0` BEFORE `step += 1`, and written when `step % log_every == 0` AFTER it (`:8250-8251`). | ⚠️⚠️ **NEVER LOGGED.** MEASURED: **0 `ga_*` keys in 4,621 rows.** With `--log-every 50` the two conditions never meet (a new finding; the builder's `code/fix/` inherits it for `ga_mh_*`). |
| 15 | **what was logged** | Train and eval (`:4506-4537`; the eval re-uses `compute_losses_v3`, `:8350-8404`): `map`, `map_iou_drivable`, `map_{gt,pred}_drivable_frac`, `map_pred_drivable_prob_mean`, `n_map_cells*`. **Why only drivable:** the call at `:4477-4481` never passes `with_metrics=True`, which is the only path to `map_iou_{i}` (`refcv6_perception_branch.py:566-572`). The drivable block was hand-written for the collision gate (the 2026-09-18 comment at `:4490-4503`). The design doc declared per-class reporting (*"MODEL_REGISTRY-bound numbers are read per class"*, `bev_encoder.py:56-59`), but it was never built. The Watch plots drivable IoU and the CE only (`build_watch_refcv6.py:648-656`). | ⚠️⚠️ **the reason the collapse went unseen for 38,000 steps.** Even if logged, the eval aggregation (a mean of per-batch values over 8 batches, `:8401`) would have biased rare-class IoU, because `map_metrics` skips classes with no union. |

**Flagged hops, in order of the class-specific signal they cost:**
- 7 (stride-16 lift resolution);
- 1 (0.5 m target: edge, part of lane);
- 10 (unweighted soft CE + argmax);
- 8 (shared encoder dominated by box3d);
- 2 (non-causal far-range labels);
- 14/15 (the instruments that would have shown it).

## 2. Task 2 — analytic per-class signal (`raw/analytic_class_signal.json`, `code/analytic_class_signal.py`)

**Function class and assumptions.** These are the per-cell LOGIT gradients of the losses as implemented:
- soft CE: `q − p`;
- weighted hard CE: `w_y (q − e_y) / Σw`.

**Attribution.** Since Σp = 1, `q − p = Σ_c p_c (q − e_c)`. So class c's part of a cell's gradient is `p_c (q − e_c)`, with size `A_c = Σ_cells p_c ‖q − e_c‖`. Two companions are reported: the own-logit pull `P_c = Σ p_c (1 − q_c)` and the loss share `L_c`.

**What this covers, and what it does not.** Everything upstream sees this vector through a class-independent Jacobian, so this is the split at the logits. Task 3 measures the split at the lift output and the trunk feature on the real model.

**States (explicit functions of the label; no training):**
- **S0**, uniform init;
- **S1**, the constant prior `q = m`;
- **S2**, "drivable-majority": `q = 0.95·p̃ + 0.05/C`, where `p̃` moves all thin-class mass to drivable. This is a model that is right about the big classes and never predicts a thin one, i.e. the refcv6@35k picture.

**Data.** The eval kit's 137 GT files, every 10th frame (2,867 frames). This is a **proxy** for the TRAIN distribution: the only GT on the dev box.

**Controls (read their known values):**
- the S0 shares equal the label-mass shares to **1.0e-15**;
- the S0 loss = **2.1972245773** = ln 9;
- the S1 loss = **1.2261834366** = H(mass).

### 2a. refcv6 as trained: 0.5 m, soft CE, 9 classes, seen ∧ lift-valid (17,819,701 cells)

| class | label mass | S0 share | S1 share (A) | **S2 share (A)** | S2 loss share |
|---|---:|---:|---:|---:|---:|
| seen-no-class | 26.23 % | 26.23 % | 28.47 % | 21.21 % | 11.71 % |
| drivable | 35.02 % | 35.02 % | 33.55 % | 19.51 % | 9.99 % |
| lane / road line | 1.694 % | 1.694 % | 2.34 % | **19.26 %** | 37.02 % |
| crosswalk | 0.532 % | 0.532 % | 0.74 % | 6.07 % | 11.64 % |
| arrow / text | 0.068 % | 0.068 % | 0.095 % | 0.78 % | 1.49 % |
| non-drivable edge | 0.585 % | 0.585 % | 0.82 % | 6.15 % | 12.79 % |
| hatched | 0.061 % | 0.061 % | 0.085 % | 0.69 % | 1.33 % |
| sidewalk / verge | 35.77 % | 35.77 % | 33.85 % | 25.98 % | 13.79 % |
| not seen | 0.045 % | 0.045 % | 0.063 % | 0.35 % | 0.24 % |
| **5 thin classes** | **2.94 %** | **2.94 %** | **4.07 %** | **32.95 %** | **64.3 %** |

**Readings:**
1. **At init the thin classes are starved:** 2.94 % of the map loss's logit gradient. That is inside a map loss that is ~3 % of the total loss, and whose gradient on the shared BEV encoder is 11× smaller than box3d's (hop 8).
2. **CE self-balances once the big classes are fit.** At S2 the thin classes carry **33 %** of the gradient and **64 %** of the loss. So unweighted CE does NOT permanently starve them at the logits. Whether refcv6 ever reached that regime is what task 3 measures.
3. **Entropy floor of the soft target: 0.079 nats/cell.** S2's loss is 0.238. refcv6's training map loss at 34.5–38.25k is **0.656** (MEASURED, `metrics.jsonl`). The run was nowhere near even the "big classes perfect" state.

### 2b. NEW-2 as designed: 10 cm, hard CE, 8 classes (500,909,554 seen cells)

The builder uses Eigen & Fergus **MF-present**. MF-global is shown for contrast.

| class | share of seen | MF-global w | MF-present w | S0 share unweighted / MF-global / MF-present | S2 share unweighted / MF-global / MF-present |
|---|---:|---:|---:|---|---|
| seen-no-class | 26.83 % | 0.041 | 0.060 | 26.8 / 12.5 / 17.6 % | 20.2 / 1.5 / 2.7 % |
| drivable | 33.75 % | 0.033 | 0.048 | 33.8 / 12.5 / 17.6 % | 25.4 / **1.5** / **2.7** % |
| lane | 1.635 % | 0.677 | 0.896 | 1.6 / 12.5 / 16.1 % | 15.4 / 19.1 / 31.1 % |
| crosswalk | 0.515 % | 2.15 | 1.13 | 0.5 / 12.5 / 6.4 % | 4.8 / 19.1 / 12.4 % |
| arrow / text | 0.066 % | 16.8 | 7.22 | 0.07 / 12.5 / 5.2 % | 0.6 / 19.1 / 10.1 % |
| edge | 0.580 % | 1.91 | 2.70 | 0.6 / 12.5 / 17.1 % | 5.5 / 19.1 / 33.2 % |
| hatched | 0.062 % | 17.95 | 3.92 | 0.06 / 12.5 / 2.7 % | 0.6 / 19.1 / 5.1 % |
| sidewalk | 36.56 % | 0.030 | 0.043 | 36.6 / 12.5 / 17.3 % | 27.5 / 1.5 / 2.7 % |

**Readings:**
- **The clip at 25 does not bind** under either definition: the maximum is 17.95 for MF-global and 7.22 for MF-present.
- **Weights fix the init starvation.** At S0 the thin classes hold 62.5 % of the gradient under MF-global and 47.5 % under MF-present.
- **They over-correct at the converged state.** The big classes, the ones the planner needs, fall to 1.5–2.7 % each.
- The builder's DRY RUN weights (5 eval clips) differ from this 137-clip proxy by up to 3.8× per class, e.g. hatched 1.035 vs 3.918. **Only the TRAIN computation counts.**

### 2c. The decision rule is the lever the design missed (ANALYTIC ceilings, 0–20 m, 675 frames)

A model trained with a weighted CE outputs `q_c ∝ w_c · P(c|x)`. The raw argmax then predicts a thin class over drivable when `P_c / P_drv > w_drv / w_c`. Under MF-present that is:
- lane at a **5.0 %** posterior;
- crosswalk at 4.0 %;
- arrow at **0.65 %**;
- edge at 1.7 %;
- hatched at 1.2 %.

**The ceiling.** The table uses the label itself, blurred by σ (a model with the right information and localisation error σ).

| rule | σ | lane | crosswalk | arrow | edge | hatched | drivable |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 m soft, argmax (refcv6) | 0.25 m | 0.76 | 0.88 | 0.81 | 0.17 | 0.91 | 0.98 |
| 0.5 m soft, argmax (refcv6) | 0.50 m | **0.26** | 0.57 | 0.34 | **0.00** | 0.57 | 0.95 |
| 0.5 m soft, argmax (refcv6) | 1.00 m | **0.05** | 0.39 | 0.09 | 0.00 | 0.38 | 0.92 |
| 10 cm, calibrated argmax | 0.10 m | 0.91 | 0.95 | 0.89 | 0.73 | 0.92 | 0.99 |
| 10 cm, calibrated argmax | 0.20 m | 0.68 | 0.83 | 0.70 | **0.04** | 0.82 | 0.97 |
| 10 cm, calibrated argmax | 0.30 m | 0.49 | 0.70 | 0.54 | 0.00 | 0.74 | 0.95 |
| 10 cm, MF-present raw argmax | 0.10 m | **0.52** | 0.60 | 0.43 | 0.21 | 0.54 | 0.91 |
| 10 cm, MF-present raw argmax | 0.20 m | 0.38 | 0.50 | 0.27 | 0.13 | 0.42 | 0.85 |

**Readings:**
- **(i)** At 0.5 m with the stride-16 localisation (0.35–1.25 m per feature), the lane-line ceiling is 0.05–0.26 and the edge ceiling is 0. refcv6's 0.009 / 0.000 sit at the floor this sets.
- **(ii)** **Hard labels at 10 cm do NOT by themselves remove the out-voting.** Under σ ≥ 0.2 m the calibrated posterior at a line is < 0.5, and edge collapses to 0.04. The lever is localisation (features), then the decision threshold. `…/2026-09-26-refcv7-map-hires/RESULT.md` §4.1 says *"the soft-target out-voting … disappears by construction"*; that holds only for σ ≲ W/2.
- **(iii)** **The raw argmax of a median-frequency-weighted head LOSES IoU** against the prior-corrected rule `argmax(z − log w)` for every class except edge at σ ≥ 0.2 m, and it costs drivable 0.08–0.12. This bears directly on BAR-M7-4.

## 3. Task 3 — MEASURED per-class loss and gradient share, refcv6 @ 38,000

**Where it ran.** The dev-box RAM gate stayed closed from 23:01 to 23:55 (3.3–7.0 GB free), so it never launched there. The Master Mind authorised the run on Thor ("GO-THOR", Thor idle).

**Setup:**
- **when and where:** Thor GPU, 2026-09-26 23:59, 27.7 s, `/home/nvidia/msa_2358/`;
- **code:** a clean `2c510fb` tree shipped by scp with md5 verification (tarball `ec162e92…`);
- **model:** the tanitad-train venv, no installs. Checkpoint `ckpt.pt`, md5 `5a2e7222…`; config md5 `a3193a46…`.
- **build checks:** strict load 0 / 0, `param_breakdown` equal, `equalize_bottom_rows` {argv 43, trunk as built 0, lift 43}, which is the run as trained. Trunk bf16 as trained.

**Instrument** (`code/measure_class_signal_38k.py`):
- The forward is the trainer's own `compute_losses_v3`, stopped by a forward hook once `model(...)` returns.
- The perception branch is then re-run from a detached copy of `fmap_s16`, with autograd on ACTIVATIONS only.
- The map loss is split exactly: `L = Σ_c L_c`, `L_c = Σ_cells p_c (−log q_c)/n`, over the trainer's cells (seen ∧ `map_valid`).
- Each `L_c` is backpropagated to the logits, the lift output and `fmap_s16`. The "share" is the additive projection `⟨g_c, g⟩/‖g‖²`, which sums to 1 over classes.

**Windows** (fixed by a rule over GT and geometry only, before any model output was read; `raw/thor_2358/measure_class_signal_38k_selection.json`):
- **8 NEUTRAL:** the median window of the 8 smallest-sha12 eval clips.
- **8 ENRICHED:** 2 per class for crosswalk, arrow/text, hatched and edge (max label mass), from 8 other clips.
- 16 clips in all. One clip is one episode, so **no interval is quoted.**

**Controls, all reproduced their known values:**

| control | max error |
|---|---:|
| re-run logits vs in-forward logits | 0.0 |
| loss vs `map_loss_row` | 0.0 |
| uniform-logit loss vs ln 9 | 4.4e-7 |
| `L_c` vs mass_c · ln 9 | 4.8e-7 |
| gradient vs analytic `(1/9 − p)/n` | 1.5e-11 |
| per-class gradient norm (relative) | 3.1e-7 |
| prior-logit loss vs H(mass) | 2.4e-7 |
| GT-as-logits vs the entropy floor | 7.7e-6 |

**All 16 windows** (105,369 supervised cells; `raw/thor_2358/measure_class_signal_38k_summary.json`). Shares are cell-weighted. "Gain" is the GT-shuffled CE minus the true CE, in nats per unit of class mass; > 0 means the head knows where the class is in THIS scene.

| class | label cells (argmax) | label mass | loss share | grad share @ logits | @ lift output | @ trunk `fmap_s16` | IoU (argmax) | mean q on own cells | mean q where absent | max q anywhere | gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seen-no-class | 14,883 | 14.3 % | 15.4 % | 19.9 % | 11.5 % | 14.8 % | 0.351 | 0.477 | 0.127 | 0.97 | 1.11 |
| drivable | 48,339 | 45.3 % | 32.1 % | 35.5 % | 44.9 % | 44.9 % | 0.592 | 0.647 | 0.202 | 1.00 | 0.47 |
| sidewalk / verge | 35,807 | 33.5 % | 22.3 % | 28.1 % | 28.7 % | 29.7 % | 0.544 | 0.632 | 0.211 | 1.00 | 0.86 |
| **lane / road line** | 2,651 | 2.67 % | 9.2 % | 6.1 % | 4.4 % | 3.5 % | **0.055** | **0.106** | 0.021 | 0.72 | 2.03 |
| **crosswalk** | 2,436 | 2.36 % | 10.3 % | 6.3 % | 7.2 % | 4.3 % | **0.012** | **0.058** | 0.010 | 0.53 | 1.69 |
| **arrow / text** | 377 | 0.37 % | 2.5 % | 1.1 % | 0.6 % | **0.14 %** | **0.000** | **0.005** | 0.001 | 0.11 | 0.71 |
| **non-drivable edge** | 84 | 0.80 % | 4.0 % | 0.9 % | 0.9 % | 1.0 % | **0.000** | **0.012** | 0.006 | 0.13 | 0.79 |
| **hatched** | 788 | 0.74 % | 4.1 % | 2.0 % | 1.9 % | 1.7 % | **0.000** | **0.015** | 0.002 | 0.10 | 2.36 |
| not seen | 4 | 0.02 % | 0.1 % | 0.0 % | 0.0 % | 0.0 % | 0.000 | — | 0.001 | 0.01 | 2.32 |
| **5 thin classes** | | **6.9 %** | **30.0 %** | **16.4 %** | **14.9 %** | **10.6 %** | | | | | |

**The same measurement on the 8 NEUTRAL windows only** (53,012 cells; the less class-enriched read):
- thin mass 4.8 %, loss 22.2 %, gradient 11.2 % at the logits / 9.3 % at the lift / **6.7 % at the trunk**;
- lane IoU 0.111, mean q on lane cells 0.161, crosswalk 0.035; arrow, edge and hatched IoU 0.000.

**How often the argmax names a label cell's own class:** lane **5.6 %**, crosswalk **1.2 %**, arrow/text **0 %**, edge **0 %**, hatched **0 %**. For comparison, drivable 74 %, sidewalk 76 %, no-class 53 %.

**Readings:**
1. **Not starved at the head at 38k.** The thin classes get 2–5× their mass share of the logit gradient: 16 % vs 6.9 %. CE's self-balancing (§2a S2) is partly under way. It is far from the S2 regime because the big classes are themselves poorly fit (drivable IoU 0.59).
2. **The gradient that reaches the TRUNK is thin-poor:** 10.6 % of the map's trunk gradient, arrow 0.14 %. And the map term itself is small against box3d in that shared path (§1 hop 8, hop 12).
3. **The head has direction but no confidence.** It ranks thin-class cells above the rest (probability 5–10× higher on own cells than where the class is absent; gain > 0 for all five). But its probabilities stay in the calibrated-under-blur range. For lane: 0.11 against a label fraction of ~0.4, a blur factor of ~4. That is exactly the state in which an argmax never names a minority class (§2c). **The failure is the decision under blur, fed by coarse features; it is not a dead wire.** The AP probe (the map-video agent's, in flight) will give the ranking view of the same fact.
4. The loss share (30 %) over-states the gradient share (16 %) because −log q is unbounded while |q − p| ≤ 1. "Thin classes dominate the loss" does NOT mean "they dominate the update".

## 4. Task 4 — review of NEW-2 as built (`…/2026-09-26-refcv7-map-hires/BUILD.md`, code at 22:37)

**What is right.** The builder's work is careful:
- the fine reader re-uses every `cart_frac` guard;
- a projection-first lift, pinned equal to `BEVLift` on values and gradients, gives 8× less sampling memory;
- a stride-8 tap re-runs stem→layer2 on the current frame, and a missing `fmap_s8` is a SystemExit (the F3 class);
- the loss uses `ignore_index` 255 with the lift-valid narrowing;
- per-class × per-band instrumentation (`per_class_signal`) has an analytic control and a mutation arm;
- IoU is POOLED in the eval row;
- a `--map-lowres` switch exists.

**Architecture** (MEASURED `param_breakdown`), 559,048 parameters:
- 0.25 m lift, 64 ch, 4 heights;
- encoder with dilations (1, 2, 4, 8, 16), ±15.75 m;
- bilinear ×2.5;
- two 3×3 convs at 10 cm and a 1×1 to 8 logits.

**Defects and requested changes, most severe first.** Each goes TO THE MASTER MIND; this audit is read-only on that package.

**Status at 22:51.** D1, D2 and D6 were sent as the interim at ~22:48. They were **ACCEPTED by the Master Mind into `SPEC_REFCV7.md` §9 (Amendment A4, commit `b3f7ea6`)**, together with the refcv6 gradient-reach finding (D3 → `D-REFCV6-GRAD-REACH-DEAD` in GOALS_AND_CLAIMS). A4 also records that **`raw/PREREG_G_MAP_OVERFIT.md` is the protocol**. The builder's code is not yet changed; the items stay open until they land.

| id | severity | finding | evidence | requested change |
|---|---|---|---|---|
| **D1** | HIGH | **The decision rule is the raw argmax of median-frequency-weighted logits.** The rule is at `map_head_hires.py:506` (`per_class_signal`) and `map_hires_metrics.py:93-97` (the refcv7 eval hook). A weighted CE learns `q ∝ w·P`, so the raw argmax calls lane at a 5 % posterior and arrow at 0.66 %. | §2c: at σ = 0.1 m, lane 0.52 vs 0.91 prior-corrected, crosswalk 0.60 vs 0.95, drivable 0.91 vs 0.99 | Declare `decision_rule` in `config.json`. Implement `argmax(z − log w)` for eval and inference; log both rules (`interraw`/`unionraw`). BAR-M7 is scored on the declared rule, and the declaration is made BEFORE any refcv7 eval. |
| **D2** | HIGH | **`s8_detached` is an invalid must-fail arm in `map_hires_overfit.py`.** With ImageNet stride-8 features the branch can still memorise 16 frames. So by the harness's own rule ("a regression arm that passes ⇒ FAIL") a HEALTHY design fails. The harness also treats a class with no cells as "absent, not gated". | the harness `run_arm`/`verdict`, read at 22:50 | Add `s8_zeros` (`zeros_like(fmap_s8)`) as the gated arm, with a `must_fail_all` rule. Keep `s8_detached` informative. Add the presence floor (1,000 cells, else FAIL), C1–C3 in-run, both decision rules, and the 1 ms time guard. See `raw/PREREG_G_MAP_OVERFIT.md` §9a. `raw/gmo_spec.json` is refused by today's `load_spec` until these exist. |
| **D3** | MED | **Gradient-reach logging is dead.** The trigger is the refcv6 off-by-one: `code/fix/…/refc_v3_train.py:8604-8616` fills `_pr_row`, now including `ga_mh_*`, on `step % log_every == 0` before `step += 1`. | MEASURED: 0 `ga_` keys in refcv6's 4,621 rows | Trigger on `(step + 1) % log_every == 0`, the rule already used at `:8220-8224`. |
| **D4** | MED | **Big classes starve at convergence under MF weights.** At S2, drivable, sidewalk and no-class get 2.7 % of the logit gradient each under MF-present. The 10 cm head's drivable quality then rides on a thin-class-dominated signal. | §2b | Keep the unweighted 0.5 m auxiliary (task 7, option a), OR use a milder weighting (e.g. `sqrt` of MF). Decide with D1: D1 fixes the decision; this item is about the TRAINING signal. |
| **D5** | MED | **Far-range supervision of unresolvable markings.** The GT is clip-lifetime (non-causal). At 20–60 m the camera cannot resolve transverse markings (0.75 px at 20 m), and the stride-8 lift has **3 feature rows for 20–40 m and 2 for 40–60 m**. MF weights amplify the loss on those cells up to 7×. | `raw/lift_resolution.json` | Log per-band loss shares (they exist: `lc` per band). Pre-register a decision: (i) supervise thin classes only at x ≤ 30 m, or (ii) band-weight, or (iii) accept it. The BARs already sit on 0–20 m. |
| **D6** | LOW | The 0.5 m keys stay `map`, `map_iou_drivable` under `--map-lowres on`. They are stamped "NOT the map" in `config.json`, but a Watch or report panel reading `map_iou_drivable` would present 0.5 m as the map (A3). | builder open question 8 | Rename to `aux05_*` for refcv7 runs (no refcv6 reader reads a refcv7 file), or at minimum label them in the Watch. |
| **D7** | LOW | **The claim "10 cm hard labels remove the out-voting by construction" is wrong in general** (`…/refcv7-map-hires/RESULT.md` §4.1). | §2c (ii) | Correct the rationale. The 0.2 m-tolerance F1 metric is the right complement. |
| D8 | INFO | `per_class_signal` runs every training step (`with_metrics=True` default in `map_hires_loss_row`). At b16 that is ~6 transient `[16,8,600,320]` fp32 tensors (~0.6 GB, no_grad). | `map_head_hires.py:419-441` | Compute it on log and eval steps only (a speed item, not correctness). |
| D9 | INFO | **Stride-8 adequacy (the brief's question), ANALYTIC-EXACT:** lateral m per stride-8 feature is **0.135 m at 10 m, 0.30 m at 20 m**, 0.46 at 30 m, 0.63 at 40 m; the image itself has 2.05 mrad/px, so a 0.15 m line is 8.9 px at 10 m, 4.0 px at 20 m and 1.9 px at 40 m. Longitudinal m per stride-8 feature is 0.9 m at 10 m, **4.3 m at 20 m**, 18.4 m at 40 m. | `raw/lift_resolution.json` | **Verdict:** adequate laterally to ~20 m, with 2–3× super-resolution asked of the decoder at 20 m. NOT adequate longitudinally beyond ~15 m. The binding limit there is the 416-row image, not the stride: the image itself has only 20 rows for 20–40 m. G-MAP-OVERFIT tests the 0–20 m band. |
| D10 | INFO | The builder's G-EVAL finding (the eval loader builds the 0.5 m lift bank without `equalize_bottom_rows`) is confirmed by reading `refcv3_arm.rebuild_perception_branch` against `refc_v3_train.py:6922-6925`. | builder open question 7 | Fixes agent. |

## 5. Task 5 — G-MAP-OVERFIT, pre-registered

The protocol is `raw/PREREG_G_MAP_OVERFIT.md`; the machine-readable spec for the builder's harness is `raw/gmo_spec.json`. It was registered at ~22:50–22:55, BEFORE any G-MAP-OVERFIT run: none exists anywhere on D: or C:. It is summarised in §9.

## 6. Task 6 — logging requirement

`raw/LOGGING_SPEC_MAP10.md`. It adopts the builder's spelling (`map_head_hires.per_class_key`). It is summarised in §9.

## 7. Task 7 — "no way to use 50 cm": what to do with the 0.5 m head

**Activation memory at b16** (`raw/arch_memory_analytic.json`, ANALYTIC from real shapes, fp32, GiB):

| item | saved for backward | forward transient |
|---|---:|---:|
| refcv6 0.5 m lift (stride 16, Z = 4, 4,096 → 128) | 1.88 | 5.63 |
| NEW-2 lift as a plain `BEVLift` (stride 8, Z = 4) | 3.75 | 11.25 |
| NEW-2 lift **as built** (projection first: `grid_sample` of 64 projected ch × Z; ESTIMATED from shapes) | ≈ 0.1 (the small projected map) | ≈ 0.94 (sampled `[16·4, 64, 240, 128]` + mask) |
| NEW-2 branch total, the builder's CPU accounting (`…/refcv7-map-hires/raw/cost_analytic_cpu.json`) | **6.85** (0.37 with `grad_ckpt`) | — |
| NEW-2 stride-8 tap, the same accounting | 8.35 unchunked; **0.20** with `--trunk-chunk-ckpt` | — |
| refcv6 peak on Thor, MEASURED | 21.63 (`cuda_max_mem_gb` max) | — |

On compute, the lift projection is 129 GFLOP fwd (refcv6) and 258 GFLOP fwd (a plain NEW-2 lift). The trunk is ~10.6 TFLOP fwd per step, so the map path is a few % of step compute.

| option | memory vs refcv6 | risk to the planner | training signal |
|---|---|---|---|
| **(a) keep the 0.5 m head as a declared internal auxiliary**, never reported as the map | +~7 GiB (the builder's estimate 30–33 GB peak, < 60 % of Thor) | **lowest.** The planner, box3d and tactical tokens read the SAME stride-16 BEV features, shaped by the SAME 0.5 m loss as refcv6. | **Consistent, not conflicting.** The 0.5 m soft target is EXACTLY the 5 × 5 average of the 10 cm labels (census control, 0 bad cells). So for unweighted CE the 0.5 m Bayes-optimal output is the average of the 10 cm Bayes-optimal outputs: the objectives are nested, a duplication. They differ only in EMPHASIS once MF weights are on the 10 cm loss. The 0.5 m aux keeps drivable at **35 %** of its gradient; the weighted 10 cm loss gives drivable **2.7 %** at convergence (§2b). The two heads share only the trunk; the conflict detector can watch it (the builder added `map_hires` to the aux sum). |
| **(b) remove it; the map is 10 cm only** | saves only the 873-param head's loss. The 0.5 m LIFT and BEV encoder must stay, because box3d, the 30 × 16 BEV tokens and the planner's cross-attention read them. | **highest.** The planner-facing BEV encoder loses its only dense supervision and is shaped by box3d and tactical alone. That changes a refcv6-validated input for no memory gain. | only the weighted 10 cm signal reaches the trunk: thin-heavy at convergence, big classes 2.7 % |
| **(c) one high-res lift (stride 8 @ 0.25 m)** feeding the 10 cm decoder AND a pooled 120 × 64 BEV for planner, box3d and cross-attention | saves the stride-16 lift: −1.9 GiB saved and −5.6 GiB transient vs (a) | **medium-high.** The planner, box3d and tokens would read layer2 features (512 ch) instead of layer3 (1024 ch), which are less semantic, and the geometry changes. A second variable on top of NEW-1. | one consistent BEV. It still wants an unweighted big-class term on the pooled features, i.e. (c) + the aux loss. |

> **SUPERSEDED by the PI's decision (2026-09-27): option (c)** (SPEC_REFCV7 §11.1, A6, `546f34c`). The recommendation below is kept as written, for the record. Its argument about the big-class signal still applies to (c): the pooled planner BEV now receives only the class-weighted 10 cm gradient. The logging spec's per-class loss shares (§6) are the instrument that will show whether drivable starves.

**Recommendation: (a)** for refcv7 (the PI decides):
- it is `--map-lowres on`, keys renamed `aux05_*` (D6), reported only as "0.5 m auxiliary";
- (c) is a separate, pre-registered v7-tiny-ladder experiment after refcv7's first bars. It is the cleaner end-state, but it is a planner-input change and must not ride along with NEW-1.

**Condition on (a):** the two map terms' trunk-gradient cosine is logged (the conflict detector's per-group rows). If it goes persistently negative, that is the signal to revisit.

## 8. Ranked causes — why refcv6 was not trained to extract all classes

**Ranking rule.** Causes are ranked by how much of the thin-class IoU deficit each can account for, from measured or exact numbers. No counterfactual retraining was run: the sizes are measured states and analytic ceilings, NOT A/B deltas.

| rank | cause | size | evidence class | explains |
|---:|---|---|---|---|
| 1 | **Localisation of the stride-16, 0.5 m lift, decoded by argmax on a soft target** | 0.35–1.25 m of road per feature laterally over 0–40 m; 20–40 m in **2** feature rows, 40–60 m in **1**. Perfect-information ceiling at σ = 0.5 / 1.0 m: lane **0.26 / 0.05**, crosswalk 0.57 / 0.39, edge **0.00 / 0.00** (0–20 m). Measured head: mean q on lane cells 0.11 vs label fraction ~0.4; argmax lane on 5.6 % of lane cells. | EXACT geometry; ANALYTIC ceiling; MEASURED head state | lane (fully), crosswalk (mostly, with transverse blur ≫ lateral), edge (with rank 2) |
| 2 | **The 0.5 m target itself** | edge keeps 0.9 % of its area; lane 61.8 %, arrow 64.7 % under the eval's ≥ 0.5 rule | MEASURED (census) | edge: unlearnable at 0.5 m under any model |
| 3 | **Signal share in the shared path** | thin classes = 10.6 % of the map's trunk gradient (arrow 0.14 %, hatched 1.7 %); the map = 3.4 % of total loss; map vs box3d gradient on lift / BEV encoder / trunk = 1 : 12 / 1 : 11 / 1 : 145 at init | MEASURED (38k probe; `metrics.jsonl`; 2026-09-17 init reach) | arrow and hatched below even the σ = 1 m ceiling (0.09 / 0.38 vs 0.000 observed). INFERRED, not isolated. |
| 4 | **Non-causal, sub-pixel far-range labels** | beyond 20 m: a 0.4 m stop line ≤ 0.75 px, a 0.15 m line ≤ 2.6 px at 30 m; the GT still labels them (clip-lifetime) | EXACT geometry | label noise concentrated in 20–60 m |
| 5 | **No per-class instrument** (why it went unseen for 38,000 steps) | 0 per-class keys and 0 `ga_*` keys in 4,621 rows | MEASURED | none of the collapse, all of the delay |

**Not causes (checked):**
- **The lift-valid mask:** it removes 11.03 % of cells, and not thin-rich ones (lane mass 1.635 % → 1.694 %).
- **The half-applied C26:** 0.86 % of 0–20 m cells lose their road-plane sample, and `map_valid` is unchanged.
- **The optimiser grouping:** the branch is in the head group, and AdamW normalises per parameter.
- **Coverage:** 1.0 train.
- **Label alignment:** the 1 ms `t_img` guard; the frame axis is pinned.

## 9. Pre-registered checks (summary)

- **G-MAP-OVERFIT** (`raw/PREREG_G_MAP_OVERFIT.md`, `raw/gmo_spec.json`):
  - **frames:** 16 TRAIN frames, 4 clips (sha12 `10497f0d664b`, `05c575ed45be`, `104d79ae052d`, `0dbd6c9cd776`), frameset md5 `4eafa03c`, selected label-only on Thor;
  - **training:** 1,000 steps, batch 4, AdamW lr 1e-3, wd 0, seed 0; the branch + trunk s8 stage trainable;
  - **PASS** (0–20 m, pooled, declared rule): big classes ≥ **0.85**, the five thin classes ≥ **0.50**, ≥ **1,000** cells per class, and per-class CE ≤ 0.5 × its step-0 value;
  - **MUST FAIL:** `lane_w0` (on lane) and `s8_zeros` (on ALL five thin classes);
  - **controls:** constant, GT-as-logits and rule-identity must read their known values.
- **G-DVB/map-logging** (`raw/LOGGING_SPEC_MAP10.md`):
  - the first eval row of the real-config smoke carries all 24 × 2 per-class-band IoU/loss-share keys plus the 48 inter/union counts;
  - Σ lc = loss;
  - the Watch shows all 24 series and the thin-class alarm;
  - the `ga_mh_*` keys appear (> 0);
  - **must fail:** drivable-only logging, per-batch IoU instead of counts, a Watch without the panel, and the reach off-by-one.

## 9A. Task 8 (SPEC_REFCV7 §11.2, A6) — the maximal map extent: the coverage census

**PI (verbatim):** *"do c and assure that the range of the map is maximal and not only 20 m"*. The rule is fixed in the SPEC; this census sets the number.

**Order of work.**
1. **PREREG first:** `raw/PREREG_MAP_EXTENT_CENSUS.md`, md5 `d4d02505…`, mtime 00:19:32. It was written before any world map was read; only the exporter source and file listings had been read.
2. **The run:** Thor, 00:21–00:22, fresh dir `/home/nvidia/msa_0021/`, script md5 `3da14c3f…` verified after scp. CPU, `nice 19`, 6 workers, the production interpreter (`tanitad-edge`, numpy 2.5.1), read-only on the corpus.

**Validity gate (it ran first; the census would have aborted otherwise):**

| control | frames | identical | differing bytes |
|---|---:|---:|---:|
| **C1** — the exporter's own expression re-crops the `[600, 320]` window from the world map on the stored `T_world_rig`, vs the stored `/2` `fine_codes` | 24 (2 per clip × 12 TRAIN clips) | **24 / 24** | **0** |
| **C2** — the census sampler inside the old window vs `fine_codes[2::5, 2::5]` | 24 | **24 / 24** | 0 |

**Coverage.** 4,369 / 4,369 TRAIN clips (0 missing), **78,321 frames** (17–18 per clip, one per second over the training label frames).
- "Seen in a ring" means ≥ 20 % of the ring's 10 cm samples are not-255.
- Coverage is frame-weighted; the clip-weighted values agree to ≤ 0.001.
- `raw/thor_0021_census/census_table.json` also carries the full block table x 0–200 m × |y| 0–60 m.

| x-ring (ahead, \|y\| < 16 m) | 0–10 | 10–20 | 20–30 | 30–40 | 40–50 | 50–60 | 60–70 | 70–80 | 80–90 | **90–100** | 100–110 | 110–120 | 140–150 | 190–200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| coverage | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 | 0.846 | 0.754 | 0.672 | 0.598 | **0.532** | 0.473 | 0.422 | 0.300 | 0.174 |

| y-ring (to the side, x < 60 m) | 0–10 | 10–20 | **20–30** | 30–40 | 40–50 | 50–60 |
|---|---|---|---|---|---|---|
| coverage | 1.000 | 1.000 | **0.9996** | 0.205 | 0.128 | 0.097 |

**THE RULE SELECTS x_max = 100 m and y_half = ±30 m**, i.e. a 100 m × 60 m map (3.125× today's 60 × 32 m area). Neither lies on the census edge.
- **Robust:** the same 100 / 30 at the 5 % and the 50 % seen-thresholds, with x-rings taken within |y| < 10 m, with y-rings taken within x < 30 m, and clip-weighted. The rule could have moved; it did not.

**Why these numbers (READ from the renderer source; the source is PUBLISHED-CODE, the attribution is INFERRED):**
- `sam3map_render_v5m.py:33` sets `R_MAX = 35.0` m. The renderer labels a world cell only within 35 m of SOME camera position in the clip, and the world map's extent is the trajectory padded by R_MAX + 12 m (`:234`).
- **Lateral:** the ±30 m cliff (1.00 → 0.21) is that 35 m camera range around the driven path. The camera's field of view does not set it. More lateral range needs a re-RENDER, which the SPEC excludes ("SAM3 is NOT re-run").
- **Ahead:** coverage past 35 m falls with the distance the ego still drives in the clip. At 100 m, 53 % of TRAIN frames still have the rest of the drive in front of them.

**Physics caveat.** It changes no number, but it matters for what the far bands can show. The far GT is non-causal: a cell 60–100 m ahead is labelled from a LATER camera position.
- From the current frame, 60–100 m of road spans **~4.4 image rows**. This is ANALYTIC: `row = 203.1 + 630/d`, fitted to the measured rows at 40, 50 and 59 m, with d the distance from the camera.
- That is **< 1 stride-8 feature row**. At 100 m a 0.15 m line is 0.75 px wide.
- Expect the thin-class bars in the 60–100 m bands to read ~0 for both refcv6 and refcv7. SPEC §11.2.1 already requires such bands to be reported as such, not dropped.

**The `/3` re-export**, timed on 20 TRAIN clips at 100 × ±30 m (`code/reexport_v3_timing.py`, `raw/thor_0021_census/reexport_v3_timing.json`; built in memory, nothing written to disk):
- **7.26 s per clip** (median 7.23, max 8.31) in one process.
- ⇒ **all 4,508 clips (train + eval) in ~9.1 h on one process, ~1.5 h on 6**.
- 4.35 MB per clip, against 2.10 MB for `/2`; **~19.6 GB** in all.
- **Inside the old 60 × 32 m window, `/3` equals `/2` byte-for-byte:** `fine_codes` AND `cart_frac`, on **all 4,020 frames**.

⚠️ **That identity needs ANCHORED coordinates.** `/2` writes a lateral cell centre as `−y_half + (j + 0.5)·cell`. At y_half = 30 the same cell gets a different float, and a `floor()` on a world-map cell boundary can flip a code. The prototype writes `y = −16 + (j_rel + 0.5)·cell` with `j_rel = j − 140` (fine) or `j − 28` (0.5 m). **Requested of the `/3` exporter's author (via the Master Mind): use this form.**

**The 10 cm decoder at b16, ANALYTIC from the builder's shapes** (`code/decoder_cost_extent.py`, `raw/decoder_cost_extent.json`; fp32; the builder's measured per-sample accounting scaled by area, with this file's own layer count as the check):

| | 60 m × ±16 m | **100 m × ±30 m** |
|---|---:|---:|
| lift grid @ 0.25 m / logit grid @ 0.1 m | 240 × 128 / 600 × 320 | 400 × 240 / 1000 × 600 |
| saved for backward, b16 (builder's measure, scaled) | 6.85 GB | **21.4 GB** |
| the same, own layer count | 7.48 GB | 23.15 GB |
| with `grad_ckpt` (builder, scaled) | 0.36 GB | 1.14 GB |
| logits / their gradient, b16 | 0.09 / 0.09 GB | 0.29 / 0.29 GB |
| largest transient activation gradient, b16 | 0.37 GB | 1.14 GB |
| fwd FLOP per sample | 34 GFLOP | 102 GFLOP |
| fwd + bwd per step, b16 | 1.6 TFLOP (5 % of the trunk's ~32) | **4.9 TFLOP (15 %)** |

- **Memory.** Against refcv6's measured 21.6 GB Thor peak, the new extent adds ~21 GB without checkpointing (≈ 43+ GB) or ~1 GB with `grad_ckpt`, which recomputes the branch (~+30 % of its FLOPs).
- **Time.** The FLOP share alone predicts +5 % (today's extent) to +15 % (100 × 60) per step. The GroupNorm/GELU passes over 21 GB of activations are memory-bound on top of that. Whether 100 × ±30 m stays under the PI's +25 % s/step line is **UNMEASURED**: the Thor G-LIVE smoke decides, and above +25 % the PI chooses between range and cost (§11.2.2).

**Design note for option (c).** The pooled 0.5 m BEV that now feeds box3d, the planner's cross-attention and the 30 × 16 tokens would grow from 120 × 64 to 200 × 120. Each token would then cover ~3.3 m × 3.75 m instead of 2 m × 2 m. Either crop the planner's pooled BEV to 60 × 32 m or DECLARE the geometry change (G-DVB). It must not change silently.

## 10. Disclosures

1. **22:26:** `code/analytic_class_signal.py` (CPU, ~300 MB, 130 s) was started at **4.32 GB free**, below my own 7.5 GB start gate. It finished cleanly and has not recurred: the heavy probe is gated by code (`wait_then_probe.py` + the probe's own 3-sample gate and a 4.0 GB abort watchdog).
2. **22:33:** one `git status` ran against the shared GIT_DIR `C:/Users/Admin/tanitad-push/.git` with the D: work tree. It may have rewritten that index's stat cache (index mtime 22:33:54, 497 bytes, 5 fixture entries: `kept.txt`, `pkg/a.txt`, `pkg/sub/b.*`, `target.txt`, `top.txt`). No entry was added or removed by it, and no lock was left. After that I used only `show`, `ls-tree`, `log`, `rev-parse`, `archive` and tree-to-tree `diff`.
3. **22:35:** my first waiter (v1, PIDs 46668/47336) read "no boxes render process" as completion while the full render was still waiting for RAM. I stopped it by explicit PID before it launched anything. v2 waits for the BOXES_DONE marker only.
4. **Thor, label scan:** one read-only, `nice 19`, 20-second label scan (`code/gmo_select_train_frames.py` over stdin), run twice. Nothing was written on Thor by it.
5. **Dev-box waiters.** Every one was stopped by explicit PID and none ever launched the probe on the dev box:
   - v2 (PIDs 46344/36600) at 23:14, to add the GPU-gate device choice;
   - v3 (PIDs 34860/51160) at 23:55, on the Master Mind's GO-THOR.
   The log is `raw/wait_then_probe.log`.
6. **Thor, the 38k probe** (authorised: "GO-THOR", 23:55; Thor idle):
   - it ran in the fresh dir `/home/nvidia/msa_2358/` (tree, tarball, loader, probe, `out/`), PID 3461519, 27.7 s on the GPU;
   - no pip installs; no other process touched;
   - its outputs were pulled back md5-verified into `raw/thor_2358/`. The dir is left in place and holds no clip id.
7. **Box-presence task.** It was added and then moved to the box-head audit mid-run. My partial is banked as `raw/box_presence_partial.{md,json}`, and the Master Mind has the paths.
8. **Thor, the extent census and `/3` timing** (the Master Mind's brief, GO-THOR rules):
   - run in the fresh dir `/home/nvidia/msa_0021/`, CPU, `nice 19`, `tanitad-edge`, read-only on `/home/nvidia/sam3map` and `/home/nvidia/data`;
   - the `/3` prototype built its arrays in memory and wrote nothing but its JSON; no other process was touched;
   - outputs are pulled back md5-verified into `raw/thor_0021_census/`.
9. **`code/decoder_cost_extent.py`** is pure arithmetic (< 1 s, no data). It ran on the dev box at 6.39 GB free, below my 7.5 GB rule, which is meant for compute jobs. Disclosed anyway.

## 11. Deliverable manifest

**Status of every file:**
- Nothing is committed or staged by this agent. The Master Mind lands the package from `LANDING_READY.txt`.
- Every repo path below lives ONLY on the D: working tree (`D:/Projects/TanitAD/`) until it is landed.
- Package prefix: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-map-signal-audit/`.

| artifact | where | only one place? |
|---|---|---|
| `RESULT.md` | repo (D: working tree) | yes, until landed |
| `LANDING_READY.txt` | repo (D:) | yes, until landed |
| `raw/PREREG_G_MAP_OVERFIT.md` | repo (D:) | yes, until landed |
| `raw/gmo_spec.json` (md5 `d3a41b92…`) | repo (D:) | yes, until landed |
| `raw/gmo_frameset.json` (md5 `4eafa03c…`) | repo (D:) | yes; it was also printed to stdout on Thor, never written there |
| `raw/gmo_frameset_v1_SUPERSEDED_duplicate_frames.json`, `raw/gmo_frameset.stderr.log` | repo (D:) | yes |
| `raw/LOGGING_SPEC_MAP10.md` | repo (D:) | yes, until landed |
| `raw/lift_resolution.json` | repo (D:) | yes |
| `raw/analytic_class_signal.json`, `raw/analytic_class_signal.log` | repo (D:) | yes |
| `raw/arch_memory_analytic.json` | repo (D:) | yes |
| `raw/thor_2358/measure_class_signal_38k.json` (`6ab016f7…`), `…_selection.json` (`69a70d62…`), `….log` (`dfd99729…`), `probe_stdout.log` (`1d78b80c…`) | repo (D:) **and** `thor:/home/nvidia/msa_2358/out/` | no (two places, md5-equal) |
| `raw/thor_2358/measure_class_signal_38k_summary.json` (`2506dcd3…`) | repo (D:) | yes |
| `raw/box_presence_partial.md`, `raw/box_presence_partial.json` | repo (D:) | yes (for the box-head audit) |
| `raw/wait_then_probe.log`, `raw/wait_then_probe.stdout.log`, `raw/measure_class_signal_38k.log` (empty: the dev box never ran it) | repo (D:) | yes |
| `code/lift_resolution.py`, `code/analytic_class_signal.py`, `code/arch_memory_analytic.py`, `code/gmo_select_train_frames.py`, `code/measure_class_signal_38k.py` (Thor copy md5 `edb608bb…`), `code/wait_then_probe.py`, `code/summarize_probe.py` | repo (D:) | yes, until landed; the probe also sits at `thor:/home/nvidia/msa_2358/` |
| clean tree of `2c510fb` (stack, taniteval, tools) | `C:/Users/Admin/msa_tree_2210` (dev box); `thor:/home/nvidia/msa_2358/tree` + `tree_2c510fb.tar` (`ec162e92…`) | scratch, reproducible from git `2c510fb`; not landable |
| Thor ship bundle | `C:/Users/Admin/msa_thor_bundle_0000/` | scratch, reproducible |
| `raw/PREREG_MAP_EXTENT_CENSUS.md` (md5 `d4d02505…`) | repo (D:) | yes, until landed |
| `code/map_extent_census.py` (`3da14c3f…`), `code/reexport_v3_timing.py` (`b2a7c4b6…`) | repo (D:) and `thor:/home/nvidia/msa_0021/` | no (md5-equal) |
| `code/decoder_cost_extent.py`, `raw/decoder_cost_extent.json` | repo (D:) | yes |
| `raw/thor_0021_census/census_table.json` (`b61dfa50…`), `census_controls.json` (`acb96b92…`), `census_stdout.log`, `reexport_v3_timing.json` (`d81285ba…`), `reexport_stdout.log` | repo (D:) and `thor:/home/nvidia/msa_0021/out/` | no (md5-equal) |

