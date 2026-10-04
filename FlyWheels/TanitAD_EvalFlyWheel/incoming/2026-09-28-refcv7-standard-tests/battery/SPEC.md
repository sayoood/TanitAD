# SPEC — refcv7 four-family milestone battery (PRE-REGISTRATION)

**Stream:** EvalFlyWheel, refcv7 standard tests (the NavSim suite is a sibling stream; BAR-R7-N1 is NOT scored here).
**Written:** 2026-09-28, finished 05:29 Europe/Berlin (dev-box clock), **BEFORE any refcv7 forward pass by this stream.** The first refcv7 forward is G0 (§2) on the validation checkpoint. The sha256 of this file at registration is in `raw/SPEC_SHA256_AT_REGISTRATION.txt`; every runner output stamps the sha256 of the SPEC it ran under. Amendments are appended below with their time and what had been read when each was written.
**refcv7 numbers the author had seen before writing:** the KEY NAMES of the run's in-run eval rows at steps 500 / 1,000 / 1,500 (1,110 keys; 98 of them null in the step-1,500 row) and the `step` field of the metrics rows (last logged step 1,861 at 05:17). **No value** of any eval or training term had been read. The pull record `raw/pull_ckpt_1500.json` reads only counts.
**Authority:** `Project Steering/SPEC_REFCV7.md` (§3 BAR-R7, §6.2/§11.2/§13 BAR-M7 + A8 per-band rule and 40 keys, §14 BAR-B7, §15.3 box ratio). ⚠️ `Project Steering/PREREG_REFCV7.md` is the 2026-09-19 **DrivoR-T** pre-registration (arms A0/B/C/D, `--refcv7` flags). Per SPEC_REFCV7 A1 those flags are OFF for this refcv7 (the launch argv carries none of them), so its gates G1-G5 do not apply to this run; only its estimator/tier/four-family discipline is inherited. Reported as a document conflict.

---

## 0. Object under test, code, data (all MEASURED 2026-09-28 unless stated)

| | |
|---|---|
| run | `refcv7-r101-s0`, Thor, training since 2026-09-28 00:03 Berlin (MODEL_REGISTRY.md at blob `45e76fba`, row "refcv7-r101-s0") |
| launch | commit `fec3a0dccfda7eeaf34b537337370f1659fd2429`; `config.json` md5 `e6512a01b9c70f0e4a4dac581621984a` (Thor == dev-box copy); argv 154 tokens |
| code | `C:/Users/Admin/ev7` = `git archive 0c444082a08e… stack taniteval tools`. `git diff --raw --abbrev=40 fec3a0dc 0c444082 -- stack taniteval tools` lists exactly 2 paths, both Training-Watch files (`build_watch_refcv7.py` + its test): **every eval-relevant blob is the launch blob.** Thor's trainer file md5 `0a6fb0d8…` == `git show fec3a0dc:stack/scripts/refc_v3_train.py` (blob `203b437f`); Thor's loader md5 `7a01aba3…` == blob `a8eb3e7e` |
| loader | `stack/tanitad/eval/refcv7_loader.py` (blob `a8eb3e7e`, passed the launch gate's G-EVAL). `--trunk-compile` is dropped (no Triton); Thor data paths are remapped per flag, basename-checked |
| kit | `D:/refcv7_eval_kit/` holds `ckpt/`, `thor_reads/`, `kit_checks/`, `KIT.log`. **The data root is `D:/refcv6_eval_kit/data`** (the loader's default `KIT`): all **290/290** eval-side inputs the run's argv names (eval139 cache 141 files, eval labels, max-speed eval sidecar + meta, extrinsics, clip-clock sidecar, anchors, 2-D and 3-D joins, the 4 `refcv7/` files: VIS-1 sidecar, nav τ file, map class weights + provenance, and the 137 eval-clip `/3` map-GT files) are **md5-identical** to Thor's `/home/nvidia/data` (`kit_checks/eval_inputs_md5_compare.json`). Nothing was pulled for data; nothing is copied twice (D: has 28.8 GB free) |
| validation ckpt | `D:/refcv7_eval_kit/ckpt/ckpt_1500.pt`, the ROLLING `ckpt.pt` at step 1,500, md5 `c35966f799b1c724515ba1281736fd48` (Thor before == Thor after == local; step key 1,500 read locally). **PIPELINE VALIDATION ONLY** (§6) |
| milestone ckpt | step 5,000 → `ckpt_5000.pt`. From source: `MILESTONES = (5000, 15000, 20000, 30000)` (`refc_v3_train.py:225`) and `torch.save({"model", "step"}, out_dir / f"ckpt_{step}.pt")` at `:9645-9647`, written after the rolling `ckpt.pt` and BEFORE that step's eval row, **never rotated**. Fallback: the rolling `ckpt.pt` while the run's last step is in [5,000, 5,500), with the step key asserted locally |
| baselines | refcv6@38k = `D:/refcv6_eval_kit/ckpt/ckpt_step38000_stopped.pt`, md5 `5a2e7222a9f5f8c7aa7bf38ef4698d8a` (MEASURED, == the registry's), config `ckpt_final/config.json` md5 `a3193a46…` (the post-switch resume config), rolled on the refcv6 package's harness on the `82c2331` tree (`C:/Users/Admin/ev6_82c2331`). Banked model-free controls: the refcv6 battery's S2 grid dumps (`C:/Users/Admin/refcv5cmp/out/*`) |

---

## 1. Tiers, estimator, variance

* **T1 = self-action OPEN LOOP:** one forward at the window origin; the path is the model's own selection (`out["traj"]`). **Not closed loop, not driving performance** (EVAL_DOCTRINE, amended 2026-09-02). refcv7 consumes no actions; the tier ruling for an action-free model (BACKLOG R30) is **UNRULED**, so `os` is stamped `T1 / status UNRULED`, as refcv3/4b/5-v2/6 were. **T0** (`oracle_sel`, the GT-nearest decoded anchor) is a **ceiling diagnostic only** and is never compared to a T1 number.
* **Point estimate:** the **FULL-SET** pooled mean over windows (map/box: pooled ratio of sums, §3.4).
* **Interval:** `taniteval.ci.episode_cluster_bootstrap`; for two arms on the same windows `taniteval.ci.paired_episode_cluster_bootstrap`. `n_boot 2000`, `seed 0`, cluster = episode (clip), 95 % percentile. ⛔ `overlapping_holdout_se` appears nowhere.
* **Which variance each interval answers** (CLAUDE.md):
  * another draw of EPISODES: the bootstrap;
  * another INFERENCE run: refcv7 samples (DDIM draws `eps` at eval). **Two inference seeds (0 and 1)**; every planner bar must hold at BOTH; the inference-seed floor `|ADE(os_s0) − ADE(os_s1)|` and the paired `os_s0 − os_s1` cell are reported;
  * another TRAINING run: **NOT MEASURED** — one training seed (`H-ESTIM-SEED-1`; replicate false-positive rate for "separated" on the tiny rig 6/42 = 14.3 %). Per SPEC_REFCV7 §3, a bar whose |delta| is within **2× the replicate floor** is **NOT PROVEN**. The only replicate floor this battery MEASURES is the inference-seed floor; that is the floor applied, and every PASS is written "PASS (single training seed; training-seed floor unmeasured)".
* **Four families, per family, never pooled, never ADE alone** (§3.3). STRATEGIC is **NOT APPLICABLE, n = 0**: the strategic layer is OFF (`--no-strategic`, SPEC_REFCV7 §1); its route loss is gated off in training, so the route head is untrained and is not scored.

---

## 2. GATE G0 — the loader reproduces refcv7's own in-run eval row (nothing downstream counts until it passes)

**What is reproduced.** `metrics.jsonl`'s eval row at the kit checkpoint's own `step`: the mean over **8 fixed batches of 16 = 128 windows**, `inrun_eval_perm` = `torch.randperm(len(e_ds), generator=manual_seed(12345))[:128]` over the loader's eval dataset, iterated in order (`refc_v3_train.py:8905-8911`). Each batch value is the trainer's own `compute_losses_v3` under `model.eval()` + `no_grad`; the row is `_eval_row_from_acc` (mean over batches, 5 dp except the exact `map_hires_*` keys, plus `derived_per_class`), and the detection keys are `detection_metrics.summarise` over the pooled per-window packs (`:9703-9714`). G0 calls those same trainer functions.

**How** (`code/g0_refcv7.py`):
* model and eval dataset from `refcv7_loader.build_model` / `build_eval_dataset` (strict load); `--trunk-compile` dropped; bf16 + NHWC, fold, dedup and chunk-8 kept as run.
* ⭐ **requires_grad as TRAINED.** The loader leaves every parameter `requires_grad=False`; the trainer's in-run eval ran with its training flags (True everywhere except `declare_grad_unreachable` subtrees). The launch gate MEASURED on Thor that this flag state changes 74/188 outputs (the slot heads' attention path). G0 therefore restores the training flags (True, except modules carrying `_gradreach.GRAD_UNREACHABLE_FLAG`), and records a control: one batch under both flag states, with the number of differing terms (the battery rolls in the loader's state; if the control shows a difference on the dev box it is reported beside every battery number).
* ⚠️ Batch 16 does not fit the 8 GB card: the model forward is **micro-batched inside `compute_losses_v3`'s single `model(...)` call** (`code/microbatch.py`, the refcv6 G0 wrapper extended with `ego_actions` / `_ego_actions`); outputs are concatenated before the loss, so every loss reduction runs over the same 16 rows. Sizes: `2,2,3,3,3,3` (unequal, for batch-dim detection); if they do not fit, `1,1,2,2,2,2,3,3` (recorded).
* only the LAW frame of `future_frames` reaches the device (`compute_losses_v3` reads index `LAW_AHEAD-1` only).
* the DDIM draw is seeded per inference seed (the run's draw came from its training RNG and cannot be replayed). **K = 8 seeds (0..7).**

**Term classes — assigned by NAME before any value is read:**

| class | members | tolerance vs the in-run value |
|---|---|---|
| **EXCLUDED** | `eval_goal_gate_grad` (the last TRAINING backward's grad; not an eval quantity); `eval_{agent,box3d}_calib_*` (10 keys: A10 §15.3's INFORMATIVE P = R gate on the FIXED **TRAIN** calibration windows, a separate TRAIN loader; the kit holds no TRAIN clips); `eval_step` | none |
| **UNDEFINED rule** (every class) | a key that is `null` in the in-run row must be null/NaN in every seed's reproduction, and vice versa | else OUT |
| **COUNT** (data) | every key with a standalone `n` or `npos` token (`agent_n_*`, `box3d_n_*`, `map_hires_n_*`, `n_map_hires_cells*`, `tacv6_n_*`, `*_npos_*`, `*_det_npos_*`) **except** the MODEL-DEPENDENT counts below; plus `eval_batches`, `eval_windows`, `slot_valid_frac`, `tac_label_rows`, `tac_label_v7`, `nav_injected`, `ego_injected`, `box3d_visible_filter`, `agent_rows_*` | exact at 5 dp (abs ≤ 1e-5; exact keys abs ≤ 1e-6·max(1, \|x\|)) |
| **DETECTION** (pooled over 128 windows; a threshold or greedy match can flip on cross-hardware numerics) | every other key `detection_metrics.summarise` emits: `*_prec@gate`, `*_rec@gate*`, `*_f1@gate`, `*_ap2m`, `*_auroc_*`, `*_cls_acc_tp*`, `*_det_ap*`, `*_det_map*`, `*_conf_ratio`, `*_centre_err_p50`; and the MODEL-DEPENDENT counts `*_n_conf`, `*_tp@gate`, `*_n_ignore_masked_slots`, `*_n_presence_exempt`. `*_conf_ratio_alarm` is judged through `conf_ratio` (it is a function of it) | [0,1]-valued: abs ≤ **0.02**; `conf_ratio`, `centre_err_p50`, model-dependent counts: rel ≤ **5 %** (counts: or abs ≤ 2); class median of abs deviation over the [0,1]-valued keys ≤ **0.005** |
| **MATCHED** (Hungarian set losses) | `agent_{centre,cls,presence,size,yaw}`, `agent_presence_layer*`, `agent_layer*`, `box3d`, `box3d_{centre,cls,h,occ,presence,presence_focal,rates,size,yaw,z,vis1}`, `box3d_presence_layer*`, `box3d_layer*` | rel ≤ **15 %** each and class median ≤ **5 %** (refcv6 G0 basis, MEASURED there) |
| **MAP10** (argmax over ~6×10⁵ cells per window) | `map_hires_{inter,union,interraw,unionraw}_*` | abs ≤ max(0.02·\|x\|, 25 cells); `map_hires_{iou,iouraw}_*`: abs ≤ **0.01**; class median rel dev over keys with \|x\| ≥ 1,000 cells ≤ **0.2 %** |
| **STOCHASTIC** | every non-excluded term above not in COUNT/DETECTION/MATCHED/MAP10 whose value across the K seeds varies by more than rel 1e-5 | the in-run value inside the **99 % prediction interval** of the K = 8 values, `mean ± t(0.995, 7)·sd·√(1+1/8)` (t = 3.499), widened by 1 % of \|mean\| for the cross-hardware bf16 shift |
| **SMOOTH** | all remaining (deterministic in the seed), e.g. `lat`, `lon`, `law`, `goal_tac`, `cascade`, `map_hires`, `map_hires_{lc,gn,gno,lshare}_*` | rel ≤ **1 %** (abs ≤ 1e-3 when \|in-run\| < 0.1); class median ≤ **0.2 %** |

**Deliberate regression (the gate must be able to FAIL).** Arm **M1** re-runs inference seed 0 with the **trunk's bottom-row equalisation removed** (`trunk_equalize_bottom_rows` 43 → 0 on the built trunk): exactly D-REFCV6-EQUALIZE-DROPPED, the defect SPEC_REFCV7 FIX-3 exists for, i.e. a loader that rebuilds the trunk as the pre-FIX argv declares. **M1 must move ≥ 1 non-COUNT term outside its class tolerance. If M1 passes, G0 is VOID, not PASS.**
**Power probes (reported, not gating; each undetected one is named as a blind spot):** **M2** the map's declared decision rule dropped (`prior_corrected` → `raw` on the built branch); **M4** the NEW-1 residual prior switched off at eval (`decoder.residual_prior` → off), a loader that loses the prior. Each at seed 0 on the same 128 windows.

**Wrapper control (refcv6 AMENDMENT A2 form, adopted from the start).** Batch = the first 4 windows of in-run batch 1; DDIM `eps` zeroed in every arm. Arms `plain`, `repeat`, `permuted` (rows reversed), `micro[1,3]`, `micro[3,1]`. `Floor = max rel(|plain−repeat|, |plain−permuted|)`; `Wrapper = max rel(|plain−micro13|, |plain−micro31|)`. Conditions: P1 `as_run`, P2 `cudnn_det` (cuDNN TF32 off, deterministic), P3 `fp32_det` (P2 + trunk fp32, NHWC off; chunk 4 if it does not fit, recorded). **PASS iff under P3 every term has `Wrapper ≤ max(3·Floor, 1e-5)`, AND the wrapper regressions W1 (0-dim outputs merged as micro-part 0) and W2 (parts concatenated in reverse) each exceed that bar on ≥ 1 term; otherwise VOID.** If P3 cannot run at all on the card, the clause is evaluated under P2 and that is recorded. P1/P2 are reported.

**Verdict.** **G0 PASS** iff: strict load clean (0 missing / 0 unexpected); `param_breakdown` equals the stamp; the kit anchor file equals the checkpoint's anchor buffers (max |Δ| 0.0); the loader's G-DVB and stamp checks pass; every term meets its class tolerance and every class median holds; the wrapper clause PASSES; **M1 is detected**. Otherwise **STOP**: every differing term is reported with both values, and no battery number is computed for that checkpoint.

---

## 3. The battery

### 3.1 Surfaces
* **S2, PRIMARY:** the refcv6 battery's S2 grid, unchanged — the banked refcv4b dev-box 2 s grid (stride 5, instants 0.5/1.0/1.5/2.0 s) restricted to the 139 kit clips: **4,754 windows / 139 episodes** (refcv6 SPEC §3.1: GT recomputed through the trainer's `waypoint_targets` equals the banked `g` on 4,754/4,754). refcv7 reads the SAME eval cache files (md5, §0), so it is scored on identical clips, windows and GT futures.
* **S6, SECONDARY (non-regression, 6 s):** the S2 windows whose 60-step future is inside the clip; instants 1..6 s.
* **M, MAP/BOX:** **every window of the loader's eval dataset index** over the 139 eval clips (SPEC_REFCV7 §6.2 "every eval window"), map scored on the 137 clips that have `/3` GT, box on every window with a VIS-1 row. n (windows, clips) printed with every cell.

### 3.2 Arms (T1 unless stated)

| arm | what |
|---|---|
| `os` | refcv7 one forward, own selection, **speed-ceiling filter ON as configured** (the PRIMARY reading, SPEC_REFCV7 A2). Fed exactly as `compute_losses_v3` feeds it: true v7 nav token, measured v0 at t0, ego-history window `poses[t:t+W]` (which carries the NEW-1 `ha0_ext_pose` prior), eval max-speed sidecar value, the per-clip 0.25 m lift geometry |
| `os_filteroff` | the same forward with the ceiling filter OFF (SPEC_REFCV7 A2's sensitivity reading, without the oracle ceiling) |
| `os_navshuf` / `os_navzero` / `os_navflip` | nav paired to the wrong window / nav withheld / left↔right |
| `os_vmaxzero` | max-speed input withheld (`v_max_valid = 0`) |
| `ha` | hold last observed (a, steer) |
| `ha0` | **constant velocity** at measured v0 |
| `ha0_ext` | the **echo** control (`refav1_arm.hold_ext_controls`; constant accel + constant curvature, reads the recorded steer at t0 — SPEC_REFCV7 A5) |
| `stop` | zero displacement |
| `oracle_sel` | GT-nearest anchor of this forward's fan — **T0 ceiling only** |
| `b_refcv6_38k` (seeds 0, 1) | refcv6@38k `os` on the SAME S2 windows, rolled by the refcv6 package's own validated harness on the 82c2331 tree (§0) |
| `b_refcv4b`, `b_refcv5v2_s0/s1` | banked dev-box dumps (context rows; model facts from MODEL_REGISTRY §4.6 / §4.8 only) |

### 3.3 Metrics per family (the same instrument for every arm)
`refcv3_arm.analyze_refcv3` (imported, not copied) per dump, plus the refcv6 package's A3 masked yaw-rate cell, and:
* **ADE / FDE** over the grid.
* **LONGITUDINAL:** speed MAE, target-speed accuracy @0.5 m/s, along-track MAE, accel; **distance keeping** (min headway, time gap, min TTC) from the banked B1 eval lead block on the shared {1.0, 2.0} s instants, with the TTC-cap censoring count.
* **LATERAL:** heading, curvature (masked, beside the `ha0` straight-line floor), yaw-rate (`LAT_yaw_rate_mae_radps_valid`), cross-track.
* **TACTICAL** (true label clock, `--clip-clock-sidecar`, G3 exclusions printed with n): lat/lon decision accuracy + Cohen's κ for the z_tac heads and the v6 behaviour decoder (`tacv6_lat/lon_logits`), confusion matrices on in-band (±2 s) labelled windows; trajectory-derived manoeuvre agreement; anchor selection vs chance 1/117; **the 22-token goal selection** per class AUROC / AP / P / R @0.5 vs `tac_goal_y/w` with n_pos, n_neg (classes under n_pos 200 flagged UNSCOREABLE, nothing pooled).
* **STRATEGIC:** NOT APPLICABLE, n = 0 (§1).
* **MAP (10 cm)** and **BOX (VIS-1)**: §3.4.

### 3.4 Map and box scoring (surface M)
* **Map:** `taniteval/taniteval/map_hires_metrics.py` (landed, unchanged). refcv7 codes = `logits_to_codes(map_hires_logits, rule="prior_corrected", class_weight=<the run's frozen sqrt_mf weights>)`; refcv6@38k codes = `coarse_to_fine_codes(its 0.5 m map prediction, extent=100×±30)` (argmax over the 8 class channels, nearest 5×5, `NO_PREDICTION` outside its 60 m × ±16 m window — the favourable reading for the baseline); positional prior = `PositionalPrior` fitted on the **300 TRAIN clips with the smallest sha12** among Thor's `sam3_gt_v3` files that are not eval clips (pulled read-only; its fit-set refusal is kept). Scored cells: seen GT (`code != 255`) — the same cells for every arm. Per-window sufficient statistics in a `WindowTable`; `summarize` and `evaluate_bars_m7` produce the bars. ⚠️ SPEC_REFCV7 says the prior is "each cell's TRAIN-set majority class"; a 300-clip subset is a registered departure (compute/disk), not a tuned choice.
* **Box:** the trainer's own `detection_metrics.window_packs` / `summarise` on per-window packs (VIS-1 positives; IGNORE rows DontCare; last decoder layer; gate σ ≥ 0.5). refcv6@38k's box3d slots are scored against the SAME VIS-1 target blocks (the refcv7 eval dataset's) through the same functions. Interval: clip-cluster bootstrap that re-pools the packs of the resampled clips; paired for refcv7 − refcv6@38k.
* **Box confidence ratio** (A10 §15.3): confident slots (σ ≥ 0.5, IGNORE-matched excluded) / VIS-1 positives, per head; outside [0.5, 1.5] is a **Watch ALARM, not a bar**.

### 3.5 Bars — MILESTONE checkpoints only (step ≥ 5,000); NEVER on the validation checkpoint
Copied from SPEC_REFCV7; nothing is re-worded into a weaker form.

| id | statement (SPEC_REFCV7) | pass iff (paired episode-cluster bootstrap, n_boot 2000) |
|---|---|---|
| **BAR-R7-1** | `os − echo` (ADE 0–2 s) < 0, separated, at both inference seeds | S2, `os − ha0_ext` ADE: delta < 0 and CI upper < 0, at seed 0 AND seed 1 |
| **BAR-R7-2** | `os − refcv6@38k` < 0, separated, on the same windows | S2, `os_s − b_refcv6_38k_s` ADE, same inference seed s, at s = 0 AND 1 |
| **BAR-R7-3** | (non-regression) at 6 s, `os − echo` < 0, separated | S6, `os − ha0_ext` ADE 0–6 s, at both seeds |
| **BAR-M7-1** | lane / road-line IoU: refcv7 − refcv6@38k > 0, separated | per band (0–20 … 80–100 m), AND beats the positional prior, separated; PASS iff every band with GT cells passes (A8) |
| **BAR-M7-2** | crosswalk IoU, same rule | as M7-1 |
| **BAR-M7-3** | non-drivable edge F1 at 0.2 m tolerance, same rule | as M7-1 |
| **BAR-M7-4** | drivable IoU not separated WORSE than refcv6@38k in any band | per band |
| **BAR-B7-1** | box3d mAP@2 m (VIS-1, pooled classes): refcv7 − refcv6@38k > 0, separated | `eval_box3d_ap2m` (= `det_ap2_all_all`, AP over pooled classes); `det_map2_all` (class-mean) reported beside it, no verdict |
| **BAR-B7-2** | F1 at the declared rule (focal, gate 0.5): refcv7 − refcv6@38k at refcv6's best-F1 gate > 0, separated | refcv7 F1 @ σ ≥ 0.5 vs refcv6 F1 @ its best-F1 gate; that gate is chosen on the SAME scored windows (favours the baseline: conservative for the bar) and then held fixed inside the bootstrap |
| BAR-R7-N1 | NavSim PDMS > STOP | **not this battery** (NavSim stream) |

* Every bar is reported **PASS / FAILED / NOT PROVEN / NOT EVALUABLE** with its cell(s), n and interval. **NOT PROVEN:** separated in the right direction but |delta| ≤ 2 × the measured inference-seed floor (planner bars; for map/box bars no replicate floor is measured and that is stated). ⛔ A missed bar is FAILED; no goalpost moves after data.
* Per family, each family's headline metric is also reported paired against `ha0_ext` and against refcv6@38k (cells, no verdict).
* A2 sensitivity: every planner bar cell is also reported for `os_filteroff` (no verdict).

### 3.6 VOID gates (a panel that fails one is VOID, not negative)
1. `ha0`'s curvature and tactical κ are exactly 0. 2. `stop`'s displacement is exactly 0. 3. The model-free arms (`ha`, `ha0`, `ha0_ext`, `stop`) and `g` are **bit-identical** between the refcv7 dump and every baseline dump (paired delta 0.0, CI [0, 0]). 4. The trivial / selection profiles of `os` are non-degenerate (at the validation checkpoint a degenerate profile is expected and recorded, not gated). 5. Map: the `/3` GT inside the old 60 m × ±32 m window equals the `/2` `fine_codes` byte for byte on every scored clip (the A7 anchored-coordinate control); the refcv6 upsampling is checked on one window against a literal. 6. Perception is seed-invariant: the map/box keys have zero spread over G0's 8 seeds (else the perception pass is rolled at both seeds).

---

## 4. Runner and the armed milestone
* `code/pull_ckpt.py` — read-only, 3-way-md5 pull (Thor md5 before and after, local), step key asserted locally, REFUSED otherwise.
* `code/g0_refcv7.py` (+ `microbatch.py`, `wrapper_probe_r7.py`) — §2.
* `code/run_battery_r7.py` — G0 gate → rolls (child processes, GPU lock) → panels → families → bars → `raw/<tag>/battery_summary.json` + `RESULT_SECTION.md`.
* `code/chain_step5000.sh` + `code/waiter_step5000.sh` — detached; wakes on the EVENT (the step-5,000 checkpoint file complete on Thor and the step-5,000 eval row present), pulls, takes the GPU lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (exclusive create, job + pid) and requires `nvidia-smi` to show no other python compute app, then G0 → battery at both seeds. Assertions are on artifacts, never on an exit code through a pipe. ⛔ Nothing is pushed to HF.
* Priority at the milestone: this battery over NavSim (brief). Thor: `ls` / `md5sum` / `scp` pulls only.

## 5. What is NOT done here, stated in advance
* BAR-R7-N1 (NavSim) — sibling stream.
* The training-seed replicate — no second training run exists.
* Closed loop — none (T1 only).
* The A10 §15.3 INFORMATIVE P = R calibration readout — needs TRAIN calibration clips not in the kit.

## 6. The validation checkpoint (step 1,500)
**PIPELINE VALIDATION ONLY.** G0 is evaluated on it, and the battery is run end to end on it (a restricted smoke is allowed and is stamped `SMOKE_RESTRICTED`). **No bar is evaluated and no number from it is a result**; every artifact from it carries `"is_milestone": false`.

---

## AMENDMENT A1 — registered 2026-09-28 05:39 Europe/Berlin, BEFORE any refcv7 number existed (the G0 probe was still building its dataset; no loss, metric or output had been produced)

**What was learned (model-free, MEASURED from the loader's own dataset census, `raw/g0_step1500/probe.log`):** the loader's eval dataset index over the 139 eval clips holds **23,772 windows** (171 per clip; SAM3 10 cm GT `frac_ok` 0.9856; agent join 22,663/23,772 labelled). §3.1 registered surface **M** as "every window of the loader's eval dataset index". On the 8 GB dev box that is a full refcv7 forward on 23,772 windows plus the same for refcv6@38k — about five times the S2 roll, per model.

**A1.** Surface **M** is replaced by **the S2 windows** (4,754 windows / 139 clips; map scored on the 137 clips with `/3` GT; box on every S2 window with a VIS-1 row), i.e. the SAME windows the planner bars use.
* S2 is a fixed, model-free, pre-existing grid (the banked refcv4b 2 s grid; stride 5 frames = 0.5 s between window starts): exactly a 1-in-5 subsample of the index, chosen by a rule that predates refcv7. Consecutive index windows are 0.1 s apart, so the subsample drops mostly redundant windows. Nothing about it is tuned on any model output.
* ⚠️ **This departs from SPEC_REFCV7 §6.2's literal "every eval window".** It is registered here before any map or box number exists, and it is named as a departure beside every BAR-M7 / BAR-B7 verdict. The full-index read remains the registered upgrade if compute allows (a 23,772-window perception pass per model).
* Everything else in §3.4 is unchanged.

---

## AMENDMENT A2 — registered 2026-09-28 06:10 Europe/Berlin, AFTER the validation checkpoint's G0 read FAIL and BEFORE any milestone (step ≥ 5,000) number exists

**What happened (MEASURED, `raw/g0_step1500/g0.json`, step 1,500 = pipeline validation only).**
* **G0 as registered at step 1,500 = FAIL, on exactly ONE term:** `eval_box3d_det_ap4_stroller_0_20` (DETECTION), in-run 0.11111 vs reproduction 0.14286 on all 8 seeds (sd 0). That AP is computed on **n_pos = 1** (the row's own `eval_box3d_det_npos_stroller_0_20`): with one GT it is `1 / rank(TP)`, and 1/9 vs 1/7 is two low-confidence detections swapping ranks. Every other term held: 179 SMOOTH (median rel dev 3.6e-5), 6 STOCHASTIC inside their 99 % PI, 29 MATCHED (median 4.6e-4), 160 MAP10 counts (median 4.4e-5), 80 MAP10 IoU, 334 DETECTION (median abs dev 0.0), 210 COUNT exact, 98 null keys null in both. Strict load clean, `param_breakdown` equal, wrapper clause PASS under P3 (max wrapper rel 6.4e-6 vs floor 1.4e-7; W1 and W2 both detected).
* **That verdict stands as registered: G0 at step 1,500 = FAIL.** It is shown in every report beside the amended one.
* **A defect in this package's judge, found reading the same artifact:** the mutation count included terms already OUT under the unmutated reproduction, so M1 read "2 terms" where one was the stroller key above. SPEC §2 says a mutation must MOVE a term outside its tolerance; a term already outside is not moved. Corrected in `g0_refcv7.judge` for BOTH verdicts. Corrected M1 at step 1,500: 1 term (`eval_lon_tac`, SMOOTH, 1.68761 → 1.64529, rel 2.5 % vs 1 %). M2: 40, M4: 4.

**Why a fixed abs 0.02 cannot gate a low-support detection key.** A per-class / per-band detection metric on n GT events moves by ~1/n or more when ONE detection flips rank, class argmax or a greedy match across a distance threshold — and cross-hardware bf16 numerics (Thor vs RTX 4060; the requires-grad control MEASURED 1e-7–1.4e-6 relative output shifts on the slot heads) can flip exactly one. At n = 1 an AP can move by up to 0.67. Such a key has no power to tell a loader defect from numerics.

**A2 (applies to every milestone checkpoint; G0 as registered is ALWAYS reported beside it).**
1. A per-class or per-band DETECTION key (`*_det_ap{thr}_{cls}_{band}` with cls ≠ all, `*_det_map{thr}_{band}`, `*_rec@gate_{cls}`) whose support n — its own GT count from the in-run row (`*_det_npos_{cls}_{band}`, the smallest present class count for `det_map`, `*_npos_{cls}`) — is **below 30** is **REPORTED with its n and does not gate**. At n ≥ 30 its tolerance is **max(0.02, 2/n)** ("two discrete events"). Pooled detection keys (`ap2m`, `det_ap*_all_*`, `prec/rec/f1@gate`, `auroc_*`, `cls_acc_tp*`, `conf_ratio`, `centre_err_p50`, model-dependent counts) keep the registered tolerance.
2. Mutation detection counts only terms that are OK under the unmutated reproduction (the correction above).
3. Everything else in §2 is unchanged: every class tolerance and median, the wrapper clause, M1 must be detected (else VOID), the power probes.
4. The battery at a milestone proceeds iff **G0-A2 = PASS**. The step-1,500 re-judge under A2 is computed from the banked reproduction with zero GPU (`code/g0_rejudge.py`) and reported; it is still pipeline validation only.

---

## AMENDMENT A3 — registered 2026-09-28 06:55 Europe/Berlin, BEFORE any milestone number exists (compute; no bar, tolerance or verdict rule changes)

**What was measured (pipeline smoke on the step-1,500 checkpoint, `D:/refcv7_eval_kit/battery/smoke_step1500/roll_s1.json`):** the full §3.2 arm set costs **4.1 s per S2 window** on the RTX 4060 (8–9 forward rows per window: `os` + shuffled + flipped in one batch, the nav-null call, the 3-row `os_filteroff` replay, `os_vmaxzero`, and the refcv6 OBEDIENCE row on fast windows), i.e. **~5.4 h per inference seed** over 4,754 windows. The same card is shared with the NavSim and Research Lab (REFe) streams under one lock.

**A3.**
1. **Inference seed 0 rolls every §3.2 arm. Inference seed 1 rolls `os` only** (plus the model-free controls `ha`, `ha0`, `ha0_ext`, `stop`, which cost no forward). Every BAR-R7 reads only `os` (and the controls and baselines), so every bar is still evaluated at BOTH seeds, and the inference-seed replicate (`os_s0` vs `os_s1`) is unchanged. The perturbation and sensitivity arms (`os_navshuf`, `os_navzero`, `os_navflip`, `os_vmaxzero`, `os_filteroff`) and the refcv6 OBEDIENCE instrument are seed-0 readings and are labelled so.
2. **Perception pass speed-up, exactness VERIFIED per run:** the LAW target frame is a zero frame (the forward never sees future frames; `compute_losses_v3`'s `law` term is not read), guarded by a control on the first 2 windows of every pass: every perception tensor must be `torch.equal` to the real-LAW-frame forward, and the `law` term must DIFFER (the positive control that the swap took effect); a failed control REFUSES the pass. Frames are decoded one window ahead in a background thread; the map statistics run in a thread pool; the WindowTable keeps window order.
3. **Resumability (Master Mind 2026-09-28):** a G0 artifact for the same checkpoint md5 is reused; a seed whose dump is complete is not re-rolled; each stage is retried once; a collision watch logs any other python GPU process while the chain holds the lock (nothing is ever killed).

---

## AMENDMENT A4 — registered 2026-09-28 07:11 Europe/Berlin (Master Mind's direction), BEFORE any milestone number exists

**The refcv6@38k baselines are rolled ONCE and reused by every refcv7 milestone (5k / 15k / 20k / 30k / final).** refcv6@38k is a fixed checkpoint (md5 `5a2e7222a9f5f8c7aa7bf38ef4698d8a`), the S2 windows are fixed (4,754 / 139), and the inputs are md5-fixed (§0); re-rolling it per milestone would only re-spend ~4.5 h of GPU on identical inputs.
* **Where:** `D:/refcv7_eval_kit/baseline_refcv6_38k/` — `refcv6_step38000/dump_s{0,1}` (T1 `os`, the refcv6 package's own harness, with that package's own G0 / G0-A1 / G0-A2 on the 38k checkpoint) and `perc_refcv6_38k.{pkl,json}` (the perception dump). A stamp file `BASELINE_STAMP.json` records the code tree (`C:/Users/Admin/ev6_82c2331`, commit `82c2331`), the checkpoint md5, the config md5 (`a3193a46…`), the refcv6 package's SPEC sha256 it ran under, this SPEC's sha256, the S2 window digest, and the sha256 of every dump manifest. A sanitized copy is banked in this package under `raw/baseline_refcv6_38k/`.
* **Reuse rule:** a later milestone reuses the baseline iff the stamp's checkpoint md5, config md5, S2 window digest and dump-manifest sha256s all re-verify; otherwise it re-rolls and says why. The inference seeds pair exactly as before: refcv7 `os` at seed s against refcv6@38k `os` at seed s.
* **Scheduling (not a SPEC change; see README):** in the step-5,000 chain the 4.5 h refcv6@38k rolls run LAST, after refcv7's G0, rolls, map/box bars and an interim RESULT are banked. Until they land, BAR-R7-2 reads **PENDING**, never absent.

---

## AMENDMENT A5 — registered 2026-10-04 ~01:00 Europe/Berlin (time and sha256 in `raw/SPEC_SHA256_AMENDMENT_A5.txt`), BEFORE any G0 or battery number for step 15,000 / 20,000 / 30,000 / 50,400, and BEFORE any number from the step-5,000 diagnostic (`code/g0_diag_r7.py`, which had not started: it was waiting for host memory)

**What had been read when this was written.** The step-5,000 G0 artifact (`raw/step5000/g0.json`, incl. every seed's per-batch values) and the step-1,500 one; the run's own in-run eval rows for SIX keys (`eval_traj`, `eval_cascade`, `eval_sel_v3`, `eval_loss`, `eval_lat`, `eval_lon_tac`) at steps 500–12,000, 15,000, 20,000, 25,000, 30,000, 35,000, 40,000, 45,000, 49,000–50,400 (`metrics_final_50400.jsonl`), and Thor's `train.log` `[v3:eval]` lines. **No replay of any checkpoint other than 1,500 and 5,000 exists.** The rule below does not depend on any of those values.

**Why (the step-5,000 G0 failure, diagnosed from source and from the banked artifact — zero GPU).**
* G0 as registered and G0-A2 FAIL on ONE term: `eval_traj` [STOCHASTIC], in-run 0.64714 vs the 8-seed mean 0.673517, sd 0.00212 (`raw/step5000/g0.json`).
* Ruled OUT with file:line evidence (launch blob `refc_v3_train.py` 203b437f = ev7 copy, md5 0a6fb0d8…):
  * a step-keyed schedule in the eval path: the training loop mutates exactly one model attribute per step, `model.core.decoder.anchor_withheld_bank` (`:9410-9412`), and this run's `config.json` has `withheld_bank` mode `fixed`, warmup 0, so it is `"fixed"` at every step, in the run and in the loader; the LR schedule (`:8969-8971`, `apply_lr_schedule` `:9406`) touches only the optimiser; the tactical-prior EMA is gated on `model.training` (`:4608-4609`) and G0 records the buffers unchanged;
  * EMA / averaged weights: none exist in the trainer (no `ema`/`swa` state anywhere in `refc_v3_train.py`);
  * eval mode: the in-run eval sets `model.eval()` (`:9658`) inside `torch.no_grad()` (`:9672`); refcv6 F1's random-t draw runs only `if rv6.f1_random_t and self.training` (`refc.py:2800`), so both sides walk the published ladder;
  * the weights: `ckpt.pt` (`:9634-9644`) and `ckpt_{step}.pt` (`:9645-9647`) are written BEFORE the eval block (`:9656`) at the same `step`, from the same `model` object; nothing touches the parameters in between; `train.log` shows `ckpt step 5000 -> ckpt.pt` then `[v3:eval] step 5000 … traj 0.6471` with no restart anywhere in the run;
  * the windows: `inrun_eval_perm` sha256 identical; `V3Dataset.__getitem__` (`:3808-3937`) draws no random number; `num_workers=0`, `shuffle=False` (`:8908-8911`);
  * micro-batching: wrapper clause P3 max rel 1.5e-6 on `traj`;
  * trunk numerics (`--trunk-compile` dropped, Thor vs RTX 4060, bf16): **quantitatively excluded** — M1 perturbs the trunk enough to move `lat` 1.4 %, `goal_tac` 3.7 %, `lon_tac` 2.6 %, and moved `eval_traj` by only +0.12 % (0.67416 vs seed-0 0.67337); the in-run row agrees with the replay on those heads to ≤ 0.16 %, so a trunk-side difference cannot carry a 3.9 % `traj` shift.
* **What remains is the DDIM draw** (`refc.py:2802`, `eps = torch.randn_like(x0_n)` at eval, "stochastic at eval, by design"), which the run drew from its training RNG and the replay draws per inference seed — and the ESTIMATOR built on it. MEASURED on the banked per-batch values: the per-batch seed spread of `traj` (sd 0.017–0.033 per batch, 8 batches × 8 seeds) implies, if the 8 batches' draws are independent (disjoint windows, separate `randn_like` calls), an sd of the 128-window mean of **0.00878**; the 8 seed means have sd **0.00212** (ratio 0.24; under independence a 1-in-3,500 small-sample event, χ²₇ lower tail p = 2.8e-4). At step 1,500 the same ratio is 1.30. Either the K = 8 sd is an unlucky under-estimate, or the draws are structurally anti-correlated within a seed; the two are told apart by more seeds, not by more windows.

**A5.**
1. **STOCHASTIC class at every milestone checkpoint: K = 24 inference seeds (0..23).** The prediction interval is unchanged in form: `mean ± t(0.995, K−1)·sd·√(1+1/K) + 1 %·|mean|`, with `t(0.995, 23) = 2.807`. ⭐ This is **not a weaker bar**: under the same i.i.d.-normal seed-mean model both are exact 99 % PIs, and the expected half-width (excluding the unchanged 1 % term) falls from 3.58 σ (K = 8) to 2.83 σ (K = 24) — stricter on average; what it removes is the 1-in-thousands lottery of an 8-sample sd.
2. **Everything else in §2 and A2 is unchanged**: classes (assigned by name; SMOOTH vs STOCHASTIC by the seed spread, now over 24 seeds), tolerances, class medians, A2's low-support rule, M1 must be detected (else VOID), M2/M4 power probes, the wrapper clause. Mutations still run at seed 0 and are judged against the A5 intervals.
3. **Reporting order:** G0 as registered (seeds 0..7) FIRST, then G0-A2 (seeds 0..7), then **G0-A5 (seeds 0..23) — the gate**: the battery proceeds iff G0-A5 = PASS (this replaces A2 item 4).
4. **What A5 does NOT license:** if the 24 seeds confirm a small seed spread and the in-run value stays outside the A5 interval, G0 FAILS and the battery does not run for that checkpoint. A5 is a fix for an estimator, never for a mechanism.
5. **Implementation (memory, not numbers):** the 8 collated batches are written once to disk and re-read through a file-backed mmap (MEASURED 2026-10-04 00:33: the shared dev box at 2.0 GB free commit of 50 GB; a G0 process died in its first 4 MiB read with `MemoryError`; the 8 in-RAM batches are ~3.9 GB of G0's 9.9 GB RSS). The tensors are bit-identical; the diagnostic's `s0` arm must reproduce G0's stored seed-0 row (`eval_traj` 0.67337) through the same path. G0 also waits for free host commit before it starts.
6. **Step 5,000:** its registered verdicts (as registered FAIL, A2 FAIL) **stand**. An A5 re-judge of step 5,000 from seeds 0..23 (seeds 8..23 from the diagnostic) is computed and reported **POST HOC** — A5 was written after the step-5,000 numbers were seen — and is never a gate result for that checkpoint.
7. **The diagnostic (pre-registered here, both outcomes committed before any number):** `code/g0_diag_r7.py` on `ckpt_5000.pt`, the same 128 windows / 8 batches, arms `s0` (control: must reproduce 0.67337), `eps0` (DDIM draw zeroed), `seed8`..`seed23`, `fp32_s0` (trunk fp32 + NCHW, cuDNN TF32 off), `loaderflags_s0`, `micro_alt_s0` (3,3,3,3,4).
   * **24-seed sd of the mean ≥ 0.006** ⇒ the K = 8 sd was a small-sample under-estimate; mechanism = estimator; A5 is the fix.
   * **24-seed sd ≤ 0.004** ⇒ the draws are structurally anti-correlated within a seed (or the replay does not reproduce the in-run decode); then `eps0`, `fp32_s0`, `loaderflags_s0`, `micro_alt_s0` localise it: an arm whose `eval_traj` lands within 0.005 of 0.64714 names the lever; none ⇒ UNEXPLAINED, reported as such.
   * between 0.004 and 0.006 ⇒ INCONCLUSIVE, reported with the number.
