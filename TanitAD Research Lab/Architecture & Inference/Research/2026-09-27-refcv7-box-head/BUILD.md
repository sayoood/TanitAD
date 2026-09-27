# BUILD: refcv7's REFINED box heads (SPEC_REFCV7 §14 A9, §14.1, §15 A10, §15.5 A10.1, §16 A11)

Architecture & Inference, the refcv7 box-head builder, 2026-09-27 (Berlin). Reports to the Master Mind.
PI, verbatim (2026-09-27): *"based on the compariosn with proven refernce heads, refine our bb head and validate it"*.

Evidence classes: MEASURED (a run here, artifact named) · ANALYTIC (closed form / a script) · STATIC (a file:line at a
named tip) · INFERRED · UNVERIFIED. Tips: built on `68e8ec6`, re-based and tested on `49f10b4`, `4797ffb` (fixes
batch 3), `03e6ed9` (A11) and, FINAL, **`cef9709` (NEW-2)**. The tip has since moved to `ab1fb45` (SPEC and register
commits only): none of the 14 shared files changed, so the landing is valid there unchanged (STATIC, blob compare).

---

## 0. Headline

1. ⛔ **The refined box head FAILS its early (non-binding) G-BOX-OVERFIT and cannot overfit ONE frame -- and neither
   can refcv6's head.** MAIN at step 2,000: AP@2 m 0.013, 0 confident slots, below its image-zeroed control (0.037);
   +R6 0.102 (§7.2). The diagnosis (§7.4) finds the loss wiring CORRECT; a one-frame ladder fails in EVERY rung (frozen
   trunk, lr 2e-5, BCE, no deep supervision, 100 queries, the refcv6 head bundle): the Hungarian assignment never
   stabilises (0.4–6.5 % of targets keep their slot). The defect is the UN-ANCHORED slot decoder, and it predates A9.
2. ⭐ **Anchoring fixes it.** On the full model (trunk training, the launch optimiser, the canonical config) the
   one-frame test (A14.1) PASSES with learned reference points (`--slot-query-select learned_ref`: matched presence
   0.873, anchor kept 1.000, 12/13 confident); the unanchored red arm fails as required; HQS fails (ii) and is dropped
   (§7.7). The early G-BOX-OVERFIT on LRP memorises the 16 frames (10 of the 12 readings from step 900 pass all six
   criteria) but FAILS the registered step-2,000 reading on a late transient (R 0.832, 95 confident) -- a read-out
   decision is escalated.
3. ⚠️ **The prereg did not test the launch optimiser** (§7.5) -- corrected as A13 before any binding run: the harness
   now builds the launch's own optimiser (DD groups, trunk 0.5 x lr, weight decay 1e-4, clip 10) at its peak lrs.
4. **Built, tested, landing-ready -- three variants**, re-based on `1b8170e` (NEW-2 R3) with 0 conflicts and verified
   in place: ANCHORED `LANDING_READY_HQS.txt` md5 `9f1a11dc75c1ee660fcdc6211c9c4b35` (29 files; the repo-format list
   `LANDING_READY_ANCHORED.txt` carries the same entries + this package's own files); MAIN `LANDING_READY.txt`
   `e080785837ccdd941e1857d734bf042c` (27, HELD); +R6 `LANDING_READY_R6.txt` `5dcf7d884a3f168177bb362c0f56e1c7` (30).
   Tests: 101 A9 + 18 HQS/LRP + 11 R6, each guard with a red arm. Thor GATE environment, ANCHORED vs the clean tip:
   **0 attributable** on 2374cd2 (2,588 vs 2,470 passed) and on 1b8170e (2,606 vs 2,488; identical 42 failed + 19
   errors both times; `raw/tests_2374cd2/`, `raw/tests_1b8170e/`).
5. **The VIS-1 sidecar exists and is exact.** 4,508 clips (4,369 TRAIN + 139 EVAL), 859,497 NOW frames, 4,862,788
   rows, sha256 `278443b3356bca054c15753e7c08d465d71b0327a2ce564e143d091261349dd4` (MEASURED). Control 1 (the
   computation): the audit's 1,757 banked windows / 101,734 rows recomputed through the trainer's `_agent_item` + the
   vendored z-buffer, **0 mismatches**; control 2 (the STORED FILE): 28,633 in-scope rows, **0 mismatches**. 44 min
   on Thor, 12 workers, nice 19.

### Needs integration or a decision (escalated here, not in a README)

| # | what | who |
|---|---|---|
| I1 | Gate + land the ANCHORED variant (`LANDING_READY_HQS.txt`) once an anchored arm passes G-BOX-OVERFIT (§7.7); MAIN stays held unless the corrected MAIN or +R6 passes on its own. G-SUITE on Thor. | Master Mind |
| I2 | ~~PREREG CORRECTION~~ RESOLVED: registered as A13 (SPEC §18, option (a): the launch optimiser as built, peak lrs constant) and implemented + tested in the harness (§7.1, §7.5). | -- |
| I3 | **Old-record rebuilds (R4).** `N_QUERIES_DEFAULT` 100 -> 300 moves the refc argparse default. `refcv3_arm.rebuild_config` rebuilds a pre-A9 record at its STAMPED `seams.agents.queries`; other loaders that re-parse a recorded argv (`experiments/alpasim-gsplat/closedloop_drive.py`, `taniteval/tools/seam_probe.py`, `bench/cli.py`) were NOT patched. They fail LOUDLY (strict-load size mismatch), never silently. | Master Mind: patch or accept |
| I4 | **The sidecar's home.** 186 MB binary on Thor (`/home/nvidia/bx_0252/full/vis1_sidecar_refcv6b1_train4369_eval139.npz`) and the dev box (`C:/Users/Admin/bxh_work/sidecar/`), sha256-verified; NOT in git. The launch reads it with `--vis1-sidecar <path>`; the Master Mind's launch dir is `/home/nvidia/data/refcv7/`. | Master Mind / PI |
| I5 | **BAR-B7-1 "pooled classes"** read as the class-AGNOSTIC pooled AP@2 m (`eval_box3d_ap2m`); the class mean is also logged. Confirm. | Master Mind |
| I6 | The launch gate should call `tanitad.train.box_head_guard.check_refcv7_box_required(model, args)`; G-LIVE-PRES reads `{box3d,agent}_presence_frac_confident`. ⚠️ The refcv7 canonical argv (gate stream, 140 tokens) does not yet carry the A9 flags (`--slot-presence-loss focal --slot-presence-prior 0.01 --slot-deep-supervision --slot-vis1 --vis1-sidecar <path>`). | gate agent |
| I7 | **A11 closure.** The harness imports the audit loader from `stack/tanitad/eval/refcv6_loader.py` (vendored byte-for-byte + a provenance block; blob `11808258…` re-checked on import), so its closure lies inside the launch tree. | gate agent |
| I8 | ⚠️ The refcv7 canonical argv reads the box memory from NEW-2's pooled BEV (`--bev-source map_hires_pool`); every overfit number here is on the s16 lift (refcv6's argv + the A9 flags). A MAIN re-run must use the canonical config + the A9 flags. | Master Mind |

---

## 1. What was built, and where it plugs in

### 1.1 New modules (full files, `code/new/`)

| file | what |
|---|---|
| `stack/tanitad/models/slot_presence.py` | R1: `sigmoid_focal_elementwise` (== torchvision), `focal_presence_cost` (mmdet FocalLossCost), the ANALYTIC optimum (`presence_optimum`, `gate_match_belief`); R2: `layer_preds`, `select_slots`; R3: `ignore_presence_weight`; the refined losses `refined_slot_losses` / `refined_agent_losses` / `refined_box3d_losses` (re-matched per layer, summed); `presence_sanity` (G-LIVE-PRES); `refine_stamp`. The legacy configuration never reaches it (`refined_is_legacy`). |
| `stack/tanitad/data/vis1.py` | R3: the rule (T_POS 0.30, PX_MIN 100, radius 2.0 m), THE ONE split function `vis1_split` (train targets AND eval scoring), the deterministic sidecar writer, `VIS1Sidecar` (refuses missing clip / frame / row, wrong track, wrong centre, wrong store scope, wrong schema), `vis1_block_for_rows` (the dataset join). |
| `stack/tanitad/data/vis1_zbuffer.py` | the audit's z-buffer VENDORED VERBATIM (`vis_zbuf.py:39-144`, md5 `3721a83f…`), 4 function sha256s pinned and re-checked against the audit file; `clip_camera` = the audit's `camera()` with explicit inputs. |
| `stack/tanitad/eval/detection_metrics.py` | P0: per-window packs, the greedy DontCare matcher (`box3d_match_rows` on POSITIVE ∪ IGNORE), AP grid, AUROC ×2, the census, P/R/F1 at the gate, the A10 alarm, the INFORMATIVE P = R calibration gate and prior-corrected argmax, the calibration-window loader. |
| `stack/tanitad/data/box_calib_train256.json` | A10 §15.3: the FIXED TRAIN calibration windows (the audit's 256 / 64 clips), sha12 + window start only, digest verified on load. |
| `stack/tanitad/train/box_head_guard.py` | the refcv7 launch requirement: A9's levers ON in argv AND built on BOTH slot heads, 300 queries. |
| `stack/scripts/precompute_vis1_sidecar.py` | the sidecar builder (`--mode control` / `--mode full`), sharded, forked pool, nice 19, the trainer's own `_agent_item` for the target rows. |
| `stack/scripts/vis1_sidecar_control.py` | control 2: the stored file vs the audit's banked rows. |
| `stack/scripts/g_box_overfit.py` | the G-BOX-OVERFIT harness (§7): the prereg + A10; A11-ready (the loader vendored into stack/, the canonical argv OBJECT accepted, the argv hashed as compact JSON); **A13**: the LAUNCH optimiser as built (`build_optimizer` + clip 10, peak lrs constant); the refcv7 canonical config replayed (NEW-2's 10 cm branch + its fine store, `map_hires_pool`); **A14/A14.1**: the `--one-frame` mode; HQS-aware arms (memory_zeros blinds the heatmap too; `anchors_removed`). |
| `stack/tanitad/eval/refcv6_loader.py` | the box-head audit's loader VENDORED byte-for-byte + a provenance block (audit blob `11808258…`, re-hashed on import and by a test). |
| `stack/tests/test_refcv7_vis1.py`, `test_refcv7_box_head.py`, `test_refcv7_box_head_wiring.py`, `test_g_box_overfit.py` | 24 + 33 + 26 + 18 = 101 tests. |

### 1.2 Shared-file edits (`code/apply_box_head_edits.py`, anchored, EOL-preserving)

| file | edits |
|---|---|
| `agent_slots.py` (CRLF) | `N_QUERIES_DEFAULT` 100 -> **300** + the R4 re-ruling note; `PRESENCE_COSTS`; `AgentSlotDecoder(presence_prior=)` (None keeps 0.05); `deep_supervision` attribute; the per-layer forward (off path unchanged; last layer bit-identical); `match_slots(presence_cost="sigmoid"|"focal")`. |
| `box3d_head.py` (CRLF) | `Box3DSlotDecoder(presence_prior=)` passes through. |
| `refc_agents.py` (CRLF) | `AgentSeamConfig.queries = N_QUERIES_DEFAULT`; 4 DECLARED fields (`presence_loss`, `presence_prior`, `deep_supervision`, `vis1`) + stamps; the builder plumbs them; `agent_losses(vis=)` dispatches to the refined path only when asked. |
| `refcv6_perception_branch.py` (CRLF) | the same 4 fields on `PerceptionBranchConfig` (frozen) + stamps; the box decoder gets the prior and `deep_supervision`; `box3d_loss_row(presence_loss=, vis1=, vis=)`. |
| `refc_v3_train.py` (CRLF) | 5 flags; `_pin_slot_refine` (refuses every dead combination); `agent_queries_as_trained`; `_vis1_batch_block` (REFUSE, do not skip); V3Dataset `enable_vis1` / `_vis1_item`; the loss wiring for both heads (`select_slots`, per-layer extras, LOGGING_SPEC census, packs in EVAL); the in-run eval pools the packs (+ the alarm print) and runs the calibration pass; config.json `slot_refine` + `vis1`. |
| `declared_vs_built.py` (LF) | 5 entries: `slot_presence_loss`, `slot_presence_prior`, `slot_deep_supervision`, `slot_vis1` (built: read on BOTH heads and the loss configs), `vis1_sidecar` (data). |
| tests pinned at 100 / at the frozen stamp keys / at source text | `test_v6_agent_slots.py`, `test_refcv6_perception_supervision_fixes.py`, `test_refc_v3_agent_provenance.py`, `test_occ_knob_is_stamped.py`, `test_declared_vs_built.py` (count **213 -> 218**, bumped relative), `test_refc_v3_save_before_eval.py` (the step loop now holds THREE held-out passes -- the A10 calibration pass enters eval mode itself; a positional check added). |
| `taniteval/tools/refcv3_arm.py` (CRLF) | a pre-A9 record rebuilds at its stamped query count; the perception rebuild reads the 4 new stamp fields. |

---

## 2. R1: the presence objective and the DECLARED decision rule

- Loss: sigmoid focal, α 0.25, γ 2, weight 2.0, `sum / max(n_matched, 1)` over all slots of the batch (mmdet
  `avg_factor`). Matching: `2.0 × FocalLossCost(presence)` replaces `1.0 × (-σ(presence))`; class, centre, size terms
  unchanged (one variable per arm). Prior: 0.01. The audit's ln 10 tilt is NOT applied.
- **Decision rule (declared):** a detection is a slot with `σ(presence_logit) ≥ 0.5` on the LAST decoder layer.
  ANALYTIC (`gate_match_belief`, pinned by a test with a swapped-α red arm): under focal the 0.5 gate is a match belief of
  **0.75**; under the pre-A9 BCE-0.1 it was **1/11 = 0.091**.
- **What the planner consumes (STATIC, tip 4797ffb), unchanged except for the loss:** `refc.py:4684-4686` runs the agent
  head and hands its dict to `AgentTokenEmbed.forward` (`refc_agents.py:318-337`): `σ(presence_logit)` (`:323`) is
  feature 15 of `slot_features` (`:294`) and, in the default soft mode, the scale on every agent token (`:337`; the
  hard mode pads at `presence_gate` 0.5, `:330`). `refc.py:4694` reads the last layer's `box[..., :2]` (WP-B address).
  The per-layer `aux` list is read by no consumer but the loss.

## 3. R2: per-layer supervision

Shared `norm` + `head` after each of the 3 layers (0 parameters, MEASURED by test), each layer RE-MATCHED, losses
summed with weight 1; the off path and the last layer are bit-identical to the pre-A9 forward (test). Per-layer terms
reach the log as `{h}_layer{i}` / `{h}_presence_layer{i}`, count `{h}_n_layers` (= decoder depth, pinned for 1/2/3).

## 4. R3: VIS-1

- POSITIVE = in `visible_target_filter` ∧ `vis_frac ≥ 0.30` ∧ `n_vis ≥ 100 px`; IGNORE = the in-filter remainder
  (vis 0.05–0.30, < 100 px, vis < 0.05, no 3-D label, no silhouette); rows outside the filter are NEITHER (A10 §15.1).
  ⚠️ The first build made out-of-filter rows IGNORE too; A10 reconciled it before any number; a red arm pins it.
- Loss: IGNORE rows are out of matching; an UNMATCHED slot within 2 m (BEV) of an IGNORE row gets presence weight 0
  (its gradient is exactly 0, test). Eval: a detection greedy-matched to an IGNORE row is DontCare.
- `vis_frac = n_vis / n_full` formed in float64 from the integers (the audit's arithmetic).
- MEASURED census of the sidecar (`raw/vis1_sidecar_census.json`), per labelled NOW frame:

| split | frames | in-filter targets mean / p99 / max | VIS-1 positives mean / p99 / max | IGNORE mean / max | positives / in-filter |
|---|---:|---|---|---|---:|
| TRAIN | 833,247 | 5.40 / 33 / 103 | 3.31 / 15 / 40 | 2.09 / 94 | 61.4 % |
| EVAL | 26,250 | 5.47 / 31 / 120 | 3.22 / 14 / 43 | 2.25 / 95 | 58.8 % |

## 5. R4: 300 queries

`N_QUERIES_DEFAULT = 300`, the ONE spelling (`AGENT_QUERIES_DEFAULT`, `AgentSeamConfig.queries`,
`PerceptionBranchConfig.n_queries` now read it). Basis: ≥ 2 × the pre-VIS-1 max (120 on EVAL, 103 on TRAIN, MEASURED
above); after VIS-1 the max is 40 / 43, so 300 is ≥ 7 ×. Drop counter: 120 targets drop 0 at 300 and 20 at 100
(test). PARAM_BAND (ANALYTIC + MEASURED on the Thor build): box3d decoder **3,858,199** (was 3,806,999), agent head
**3,874,069** (was 3,822,869), both inside 2–4 M; the box decoder's remaining headroom is 553 memory tokens (NEW-2 must
stay inside it).

## 6. P0: detection metrics and the Watch contract

Module `tanitad.eval.detection_metrics`; the full key list is `raw/p0_watch_keys.json` (274 eval keys per head).
- **Eval row (pooled, never a batch mean):** `eval_{h}_prec@gate`, `_rec@gate`, `_f1@gate`, `_conf_ratio`,
  `_conf_ratio_alarm` (1 outside [0.5, 1.5]), `_ap2m`, `_auroc_matched`, `_auroc_objectness`, `_cls_acc_tp`,
  `_cls_acc_tp_priorcorr` (INFORMATIVE), `_centre_err_p50`, `_rec@gate_<cls>`, `_npos_<cls>`, the census
  `_n_conf`, `_tp@gate`, `_n_pos`, `_n_ignore`, `_n_dropped_hidden`, `_n_ignore_masked_slots`, `_n_windows`; the A9 grid
  `eval_{h}_det_ap{0p5,1,2,4}_{all|<cls>}_{all|0_20|20_40|40_60}`, `eval_{h}_det_map{thr}_{band}`,
  `eval_{h}_det_npos_{cls}_{band}`.
- **Calibration pass (INFORMATIVE, A10 §15.3):** `eval_{h}_calib_pr_gate`, `_calib_prec`, `_calib_rec`,
  `_calib_n_pos`, `_calib_n_windows` on the 256 banked TRAIN windows.
- **Train row:** `{h}_n_pos`, `_n_ignore`, `_n_dropped_hidden`, `_n_ignore_masked_slots`, `_n_conf`, `_conf_ratio`,
  `_tp@gate`, `{h}_presence_frac_confident` (G-LIVE-PRES), `{h}_layer{0,1,2}`, `{h}_presence_layer{0,1,2}`.
- Declared rules: bands by forward `cx`; per class nuScenes-style; NaN written as null. Not emitted (A10): any
  calibrated-probability key (sum_phat, G-LIVE-COUNT).

## 7. G-BOX-OVERFIT, the diagnosis, and the anchored lever

### 7.1 The harness
Prereg (md5 `594c7119…`) + A10: 16 frames, 113 POSITIVE / 77 IGNORE, batch 4, 2,000 steps, seed 0, the LAUNCH box loss
(the trainer's own `compute_losses_v3` `box3d` term), scored under the eval rule at σ ≥ 0.5; the model is a replay of
`train()`'s build CHECKED by `declared_vs_built.check` (0 mismatches on every config below). Must-fail `memory_zeros`,
`presence_w0`; controls C1–C4. **A13** (SPEC §18): the optimiser is the trainer's own `build_optimizer` on the launch
argv (`--opt dd --lr 1e-4`: AdamW, trunk at 0.5 x lr, weight decay 1e-4) + `clip_grad_norm_` 10.0, held at the PEAK
lrs; a test holds the harness's groups / lrs / weight decay / clip EQUAL to the trainer's build (red arm: the early
constant-2e-4 config). **A14.1** (§19.1): `--one-frame` = the ladder's frame, 500 steps, PASS iff (i) matched presence
median ≥ 0.50 and (ii) the matched ANCHOR is kept over the final 100 steps ≥ 0.80 (the cell under HQS, the query
otherwise); the red arm `unanchored` (= the MAIN decoder) must fail.

### 7.2 The early non-binding runs (Thor GPU; tip 4797ffb + this patch; refcv6 argv + A9 flags; the OLD optimiser)
MEASURED, `raw/thor_gbo/`. Every arm FAILS; MAIN scores below its image-zeroed control.

| arm | AP@2 m | P / R at σ ≥ 0.5 | Σ confident (113) | centre / size / z p50 | presence(2000) / presence(0) | verdict |
|---|---|---|---|---|---|---|
| MAIN | 0.013 | nan / 0.000 | 0 | 2.99 / 0.99 / 0.20 m | 0.445 / 0.906 | FAIL |
| memory_zeros | 0.037 | nan / 0.000 | 0 | 1.05 / 0.64 / 0.17 m | 0.449 / 0.979 | failed as required |
| presence_w0 | 0.001 | nan / 0.000 | 0 | 4.35 / 0.60 / 0.20 m | 1.164 / 0.906 | failed as required |
| +R6 (A10.1, 5 DN groups) | 0.102 | nan / 0.000 | 0 | 0.65 / 0.40 / 0.10 m | 0.438 / 0.906 | FAIL |

### 7.3 Toy diagnosis (INFERRED; a toy, not the launch path)
`code/tools/toy_presence_probe.py`, `toy_slot_diversity.py`, `code/r6_banked/toy_r6_probe.py`: with surplus queries the
focal presence of EVERY slot sits at its base rate and the Hungarian winner of a target changes between evaluations
(12–38 % kept) -- the first sign of what §7.4 measured on the real path.

### 7.4 The diagnosis (MEASURED, Thor GPU, NON-BINDING; `raw/diag/`)
- **Signal audit** (MAIN config, the first 200 steps of MAIN's schedule): the loss wiring is CORRECT -- matched slots
  get presence gradient 0.012–0.035, 100 % pushed up; IGNORE never zeroes a matched slot (33/33 layer readings);
  every layer's presence targets are that layer's own `match_slots` result (33/33). The presence logits reach the
  focal BASE-RATE optimum (~-1.8, ANALYTIC p* ≈ 0.14) by step ~60 with matched ≈ unmatched. The memory entering the
  decoder loses its frame-specific share 37 % -> 6 % within 20 steps.
- **One-frame overfit** (frame `384cb23868d0` t = 149, 13 POSITIVE): FAIL -- matched presence ≤ 0.29 in 500 steps;
  the Hungarian assignment keeps 0–15 % of targets' slots between readings 25 steps apart.
- **One-frame ladder** (one variable per rung; `gbo_ladder.json`): EVERY rung fails, the refcv6 head included --

| rung | matched p median @500 | unmatched max | conf | AP@2m | slot kept over 25 steps (mean) | memory frame-share 0 -> 500 |
|---|---|---|---|---|---|---|
| MAIN | 0.188 | 0.194 | 0 | 0.009 | 0.020 | 0.408 -> 0.126 |
| frozen_trunk | 0.145 | 0.236 | 0 | 0.036 | 0.016 | 0.408 -> 0.223 |
| lr 2e-5 | 0.177 | 0.231 | 0 | 0.085 | 0.012 | 0.408 -> 0.204 |
| BCE presence | 0.280 | 0.447 | 0 | 0.006 | 0.004 | 0.408 -> 0.174 |
| no deep supervision | 0.184 | 0.202 | 0 | 0.237 | 0.012 | 0.408 -> 0.044 |
| 100 queries | 0.244 | 0.342 | 0 | 0.213 | 0.065 | 0.408 -> 0.126 |
| refcv6 head (BCE-0.1, prior 0.05, no deep sup, 100 q, refcv6 targets) | 0.628 | 0.646 | 100/100 | 0.081 | 0.056 | 0.408 -> 0.039 |

  The 16-frame refcv6-head CONTROL (the harness loop, BCE-0.1, 100 queries, refcv6 targets, VIS-1 scoring) was paused
  for the ladder and STOPPED after the one-frame tests (the ladder answered its question); its partial rows (MEASURED,
  `raw/thor_diag/diag.log`): step 100 AP 0.006, 0 confident; step 200 AP 0.012, P 0.028 / R 0.389, **1,573 of 1,600
  slots confident** -- BCE-0.1's base rate above the gate, the audit's finding reproduced under the harness.
  Reading (INFERRED, published support): the known bipartite-matching instability of learned, UN-ANCHORED queries
  with absolute box regression; it predates A9. BCE-0.1's base rate sits ABOVE the 0.5 gate (every slot fires), focal's
  below it (none fires) -- the PI's "messy boxes" in one line.

### 7.5 The optimiser the prereg did not test (STATIC; registered as A13, SPEC §18)
The launch trains AdamW with DD's two groups (`refc_v3_train.py:6988 build_optimizer`, `:7025`, `timm_trunk.py:1316`),
trunk 5e-5 / head 1e-4 peak (`:10644`, `:10141`), 2,000-step warm-up then cosine (`:8644`), weight decay 1e-4
(`:10641`), `clip_grad_norm_` 10.0 (`:9203`), batch 16; the early harness used 2e-4 constant on every tensor, no
clip. The corrected harness (A13) builds the launch optimiser itself; CPU check-builds on the canonical config read
`opt=dd AdamW wd=1e-4 encoder_lr=5e-5 (316 tensors) head_lr=1e-4 (486-494 tensors)`, G-DVB 0 mismatches.

### 7.6 The decoder bench (MEASURED, dev-box GPU, NON-BINDING, INFORMATIVE; `raw/diag/bench_*.log`)
The box head alone on the box memory's inputs CAPTURED from the corrected canonical config (`capture_oneframe.py`, md5
`eaffde47`), trunk + BEV frozen, the launch head group (lr 1e-4, wd 1e-4, clip 10), 500 steps, the landed loss:

| mode | matched p median @500 | unmatched max | conf / 13 | slot kept, final 100 | anchor kept, final 100 |
|---|---|---|---|---|---|
| learned (MAIN) | 0.172 | 0.263 | 0 | 0.000 | -- |
| heatmap (HQS) | **0.764** | 0.382 | **12** | 0.115 (top-K rank churn) | **0.942** |
| learned reference points | **0.879** | 0.053 | **13** | **1.000** | (the points move) |

ANCHORING fixes the one-frame failure. The bench is why A14 was amended BEFORE its first Thor number (A14.1: (ii) on
the ANCHOR; red arm = `unanchored`; learned reference points = candidate `learned_ref`).

### 7.7 The anchored arms on the full model (Thor GPU, trunk training, A13 optimiser, canonical config) -- MEASURED
One-frame tests under A14.1's literals (`raw/thor_oneframe/`, NON-BINDING), frame `384cb23868d0` t = 149:

| config | (i) matched p median @500 (bar 0.50) | (ii) anchor kept, final 100 (bar 0.80) | confident / 13 | AP@2m | verdict |
|---|---|---|---|---|---|
| `learned` = unanchored (the red arm, the MAIN decoder) | 0.168 | 0.000 (query) | 0 | 0.022 | FAIL, as required |
| `heatmap` (HQS) | 0.589 | 0.442 (cell) | 11 | 0.906 | FAIL on (ii) -> dropped |
| `learned_ref` (LRP) | **0.873** | **1.000** (query) | 12 | 0.839 | **PASS** |

HQS detects (AP 0.91 on the frame) but its heatmap peaks move between neighbouring 0.5 m cells while the trunk trains,
so the exact-cell anchor changes about half the time (INFERRED); (ii) is a literal and the FAIL stands. The early
G-BOX-OVERFIT on LRP (MAIN arm only -- the Master Mind stops the early run after MAIN; the must-fail arms count in
the BINDING run on the launch closure) -- MEASURED, NON-BINDING (`raw/thor_gbo_lrp/`), stopped by PID after MAIN:

| # | criterion @ step 2,000 | value | pass |
|---|---|---|---|
| 1 | AP@2 m >= 0.90 | 0.996 | yes |
| 2 | P >= 0.90 and R >= 0.90 at sigma >= 0.5 | 0.989 / 0.832 | **no** |
| 3 | abs(sum confident - 113) / 113 <= 0.10 | 95 -> 0.159 | **no** |
| 4 | centre / abs(dl)+abs(dw) / abs(dz) medians <= 0.30 / 0.30 / 0.15 m | 0.141 / 0.086 / 0.078 | yes |
| 5 | greedy-TP class accuracy >= 0.90 | 0.989 | yes |
| 6 | finite and presence(2000) <= 0.25 x presence(0) | 0.028 / 1.275 | yes |

**FAIL as registered -- on the last reading only.** From step 900 on, 10 of the 12 readings pass all six criteria
(AP 1.000, P 1.000, R 0.92-1.00, 104-113 confident, AUROC 1.000); the two that fail are step 1,200 (z 0.199) and step
2,000 (a late transient: the loss rose 2.16 -> 3.05 and the presence term 0.0018 -> 0.0285 over the last 100 steps,
while the ranking stayed perfect -- AUROC 1.000, the informative P = R 0.973 / 0.973). INFERRED: the constant peak lr
(A13) at batch 4 makes a single-snapshot read-out a coin flip against transients. Decision escalated (the Master Mind /
PI): keep the literal, or register a transient-robust read-out (e.g. every criterion on the median of the last 5 log
points: here R 1.000, 113 confident, z 0.070) BEFORE the binding run.

## 8. The variants

- **MAIN** (`LANDING_READY.txt`, 27 files): A9/A10 + the harness. Held (the Master Mind) unless the corrected MAIN or
  +R6 passes G-BOX-OVERFIT on its own.
- **+R6** (`LANDING_READY_R6.txt`, 30 files): A10.1's denoising queries behind `--slot-dn-groups` (default 0),
  `code/r6_banked/`; 11 tests with a group-leak red arm; deprioritised by A14.1.
- **ANCHORED = MAIN + HQS/LRP** (`LANDING_READY_HQS.txt`, 29 files): `stack/tanitad/models/slot_query_select.py` +
  `code/hqs/apply_hqs_edits.py` on 5 shared files. `--slot-query-select {learned,heatmap,learned_ref}`:
  - `learned` (DEFAULT) is BIT-IDENTICAL to MAIN: digest `7d7c61e6…` recorded on MAIN and reproduced (a digest test),
    and a raise-on-touch test proves the default never reaches the module (red arm: a heatmap build);
  - `heatmap` (HQS, A14): a 2-conv BEV centre heatmap (prior 0.01; CenterNet Gaussian targets of the VIS-1 POSITIVES,
    IGNORE cells masked; penalty-reduced focal α 2 β 4 / #positives, weight 1.0 INSIDE the box term), 3x3 NMS +
    top-K anchors (detached), the anchor's sine position through an MLP added to the self-attention Q/K and the
    cross-attention Q at every layer (the layer's own modules, no new layer parameter), box centre = anchor +
    tanh(raw) x 4 m;
  - `learned_ref` (LRP, A14.1): one trainable (x, y) anchor per query (own generator: no global-RNG draw), the same
    position embed and anchor-relative centre, no heatmap, no BEV requirement;
  - pins refuse both with `--w-box3d 0` and `heatmap` without BEV features; G-DVB `slot_query_select` (count +1) reads
    the built modules both ways; the stamp key `query_select`; 18 tests + the harness's HQS arm tests.

## 9. UNVERIFIED / not done

- The anchored arms' one-frame and G-BOX-OVERFIT verdicts on the full model (§7.7) are pending on Thor.
- The in-run P0 path and the calibration pass have run in unit tests and in the harness's eval (the same
  `compute_losses_v3`), but NOT inside `train()`'s eval loop on real data: the Thor G-LIVE smoke is the first run that
  exercises it.
- The paused refcv6-head control (pid 3603614) is to be stopped after the one-frame tests (the Master Mind), its partial
  rows banked from its log.
- Nothing was pushed or committed; nothing was written to G:.

## 10. BINDING-run prep (for the Master Mind; nothing here has run)

- **Canonical argv additions** (`stack/ops/runs.d/refcv7-r101-s0.argv.json`, whose `todo_box_head` leaves the box flags to
  this package): `--slot-presence-loss focal --slot-presence-prior 0.01 --slot-deep-supervision --slot-vis1
  --vis1-sidecar /home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz --slot-query-select learned_ref`
  (the last one only if the LRP amendment is registered).
- **The VIS-1 sidecar at launch**: copy `/home/nvidia/bx_0252/full/vis1_sidecar_refcv6b1_train4369_eval139.npz` to
  `/home/nvidia/data/refcv7/` (sha256 `278443b3356bca054c15753e7c08d465d71b0327a2ce564e143d091261349dd4`); the flag
  that reads it is `--vis1-sidecar`; the dataset REFUSES a missing clip / frame / row or a mismatched track / centre.
- **The binding G-BOX-OVERFIT under A11** (the launch tree is stack/ + taniteval/ + tools/; the audit's prereg and
  frame set are DATA, so `--audit-dir` may point at a copy of the audit's `raw/` on Thor -- the harness reads only
  `raw/PREREG_G_BOX_OVERFIT.md` and `raw/visibility/gbo_frameset.json`; its loader is vendored in stack/):

      python stack/scripts/closure_run.py --out <dir>/gbo_closure.json --result <dir>/gbo_binding.json \
        --data <audit>/raw/PREREG_G_BOX_OVERFIT.md --data <audit>/raw/visibility/gbo_frameset.json \
        --data /home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz \
        --binding --commit <launch sha> --tree <launch tree> -- \
        stack/scripts/g_box_overfit.py --launch-argv stack/ops/runs.d/refcv7-r101-s0.argv.json --audit-dir <audit> \
          --out <dir>/gbo_binding.json --device cuda --binding --commit <launch sha> \
          --candidate "BINDING: launch commit <sha>" --arms main,memory_zeros,presence_w0

  The harness drops `--trunk-compile` itself and records it; its optimiser is the launch's (A13); the run is ~2.7 s/step
  per arm on Thor alone (MEASURED on the early LRP run), i.e. ~4.5 h for the three arms.
