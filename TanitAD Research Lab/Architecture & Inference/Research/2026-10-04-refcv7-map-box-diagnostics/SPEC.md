# SPEC — refcv7-r101-s0 map and box diagnostics (PRE-REGISTRATION)

Architecture & Inference, 2026-10-04. Written BEFORE any diagnostic number was computed; its sha256 and UTC
time are in `raw/SPEC_SHA256.txt`. Anything added after that time is marked **POST-HOC** in RESULT.md.

**Evidence tier: these are OPEN-LOOP PERCEPTION diagnostics** (a single forward per window, teacher-forced
inputs, no planner in the loop). Nothing here is a T1 or driving number.

**What I had seen before writing this (disclosure).** The run's own `metrics.jsonl` eval row at step 50,400
(the in-run monitor, 128 windows): it logs per-class × band `inter`/`union` under BOTH the declared
`prior_corrected` rule and the raw argmax (`interraw`/`unionraw`), and `eval_box3d_conf_ratio 0.0059`
(`n_conf 2`, `n_pos 339`), `eval_agent_conf_ratio 0.0`. The raw-argmax keys suggest that the declared rule
predicts far fewer thin-class cells than the raw argmax. That is the hypothesis this SPEC tests; it was
formed from those 128 logged windows, so the decision rules below are scored on the 1,112-window eval
sample and with fits made on TRAIN only.

## 0. Object under test

* Run `refcv7-r101-s0`, launch tree `/home/nvidia/refcv7_run/fec3a0dccf`, run dir read-only.
* Final checkpoint `ckpt.pt` step 50,400, md5 `d5f104ee54ba6b2861e38030b4f6fcf1` (verified 2026-10-04).
  Milestones `ckpt_{5000,15000,20000,30000}.pt` for M-e.
* Model built by the run's own eval loader `stack/tanitad/eval/refcv7_loader.py` (`build_model`, STRICT load,
  G-DVB + stamp checks) on Thor with `REFCV6_KIT=/home/nvidia`, `REFCV6_REPO=<launch tree>`; `--trunk-compile`
  dropped (forward-identical, recorded by the loader). Forward + targets through the trainer's own
  `compute_losses_v3` in `model.eval()` under `torch.no_grad()`. The map logits are captured by wrapping
  `map_head_hires.map_hires_loss_row` (the original still runs; its `inter`/`union` output is CONTROL C1).
  Box rows come from the trainer's own `_det_pack_box3d` / `_det_pack_agent` (`detection_metrics.window_packs`).

## 1. Window sets (split discipline)

* **EVAL-INRUN** — the trainer's fixed in-run subset: `torch.randperm(len(e_ds), Generator().manual_seed(12345))[:128]`
  (`refc_v3_train.py:8905-8907`). Used ONLY for reproduction controls (C1, C2/B-a).
* **EVAL-DIAG** — all 139 eval139 episodes × K = 8 windows each = **1,112 windows**. Per episode the windows
  are `ts[floor((j + 0.5) * len(ts) / 8)]`, j = 0..7, over the episode's sorted window starts `ts`
  (`e_ds.index`). **SCORED, never tuned on.**
* **TRAIN-DIAG** — 139 TRAIN episodes: the train cache's clips sorted by `sha12 = sha256(clip_id)[:12]`,
  the first 139 whose 10 cm GT file `<sha12>.sam3mapgt.npz` exists in `sam3_gt_v3`; same 8 window positions
  = **1,112 windows**. Built through the same loader path (`build_eval_dataset`) on a symlink subset cache, so
  train and eval windows go through ONE pipeline (no augmentation on either). **All fits are made here.**
* **TRAIN-CALIB256** — the banked A10 calibration set `stack/tanitad/data/box_calib_train256.json`
  (64 train clips × 4 windows, digest-verified by `detection_metrics.load_calib_windows`). Box SENSITIVITY fit
  set only (secondary).
* Every threshold, temperature, logit shift and logit offset is FITTED on a TRAIN set and SCORED on EVAL-DIAG.
  Anything tuned on eval is labelled **ORACLE (ceiling, inadmissible as a result)**.

## 2. Map definitions (10 cm head, 100 m × ±30 m, 1000 × 600 cells)

* **Classes** 0 nocls, 1 drivable, 2 lane, 3 crosswalk, 4 arrow, 5 edge, 6 hatched, 7 sidewalk
  (`map_head_hires.CLASS_KEYS`). **Thin classes (declared): lane, crosswalk, arrow, edge, hatched** = every class
  whose corpus frequency in the run's class-weights file is < 2 % (1.02 / 0.78 / 0.10 / 0.36 / 0.19 %).
* **Supervised cells** = GT code ≠ 255 AND lift-valid (0.25 m `map_hires_lift_valid` mapped to 0.1 m by
  `lift_valid_to_fine`), exactly the in-run loss/metric cells (`per_class_signal`).
* **Bands** = the trainer's 20 m forward bands by fine row (`band_keys_for_rows`): 0_20, 20_40, 40_60,
  60_80, 80_100; plus `all`.
* **Decision rules.** `pc` = declared `prior_corrected` = `argmax_c (z_c − ln w_c)` (`map_head_hires.decide`,
  w = the run's frozen class weights); `raw` = `argmax_c z_c`; `off` = `argmax_c (z_c − ln w_c + δ_c)` with δ
  fitted on TRAIN-DIAG (M-c iv).
* **IoU** (M-a) = pooled `Σ|P∩G| / Σ|P∪G|` over windows (sum/sum, never a mean of ratios), per class × band,
  null when the union is 0.
* **Tolerant metrics** (M-b), k ∈ {0, 1, 2} cells (0 / 0.1 / 0.2 m), Chebyshev neighbourhood (square
  (2k+1)²), supervised cells only: `TPp_k` = predicted-c cells with a GT-c cell within k; `TPg_k` = GT-c cells
  with a predicted-c cell within k; `P_k = ΣTPp_k/Σ|P|`, `R_k = ΣTPg_k/Σ|G|`, `F_k = 2PR/(P+R)`,
  **`IoU_k = F_k / (2 − F_k)`** (at k = 0 this is EXACTLY the IoU, since F_0 is the Dice coefficient).
* **Counts** (M-c i/ii): `N_pred(rule)`, `N_gt`, ratio `N_pred/N_gt` per class × band.
* **Threshold sweep** (M-c iii): per class, one-vs-rest mask `{s_c ≥ τ}`, for two scores: `s = p̂` (the
  calibrated posterior `softmax(z − ln w)`) and `s = q` (raw `softmax(z)`). Computed exactly from per-class ×
  band histograms of `logit(s)` on GT-positive and GT-negative cells (4,000 bins over [−20, 20], clipped).
  `τ*_c` (one per class and score, over all bands) maximises pooled TRAIN-DIAG IoU; EVAL IoU is reported at
  `τ*_c`, beside the EVAL-optimal ORACLE ceiling.
* **Offset rule** (M-c iv): δ (δ_0 = 0) fitted by coordinate ascent (grid −3…+6 step 0.25, 3 sweeps) to
  maximise mean IoU over classes 1-7 (all bands pooled) on a uniform 1 % cell subsample of TRAIN-DIAG
  (fixed seed); scored EXACTLY on all EVAL-DIAG cells.
* **AUROC** (M-d) per class × band of `p̂_c` (and `q_c`) on GT-c vs GT-not-c supervised cells, from the same
  histograms (ties within a bin counted half), plus the mean score on each side.
* **M-e**: EVAL-DIAG at 5,000 / 15,000 / 20,000 / 30,000 / 50,400: IoU (pc, raw), counts, AUROC and the
  ORACLE best-threshold IoU (trend diagnostic only — it says whether a separable signal existed).
* **M-f** (labels): (1) GT class fraction per band on supervised cells, both splits; (2) GT line width = the
  lengths of maximal lateral (along y) runs of class c per fine row, median / p90 per band, for the thin
  classes; (3) registration spot-check: Chebyshev distance (cells, capped at 20) from each GT-edge cell to the
  nearest drivable-boundary cell of (a) the GT drivable mask and (b) the PREDICTED (pc rule) drivable mask,
  histogram {0, 1, 2, 3-5, 6-10, 11-20, >20}; boundary = a drivable cell with a 4-neighbour supervised
  non-drivable cell.
* **Interval**: episode-cluster bootstrap over the 139 episodes (resample episodes, recompute the pooled
  ratio), B = 1,000, 95 % percentile. Train-vs-eval differences: independent resamples of the two splits
  (different episodes, unpaired). This interval answers *another draw of EPISODES*; it is blind to training-run
  variance (one seed, no replicate) — stated beside every interval.

## 3. Box definitions (box3d head first, then the agent head = **B-d**)

* **B-a** `conf_ratio = n_conf / n_pos` as defined in `stack/tanitad/eval/detection_metrics.py` `_census`
  (greedy 2 m rows over slots with `sigmoid(presence_logit) ≥ 0.5`, IGNORE-matched rows = DontCare excluded;
  `n_pos` = VIS-1 positives) — file:line quoted in RESULT. Reproduced on EVAL-INRUN.
* **B-b** rows = `detection_metrics.greedy_rows(pk, 2.0)` over ALL slots (no gate): TP (1), FP (0), DontCare
  excluded. Histograms of p for TP vs FP (20 equal bins); reliability diagram, 15 equal-width bins;
  `ECE_all` over all rows, `ECE_≥0.05` over rows with p ≥ 0.05; `Σp / n_pos` (expected-count ratio).
* **B-c** transforms of the presence logit z (monotone, global, so rows' TP/FP labels and AP are unchanged):
  * T0 identity (declared, gate 0.5);
  * T1a analytic prior shift `z + ln(0.99/0.01) = z + 4.5951` (no fit);
  * T1b fitted shift `z + b*`, b* = argmin BCE(TP vs FP rows) on TRAIN-DIAG;
  * T1c analytic focal inversion: p → π(p), the match belief whose focal optimum is p
    (`slot_presence.presence_optimum`, α 0.25, γ 2), tabulated (no fit);
  * T2 temperature `z / T*`, T* = argmin BCE on TRAIN-DIAG; T2b Platt `a z + b` (BCE on TRAIN-DIAG);
  * T3 threshold at the P = R point: gate `g*` = `detection_metrics.pr_equal_gate(TRAIN-DIAG packs)["gate"]`,
    applied to the raw score.
  Reported on EVAL-DIAG for each: conf_ratio, prec / rec / F1 at the gate, AP@2 m and AP@4 m (all classes, all
  bands; must read IDENTICALLY across T0-T2b — CONTROL C5), ECE_all, ECE_≥0.05, Σp/n_pos. Sensitivity: the
  same fits on TRAIN-CALIB256.
* **Interval** (box): episode-cluster bootstrap over eval episodes, B = 1,000, for conf_ratio / P / R / F1 at
  the gate (paired: T0 vs the chosen transform on the same resamples).

## 4. Controls that must read known values (tests in `code/`)

* C1 — my map inter/union on EVAL-INRUN under both rules == the trainer's own `map_hires_inter*`/`union*` for
  the same batches (exact), and == the logged step-50,400 eval row (float tolerance 1e-6 relative).
* C2 — B-a: `n_conf`, `tp@gate`, `n_pos` on EVAL-INRUN at ckpt.pt == the logged 2 / 2 / 339 for box3d and
  the logged agent values.
* C3 (unit) — GT-as-prediction → IoU 1.0 (and IoU_k = 1 for all k); empty prediction → IoU 0, recall 0 for
  every class; 1-cell-shifted GT line → IoU_0 low (< 0.1 for a 1-cell line), IoU_1 = 1.0.
* C4 (unit) — histogram AUROC: perfect score → 1.0, constant score → 0.5; histogram threshold-IoU of a
  perfect score peaks at 1.0; histogram IoU at τ equals the direct mask IoU.
* C5 — AP@2 m / AP@4 m identical (to 1e-12) across T0, T1a, T1b, T1c, T2, T2b on EVAL-DIAG.
* C6 (unit) — the shift / temperature fit recovers a known planted b / T on synthetic rows; ECE of a
  perfectly calibrated synthetic sample < 0.02.

## 5. Decision rules (committed before the data)

**Alarm 1 — map thin classes.** For each thin class, on EVAL-DIAG, all bands pooled:
* **D1 DECISION-RULE SUPPRESSION** if `N_pred/N_gt (pc) < 0.2` AND the best TRAIN-fitted alternative (raw,
  τ*, or δ) gives EVAL IoU ≥ 2 × the pc IoU with the bootstrap interval of the difference excluding 0.
  ⇒ inference-time fix (change the decision rule; no retraining).
* **D2 NO SIGNAL** if AUROC(p̂) < 0.80 AND no TRAIN-fitted alternative reaches EVAL IoU 0.05.
  ⇒ representation / optimisation failure ⇒ recipe change.
* **D3 LOCALISATION-LIMITED** if AUROC ≥ 0.90 but the best alternative's IoU_0 < 0.15 AND IoU_2 ≥ 2 × IoU_0.
  ⇒ the head sees the line but not to the cell ⇒ recipe (resolution / boundary-tolerant loss) or GT
  registration (M-f decides which).
* **D4 GENERALISATION GAP** if TRAIN τ* IoU − EVAL τ* IoU > 0.10 with the interval excluding 0; **D4'
  OPTIMISATION** if TRAIN τ* IoU < 0.15 as well (the head does not fit its own training windows).
* These are not exclusive; each is reported per class. M-e dates the change (the first milestone at which
  pc IoU falls below half its 5,000-step value, and whether AUROC moved with it).

**Alarm 2 — box conf_ratio.**
* **E1 CALIBRATION (ranking intact)** if a TRAIN-fitted monotone transform (T1b, T2, T2b or T3) puts EVAL
  conf_ratio in [0.5, 1.5] with AP unchanged (C5) ⇒ inference-time fix for the DETECTION readout. Preferred
  transform = the in-band one with the highest EVAL F1 at its gate (ties → fewest fitted parameters).
* **E2 TRANSFER FAILURE** if the TRAIN-fitted transforms land outside [0.5, 1.5] on EVAL ⇒ train/eval score
  shift; report the ORACLE eval transform beside it and classify as a recipe question.
* **E3 the focal-prior hypothesis** is SUPPORTED if the analytic transforms (T1a / T1c, zero fit) move
  conf_ratio by ≥ 10× toward 1; it is a cross-run claim (refcv6 vs refcv7, one seed each) and is reported as
  a mechanism, never as a measured lever effect.
* Agent head: the planner CONSUMES `sigmoid(presence_logit)` (feature + soft scale,
  `refc_agents.py:346/:360` per `slot_presence.py` docstring). A recalibration of the detection readout is
  inference-time; feeding a recalibrated score to the planner is a distribution shift ⇒ classified as a
  recipe change (fine-tune / branch run) regardless of the readout result.

## 6. Priority / banking order

M-c → B-c → M-a → M-b → B-b → M-d → M-e → M-f → B-d (the agent head, B-a..B-c). Each is banked to `raw/*.json`
as it finishes. Compute: Thor GPU under `flock /home/nvidia/refcv7_post/thor_gpu.lock`, `OMP_NUM_THREADS`
pinned, batch 8–16; no full logit dumps (histograms + a 1 % TRAIN cell subsample only).
