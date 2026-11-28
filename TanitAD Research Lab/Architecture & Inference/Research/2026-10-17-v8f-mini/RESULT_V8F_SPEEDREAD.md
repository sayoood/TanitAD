# RESULT_V8F_SPEEDREAD: is the per-window speed information O needs present in T's own input? Outcome **NOT IN INPUT**

Claim row `H-V8F-SPEEDREAD-1`. Pre-registration `Project Steering/PREREG_V8F_SPEEDREAD.md` with **Amendment 1** (`A1-2026-10-27`, landed `569a9d9b` before any SPEEDREAD number; instrument sha256 `fd5ce0be…`). Read 2026-10-28 01:09 by the pre-registered `decide()` (`stack/scripts/v8f_mini_eval_speedread.py`, stage `read`); all **19 control flags PASS**. Output `raw/2026-10-28-speedread/read/READ.json` (sha256 `46543c392ee412c76edab48ec95c993e11ba621cc43188ee8e25f7b966d4579f`). Every number below is MEASURED in this run (`READ.json`, `fit.json`, `extract.json`, the per-mode eval JSONs) unless it is marked INHERITED or ESTIMATED. The family rows of §4 come from the banked dumps (`families/`), read after §1 and changing nothing in it. READ.json `outcome`: **NOT IN INPUT**.

* **Tier.** READ.json `tier`: `T1-class open loop (X_s's own held S / T outputs; only the five speed cells O reads are replaced)`. READ.json `tier_gt`: `T0 (reads the eval GT): ceiling / control only, never a result`. The `gt` arm is the T0 ceiling and a control, never a result, and nothing here is closed loop. The unmodified harness stamps each dump `raw.tier` = `T0-open-loop-own-held-outputs (trunk-probe grid)`; the tier of the read is the instrument's stamp above, the tier of SPEEDCAL's and KINDX's ADE6 rows (PREREG §3).
* **Estimator.** READ.json `estimator`: `taniteval.ci.paired_episode_cluster_bootstrap, B=2000, seed 0; answers 'another draw of EPISODES' only (H-ESTIM-SEED-1)`. 141 episodes / 2,399 windows per seed, paired. The read-out is seed-independent: one fit (`readout_sha256` `a205609a…` in all 14 dumps) serves both seeds, so the two seeds are O's two training runs, the replicate `H-ESTIM-SEED-1` asks for at this stage. The planner is deterministic, so there is no inference-seed term. The bootstrap is blind to training variance; the second seed carries that question.
* **Arms.** X_s = `M2b-mixo_s<s>`, the SPEEDCAL / KINDX base. No model is trained in this experiment. Checkpoint sha256 s0 `c6a62876…`, s1 `68a0f335…`, each checked by `--expect-sha` at launch and recorded in every dump (`speedread.x_sha256`). The read-out file `fit/readout.pt` has sha256 `a205609a…` = `fit.json` `readout_sha256`.

## 1. The outcome, as the pre-registered table decides it

| | s0 | s1 |
|---|---|---|
| **PRIMARY** (`lin`) ADE6, read − own (m) | **−0.0097 [−0.0871, +0.0620]** | **+0.0022 [−0.0534, +0.0564]** |
| bar: ≤ −F = −0.1207 and upper end < 0 | not met | not met |
| ADE6 mean, own → read (m) | 2.1977 → 2.1880 | 2.1201 → 2.1223 |
| ALT (`mlp`) ADE6, alt − own | +0.2122 [+0.1101, +0.3158], separated-worse | +0.2040 [+0.1118, +0.2934], separated-worse |
| CAPTURE = PRIMARY / (gt − own) | 0.0135 | −0.0033 |
| `gt` − own (T0 ceiling) | −0.7164 [−0.8182, −0.6234]; mean 2.1977 → 1.4813 | −0.6676 [−0.7613, −0.5753]; mean 2.1201 → 1.4526 |
| `const` − own (control) | +5.3455 [+4.0805, +6.7295] | +5.7670 [+4.4988, +7.1424] |
| `perm` − own (control) | +5.3612 [+4.1072, +6.7345] | +5.7569 [+4.4971, +7.1255] |
| `perm` not separated-better | true | true |

Keys: `READ.json` `rows.<s>.read | alt | gt | const | perm | capture`; the means are the dumps' `raw.traj.ade6_m.mean` (4 decimals), so a delta can differ from the difference of two printed means by 0.0001. "Separated-better" is the CI's upper end below 0 (PREREG §5); "separated-worse" is its lower end above 0. F = 0.1207 m is the cross-seed spread of M2b (INHERITED, `RESULT_V8F_GRID.md`).

The pre-registered conditions (PREREG §5, first match wins), restated against the rows. `decide()`'s word in READ.json is the decision of record.

| outcome | condition | this run |
|---|---|---|
| INFO IN INPUT | PRIMARY_s ≤ −F and separated-better, at both seeds | no: PRIMARY is −0.0097 / +0.0022, neither ≤ −0.1207, neither upper end below 0 |
| PARTIAL | PRIMARY separated-better at both seeds | no: upper ends +0.0620 / +0.0564 |
| **NOT IN INPUT** | neither PRIMARY nor ALT separated-better at either seed | **yes**: PRIMARY is not, and ALT is separated-worse at both seeds |
| MIXED | anything else | not reached |

**Committed next, quoted from READ.json `next`:** *"perception stays the lever: PREREG_V8F_LORA on the A40 (D-V8F-SPEED-PERCEPTION, the PI's go); nothing on box 2 on this line"*.

⚠ **Scope of the negative** (PREREG §5). NOT IN INPUT means *"not recoverable by a ridge or a 2×256 MLP from the pooled 4-frame summary, on unseen clips"*. It does not cover the full 40-token attention view T itself has; T fails on that view too (SPEEDCAL, INHERITED), and the two negatives together are what route to perception. Disclosed design limits (PREREG §10): the PCA-512 summary is lossy; the 2×5 pooling assumes the `pool4x10` tokens are row-major (if they are not, it is a different fixed linear summary, not a leak); S's checkpoints are not read-out inputs; the fit uses the eval's age cycle, not the trainer's random ages.

## 2. Controls: each must read its known value before any row is read

| id | must read (PREREG §4) | s0 | s1 | measured detail |
|---|---|---|---|---|
| C1 | identity − own ADE6 max \|Δ\| = 0.0 and plans equal | true | true | ADE6 means, own / identity: s0 2.1977 / 2.1977 ; s1 2.1201 / 2.1201 |
| C2 | non-speed cells O reads = X_s's own, max \|Δ\| = 0.0, every hooked mode | true | true | `speedread.hook.nonspeed_max_abs_dev` = 0.0 in all 6 hooked modes at both seeds |
| C3 | the hook's tick slice = the model's own `_scene` pad mask for T (first batch) | true | true | `speedread.hook.tick_mirror_ok` true in all 6 hooked modes at both seeds; 38 batches each |
| C4 | episode ids equal across modes | true | true | flag only |
| C5 | own ADE6 = SPEEDCAL's banked `own` per window, max \|Δ\| = 0.0 | true | true | flag only (the per-window deviation is computed inside `read` and is not stored in READ.json) |
| C6 | gt ADE6 = SPEEDCAL's banked `gt` per window, ≤ 10⁻⁴ m | true | true | flag only (same) |
| C7 | what O reads in `read` = the identity run's read-out prediction, ≤ 10⁻⁵ (std units) | true | true | flag only (same) |
| C8 | CONST eval skill = 0.0 exactly | true | true | `cells.<s>.const.skill` = 0.0 in all five cells at both seeds |
| C9 | PERM held-out-TRAIN skill ≤ 0.02, and PERM not separated-better than own | true | true | one flag for both seeds; `fit.json` `perm_holdout_skill_max` 0.0069 (≤ 0.02); `perm_not_separated_better` true at both seeds |
| C10 | train parity (4,512 / 306,566 / `f6a1dee0…`) | true | true | one flag for both seeds; `fit.json` `train_parity_ok` true; `extract.json` parity: 4,512 clips, 306,566 windows, excluded sha256 `f6a1dee0…` |
| C11 | eval-141 ∩ fit clips = ∅ | true | true | one flag for both seeds; `fit.json` `eval_train_disjoint` true |

All 19 flags are true. Keys: `READ.json` `controls.<s>.C1..C8`, `controls.C9_perm_no_skill`, `controls.C10_train_parity`, `controls.C11_disjoint`.

## 3. The read-outs: PRIMARY = `lin`, selected before any eval batch

`fit.json` `primary` = `lin`, `alt` = `mlp`. The rule (PREREG §3): the one of LIN / MLP with the lower fold-0 RMSE (train-SD units, mean over cells), a tie to LIN. It was selected on held-out TRAIN fold 0 and written to `fit.json` when the fit stage finished (00:54:18, `logs/run.out`); the evals stage started after it, and the first eval dump is stamped 00:57. Fit on 306,566 windows of 4,512 TRAIN clips; the design is 534 dims (512 PCA of 14,336 raw scene dims, plus 22 ego dims); 5 folds by clip.

Fold-0 hold-out RMSE, train-SD units (`fit.json` `<read-out>.holdout_rmse_sd`):

| read-out | role | `lon_v_target_ms` | `v_a_ms` | `v_end_ms` | `v_lo_ms` | `v_hi_ms` | mean over cells (derived) | how it was fitted |
|---|---|---|---|---|---|---|---|---|
| `lin` | **PRIMARY** | 0.3572 | 0.1054 | 0.3651 | 0.2710 | 0.2631 | 0.2724 | ridge λ per cell: 1000 / 0.1 / 1000 / 0.1 / 0.1 |
| `mlp` | ALT | 0.3717 | 0.1560 | 0.3749 | 0.2803 | 0.2769 | 0.2920 | best epoch 10 |
| `lin_ego` | beside | 0.3707 | 0.1082 | 0.3753 | 0.2778 | 0.2716 | 0.2807 | ridge λ per cell: 10 / 0.1 / 10 / 100 / 100 |
| `mlp_ego` | beside | 0.3290 | 0.1036 | 0.3291 | 0.2313 | 0.2444 | 0.2475 | best epoch 8 |

The mean over cells is derived here (arithmetic mean of the five `fit.json` values); the rule's own comparison is `lin` 0.2724 against `mlp` 0.2920, so `lin` is PRIMARY, consistent with `fit.json` `primary`. PERM (the PRIMARY's class on window-permuted targets, fitted on folds 1–4) reads a held-out TRAIN skill of at most 0.0069 (`perm_holdout_skill_max`).

## 4. The four families (beside; they decide nothing)

READ.json carries ADE6 only, so the four families were computed afterwards on the banked dumps: read − own at each seed (the PRIMARY contrast) and gt − own as the T0 ceiling. Nothing here moves §1.

* **TACTICAL and STRATEGIC.** `v8f_paired_families.py` from the run tree (the `f3971165` copy, unmodified), run the way `code/devbox_ops/run_speedcal.sh` stage `pairfam` runs it, with the same substitute control. Outputs: `families/pairfam_<read|gt>_s<s>.json`.
* **LONGITUDINAL and LATERAL.** That tool does not cover them (its docstring places their paired blocks in `taniteval.four_families` and the readers). `gen/families_longlat.py` reuses the committed per-window row definitions and writes none: `comps` of `t1k_read.py` and `lat_rows` of `t1klc_read.py` at tip `e7ea20d1`, extracted by AST, paired by the same bootstrap. Outputs: `families/longlat_s<s>.json`.
* **Tier.** READ.json's T1-class open loop (own held outputs, no recorded future action; the tool stamps the same words). The gt − own table is the T0 ceiling and a control, never a result. The harness's own `raw.tier` stamp is quoted in the header.
* **Estimator.** Paired episode-cluster bootstrap, B = 2000, seed 0, 141 episodes: another draw of EPISODES only. There is one training run per seed, so a separated cell is a pointer, not a finding (`H-ESTIM-SEED-1`).
* **STRATEGIC is reported, not paused, for v8F.**
* **Markers.** B = separated-better, W = separated-worse, – = not separated, S+ / S− = separated on a row with no better direction (a bias or a ratio). Bold = separated. Error rows are lower-better; accuracy and F1 rows are higher-better.

### read − own (the PRIMARY contrast; point [95 % CI])

| family | row | s0 | s1 | n windows s0 / s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 (m) | −0.0097 [−0.0871, +0.0620] – | +0.0022 [−0.0534, +0.0564] – | 2,399 / 2,399 |
|  | ADE2 (m) | +0.0102 [−0.0019, +0.0195] – | **+0.0060 [+0.0001, +0.0119] W** | 2,399 / 2,399 |
| LONGITUDINAL | speed MAE (m/s) | +0.0035 [−0.0380, +0.0409] – | +0.0069 [−0.0223, +0.0347] – | 2,399 / 2,399 |
|  | along-track MAE (m) | −0.0188 [−0.1005, +0.0579] – | −0.0103 [−0.0668, +0.0447] – | 2,399 / 2,399 |
|  | accel MAE (m/s²) | −0.0215 [−0.0542, +0.0060] – | −0.0159 [−0.0328, +0.0008] – | 2,399 / 2,399 |
|  | DK-VIOL / DK-HW / DK-TG | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; the dump's reason: "no lead-agent track supplied" |
|  | anti-echo (hold-v0 baseline) | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; 2,399 windows available; the dump's reason: "no ego speed at t0 in the window dict" |
| LATERAL | heading MAE (deg) | −0.0999 [−0.3285, +0.0747] – | −0.0703 [−0.2577, +0.0700] – | 2,348 / 2,373 |
|  | yaw-rate MAE (deg/s) | −0.2969 [−0.5879, +0.0019] – | **−0.3849 [−0.6758, −0.1549] B** | 2,329 / 2,365 |
|  | curvature MAE (1/m) | −0.0045 [−0.0109, +0.0021] – | **−0.0088 [−0.0156, −0.0033] B** | 2,329 / 2,365 |
|  | cross-track MAE (m) | **+0.0193 [+0.0036, +0.0348] W** | +0.0108 [−0.0034, +0.0255] – | 2,399 / 2,399 |
|  | L1 lateral-accel MAE (m/s²) | **−0.0110 [−0.0169, −0.0055] B** | −0.0005 [−0.0039, +0.0027] – | 2,329 / 2,365 |
|  | L2 curvature MAE, GT ≥ 5.2 m/s (1/m) | −0.0032 [−0.0069, +0.0000] – | **−0.0042 [−0.0097, −0.0005] B** | 2,094 / 2,103 |
|  | L3 yaw-rate MAE, GT ≥ 5.2 m/s (deg/s) | **−0.2536 [−0.4139, −0.1247] B** | **−0.2335 [−0.4972, −0.0472] B** | 2,094 / 2,103 |
| TACTICAL | TAC-LAT (lateral-decision accuracy) | +0.0033 [−0.0046, +0.0125] – | **+0.0063 [+0.0008, +0.0125] B** | 2,399 / 2,399 |
|  | TAC-LON (longitudinal-decision accuracy) | −0.0188 [−0.0413, +0.0025] – | +0.0025 [−0.0200, +0.0271] – | 2,399 / 2,399 |
|  | 5-way manoeuvre accuracy | −0.0133 [−0.0359, +0.0079] – | +0.0083 [−0.0134, +0.0304] – | 2,399 / 2,399 |
|  | goal-point error (m) | +0.0234 [−0.2035, +0.2460] – | +0.0565 [−0.1156, +0.2216] – | 2,399 / 2,399 |
|  | goal longitudinal bias (m) | **+0.5985 [+0.3008, +0.9251] S+** | **+0.4706 [+0.2127, +0.7544] S+** | 2,399 / 2,399 |
|  | goal lateral bias (m) | **−0.1270 [−0.1994, −0.0518] S−** | −0.0249 [−0.0910, +0.0452] – | 2,399 / 2,399 |
|  | goal bearing MAE (deg) | −0.0531 [−0.4587, +0.2325] – | −0.1344 [−0.5023, +0.0840] – | 2,367 / 2,367 |
|  | goal range ratio | +0.0130 [−0.0030, +0.0278] – | +0.0195 [−0.0014, +0.0435] – | 2,367 / 2,367 |
|  | lateral head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,107 / 2,107 |
|  | longitudinal head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,240 / 2,240 |
|  | goal head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,365 / 2,365 |
| STRATEGIC | S-POS@30 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,322 / 2,322 |
|  | S-POS@50 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,319 / 2,319 |
|  | S-POS@80 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,294 / 2,294 |
|  | S heading@30 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,322 / 2,322 |
|  | S heading@50 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,319 / 2,319 |
|  | S heading@80 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,294 / 2,294 |

### gt − own (T0 ceiling, a control only; point [95 % CI])

| family | row | s0 | s1 | n windows s0 / s1 |
|---|---|---|---|---|
| PRIMARY | ADE6 (m) | **−0.7164 [−0.8182, −0.6234] B** | **−0.6676 [−0.7613, −0.5753] B** | 2,399 / 2,399 |
|  | ADE2 (m) | −0.0109 [−0.0235, +0.0011] – | **−0.0119 [−0.0196, −0.0041] B** | 2,399 / 2,399 |
| LONGITUDINAL | speed MAE (m/s) | **−0.3544 [−0.4059, −0.3064] B** | **−0.3424 [−0.3878, −0.2990] B** | 2,399 / 2,399 |
|  | along-track MAE (m) | **−0.7644 [−0.8685, −0.6692] B** | **−0.7122 [−0.8054, −0.6194] B** | 2,399 / 2,399 |
|  | accel MAE (m/s²) | **−0.0536 [−0.0970, −0.0140] B** | **−0.0303 [−0.0543, −0.0084] B** | 2,399 / 2,399 |
|  | DK-VIOL / DK-HW / DK-TG | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; the dump's reason: "no lead-agent track supplied" |
|  | anti-echo (hold-v0 baseline) | UNAVAILABLE | UNAVAILABLE | n = 0 / 0; 2,399 windows available; the dump's reason: "no ego speed at t0 in the window dict" |
| LATERAL | heading MAE (deg) | +0.0808 [−0.1576, +0.3139] – | −0.0839 [−0.3452, +0.0962] – | 2,349 / 2,373 |
|  | yaw-rate MAE (deg/s) | +0.0570 [−0.4042, +0.4881] – | −0.0760 [−0.3837, +0.1851] – | 2,329 / 2,364 |
|  | curvature MAE (1/m) | +0.0005 [−0.0122, +0.0122] – | −0.0032 [−0.0099, +0.0027] – | 2,329 / 2,364 |
|  | cross-track MAE (m) | **−0.0378 [−0.0601, −0.0183] B** | **−0.0430 [−0.0657, −0.0231] B** | 2,399 / 2,399 |
|  | L1 lateral-accel MAE (m/s²) | −0.0044 [−0.0120, +0.0027] – | +0.0022 [−0.0029, +0.0067] – | 2,329 / 2,364 |
|  | L2 curvature MAE, GT ≥ 5.2 m/s (1/m) | **−0.0158 [−0.0381, −0.0042] B** | **−0.0108 [−0.0220, −0.0026] B** | 2,097 / 2,104 |
|  | L3 yaw-rate MAE, GT ≥ 5.2 m/s (deg/s) | **−0.7495 [−1.6944, −0.2279] B** | **−0.5456 [−1.0354, −0.1744] B** | 2,097 / 2,104 |
| TACTICAL | TAC-LAT (lateral-decision accuracy) | −0.0071 [−0.0192, +0.0042] – | −0.0042 [−0.0142, +0.0054] – | 2,399 / 2,399 |
|  | TAC-LON (longitudinal-decision accuracy) | **+0.1159 [+0.0875, +0.1450] B** | **+0.1446 [+0.1188, +0.1706] B** | 2,399 / 2,399 |
|  | 5-way manoeuvre accuracy | **+0.0805 [+0.0513, +0.1108] B** | **+0.0988 [+0.0738, +0.1254] B** | 2,399 / 2,399 |
|  | goal-point error (m) | **−2.2239 [−2.5417, −1.9337] B** | **−2.0637 [−2.3514, −1.7739] B** | 2,399 / 2,399 |
|  | goal longitudinal bias (m) | **+0.6815 [+0.1831, +1.2044] S+** | +0.2837 [−0.1308, +0.7242] – | 2,399 / 2,399 |
|  | goal lateral bias (m) | **−0.1294 [−0.2446, −0.0261] S−** | −0.0302 [−0.1403, +0.0734] – | 2,399 / 2,399 |
|  | goal bearing MAE (deg) | −0.1541 [−0.6931, +0.3614] – | −0.2168 [−0.5473, +0.0024] – | 2,367 / 2,367 |
|  | goal range ratio | −0.0080 [−0.0402, +0.0214] – | −0.0233 [−0.0522, +0.0034] – | 2,367 / 2,367 |
|  | lateral head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,107 / 2,107 |
|  | longitudinal head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,240 / 2,240 |
|  | goal head macro-F1 | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,365 / 2,365 |
| STRATEGIC | S-POS@30 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,322 / 2,322 |
|  | S-POS@50 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,319 / 2,319 |
|  | S-POS@80 (m) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,294 / 2,294 |
|  | S heading@30 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,322 / 2,322 |
|  | S heading@50 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,319 / 2,319 |
|  | S heading@80 (deg) | +0.0000 [+0.0000, +0.0000] – | +0.0000 [+0.0000, +0.0000] – | 2,294 / 2,294 |

### What the rows say (beside; none of this moves the outcome)

Separated cells of the read, by seed:

* **s0:** 6 of 49 cells separated (read − own): `cross_mae_m` +0.0193 [+0.0036, +0.0348] W; `L1_lat_accel_mae_mps2` −0.0110 [−0.0169, −0.0055] B; `L3_yaw_rate_mae_fast_degps` −0.2536 [−0.4139, −0.1247] B; `tactical.goal_setting.goal_long_bias_m` +0.5985 [+0.3008, +0.9251] S+; `tactical.goal_setting.goal_lat_bias_m` −0.1270 [−0.1994, −0.0518] S−; `tactical.goal_setting.goal_bearing_bias_deg` −0.3766 [−0.7734, −0.0716] S−.
* **s1:** 8 of 49 cells separated (read − own): `ade2_m` +0.0060 [+0.0001, +0.0119] W; `yaw_rate_mae_degps` −0.3849 [−0.6758, −0.1549] B; `curvature_mae_1pm` −0.0088 [−0.0156, −0.0033] B; `L2_curvature_mae_fast_1pm` −0.0042 [−0.0097, −0.0005] B; `L3_yaw_rate_mae_fast_degps` −0.2335 [−0.4972, −0.0472] B; `tactical.plan_derived.lateral_decision.accuracy` +0.0063 [+0.0008, +0.0125] B; `tactical.plan_derived.lateral_decision.kappa` +0.0132 [+0.0012, +0.0265] B; `tactical.goal_setting.goal_long_bias_m` +0.4706 [+0.2127, +0.7544] S+.

1. **No LONGITUDINAL row moves.** Speed, along-track and acceleration MAE are not separated for the read at either seed.
2. **The hierarchy's own outputs are untouched, as they must be.** The substitution acts on the code O reads, after S and T have run, so the three head macro-F1 rows and all six S rows read exactly +0.0000 [+0.0000, +0.0000] at both seeds, for the read and for the gt arm.
3. **UNAVAILABLE rows are work items, not passes.** DK-VIOL / DK-HW / DK-TG and the anti-echo block have status UNAVAILABLE with n = 0 in all 14 dumps: the scoring call was given no lead-agent track and no t0 speed. The dumps' own wording is that every longitudinal number beside the anti-echo block is UNVERIFIED against a hold-v0 copy.
4. **Two rows are separated at both seeds.** L3 yaw-rate MAE (GT ≥ 5.2 m/s) is separated-better at both seeds (−0.2536 / −0.2335 deg/s), and the goal longitudinal bias moves by +0.5985 / +0.4706 m (S+, a bias has no better direction). The separated-worse rows are cross-track MAE at s0 (+0.0193 m) and ADE2 at s1 (+0.0060 m); no row is separated-worse at both seeds, and ADE6 does not move (§1).
5. **Read the separated cells as pointers.** They come from the cells counted above, each at one nominal level, with no replicate arm.

### Controls (each ran before its rows were read)

| run | tool `controls_ok` | identity: cells not zero / cells | reproduction of A and B: failed / checked, max abs diff | stored ADE6 vs recomputed, max dev (A, B) |
|---|---|---|---|---|
| read − own, s0 | true | 0 / 45 | 0 / 119 (A), 0 / 119 (B); 0.0 | 0.0, 0.0 |
| gt − own, s0 | true | 0 / 45 | 0 / 119 (A), 0 / 119 (B); 0.0 | 0.0, 0.0 |
| read − own, s1 | true | 0 / 45 | 0 / 119 (A), 0 / 119 (B); 0.0 | 0.0, 0.0 |
| gt − own, s1 | true | 0 / 45 | 0 / 119 (A), 0 / 119 (B); 0.0 | 0.0, 0.0 |
| substitute control, s0: `own` vs the banked X_s0 | true | 0 / 45 | 0 / 119 (A), 0 / 119 (B); 0.0 | max abs delta over all cells 8.0e-07 (bar: at most 1e-6) |
| substitute control, s1: `own` vs the banked X_s1 | true | 0 / 45 | 0 / 119 (A), 0 / 119 (B); 0.0 | max abs delta over all cells 7.0e-07 (bar: at most 1e-6) |

| LONGITUDINAL / LATERAL rows | LL-1 identity (rows not zero / rows) | LL-2 ADE6 row = READ.json (read, gt, alt, const, perm) | LL-3 per-arm values = the dump's own |
|---|---|---|---|
| s0 | true (0 / 12) | true | true (max abs dev 5.0e-05 over 42 checks; bar 1e-4) |
| s1 | true (0 / 12) | true | true (max abs dev 5.0e-05 over 42 checks; bar 1e-4) |

The tool's controls are: identity (the tool run on A against A reads exactly 0.0 on every cell), reproduction (each arm's point values equal its own eval JSON's) and alignment (same windows; the stored ADE6 equals ADE6 recomputed from `pred` and the grid GT). The substitute control is SPEEDCAL's: the SPEEDREAD `own` dump against the banked X_s dump, both read against the banked eval JSON. The few cells it flags as separated are float noise: their intervals sit at the 1e-7 level, below its bar. LL-1..LL-3 are defined in the header of `gen/families_longlat.py`.

## 5. Beside (does not move the outcome)

Per-cell eval RMSE to the GT, m/s, on the eval-141 windows where the GT cell is present (`cells.<s>.n_present`; keys `READ.json` `cells.<s>.T_own | lin | mlp | lin_ego | mlp_ego .rmse`). T_own is X_s's own speed cells as O reads them. Every column except T_own is identical at the two seeds, because the read-outs do not depend on the seed. `lin` is the PRIMARY (the `read` mode) and `mlp` the ALT.

| cell | n present | T_own s0 | T_own s1 | `lin` (the read) | `mlp` (ALT) | `lin_ego` | `mlp_ego` |
|---|---|---|---|---|---|---|---|
| `lon_v_target_ms` | 2,334 | 2.8940 | 2.7470 | 2.8934 | 3.0541 | 2.9663 | 2.5404 |
| `v_a_ms` | 2,351 | 1.9190 | 1.3020 | 0.7796 | 1.2178 | 0.7874 | 0.7517 |
| `v_end_ms` | 2,317 | 2.7953 | 2.6916 | 3.0009 | 3.1354 | 3.0811 | 2.5570 |
| `v_lo_ms` | 2,351 | 2.2571 | 2.1425 | 2.3032 | 2.4565 | 2.3346 | 1.8607 |
| `v_hi_ms` | 2,351 | 2.0927 | 1.8020 | 1.8744 | 2.0686 | 1.9151 | 1.7143 |

1. **The ego-only MLP is the most accurate read-out by every beside measure.** `mlp_ego` has the lowest fold-0 hold-out RMSE of the four read-outs in all five cells (§3; mean over cells 0.2475 against 0.2724 for `lin`, derived) and the lowest eval cell RMSE in all five cells, and its eval RMSE is below T_own's at both seeds in all five cells. **Its ADE6 substitution was NOT run**: the pre-registered modes are own, identity, gt, read, alt, const and perm, and `mlp_ego` appears only in the identity run's cell dump. No claim is made here about what it would do to ADE6.
2. **Vision plus ego is worse than ego alone for the MLP.** `mlp` is above `mlp_ego` in all five cells, on fold-0 hold-out and on eval (mean over cells 0.2920 against 0.2475, derived), and the ALT substitution costs +0.2122 / +0.2040 m of ADE6, separated-worse at both seeds. For the ridge the vision summary lowers the error a little: `lin` is below `lin_ego` in all five cells on both measures (hold-out mean 0.2724 against 0.2807, derived).
3. **A lower RMSE in one cell did not move ADE6.** Against T's own cells the `lin` read-out is lower in `v_a_ms` at both seeds (0.7796 against 1.9190 at s0 and 1.3020 at s1), lower in `v_hi_ms` at s0 only, higher in `v_end_ms` and `v_lo_ms` at both seeds, level in `lon_v_target_ms` at s0 (2.8934 against 2.8940) and higher at s1 (T_own 2.7470). ADE6 read − own is −0.0097 / +0.0022 (§1).
4. **O does respond to these five cells; the ridge read-out's cells do not carry that response.** ADE6 moves by −0.7164 / −0.6676 m under the `gt` ceiling, by +5.3455 / +5.7670 m under the constant substitute and by +5.3612 / +5.7569 m under the shuffled one (s0 / s1), and by −0.0097 / +0.0022 m under the read. CAPTURE is 0.0135 at s0 and −0.0033 at s1.

## 6. Next (Rule Zero)

* **The committed next is blocked, and here is exactly what blocks it.** READ.json `next` names perception: `PREREG_V8F_LORA` (DINOv3-L/16 + LoRA on the fly) on the A40. The A40 is not allocated (PI: box 2 and dev-box CPU only; INHERITED from the Master Mind's brief of 2026-10-28). What unblocks it, all PI decisions (PREREG §7): (1) the A40 provisioned (`D-V8F-SPEED-PERCEPTION`, `PI_DECISION_QUEUE.md`); (2) the frames on HF private or a disk provision (LORA §7, AR-4); (3) the go for ~200 GPU-h (2 seeds). LORA projects 494 h per seed on box 2 against its venue rule of ≤ 36 h per seed (INHERITED, LORA Amendment 3), and about 100+ GPU-h per seed on the A40 (ESTIMATED, LORA Amendment 3).
* **Nothing on box 2 on this line.** The box-2 streams already queued are unaffected (PREREG §7).
* The Master Mind pre-registers the next box-2/CPU-feasible arm separately.
* **How the Master Mind ranks it (beside; scoped to M2b-mixo, T1-class open loop, not DAG-OACC under T1-kin).** Even the GT speed cells
  (`gt` − own, T0 ceiling, §4) leave all-speed yaw-rate MAE and curvature MAE not separated at either seed; only the fast-window L2 and L3
  rows move. So the speed line reaches L2 of DAG-OACC's open crawl set and not yaw-rate, curvature or M1. The lever aimed at those rows is
  DAG-ORC, running on box 2 since 2026-10-28 01:16. An ego-only speed-cell arm (the `mlp_ego` read-out of §5, not yet run as a substitution)
  is ranked below it.

## 7. Provenance

* **Data parity** (`train/extract.json`, `fit/fit.json`). 4,512 clips, 306,566 windows, excluded sha256 `f6a1dee0…`, `parity_ok` true; grid `pool4x10`; `d_ego` 22; window-list sha256 `61a6126e…`; join sha256 `978f41bf…` (equal to the trainer's `--join-train`, PREREG §2); tracks sha256 `1db431f1…`. Windows with the GT cell present, per cell (cell order as in §3): 303,070 / 305,563 / 300,731 / 305,563 / 305,563. Eval-141 ∩ fit clips is empty (`eval_train_disjoint` true, C11).
* **Three stops, each before any SPEEDREAD number.** None wrote or displayed a result.

| stop | when | what stopped | opened | class and fix | log |
|---|---|---|---|---|---|
| 1 | 2026-10-27 23:09 | `extract`: `ModuleNotFoundError: No module named 'taniteval'` (`v8f_mini_data.py:81`), after the parity line | the first item of the first batch, nothing decoded | run mechanics: `taniteval/` was missing from the exported tree. The wrapper `run/run_with_taniteval.py` puts the tree's `taniteval/` on `sys.path` under `-I`; the instrument is unchanged | `stops/stop1/extract_no_taniteval.log`, `run_stop1.out` |
| 2 | 2026-10-27 23:10 | `extract`: `ValueError: could not broadcast input array from shape (128,22) into shape (128,23)` (instrument line 611) | one batch of 128 TRAIN windows decoded; the partly filled scene memmap was moved aside unread (`F:/TanitAD-scratch/speedread/stop2/train_partial_unread/`, not banked) | the ego vector is 22 dims, not the 23 PREREG §3 counted: **Amendment 1**, landed `569a9d9b`, instrument `d5545751…` → `fd5ce0be…`. The pre-Amendment instrument is banked beside the log | `stops/stop2/extract_ego23.log`, `run_stop2.out`, `v8f_mini_eval_speedread_d5545751.py` |
| 3 | 2026-10-28 00:54 | `evals_s0`: `FileNotFoundError` on `…/raw/eval_grid_windows.json` (`v8f_mini_eval.py:57`, `ctx_from_args`), 4 s after the fit stage finished | no eval batch; `eval_s0_partial/` was empty | run mechanics: the log names `eval_grid_windows.json`. The Master Mind exported the eval inputs from the same commit `f3971165` into the run tree; the 12 files now there are byte-identical to that commit's blobs (sha256 compared here). The run restarted at 00:56:18 with extract and fit skipped as done | `stops/stop3/evals_s0_no_grid.log`, `run_stop3.out` |

* **Instrument.** `stack/scripts/v8f_mini_eval_speedread.py`, sha256 `fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe`, pinned in `run/run_speedread.sh`. The run tree's file, the blob at `569a9d9b` and the blob at `5fee0064` all hash to it (compared here). The run tree is a `git archive` of `f3971165` (`stack/`, `taniteval/`, SPEEDCAL's banked eval) with the Amendment 1 instrument. Self-test 27 passed and 19 / 19 mutants killed (INHERITED, PREREG Amendment 1 §3, `raw/2026-10-27-speedread-a1/`).
* **Run** (dev-box CPU, `CUDA_VISIBLE_DEVICES=-1`; `logs/run.out`). extract 23:21:49–00:17:14; fit 00:17:14–00:54:18; evals_s0 00:56:19–01:03:22; evals_s1 01:03:22–01:09:19; read 01:09:19–01:09:25.
* **The pull** (`pull/pull.out`). 4,572 clips md5-verified against the manifest (`ZZSPEEDREAD-PULL-DONE` 23:08:43; `pull/verified.txt`); the extract used the 4,512 its parity rule keeps. It read box 2's USB HDD (per the Master Mind) and ran without the temperature guard (`run/pull_speedread.sh` has no temperature check).
* **Banked** in `raw/2026-10-28-speedread/`: `read/`, `fit/fit.json`, `train/extract.json`, the 14 per-mode eval JSONs under `eval_s0/` and `eval_s1/`, `families/` (the §4 outputs), `logs/`, `pull/`, `stops/`, `run/`, `landing/`.
* **Not in the repo, single copy on F: (dev-box scratch).** `fit/readout.pt` (118,851,106 B, sha256 `a205609a…`); `train/train_arrays.npz` (94,730,100 B) and `train/scene_raw.npy` (8,789,860,480 B); the per-mode `eval_*.npz` and `speedread_cells_*.npz` dumps; the 94.27 GB train-cache pull (PREREG §2, for the 4,512 clips); `stop2/train_partial_unread/`.
