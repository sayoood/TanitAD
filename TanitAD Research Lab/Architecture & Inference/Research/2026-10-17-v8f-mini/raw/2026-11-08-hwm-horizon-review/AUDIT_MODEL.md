# AUDIT_MODEL: v8F-mini against the horizon requirements (REQ_HORIZONS.md, R-OP, R-TAC, R-STR, R-TRAJ, R-ABS, R-VIS, R-DATA)

Auditor: model-design audit agent, 2026-11-08. No code, test, register or launch_gate.py edit; nothing staged, committed, pushed or launched; no GPU.

Layers named in every claim (the "state the layer" rule): **TIP** = mirror commit `26fb0bc1b86bd290c9913ecee0b3676ba7684f05`, read with `git show <tip>:<path>` (I extracted copies of the six key files to my scratch folder and cite the tip line numbers). **UNLANDED** = `theory/mrate/tree/.../v8f_mini_train.mrate_lp4.py` and `theory/levers2/diffs/{1_strides,2_divrs,3_rel}.diff`. **LABELS** = `s2_labels_v7.2_train.jsonl.gz` (4,572 per-clip records) and the v9 per-frame release (`build_v9_labels.py`).

Evidence classes: MEASURED (ours, artifact path given), PUBLISHED, INHERITED (another doc, not re-verified here), ESTIMATED, HYPOTHESIS.

Short paths used below:
* `hjepa.py` = `stack/tanitad/models/hjepa.py`; `v8f_mini.py` = `stack/tanitad/models/v8f_mini.py`; `train.py` = `stack/scripts/v8f_mini_train.py`; `data.py` = `stack/tanitad/data/v8f_mini_data.py`; `ff.py` = `taniteval/taniteval/four_families.py`; `cons.py` = `taniteval/taniteval/v8f_consistency.py`.

---

## 0. Findings in one page (the ones that change the plan)

| # | finding | class |
|---|---|---|
| F1 | The world model (WM) loss is **one-step teacher-forced** at every level (`hjepa.py:671-695`). A level's "reach" is therefore its step: 0.2 / 0.4 / 0.8 s. The 0.6 / 1.2 / 2.4 s figures are the **predictor context span** (context 3, `hjepa.py:452`), not a predicted horizon. No multi-step (free-run) term exists in the loss; `rollout()` (`hjepa.py:732`) is only used by eval and planning code. Nothing in the WM reaches 2 s, 6 s or beyond as a **trained** prediction. | MEASURED (code read) |
| F2 | The WM window is **past-only**: `idx = arange(t - hmax + 1, t + 1)` (`data.py:580`), `hmax` = 16 whenever `--wm` is on (`train.py:316`). Window span = 15 steps = 3.0 s. Transitions per window: L1 15, L2 7, L3 3. | MEASURED (arithmetic on `hjepa.py:637-668`) |
| F3 | Strides are **hardcoded** at the tip: `wm_strides = (2, 2)` is a dataclass default (`v8f_mini.py:195`), applied at `v8f_mini.py:659-660`; the trainer has **no** `--wm-strides` flag (grep of `train.py` for `wm-strides` and `wm_k`: 0 hits; the same grep on the UNLANDED trainer: 15 hits for `wm_k`; the flag exists only in `levers2/diffs/1_strides.diff`). Context (3), predictor width/depth, window `w = 1`, `act_sigreg` are also fixed in `LevelConfig` defaults (`hjepa.py:423-458`) and not exposed. | MEASURED |
| F4 | The planner reads **no** WM output: the only use of `self.wm` in `forward`/`loss` is the aux loss term `T["wm"]` (`v8f_mini.py:1045-1047`, added at `:1056-1057` with weight `w_wm = 0.1`). So a WM-only retrain cannot change driving. | MEASURED |
| F5 | O's output is a 6 s plan (60 x 0.1 s, `N60`, `v8f_mini.py:42`; 30 knots at 0.2 s, `K30`) as P10 prior plus a 12-cosine residual (`:710-713`, `:845-847`). The loss weights all 30 knots equally (`traj_terms`, `:418`). There is no operative/tactical split in loss, head or metric. | MEASURED |
| F6 | T's labels come from the **v9 per-frame release**, not from the v7.2 release. v8F-mini never reads `a_str`, `g_str`, `a_tac`, `g_tac`, `bands` or `horizon` (grep of `v8f_mini*.py`, `data.py`, `train.py`, and a second path: all `stack/tanitad/models/v8f*`, `stack/tanitad/data/v8f*`, `ra1_pooled.py`: 0 hits; the only hits for `g_str` in `stack/tanitad` are old refcv-line names in `declared_vs_built.py:1432-1433`). | MEASURED |
| F7 | T uses the **v7 tactical GOAL vocabulary** (22 tokens, identical order to `vocab_v7.py:95-117`, restated in `data.py:50-54` and `build_v9_labels.py:468`). Its **action** classes are **not** the v7 action vocabulary: T has 5 lat and 7 lon classes (`data.py:55-56`, `v8f_mini.py:44`), and `STOP`, `DECELERATE`, `KEEP` are not in `vocab_v7.py:320-331`. The goal, lat, lon and constraint labels all reach the loss (`v8f_mini.py:996-1008`, total at `:1048-1049`). | MEASURED |
| F8 | **Band mismatch in the labels.** v9 tactical labels are built over `[NOW+2, NOW+8]` (`build_v9_labels.py` LITERALS `band_s: (2.0, 8.0)`, `h_abs_min_s: 8.0`). The v7.2 emitter bands are operative (0,2), tactical (2,6), strategic (8,30) (`s2_geom_emit_v7.py:50-52`; the bands field of the record reads `[0,2] [2,6] [8,30]`, MEASURED on record 1 of the train release). The PI's R-TAC is 2-6 s and R-STR is 6-30 s. Two gaps: T's labels reach 2 s past the tactical limit; and 6-8 s is owned by no layer in the v7.2 release. The v9 builder fell back from 6 s to 8 s because truncating to 6 s changed 5.3 % (lat-a), 7.3 % (lat-b), 15.4 % (lon) of labels against a bar of 5 % (comment on `h_abs_min_s`, evidence `raw/s3_validation_eval139.json`, INHERITED). | MEASURED / INHERITED |
| F9 | **Strategic GT exists in v7.2 only per clip, and is thin.** `g_str` tokens in the train release (n = 4,572): `FOLLOW_ROUTE` 2,948, `TURN_RIGHT_FOLLOW_ROUTE` 596, `TURN_LEFT_FOLLOW_ROUTE` 574, `STOP_AT_FOLLOW_ROUTE` 454; the other 4 of 8 tokens never occur. `a_str`: 5 of 7 tokens occur. Among the 1,624 non-`FOLLOW_ROUTE` records, `by_time_s` p5/p25/p50/p75/p95 = 8.0 / 8.0 / 12.4 / 19.8 / 27.0 s and `within_m` = 8 / 37 / 73 / 140 / 256 m. MEASURED from `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/s2_labels_v7.2_train.jsonl.gz`. | MEASURED |
| F10 | Context labels: `strata` in each v7.2 record carries `country`, `daynight_clock`, `road_class`, `strata_cell` (MEASURED, record 1). The published metadata parquet has `country`, `month`, `hour_of_day`, `platform_class`, `radar_config` for 306,152 clips and **no weather column** (`theory/strat0/CENSUS_STRAT0.json`, MEASURED). The hour is a clock label: only about 8 % of clips are dark against 46 % "night" (`PLAN_REFCV8.md` K16, MEASURED D3, INHERITED here), so illumination must be derived from the image, not from the hour. | MEASURED / INHERITED |
| F11 | The cache holds 101 frames (20 s, user-stated, INHERITED), so a 30 s **image** window is impossible. But the ego log is long: `horizon.available_s` p5/p25/p50/p75/p95 = 22.9 / 35.0 / 35.0 / 35.0 / 35.0 s (35 is the builder's cap) and `recording_span_s` median 139.7 s (MEASURED, same release). **Labels to 30 s are feasible for about 95 % of clips; WM reach on images beyond about 20 s is not.** | MEASURED |
| F12 | Trainer gate: for the mini trainer, G-DVB is `FLAG_CHECKS` plus `declared_vs_built` (`train.py:68`, `:941-952`) and `live_check` (`:1054`). `launch_gate.py` at the tip has **no** profile for the mini trainer (probes: grep for `mini`, `wm_only`, `wm-only` in `launch_gate.py`: only comments and the v8f-mini ENV-id pin lists; its `v8f` / `v8f_steer` profiles at `:979-1044` are the refcv8-derived P10 rung). Whether the MRATE card's "launch-gate PASS token" is issued by `launch_gate.py` for mini runs is therefore **unresolved by this audit** (see section 5). | MEASURED / OPEN |

---

## 1. World model per level

### 1.1 What is there (flag to loss, with file:line)

| item | where set | hardcoded or flag | value today |
|---|---|---|---|
| number of levels | `v8f_mini.py:659-660` (three `LevelConfig`) | hardcoded | 3 |
| level 1 step | grid: z1 is the pooled adapter token of every cache frame (`WMAux.z1`, `v8f_mini.py:666`), cache step `DT = 0.2` (`:41`) | hardcoded | 0.2 s |
| level 2 / 3 stride | `cfg.wm_strides` (`v8f_mini.py:195`) -> `LevelConfig(stride=...)` (`:660`) -> `_apply_encoder` windows (`hjepa.py:601-614`, `_windows` `:258`) and `MacroActionEncoder(stride)` (`hjepa.py:285-326`) | config default; **no CLI flag** at TIP; flag `--wm-strides S2 S3` exists only in UNLANDED `1_strides.diff` | (2, 2) -> steps 0.2 / 0.4 / 0.8 s |
| window length (frames) | `hmax` (`v8f_mini.py:184`) <- `--hmax`, auto 16 if `--wm`, else 10 (`train.py:270`, `:316`); dataset frames `t-hmax+1..t` (`data.py:580`) | flag (G-DVB `cfg:hmax`, `train.py:75`) | 16 (MRATE arms: 28) |
| states / transitions per level | `HJEPAStack.encode` (`hjepa.py:637-668`): `T_l = (T_{l-1} - w)//s + 1`; error if `T_l < 2` (`:657`) | derived | 16/8/4 states, **15/7/3 transitions** (28 frames: 28/14/7, 27/13/6) |
| predictor context | `LevelConfig.context = 3` (`hjepa.py:452`) -> `CausalPredictor.pos` shape (`hjepa.py:402`) | hardcoded default; changing it changes parameter shapes | 3 |
| training horizon of one prediction | `_pred_loss` (`hjepa.py:671-695`): chunks of `c` positions, position `i` predicts `z_{i+1}` | hardcoded | **1 step** = 0.2 / 0.4 / 0.8 s |
| loss weights per level | `LevelConfig.beta/gamma` set in `WMAux` (`v8f_mini.py:655-658`: beta `wm_beta` 0.08, gamma 0 or `wm_gamma` 100); total weight `w_wm` | flags `--w-wm`; wm_beta/gamma are `cfg` fields, trainer flags for them: `wm_dim`, `wm_beta`, `wm_gamma` appear in `INIT_CKPT_MATCH` (`train.py:657`) but no argparse flag found | 0.1 / 0.08 / 100 |
| tie of WM output to planner | none (F4) | n/a | WM is an aux loss |
| minimum-length guard | `v8f_mini.py:265`: `hmax >= 2*s1*s2 + 4` | hardcoded formula | 12 <= 16 |

Reach summary at the tip (MEASURED by arithmetic): L1 predicts 0.2 s ahead, L2 0.4 s, L3 0.8 s, each from a context of up to 3 states (so 0.6 / 1.2 / 2.4 s of state history inside one chunk). At eval, WMACT/IMAG read free-run rollouts at L1 k 1/5/10 (up to 2.0 s) and L2, L3 k 1/3 (`PREREG_V8F_MRATE.md` E1); those are **extrapolations beyond the trained one-step objective**, not trained reach.

Against the requirements: R-OP (prediction reaches 2 s): not met (0.2 s trained). R-TAC (6 s): not met (0.4 s). R-STR (6-30 s): not met (0.8 s). R-ABS: L3 at 0.8 s is exactly the "0.8 s as long-term abstract level" the PI rejected.

### 1.2 Definitions for the candidates

Base grid g = 0.2 s. Level l has cumulative stride `S_l` (product of strides), step `tau_l = 0.2 * S_l`, context `c_l` states, rollout depth `k_l` steps (new: a free-run or k-step loss; today k = 1).

* Reach `R_l = k_l * tau_l`.
* Frames needed by one anchor (context states, then k targets): `F_l = (c_l - 1 + k_l) * S_l + 1`.
* The joint window is `F = max_l F_l`.
* Transitions per window `floor((F-1)/S_l)`.

### 1.3 Candidates (the 0.2 s grid; the cache has 101 frames)

| cand | strides (S2,S3) | steps tau1/tau2/tau3 | k1/k2/k3 | c1/c2/c3 | reach R1/R2/R3 | F1/F2/F3 frames | joint F (span) | transitions per window L1/L2/L3 | meets |
|---|---|---|---|---|---|---|---|---|---|
| now (M2b) | (2,2) | 0.2/0.4/0.8 | 1/1/1 | 3/3/3 | 0.2/0.4/0.8 s | 5/9/17 | 16 (3.0 s) | 15/7/3 | none |
| **A (band-matched, reuses the LFS (3,3) flag)** | (3,3) | 0.2/0.6/1.8 | 10/10/10 | 3/3/1 | 2.0/6.0/18.0 s | 13/37/91 | **91 (18.0 s)** | 90/30/10 | R-OP yes, R-TAC yes, R-STR partial (to 18 s) |
| B (round steps) | (5,3) | 0.2/1.0/3.0 | 10/6/6 | 3/3/1 | 2.0/6.0/18.0 s | 13/41/91 | 91 (18.0 s) | 90/18/6 | same as A; L2 step 1.0 s |
| C (strategic to 20 s, the cache limit) | (3,5) | 0.2/0.6/3.0 | 10/10/6 | 3/3/1 | 2.0/6.0/18.0 s | 13/37/91 | 91 | 90/30/6 | same reach, coarser L3 |
| D (30 s) | (3,5) | 0.2/0.6/3.0 | 10/10/10 | 3/3/1 | 2.0/6.0/30.0 s | 13/37/151 | **151 (30.0 s)** | 150/50/10 | needs 151 contiguous frames; **cache has 101: blocked** |

Notes (all arithmetic: ESTIMATED to the extent the choice of k and c is a design choice, MEASURED to the extent the formulas are read from `hjepa.py:258-263`, `:637-668`):
* "2 s" at L1 needs k = 10 steps at 0.2 s. "6 s" at L2 needs k = 10 at 0.6 s, or k = 6 at 1.0 s. "6-30 s" at L3 needs k x tau3 up to 30 s: with tau3 = 3.0 s that is k = 10 and a 151-frame window.
* c = 1 at L3 is the planner's own context (`rollout(context=1)`, `hjepa.py:732`, planning uses c = 1 per the docstring at `hjepa.py:384-386`). It saves 2 x S3 frames. A pure k-step training with c = 1 trains exactly the planner's use.
* A cheaper way to touch the 6-30 s band without a 30 s image window: train L3 at tau3 = 1.8-3.0 s, k = 6-10 to 18 s, and carry 18-30 s only in the **labels** (F11), through the strategic head, not through WM reach.
* The existing guard at `v8f_mini.py:265` (`hmax >= 2*s1*s2 + 4`) does not describe any of this; for A it would demand 22 and say nothing about the 91 frames A needs. UNLANDED `1_strides.diff` replaces the guard by `LFS_MIN_TRANSITIONS = 2` per level (`bad_wm_strides`), which also does not encode `F_l`. A new guard `F >= max_l F_l` is needed.

### 1.4 Window availability and what 30 s needs

* Training windows are `t in [start, T-31]` because 30 future knots are required (`data.py:607-620`, `HORIZON = 30` at `data.py:48`). With T = 101 that is at most 71 anchors per clip, and a past-only WM window of F = 91 frames requires `t >= 90`: **no window satisfies both** (`t <= 70` and `t >= 90`). So candidates A-C cannot reuse `train_windows`. They need a **WM window sampler of their own** (any `t >= F-1`, no label requirement, because WM-only needs none): 11 anchors per clip for F = 91 (`t = 90..100`). That is a data change (medium), not a flag.
* A 30 s target needs 151 contiguous frames (30.2 s of span). The cache clip has 101. Options to unblock: (a) re-encode longer spans of the source recordings (the ego log spans 140 s median, F11, but whether the **video** extends beyond 20 s per cached clip is not established by this audit: HYPOTHESIS that it does not, since PhysicalAI-AV clips are 20 s, `REFCV5_DESIGN_PLAN.md:140`, INHERITED); (b) restrict L3 to 18-20 s on images and take 20-30 s from labels only.

### 1.5 Memory and time on an 8 GB GPU (ESTIMATED unless marked)

Per-frame cost in the joint model (adapter trained): `feats.float()` 40 x 1024 x 4 B = 163,840 B, the LayerNorm output (same size, kept for backward) 163,840 B, linear output and positional add 2 x 40 x 256 x 4 B = 81,920 B; about 0.41 MB per frame per window (ESTIMATED from `v8f_mini.py:_encode` and `d = 256`, `d_feat = 1024`). At the default batch 64:

| F (frames) | adapter activations per step | note |
|---|---|---|
| 16 (now) | about 0.42 GB | M2b |
| 28 (MRATE) | about 0.73 GB | MRATE arms |
| 37 | about 0.97 GB | L2 6 s alone |
| 91 (A, B, C) | about 2.4 GB | fits an 8 GB card with the readers (they see 4 frames) but little headroom with AMP off |
| 151 (D) | about 4.0 GB | does not fit with the joint model at batch 64; batch 32 or chunked frames needed |

MEASURED anchors: WM parameters 2,007,874 (`theory/mrate/logs/time_step_4thr.json`); the WM-only CPU step at hmax 28, batch 64 is 0.71 s median, of which the adapter forward is 0.26 s (same file; contended dev box, so an upper bound); full M2b on box 2 is 0.82 s per step at hmax 16 (`IMPL_CHOICES_MRATE.md` section 3, INHERITED). ESTIMATED consequence if the step is loader/adapter bound and linear in F: F = 91 gives about 4.7 s per full step on box 2 (that is 25 h for the 19,164-step budget) and F = 37 about 1.9 s (10 h). **The cheap fix for time and memory is WM-only mode with the adapter frozen**: the WM consumes only `adapt.mean(2)` (`v8f_mini.py:666-669`), a [B, F, D] tensor, so for a frozen adapter it can be computed once per frame, under no-grad and in chunks, or cached per clip (one 256-wide vector per frame instead of 40 x 1024 fp8). That removes the F-fold activation memory and makes F = 91 or 151 a small cost. This is a new data path (medium), and it is the reason the WM ladder should be trained as a **WM-only** run.

The WM itself stays small: dim 128, predictor width 128, depth 2 (`v8f_mini.py:650-653`); the macro-action MLP at S3 = 9 or 15 concatenates `S3 x 32` inputs (`hjepa.py:296-310`), so its first layer grows from 64 x 32 to 288 or 480 x 32; negligible.

### 1.6 What each candidate costs to train

* The WM ladder change (strides, k, c, windows) touches **only `wm.*` parameters** (F4). The reader/adapter/planner weights are not used by the WM loss except through `adapt` (the WM loss does reach the adapter in the joint run: `v8f_mini.py:1046` passes `out["adapt"]` with grad). A WM-only run (`--wm-only` in the UNLANDED trainer, adapter/readers/planner frozen and bit-equal) retrains the WM only and leaves driving unchanged (planner-identity control, MRATE card control 1).
* If the PI wants the new WM to also shape the adapter (joint), that is a full retrain; I do not recommend it before the WM-only result is read (the MRATE/LFS cards already take the WM-only route).

---

## 2. Readers: what S, T and O read, what supervises them, what they output

All three readers share one scene adapter (`v8f_mini.py` around line 690: LayerNorm + Linear on the frozen DINOv3-L pooled 4 x 10 tokens), see `hist = 4` frames (0.8 s of history, `v8f_mini.py:183`, `_scene` at `:790`) and run **sample-and-hold**: S ticks about 1 Hz, T about 2 Hz, O every step; the upper output is held and re-expressed with measured odometry (module docstring `v8f_mini.py:1-20`; ages `max_age_t = 2`, `max_age_s = 6` cache steps, `:207-208`).

| reader | inputs (code) | output | label source and loss (flag to loss) | time horizon of the output |
|---|---|---|---|---|
| **S** `run_s` (`v8f_mini.py:799-808`) | ego features `p10f` via `e_ego`, nav `e_nav` (5 values: FOLLOW/TURN_L/TURN_R one-hot, distance/100, valid), speed limit `e_lim`, scene tokens (unless `navonly`) | `head_s` gives 9 numbers = 3 route checkpoints (x, y, heading) at arcs `CKPT_ARCS = (30, 50, 80)` m (`:43`, `:706`, `:806-808`) | v9 `rc_A30/A50/A80` (x, y, psi, valid) joined at `data.py:134-146`; L1 loss `T["ckpt"]` (`v8f_mini.py:988-994`), weight `w_ckpt`, total at `:1048`; teacher forcing: with probability `p_gt` (1.0 to 0.2) T reads the GT checkpoints instead (`:880-890`) | arc-indexed (30 / 50 / 80 m), no time axis; see section 3 |
| **T** `run_t` (`v8f_mini.py:810-823`) | as S plus S's held checkpoints (`_ck_cond`, `:768`) | 22 goal logits, 5 lat logits, 7 lon logits, 26 constraint regressions (`h_goal/h_lat/h_lon/h_cons`, `:707-708`); `CODE_DIM = 60` (`:46`) | v9 `goal_y/goal_w` (22-token mask), `lat_cls/lon_cls` + allowed masks, `cons` (26 cells, NaN = undefined), joined at `data.py:211-222`; losses `T["goal"]` BCE, `T["lat"]`, `T["lon"]` partial CE, `T["cons"]` Huber (`v8f_mini.py:996-1008`); total `w_tac * (goal + 0.5 (lat + lon) + w_cons * cons)` (`:1048-1049`) | labels over `[NOW+2, NOW+8]` (F8); consistency rules use `horizon_s = 6.0`, `band_start_s = 2.0` (`cons.py:50`); constraint times (`turn_t_*`, `lon_t_reach_s`, ...) are seconds inside that band |
| **O** `run_o` (`v8f_mini.py:825-848`) | ego token, nav, limit, S's checkpoints (held, `_ck_cond`), **T's code** (`e_code`, GT mixed with predicted by `p_gt` / `code_mix`), ages `e_age`, optional level/lead tokens, scene tokens | `head_o` gives `[B, 2, nb=12]` cosine coefficients x `res_scale_m`, times the ramped basis (`:711-713`), plus the P10 prior upsampled to 60 points (`upsample_prior`, `:355`); `traj30` = the 0.2 s knots | GT future waypoints `gt` [30, 2] (`data.py:387-397`, `HORIZON = 30`); `T["ade"], T["fde"] = traj_terms(...)` (`:1011`, def `:418`), total `w_traj * (ade / 3 + 0.5 fde / 6)` (`:1049`); optional `acc`, `launch_along`, `level`, `lead` terms | 0 to 6.0 s, 30 knots at 0.2 s, 60 points at 0.1 s (`K30`, `N60`, `:42`) |

Does T use the v7 tactical vocabulary and do its labels reach the loss? **Partly yes, and the part that is not the v7 vocabulary matters.**
* Goals: yes. `GOAL22` in `data.py:50-54` is the same 22 tokens in the same order as `TACTICAL_GOAL_TOKENS_V7` (`vocab_v7.py:95-117`); the v9 builder restates them (`build_v9_labels.py:463-476`) rather than importing them. Traffic-light goals carry their colour.
* Actions: no. T's lat classes are 5 (`LANE_KEEP, TURN_L, TURN_R, LANE_CHANGE_L, LANE_CHANGE_R`, `data.py:55`) and lon classes 7 (`HOLD, CREEP, STOP, FOLLOW, DECELERATE, ACCELERATE, KEEP`, `data.py:56`). The v7 action tuples (`vocab_v7.py:320-331`) are 8 lat (with `ABORT_LC`, `NUDGE_L/R`) and 8 lon (`CRUISE`, `YIELD_MERGE`, `BRAKE_TO`, `ADAPT_SPEED_FOR_CURVE`, ...). So "T includes our tactical vocab" is true for goals and false for actions. The v9 builder collapses variants (`LAT9_TO_7`, `LON9_TO_7`, `build_v9_labels.py:1253-1254`); I did not trace which collapse maps into the 5/7 classes (open, small).
* Reach to the loss: yes for goal, lat, lon and constraints (`T["goal"]`, `T["lat"]`, `T["lon"]`, `T["cons"]` are in the total, `:1048-1049`). The trainer's `live_check` requires every declared term finite and every parameter group (`S`, `T`, `O`, `WM` groups at `:1074-1077`) to receive a non-zero gradient, against a frozen-P10 control that must read exactly 0.0 (`train.py:1054`, module docstring). I did not run it (no GPU).
* Band: T's labels run to NOW+8 s (F8), 2 s beyond R-TAC.

### 2.1 Composing O's 6 s trajectory as operative (knots 1-10) + tactical (knots 11-30)

Today the plan is one object: `traj30 = prior + residual` with one 12-mode basis over 6 s. The 0.2 s knots 1-10 are tau = 0.2-2.0 s; knots 11-30 are tau = 2.2-6.0 s; `cons.py:51` already names this boundary (`_BAND_KNOT = 10`). Two designs:

* **(i) Split by loss and metric only (small).** Keep `head_o` and the basis. Replace the single ADE/FDE by `ade_op` (knots 1-10), `ade_tac` (knots 11-30), `fde_2s` (knot 10), `fde_6s` (knot 30), plus a continuity term at knot 10 (GT-referenced position error and the heading/speed from finite differences of knots 9-11). Flags `--w-op`, `--w-tac-traj`, `--w-cont`, defaults chosen so the default argv reproduces today's total bit for bit (the pattern used by `w_acc` and `launch_intent_w`: "the term is NOT in the total and NOT computed", `v8f_mini.py:215-224`). O-only retrain.
* **(ii) Split by head (medium).** `head_op`: residual over knots 1-10 with its own basis (about 5-6 modes over 2 s), reading O's reader as today. `head_tac`: residual over knots 11-30, conditioned on T's code **and** on `head_op`'s end state (position, heading, speed at knot 10), zero-initialised and ramped from the knot-10 value so the concatenation is continuous. The output stays one trajectory of 30 knots (R-TRAJ). The HACC post-map table is already per knot ([30, 2], `hacc_check_weights`) and needs no change; `launch_intent_w` already targets knots 1-12 (`LAUNCH_KNOT_*`, `:445`); `accel_term` works on the 29 interior centres (`:426-440`) and can stay on the full plan.
* Loss/head changes needed either way: per-band ADE/FDE, the continuity term, per-band weights; for (ii) the second head, a new `groups["O"]` entry so G-LIVE sees it, and `INIT_CKPT_MATCH` keys. Reuse: `P10Prior`, `upsample_prior`, `traj_terms` (split it, keep the old call as the default path), `accel_term`, HACC post-map, `code_mix`.
* Retrain class: (i) and (ii) are **O-only** if S, T and the adapter are frozen at an existing checkpoint (the `--init-ckpt` loader exists, `train.py:629`); T's code distribution does not change because T is untouched.

---

## 3. Strategic

### 3.1 What S's 30 / 50 / 80 m checkpoints mean in seconds

MEASURED on the eval-141 T1-kin dump `E:/Projects/TanitAD-artifacts/moved_from_c/hlat/run/dump_B_s0_m10/ep*.npz` (141 episodes, n = 2,399 windows, GT waypoints `g` at 0.2 s knots, ego speed `v0`; computed read-only with `venvs/tanitad`). Caveat: this is the **eval** window grid, not the training distribution (not measured here).

* Ego speed `v0`: p5 / p10 / p25 / p50 / p75 / p90 / p95 = 0.8 / 2.4 / 6.0 / 10.3 / 15.2 / 22.0 / 27.2 m/s; mean 11.43, std 7.58 (the std agrees with the 7.7 logged in `DESIGN_T1_V8F.md:118`, INHERITED). 20.2 % of windows are below 5 m/s.
* GT arc over 6 s: p5 / p25 / p50 / p75 / p95 = 6.8 / 35.2 / 60.1 / 89.8 / 164.7 m.
* Time to reach the arc, where the GT path gets there inside the 6 s plan: 30 m in 79.7 % of windows (median 2.6 s, p10-p90 1.4-4.8 s); 50 m in 59.9 % (median 3.8 s, 2.0-5.4); 80 m in 31.0 % (median 4.4 s, 2.8-5.8). Conditioning on "reached in 6 s" biases these down.
* Whole-distribution time `t = D / (GT 6 s mean speed, floor 0.5 m/s)` (ESTIMATED: extrapolates the 6 s mean speed):

| checkpoint | p10 | p25 | **p50** | p75 | p90 | share in 0-6 s | share in 6-30 s | share beyond 30 s |
|---|---|---|---|---|---|---|---|---|
| 30 m | 1.3 s | 2.0 | **3.0 s** | 5.1 | 11.9 | 79.7 % | 15.9 % | 4.4 % |
| 50 m | 2.2 | 3.3 | **5.0 s** | 8.5 | 19.8 | 59.9 % | 33.1 % | 7.1 % |
| 80 m | 3.6 | 5.3 | **8.0 s** | 13.6 | 31.6 | 31.0 % | 58.7 % | 10.3 % |

Reading against the requirements: the 30 m and 50 m checkpoints sit **inside the tactical band** at the median (3 s and 5 s); only the 80 m checkpoint is mainly in the 6-30 s band (median 8 s). An arc-indexed checkpoint is not a time-indexed one: at 1 m/s the 80 m checkpoint is 80 s away, at 25 m/s it is 3.2 s away. **S as built does not cover R-STR in seconds**, and covers 12-30 s hardly at all (the 80 m checkpoint lies beyond 13.6 s only for the slowest quarter of windows).

Strategic ground truth in the v7.2 release is better aligned to the band than S's checkpoints: for the 1,624 non-trivial `g_str` records, `by_time_s` has p50 12.4 s, p75 19.8, p95 27.0 (F9).

### 3.2 Do `g_str` and `a_str` reach any loss?

**No, not in v8F-mini.** Probes: (1) grep for `g_str|a_str|g_tac|a_tac|s2_labels` over `v8f_mini.py`, `v8f_mini_data.py`, `v8f_mini_train.py`, `launch_gate.py`: 0 hits in the model, data and trainer (hits in `launch_gate.py` are the string `refcv7`, unrelated); (2) the same pattern over `stack/tanitad/models/v8f*`, `stack/tanitad/data/v8f*`, `ra1_pooled.py`, `stack/scripts/v8f*`: 0 hits; (3) a pattern control: the same grep finds `g_str`, `a_str`, `bands`, `horizon` in `v7_labels.py:191-193, 409-418`, so the probe works. The strategic content of the v9 join is the route checkpoint only (`rc_*`, `data.py:134-146`). The v7.2 strategic tokens, their `by_time_s` / `within_m` arguments and the `strata` context are in the v7.2 file and are read by nothing in this stack.

### 3.3 A head for 6-30 s checkpoints, detected context and derived goals

Proposed S-side additions (new heads on S's pooled reader output `h` or on vision-only pooled tokens; none changes T's or O's input shapes, so the existing T/O weights stay valid):

| head | output | inputs at inference | labels (training only; R-DATA) | loss |
|---|---|---|---|---|
| `head_s2` time-indexed route checkpoints | (x, y, heading) at tau in {8, 12, 18, 24, 30} s (15 numbers; the exact set is a PI/MM choice) in the ego frame at NOW | as S today | ego-log positions at NOW + tau, smoothed as `rc` is (`build_v9_labels.py` section 6); masked where the log is shorter (F11: about 5 % of clips end before 30 s) | masked L1, same form as `T["ckpt"]` |
| `head_ctx` detected context | country (25 classes, `CENSUS_STRAT0.json`; the US is 50.7 % of the 306,152 clips, so class-balanced loss and macro metrics), period of day (classes TBD), illumination (night / dawn-dusk / day, from image statistics, not from the hour: F10), weather (classes TBD; no metadata column, so the label source must be built, for example from the Alpamayo CoT text in the v7.2 `cot_source` / `scene` blocks or a VLM pass; HYPOTHESIS that the CoT carries enough weather text) | **vision only**: the pooled scene tokens of the `hist` frames, optionally a slower summary of the 3 s WM window; no ego, nav, limit or T code (R-VIS) | `strata.country`, metadata `month` / `hour_of_day` for period of day (label construction only), image-derived illumination, weather from the chosen source | CE per attribute with per-attribute masks; the first three are per-clip constants, so they are cheap to label |
| `head_gstr` derived strategic goals | the 8 v7 strategic goal tokens (`vocab_v7.py:62-71`) with `by_time_s` and `within_m` regression; optionally a small set of context-derived goals (for example speed margin at night, right-of-way convention by country; the set is a design item for the PI) | S's reader output `h` + the **detached** `head_ctx` outputs + nav (nav is an input at train and inference) | per-frame re-emission of `g_str` / `a_str` over the 6-30 s band from the whole-recording track (F9: only 4 of 8 tokens occur today; `EXIT_*` and `LANE_CHANGE_*` never occur, so those four are untrainable) | BCE or CE with mask, args Huber |

Where it sits so the goal path stays disjoint from the situation classifier (2026-08-03 ruling):
* The "situation classifier" is the ego-dynamics situation classifier (`situations.py`); v8F-mini contains none, and pins this with `assert_no_situation_classifier_input` (`v8f_mini.py:1179`) and `INFERENCE_INPUT_KEYS` (`:169`). `head_ctx` reads only image tokens, exactly like the level head, whose vision-only property is pinned by `assert_level_head_vision_only` (`:1221`) and `level_head_probe` (`train.py:1578`). Copy that template: it supplies the guard, the probe and the declared-vs-built clause (`bad_level`).
* `head_gstr` is downstream of `head_ctx` (detached) and of S's own inputs; it reads **nothing from T** (T is below S in the cascade) and nothing produced by the situation classifier. State in the PREREG what the goal is computed from: scene tokens (vision) + nav + the vision-only context. The admissibility question "could this have been computed from the situation classifier's output?" is answered no, because the classifier is not an input to S.
* Goal to T: only through a detached token added to T's conditioning (the `aux_cond = token` path, `e_aux`, `:735-737`), default off, so attribution stays separable (a goal-token arm versus the same arm without it).
* PAUSED: item 25 switches the strategic layer off temporarily. Training these heads is allowed in principle; **reporting** them is the PAUSED row. Whether to spend training time on S-strategic before item 25 is lifted is a PI decision; the data release (section 6, item 7) is useful regardless, and R-DATA asks for it.

---

## 4. Eval: what assumes the old horizons and what must be added

| instrument | assumption today | file:line |
|---|---|---|
| ADE / FDE of O | whole 30-knot mean; readers carry `ade2`, `ade4`, `ade6`, `fde6` arrays (`v8f_oacc_read.py:446`, `v8f_speedavg_read.py:141`, in the Research folder `2026-10-17-v8f-mini/code/`). Only `ade2` is an operative-band quantity; `ade4` is in no band; there is no tactical-band ADE (2-6 s) and no FDE at 2 s | `v8f_mini.py:418` |
| LONGITUDINAL, LATERAL families | computed on the whole predicted trajectory; `dt` inferred from `wp_steps` (`ff.py:190-219`), no band argument | `ff.py:447`, `:806` |
| TACTICAL family | `tactical_goal` = goal point at the end of the plan ("FDE at the tactical horizon", `ff.py:1240-1330`); `tactical_from_trajectory` labels the manoeuvre of the whole plan (`ff.py:1330`); T's decision quality against v9 labels is a separate reader, band 2-8 s (F8) | `ff.py` |
| consistency rules | `horizon_s = 6.0`, `band_start_s = 2.0`, `ckpt_arcs_m = (30, 50, 80)` | `cons.py:48-57` |
| STRATEGIC family | reported n/a with a corpus reason (`ff.py:1405-1500`), PAUSED by item 25 | `ff.py:1487` |
| WM diagnostics (T0) | WMACT / IMAG reads: L1 k 1/5/10, L2 and L3 k 1/3 (`PREREG_V8F_MRATE.md` E1); IA reads at horizons "each arm's window allows" (LFS card) | cards |

What to add so each level is scored in its own band (the four families stay, per family, never pooled; paired episode-cluster bootstrap B = 2,000; tier stamp on every number):
1. **OPERATIVE band (0-2 s):** `ade_op`, `fde_2s`, longitudinal and lateral families computed on knots 1-10 (slice before calling `longitudinal` / `lateral`, passing `dt`); WM L1 reads at k = 1..10.
2. **TACTICAL band (2-6 s):** `ade_tac` (knots 11-30), `fde_6s`, families on knots 11-30; T's decision quality (goals, lat, lon, constraints) against labels in the **same** band as R-TAC (depends on item 6 of section 6; until then report T against the 2-8 s labels and say so); goal point at 6 s; WM L2 reads up to k x tau = 6 s.
3. **TRAJECTORY (R-TRAJ, 6 s):** today's metrics unchanged, plus the knot-10 continuity residual.
4. **STRATEGIC band (6-30 s), computed but printed in the PAUSED row with the item-25 ruling:** time-indexed checkpoint error per tau; context accuracy (country, period of day, illumination, weather) with n, macro-F1, confusion and the **episode-cluster** estimator (country, hour and illumination are per-clip constants, so n_eff is the episode count, not the window count); goal match against `g_str`. WM L3 reads up to its reach.
5. **Baselines per band:** copy-the-present and P10 per band (IMAG1B found no WM level beats copying the present, `PREREG_V8F_REL.md` header), and a WM-off control per band.
6. **Tier:** the WM reads are T0 (diagnostic); anything that claims driving is T1 (`EVAL_DOCTRINE.md`); do not mix.
7. An eval-time check that the loader builds the identical model (G-EVAL), with the new heads present.

Cheap first step, no training: evaluate the banked M2b dumps (the `dump_B_s*_m10` family) per band with the sliced functions to give the **baseline per band** before any retrain.

---

## 5. Launch gate

For the mini trainer, G-DVB is inside the trainer: `FLAG_CHECKS` (`train.py:68`), `declared_vs_built` (`:941`) which **refuses any argparse flag without an entry** (`:951-953`), the `bad_*` clauses per lever, and `live_check` (`:1054`). `launch_gate.py` at the tip has no profile for this trainer (F12). **Open, for the Master Mind:** the MRATE and LFS cards say "each needs a launch-gate PASS token"; if that means `launch_gate.py run --profile ...`, a mini profile does not exist, and adding one edits `launch_gate.py`, which moves profile fingerprints pinned in three tests (memory note "Pin edit moves profile fingerprints", INHERITED). Not done by this audit.

Entries a new ladder and new heads need (each with a deliberate-regression arm that must turn the gate red):

| lever | G-DVB kind and reader | clause | deliberate-regression arm |
|---|---|---|---|
| `--wm-strides` | `lever:wm_strides` (exists in `1_strides.diff`: reads the `LevelConfig` stride, macro encoder attribute, macro parameter shape, `built_levers`) | `bad_wm_strides` | flag set, config built at (2,2) |
| `--wm-ctx` (per level) | `lever:wm_ctx`, read from `predictor.pos.shape[0]` | new `bad_wm_ctx` | context declared 1, built 3 |
| `--wm-rollout-k` (per level) | `lever:wm_rollout_k`, read from the loss's own loop count (a probe that counts predictor calls per level) | new `bad_wm_k` | k declared 10, loss builds 1 (the F1 defect) |
| `--wm-windows` | `data:wm_windows`: sampler kind, F, anchors per clip, read back from the built dataset | new `bad_wm_windows` | sampler declared, `train_windows` built |
| F-versus-guard | replace `v8f_mini.py:265` by `F >= max_l F_l` and per-level transitions >= 2 | in `__post_init__` and `bad_wm_strides` | stride that leaves L3 with 1 transition |
| `--w-op`, `--w-tac-traj`, `--w-cont`, `--o-heads` | `cfg:` and `lever:` | new `bad_oband`: each term moves the total by exactly its weight; head count built = declared | weight 0 with the head built |
| `--ctx-head`, `--w-ctx`, `--str-ckpt-times`, `--w-str-ckpt`, `--w-gstr` | `lever:` and `cfg:`, copying `bad_level` / `bad_lead` | head with no loss, loss with no head, a non-vision input in `head_ctx` (source guard + a `ctx_head_probe` like `level_head_probe`) | a head that reads `p10f` |
| label release flags (`--v10-train`, md5) | `data:` md5 in an accepted list (`data.py:62-64`) | refuse an unknown md5 | a stale v9 file |
| `INIT_CKPT_MATCH` (`train.py:657`) | add the new keys; `WM_ONLY_NOT_COMPARED` (from `1_strides.diff`) already lists `hmax`, `wm_strides` | | |

Other gate parts: **G-LIVE** needs each new group in `groups` (`v8f_mini.py:1074-1077`) so "every trainable group receives gradient" covers it, with the exact-0.0 control; **G-CLOCK**: time-indexed labels read NOW + tau from the log, so the join key (`k = 2j`, `data.py:9-12`) must be re-verified for the new release; **G-CKPT**: round-trip with new heads; **G-EVAL**: the eval loader builds the same heads; **G-SUITE**: the pinned v8f tests.

---

## 6. Ordered minimal change list

Size: S small, M medium, L large. Retrain: F = full cascade, WM = WM-only, O = O-only, H = head-only on frozen trunks, none.

| # | change | size | reuse | tests | retrain |
|---|---|---|---|---|---|
| 0 | **PI/MM decisions** that block labels: tactical label band 2-6 (or keep 2-8 and report it); strategic 6-30 with the 6-8 s hole closed; the time set for checkpoints; context class lists; whether to train strategic heads before item 25 is lifted | S (decision) | F8, F9 numbers | none | none |
| 1 | **Per-band eval, baseline first.** Slice `ade` / `fde` / families by band; add `ade_op`, `ade_tac`, `fde_2s`, `fde_6s`; run on the banked M2b dumps | S-M | `ade2/ade4/ade6/fde6` arrays, `longitudinal` and `lateral` take `dt`, `cons.py` band | analytic trajectories (constant speed, constant curvature) per band; slice equals full when band = full | none |
| 2 | **Horizon constants in one place** (OP_S 2.0, TAC_S 6.0, STR_S (6, 30); `K_OP` 10, `K_TAC` 30) used by `v8f_mini.py:41-43`, `data.py:48`, `cons.py`, eval | S | existing literals | a pin test that greps for stray literals (the pattern of `test_physicalai_feature_readset.py`) | none |
| 3 | **WM ladder flags:** land `--wm-strides` (`1_strides.diff`); add per-level `--wm-ctx` and `--wm-rollout-k` (a k-step free-run loss built on `HJEPAStack.rollout`, `hjepa.py:732`); new guard `F >= max F_l` | M | `1_strides.diff`, `wm_k` / `wm_only` of `mrate_lp4`, `rollout()`, `LFS_MIN_TRANSITIONS` | k = 1 reproduces today's loss bit for bit; gradient on each level's predictor at k > 1; **mutation:** delete the k-step line and the gate must go red; stride shape tests; the G-DVB clauses of section 5 | WM |
| 4 | **WM window sampler and frozen-adapter pooled path** (anchors with `t >= F-1`, no label requirement; per-frame mean token computed once under no-grad or cached) | M | `V8FMiniFrameWindows` pattern (`data.py:913`), `WMAux.z1` | pooled path equals the live adapter mean to 1e-6; windows never cross a clip end | WM |
| 5 | **O band split:** loss and metric split + continuity (design i), optional two-head (design ii) | S then M | `traj_terms`, `accel_term`, HACC table, `--init-ckpt` | default argv total bit-equal; each band term moves the total by exactly its weight; continuity zero on GT; per-head gradient | O |
| 6 | **T band re-label:** per-frame release with the tactical band (2, 6) if the PI rules so; T and O retrain on it. Cheaper interim: keep v9 and mask constraint cells with times above 6 s | L (data) / S (interim) | `build_v9_labels.py` and its truncation test (5.3 / 7.3 / 15.4 % label change at 6 s, F8) | the truncation test again, label-change table against v9; join-key check | T + O (S frozen) |
| 7 | **Strategic label release (R-DATA):** per-frame `g_str` / `a_str` over (6, 30), time-indexed route checkpoints, `strata` context copied per frame, illumination from image statistics, weather from the chosen source; saved private on HF with sha256 | L | `s2_geom_emit_v7.py`, `build_v9_labels.py`, `CENSUS_STRAT0.json`, `strata` in v7.2, the HF upload tooling | schema pin, md5 list, joins, coverage per attribute and per class, G-CLOCK; HF sha256 verify | none (data) |
| 8 | **Strategic heads** `head_ctx`, `head_s2`, `head_gstr` on frozen trunks | M-L | the level-head template (`_level`, `bad_level`, `level_head_probe`, vision-only guard) | vision-only guard with a mutation (add `p10f` and it must go red); gradient per head; exact-0 detach control; eval PAUSED row prints the ruling | H (S/T/O frozen), then optional joint |
| 9 | **Cards:** a new ladder card (reach x step) and amendments to MRATE / DIVRS / REL as below; each frozen with sha256 before code | S | existing cards | n/a | n/a |

Order rationale: 1 and 2 cost nothing and baseline every later claim; 3 and 4 are the WM change the PI asked for and are WM-only (driving cannot change, F4); 5 is O-only and gives R-TRAJ; 6-8 need data and PI decisions and gate everything strategic.

### 6.1 The four frozen cards on the new ladder

| card | stays valid? | why | what is needed |
|---|---|---|---|
| **MRATE** (`PREREG_V8F_MRATE.md`, frozen sha `50667a79...`) | The **question** stays valid; the **arms and the chain it anchors** do not carry over. | W0 / W1 are defined on strides (2, 2) and hmax 28 (L1 0.2 s, L2 0.4 s, L3 0.8 s); W1's K = 3 history at "that level's own step" gives 0.4 / 1.2 / 2.8 s of history only for that geometry (card text). Its outcomes decide "W1 becomes the WM of record", which LFS, DIVRS and REL all take as base B. On ladder A the same K = 3 reads 0.4 / 1.2 / 5.4 s of history. | A new or amended card with B = the new ladder; the `wm_k` code is reusable unchanged. Running MRATE on (2, 2) first costs four box-2 runs for a geometry the PI rejected as the base. |
| **LFS** (`PREREG_V8F_LFS.md`) | **Needs a new card.** | Its arm F33 (strides 3, 3: 0.2 / 0.6 / 1.8 s) is candidate A's geometry **without** the reach lever and with hmax 28 (transitions 27 / 9 / 3). LFS asks "does a longer step make levels hold different timescales (TSI)"; the new ladder asks "does a band-matched reach make each level predict inside its band". The two share the flag. | A LADDER card: arms = current (2, 2) k = 1 as control, F33 k = 1 (this is LFS's F33), A (k = 10), B; LFS's TSI read stays as a beside-read. LFS is subsumed; its outcome table does not decide the new question. |
| **DIVRS** (`PREREG_V8F_DIVRS.md`) | **Needs a new card or an amendment.** | Bins use the per-window mean abs yaw rate and mean acceleration over the window's transitions (5 x 5 grid). With F = 91 (18 s) or 151 frames those means average turns and straight driving together, and the grid collapses toward the centre. | Re-bin on the per-level reach window (2 s for L1, 6 s for L2); then the sampler flag and its G-DVB (`2_divrs.diff`) are reusable. |
| **REL** (`PREREG_V8F_REL.md`) | **The mechanism stays valid; the token spec needs an amendment.** | `--o-reads-wm pred` feeds L2 and L3 states for k = 1, 2, 3. On ladder A that is 0.6-1.8 s (L2) and 1.8-5.4 s (L3); the useful tokens for an operative/tactical O are L1 up to 2 s and L2 up to 6 s. The rollout action is the P10 mean held over the horizon: acceptable at 2 s, a weak assumption at 6 s. The "reads the BUILT token count" control (`3_rel.diff`) still applies. | Amend the token list and the k set; keep the stop-gradient and admissibility statements. |

---

## 7. Open items, stated plainly

1. Whether the cached video extends beyond 20 s per clip (decides candidate D, the 30 s image window): HYPOTHESIS no; one cheap probe is the cache manifest `n_frames` distribution.
2. Whether a PASS token for the mini trainer is issued by `launch_gate.py` (F12, section 5).
3. The mapping of the v9 5/7 action classes onto the v7 8/8 action vocabulary (section 2).
4. Train-side speed distribution for the checkpoint-seconds table: only the eval-141 grid was measured (section 3.1).
5. Peak GPU memory of any candidate was not measured (no GPU work): section 1.5 is arithmetic, ESTIMATED.
6. The weather label source does not exist in the metadata; its feasibility from CoT text is a HYPOTHESIS until a coverage probe runs.
