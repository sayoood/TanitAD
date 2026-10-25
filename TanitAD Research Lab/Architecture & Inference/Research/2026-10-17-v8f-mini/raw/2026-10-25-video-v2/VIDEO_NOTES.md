# v8F-mini evidence video v2 -- VIDEO_NOTES (every number, its artifact, evidence class, tier, estimator)

Written 2026-10-25 by the viz2 subagent. Dev box only, CPU only, nothing staged, committed or pushed. All paths are under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/` (`SC` below).

## 0. Status and honest caveats

* The shipped `viz2/v8f_evidence_v2.mp4` is the FIRST full build (146.4 s, 1920x1080, H.264, 10 fps, 6 segments). The build was stopped there because `C:` free space fell below the 1.5 GB hand-back threshold (about 0.38 GB free, caused by other processes; the viz2 footprint is 57 MB). Three small source fixes were made AFTER that build and are in `viz2/v2_scenes.py` but NOT in the mp4: (1) segment 5's S card gains an S-age chart (50 m error of the held vs fresh checkpoint by the age of S's last tick, both seeds, from `SC/sfresh/sfresh_extra.json`; preview `viz2/_t_s_card.png`); (2) the caption of the two segment-5 window frames is shortened (in the built video its last line touches the bottom edge of the caption box, text still legible) and now points to the segment's closing card for seed 1; (3) the segment-3 summary caption now says the 23-of-24 T-lateral count is with the nav command at all three layers. One command rebuilds the video from source once C: has room: `CUDA_VISIBLE_DEVICES=-1 V2_THREADS=1 OMP_NUM_THREADS=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe viz2/build_video.py --workers 4` (writes atomically via a .tmp.mp4).
* Tier of everything: **T1-class open loop** (the model reads its own held outputs; nothing it does feeds back). The nav counterfactual is an **INTERVENTION on the model's input** (a diagnostic of dependence), never driving. Fresh-S ages are a diagnostic, not the deployed schedule.
* Estimator: episode-cluster bootstrap, B = 2,000, 141 episodes (`taniteval/ci.py`), paired for differences. It answers *another draw of EPISODES* only. Training-seed variance = the two seeds M2b_s0 / M2b_s1 side by side (one model per seed, no replicate arm). Inference is deterministic (no sampling), so there is no inference-seed floor. A separated CI from a one-seed arm is necessary, not sufficient, for a *lever* claim; nothing here claims a lever.
* STRATEGIC family: **PAUSED** by the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` item 25). Segment 5 shows S's checkpoints as a diagnostic of the layer only; it is not a strategic-family pass and does not lift the pause.
* Evidence classes: MEASURED (ours) = computed in this task from the banked checkpoints/arrays (artifact paths below); INHERITED = quoted from another agent's result file, with the number checked here against that file's table (LISTEN, SFRESH, HBLEND, HSPEC). Nothing is registered or promoted by this video.
* One window is an example; the eval-wide numbers are the evidence. The 4 windows are exactly those of `SC/viz/out/selection.json`, declared in `viz2/SELECTION_V2.md` before any run (that file was not edited afterwards; see CONTROLS_V2.json `pre_declaration`).

## 1. Controls (`viz2/CONTROLS_V2.json`): ALL_PASS = True

| control | result | tolerance |
|---|---|---|
| 1. window's own nav at S, T and O reproduces B's own-code plan (`forward_infer`) | max abs difference **0.0** (traj30, S ckpt, T lat/lon logits), 8 window-seed pairs | exactly 0.0 |
| 2. H / HS / B / P10 as drawn equal the banked per-window arrays | max abs difference **0.0 m** for all four plans; ADE6 recomputed from the drawn arrays vs banked: max 3.5e-07 m | <= 1e-5 m |
| 3. frame alignment (v1 control, all 101 cache steps of the 4 clips) | argmax = self **404 / 404**, min PSNR 48.6 dB | all self |
| 4. zeroed `e_nav` model: D6, D6_TURN_L/R, L-R spread, LR_dist (structural zero) | **0.0** over 160 cells | exactly 0.0 |
| info: this CPU float32 re-run vs the banked box-2 GPU arrays | own plan 1.6e-05 m, S ckpt 7.6e-06 m, fresh plan 8.6e-06 m (float32 round-off) | <= 1e-4 m |

## 2. Segment 3: nav counterfactual -- MEASURED (ours), T1-class open loop, intervention

Artifacts: `viz2/data/cf_windows.npz` (4 windows x 2 seeds x 4 injection points x 3 commands; keys `<seed>/<window>/cf/<cfg>/<cmd>/{traj30,ckpt_now,lat_p,lon_p}` and `.../D6`, `.../lat_sep_LR`), produced by `viz2/cf_run.py`; eval-wide `viz2/data/cf_evalwide.{npz,json}` (all 2,399 windows, `viz2/cf_evalwide.py`). Commands: constant over the whole nav history, d_next = 30 m, args_valid = 1; the model's OWN `run_s -> run_t -> run_o` chain, only the named layer's `b['nav']` replaced.
**D6** = mean over {TURN_L, TURN_R} of the Euclidean distance at 6 s between that plan and the FOLLOW plan at the same injection point (m). **L-R** = |lateral(TURN_L) - lateral(TURN_R)| at 6 s (m).

### 2a. The four pre-declared windows (D6 m / L-R m)

| window (eval index, clip, cache step t; 0.2 s per step) | seed | (a) S+T+O | (b) O only | (c) T only | (d) S only |
|---|---|---|---|---|---|
| JUNCTION (#1131, 79e30ac2, t=35) | 0 | 4.90 / 8.74 | 2.61 / 0.23 | 4.15 / 8.22 | 4.25 / 8.43 |
| JUNCTION (#1131, 79e30ac2, t=35) | 1 | 10.71 / 10.42 | 3.57 / 0.55 | 2.65 / 0.24 | 7.62 / 12.02 |
| CURVE (#654, 45aea78c, t=31) | 0 | 22.07 / 10.47 | 1.70 / 0.57 | 3.73 / 0.10 | 18.30 / 21.22 |
| CURVE (#654, 45aea78c, t=31) | 1 | 19.09 / 22.13 | 1.52 / 2.16 | 4.74 / 0.74 | 26.07 / 46.59 |
| NO-TURN (#763, 51c01ba4, t=59) | 0 | 28.99 / 21.63 | 1.33 / 1.88 | 13.82 / 8.76 | 21.80 / 39.61 |
| NO-TURN (#763, 51c01ba4, t=59) | 1 | 20.87 / 20.25 | 2.29 / 3.77 | 14.22 / 7.16 | 27.71 / 51.22 |
| FAILURE(worst) (#2158, ed87040c, t=59) | 0 | 25.14 / 47.36 | 2.65 / 0.49 | 3.71 / 4.66 | 22.48 / 43.28 |
| FAILURE(worst) (#2158, ed87040c, t=59) | 1 | 16.96 / 31.57 | 1.30 / 0.93 | 2.62 / 3.84 | 16.03 / 29.99 |

T's lateral decision under the all-layers command (argmax of the T lateral head; expected LANE_KEEP / TURN_L / TURN_R): follows the command in **23 of 24** cases (4 windows x 2 seeds x 3 commands); the one exception is FAILURE(worst) seed 1, TURN_R.

### 2b. Eval-wide, all 2,399 windows (mean, 95 % episode-cluster CI)

| seed | metric | (a) S+T+O | (b) O only | (c) T only | (d) S only |
|---|---|---|---|---|---|
| 0 | D6 (m) | 11.16 [9.77, 12.62] | 1.06 [0.98, 1.14] | 4.24 [3.56, 4.98] | 9.75 [8.48, 11.09] |
| 0 | lat_sep_LR (m) | 9.82 [8.51, 11.13] | 0.73 [0.64, 0.84] | 2.03 [1.66, 2.42] | 15.16 [12.90, 17.52] |
| 1 | D6 (m) | 10.76 [9.71, 11.85] | 1.22 [1.12, 1.31] | 4.04 [3.32, 4.84] | 11.12 [9.77, 12.53] |
| 1 | lat_sep_LR (m) | 10.50 [9.34, 11.66] | 1.26 [1.07, 1.46] | 1.80 [1.49, 2.13] | 19.16 [16.68, 21.88] |

M2b_s0: with nav at all layers, T's lateral argmax equals the command in FOLLOW 94 % / TURN_L 52 % / TURN_R 62 % of windows.

M2b_s1: with nav at all layers, T's lateral argmax equals the command in FOLLOW 90 % / TURN_L 54 % / TURN_R 73 % of windows.

Reading (all MEASURED, both seeds): O only (about 1 m) < T only (about 4 m) < all three layers ~ S only (about 10-11 m). The route reaches the plan mostly through S's checkpoints and T's code; O's own nav token is nearly inert. 'Through T' holds in the sense T-only >> O-only; S-only is the largest single-point effect, so the nav also reaches O through S's checkpoints, which O reads directly. On single windows the O-only / T-only order can flip (T only < O only on JUNCTION seed 1).

## 3. Segment 2: who listens to what -- INHERITED from LISTEN, each number re-checked against the table in the file

Source: `SC/listen/LISTEN_RESULT.md` (sections 2 and 4; `READ.json`; `fig_listen_grid.png` is shown as the grid). Tier T1-class open loop, donor-swap interventions; paired episode-cluster bootstrap B = 2,000, 141 episodes; both seeds M2b_s0 / M2b_s1. Numbers on screen (seed 0 / seed 1): S nav +4.197 / +3.985 m at 50 m, S scene +3.214 / +3.632 m, S all-inputs +8.057 / +8.375 m; T-lateral macro-F1 drop for S checkpoints +0.391 / +0.406, nav +0.173 / +0.216, scene -0.003 / +0.010 (not separated); T-longitudinal ego +0.247 / +0.261, scene +0.185 / +0.187; O ADE6 ego +5.697 / +4.716 m, T code +5.395 / +4.132 m, S checkpoints +0.873 / +1.894 m, nav +0.015 / +0.011 m (seed 1 not separated), all-inputs +1.394 / +1.315 m. Q-NAV (section 4): ADE6 change with the donor nav at S only +0.726 / +0.821, T only +0.476 / +0.521, O only +0.015 / +0.011, all three +1.174 / +1.180 m; D = T-only minus O-only +0.461 [+0.254, +0.687] / +0.510 [+0.273, +0.793], separated at both seeds.

## 4. Segment 4: better plans -- INHERITED from HBLEND and HSPEC per-window arrays and score files, re-read programmatically

Artifacts: `viz2/data/tables.json` and `viz2/data/plans_pack.npz` (from `SC/hblend/score_M2b_s{0,1}.json`, `score_M2b_s{0,1}_perwindow.npz` and `SC/hspec/score_hs2_M2b_s{0,1}.json`, `..._perwindow.npz`; `viz2/plans_pack.py`). Plans: B = M2b's own plan, P10 = frozen kinematic prior, H = HBLEND shrinkage of B toward P10, HS = HSPEC stage 2. Tier T1-class open loop (held outputs scored against the logged human path).

### 4a. ADE by horizon over all 2,399 windows (m; mean [95 % CI])

| seed | plan | 1 s | 2 s | 3 s | 4 s | 5 s | 6 s |
|---|---|---|---|---|---|---|---|
| 0 | B | 0.082 [0.071, 0.095] | 0.316 [0.277, 0.362] | 0.577 [0.525, 0.640] | 0.984 [0.908, 1.070] | 1.518 [1.403, 1.647] | 2.175 [2.007, 2.357] |
| 0 | H | 0.060 [0.051, 0.073] | 0.212 [0.192, 0.239] | 0.473 [0.432, 0.520] | 0.854 [0.786, 0.929] | 1.353 [1.246, 1.468] | 1.968 [1.811, 2.131] |
| 0 | HS | 0.064 [0.055, 0.077] | 0.217 [0.196, 0.244] | 0.478 [0.438, 0.525] | 0.855 [0.785, 0.932] | 1.354 [1.245, 1.470] | 1.976 [1.819, 2.140] |
| 0 | P10 | 0.063 [0.054, 0.076] | 0.222 [0.201, 0.249] | 0.534 [0.486, 0.587] | 1.013 [0.925, 1.107] | 1.663 [1.516, 1.811] | 2.487 [2.261, 2.713] |
| 1 | B | 0.071 [0.061, 0.083] | 0.291 [0.254, 0.340] | 0.552 [0.498, 0.618] | 0.936 [0.859, 1.020] | 1.440 [1.325, 1.563] | 2.055 [1.886, 2.232] |
| 1 | H | 0.060 [0.052, 0.073] | 0.214 [0.193, 0.240] | 0.481 [0.441, 0.527] | 0.866 [0.798, 0.940] | 1.369 [1.260, 1.483] | 1.985 [1.820, 2.159] |
| 1 | HS | 0.063 [0.054, 0.076] | 0.217 [0.196, 0.243] | 0.486 [0.446, 0.533] | 0.869 [0.799, 0.945] | 1.373 [1.263, 1.488] | 1.997 [1.833, 2.169] |
| 1 | P10 | 0.063 [0.054, 0.076] | 0.222 [0.201, 0.249] | 0.534 [0.486, 0.587] | 1.013 [0.925, 1.107] | 1.663 [1.516, 1.811] | 2.487 [2.261, 2.713] |

### 4b. Paired differences at 6 s, acceleration MAE and constraint violations

| seed | pair | ADE6 delta (m) | accel MAE delta (m/s^2) | CONSISTENCY(a) delta (share of windows) |
|---|---|---|---|---|
| 0 | H-B | -0.207 [-0.277, -0.142] sep. | -0.464 [-0.554, -0.380] sep. | +0.101 [+0.071, +0.133] sep. |
| 0 | H-P10 | -0.518 [-0.635, -0.407] sep. | +0.104 [+0.080, +0.134] sep. | n/a |
| 0 | HS-B | -0.200 [-0.286, -0.121] sep. | -0.542 [-0.643, -0.447] sep. | +0.101 [+0.067, +0.135] sep. |
| 0 | HS-P10 | -0.511 [-0.617, -0.411] sep. | +0.026 [+0.011, +0.044] sep. | n/a |
| 0 | HS-H | +0.007 [-0.012, +0.025] not sep. | n/a | +0.000 [-0.010, +0.010] not sep. |
| 1 | H-B | -0.070 [-0.101, -0.042] sep. | -0.228 [-0.289, -0.174] sep. | +0.075 [+0.057, +0.095] sep. |
| 1 | H-P10 | -0.502 [-0.621, -0.385] sep. | +0.066 [+0.040, +0.102] sep. | n/a |
| 1 | HS-B | -0.058 [-0.105, -0.015] sep. | -0.288 [-0.364, -0.226] sep. | +0.069 [+0.046, +0.092] sep. |
| 1 | HS-P10 | -0.490 [-0.598, -0.385] sep. | +0.006 [-0.011, +0.028] not sep. | n/a |
| 1 | HS-H | +0.012 [-0.006, +0.029] not sep. | n/a | -0.007 [-0.016, +0.003] not sep. |

Acceleration MAE (m/s^2): seed 0: B 1.206, H 0.741, HS 0.663, P10 0.638; seed 1: B 0.931, H 0.704, HS 0.643, P10 0.638. Consistency violation rate: seed 0: B 0.160, H 0.260, HS 0.261, P10 0.311; seed 1: B 0.210, H 0.285, HS 0.278, P10 0.311.

The next step named on the gap card (a second-difference penalty in the training loss) needs a launch gate and box-2 time; it is not run here.

### 4c. The four windows, ADE6 of each plan (m; seed 0 / seed 1), `viz2/data/plans_pack.npz`

| window | P10 | B | H | HS |
|---|---|---|---|---|
| JUNCTION | 4.46 / 4.46 | 2.66 / 1.58 | 1.09 / 1.24 | 0.79 / 1.00 |
| CURVE | 5.25 / 5.25 | 2.72 / 5.37 | 3.66 / 5.32 | 3.99 / 5.29 |
| NO-TURN | 1.75 / 1.75 | 1.39 / 2.15 | 1.42 / 2.08 | 1.44 / 2.04 |
| FAILURE(worst) | 11.73 / 11.73 | 16.95 / 16.90 | 15.85 / 16.03 | 15.38 / 15.69 |

## 5. Segment 5: the strategic layer at its own tick (DIAGNOSTIC; family PAUSED)

Eval-wide, INHERITED from `SC/sfresh/SFRESH_RESULT.md` / `sfresh_numbers.json` (PREREG_V8F_SFRESH, tip 583cfc02), re-read into `viz2/data/tables.json`; n = 2,319 windows with a valid 50 m label; paired episode-cluster bootstrap B = 2,000, 141 episodes. Per-window S checkpoints are MEASURED here (`viz2/data/cf_windows.npz`: `own_ckpt_now`, `fresh_ckpt_now`, `rc` = the v9 RC-A GT checkpoints in the NOW frame).

| seed | level | 50 m position error, mean [95 % CI] (m) |
|---|---|---|
| 0 | banked eval ages (S held) | 8.01 [7.31, 8.79] |
| 0 | S fresh (ticked at NOW) | 2.30 [1.76, 2.96] |
| 0 | nav-only kinematic baseline | 4.73 [3.72, 5.90] |
| 1 | banked eval ages (S held) | 7.94 [7.29, 8.65] |
| 1 | S fresh (ticked at NOW) | 2.20 [1.75, 2.74] |
| 1 | nav-only kinematic baseline | 4.73 [3.72, 5.90] |

Seed 0 paired at 50 m: fresh S - nav-only baseline -2.43 [-3.22, -1.74]; fresh - held (stale cost) -5.72 [-6.44, -5.06]; banked cycle S - nav-only baseline +3.28 [+2.06, +4.47].

Seed 1 paired at 50 m: fresh S - nav-only baseline -2.53 [-3.33, -1.79]; fresh - held (stale cost) -5.73 [-6.46, -5.07]; banked cycle S - nav-only baseline +3.21 [+1.96, +4.38].

S-age chart (source only, not yet in the mp4): `SC/sfresh/sfresh_extra.json::age_profile_50m`, held vs fresh by S age 0.0-1.2 s, both seeds.
Part of the held-vs-fresh gap is DEFINITIONAL: the banked eval scores S's checkpoints (predicted in the frame of S's tick) against labels in the NOW frame, so it pays for the distance driven since the tick. The deployed-schedule cost is the next strategic lever (not measured here).

## 6. Frames and windows

Frames: `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/{clip}.v2ep.pt`, cache step j = native frame 2j (cylindrical 256 x 640). Windows (eval index, clip, t in s): JUNCTION #1131 79e30ac2-c619-4d61-98a3-1007037b4d05 t=35; CURVE #654 45aea78c-7762-4e46-aef1-e9e22602b8fd t=31; NO-TURN #763 51c01ba4-9000-4370-8abb-9fd22b397d05 t=59; FAILURE(worst) #2158 ed87040c-cf3c-4fe4-9a95-50c1114f0d38 t=59.

## 7. Manifest (all under `SC/viz2/`, dev box, nothing in git)

`v8f_evidence_v2.mp4`, `video_chapters.json`, `SELECTION_V2.md` (pre-declaration, unchanged), `CONTROLS_V2.json` (+ `_part1/_part2`), `VIDEO_NOTES.md`, `qa_*.png` (25 stills), `data/{cf_windows.npz,cf_evalwide.npz,cf_evalwide.json,plans_pack.npz,tables.json}`, scripts `v2_lib.py v2_draw.py v2_scenes.py build_video.py cf_run.py cf_evalwide.py plans_pack.py controls.py notes_gen.py qa_one.py`, `code/` (tip 9a6551d5 export, 38 MB), `_t_s_card.png` (preview of the S-age chart).
