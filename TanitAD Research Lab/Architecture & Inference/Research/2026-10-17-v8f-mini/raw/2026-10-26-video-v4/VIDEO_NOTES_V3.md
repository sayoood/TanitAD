# v8F-mini evidence video v3 (HACC) -- VIDEO_NOTES_V3 (every number, its artifact, evidence class, tier, estimator)

Written 2026-10-25 by the viz3 subagent. Dev box, CPU only, ONE render process, nothing staged, committed or pushed. All paths are under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/` (`SC` below). Video: `viz3/v8f_evidence_v3.mp4`, 215.6 s, 1920x1080, H.264, 10 fps, 7 segments, 177 unique frames.

## 0. Status, caveats, and what changed from v2

* **Tier of every number: T1-class open loop (own held outputs; NOT closed loop).** The model reads its own held outputs; the plan is scored against the logged human path; nothing feeds back. Interventions (the nav counterfactual, the LISTEN donor swaps) are diagnostics of dependence.
* **Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes (taniteval/ci.py); another draw of EPISODES only; training variance = the two seeds side by side; inference deterministic.** A separated interval from a one-seed arm is necessary, not sufficient, for a lever claim (H-ESTIM-SEED-1); the control B1 below re-derives the four primary dominance intervals with an independent bootstrap.
* **Baseline:** P10 is the programme's own kinematic floor (it extrapolates ego motion and ignores the scene). The dominance claim is against that floor, not against the state of the art; no external benchmark is shown.
* **Seed 1's KEEP margin is thin** (on screen in segment 5): ADE6 HACC - H = +0.0313 [+0.0026, +0.0586] m is SEPARATED-WORSE than H and passes the F/2 bar (+0.06037 m) by 0.0018 m. Seed 0: +0.0151 [-0.0030, +0.0320] (not separated).
* **STRATEGIC family: IN FORCE for v8F** (PI-V8F-2026-10-17; R72 / R73; confirmed by `hacc/RESULT_V8F_HACC.md` section 8 and `score_<seed>.json::structural_identity.strategic`). **v2's notes and caption said PAUSED (PI ruling 2026-09-19): that is stale for v8F and is corrected in every v3 caption** (segment 6's caption and the closing card say IN FORCE; control S finds no 'paused' string among the drawn text). HACC touches the 6 s plan only; S's checkpoints and T's decision heads are the banked B arrays, so the strategic block is unchanged by HACC.
* Never drawn: the words 'drives' and 'SOTA' (control W); 'closed loop' appears only in the mandated tier string 'NOT closed loop'.
* **v2 -> v3 changes.** (1) HACC drawn beside GT / P10 / B / H on the same 4 pre-declared windows (`SELECTION_V2.md`, unchanged, sha256 below); HS (v2's fifth line) is kept as a thin dotted line and a table row so v2's content is not lost. (2) New segment 5 'Dominance over the kinematic prior'. (3) The three pending v2 source fixes are applied and visible in the v3 render: segment 6's S card has the S-age chart (held vs fresh 50 m error by S's age; both seeds, from `sfresh/sfresh_extra.json::age_profile_50m`), the two segment-6 window frames carry the shortened caption that points to the card for seed 1, and the segment-3 summary caption says the 23-of-24 T-lateral count is with the nav command at all three layers. (4) v2's stale STRATEGIC caption corrected (above). (5) The v2 gap card ('what is NOT yet better', which named a second-difference penalty in training as the next step) is replaced by the HACC version: that penalty was realised as HACC's fit objective, with zero training. (6) Segment 3's plans are the model's own plan B BEFORE the post-map; this is now written on the method card and in the footers (HACC is a post-map of B's plan; the counterfactual is a diagnostic of the model's chain).
* **Evidence classes.** HACC numbers (segments 4 and 5) are **MEASURED (ours)** by PREREG_V8F_HACC's run (`SC/hacc/`, tip c8ea1623, `raw/2026-10-25-hacc/`) and are **re-read here, not re-fitted**: every on-screen number is bound to its JSON source (control N1) and to `RESULT_V8F_HACC.md` (N2); the plan arrays are re-derived from the banked weights (H1); the per-window and eval-wide ADE means are recomputed from the banked per-window arrays (H2); the four primary intervals are re-derived with an independent bootstrap (B1). The paired intervals themselves are **INHERITED** from `READ.json` / `score_<seed>.json` (taniteval/ci.py, seed 0) apart from that cross-check. Segment 3 (nav counterfactual) is MEASURED (ours) in v2 and regenerated here (controls V2). Segment 2 (LISTEN) and SFRESH numbers in segment 6 are **INHERITED** from their result files, each checked against the file's table in v2 and re-found in the source corpus here (control N4).
* Nothing is registered, promoted or retracted by this video; landing and the GOALS_AND_CLAIMS register are the Master Mind's step. 'One window is an example; the eval-wide numbers are the evidence.'

## 1. Controls (`viz3/CONTROLS_V3.json`): ALL_PASS = True

| control | result | tolerance |
|---|---|---|
| H1. HACC as drawn = P + w (.) (tau - P) from the banked weights (30 x 2) AND = the banked per-window plan `hacc/score_<seed>_perwindow.npz`; all 2,399 windows x 2 seeds + the 4 drawn windows x 2 seeds | max abs **0.0e+00 m** | <= 1e-5 m |
| H1 deliberate regressions (the check must be able to fail): axes of w swapped / w = 1 | 6.66 m / 20.43 m: both detected = True | > 1e-3 m |
| H2. ADE6 recomputed from the drawn plan and GT = the banked per-window ADE6, all 5 plans, 8 window-seed pairs; eval-wide per-window ADE1-6 means = the score JSON; B/H/P10 in `hacc/score_*.json` = `hblend/score_*.json` exactly; F = |ADE6(B s0) - ADE6(B s1)| = READ.json F | PASS = True | 1e-5 m (means: json rounding 5.1e-5) |
| N1. every number the scenes registered (1089) re-resolved from its JSON file in a fresh process, re-formatted, equal to the drawn string, and found in that frame's drawn text | 1089 / 1089 bound, 0 failures | exact string |
| N2. dominance table (15 rows x 2 seeds), accel-band rows, KEEP strings = `RESULT_V8F_HACC.md` text, verbatim on screen | 30 cells, 10 / 10 band rows, KEEP s0 / s1 on screen = {'s0': True, 's1': True} | exact string |
| N3. derived aggregates recomputed independently and found on screen: separated-better rows {'M2b_s0': 13, 'M2b_s1': 12}, separated-worse 0, max abs ADE1-5 HACC - H 0.0101 m, KEEP margin s1 0.0018 m, non-monotone steps {'M2b_s0': {'along': 4, 'cross': 18}, 'M2b_s1': {'along': 7, 'cross': 18}} | PASS = True | exact |
| N4. existence scan: every numeric token of every drawn string (6977 tokens, all 177 frames) is a declared constant or a value present in the source corpus | 0 unmatched | 0 |
| B1. the 4 primary HACC - P10 intervals (ADE2, ADE6, accel MAE, curvature MAE; both seeds) re-derived with an independent numpy bootstrap (B = 2,000, 141 episodes) | PASS = True; e.g. ADE6 s0 own -0.5034 [-0.6116, -0.4009] vs READ.json -0.5034 [-0.6073, -0.4050] | within 3 % of the width (different random stream) |
| V2. v2's controls still pass: CONTROLS_V2.json ALL_PASS as recorded = True, re-derived from its numbers = True; control 1 (own nav at S, T, O reproduces `forward_infer`) RE-RUN with v2's harness: max abs **0.0** over 8 window-seed pairs (exactly 0.0); the 96 counterfactual cells regenerated, max abs vs the drawn arrays 1.5e-05 m (<= 1e-4); structural zero 160 cells max 0.0; D6 recomputed from the drawn plans max 9.5e-07 m; v2 pack arrays unchanged 0.0; data copies byte-identical True; frame accessor pixel-identical to v2 on 8 frames; SELECTION_V2.md unchanged (sha256 c98f72aa8107dd92) | PASS = True | as v2 |
| W / S. no 'drives' / 'SOTA' drawn, 'closed loop' only inside 'NOT closed loop'; no 'paused' string and 'IN FORCE' drawn | PASS = True / True | - |

Control 3 of v2 (frame alignment, 404 / 404 frames, min PSNR 48.6 dB) is v1's `align_control` and applies unchanged: v3 draws frames with a byte-identical copy of the accessor (checked pixel-identical to v2's on every frame drawn).

## 2. Segment 3 (nav counterfactual) -- MEASURED (ours) in v2, unchanged; regenerated here (controls V2)

The numbers and tables below are v2's (`viz2/VIDEO_NOTES.md` section 2), sliced verbatim; the data file `viz3/data/cf_windows.npz` is a byte-identical copy of v2's and every drawn counterfactual array was regenerated by `viz3/controls_v2_rerun.py` (max abs 1.5e-5 m). The plans in this segment are the model's own plan B BEFORE the HACC post-map. Paths `viz2/data/...` in the slice are read-only originals; `viz3/data/` holds the copies the render reads.

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

## 3. Segment 2 (who listens to what) -- INHERITED from LISTEN, unchanged from v2

Source: `SC/listen/LISTEN_RESULT.md` (sections 2 and 4; `READ.json`; `fig_listen_grid.png` is shown as the grid). Tier T1-class open loop, donor-swap interventions; paired episode-cluster bootstrap B = 2,000, 141 episodes; both seeds M2b_s0 / M2b_s1. Numbers on screen (seed 0 / seed 1): S nav +4.197 / +3.985 m at 50 m, S scene +3.214 / +3.632 m, S all-inputs +8.057 / +8.375 m; T-lateral macro-F1 drop for S checkpoints +0.391 / +0.406, nav +0.173 / +0.216, scene -0.003 / +0.010 (not separated); T-longitudinal ego +0.247 / +0.261, scene +0.185 / +0.187; O ADE6 ego +5.697 / +4.716 m, T code +5.395 / +4.132 m, S checkpoints +0.873 / +1.894 m, nav +0.015 / +0.011 m (seed 1 not separated), all-inputs +1.394 / +1.315 m. Q-NAV (section 4): ADE6 change with the donor nav at S only +0.726 / +0.821, T only +0.476 / +0.521, O only +0.015 / +0.011, all three +1.174 / +1.180 m; D = T-only minus O-only +0.461 [+0.254, +0.687] / +0.510 [+0.273, +0.793], separated at both seeds.

## 4. Segment 4: better plans, now with HACC -- MEASURED (ours), T1-class open loop (own held outputs; NOT closed loop)

Artifacts: `viz3/data/plans_pack3.npz` (GT, B, P10, H, HS, HACC of the 4 windows x 2 seeds + per-step and ADE6; v2's arrays unchanged + HACC from `hacc/score_<seed>_perwindow.npz`; builder `viz3/plans_pack3.py`), `hacc/score_M2b_s{0,1}.json` (single, paired, consistency), `hspec/score_hs2_M2b_s{0,1}.json` (HS via `viz3/data/tables.json`). Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes (taniteval/ci.py); another draw of EPISODES only; training variance = the two seeds side by side; inference deterministic.

### 4a. ADE by horizon over all 2,399 windows (m; mean [95 % CI])

| seed | plan | 1 s | 2 s | 3 s | 4 s | 5 s | 6 s |
|---|---|---|---|---|---|---|---|
| 0 | B | 0.082 [0.071, 0.095] | 0.316 [0.277, 0.362] | 0.577 [0.525, 0.640] | 0.984 [0.908, 1.070] | 1.518 [1.403, 1.647] | 2.175 [2.007, 2.357] |
| 0 | H | 0.060 [0.051, 0.073] | 0.212 [0.192, 0.239] | 0.473 [0.432, 0.520] | 0.854 [0.786, 0.929] | 1.353 [1.246, 1.468] | 1.968 [1.811, 2.131] |
| 0 | HS | 0.064 [0.055, 0.077] | 0.217 [0.196, 0.244] | 0.478 [0.438, 0.525] | 0.855 [0.785, 0.932] | 1.354 [1.245, 1.470] | 1.976 [1.819, 2.140] |
| 0 | HACC | 0.061 [0.052, 0.074] | 0.211 [0.191, 0.238] | 0.473 [0.432, 0.519] | 0.852 [0.783, 0.928] | 1.354 [1.245, 1.470] | 1.984 [1.822, 2.151] |
| 0 | P10 | 0.063 [0.054, 0.076] | 0.222 [0.201, 0.249] | 0.534 [0.486, 0.587] | 1.013 [0.925, 1.107] | 1.663 [1.516, 1.811] | 2.487 [2.261, 2.713] |
| 1 | B | 0.071 [0.061, 0.083] | 0.291 [0.254, 0.340] | 0.552 [0.498, 0.618] | 0.936 [0.859, 1.020] | 1.440 [1.325, 1.563] | 2.055 [1.886, 2.232] |
| 1 | H | 0.060 [0.052, 0.073] | 0.214 [0.193, 0.240] | 0.481 [0.441, 0.527] | 0.866 [0.798, 0.940] | 1.369 [1.260, 1.483] | 1.985 [1.820, 2.159] |
| 1 | HS | 0.063 [0.054, 0.076] | 0.217 [0.196, 0.243] | 0.486 [0.446, 0.533] | 0.869 [0.799, 0.945] | 1.373 [1.263, 1.488] | 1.997 [1.833, 2.169] |
| 1 | HACC | 0.061 [0.052, 0.074] | 0.214 [0.193, 0.240] | 0.484 [0.444, 0.531] | 0.870 [0.802, 0.945] | 1.379 [1.268, 1.493] | 2.016 [1.850, 2.192] |
| 1 | P10 | 0.063 [0.054, 0.076] | 0.222 [0.201, 0.249] | 0.534 [0.486, 0.587] | 1.013 [0.925, 1.107] | 1.663 [1.516, 1.811] | 2.487 [2.261, 2.713] |

### 4b. Paired differences (m unless stated)

| seed | pair | ADE6 | accel MAE (m/s^2) | curvature MAE (1/m) | CONSISTENCY(a) 'any' (share) |
|---|---|---|---|---|---|
| 0 | HACC - P10 | -0.5034 [-0.6073, -0.4050] sep. | -0.0215 [-0.0345, -0.0087] sep. | -0.00560 [-0.00960, -0.00210] sep. | -0.0750 [-0.1017, -0.0496] sep. |
| 0 | HACC - H | +0.0151 [-0.0030, +0.0320] n.s. | -0.1253 [-0.1504, -0.1065] sep. | -0.01310 [-0.01960, -0.00820] sep. | -0.0242 [-0.0350, -0.0133] sep. |
| 0 | HACC - B | -0.1917 [-0.2759, -0.1139] sep. | -0.5897 [-0.6994, -0.4871] sep. | -0.04350 [-0.07100, -0.02520] sep. | +0.0763 [+0.0450, +0.1092] sep. |
| 1 | HACC - P10 | -0.4708 [-0.5695, -0.3748] sep. | -0.0402 [-0.0509, -0.0293] sep. | -0.00510 [-0.00830, -0.00230] sep. | -0.0463 [-0.0684, -0.0242] sep. |
| 1 | HACC - H | +0.0313 [+0.0026, +0.0586] sep. | -0.1062 [-0.1360, -0.0861] sep. | -0.01640 [-0.02470, -0.00930] sep. | -0.0204 [-0.0330, -0.0088] sep. |
| 1 | HACC - B | -0.0384 [-0.0948, +0.0137] n.s. | -0.3339 [-0.4234, -0.2660] sep. | -0.02420 [-0.03820, -0.01210] sep. | +0.0550 [+0.0317, +0.0779] sep. |

v2 rows kept for context (from `viz3/data/tables.json`): seed 0 H - P10 accel +0.104, ADE6 H - B -0.207; seed 1 H - P10 accel +0.066, ADE6 H - B -0.070.

### 4c. The four windows: ADE6 of each plan (m; seed 0 / seed 1), `viz3/data/plans_pack3.npz` (recomputed from the drawn plan and GT, control H2)

| window | P10 | B | H | HS | HACC |
|---|---|---|---|---|---|
| JUNCTION (#1131) | 4.46 / 4.46 | 2.66 / 1.58 | 1.09 / 1.24 | 0.79 / 1.00 | 1.30 / 1.39 |
| CURVE (#654) | 5.25 / 5.25 | 2.72 / 5.37 | 3.66 / 5.32 | 3.99 / 5.29 | 3.92 / 5.29 |
| NO-TURN (#763) | 1.75 / 1.75 | 1.39 / 2.15 | 1.42 / 2.08 | 1.44 / 2.04 | 1.47 / 2.02 |
| FAILURE(worst) (#2158) | 11.73 / 11.73 | 16.95 / 16.90 | 15.85 / 16.03 | 15.38 / 15.69 | 15.18 / 15.42 |

On the four windows HACC is not uniformly better (it is worse than P10 on FAILURE at both seeds and on CURVE at seed 1, and worse than H on JUNCTION and CURVE s0): the captions say so per window; one window is an example, the eval-wide rows are the evidence.

### 4d. The gap card: what HACC closes and what it does not (`hacc/score_<seed>.json`; both seeds on screen)

Closed vs P10: accel MAE and curvature MAE (both separated-better at both seeds, table 4b). Not better: CONSISTENCY(a) 'any' vs B (HACC - B +0.0763 / +0.0550 separated-worse), vs H on cross-track MAE (+0.0274 / +0.0138), FDE6 (+0.1195 / +0.1886) and, at seed 1, ADE6; distance keeping vs P10 not separated; the accel error at 5.2-5.8 s still above P10 (section 5c). Quoted from the JSON by the scene code (control N1).

## 5. Segment 5 (NEW): Dominance over the kinematic prior -- MEASURED (ours, `hacc/`), T1-class open loop (own held outputs; NOT closed loop)

### 5a. The dominance row, HACC - P10 (paired; `hacc/READ.json::per_seed.<seed>.PRIMARY_DOM.rows`; the same numbers verbatim in `hacc/RESULT_V8F_HACC.md` section 4, control N2)

Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes (taniteval/ci.py); another draw of EPISODES only; training variance = the two seeds side by side; inference deterministic. Outcome (PREREG outcome table, in order): **DOMINATES P10** (PRIMARY-DOM and PRIMARY-KEEP at both seeds). Rows: `*` = interval excludes 0.

| metric | requirement | seed 0: mean [95 % CI] | verdict s0 | seed 1: mean [95 % CI] | verdict s1 |
|---|---|---|---|---|---|
| ADE2 | separated-better | -0.0106 [-0.0150, -0.0056] | better, separated | -0.0083 [-0.0133, -0.0026] | better, separated |
| ADE6 | separated-better | -0.5034 [-0.6073, -0.4050] | better, separated | -0.4708 [-0.5695, -0.3748] | better, separated |
| accel MAE | separated-better | -0.0215 [-0.0345, -0.0087] | better, separated | -0.0402 [-0.0509, -0.0293] | better, separated |
| curvature MAE | separated-better | -0.00560 [-0.00960, -0.00210] | better, separated | -0.00510 [-0.00830, -0.00230] | better, separated |
| speed MAE | not separated-worse | -0.1132 [-0.1547, -0.0666] | better, separated | -0.1306 [-0.1646, -0.0934] | better, separated |
| along MAE | not separated-worse | -0.2323 [-0.3042, -0.1569] | better, separated | -0.2390 [-0.2981, -0.1754] | better, separated |
| cross-track MAE | not separated-worse | -0.4027 [-0.4906, -0.3173] | better, separated | -0.3344 [-0.4239, -0.2484] | better, separated |
| heading MAE | not separated-worse | -1.5548 [-1.9150, -1.2099] | better, separated | -1.4346 [-1.8027, -1.0956] | better, separated |
| yaw-rate MAE | not separated-worse | -0.8374 [-1.1086, -0.6108] | better, separated | -0.7927 [-1.0245, -0.5881] | better, separated |
| plan-derived lateral decision agreement | not separated-worse | +0.0809 [+0.0617, +0.1017] | better, separated | +0.0704 [+0.0496, +0.0917] | better, separated |
| plan-derived longitudinal decision agreement | not separated-worse | +0.0479 [+0.0117, +0.0821] | better, separated | +0.0604 [+0.0242, +0.0963] | better, separated |
| CONSISTENCY(a) 'any' | not separated-worse | -0.0750 [-0.1017, -0.0496] | better, separated | -0.0463 [-0.0684, -0.0242] | better, separated |
| DK headway min (m) | not separated-worse | +0.2132 [-0.0879, +0.5363] | not separated | +0.1986 [-0.1648, +0.5916] | not separated |
| DK time-gap min (s) | not separated-worse | +0.0702 [-0.0086, +0.1593] | not separated | +0.0457 [-0.0489, +0.1481] | not separated |
| DK LEAD-VIOL at-risk share | not separated-worse | -0.0353 [-0.0625, -0.0124] | better, separated | -0.0218 [-0.0487, +0.0000] | not separated |

DK rows (distance keeping; `hacc/dk_read.json`): only windows with a lead agent in frame; n windows / episodes: headway seed 0 724 / 75, seed 1 723 / 74; time-gap seed 0 662 / 74, seed 1 661 / 73; LEAD-VIOL at-risk seed 0 595 / 70, seed 1 595 / 70. (The 141-episode statement applies to the other rows; the DK rows bootstrap their own episode subsets.) Decision agreements are derived from the plan (plan-derived). Separated-better rows: seed 0 13 of 15, seed 1 12 of 15; separated-worse: 0.

### 5b. PRIMARY-KEEP (ADE6, HACC - H; bar = F/2 with the exact F)

| seed | HACC - H ADE6 [95 % CI] | bar (hi <=) | pass | lambda* (m*) |
|---|---|---|---|---|
| 0 | +0.0151 [-0.0030, +0.0320] | +0.06037 | True | 9,998 (4) |
| 1 | +0.0313 [+0.0026, +0.0586] | +0.06037 | True | 36,931 (16) |

F = |ADE6(B s0) - ADE6(B s1)| = 0.12074 m (`READ.json::F_exact_m`; reproduced from the score JSONs by control H2); the seed-1 margin to the bar is 0.00177 m (derived: bar - hi).

### 5c. The per-step accel-error profile (descriptive; unpaired per-cell means over the 2,399 windows; `hacc/accel_profile.json`, tied to the banked `fw_lon_accel_mae` by control 7 of RESULT_V8F_HACC.md: 9.5e-7 / 4.8e-7)

| seed | band | HACC | H | P10 | HACC - P10 | H - P10 |
|---|---|---|---|---|---|---|
| 0 | 0.8-1.6 s (cells 4-8) | 0.707 | 0.856 | 0.747 | -0.041 | +0.109 |
| 0 | 2.2-2.6 s (cells 11-13) | 0.643 | 1.107 | 0.670 | -0.027 | +0.437 |
| 0 | 2.8-5.0 s (cells 14-25) | 0.568 | 0.564 | 0.653 | -0.085 | -0.089 |
| 0 | 5.2-5.8 s (cells 26-29) | 0.799 | 1.091 | 0.565 | +0.233 | +0.526 |
| 0 | all 29 cells | 0.616 | 0.741 | 0.638 | -0.022 | +0.104 |
| 1 | 0.8-1.6 s (cells 4-8) | 0.714 | 0.811 | 0.747 | -0.034 | +0.064 |
| 1 | 2.2-2.6 s (cells 11-13) | 0.596 | 1.200 | 0.670 | -0.074 | +0.530 |
| 1 | 2.8-5.0 s (cells 14-25) | 0.578 | 0.585 | 0.653 | -0.075 | -0.068 |
| 1 | 5.2-5.8 s (cells 26-29) | 0.692 | 0.817 | 0.565 | +0.127 | +0.252 |
| 1 | all 29 cells | 0.597 | 0.704 | 0.638 | -0.040 | +0.066 |

Reading (on screen): H's gap to P10 sat at 0.8-1.6 s, 2.2-2.6 s and 5.2-5.8 s; HACC is below P10 in the first two (gone) and still above P10 at 5.2-5.8 s (reduced, NOT gone).

### 5d. The mechanism card (descriptive; two things changed together, no ablation separates them)

Weights: `hacc/weights_HACC_M2b_s{0,1}.json` and `hblend/weights_M2b_s{0,1}.json` (`w_along_by_step`, `w_cross_by_step`; 30 each); lambda* from `READ.json`; mean |tau - P| per step from `viz3/data/aux3.json` (built by `viz3/aux3.py` from `hblend/score_*_perwindow.npz`). HACC = P + w (.) (tau - P): w fitted on position error + lambda x the second-difference error, bounded [0, 1], no isotonic step; H: position error only, then isotonic. On screen: the along weight at steps 10 -> 14 (H seed 0 0.12 -> 0.50, seed 1 0.14 -> 0.59; HACC seed 0 0.20 -> 0.41, seed 1 0.22 -> 0.46), the counts of non-monotone steps, and the first cross-axis steps (mean |tau - P| at step 4 is small).

### 5e. The caveats card (all numbers from `READ.json`, control N1)

Seven items: tier; baseline = the programme's floor; estimator and the seed floor F; seed 1's thin KEEP margin; distance keeping not separated; only the plan moves (T's heads and S are the banked B arrays; HACC - H is exactly 0 on the strategic block, `score_<seed>.json::structural_identity`); not done here (HACC not folded into `forward_infer`; D-V8F-T1KIN-1 needs a PI go for the pooled eval cache).

## 6. Segment 6: the strategic layer at its own tick (DIAGNOSTIC) -- family IN FORCE for v8F

**Status correction vs v2:** the STRATEGIC family is IN FORCE for v8F (PI-V8F-2026-10-17; R72 / R73); the v2 text 'PAUSED' is stale. The diagnostics below are INHERITED (SFRESH; `sfresh/SFRESH_RESULT.md`, `sfresh_numbers.json`) and MEASURED per window (CPU re-run, v2); they do not change with HACC. S's checkpoint error IS the family's metric: at the deployed (held) ages S loses to the nav-only baseline at 50 m; ticked fresh it beats it.


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

S-age chart (NOW in the v3 render): `SC/sfresh/sfresh_extra.json::age_profile_50m`, held vs fresh by S age 0.0-1.2 s, both seeds.
Part of the held-vs-fresh gap is DEFINITIONAL: the banked eval scores S's checkpoints (predicted in the frame of S's tick) against labels in the NOW frame, so it pays for the distance driven since the tick. The deployed-schedule cost is the next strategic lever (not measured here).

## 6b. Frames and windows (unchanged from v2)

Frames: `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/{clip}.v2ep.pt`, cache step j = native frame 2j (cylindrical 256 x 640). Windows (eval index, clip, t in s): JUNCTION #1131 79e30ac2-c619-4d61-98a3-1007037b4d05 t=35; CURVE #654 45aea78c-7762-4e46-aef1-e9e22602b8fd t=31; NO-TURN #763 51c01ba4-9000-4370-8abb-9fd22b397d05 t=59; FAILURE(worst) #2158 ed87040c-cf3c-4fe4-9a95-50c1114f0d38 t=59.

## 7. Source files read by the render and the controls (sha256 of the bytes read)

| key | path | sha256 |
|---|---|---|
| hacc_read | `SC/hacc/READ.json` | `f006cfcde9b88603756294611d1ff3b8a7f0aace386304b235e62b0bc3f3fcef` |
| hacc_score_s0 | `SC/hacc/score_M2b_s0.json` | `7a878b164074000c02452c8b591e2ee57c2daa1e53564b1dd6baa2ebbd58045c` |
| hacc_score_s1 | `SC/hacc/score_M2b_s1.json` | `2111461ea7cc5fd9fe59836175ec5d0b4d5f2eee68b01c0f128558746d9b9ec9` |
| hacc_accel | `SC/hacc/accel_profile.json` | `fb0a489e06f53def62d7b5e03b7f5d898b78ce7dac3ee0679051c5c0fa4c7bc2` |
| hacc_dk | `SC/hacc/dk_read.json` | `8e22daae19df62daada735108896965e155f3f06165283b782d5b954f98f800c` |
| hacc_w_s0 | `SC/hacc/weights_HACC_M2b_s0.json` | `5b8e368fb06b6fea4ab710d071d2d94df58f1e28d4f0648ef6fab79ba664b1ed` |
| hacc_w_s1 | `SC/hacc/weights_HACC_M2b_s1.json` | `56f78bfefe6fed234bc44e1edb2a97cd47ac47f1a8c4b3ced8a69d1aded18469` |
| hblend_w_s0 | `SC/hblend/weights_M2b_s0.json` | `dd260b4eaf114454bdc2f0a55c9eb601fd62de4e4c13ec0dad590b4b30eeff9d` |
| hblend_w_s1 | `SC/hblend/weights_M2b_s1.json` | `987a0b5f07393e399d1482f96ca9a01c338386d952594776edb3426cc6d65227` |
| hspec_s0 | `SC/hspec/score_hs2_M2b_s0.json` | `efebf220fc1d3746c5d68baa65abf65279c5c9520a5254d741a9cc02df566baf` |
| hspec_s1 | `SC/hspec/score_hs2_M2b_s1.json` | `96dc21277b452ae41b9d758e591cd8805d0917bf540b21510eeb536433281fac` |
| sfresh | `SC/sfresh/sfresh_numbers.json` | `3c8081e499799b38816c8bc8ba56b91e0b28945a132241a31b195b6d953dde17` |
| sfresh_extra | `SC/sfresh/sfresh_extra.json` | `da2c0e5d5f3ca19cc940153b68af664cea55d8a1b2944ec6dd07107d56a77d22` |
| tables_v2 | `SC/viz3/data/tables.json` | `9fcf7c2246d1cad16675b4a1f60b47b514002ce862bfdee7ae01eb71443e8a15` |
| cf_evalwide | `SC/viz3/data/cf_evalwide.json` | `93d71399b84ee504abfe344ae8d5aaa68255c3c159ccb22403acbfd01c1c1535` |
| aux3 | `SC/viz3/data/aux3.json` | `bcd35deb72eb6b50f9c876e5c49217c9a6ef807c3c64d33471d57e7aab439909` |
| (file) | `SC/hacc/RESULT_V8F_HACC.md` | `6688a4f3a720cb323bbfa21b1c0f1824d4ec9e4061ad05399d0f276c675f7500` |
| (file) | `SC/hacc/PREREG_V8F_HACC.md` | `0128cac0c5bcdc62c967b1e47b9dbe54195ee82f83811a716b2666a401c1e554` |
| (file) | `SC/viz2/SELECTION_V2.md` | `c98f72aa8107dd9240da0bb05b3f32b6caa5487f1045e9ba746db036ad7a15a1` |
| (file) | `SC/viz2/CONTROLS_V2.json` | `fd47cd3f511ebeeae8e42d087629a65f41d0ca8db29503fdeda3fb5192bc257d` |
| (file) | `SC/listen/READ.json` | `453bc7100d0806c67fe5cefc936081fdd549c4e6cbb57b7769ca04c170a8bee0` |
| (file) | `SC/listen/fig_listen_grid.png` | `2b27d65a53f01a03a3b473ce6ed7797639e12099c8a74800cb8c5360f43fd842` |

Per-window plan arrays: `hblend/score_M2b_s{0,1}_perwindow.npz`, `hspec/score_hs2_M2b_s{0,1}_perwindow.npz`, `hacc/score_M2b_s{0,1}_perwindow.npz` (read by `plans_pack3.py` and, again, by the controls).

## 8. Manifest (all under `SC/viz3/`; nothing in git; list with bytes and raw-bytes git blob sha1 in `FILES.txt`)

`v8f_evidence_v3.mp4`, `CONTROLS_V3.json` (+ `CONTROLS_V3_part1.json`, `CONTROLS_V3_part2.json`), `VIDEO_NOTES_V3.md`, `qa_*.png` (6 stills), `video_chapters.json`, `FILES.txt`, `HANDOFF.md` (if present), `data/` (cf_windows.npz, cf_evalwide.json, tables.json, plans_pack3.npz, aux3.json, numreg.json, onscreen_text.json), scripts `v3_lib.py v3_draw.py v3_scenes.py build_video3.py plans_pack3.py aux3.py controls3.py controls_v2_rerun.py notes_gen3.py`.
