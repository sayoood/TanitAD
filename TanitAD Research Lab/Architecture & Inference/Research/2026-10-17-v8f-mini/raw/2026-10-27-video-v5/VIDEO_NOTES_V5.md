# v8F-mini evidence video v5 (HACC + T1-kin + DAgger) -- VIDEO_NOTES_V5 (chapter 8: every number, its artifact, evidence class, tier, estimator)

Written 2026-10-27 by the video5 subagent. Dev box, CPU only, ONE render process, nothing staged, committed or pushed. Work dir `E:/Projects/TanitAD-artifacts/video5/`. Video: `v8f_evidence_v5.mp4`, 536.2 s, 1920x1080, H.264, 10 fps, 9 segments, 622 unique frames. v4's notes are `v4_ref/VIDEO_NOTES_V4.md` (a copy, unchanged) and apply to segments 1-7.

## 0. Status and wording

* **Tier of chapter 8: T1-kin = self-action OPEN loop** (the DAG and DAG-0 runs `--limit causal --dump-ticks --dump-heads`, plain, replan every 2 s from the model's own executed motion; cameras = the human's logged frames). Never 'closed loop' (only inside 'NOT closed loop'), never 'drives' / 'autonomous' (control W). R-T0 is the guard tier: T0 teacher-forced own code. STRATEGIC is NOT paused for v8F: S-POS vs CTRV is shown beside and decides nothing.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows (`taniteval/ci.py`), DAG - DAG-0 per training seed: 'another draw of EPISODES' only. Another TRAINING run is answered by the two seeds and the replicate floor RF = |F1(DAG-0) - F1(ORC-0)| (ORC-0 is a same-argv replicate of DAG-0 trained by another build); the planner is deterministic, so no inference-seed question arises.
* **Evidence class: MEASURED (ours).** Every interval, bar, F1 and S-POS cell is a pointer into `data/DAG_READ.json` (sha256 `3393fdd56bd561fd5181a8219d2c75d700036b9373da6ec12a8bcba51decde3d`, the pre-registered read, PREREG_V8F_DAGGER.md at 2a78f059 incl. A1, A2; OUTCOME **ROBUST**), re-read by an independent pointer walker (control J1); the R-ADE and R-T0 intervals are re-derived from the float64 dumps / T0 arrays with the repo's estimator (control B1). Per-window numbers are computed from the float64 dumps (control A1). Per-class F1 is derived here from the label cells, beside, and is not in the read. Baseline = DAG-0, the same fine-tune without the lever; not a published state of the art; no external benchmark.
* **Where DAG looks worse (all on screen):** see section 5. Nothing is registered, promoted or retracted by this video; landing and the GOALS_AND_CLAIMS register are the Master Mind's step.

## 1. What changed against v4

* **New chapter 8** 'DAgger: trained on its own states', inserted before the closing (now chapter 9). Code: `v5_lib.py` (data layer), `v5_scenes.py` (scenes), `extract_dag_cells.py` (label cells via the programme's own reader), `select_scenes5.py`, `build_video5.py` (schedule), `controls5.py`.
* **Edited in segments 1 and 9, nothing else of 1-7:** (a) the header string of every frame says v5; (b) the title card has a sixth question and a compacted layout; (c) the closing card is rewritten: a fourth, full-width box carries chapter 8's numbers and what is still open. Control U1: every other frame of segments 1-7 is text-identical to v4's after the header swap; U2: v4's registered numbers are unchanged. v4's chapter-7 code (`v4_lib.py`, `v4_scenes.py`) is byte-identical to the banked v4 package.
* Segment timing (s): 1-7 as in v4 (0 - 372.4); **8 DAgger 372.4 - 526.2**; 9 closing 526.2 - 536.2.

## 2. Chapter 8 outline

| t (s, in the chapter) | scene | what is on screen |
|---|---|---|
| 0 - 14 | intro card | what DAgger changes (p = 0.1 of the training samples replaced by the model's own rolled-out states, human labels, M2b init, 3,000 steps, two seeds), the control DAG-0, what the viewer will see |
| 14 - 28 | rule card | the selection rule (verbatim, sha256 head), the seven windows with their stratum and delta, why they are not a sample |
| 28 - 108 | 7 windows x 11.4 s | per window: the logged camera frame, DAG and DAG-0 BEVs (seed 0) with the human's path (GT), each arm's EXECUTED path growing in real time (O, orange), S's three checkpoints of the current tick (pink diamonds) and the human's route checkpoints (hollow); under each BEV three tick boxes (+0, +2, +4 s) with S position error at 30/50/80 m and T's lateral / longitudinal decision (head argmax) against the label; at the end the per-window table (ADE6, path length, T right counts, both seeds) and a caption that says which arm is closer |
| 108 - 122 | distribution card | per-window DAG - DAG-0 ADE6 over all 2,399 windows, both seeds: shares, means (= the R-ADE point estimates), the seven windows marked |
| 122 - 138 | T-head card | forest of R-TLAT / R-TLON with the pre-registered bars, the F1 table, the OUTCOME; per-class F1 (derived, beside) |
| 138 - 154 | guards card | R-ADE, R-T0 vs the F/2 bar (forest), S-POS vs CTRV within each run, tick-0 skill, crawl fraction, what it is not |

## 3. Scene selection (`SELECTION_V5.md`, `select_scenes5.py`, `data/selection_v5.json`)

Written by the script BEFORE any chapter-8 frame was rendered (control S1 checks the file mtimes, re-derives the rows with independent code, and that the script reads only g / cl / ws / clip_index / eid). **Rule sha256 `ee759319dfafa149f8ed96da2d1893944569f9cdf5675399ebda9aac76d000c5`** (the string `RULE_TEXT` in the script); script sha256 `d3063aaeef21f352d8ed13eb6380f744e6e169f5f02cfde9d3598b09d3f189f3`.

> delta_w = ADE6(DAG, seed 0) - ADE6(DAG-0, seed 0) of window w, T1-kin, causal, m = 10 (negative: DAG closer to the human's path), over the 2,399 windows in the dump order.
> 1. For each GT stratum (JUNCTION, CURVE, NO-TURN; window_strata_eval141.npz, defined from the human's labels) sort the stratum's windows by (|delta_w|, grid row) ascending and take position (n - 1) // 2: the lower median of |delta|, the typical size of change in that stratum.
> 2. WORST: the window with the largest delta_w over all 2,399 windows (ties: lowest grid row).
> 3. Three DRAWS: numpy.random.default_rng(20261027).choice(n_cand, 3, replace=False) over the windows (dump order) with GT mean speed >= 5 m/s that are not already chosen by 1 or 2.
> No window is swapped or dropped for how it looks; a window where DAG is worse stays in.

| # | tag | how chosen | stratum | eval window (grid row) | clip | clip_index | ws | GT mean speed (m/s) | delta s0 (m) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | JUNCTION | median abs(delta) of stratum JUNCTION (n = 298) | JUNCTION | #496 | 37e61793 | 29 | 11 | 5.56 | +0.6526 |
| 2 | CURVE | median abs(delta) of stratum CURVE (n = 140) | CURVE | #944 | 63434e0b | 55 | 35 | 14.06 | -0.3069 |
| 3 | NO-TURN | median abs(delta) of stratum NO-TURN (n = 1961) | NO-TURN | #59 | 10804db8 | 3 | 35 | 4.60 | +0.4664 |
| 4 | WORST | largest delta over all windows | NO-TURN | #1795 | c8a39711 | 105 | 39 | 6.49 | +8.8633 |
| 5 | DRAW1 | default_rng(20261027).choice | NO-TURN | #1715 | b99058dc | 100 | 59 | 9.60 | -0.6531 |
| 6 | DRAW2 | default_rng(20261027).choice | NO-TURN | #2344 | fb8e8200 | 137 | 55 | 22.47 | -2.6123 |
| 7 | DRAW3 | default_rng(20261027).choice | NO-TURN | #526 | 38fe80c1 | 30 | 63 | 14.15 | +0.3004 |

## 4. The numbers on the cards (tier T1-kin unless stated; estimator paired episode-cluster bootstrap, B = 2,000, 141 episodes; seeds s0 / s1; all from `data/DAG_READ.json`)

| card | quantity (DAG - DAG-0) | s0 | s1 | pointer |
|---|---|---|---|---|
| T heads | lateral head macro-F1 (R-TLAT) | +0.1082 [+0.0491, +0.1572], bar 0.0248, F1 0.6029 vs 0.4947, clears True | +0.1184 [+0.0755, +0.1585], bar 0.0248, F1 0.6385 vs 0.5201, clears True | rows/s*/R-TLAT/* |
| T heads | longitudinal head macro-F1 (R-TLON) | +0.0554 [+0.0300, +0.0847], bar 0.0502, F1 0.3213 vs 0.2659, clears True | +0.0806 [+0.0560, +0.1043], bar 0.0278, F1 0.3311 vs 0.2505, clears True | rows/s*/R-TLON/* |
| guards | T1-kin ADE6 (m), guard R-ADE | −0.5435 [−0.7564, −0.3690], means 2.278 vs 2.821 | −0.4046 [−0.5496, −0.2782], means 2.456 vs 2.861 | rows/s*/R-ADE/* |
| guards | T0 own-code ADE6 (m), guard R-T0 | +0.0019 [−0.0230, +0.0266], means 1.974 vs 1.972 | +0.0483 [+0.0178, +0.0807], means 2.098 vs 2.050 | rows/s*/R-T0/* |
| guards | F/2 bar for R-T0 (m) | 0.0604 | same | bars/F_over_2_t0_point_m |
| (OUTCOME) | ROBUST: R-TLAT and R-TLON both clear (§5) at both seeds; R-ADE not separated-worse at either seed; R-T0 ≤ +F/2 at both | guards_held True | | OUTCOME/* |

S-POS vs CTRV (within each run, ticks >= 1, unpaired, beside, decides nothing), delta (m) [lo, hi] and the read's `separated` flag:

| run, arc | s0 | s1 |
|---|---|---|
| DAG @30 m | -1.72 [-2.27, -1.28] separated True | -1.87 [-2.36, -1.44] separated True |
| DAG @50 m | -4.48 [-5.63, -3.50] separated True | -4.90 [-5.96, -3.92] separated True |
| DAG @80 m | -9.69 [-11.84, -7.57] separated True | -10.33 [-12.44, -8.35] separated True |
| DAG0 @30 m | +0.12 [-0.27, +0.49] separated False | +0.81 [+0.35, +1.29] separated True |
| DAG0 @50 m | -0.91 [-1.97, +0.07] separated False | +0.33 [-0.81, +1.46] separated False |
| DAG0 @80 m | -3.89 [-6.09, -1.85] separated True | -1.95 [-4.37, +0.49] separated False |

Per-window values (the end frame of each window; ADE6 in m, T1-kin, no interval), from `data/numreg.json` (kinds ade6 / ade6d, recomputed by control A1 from the float64 dumps):

| # | tag | s0 DAG | s0 DAG-0 | s0 delta | s1 DAG | s1 DAG-0 | s1 delta |
|---|---|---|---|---|---|---|---|
| 1 | JUNCTION | 2.95 | 2.30 | +0.65 | 2.24 | 2.44 | −0.20 |
| 2 | CURVE | 1.69 | 1.99 | −0.31 | 2.01 | 1.10 | +0.90 |
| 3 | NO-TURN | 1.04 | 0.58 | +0.47 | 0.53 | 1.21 | −0.69 |
| 4 | WORST | 10.81 | 1.95 | +8.86 | 6.15 | 4.11 | +2.05 |
| 5 | DRAW1 | 1.16 | 1.81 | −0.65 | 2.31 | 2.54 | −0.23 |
| 6 | DRAW2 | 1.58 | 4.20 | −2.61 | 1.97 | 1.98 | −0.01 |
| 7 | DRAW3 | 2.19 | 1.89 | +0.30 | 1.95 | 1.68 | +0.26 |

Source dumps (read-only): `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_{DAG,DAG0}_s{0,1}_m10_causal_heads/ep*.npz` (keys g, cl), per-tick files `ticks/ticks_*_m10_causal_heads.npz` (sha256 in `data/DAG_READ.json` `runs`), run manifest `runs.json`. Label cells: `data/dag_cells_*.npz` from `extract_dag_cells.py` (imports the programme's own `t1hd_read.py`, sha256 `3c6d9f790f855115...`, from the pinned tool tree `E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a`), manifest `data/dag_cells_manifest.json`.

## 5. Where DAG looks worse (all of it on screen)

* **Windows (seed 0, ADE6 DAG - DAG-0 > 0): JUNCTION +0.65 m, NO-TURN +0.47 m, WORST +8.86 m, DRAW3 +0.30 m** (4 of 7). The WORST window (#1795, NO-TURN) is DAG +8.86 m at seed 0 and +2.05 m at seed 1: DAG executes a 73 m path where the human travels 39 m (the human stops: label STOP; DAG's longitudinal head says DECEL, DAG-0's says STOP at tick 1).
* **Distribution:** DAG wins about two thirds of the windows at both seeds and loses a third (histogram card); the mean difference is the R-ADE point (guard held, both seeds separated BETTER).
* **R-T0 at seed 1 is separated WORSE** (+0.0483 m [+0.0178, +0.0807]) but under the pre-registered F/2 bar of 0.0604 m: the guard held, the cost is on the card.
* **Tick 0 (the logged state):** DAG's macro-F1 is lower than DAG-0's in all four numbers (lat 0.8469 vs 0.8504 / 0.8574 vs 0.8621, lon 0.4656 vs 0.4709 / 0.4835 vs 0.5111; no interval in the read).
* **Still open:** images stay logged, so lateral recovery is not taught (PREREG section 3); one bank pass and one fine-tune of 3,000 steps per arm; the crawl rows M1 / L2 are carried per run, not paired; per-turn-class F1 and the four families at T0 and T1 have no instrument in the read (per-class F1 on the card is derived here, beside); the HACC curvature gap of chapter 7 is not re-read for DAG.

## 6. Controls (`CONTROLS_V5.json`)

ALL_PASS before the render = **True**; post-render controls done = **True**; ALL_PASS = **True**.

| control | result |
|---|---|
| T1. every plotted line (GT, executed path, S, S GT; both arms) = the raw arrays, independent loader | 217 window frames, 1736 lines, max abs **0.0e+00 m**, PASS = True (tolerance 1e-6 m) |
| A1. per-window numbers on screen = recomputed from the raw arrays | 2828 numbers {'serr': 882, 'tcell': 1764, 'ade6': 56, 'ade6d': 28, 'plen': 42, 'tcount': 56}, 0 failures, PASS = True |
| L1. label chain over all windows x 3 ticks x 4 runs x 2 heads: pred = argmax of the tick file's logits; true = the join class at the matched route row with the on-route cut (own formula) | 8 (run, head) pairs, failures [], PASS = True |
| F1. own macro-F1 of the extracted cells = the read's own-cell F1 (8 numbers) and matched-cell F1 (4 rows x 2) | max abs diff 0.0e+00 over 12 rows, PASS = True (tolerance 1e-9) |
| S1. S error = abs(S checkpoint - S GT) for all windows (4 runs); selection re-derived from the raw dumps with own code = `selection_v5.json`; script reads only g / cl / ws / clip_index / eid; selection files older than every chapter-8 artifact; rule sha256 matches the text | max abs diff 0.0e+00 m, rows [496, 944, 59, 1795, 1715, 2344, 526] = [496, 944, 59, 1795, 1715, 2344, 526], PASS = True |
| H1. histogram counts, shares, means recomputed from the dumps; each mean = the R-ADE delta (4 decimals) | 2 panels, 0 failures, PASS = True |
| J1. every DAG_READ pointer on screen re-read by own code, value in the frame's drawn text | 174 items {'dagj': 135, 'argv': 4, 'dagb': 17, 'dagt': 6, 'dagw': 12}, 0 failures, PASS = True |
| B1. R-ADE and R-T0 intervals re-derived with the repo's estimator (B = 2,000, seed 0) = the read's cells | R-ADE s0 [-0.5435, -0.7564, -0.369]; R-T0 s0 [0.0019, -0.023, 0.0266]; R-ADE s1 [-0.4046, -0.5496, -0.2782]; R-T0 s1 [0.0483, 0.0178, 0.0807]; PASS = True |
| N4. every numeric token of every chapter-8 drawn string (and the closing card) is registered, in the selection, or a declared constant | 20219 tokens, 0 unmatched, PASS = True |
| W. no 'drives' / 'driving' / 'autonomous' / 'SOTA'; 'closed loop' only inside 'NOT closed loop'; T1-kin on every chapter-8 frame; estimator and seed on the cards | PASS = True |
| G1. every window ends inside its clip | PASS = True |
| **X. deliberate regression: DAG and DAG-0 arrays of scene WORST swapped at the data-access layer (`V5_SWAP_ARMS`)** | T1 FAILED as it must (14 line failures, up to 28.85 m), A1 FAILED as it must (42 number failures); regression arm PASS = True. Still: `regression_swap_WORST_end.png` (labels say DAG, the data are DAG-0's) |
| V1. video: codec / size / frame count | h264 1920x1080, 5362 frames (expected 5362), PASS = True |
| P1. the full render's drawn text = the control render's, chapters 8-9 | 223 scenes, PASS = True |
| P2. the plotted arrays of the video's frames = the controlled ones (sha1 per frame) | 217 frames, PASS = True |
| U1. segments 1-7 text-identical to v4 after the header swap, except the title card | 398 of 399 identical, differing = ['title_card()'], PASS = True |
| U2. v4's registered numbers (minus closing / title) unchanged | 1318 items, PASS = True |

## 7. Re-run

```
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe ; export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1   # cwd = E:/Projects/TanitAD-artifacts/video5/
$PY -I extract_dag_cells.py                  # data/dag_cells_*.npz (pinned tool tree, the programme's own reader)
$PY -I select_scenes5.py                     # SELECTION_V5.md + data/selection_v5.json
$PY -I controls5.py --main ; V5_SWAP_ARMS=WORST $PY -I controls5.py --swap
$PY -I build_video5.py [--qa-only --tags 08a_intro,...]   # video + data/onscreen_text.json, numreg.json, ch78_lines_digest.json
$PY -I controls5.py --post                   # CONTROLS_V5.json
```

Inputs outside this dir (read-only): the scratchpad dirs `viz/`, `viz2/`, `hacc/`, `hblend/`, `hspec/`, `listen/` of the v3/v4 pipeline (chapters 1-6), `t1kc_run/` + `t1ks_run/` (chapter-7 dumps), `tanitad-data/refav1-eval141-256rb/` (frames). Chapter-7's selection (`SELECTION_V4.md`, `select_scenes.py`) is in `v4_ref/` and the v4 package.
