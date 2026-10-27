# v8F-mini evidence video v4 (HACC + T1-kin) -- VIDEO_NOTES_V4 (chapter 7: every number, its artifact, evidence class, tier, estimator)

Written 2026-10-26 by the viz4 subagent. Dev box, CPU only, ONE render process, nothing staged, committed or pushed. All paths are under `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/` (`SC`). Video: `viz4/v8f_evidence_v4.mp4`, 381.4 s, 1920x1080, H.264, 10 fps, 8 segments, 400 unique frames. v3's notes are `viz4/VIDEO_NOTES_V3.md` (a copy, unchanged) and apply to segments 1-6.

## 0. Status and wording

* **Tier of chapter 7: T1-kin = self-action OPEN loop.** Every 2 s (m = 10 steps of 0.2 s) the model replans from its own executed motion; the cameras are the human's logged frames (CAUSAL) or frozen at the first frame (STRICT). P10 is run the same way in its own loop. Never 'closed loop' (only inside the mandated 'NOT closed loop'), never 'drives' / 'autonomous' (control W). Segments 1-6 keep v3's tier string 'T1-class open loop (own held outputs)'.
* **Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows (`taniteval/ci.py`); it answers 'another draw of EPISODES'. Training variance = the two seeds side by side (s0, s1); a separated interval from a one-seed arm is necessary, not sufficient (H-ESTIM-SEED-1), and here both seeds agree.
* **Evidence class: MEASURED (ours).** Per-window values and the histogram are computed here from the float64 dumps; the paired intervals and the other table cells are copied verbatim from the four RESULT files (control R1) and the four primary intervals are re-derived from the dumps with the repo's estimator (control B1). Baseline = the programme's own floor P10, not a published state of the art; no external benchmark.
* **The open gap is on screen** (lateral card): curvature at GT speed >= 5.2 m/s is separated-worse than the floor's at both seeds; the canonical curvature / yaw-rate rows are worse too and noise-dominated below 5 m/s. T1-kin has no tactical / strategic / distance-keeping rows (NOT COMPUTED in the RESULT files): shown as a work item, never as a pass.
* Nothing is registered, promoted or retracted by this video; landing and the GOALS_AND_CLAIMS register are the Master Mind's step.

## 1. What changed against v3

* **New chapter 7** 'T1-kin: the model on its own motion', inserted before the closing (now chapter 8). Code: `v4_lib.py` (data layer), `v4_scenes.py` (scenes), `build_video4.py` (schedule).
* **Edited in segments 1 and 8, nothing else of 1-6:** (a) the header string of every frame says v4; (b) the title card has a fifth question and a compacted layout; (c) the closing card is rewritten: v3 said 'Any tier above this one: not run (D-V8F-T1KIN-1 needs a PI go ...)', which the T1-kin result makes stale, so it now carries the T1-kin numbers and the open lateral gap. Control U1 shows every other frame of segments 1-6 is text-identical to v3's after the header swap; control U2 that v3's registered numbers are unchanged.
* Segment timing (s): 1-6 as in v3 (0 - 206.6); **7 T1-kin 206.6 - 372.4**; 8 closing 372.4 - 381.4.

## 2. Chapter 7 outline

| t (s, in the chapter) | scene | what is on screen |
|---|---|---|
| 0 - 14 | intro card | what T1-kin is, the two configurations, what the viewer will see, tier / estimator / baseline |
| 14 - 28 | rule card | the scene-selection rule (stated on screen), the seven windows, why they are not a sample of the typical window |
| 28 - 108 | 7 windows x 11.4 s | per window: logged camera frame (CAUSAL) and the first frame held (STRICT); two BEVs (CAUSAL, STRICT; seed 0) of GT, HACC's executed track and P10's executed track growing in real time (one knot = 0.2 s), rings at the 2 s and 4 s replan ticks; at the end the per-window ADE6 table (both runs, both seeds) and a caption that says where the floor is better |
| 108 - 122 | distribution card | per-window HACC - P10 ADE6 over all 2,399 windows: 4 histograms (CAUSAL / STRICT x s0 / s1), share of windows each side wins, the seven windows marked |
| 122 - 136 | result card CAUSAL | forest plot of the primary row, the nine-row table (LONG / LAT families), mean ADE6, what the causal bucket costs, what it is not |
| 136 - 150 | result card STRICT | the same for cameras frozen + causal bucket |
| 150 - 166 | lateral-gap card | L2 curvature at speed (WORSE both seeds), L1 lateral acceleration (not worse), L3 yaw rate, heading, cross-track, canonical curvature; the noise-floor chart; the next step |

## 3. Scene selection (`SELECTION_V4.md`, `select_scenes.py`, `data/selection_v4.json`)

Fixed before any per-window T1 number was read: `select_scenes.py` loads only `g`, `ws`, `clip_index` (control S1 checks the source, the file mtimes, and re-derives the rows).

1. The four v3 windows (JUNCTION #1131, CURVE #654, NO-TURN #763, FAILURE #2158), if in the T1 grid: all four are, and the grid row equals the v3 window id.
2. Three windows from `numpy.random.default_rng(20261026).choice(1907, 3, replace=False)` over the T1-grid windows (dump order) with GT mean speed >= 5 m/s (1,911 of 2,399), excluding the four of rule 1. Draw positions [705, 1153, 1499].

| # | tag | eval window (grid row) | clip | clip_index | ws (cache step) | GT mean speed (m/s) |
|---|---|---|---|---|---|---|
| 1 | JUNCTION | #1131 | 79e30ac2 | 66 | 35 | 7.43 |
| 2 | CURVE | #654 | 45aea78c | 38 | 31 | 11.63 |
| 3 | NO-TURN | #763 | 51c01ba4 | 44 | 59 | 18.50 |
| 4 | FAILURE(worst) | #2158 | ed87040c | 126 | 59 | 11.50 |
| 5 | DRAW1 | #834 | 59c25a0f | 49 | 3 | 36.50 |
| 6 | DRAW2 | #1400 | 9b4347d2 | 82 | 23 | 11.28 |
| 7 | DRAW3 | #1844 | d1d382aa | 108 | 27 | 15.29 |

No window was picked, swapped or dropped for how the model looks on it. Per-window result (ADE6, m): see the end frame of each window; the FAILURE window is worse than the floor in all four (run, seed) cells and CURVE in two of four; both stay in the video.

## 4. The numbers on the result cards (verbatim from the RESULT files; tier T1-kin; estimator paired episode-cluster bootstrap, B = 2,000, 141 episodes; seeds s0 / s1)

| card | quantity (HACC - P10, lower is better) | s0 | s1 | source (git blob sha in `data/results/`) |
|---|---|---|---|---|
| causal | ADE6 (m), PRIMARY | −0.7680 [−0.9177, −0.6258] | −0.7336 [−0.8776, −0.5984] | `RESULT_V8F_T1KIN_CAUSALBUCKET.md` (c2056a60) |
| causal | curvature MAE (1/m), canonical row | +0.0302 [+0.0188, +0.0442] WORSE | +0.0130 [+0.0057, +0.0217] WORSE | `RESULT_V8F_T1KIN_CAUSALBUCKET.md` (c2056a60) |
| strict | ADE6 (m), PRIMARY | −0.7278 [−0.8680, −0.5942] | −0.6413 [−0.7696, −0.5190] | `RESULT_V8F_T1KIN_STRICT.md` (fc1e8cf3) |
| strict | curvature MAE (1/m), canonical row | +0.032, worse | +0.014, worse | `RESULT_V8F_T1KIN_STRICT.md` (fc1e8cf3) |
| lateral | L2 curvature MAE at GT >= 5.2 m/s (1/m) | +0.0210 [+0.0099, +0.0351] WORSE | +0.0082 [+0.0026, +0.0148] WORSE | `RESULT_V8F_T1_LATCOND.md` (b8a15b84) |
| lateral | L1 lateral-acceleration MAE (m/s^2) | −0.107 [−0.606, +0.388], n.s. | −0.115 [−0.703, +0.487], n.s. | `RESULT_V8F_T1_LATCOND.md` (b8a15b84) |
| lateral | L3 yaw-rate MAE at GT >= 5.2 m/s (deg/s) | +1.98 [+0.34, +3.88] WORSE | +1.01 [−0.30, +2.35], n.s. | `RESULT_V8F_T1_LATCOND.md` (b8a15b84) |

The other table rows (ADE2, speed, along-track, accel, heading, yaw rate, cross-track) are copied the same way and registered (control R1). Mean ADE6 (CAUSAL 2.675 / 2.710, STRICT 2.716 / 2.802, P10 3.443 m) and the other sentences are verbatim snippets of the RESULT files.

Source commit of the RESULT files: `abc12d3f` (`git -C C:/Users/Admin/tanitad-push -c safe.directory=* show abc12d3f:"TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/<name>"`); the saved copies in `viz4/data/results/` hash to the blob shas above (checked: `git hash-object`).
Dumps (read-only): `C:/Users/Admin/t1kc_run/dump_HACC_s{0,1}_m10_causal/ep*.npz`, `C:/Users/Admin/t1ks_run/dump_HACC_s{0,1}_m10_strict/ep*.npz` (141 files each; keys g, cl, fl, ol, ha0, ws, eid, clip_index, v0). Their float64 originals are also in `E:/Projects/TanitAD-artifacts/v8f-mini-hacc/t1kin_causalbucket_dumps_f64_2026-10-26.tar` and `t1kin_strict_dumps_f64_2026-10-26.tar`. Join: `C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz`. Frames: `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/<clip>.v2ep.pt` via v3's accessor.

## 5. Controls (`CONTROLS_V4.json`)

ALL_PASS before the render = **True**; post-render controls done = **True**; ALL_PASS (needs the post-render controls) = **True**.

| control | result | tolerance |
|---|---|---|
| T1. every plotted track (the matplotlib artists of the figure, read back) = the raw dump arrays, from an independent loader | 217 scene frames x 6 lines = 1302 lines, max abs **0.0e+00 m**, PASS = True | <= 1e-6 m |
| A1. per-window ADE6 and deltas on screen = recomputed from the raw dumps, string-equal and in the frame's text | 140 numbers, 0 failures, PASS = True | exact string |
| H1. histogram counts, shares and means recomputed from all 141 episode files; each mean = the RESULT point estimate | 4 panels, 0 failures, PASS = True | exact |
| R1. every RESULT cell / snippet a scene shows = the cell of the file re-parsed here, and its value is in the frame's text | 77 cells (65 shown in full), 14 snippets, 0 failures, PASS = True | exact string |
| B1. the four primary intervals re-derived from the float64 dumps with the repo's estimator (B = 2,000, seed 0) = the RESULT cells | causal s0 −0.7680 [−0.9177, −0.6258]; causal s1 −0.7336 [−0.8776, −0.5984]; strict s0 −0.7278 [−0.8680, −0.5942]; strict s1 −0.6413 [−0.7696, −0.5190] ; PASS = True | exact 4-decimal string |
| N4. every numeric token of every chapter-7 drawn string (and the closing card) is in a RESULT file, a registered number, the selection, or a declared constant | 10273 tokens, 0 unmatched, PASS = True | 0 |
| W. no 'drives' / 'driving' / 'autonomous' / 'SOTA'; 'closed loop' only inside 'NOT closed loop'; T1-kin on every chapter-7 frame; estimator and seed on every card | PASS = True | - |
| S1. selection re-derived from the raw GT = `selection_v4.json`; `select_scenes.py` reads only g / ws / clip_index; selection file older than the data layer; v3 windows in the grid | PASS = True | exact |
| G1. chapter-7 frame accessor pixel-identical to v3's on the four v3 windows; dump GT = v3's `cf_windows` GT of the same window | max GT diff 3.8e-06 m (float32 source), PASS = True | 1e-3 m |
| F1. the floor's executed tracks are identical in all four dump sets | max abs 0.0 m, PASS = True | 0 |
| **X. deliberate regression: cl and fl of scene JUNCTION swapped at the data-access layer (`V4_SWAP_CLFL`)** | T1 FAILED as it must (12 line failures, tracks off by up to 13.02 m), A1 FAILED as it must (20 number failures); regression arm PASS = True. Still: `viz4/regression_swap_JUNCTION_end.png` (labels say HACC, the data are the floor's) | both must fail |
| V1. video: codec / size / frame count | h264 1920x1080, 3814 frames (expected 3814), PASS = True | exact |
| P1. the full render's drawn text = the control render's, chapters 7-8 | PASS = True | exact |
| P2. the plotted arrays of the video's frames = the controlled ones (sha1 per frame) | 223 frames, PASS = True | sha1 |
| U1. segments 1-6 text-identical to v3 after the header swap, except the title card | 175 of 176 identical, differing = ['title_card()'], PASS = True | exact |
| U2. v3's registered numbers (1,089 minus the closing card) unchanged | PASS = True | exact |

## 6. Re-run

```
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe ; export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1   # cwd = viz4/
$PY select_scenes.py                       # SELECTION_V4.md (GT only)
$PY build_video4.py [--qa-only --tags 07d_hist,...]   # video + data/onscreen_text.json, numreg.json, ch7_lines_digest.json
$PY controls4.py --main ; V4_SWAP_CLFL=JUNCTION $PY controls4.py --swap ; $PY controls4.py --post   # CONTROLS_V4.json
```

