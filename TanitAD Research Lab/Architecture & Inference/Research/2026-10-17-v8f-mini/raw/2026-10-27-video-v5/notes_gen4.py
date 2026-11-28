"""notes_gen4.py -- writes VIDEO_NOTES_V4.md from CONTROLS_V4*.json, video_chapters.json, SELECTION_V4.md and the RESULT files (so the notes cannot drift from the controls).  CPU only."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
C = json.load(open(f"{HERE}/CONTROLS_V4.json"))
M = C["main"]["controls"]
X = C["X_regression_arm_swap_cl_fl"]
CH = json.load(open(f"{HERE}/video_chapters.json"))
SEL = json.load(open(f"{HERE}/data/selection_v4.json"))
post = C.get("post_render")


def rf(n):
    return open(f"{HERE}/data/results/{n}.md", "rb").read().decode("utf-8")


def cell(name, key, col):
    for line in rf(name).split("\n"):
        if line.startswith("|"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if key in c[0]:
                return c[col].replace("**", "").replace("\u2705", "").strip()
    raise KeyError((name, key))


chs = {c["name"]: c["t_s"] for c in CH["chapters"]}
t7 = chs["7 T1-kin: the model on its own motion"]
t8 = chs["8 closing"]
T = lambda k: M[k]
L = []
w = L.append
w("# v8F-mini evidence video v4 (HACC + T1-kin) -- VIDEO_NOTES_V4 (chapter 7: every number, its artifact, evidence class, tier, estimator)")
w("")
w("Written 2026-10-26 by the viz4 subagent. Dev box, CPU only, ONE render process, nothing staged, committed or pushed. All paths are under "
  "`C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/` (`SC`). Video: `viz4/v8f_evidence_v4.mp4`, "
  f"{CH['duration_s']:.1f} s, 1920x1080, H.264, 10 fps, 8 segments, {CH['unique_frames']} unique frames. v3's notes are `viz4/VIDEO_NOTES_V3.md` (a copy, unchanged) and apply to segments 1-6.")
w("")
w("## 0. Status and wording")
w("")
w("* **Tier of chapter 7: T1-kin = self-action OPEN loop.** Every 2 s (m = 10 steps of 0.2 s) the model replans from its own executed motion; the cameras are the human's logged frames (CAUSAL) or frozen at the first frame (STRICT). "
  "P10 is run the same way in its own loop. Never 'closed loop' (only inside the mandated 'NOT closed loop'), never 'drives' / 'autonomous' (control W). Segments 1-6 keep v3's tier string 'T1-class open loop (own held outputs)'.")
w("* **Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows (`taniteval/ci.py`); it answers 'another draw of EPISODES'. Training variance = the two seeds side by side (s0, s1); a separated interval from a one-seed arm is necessary, not sufficient (H-ESTIM-SEED-1), and here both seeds agree.")
w("* **Evidence class: MEASURED (ours).** Per-window values and the histogram are computed here from the float64 dumps; the paired intervals and the other table cells are copied verbatim from the four RESULT files "
  "(control R1) and the four primary intervals are re-derived from the dumps with the repo's estimator (control B1). Baseline = the programme's own floor P10, not a published state of the art; no external benchmark.")
w("* **The open gap is on screen** (lateral card): curvature at GT speed >= 5.2 m/s is separated-worse than the floor's at both seeds; the canonical curvature / yaw-rate rows are worse too and noise-dominated below 5 m/s. T1-kin has no tactical / strategic / distance-keeping rows (NOT COMPUTED in the RESULT files): shown as a work item, never as a pass.")
w("* Nothing is registered, promoted or retracted by this video; landing and the GOALS_AND_CLAIMS register are the Master Mind's step.")
w("")
w("## 1. What changed against v3")
w("")
w("* **New chapter 7** 'T1-kin: the model on its own motion', inserted before the closing (now chapter 8). Code: `v4_lib.py` (data layer), `v4_scenes.py` (scenes), `build_video4.py` (schedule).")
w("* **Edited in segments 1 and 8, nothing else of 1-6:** (a) the header string of every frame says v4; (b) the title card has a fifth question and a compacted layout; (c) the closing card is rewritten: v3 said 'Any tier above this one: not run (D-V8F-T1KIN-1 needs a PI go ...)', which the T1-kin result makes stale, "
  "so it now carries the T1-kin numbers and the open lateral gap. Control U1 shows every other frame of segments 1-6 is text-identical to v3's after the header swap; control U2 that v3's registered numbers are unchanged.")
w(f"* Segment timing (s): 1-6 as in v3 (0 - {t7:.1f}); **7 T1-kin {t7:.1f} - {t8:.1f}**; 8 closing {t8:.1f} - {CH['duration_s']:.1f}.")
w("")
w("## 2. Chapter 7 outline")
w("")
w("| t (s, in the chapter) | scene | what is on screen |")
w("|---|---|---|")
w("| 0 - 14 | intro card | what T1-kin is, the two configurations, what the viewer will see, tier / estimator / baseline |")
w("| 14 - 28 | rule card | the scene-selection rule (stated on screen), the seven windows, why they are not a sample of the typical window |")
w("| 28 - 108 | 7 windows x 11.4 s | per window: logged camera frame (CAUSAL) and the first frame held (STRICT); two BEVs (CAUSAL, STRICT; seed 0) of GT, HACC's executed track and P10's executed track growing in real time (one knot = 0.2 s), rings at the 2 s and 4 s replan ticks; "
  "at the end the per-window ADE6 table (both runs, both seeds) and a caption that says where the floor is better |")
w("| 108 - 122 | distribution card | per-window HACC - P10 ADE6 over all 2,399 windows: 4 histograms (CAUSAL / STRICT x s0 / s1), share of windows each side wins, the seven windows marked |")
w("| 122 - 136 | result card CAUSAL | forest plot of the primary row, the nine-row table (LONG / LAT families), mean ADE6, what the causal bucket costs, what it is not |")
w("| 136 - 150 | result card STRICT | the same for cameras frozen + causal bucket |")
w("| 150 - 166 | lateral-gap card | L2 curvature at speed (WORSE both seeds), L1 lateral acceleration (not worse), L3 yaw rate, heading, cross-track, canonical curvature; the noise-floor chart; the next step |")
w("")
w("## 3. Scene selection (`SELECTION_V4.md`, `select_scenes.py`, `data/selection_v4.json`)")
w("")
w("Fixed before any per-window T1 number was read: `select_scenes.py` loads only `g`, `ws`, `clip_index` (control S1 checks the source, the file mtimes, and re-derives the rows).")
w("")
w("1. The four v3 windows (JUNCTION #1131, CURVE #654, NO-TURN #763, FAILURE #2158), if in the T1 grid: all four are, and the grid row equals the v3 window id.")
w(f"2. Three windows from `numpy.random.default_rng({SEL['seed']}).choice({SEL['n_candidates']}, 3, replace=False)` over the T1-grid windows (dump order) with GT mean speed >= {SEL['speed_min_mps']:g} m/s ({SEL['n_ge5']:,} of {SEL['n_windows']:,}), excluding the four of rule 1. Draw positions {SEL['draw_positions_in_candidates']}.")
w("")
w("| # | tag | eval window (grid row) | clip | clip_index | ws (cache step) | GT mean speed (m/s) |")
w("|---|---|---|---|---|---|---|")
for i, s in enumerate(SEL["scenes"]):
    w(f"| {i + 1} | {s['tag']} | #{s['grid_row']} | {s['clip'][:8]} | {s['clip_index']} | {s['ws']} | {s['gt_mean_speed']:.2f} |")
w("")
w("No window was picked, swapped or dropped for how the model looks on it. Per-window result (ADE6, m): see the end frame of each window; the FAILURE window is worse than the floor in all four (run, seed) cells and CURVE in two of four; both stay in the video.")
w("")
w("## 4. The numbers on the result cards (verbatim from the RESULT files; tier T1-kin; estimator paired episode-cluster bootstrap, B = 2,000, 141 episodes; seeds s0 / s1)")
w("")
w("| card | quantity (HACC - P10, lower is better) | s0 | s1 | source (git blob sha in `data/results/`) |")
w("|---|---|---|---|---|")
src = {"RESULT_V8F_T1KIN_CAUSALBUCKET": "c2056a60", "RESULT_V8F_T1KIN_STRICT": "fc1e8cf3", "RESULT_V8F_T1_LATCOND": "b8a15b84", "RESULT_V8F_HACC_LAT": "53b36103"}
rows = [("causal", "RESULT_V8F_T1KIN_CAUSALBUCKET", "ADE6, PRIMARY", "ADE6 (m), PRIMARY"), ("causal", "RESULT_V8F_T1KIN_CAUSALBUCKET", "curvature MAE", "curvature MAE (1/m), canonical row"),
        ("strict", "RESULT_V8F_T1KIN_STRICT", "ADE6, PRIMARY", "ADE6 (m), PRIMARY"), ("strict", "RESULT_V8F_T1KIN_STRICT", "curvature MAE", "curvature MAE (1/m), canonical row"),
        ("lateral", "RESULT_V8F_T1_LATCOND", "L2:", "L2 curvature MAE at GT >= 5.2 m/s (1/m)"), ("lateral", "RESULT_V8F_T1_LATCOND", "L1:", "L1 lateral-acceleration MAE (m/s^2)"),
        ("lateral", "RESULT_V8F_T1_LATCOND", "L3:", "L3 yaw-rate MAE at GT >= 5.2 m/s (deg/s)")]
for card, f, key, lab in rows:
    w(f"| {card} | {lab} | {cell(f, key, 1)} | {cell(f, key, 2)} | `{f}.md` ({src[f]}) |")
w("")
w("The other table rows (ADE2, speed, along-track, accel, heading, yaw rate, cross-track) are copied the same way and registered (control R1). Mean ADE6 (CAUSAL 2.675 / 2.710, STRICT 2.716 / 2.802, P10 3.443 m) and the other sentences are verbatim snippets of the RESULT files.")
w("")
w("Source commit of the RESULT files: `abc12d3f` (`git -C C:/Users/Admin/tanitad-push -c safe.directory=* show abc12d3f:\"TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/<name>\"`); the saved copies in `viz4/data/results/` hash to the blob shas above (checked: `git hash-object`).")
w("Dumps (read-only): `C:/Users/Admin/t1kc_run/dump_HACC_s{0,1}_m10_causal/ep*.npz`, `C:/Users/Admin/t1ks_run/dump_HACC_s{0,1}_m10_strict/ep*.npz` (141 files each; keys g, cl, fl, ol, ha0, ws, eid, clip_index, v0). Their float64 originals are also in "
  "`E:/Projects/TanitAD-artifacts/v8f-mini-hacc/t1kin_causalbucket_dumps_f64_2026-10-26.tar` and `t1kin_strict_dumps_f64_2026-10-26.tar`. Join: `C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz`. Frames: `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/<clip>.v2ep.pt` via v3's accessor.")
w("")
w("## 5. Controls (`CONTROLS_V4.json`)")
w("")
w(f"ALL_PASS before the render = **{C['ALL_PASS_before_render']}**; post-render controls done = **{C['post_render_done']}**; ALL_PASS (needs the post-render controls) = **{C['ALL_PASS']}**.")
w("")
w("| control | result | tolerance |")
w("|---|---|---|")
t1 = T("T1_tracks_equal_dumps")
w(f"| T1. every plotted track (the matplotlib artists of the figure, read back) = the raw dump arrays, from an independent loader | {t1['n_frames']} scene frames x 6 lines = {t1['n_lines']} lines, max abs **{t1['max_abs_m']:.1e} m**, PASS = {t1['PASS']} | <= 1e-6 m |")
a1 = T("A1_window_ade6_recomputed")
w(f"| A1. per-window ADE6 and deltas on screen = recomputed from the raw dumps, string-equal and in the frame's text | {a1['n_numbers']} numbers, {len(a1['failures'])} failures, PASS = {a1['PASS']} | exact string |")
h1 = T("H1_histogram_recomputed")
w(f"| H1. histogram counts, shares and means recomputed from all 141 episode files; each mean = the RESULT point estimate | {h1['n']} panels, {len(h1['failures'])} failures, PASS = {h1['PASS']} | exact |")
r1 = T("R1_result_cells_equal_files")
w(f"| R1. every RESULT cell / snippet a scene shows = the cell of the file re-parsed here, and its value is in the frame's text | {r1['n_cells']} cells ({r1['full_cell_on_screen']} shown in full), {r1['n_snippets']} snippets, {len(r1['failures'])} failures, PASS = {r1['PASS']} | exact string |")
b1 = T("B1_primary_intervals_reproduced_from_dumps")
w("| B1. the four primary intervals re-derived from the float64 dumps with the repo's estimator (B = 2,000, seed 0) = the RESULT cells | " + "; ".join(f"{r['run']} s{r['seed']} {r['rederived']}" for r in b1["rows"]) + f" ; PASS = {b1['PASS']} | exact 4-decimal string |")
n4 = T("N4_existence_scan")
w(f"| N4. every numeric token of every chapter-7 drawn string (and the closing card) is in a RESULT file, a registered number, the selection, or a declared constant | {n4['n_tokens']} tokens, {n4['n_unmatched']} unmatched, PASS = {n4['PASS']} | 0 |")
w(f"| W. no 'drives' / 'driving' / 'autonomous' / 'SOTA'; 'closed loop' only inside 'NOT closed loop'; T1-kin on every chapter-7 frame; estimator and seed on every card | PASS = {T('W_wording_tier_estimator_seed')['PASS']} | - |")
s1 = T("S1_selection_rederived")
w(f"| S1. selection re-derived from the raw GT = `selection_v4.json`; `select_scenes.py` reads only g / ws / clip_index; selection file older than the data layer; v3 windows in the grid | PASS = {s1['PASS']} | exact |")
g1 = T("G1_frames_and_gt_alignment_with_v3")
w(f"| G1. chapter-7 frame accessor pixel-identical to v3's on the four v3 windows; dump GT = v3's `cf_windows` GT of the same window | max GT diff {max(g1['gt_vs_v3_cf_max_abs_m'].values()):.1e} m (float32 source), PASS = {g1['PASS']} | 1e-3 m |")
w(f"| F1. the floor's executed tracks are identical in all four dump sets | max abs {max(T('F1_floor_identical_all_four_dump_sets')['max_abs_m'].values()):.1f} m, PASS = {T('F1_floor_identical_all_four_dump_sets')['PASS']} | 0 |")
w(f"| **X. deliberate regression: cl and fl of scene {X['scene_swapped']} swapped at the data-access layer (`V4_SWAP_CLFL`)** | T1 FAILED as it must ({X['n_T1_failures']} line failures, tracks off by up to {X['T1_max_abs_m']:.2f} m), A1 FAILED as it must ({X['n_A1_failures']} number failures); regression arm PASS = {X['PASS']}. "
  "Still: `viz4/regression_swap_JUNCTION_end.png` (labels say HACC, the data are the floor's) | both must fail |")
if post:
    p = post
    w(f"| V1. video: codec / size / frame count | {p['V1_video']['probe']['codec_name']} {p['V1_video']['probe']['width']}x{p['V1_video']['probe']['height']}, {p['V1_video']['probe']['nb_read_frames']} frames (expected {p['V1_video']['expected_frames']}), PASS = {p['V1_video']['PASS']} | exact |")
    w(f"| P1. the full render's drawn text = the control render's, chapters 7-8 | PASS = {p['P1_full_render_text_equals_control_render']['PASS']} | exact |")
    w(f"| P2. the plotted arrays of the video's frames = the controlled ones (sha1 per frame) | {p['P2_plotted_arrays_of_the_video_equal_the_controlled_ones']['n_frames']} frames, PASS = {p['P2_plotted_arrays_of_the_video_equal_the_controlled_ones']['PASS']} | sha1 |")
    u1 = p["U1_chapters_1_to_6_unchanged_vs_v3"]
    w(f"| U1. segments 1-6 text-identical to v3 after the header swap, except the title card | {u1['identical_after_header_swap']} of {u1['n_frames']} identical, differing = {u1['differing_frames']}, PASS = {u1['PASS']} | exact |")
    w(f"| U2. v3's registered numbers (1,089 minus the closing card) unchanged | PASS = {p['U2_v3_registered_numbers_unchanged']['PASS']} | exact |")
else:
    w("| V1, P1, P2, U1, U2 (post-render: video integrity, the video's text and plotted arrays = the controlled ones, segments 1-6 unchanged vs v3) | **pending**: `controls4.py --post` runs at the end of the detached chain (`run_chain.cmd`) | - |")
w("")
w("## 6. Re-run")
w("")
w("```")
w("PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe ; export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1   # cwd = viz4/")
w("$PY select_scenes.py                       # SELECTION_V4.md (GT only)")
w("$PY build_video4.py [--qa-only --tags 07d_hist,...]   # video + data/onscreen_text.json, numreg.json, ch7_lines_digest.json")
w("$PY controls4.py --main ; V4_SWAP_CLFL=JUNCTION $PY controls4.py --swap ; $PY controls4.py --post   # CONTROLS_V4.json")
w("```")
w("")
open(f"{HERE}/VIDEO_NOTES_V4.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote VIDEO_NOTES_V4.md", len("\n".join(L)))
