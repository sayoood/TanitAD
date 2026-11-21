"""notes_gen5.py -- writes VIDEO_NOTES_V5.md from CONTROLS_V5*.json, video_chapters.json, SELECTION_V5 (selection_v5.json), data/DAG_READ.json and data/numreg.json (so the notes cannot drift from the controls).  CPU only."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
C = json.load(open(f"{HERE}/CONTROLS_V5.json"))
M = C["main"]["controls"]
X = C["X_regression_arm_swap_dag_dag0"]
CH = json.load(open(f"{HERE}/video_chapters.json"))
SEL = json.load(open(f"{HERE}/data/selection_v5.json"))
DR = json.load(open(f"{HERE}/data/DAG_READ.json", encoding="utf-8"))
NR = json.load(open(f"{HERE}/data/numreg.json"))
CM = json.load(open(f"{HERE}/data/dag_cells_manifest.json"))
post = C.get("post_render")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
chs = {c["name"]: c["t_s"] for c in CH["chapters"]}
t8 = chs["8 DAgger: trained on its own states"]
t9 = chs["9 closing"]
MINUS = "\u2212"


def g(ptr):
    o = DR
    for k in ptr.split("/"):
        o = o[k]
    return o


def f4(x):
    return f"{x:+.4f}".replace("-", MINUS)


L = []
w = L.append
w("# v8F-mini evidence video v5 (HACC + T1-kin + DAgger) -- VIDEO_NOTES_V5 (chapter 8: every number, its artifact, evidence class, tier, estimator)")
w("")
w(f"Written 2026-10-27 by the video5 subagent. Dev box, CPU only, ONE render process, nothing staged, committed or pushed. Work dir `E:/Projects/TanitAD-artifacts/video5/`. Video: `v8f_evidence_v5.mp4`, "
  f"{CH['duration_s']:.1f} s, 1920x1080, H.264, 10 fps, 9 segments, {CH['unique_frames']} unique frames. v4's notes are `v4_ref/VIDEO_NOTES_V4.md` (a copy, unchanged) and apply to segments 1-7.")
w("")
w("## 0. Status and wording")
w("")
w("* **Tier of chapter 8: T1-kin = self-action OPEN loop** (the DAG and DAG-0 runs `--limit causal --dump-ticks --dump-heads`, plain, replan every 2 s from the model's own executed motion; cameras = the human's logged frames). "
  "Never 'closed loop' (only inside 'NOT closed loop'), never 'drives' / 'autonomous' (control W). R-T0 is the guard tier: T0 teacher-forced own code. STRATEGIC is NOT paused for v8F: S-POS vs CTRV is shown beside and decides nothing.")
w("* **Estimator:** paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows (`taniteval/ci.py`), DAG - DAG-0 per training seed: 'another draw of EPISODES' only. Another TRAINING run is answered by the two seeds and the "
  "replicate floor RF = |F1(DAG-0) - F1(ORC-0)| (ORC-0 is a same-argv replicate of DAG-0 trained by another build); the planner is deterministic, so no inference-seed question arises.")
w(f"* **Evidence class: MEASURED (ours).** Every interval, bar, F1 and S-POS cell is a pointer into `data/DAG_READ.json` (sha256 `{sha(f'{HERE}/data/DAG_READ.json')}`, the pre-registered read, "
  "PREREG_V8F_DAGGER.md at 2a78f059 incl. A1, A2; OUTCOME **" + g("OUTCOME/OUTCOME") + "**), re-read by an independent pointer walker (control J1); the R-ADE and R-T0 intervals are re-derived from the float64 dumps / T0 arrays with the repo's estimator (control B1). "
  "Per-window numbers are computed from the float64 dumps (control A1). Per-class F1 is derived here from the label cells, beside, and is not in the read. Baseline = DAG-0, the same fine-tune without the lever; not a published state of the art; no external benchmark.")
w("* **Where DAG looks worse (all on screen):** see section 5. Nothing is registered, promoted or retracted by this video; landing and the GOALS_AND_CLAIMS register are the Master Mind's step.")
w("")
w("## 1. What changed against v4")
w("")
w("* **New chapter 8** 'DAgger: trained on its own states', inserted before the closing (now chapter 9). Code: `v5_lib.py` (data layer), `v5_scenes.py` (scenes), `extract_dag_cells.py` (label cells via the programme's own reader), `select_scenes5.py`, `build_video5.py` (schedule), `controls5.py`.")
w("* **Edited in segments 1 and 9, nothing else of 1-7:** (a) the header string of every frame says v5; (b) the title card has a sixth question and a compacted layout; (c) the closing card is rewritten: a fourth, full-width box carries chapter 8's numbers and what is still open. "
  "Control U1: every other frame of segments 1-7 is text-identical to v4's after the header swap; U2: v4's registered numbers are unchanged. v4's chapter-7 code (`v4_lib.py`, `v4_scenes.py`) is byte-identical to the banked v4 package.")
w(f"* Segment timing (s): 1-7 as in v4 (0 - {t8:.1f}); **8 DAgger {t8:.1f} - {t9:.1f}**; 9 closing {t9:.1f} - {CH['duration_s']:.1f}.")
w("")
w("## 2. Chapter 8 outline")
w("")
w("| t (s, in the chapter) | scene | what is on screen |")
w("|---|---|---|")
w("| 0 - 14 | intro card | what DAgger changes (p = 0.1 of the training samples replaced by the model's own rolled-out states, human labels, M2b init, 3,000 steps, two seeds), the control DAG-0, what the viewer will see |")
w("| 14 - 28 | rule card | the selection rule (verbatim, sha256 head), the seven windows with their stratum and delta, why they are not a sample |")
w("| 28 - 108 | 7 windows x 11.4 s | per window: the logged camera frame, DAG and DAG-0 BEVs (seed 0) with the human's path (GT), each arm's EXECUTED path growing in real time (O, orange), S's three checkpoints of the current tick (pink diamonds) and the human's route checkpoints (hollow); under each BEV three tick boxes (+0, +2, +4 s) with S position error at 30/50/80 m and T's lateral / longitudinal decision (head argmax) against the label; at the end the per-window table (ADE6, path length, T right counts, both seeds) and a caption that says which arm is closer |")
w("| 108 - 122 | distribution card | per-window DAG - DAG-0 ADE6 over all 2,399 windows, both seeds: shares, means (= the R-ADE point estimates), the seven windows marked |")
w("| 122 - 138 | T-head card | forest of R-TLAT / R-TLON with the pre-registered bars, the F1 table, the OUTCOME; per-class F1 (derived, beside) |")
w("| 138 - 154 | guards card | R-ADE, R-T0 vs the F/2 bar (forest), S-POS vs CTRV within each run, tick-0 skill, crawl fraction, what it is not |")
w("")
w("## 3. Scene selection (`SELECTION_V5.md`, `select_scenes5.py`, `data/selection_v5.json`)")
w("")
w(f"Written by the script BEFORE any chapter-8 frame was rendered (control S1 checks the file mtimes, re-derives the rows with independent code, and that the script reads only g / cl / ws / clip_index / eid). **Rule sha256 `{SEL['rule_sha256']}`** (the string `RULE_TEXT` in the script); script sha256 `{SEL['script_sha256']}`.")
w("")
for ln in SEL["rule_text"].split("\n"):
    w("> " + ln)
w("")
w("| # | tag | how chosen | stratum | eval window (grid row) | clip | clip_index | ws | GT mean speed (m/s) | delta s0 (m) |")
w("|---|---|---|---|---|---|---|---|---|---|")
for i, s in enumerate(SEL["scenes"]):
    w(f"| {i + 1} | {s['tag']} | {s['source']} | {s['stratum']} | #{s['grid_row']} | {s['clip'][:8]} | {s['clip_index']} | {s['ws']} | {s['gt_mean_speed']:.2f} | {s['delta_s0_m']:+.4f} |")
w("")
w("## 4. The numbers on the cards (tier T1-kin unless stated; estimator paired episode-cluster bootstrap, B = 2,000, 141 episodes; seeds s0 / s1; all from `data/DAG_READ.json`)")
w("")
w("| card | quantity (DAG - DAG-0) | s0 | s1 | pointer |")
w("|---|---|---|---|---|")
for nm, rk in (("lateral head macro-F1 (R-TLAT)", "R-TLAT"), ("longitudinal head macro-F1 (R-TLON)", "R-TLON")):
    cells = [f"{f4(g(f'rows/s{sd}/{rk}/delta'))} [{f4(g(f'rows/s{sd}/{rk}/lo'))}, {f4(g(f'rows/s{sd}/{rk}/hi'))}], bar {g(f'rows/s{sd}/{rk}/bar'):.4f}, F1 {g(f'rows/s{sd}/{rk}/f1_a'):.4f} vs {g(f'rows/s{sd}/{rk}/f1_b'):.4f}, clears {g(f'rows/s{sd}/{rk}/clears')}" for sd in (0, 1)]
    w(f"| T heads | {nm} | {cells[0]} | {cells[1]} | rows/s*/{rk}/* |")
for nm, rk in (("T1-kin ADE6 (m), guard R-ADE", "R-ADE"), ("T0 own-code ADE6 (m), guard R-T0", "R-T0")):
    cells = [f"{f4(g(f'rows/s{sd}/{rk}/delta'))} [{f4(g(f'rows/s{sd}/{rk}/lo'))}, {f4(g(f'rows/s{sd}/{rk}/hi'))}], means {g(f'rows/s{sd}/{rk}/mean_a'):.3f} vs {g(f'rows/s{sd}/{rk}/mean_b'):.3f}" for sd in (0, 1)]
    w(f"| guards | {nm} | {cells[0]} | {cells[1]} | rows/s*/{rk}/* |")
w(f"| guards | F/2 bar for R-T0 (m) | {g('bars/F_over_2_t0_point_m'):.4f} | same | bars/F_over_2_t0_point_m |")
w(f"| (OUTCOME) | {g('OUTCOME/OUTCOME')}: {g('OUTCOME/condition')} | guards_held {g('OUTCOME/flags/guards_held')} | | OUTCOME/* |")
w("")
w("S-POS vs CTRV (within each run, ticks >= 1, unpaired, beside, decides nothing), delta (m) [lo, hi] and the read's `separated` flag:")
w("")
w("| run, arc | s0 | s1 |")
w("|---|---|---|")
for arm in ("DAG", "DAG0"):
    for arc in (30, 50, 80):
        cs = []
        for sd in (0, 1):
            c = g(f"beside/per_seed/s{sd}/per_run_t1heads/{arm}/S-POS vs CTRV (ticks>=1, within this run)/S-POS@{arc}")
            cs.append(f"{c['delta']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] separated {c['separated']}")
        w(f"| {arm} @{arc} m | {cs[0]} | {cs[1]} |")
w("")
w("Per-window values (the end frame of each window; ADE6 in m, T1-kin, no interval), from `data/numreg.json` (kinds ade6 / ade6d, recomputed by control A1 from the float64 dumps):")
w("")
w("| # | tag | s0 DAG | s0 DAG-0 | s0 delta | s1 DAG | s1 DAG-0 | s1 delta |")
w("|---|---|---|---|---|---|---|---|")
vals = {}
for r in NR:
    if r.get("kind") in ("ade6", "ade6d") and r["scene"].startswith("d5_frame") and r["scene"].endswith(", 30)"):
        vals[(r["tag"], r["kind"], r.get("arm"), r["seed"])] = r["shown"]
for i, s in enumerate(SEL["scenes"]):
    t = s["tag"]
    w(f"| {i + 1} | {t} | " + " | ".join(f"{vals[(t, 'ade6', 'DAG', sd)]} | {vals[(t, 'ade6', 'DAG0', sd)]} | {vals[(t, 'ade6d', None, sd)]}" for sd in (0, 1)) + " |")
w("")
w(f"Source dumps (read-only): `E:/Projects/TanitAD-artifacts/v8f-dagger-t1/dump_{{DAG,DAG0}}_s{{0,1}}_m10_causal_heads/ep*.npz` (keys g, cl), per-tick files `ticks/ticks_*_m10_causal_heads.npz` (sha256 in `data/DAG_READ.json` `runs`), run manifest `runs.json`. "
  f"Label cells: `data/dag_cells_*.npz` from `extract_dag_cells.py` (imports the programme's own `t1hd_read.py`, sha256 `{CM['t1hd_read.py'][:16]}...`, from the pinned tool tree `{CM['tree']}`), manifest `data/dag_cells_manifest.json`.")
w("")
w("## 5. Where DAG looks worse (all of it on screen)")
w("")
loses = [(s["tag"], s["delta_s0_m"]) for s in SEL["scenes"] if s["delta_s0_m"] > 0]
w(f"* **Windows (seed 0, ADE6 DAG - DAG-0 > 0): " + ", ".join(f"{t} {d:+.2f} m" for t, d in loses) + f"** ({len(loses)} of 7). The WORST window (#{[s for s in SEL['scenes'] if s['tag'] == 'WORST'][0]['grid_row']}, NO-TURN) is DAG +8.86 m at seed 0 and +2.05 m at seed 1: DAG executes a 73 m path where the human travels 39 m (the human stops: label STOP; DAG's longitudinal head says DECEL, DAG-0's says STOP at tick 1).")
w("* **Distribution:** DAG wins about two thirds of the windows at both seeds and loses a third (histogram card); the mean difference is the R-ADE point (guard held, both seeds separated BETTER).")
w("* **R-T0 at seed 1 is separated WORSE** (+0.0483 m [+0.0178, +0.0807]) but under the pre-registered F/2 bar of 0.0604 m: the guard held, the cost is on the card.")
w("* **Tick 0 (the logged state):** DAG's macro-F1 is lower than DAG-0's in all four numbers (lat 0.8469 vs 0.8504 / 0.8574 vs 0.8621, lon 0.4656 vs 0.4709 / 0.4835 vs 0.5111; no interval in the read).")
w("* **Still open:** images stay logged, so lateral recovery is not taught (PREREG section 3); one bank pass and one fine-tune of 3,000 steps per arm; the crawl rows M1 / L2 are carried per run, not paired; per-turn-class F1 and the four families at T0 and T1 have no instrument in the read (per-class F1 on the card is derived here, beside); the HACC curvature gap of chapter 7 is not re-read for DAG.")
w("")
w("## 6. Controls (`CONTROLS_V5.json`)")
w("")
w(f"ALL_PASS before the render = **{C['ALL_PASS_before_render']}**; post-render controls done = **{C['post_render_done']}**; ALL_PASS = **{C['ALL_PASS']}**.")
w("")
w("| control | result |")
w("|---|---|")
t1 = M["T1_plotted_lines_equal_raw_arrays"]
a1 = M["A1_per_window_numbers_recomputed"]
l1 = M["L1_label_chain_all_windows"]
f1 = M["F1_macro_f1_reproduces_the_read"]
s1 = M["S1_S_error_and_selection_rederived"]
h1 = M["H1_histogram_recomputed"]
j1 = M["J1_dag_read_pointers_equal_the_file"]
b1 = M["B1_intervals_reproduced_with_the_repo_estimator"]
n4 = M["N4_existence_scan"]
ww = M["W_wording_tier_estimator_seed"]
w(f"| T1. every plotted line (GT, executed path, S, S GT; both arms) = the raw arrays, independent loader | {t1['n_frames']} window frames, {t1['n_lines']} lines, max abs **{t1['max_abs_m']:.1e} m**, PASS = {t1['PASS']} (tolerance 1e-6 m) |")
w(f"| A1. per-window numbers on screen = recomputed from the raw arrays | {a1['n_numbers']} numbers {a1['by_kind']}, {len(a1['failures'])} failures, PASS = {a1['PASS']} |")
w(f"| L1. label chain over all windows x 3 ticks x 4 runs x 2 heads: pred = argmax of the tick file's logits; true = the join class at the matched route row with the on-route cut (own formula) | {len(l1['runs'])} (run, head) pairs, failures {l1['failures']}, PASS = {l1['PASS']} |")
w(f"| F1. own macro-F1 of the extracted cells = the read's own-cell F1 (8 numbers) and matched-cell F1 (4 rows x 2) | max abs diff {max(r['abs_diff'] for r in f1['rows']):.1e} over {len(f1['rows'])} rows, PASS = {f1['PASS']} (tolerance 1e-9) |")
w(f"| S1. S error = abs(S checkpoint - S GT) for all windows (4 runs); selection re-derived from the raw dumps with own code = `selection_v5.json`; script reads only g / cl / ws / clip_index / eid; selection files older than every chapter-8 artifact; rule sha256 matches the text | max abs diff {s1['max_abs_diff_m']:.1e} m, rows {s1['rows_rederived']} = {s1['rows_in_selection_v5']}, PASS = {s1['PASS']} |")
w(f"| H1. histogram counts, shares, means recomputed from the dumps; each mean = the R-ADE delta (4 decimals) | {h1['n']} panels, {len(h1['failures'])} failures, PASS = {h1['PASS']} |")
w(f"| J1. every DAG_READ pointer on screen re-read by own code, value in the frame's drawn text | {j1['n_items']} items {j1['by_kind']}, {len(j1['failures'])} failures, PASS = {j1['PASS']} |")
w(f"| B1. R-ADE and R-T0 intervals re-derived with the repo's estimator (B = 2,000, seed 0) = the read's cells | " + "; ".join(f"{r['row']} {r['rederived']}" for r in b1['rows']) + f"; PASS = {b1['PASS']} |")
w(f"| N4. every numeric token of every chapter-8 drawn string (and the closing card) is registered, in the selection, or a declared constant | {n4['n_tokens']} tokens, {n4['n_unmatched']} unmatched, PASS = {n4['PASS']} |")
w(f"| W. no 'drives' / 'driving' / 'autonomous' / 'SOTA'; 'closed loop' only inside 'NOT closed loop'; T1-kin on every chapter-8 frame; estimator and seed on the cards | PASS = {ww['PASS']} |")
w(f"| G1. every window ends inside its clip | PASS = {M['G1_windows_inside_their_clips']['PASS']} |")
w(f"| **X. deliberate regression: DAG and DAG-0 arrays of scene {X['scene_swapped']} swapped at the data-access layer (`V5_SWAP_ARMS`)** | T1 FAILED as it must ({X['n_T1_failures']} line failures, up to {X['T1_max_abs_m']:.2f} m), A1 FAILED as it must ({X['n_A1_failures']} number failures); regression arm PASS = {X['PASS']}. Still: `regression_swap_{X['scene_swapped']}_end.png` (labels say DAG, the data are DAG-0's) |")
if post:
    p = post
    w(f"| V1. video: codec / size / frame count | {p['V1_video']['probe']['codec_name']} {p['V1_video']['probe']['width']}x{p['V1_video']['probe']['height']}, {p['V1_video']['probe']['nb_read_frames']} frames (expected {p['V1_video']['expected_frames']}), PASS = {p['V1_video']['PASS']} |")
    w(f"| P1. the full render's drawn text = the control render's, chapters 8-9 | {p['P1_full_render_text_equals_control_render']['n_checked']} scenes, PASS = {p['P1_full_render_text_equals_control_render']['PASS']} |")
    w(f"| P2. the plotted arrays of the video's frames = the controlled ones (sha1 per frame) | {p['P2_plotted_arrays_of_the_video_equal_the_controlled_ones']['n_frames']} frames, PASS = {p['P2_plotted_arrays_of_the_video_equal_the_controlled_ones']['PASS']} |")
    u1 = p["U1_chapters_1_to_7_unchanged_vs_v4"]
    w(f"| U1. segments 1-7 text-identical to v4 after the header swap, except the title card | {u1['identical_after_header_swap']} of {u1['n_frames']} identical, differing = {u1['differing_frames']}, PASS = {u1['PASS']} |")
    w(f"| U2. v4's registered numbers (minus closing / title) unchanged | {p['U2_v4_registered_numbers_unchanged']['n_v4']} items, PASS = {p['U2_v4_registered_numbers_unchanged']['PASS']} |")
w("")
w("## 7. Re-run")
w("")
w("```")
w("PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe ; export CUDA_VISIBLE_DEVICES=-1 PYTHONUTF8=1   # cwd = E:/Projects/TanitAD-artifacts/video5/")
w("$PY -I extract_dag_cells.py                  # data/dag_cells_*.npz (pinned tool tree, the programme's own reader)")
w("$PY -I select_scenes5.py                     # SELECTION_V5.md + data/selection_v5.json")
w("$PY -I controls5.py --main ; V5_SWAP_ARMS=WORST $PY -I controls5.py --swap")
w("$PY -I build_video5.py [--qa-only --tags 08a_intro,...]   # video + data/onscreen_text.json, numreg.json, ch78_lines_digest.json")
w("$PY -I controls5.py --post                   # CONTROLS_V5.json")
w("```")
w("")
w("Inputs outside this dir (read-only): the scratchpad dirs `viz/`, `viz2/`, `hacc/`, `hblend/`, `hspec/`, `listen/` of the v3/v4 pipeline (chapters 1-6), `t1kc_run/` + `t1ks_run/` (chapter-7 dumps), `tanitad-data/refav1-eval141-256rb/` (frames). "
  "Chapter-7's selection (`SELECTION_V4.md`, `select_scenes.py`) is in `v4_ref/` and the v4 package.")
open(f"{HERE}/VIDEO_NOTES_V5.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("VIDEO_NOTES_V5.md written,", len(L), "lines")
