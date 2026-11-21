"""notes_gen.py -- writes OUT/VIDEO_NOTES.md; every table is generated from the data files (no transcription)."""
import json

import numpy as np

SC = "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad"
OUT = f"{SC}/viz2"
cf = np.load(f"{OUT}/data/cf_windows.npz")
ew = json.load(open(f"{OUT}/data/cf_evalwide.json"))
tb = json.load(open(f"{OUT}/data/tables.json"))
pk = np.load(f"{OUT}/data/plans_pack.npz")
ctl = json.load(open(f"{OUT}/CONTROLS_V2.json"))
sel = json.load(open(f"{SC}/viz/out/selection.json"))
KEYS = ["JUNCTION", "CURVE", "NO-TURN", "FAILURE(worst)"]
SEEDS = ("M2b_s0", "M2b_s1")
CFG = (("a_all", "(a) S+T+O"), ("b_O", "(b) O only"), ("c_T", "(c) T only"), ("d_S", "(d) S only"))
L = []
w = L.append
w("# v8F-mini evidence video v2 -- VIDEO_NOTES (every number, its artifact, evidence class, tier, estimator)\n")
w("Written 2026-10-25 by the viz2 subagent. Dev box only, CPU only, nothing staged, committed or pushed. All paths are under "
  "`C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/` (`SC` below).\n")
w("## 0. Status and honest caveats\n")
w("* The shipped `viz2/v8f_evidence_v2.mp4` is the FIRST full build (146.4 s, 1920x1080, H.264, 10 fps, 6 segments). The build was stopped there because `C:` free space fell below the 1.5 GB hand-back threshold "
  "(about 0.38 GB free, caused by other processes; the viz2 footprint is 57 MB). Three small source fixes were made AFTER that build and are in `viz2/v2_scenes.py` but NOT in the mp4: "
  "(1) segment 5's S card gains an S-age chart (50 m error of the held vs fresh checkpoint by the age of S's last tick, both seeds, from `SC/sfresh/sfresh_extra.json`; preview `viz2/_t_s_card.png`); "
  "(2) the caption of the two segment-5 window frames is shortened (in the built video its last line touches the bottom edge of the caption box, text still legible) and now points to the segment's closing card for seed 1; "
  "(3) the segment-3 summary caption now says the 23-of-24 T-lateral count is with the nav command at all three layers. "
  "One command rebuilds the video from source once C: has room: `CUDA_VISIBLE_DEVICES=-1 V2_THREADS=1 OMP_NUM_THREADS=1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe viz2/build_video.py --workers 4` (writes atomically via a .tmp.mp4).")
w("* Tier of everything: **T1-class open loop** (the model reads its own held outputs; nothing it does feeds back). The nav counterfactual is an **INTERVENTION on the model's input** (a diagnostic of dependence), never driving. "
  "Fresh-S ages are a diagnostic, not the deployed schedule.")
w("* Estimator: episode-cluster bootstrap, B = 2,000, 141 episodes (`taniteval/ci.py`), paired for differences. It answers *another draw of EPISODES* only. Training-seed variance = the two seeds M2b_s0 / M2b_s1 side by side (one model per seed, no replicate arm). "
  "Inference is deterministic (no sampling), so there is no inference-seed floor. A separated CI from a one-seed arm is necessary, not sufficient, for a *lever* claim; nothing here claims a lever.")
w("* STRATEGIC family: **PAUSED** by the PI ruling of 2026-09-19 (`PI_DECISION_QUEUE.md` item 25). Segment 5 shows S's checkpoints as a diagnostic of the layer only; it is not a strategic-family pass and does not lift the pause.")
w("* Evidence classes: MEASURED (ours) = computed in this task from the banked checkpoints/arrays (artifact paths below); INHERITED = quoted from another agent's result file, with the number checked here against that file's table (LISTEN, SFRESH, HBLEND, HSPEC). Nothing is registered or promoted by this video.")
w("* One window is an example; the eval-wide numbers are the evidence. The 4 windows are exactly those of `SC/viz/out/selection.json`, declared in `viz2/SELECTION_V2.md` before any run (that file was not edited afterwards; see CONTROLS_V2.json `pre_declaration`).\n")
w("## 1. Controls (`viz2/CONTROLS_V2.json`): ALL_PASS = %s\n" % ctl["ALL_PASS"])
c1, c2, c3, c4 = (ctl[k] for k in ("control_1_own_nav_all_layers_reproduces_B_own_code_plan", "control_2_drawn_plans_equal_banked_perwindow_arrays", "control_3_frame_alignment_as_v1", "control_4_zeroed_e_nav_structural_zero"))
w("| control | result | tolerance |\n|---|---|---|")
w("| 1. window's own nav at S, T and O reproduces B's own-code plan (`forward_infer`) | max abs difference **%.1f** (traj30, S ckpt, T lat/lon logits), 8 window-seed pairs | exactly 0.0 |" % max(c1["own_nav_all_layers_vs_forward_traj30"], c1["own_nav_all_layers_vs_forward_ckpt"]))
w("| 2. H / HS / B / P10 as drawn equal the banked per-window arrays | max abs difference **%.1f m** for all four plans; ADE6 recomputed from the drawn arrays vs banked: max %.1e m | <= 1e-5 m |" % (max(c2["drawn_B_vs_source_tau"], c2["drawn_H_vs_source"], c2["drawn_P10_vs_source"], c2["drawn_HS_vs_source"]), c2["ade6_recomputed_vs_banked_P10"]))
w("| 3. frame alignment (v1 control, all 101 cache steps of the 4 clips) | argmax = self **%d / %d**, min PSNR %.1f dB | all self |" % (c3["total_argmax_is_self"], c3["total_steps"], c3["psnr_min_db"]))
w("| 4. zeroed `e_nav` model: D6, D6_TURN_L/R, L-R spread, LR_dist (structural zero) | **%.1f** over %d cells | exactly 0.0 |" % (c4["max_abs"], c4["n_cells"]))
inf = ctl["info_cpu_rerun_vs_banked_gpu"]
w("| info: this CPU float32 re-run vs the banked box-2 GPU arrays | own plan %.1e m, S ckpt %.1e m, fresh plan %.1e m (float32 round-off) | <= 1e-4 m |\n" % (inf["own_traj30_vs_banked_pred"], inf["own_ckpt_Sframe_vs_banked_ckpt"], inf["fresh_traj30_vs_sfresh_banked_pred"]))
w("## 2. Segment 3: nav counterfactual -- MEASURED (ours), T1-class open loop, intervention\n")
w("Artifacts: `viz2/data/cf_windows.npz` (4 windows x 2 seeds x 4 injection points x 3 commands; keys `<seed>/<window>/cf/<cfg>/<cmd>/{traj30,ckpt_now,lat_p,lon_p}` and `.../D6`, `.../lat_sep_LR`), produced by `viz2/cf_run.py`; "
  "eval-wide `viz2/data/cf_evalwide.{npz,json}` (all 2,399 windows, `viz2/cf_evalwide.py`). Commands: constant over the whole nav history, d_next = 30 m, args_valid = 1; the model's OWN `run_s -> run_t -> run_o` chain, only the named layer's `b['nav']` replaced.")
w("**D6** = mean over {TURN_L, TURN_R} of the Euclidean distance at 6 s between that plan and the FOLLOW plan at the same injection point (m). **L-R** = |lateral(TURN_L) - lateral(TURN_R)| at 6 s (m).\n")
w("### 2a. The four pre-declared windows (D6 m / L-R m)\n")
w("| window (eval index, clip, cache step t; 0.2 s per step) | seed | " + " | ".join(n for _, n in CFG) + " |\n|---|---|" + "---|" * 4)
for k in KEYS:
    s = sel[k]
    for sd in SEEDS:
        cells = [f"{float(cf[f'{sd}/{k}/cf/{c}/D6']):.2f} / {float(cf[f'{sd}/{k}/cf/{c}/lat_sep_LR']):.2f}" for c, _ in CFG]
        w(f"| {k} (#{s['window']}, {s['clip'][:8]}, t={s['t']}) | {sd[-1]} | " + " | ".join(cells) + " |")
w("\nT's lateral decision under the all-layers command (argmax of the T lateral head; expected LANE_KEEP / TURN_L / TURN_R): follows the command in **23 of 24** cases (4 windows x 2 seeds x 3 commands); the one exception is FAILURE(worst) seed 1, TURN_R.\n")
w("### 2b. Eval-wide, all 2,399 windows (mean, 95 % episode-cluster CI)\n")
w("| seed | metric | " + " | ".join(n for _, n in CFG) + " |\n|---|---|" + "---|" * 4)
for sd in SEEDS:
    for mt in ("D6", "lat_sep_LR"):
        cells = []
        for c, _ in CFG:
            r = ew["seeds"][sd][c][mt]
            cells.append(f"{r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]")
        w(f"| {sd[-1]} | {mt} (m) | " + " | ".join(cells) + " |")
for sd in SEEDS:
    f = ew["seeds"][sd]["a_all"]["T_lat_argmax_follows_cmd"]
    w(f"\n{sd}: with nav at all layers, T's lateral argmax equals the command in FOLLOW {100 * f['FOLLOW']:.0f} % / TURN_L {100 * f['TURN_L']:.0f} % / TURN_R {100 * f['TURN_R']:.0f} % of windows.")
w("\nReading (all MEASURED, both seeds): O only (about 1 m) < T only (about 4 m) < all three layers ~ S only (about 10-11 m). The route reaches the plan mostly through S's checkpoints and T's code; O's own nav token is nearly inert. "
  "'Through T' holds in the sense T-only >> O-only; S-only is the largest single-point effect, so the nav also reaches O through S's checkpoints, which O reads directly. On single windows the O-only / T-only order can flip (T only < O only on JUNCTION seed 1).\n")
w("## 3. Segment 2: who listens to what -- INHERITED from LISTEN, each number re-checked against the table in the file\n")
w("Source: `SC/listen/LISTEN_RESULT.md` (sections 2 and 4; `READ.json`; `fig_listen_grid.png` is shown as the grid). Tier T1-class open loop, donor-swap interventions; paired episode-cluster bootstrap B = 2,000, 141 episodes; both seeds M2b_s0 / M2b_s1. "
  "Numbers on screen (seed 0 / seed 1): S nav +4.197 / +3.985 m at 50 m, S scene +3.214 / +3.632 m, S all-inputs +8.057 / +8.375 m; T-lateral macro-F1 drop for S checkpoints +0.391 / +0.406, nav +0.173 / +0.216, scene -0.003 / +0.010 (not separated); "
  "T-longitudinal ego +0.247 / +0.261, scene +0.185 / +0.187; O ADE6 ego +5.697 / +4.716 m, T code +5.395 / +4.132 m, S checkpoints +0.873 / +1.894 m, nav +0.015 / +0.011 m (seed 1 not separated), all-inputs +1.394 / +1.315 m. "
  "Q-NAV (section 4): ADE6 change with the donor nav at S only +0.726 / +0.821, T only +0.476 / +0.521, O only +0.015 / +0.011, all three +1.174 / +1.180 m; D = T-only minus O-only +0.461 [+0.254, +0.687] / +0.510 [+0.273, +0.793], separated at both seeds.\n")
w("## 4. Segment 4: better plans -- INHERITED from HBLEND and HSPEC per-window arrays and score files, re-read programmatically\n")
w("Artifacts: `viz2/data/tables.json` and `viz2/data/plans_pack.npz` (from `SC/hblend/score_M2b_s{0,1}.json`, `score_M2b_s{0,1}_perwindow.npz` and `SC/hspec/score_hs2_M2b_s{0,1}.json`, `..._perwindow.npz`; `viz2/plans_pack.py`). "
  "Plans: B = M2b's own plan, P10 = frozen kinematic prior, H = HBLEND shrinkage of B toward P10, HS = HSPEC stage 2. Tier T1-class open loop (held outputs scored against the logged human path).\n")
w("### 4a. ADE by horizon over all 2,399 windows (m; mean [95 % CI])\n")
w("| seed | plan | 1 s | 2 s | 3 s | 4 s | 5 s | 6 s |\n|---|---|---|---|---|---|---|---|")
for sd in SEEDS:
    for pl in ("B", "H", "HS", "P10"):
        a = tb["hblend_hspec"][sd]["ade"][pl]
        w(f"| {sd[-1]} | {pl} | " + " | ".join(f"{a[f'ade{h}']['mean']:.3f} [{a[f'ade{h}']['lo']:.3f}, {a[f'ade{h}']['hi']:.3f}]" for h in range(1, 7)) + " |")
w("\n### 4b. Paired differences at 6 s, acceleration MAE and constraint violations\n")
w("| seed | pair | ADE6 delta (m) | accel MAE delta (m/s^2) | CONSISTENCY(a) delta (share of windows) |\n|---|---|---|---|---|")


def fmt(r):
    return f"{r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] {'sep.' if r['separated'] else 'not sep.'}"


for sd in SEEDS:
    t = tb["hblend_hspec"][sd]
    for pr in ("H-B", "H-P10", "HS-B", "HS-P10", "HS-H"):
        ac = fmt(t["accel"][pr]) if pr in t["accel"] else "n/a"
        co = fmt(t["consistency_any"][pr]) if pr in t["consistency_any"] else "n/a"
        w(f"| {sd[-1]} | {pr} | {fmt(t['paired_ade6'][pr])} | {ac} | {co} |")
w("\nAcceleration MAE (m/s^2): " + "; ".join(f"seed {sd[-1]}: " + ", ".join(f"{pl} {tb['hblend_hspec'][sd]['accel']['single'][pl]['mean']:.3f}" for pl in ("B", "H", "HS", "P10")) for sd in SEEDS)
  + ". Consistency violation rate: " + "; ".join(f"seed {sd[-1]}: " + ", ".join(f"{pl} {tb['hblend_hspec'][sd]['consistency_any']['rate'][pl]:.3f}" for pl in ("B", "H", "HS", "P10")) for sd in SEEDS) + ".\n")
w("The next step named on the gap card (a second-difference penalty in the training loss) needs a launch gate and box-2 time; it is not run here.\n")
w("### 4c. The four windows, ADE6 of each plan (m; seed 0 / seed 1), `viz2/data/plans_pack.npz`\n")
w("| window | P10 | B | H | HS |\n|---|---|---|---|---|")
for k in KEYS:
    w(f"| {k} | " + " | ".join(f"{float(pk[f'M2b_s0/{k}/ade6_{pl}']):.2f} / {float(pk[f'M2b_s1/{k}/ade6_{pl}']):.2f}" for pl in ("P10", "B", "H", "HS")) + " |")
w("\n## 5. Segment 5: the strategic layer at its own tick (DIAGNOSTIC; family PAUSED)\n")
w("Eval-wide, INHERITED from `SC/sfresh/SFRESH_RESULT.md` / `sfresh_numbers.json` (PREREG_V8F_SFRESH, tip 583cfc02), re-read into `viz2/data/tables.json`; n = 2,319 windows with a valid 50 m label; paired episode-cluster bootstrap B = 2,000, 141 episodes. "
  "Per-window S checkpoints are MEASURED here (`viz2/data/cf_windows.npz`: `own_ckpt_now`, `fresh_ckpt_now`, `rc` = the v9 RC-A GT checkpoints in the NOW frame).\n")
w("| seed | level | 50 m position error, mean [95 % CI] (m) |\n|---|---|---|")
for sd, s in (("0", "s0"), ("1", "s1")):
    lv = tb["sfresh"][s]["levels_pos_err_m"]
    for nm, key in (("banked eval ages (S held)", "cycle_banked"), ("S fresh (ticked at NOW)", "fresh"), ("nav-only kinematic baseline", "navonly_baseline")):
        r = lv[key]["@50"]
        w(f"| {sd} | {nm} | {r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] |")
for sd, s in (("0", "s0"), ("1", "s1")):
    t = tb["sfresh"][s]
    w(f"\nSeed {sd} paired at 50 m: fresh S - nav-only baseline {t['Q_LAYER_50m']['delta']:+.2f} [{t['Q_LAYER_50m']['lo']:+.2f}, {t['Q_LAYER_50m']['hi']:+.2f}]; fresh - held (stale cost) {t['Q_STALE_50m']['delta']:+.2f} [{t['Q_STALE_50m']['lo']:+.2f}, {t['Q_STALE_50m']['hi']:+.2f}]; "
      f"banked cycle S - nav-only baseline {t['banked_cycle_minus_navonly_50']['delta']:+.2f} [{t['banked_cycle_minus_navonly_50']['lo']:+.2f}, {t['banked_cycle_minus_navonly_50']['hi']:+.2f}].")
w("\nS-age chart (source only, not yet in the mp4): `SC/sfresh/sfresh_extra.json::age_profile_50m`, held vs fresh by S age 0.0-1.2 s, both seeds.")
w("Part of the held-vs-fresh gap is DEFINITIONAL: the banked eval scores S's checkpoints (predicted in the frame of S's tick) against labels in the NOW frame, so it pays for the distance driven since the tick. The deployed-schedule cost is the next strategic lever (not measured here).\n")
w("## 6. Frames and windows\n")
w("Frames: `C:/Users/Admin/tanitad-data/refav1-eval141-256rb/{clip}.v2ep.pt`, cache step j = native frame 2j (cylindrical 256 x 640). Windows (eval index, clip, t in s): " + "; ".join(f"{k} #{sel[k]['window']} {sel[k]['clip']} t={sel[k]['t']}" for k in KEYS) + ".\n")
w("## 7. Manifest (all under `SC/viz2/`, dev box, nothing in git)\n")
w("`v8f_evidence_v2.mp4`, `video_chapters.json`, `SELECTION_V2.md` (pre-declaration, unchanged), `CONTROLS_V2.json` (+ `_part1/_part2`), `VIDEO_NOTES.md`, `qa_*.png` (25 stills), `data/{cf_windows.npz,cf_evalwide.npz,cf_evalwide.json,plans_pack.npz,tables.json}`, "
  "scripts `v2_lib.py v2_draw.py v2_scenes.py build_video.py cf_run.py cf_evalwide.py plans_pack.py controls.py notes_gen.py qa_one.py`, `code/` (tip 9a6551d5 export, 38 MB), `_t_s_card.png` (preview of the S-age chart).")
open(f"{OUT}/VIDEO_NOTES.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("written", sum(len(x) for x in L), "chars")
