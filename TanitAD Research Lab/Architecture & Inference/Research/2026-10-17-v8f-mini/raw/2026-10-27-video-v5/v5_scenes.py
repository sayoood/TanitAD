"""v5_scenes.py -- the scenes of chapter 8, "DAgger: trained on its own states" (new in video v5).  Each scene returns an RGB uint8 [1080, 1920, 3].

Everything drawn here comes from (a) the float64 T1-kin dumps of the four decisive runs (DAG / DAG-0 x seed 0 / 1: tracks, per-window ADE6, the histogram), (b) their per-tick files
and label cells (S's checkpoints, T's decisions, the route-matched labels), via v5_lib, or (c) a pointer into data/DAG_READ.json (NJ / BJ / TJ), registered for CONTROLS_V5.
No model is imported, no number is typed in by hand except the declared constants (B = 2,000, 141 episodes, 2,399 windows, m = 10, 6 s, ticks at 0 / 2 / 4 s ...).
Wording rule: never 'closed loop' except inside 'NOT closed loop'; never 'drives' / 'driving' / 'autonomous'.
"""
from __future__ import annotations

import numpy as np

import v3_lib as L
from v4_scenes import *  # noqa: E402,F401,F403  (helpers, colours, render4, TIER_K ...)
from v4_scenes import render4, LAST_LINES, _bars, plan_t1  # noqa: E402,F401
from v3_scenes import SCENES  # noqa: E402
import v3_draw as DR  # noqa: E402
import v5_lib as V  # noqa: E402  (after the star imports: v4_scenes exports its own `V` = v4_lib)

C_DAG, C_DAG0 = "#00838f", "#8a8f98"
ARM_COL = {"DAG": C_DAG, "DAG0": C_DAG0}
ARM_TXT = {"DAG": "DAG  (p = 0.1: own states in the batch)", "DAG0": "DAG-0  (p = 0: the same fine-tune, no own states)"}
EST5 = ("Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows, DAG - DAG-0 per training seed (answers 'another draw of EPISODES' only); "
        "another TRAINING run is answered by the two seeds and the replicate floor RF; the planner is deterministic.")
S_COL, T_COL, O_COL = LV_COL["S"], LV_COL["T"], LV_COL["O"]
STAT_COL = {"match": C_OK, "MISS": C_BAD, "no label": C_NS}
TICK_T = ("+0.0 s", "+2.0 s", "+4.0 s")


# ================================================================================================ 8a. what DAgger changes
def d5_intro():
    fig = new_fig(DARK)
    tx(fig, 90, 56, "8   DAgger: trained on its own states", 40, "white", "bold")
    tx(fig, 90, 120, "Does a model that is fine-tuned on the states IT reaches make its tactical decisions survive its own motion?", 20, "#e8ecf2")
    tx(fig, 90, 160, "tier T1-kin = self-action OPEN loop  |  NOT closed loop: the cameras are the human's recorded frames, nothing re-renders the scene", 14, "#ffd9b3")
    p = V.NJ("dagger_p", "{:g}")
    nw, e0, e1 = V.NJ("bank/s0/header/train/n_windows", "{:,}"), V.NJ("bank/s0/coverage/eligible_fraction", "{:.1f}", 100), V.NJ("bank/s1/coverage/eligible_fraction", "{:.1f}", 100)
    st_s, lr = V.AV("DAG", "--steps", "{:,.0f}"), V.AV("DAG", "--lr")
    boxes = [("WHAT DAGGER CHANGES", C_DAG,
              f"During the fine-tune a fraction p = {p} of the training samples is REPLACED by a state the model itself reached: its own rolled-out ego state (speed, history, odometry, route countdown) "
              "at 2 s or 4 s into a T1-kin rollout.\n\n"
              f"The states come from a bank of the starting checkpoint's own rollouts over the TRAINING clips ({nw} windows per seed; {e0} % / {e1} % of the windows have an eligible own state, seed 0 / seed 1).\n\n"
              "The label stays the HUMAN's: the 6 s future and the tactical labels at the route point the own state matches.  The images stay the human's logged frames: nothing is re-rendered."),
             ("THE CONTROL, DAG-0", C_DAG0,
              f"The same fine-tune with p = 0: same init (the M2b checkpoint), {st_s} steps, learning rate {lr}, same data, same seed.\n\n"
              "It separates 'trained on its own states' from 'trained for more steps': the effect of the lever is DAG minus DAG-0, never DAG against the old checkpoint.\n\n"
              "Both arms, two training seeds (s0, s1), are run at T1-kin on the 141 evaluation episodes (2,399 windows) and read by a rule committed before the runs (PREREG_V8F_DAGGER)."),
             ("WHAT YOU WILL SEE", "#D55E00",
              "Seven windows chosen by a rule fixed before any chapter-8 frame was rendered (next card).\n\n"
              "For each: DAG and DAG-0 side by side, each on its OWN motion, at the replan ticks (+0, +2, +4 s).  S: route checkpoints, 30 / 50 / 80 m ahead.  T: the lateral and longitudinal decision (head argmax) against the label.  "
              "O: the path it executes, against the human's.\n\n"
              "Then the distribution over all 2,399 windows, the pre-registered T-head rows, and the guards.  Where DAG is worse, the caption says so.")]
    for i, (ttl, col, body) in enumerate(boxes):
        x = 70 + i * 600
        ax = axpx(fig, x, 230, 560, 640, W, H)
        ax.set_facecolor("#18202a"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_edgecolor(col); s_.set_linewidth(2.6)
        tx(fig, x + 20, 246, ttl, 19, col, "bold")
        tx(fig, x + 20, 292, wrap(body, 39), 15.6, "#e8ecf2", ls=1.38)
    tx(fig, 70, 900, wrap(EST5, 180), 12, "#aeb7c4")
    tx(fig, 70, 956, wrap("Baseline: DAG-0 is the same fine-tune without the lever, not the state of the art; no external benchmark is shown.  Evidence class of every number in this chapter: MEASURED (ours), from the float64 T1-kin dumps, "
                          "the per-tick files, and the pre-registered read data/DAG_READ.json.", 180), 12, "#aeb7c4")
    return render4(fig)


# ================================================================================================ 8b. the selection rule
def d5_rule():
    fig = new_fig()
    header(fig, "8  DAgger: which windows are shown, and why", tier=TIER_K, sub="rule fixed BEFORE any chapter-8 frame was rendered  |  SELECTION_V5.md, select_scenes5.py (it reads g and cl of the two seed-0 runs, nothing else)")
    sj = V._SEL
    rule = "THE RULE   (sha256 " + sj["rule_sha256"][:16] + "...)\n\n" + sj["rule_text"]
    tx(fig, 24, 100, wrap(rule, 68), 15.2, INK, ls=1.36)
    tx(fig, 940, 100, "The seven windows, in the order shown", 17, INK, "bold")
    hdr = f"{'#':<3}{'tag':<9}{'stratum':<10}{'eval win':>9}{'clip':>10}{'t (s)':>7}{'m/s':>7}{'delta s0 (m)':>14}"
    tx(fig, 940, 140, hdr, 12.5, MUTED, mono=True)
    for i, s in enumerate(V.SCN):
        d = V.dlt_reg(0, i, "{:+.4f}")
        row = f"{i + 1:<3}{s['tag']:<9}{s['stratum']:<10}{'#' + str(s['grid_row']):>9}{s['clip'][:8]:>10}{0.2 * s['ws']:>7.1f}{s['gt_mean_speed']:>7.1f}{d:>14}"
        tx(fig, 940, 172 + i * 34, row, 12.5, INK, "bold", mono=True)
    tx(fig, 940, 424, wrap("delta = ADE6(DAG) - ADE6(DAG-0) of the window at seed 0, T1-kin (negative: DAG closer to the human's path): the quantity the rule sorts by, printed here for the record.  "
                           "Draw positions in the candidate list: " + ", ".join(str(p) for p in sj["draw_positions_in_candidates"]) + ".  Seed 1 is shown beside in every window and was not used to choose.", 78), 11.6, MUTED)
    tx(fig, 24, 600, "Why a rule, and why these windows are NOT a sample of the typical window", 16, INK, "bold")
    tx(fig, 24, 636, wrap("A median window shows the TYPICAL size of the change in its stratum, whichever way it goes, so a window where DAG is worse is as likely to be picked as one where it is better; the WORST window is "
                          "the largest increase over all 2,399 windows, kept because a video that shows only wins is not evidence; the three draws are random among windows that move (at least 5 m/s).  "
                          "A window is an example; the distribution card and the result cards are the evidence.", 150), 14, INK, ls=1.35)
    sc = sj["strata_counts"]
    assert [c["stratum"] for c in V.SCN] == ["JUNCTION", "CURVE", "NO-TURN", "NO-TURN", "NO-TURN", "NO-TURN", "NO-TURN"] and sc["NO-TURN"] > max(sc["JUNCTION"], sc["CURVE"])
    tx(fig, 24, 790, wrap(f"Strata (from the human's labels): JUNCTION {sc['JUNCTION']:,} windows, CURVE {sc['CURVE']:,}, NO-TURN {sc['NO-TURN']:,} (2,399 in all).  Candidates for the three draws: {sj['n_candidates_draw']:,}.  "
                          "The median windows are one JUNCTION and one CURVE and one NO-TURN window; the WORST window and the three draws all fall in NO-TURN, by far the largest stratum.", 150), 13.4, INK, ls=1.35)
    tx(fig, 24, 868, wrap("What the rule cannot do: it cannot make seven windows a sample.  The distribution card shows all 2,399, and the two result cards carry the decision-grade rows.", 150), 13.4, MUTED, ls=1.35)
    caption(fig, "Each window is shown with DAG on the left and DAG-0 on the right, both at training seed 0 in the picture, both reading the SAME logged camera frames, each running on its OWN executed motion.  "
                 "The per-window ADE6 of both seeds is in the table and the caption at the end of each window.", size=16, y0=934, h=106, n=150)
    footer(fig, "Tier T1-kin: self-action OPEN loop, NOT closed loop.  " + EST5)
    return render4(fig)


# ================================================================================================ 8c. one window
BEV_BOX = {"DAG": (700, 100, 550, 640), "DAG0": (1340, 100, 550, 640)}


def _ext5(i):
    """BEV extent: the paths of both arms and the human, AND every S checkpoint (of both arms, all three ticks) and the human's route checkpoints, so nothing is cut."""
    arrs = [V.gt_track(i), V.o_track("DAG", 0, i), V.o_track("DAG0", 0, i)]
    for arm in V.ARMS:
        for k in range(3):
            ck, gck, ok = V.s_set(arm, 0, i, k)
            arrs.append(ck[:, :2])
            if ok.any():
                arrs.append(gck[ok][:, :2])
    return fit_extent(arrs)


def _bev_arm(fig, i, arm, f, ext):
    ax = bev_axes(fig, BEV_BOX[arm], ext, title=ARM_TXT[arm].split("  (")[0] + f"  (seed 0): {'p = 0.1' if arm == 'DAG' else 'p = 0'}")
    for s_ in ax.spines.values():
        s_.set_edgecolor(ARM_COL[arm]); s_.set_linewidth(2.6)
    gt, o = V.gt_track(i), V.o_track(arm, 0, i)
    plan_t1(ax, gt, 30, f"{arm}:GT", C_GT, 3.4, "-", "GT: the human's path", 3, ms=5, full=True)
    plan_t1(ax, o, f, f"{arm}:O", O_COL, 3.8, "-", "O: the executed path", 8, ms=4.6, annotate=True)
    ax.plot([], [], "o", ms=11, mfc="none", mec=INK, mew=2.0, label="replan tick (2 s, 4 s)")
    k = min(f // 10, 2)
    ck, gck, ok = V.s_set(arm, 0, i, k)
    ax.plot(-ck[:, 1], ck[:, 0], "D", ms=10, mfc=S_COL, mec="white", mew=1.0, zorder=7, clip_on=True, gid=f"{arm}:S", label="S: checkpoints, 30/50/80 m")
    for q in range(3):
        ax.annotate("", xy=(-(ck[q, 1] + 3.6 * np.sin(ck[q, 2])), ck[q, 0] + 3.6 * np.cos(ck[q, 2])), xytext=(-ck[q, 1], ck[q, 0]), arrowprops=dict(arrowstyle="-|>", color=S_COL, lw=1.3), zorder=7)
    ax.plot(-gck[ok, 1], gck[ok, 0], "D", ms=10, mfc="none", mec=INK, mew=1.8, zorder=6, clip_on=True, gid=f"{arm}:SGT", label="S GT: human's route")
    legend(ax, "lower left", 7.8)
    return ax


def _tick_box(fig, x, y, w, h, arm, i, k, f):
    cur = min(f // 10, 2) == k
    reached = f >= 10 * k
    panel_box(fig, x, y, w, h, ec=ARM_COL[arm] if cur else "#c9ced6", lw=2.8 if cur else 1.0, fc="white" if reached else "#eef0f3")
    tx(fig, x + 7, y + 5, f"tick {k}   {TICK_T[k]}", 9.8, INK if reached else "#9aa3af", "bold")
    tx(fig, x + 7, y + 21, "logged state" if k == 0 else "OWN state", 8.8, MUTED if reached else "#9aa3af")
    if not reached:
        tx(fig, x + 7, y + 48, "not reached yet", 9.6, "#9aa3af")
        return
    se = V.serr_reg(arm, 0, i, k)
    tx(fig, x + 7, y + 38, "S pos. error 30/50/80 m", 8.2, MUTED)
    tx(fig, x + 7, y + 51, "/".join(se) + " m", 9.6, S_COL, "bold", mono=True)
    for q, hd in enumerate(("lat", "lon")):
        pr, tr, st = V.tinfo(arm, 0, i, k, hd)
        yy = y + 76 + q * 36
        tx(fig, x + 7, yy, f"T {hd}: {pr}", 9.8, T_COL, "bold")
        tx(fig, x + 7, yy + 15, (f"label {tr}  {'OK' if st == 'match' else st}" if tr else "no label at this tick"), 8.8, STAT_COL[st], "bold")


def d5_frame(i, f):
    s = V.SCN[i]
    tag = s["tag"]
    fig = new_fig()
    header(fig, f"8  DAgger: trained on its own states  |  window {i + 1} of 7: {tag}  ({s['stratum']})", tier=TIER_K,
           sub=f"DAG vs DAG-0, each in its own loop, seed 0 in the picture  |  clip {s['clip'][:8]}  |  eval window #{s['grid_row']}  |  t = {0.2 * s['ws']:.1f} s  |  GT mean speed {s['gt_mean_speed']:.1f} m/s  |  replan every 2 s")
    tick = f % 10 == 0 and f < 30
    camera(fig, (24, 112, 600, 240), V.frame_for(i, f), title="the LOGGED camera frame now (both arms read it)",
           sub=f"frame at +{0.2 * f:.1f} s  |  the model reads the logged frame at each replan tick" + ("   <- a replan tick" if tick else ""), ec=ARM_COL["DAG"] if tick else INK)
    panel_box(fig, 24, 388, 600, 140, ec="#c9ced6")
    for q, (lv, col, txt) in enumerate((("S", S_COL, "strategic: three route checkpoints 30 / 50 / 80 m ahead (filled = the model's, hollow = the human's route)"),
                                         ("T", T_COL, "tactical: lateral and longitudinal decision (head argmax) against the label (boxes under each panel)"),
                                         ("O", O_COL, "operative: the 6 s path the model EXECUTES (orange) against the human's path (black)"))):
        tx(fig, 36, 398 + q * 43, lv, 17, col, "bold")
        tx(fig, 66, 398 + q * 43, wrap(txt, 62), 9.8, INK, ls=1.18)
    ext = _ext5(i)
    for arm in V.ARMS:
        _bev_arm(fig, i, arm, f, ext)
        x0 = BEV_BOX[arm][0]
        for k in range(3):
            _tick_box(fig, x0 + k * 187, 778, 176, 150, arm, i, k, f)
    nt = int(f >= 10) + int(f >= 20)
    tx(fig, 24, 538, f"time: +{0.2 * f:.1f} s of 6.0 s   |   replans done: {nt} of 2", 12.5, INK, "bold")
    tx(fig, 24, 560, wrap("ring = a 2 s replan tick: the model rebuilds its state from its own executed motion there.  Forward-up BEV, ego frame at +0.0 s; dots every 1 s.", 80), 10.2, MUTED)
    if f == 30:
        tx(fig, 24, 606, "this window, T1-kin, per-window values, no interval", 11.5, INK, "bold")
        tx(fig, 24, 628, f"{'':<26}{'seed 0':>9}{'seed 1':>9}", 11.0, MUTED, mono=True)
        ad = {(a, sd): V.ade_reg(a, sd, i) for a in V.ARMS for sd in V.SEEDS5}
        dl = {sd: V.dlt_reg(sd, i) for sd in V.SEEDS5}
        pl = {(a, sd): V.plen_reg(a, sd, i) for a in ("GT",) + V.ARMS for sd in V.SEEDS5}
        rows = [("ADE6 DAG (m)", C_DAG, [ad[("DAG", sd)] for sd in V.SEEDS5]), ("ADE6 DAG-0 (m)", C_DAG0, [ad[("DAG0", sd)] for sd in V.SEEDS5]), ("DAG - DAG-0 (m)", INK, [dl[sd] for sd in V.SEEDS5]),
                ("path 6 s, human (m)", INK, [pl[("GT", sd)] for sd in V.SEEDS5]), ("path 6 s, DAG (m)", C_DAG, [pl[("DAG", sd)] for sd in V.SEEDS5]), ("path 6 s, DAG-0 (m)", C_DAG0, [pl[("DAG0", sd)] for sd in V.SEEDS5])]
        for hd, nm in (("lat", "T lat right"), ("lon", "T lon right")):
            for a in V.ARMS:
                rows.append((f"{nm}, {V.ARM_LAB[a]}", ARM_COL[a], [V.tcount_reg(a, sd, i, hd) for sd in V.SEEDS5]))
        for r, (lab, col, vals) in enumerate(rows):
            tx(fig, 24, 650 + r * 21, f"{lab:<26}{vals[0]:>9}{vals[1]:>9}", 11.0, col, "bold", mono=True)
        tx(fig, 24, 650 + len(rows) * 21 + 4, "T right = correct decisions among the scored ticks 1-2 (own states)", 9.4, MUTED)
    else:
        tx(fig, 24, 606, "the window's numbers appear when the 6 s are complete", 11.5, MUTED)
    if f < 30:
        cap = (f"Window {tag}, running: the human's path (black), each arm's executed path (orange) and S's checkpoints (pink diamonds, hollow = the human's route); in the boxes T's decisions against the label at the replan ticks "
               f"(+0.0, +2.0, +4.0 s), at +{0.2 * f:.1f} s.  Left DAG (p = 0.1), right DAG-0 (p = 0); both read the same logged frames and run on their own executed motion.")
    else:
        nb = 0
        parts = []
        for sd in V.SEEDS5:
            a, b, d = V.ade_reg("DAG", sd, i), V.ade_reg("DAG0", sd, i), V.dlt_reg(sd, i)
            nb += int(V.ade6("DAG", sd, i) < V.ade6("DAG0", sd, i))
            parts.append(f"s{sd}: DAG {a} vs DAG-0 {b} ({d})")
        verdict = ("DAG is closer to the human's path than DAG-0 at both seeds." if nb == 2 else
                   "DAG-0 is closer to the human's path at both seeds in this window (kept as it is)." if nb == 0 else
                   "the two seeds disagree in this window (kept as it is).")
        cap = (f"Window {tag} ({s['stratum']}): ADE6 (m), T1-kin, per-window value, no interval.  " + "  |  ".join(parts) + ".  " + verdict + "  One window is an example; the cards carry the evidence.")
    caption(fig, cap, size=15.4, y0=934, h=106, n=156)
    footer(fig, "Tracks: float64 dumps v8f-dagger-t1/dump_{DAG,DAG0}_s<seed>_m10_causal_heads/ep*.npz (g, cl); S checkpoints and T logits: ticks/ticks_*.npz; labels and S's GT: data/dag_cells_*.npz from the programme's own reader t1hd_read.py "
                "(CONTROLS_V5).  Frames: v2ep cache step = window start + 0.2 s per step.  Selection rule: SELECTION_V5.md.  Tier T1-kin, NOT closed loop.")
    return render4(fig)


# ================================================================================================ 8d. the distribution
def d5_hist():
    fig = new_fig()
    header(fig, "8  DAgger: per-window DAG - DAG-0 ADE6 over all 2,399 windows", tier=TIER_K, sub="each bar = windows; left of zero DAG's executed track is closer to the human's than DAG-0's  |  descriptive: windows of one episode are not independent")
    pos = {0: (60, 110), 1: (990, 110)}
    wi = [i for i, c in enumerate(V.SCN) if c["tag"] == "WORST"][0]
    for sd, (x, y) in pos.items():
        d = V.per_window_delta(sd)
        ax = axpx(fig, x + 30, y + 40, 780, 470, W, H)
        ax.set_facecolor(PANEL)
        h = _bars(ax, d, C_DAG, "#9aa0aa")
        ax.axvline(0, color=INK, lw=1.4)
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, h.max() * 1.22)
        ax.set_xlabel("DAG - DAG-0 ADE6 of the window (m), lower is better; |value| > 6 m is drawn in the end bins", fontsize=9)
        ax.set_ylabel("windows", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", color="#e8ebef")
        nm, nf, nt = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
        pm, pf = 100 * nm / len(d), 100 * nf / len(d)
        sm = V.mm(float(d.mean()), "{:.4f}")
        L.REG.append({"kind": "hist5", "scene": L.CUR["scene"], "seed": sd, "n_dag": nm, "n_dag0": nf, "n_tie": nt, "mean": sm, "n": len(d)})
        ax.set_title(f"seed {sd}", fontsize=12, fontweight="bold", loc="left")
        ax.text(0.985, 0.97, f"DAG better: {nm:,} windows ({pm:.1f} %)\nDAG-0 better: {nf:,} windows ({pf:.1f} %)" + (f"\ntied: {nt:,}" if nt else "") + f"\nmean over windows: {sm} m",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5, family="DejaVu Sans Mono", color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c9ced6"))
        for i, s in enumerate(V.SCN):
            di = float(d[s["grid_row"]])
            ax.plot([np.clip(di, -5.9, 5.9)], [0], "^", ms=11, mfc=C_GT, mec="white", mew=0.8, zorder=9, clip_on=False)
            ax.text(float(np.clip(di, -5.9, 5.9)), h.max() * (0.045 + 0.06 * (i % 3)), f"{i + 1}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=C_GT)
        L.REG.append({"kind": "ref5", "scene": L.CUR["scene"], "seed": sd, "ptr": f"rows/s{sd}/R-ADE/delta", "mean": sm})
        tx(fig, x + 30, y + 566, wrap(f"The mean of the bars is the point estimate of the paired read: {V.NJ(f'rows/s{sd}/R-ADE/delta', '{:+.4f}')} m, interval [{V.NJ(f'rows/s{sd}/R-ADE/lo', '{:+.4f}')}, {V.NJ(f'rows/s{sd}/R-ADE/hi', '{:+.4f}')}] m "
                                      f"(mean ADE6 DAG {V.NJ(f'rows/s{sd}/R-ADE/mean_a', '{:.3f}')} m, DAG-0 {V.NJ(f'rows/s{sd}/R-ADE/mean_b', '{:.3f}')} m).", 90), 12, INK, ls=1.3)
    tx(fig, 90, 770, wrap(f"Triangle 4 at the right edge is the WORST window ({V.SCN[wi]['tag']}, eval window #{V.SCN[wi]['grid_row']}): DAG - DAG-0 = {V.dlt_reg(0, wi, '{:+.2f}')} m at seed 0, drawn in the end bin.  It is shown in the chapter, not hidden.", 200), 12.6, C_BAD, "bold", ls=1.3)
    d0 = V.per_window_delta(0)
    n_lose7 = sum(int(V.ade6("DAG", 0, i) > V.ade6("DAG0", 0, i)) for i in range(len(V.SCN)))
    assert (d0 < 0).sum() > (d0 > 0).sum() > 0 and 0 < n_lose7 < len(V.SCN) and wi == 3
    L.REG.append({"kind": "cnt7", "scene": L.CUR["scene"], "seed": 0, "shown": str(n_lose7)})
    caption(fig, f"Per-window DAG - DAG-0 ADE6 (m), T1-kin, both seeds.  At seed 0 DAG wins {100 * (d0 < 0).mean():.1f} % of the windows and loses {100 * (d0 > 0).mean():.1f} %: the losses are the grey bars and the right tail, "
                 f"and {n_lose7} of the seven windows shown are among them (triangles 1-7 = the seven windows of this chapter; their position was not used to choose them).  "
                 "Descriptive only: the decision-grade intervals are on the next cards.", size=15.2, y0=934, h=106, n=158)
    footer(fig, "Source: float64 dumps v8f-dagger-t1/dump_{DAG,DAG0}_s<seed>_m10_causal_heads (g, cl; ADE6 = mean over the 30 knots of the L2 distance to the GT), all 2,399 windows, per seed; the interval and the mean ADE6 are DAG_READ.json "
                "rows R-ADE.  " + EST5 + "  Tier T1-kin, NOT closed loop.")
    return render4(fig)


# ================================================================================================ 8e. the pre-registered T-head rows
def d5_tcard():
    fig = new_fig()
    header(fig, "8  DAgger: result card, T's decisions on its own states  |  DAG - DAG-0", tier=TIER_K,
           sub="macro-F1 of T's lateral and longitudinal head at ticks 1 and 2 (the model's OWN states), matched cells  |  the pre-registered rows R-TLAT, R-TLON  |  source data/DAG_READ.json")
    ax = axpx(fig, 230, 150, 740, 320, W, H)
    ax.set_facecolor(PANEL)
    rows = [("lateral", "R-TLAT", 0), ("lateral", "R-TLAT", 1), ("longitudinal", "R-TLON", 0), ("longitudinal", "R-TLON", 1)]
    ax.set_xlim(-0.04, 0.34)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for k, (nm, rk, sd) in enumerate(rows):
        y = len(rows) - 1 - k
        pt = f"rows/s{sd}/{rk}"
        v, lo, hi, bar = (L.getp(V.DR, f"{pt}/{x}") for x in ("delta", "lo", "hi", "bar"))
        clears = V.BJ(f"{pt}/clears")
        ax.plot([lo, hi], [y, y], "-", color=C_DAG, lw=4.0, solid_capstyle="round")
        ax.plot([v], [y], "o", color=C_DAG, ms=11, mec="white", mew=1.2, zorder=5)
        ax.plot([bar, bar], [y - 0.32, y + 0.32], "--", color=C_BAD, lw=2.0)
        ax.text(hi + 0.006, y + 0.04, f"{V.NJ(pt + '/delta')} [{V.NJ(pt + '/lo')}, {V.NJ(pt + '/hi')}]", fontsize=10.2, va="bottom", color=INK, family="DejaVu Sans Mono")
        ax.text(hi + 0.006, y - 0.06, f"bar {V.NJ(pt + '/bar', '{:.4f}')}  " + ("CLEARS" if clears else "does not clear"), fontsize=9.6, va="top", color=C_OK if clears else C_BAD, fontweight="bold")
        ax.text(-0.045, y, f"{nm}, seed {sd}", fontsize=10.5, va="center", ha="right", color=INK, fontweight="bold", clip_on=False)
    ax.axvline(0, color=INK, lw=1.4)
    ax.set_yticks([])
    ax.set_xlabel("T head macro-F1, DAG - DAG-0 (higher is better); dashed = the pre-registered bar of that seed and head", fontsize=9.2)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", color="#e8ebef")
    ax.set_title("R-TLAT / R-TLON, T1-kin, ticks 1-2, paired episode-cluster bootstrap, B = 2,000", fontsize=10.8, fontweight="bold", loc="left")
    all_clear = all(V.BJ(f"rows/s{sd}/{rk}/clears") for _, rk, sd in rows)
    all_sep = all(V.BJ(f"rows/s{sd}/{rk}/separated") for _, rk, sd in rows)
    tx(fig, 24, 530, "macro-F1 at ticks 1-2, matched cells (the cells both runs scored)", 12.5, INK, "bold")
    tx(fig, 24, 556, f"{'row':<20}{'DAG':>9}{'DAG-0':>9}{'DAG - DAG-0':>14}{'bar':>9}{'RF':>9}", 11.4, MUTED, mono=True)
    for r, (nm, rk, sd) in enumerate(rows):
        pt = f"rows/s{sd}/{rk}"
        line = f"{nm + ', s' + str(sd):<20}{V.NJ(pt + '/f1_a', '{:.4f}'):>9}{V.NJ(pt + '/f1_b', '{:.4f}'):>9}{V.NJ(pt + '/delta'):>14}{V.NJ(pt + '/bar', '{:.4f}'):>9}{V.NJ(pt + '/RF', '{:.4f}'):>9}"
        tx(fig, 24, 584 + r * 27, line, 11.4, INK, "bold", mono=True)
    tx(fig, 24, 706, wrap("A head CLEARS at a seed iff its interval is separated-better AND the point is at least the bar max(2F, 2 RF): F is the T0 fresh-seed floor, RF the run-to-run gap between DAG-0 and ORC-0, "
                          "a same-argv replicate of DAG-0 trained by another build.  " + ("All four rows clear" if all_clear else "NOT all four rows clear") + ".  Pre-registered condition for ROBUST: " + V.TJ("OUTCOME/condition") + ".", 84), 11.8, INK, ls=1.32)
    tx(fig, 24, 838, wrap("OUTCOME (committed before the runs): " + V.TJ("OUTCOME/OUTCOME") + ".  Committed next: " + V.TJ("OUTCOME/committed_next"), 84), 11.4, C_OK, "bold", ls=1.3)
    tx(fig, 990, 98, "Where the change is: per-class F1 at ticks 1-2", 13, INK, "bold")
    tx(fig, 990, 120, "own cells of each run; derived here from the cells, beside, not in the read", 10.6, MUTED)
    for (hd, sd), (x, y) in {("lat", 0): (1000, 150), ("lat", 1): (1450, 150), ("lon", 0): (1000, 530), ("lon", 1): (1450, 530)}.items():
        names = LAT5 if hd == "lat" else LON7
        n = len(names)
        axc = axpx(fig, x, y + 30, 420, 270, W, H)
        axc.set_facecolor(PANEL)
        st = {a: V.class_f1(a, sd, hd) for a in V.ARMS}
        wd = 0.38
        for q, a in enumerate(V.ARMS):
            f1c = np.nan_to_num(st[a][1], nan=0.0)
            axc.bar(np.arange(n) + (q - 0.5) * wd, f1c, width=wd, color=ARM_COL[a], ec="white", label=V.ARM_LAB[a])
            for c in range(n):
                shown = "n/a" if not np.isfinite(st[a][1][c]) else f"{st[a][1][c]:.2f}"
                L.REG.append({"kind": "f1cls", "scene": L.CUR["scene"], "arm": a, "seed": sd, "head": hd, "cls": names[c], "shown": shown, "support": int(st[a][2][c])})
                axc.text(c + (q - 0.5) * wd, np.nan_to_num(st[a][1][c], nan=0.0) + 0.012, shown, ha="center", va="bottom", fontsize=6.3, rotation=90, color=INK)
        axc.set_ylim(0, 1.22)
        axc.set_xlim(-0.6, n - 0.4)
        axc.set_xticks(range(n))
        axc.set_xticklabels([c.replace("LANE_", "") for c in names], fontsize=7.2)
        axc.tick_params(labelsize=7.6, length=2)
        axc.grid(axis="y", color="#e8ebef")
        axc.set_title(f"{'lateral' if hd == 'lat' else 'longitudinal'} head, seed {sd}", fontsize=10.2, fontweight="bold", loc="left")
        axc.legend(loc="upper right", fontsize=7.6, framealpha=0.95, ncol=2)
        sup = "  ".join(f"{c.replace('LANE_', '')} {int(st['DAG'][2][k])}" for k, c in enumerate(names))
        L.REG.append({"kind": "support", "scene": L.CUR["scene"], "seed": sd, "head": hd, "shown": sup})
        tx(fig, x, y + 340, wrap("labelled cells per class (DAG): " + sup, 64), 8.6, MUTED, ls=1.2)
    caption(fig, "T's lateral and longitudinal decisions on the model's OWN states, DAG - DAG-0, macro-F1 at ticks 1-2, T1-kin, paired episode-cluster bootstrap (B = 2,000, 141 episodes), seeds 0 and 1: lateral "
                 + V.NJ("rows/s0/R-TLAT/delta") + " / " + V.NJ("rows/s1/R-TLAT/delta") + ", longitudinal " + V.NJ("rows/s0/R-TLON/delta") + " / " + V.NJ("rows/s1/R-TLON/delta")
                 + "; " + ("every row is separated-better and clears its bar" if (all_clear and all_sep) else "NOT every row is separated-better and clearing its bar") + ".  An interval answers 'another draw of episodes'; "
                 "the two seeds and the replicate floor answer 'another training run'.", size=14.6, y0=934, h=106, n=164)
    footer(fig, "Source: data/DAG_READ.json (sha256 3393fdd5, the pre-registered read, PREREG_V8F_DAGGER.md at 2a78f059 incl. A1, A2); per-class F1: own implementation of the programme's macro-F1 on data/dag_cells_*.npz "
                "(its macro value reproduces the read's own-cell F1 exactly, CONTROLS_V5 F1).  Tier T1-kin, NOT closed loop.")
    return render4(fig)


# ================================================================================================ 8f. guards, S, beside
def d5_gcard():
    fig = new_fig()
    header(fig, "8  DAgger: guards, S, and what is still open", tier=TIER_K, sub="DAG - DAG-0 per seed  |  guards R-ADE and R-T0 are pre-registered; S-POS and tick-0 skill are BESIDE rows (decide nothing)  |  source data/DAG_READ.json")
    tx(fig, 24, 100, "Guards (pre-registered)", 15, INK, "bold")
    tx(fig, 24, 130, f"{'row':<28}{'seed 0':>34}{'seed 1':>34}", 11.0, MUTED, mono=True)
    ga = [f"{V.NJ(f'rows/s{sd}/R-ADE/delta')} [{V.NJ(f'rows/s{sd}/R-ADE/lo')}, {V.NJ(f'rows/s{sd}/R-ADE/hi')}]" for sd in (0, 1)]
    gt_ = [f"{V.NJ(f'rows/s{sd}/R-T0/delta')} [{V.NJ(f'rows/s{sd}/R-T0/lo')}, {V.NJ(f'rows/s{sd}/R-T0/hi')}]" for sd in (0, 1)]
    tx(fig, 24, 156, f"{'R-ADE  T1-kin ADE6 (m)':<28}{ga[0]:>34}{ga[1]:>34}", 11.0, C_OK, "bold", mono=True)
    tx(fig, 24, 180, f"{'R-T0   T0 own-code ADE6 (m)':<28}{gt_[0]:>34}{gt_[1]:>34}", 11.0, INK, "bold", mono=True)
    f2 = V.NJ("bars/F_over_2_t0_point_m", "{:.4f}")
    assert V.BJ("rows/s0/R-ADE/better") and V.BJ("rows/s1/R-ADE/better") and not V.BJ("rows/s0/R-T0/separated") and V.BJ("rows/s1/R-T0/worse")
    assert float(L.getp(V.DR, "rows/s1/R-T0/point")) <= float(L.getp(V.DR, "bars/F_over_2_t0_point_m"))
    tx(fig, 24, 214, wrap(f"R-ADE: the guard is 'not separated-worse'.  Both seeds are separated BETTER: mean ADE6 DAG {V.NJ('rows/s0/R-ADE/mean_a', '{:.3f}')} vs DAG-0 {V.NJ('rows/s0/R-ADE/mean_b', '{:.3f}')} m (seed 0), "
                          f"{V.NJ('rows/s1/R-ADE/mean_a', '{:.3f}')} vs {V.NJ('rows/s1/R-ADE/mean_b', '{:.3f}')} m (seed 1).", 98), 12, INK, ls=1.32)
    tx(fig, 24, 264, wrap(f"R-T0 (teacher-forced own code, the guard tier): the guard is 'point at most F/2 = {f2} m'.  Seed 0: point {V.NJ('rows/s0/R-T0/point')} m, not separated.  "
                          f"Seed 1: point {V.NJ('rows/s1/R-T0/point')} m, SEPARATED WORSE, but under the bar: the guard held, and the cost is visible.", 98), 12, C_BAD, "bold", ls=1.32)
    tx(fig, 24, 336, wrap("Guards held at both seeds: " + ("yes" if V.BJ("OUTCOME/flags/guards_held") else "NO") + ".  A guard that held is not an improvement: DAG's T0 cost at seed 1 stays on the card.", 98), 12, INK, ls=1.32)
    # the guard rows as intervals
    axg = axpx(fig, 250, 716, 700, 150, W, H)
    axg.set_facecolor(PANEL)
    grows = [("R-ADE, T1-kin, seed 0", "R-ADE", 0), ("R-ADE, T1-kin, seed 1", "R-ADE", 1), ("R-T0, T0, seed 0", "R-T0", 0), ("R-T0, T0, seed 1", "R-T0", 1)]
    for k, (lab, rk, sd) in enumerate(grows):
        y = len(grows) - 1 - k
        pt = f"rows/s{sd}/{rk}"
        v, lo, hi = (L.getp(V.DR, f"{pt}/{x}") for x in ("delta", "lo", "hi"))
        axg.plot([lo, hi], [y, y], "-", color=C_OK if rk == "R-ADE" else INK, lw=3.6, solid_capstyle="round")
        axg.plot([v], [y], "o", color=C_OK if rk == "R-ADE" else INK, ms=9, mec="white", mew=1.0, zorder=5)
        axg.text(-0.995, y, lab, fontsize=9.6, va="center", ha="right", color=INK, fontweight="bold", clip_on=False)
    fb = float(L.getp(V.DR, "bars/F_over_2_t0_point_m"))
    axg.plot([fb, fb], [-0.45, 1.45], "--", color=C_BAD, lw=2.0)
    axg.text(fb + 0.012, 1.45, "F/2 bar " + f2, fontsize=9.2, va="top", color=C_BAD, fontweight="bold")
    axg.axvline(0, color=INK, lw=1.2)
    axg.set_xlim(-0.95, 0.22)
    axg.set_ylim(-0.6, len(grows) - 0.4)
    axg.set_yticks([])
    axg.tick_params(labelsize=8.5)
    axg.grid(axis="x", color="#e8ebef")
    axg.set_xlabel("DAG - DAG-0 ADE6 (m), lower is better; R-T0: the point must stay left of the dashed bar", fontsize=8.8)
    # S-POS beside
    tx(fig, 24, 396, "S beside: S's position error against the CTRV baseline, within each run", 13, INK, "bold")
    tx(fig, 24, 418, "(ticks 1-2; unpaired; decides nothing)", 10.6, MUTED)
    tx(fig, 24, 442, f"{'run, arc':<18}{'seed 0':>36}{'seed 1':>36}", 10.8, MUTED, mono=True)
    r = 0
    words = {}
    for arm in ("DAG", "DAG0"):
        for arc in (30, 50, 80):
            cells_ = []
            for sd in (0, 1):
                pt = f"beside/per_seed/s{sd}/per_run_t1heads/{arm}/S-POS vs CTRV (ticks>=1, within this run)/S-POS@{arc}"
                w = V.sep_word(pt)
                cells_.append(f"{V.NJ(pt + '/delta', '{:+.2f}')} [{V.NJ(pt + '/lo', '{:+.2f}')}, {V.NJ(pt + '/hi', '{:+.2f}')}] {w}")
                words[(arm, sd, arc)] = w
            tx(fig, 24, 466 + r * 22, f"{V.ARM_LAB[arm] + ' @' + str(arc) + ' m':<18}{cells_[0]:>36}{cells_[1]:>36}", 10.8, C_DAG if arm == "DAG" else C_DAG0, "bold", mono=True)
            r += 1
    assert all(words[("DAG", sd, a)] == "better" for sd in (0, 1) for a in (30, 50, 80))
    assert [k for k, v in words.items() if k[0] == "DAG0" and v == "better"] == [("DAG0", 0, 80)] and [k for k, v in words.items() if k[0] == "DAG0" and v == "WORSE"] == [("DAG0", 1, 30)]
    tx(fig, 24, 606, wrap("Negative = S's checkpoint is closer to the human's route than the CTRV extrapolation of the run's own motion.  DAG is separated-better at all three arcs and both seeds; DAG-0 only at 80 m at seed 0, and "
                          "separated WORSE at 30 m at seed 1.  STRATEGIC is " + V.TJ("strategic_family", ":") + ": reported beside, decides nothing.", 98), 11.2, INK, ls=1.32)
    # tick-0 skill and crawl, beside
    tx(fig, 1010, 100, "Beside: did DAG lose skill at the LOGGED state (tick 0)?", 13, INK, "bold")
    tx(fig, 1010, 128, f"{'T head macro-F1, tick 0':<26}{'DAG':>9}{'DAG-0':>9}", 11.0, MUTED, mono=True)
    r = 0
    for hd, nm in (("lat", "lateral"), ("lon", "longitudinal")):
        for sd in (0, 1):
            pt = f"beside/per_seed/s{sd}/tick0_f1"
            tx(fig, 1010, 152 + r * 24, f"{nm + ', s' + str(sd):<26}{V.NJ(f'{pt}/DAG/{hd}', '{:.4f}'):>9}{V.NJ(f'{pt}/DAG0/{hd}', '{:.4f}'):>9}", 11.0, INK, "bold", mono=True)
            r += 1
    assert all(L.getp(V.DR, f"beside/per_seed/s{sd}/tick0_f1/DAG/{hd}") < L.getp(V.DR, f"beside/per_seed/s{sd}/tick0_f1/DAG0/{hd}") for sd in (0, 1) for hd in ("lat", "lon"))
    tx(fig, 1010, 256, wrap("DAG is slightly lower than DAG-0 at tick 0 in all four numbers (no interval in the read): the lever may cost a little skill on the logged state while it gains on the model's own states.", 78), 11.4, C_BAD, ls=1.32)
    tx(fig, 1010, 330, "Beside: crawl fraction of ticks 1-2", 13, INK, "bold")
    tx(fig, 1010, 352, "(the model's own speed below the human's by more than 2 m/s; carried per run, not paired)", 10.4, MUTED)
    tx(fig, 1010, 378, f"{'':<14}{'DAG':>9}{'DAG-0':>9}", 11.0, MUTED, mono=True)
    for sd in (0, 1):
        pt = f"beside/per_seed/s{sd}/per_run_t1heads"
        tx(fig, 1010, 402 + sd * 24, f"{'seed ' + str(sd):<14}{V.NJ(f'{pt}/DAG/crawl_fraction_ticks_ge1/lat', '{:.4f}'):>9}{V.NJ(f'{pt}/DAG0/crawl_fraction_ticks_ge1/lat', '{:.4f}'):>9}", 11.0, INK, "bold", mono=True)
    tx(fig, 1010, 462, wrap("A fraction, not a decision row.  The pre-registered next step, if the crawl rows still fail, is DAgger iteration 2.", 78), 11.4, MUTED, ls=1.32)
    tx(fig, 1010, 540, "What it is not", 14.5, C_BAD, "bold")
    nots = ["NOT closed loop: the cameras are the human's recorded frames; T2 (re-perception) is not provisioned.",
            "The images stay logged, so a lateral offset is invisible to the sample: this lever cannot teach lateral recovery (PREREG section 3).",
            "One bank pass, one fine-tune of " + V.AV("DAG", "--steps", "{:,.0f}") + " steps per arm and seed; the next pass (iteration 2) is not run.",
            "NOT COMPUTED in this read: " + V.TJ("beside/not_computed/four families at T0 and T1") + "; per-class F1 on the previous card is derived here, beside.",
            "The control is DAG-0 (same fine-tune, p = 0), not a published state of the art; no external benchmark."]
    tx(fig, 1010, 574, wrap("\n".join("- " + n for n in nots), 80), 12.2, INK, ls=1.36)
    caption(fig, "Guards and beside rows, DAG - DAG-0, T1-kin and T0, seeds 0 and 1.  R-ADE: both seeds separated better, guard held.  R-T0: seed 0 not separated; seed 1 separated worse by " + V.NJ("rows/s1/R-T0/point") +
                 " m, below the F/2 bar of " + f2 + " m, guard held.  S: DAG beats the CTRV baseline at every arc and seed, DAG-0 mostly does not.  DAG loses a little at the logged state (tick 0).", size=14.8, y0=934, h=106, n=162)
    footer(fig, "Source: data/DAG_READ.json (rows R-ADE, R-T0, bars; beside/per_seed/*).  " + EST5 + "  Tier T1-kin (R-ADE, tick-0, S-POS, crawl) and T0 teacher-forced own code (R-T0, the guard tier); NOT closed loop.")
    return render4(fig)


SCENES.update({"d5_intro": d5_intro, "d5_rule": d5_rule, "d5_frame": d5_frame, "d5_hist": d5_hist, "d5_tcard": d5_tcard, "d5_gcard": d5_gcard})
