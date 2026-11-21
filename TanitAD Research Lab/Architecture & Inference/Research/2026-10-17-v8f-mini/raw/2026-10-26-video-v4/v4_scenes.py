"""v4_scenes.py -- the scenes of chapter 7, "T1-kin: the model on its own motion" (new in video v4).  Each scene returns an RGB uint8 [1080, 1920, 3].

Everything drawn here comes from (a) the float64 T1-kin dumps (tracks, per-window ADE6, the histogram) via v4_lib, or (b) a verbatim cell / snippet of one of the four
RESULT files via v4_lib.RC / RT.  No model is imported, no number is typed in by hand except the declared constants (B = 2,000, 141 episodes, 2,399 windows, m = 10, 6 s ...).
Wording rule: never 'closed loop' except inside 'NOT closed loop'; never 'drives' / 'driving' / 'autonomous'.
"""
from __future__ import annotations

import re

import numpy as np

import v3_lib as L
import v4_lib as V
from v3_scenes import *  # noqa: E402,F401,F403  (helpers, colours, SCENES of chapters 1-6)
from v3_scenes import SCENES  # noqa: E402
import v3_draw as DR  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TIER_K = "T1-kin: self-action OPEN loop (NOT closed loop)"
C_MODEL, C_FLOOR, C_GT = PLAN_COL["HACC"], "#6d737d", PLAN_COL["GT"]
C_OK, C_BAD, C_NS = "#00704f", "#b30000", "#5b6470"
EST = ("Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, 2,399 windows (answers 'another draw of EPISODES' only); seed = training seed, "
       "the two seeds are two independent training runs shown side by side.")
RUN_LAB = {"causal": "CAUSAL", "strict": "STRICT"}
RUN_DESC = {"causal": "logged frames + causal bucket", "strict": "first frame frozen + causal bucket"}
NBOOT_TXT = "B = 2,000"

LAST_LINES: list = []          # [(gid, x, y)] of the figure just rendered (control hook)


def _capture(fig):
    LAST_LINES[:] = [(ln.get_gid(), np.array(ln.get_xdata(), float), np.array(ln.get_ydata(), float)) for ax in fig.axes for ln in ax.lines if ln.get_gid()]


def render4(fig):
    _capture(fig)
    return render(fig)


def stat_col(word):
    w = word.lower()
    return C_OK if w.startswith("better") else C_BAD if w.startswith("worse") else C_NS


def label_of(name, key, nth=0):
    return V.row_cells(name, key, nth)[0].replace("**", "").strip()


# ================================================================================================ 7a. what T1-kin is
def t1k_intro():
    fig = new_fig(DARK)
    tx(fig, 90, 56, "7   T1-kin: the model on its own motion", 40, "white", "bold")
    tx(fig, 90, 120, "Against chapters 1-6 the model no longer reads the human's logged motion: it reads its OWN executed motion", 20, "#e8ecf2")
    tx(fig, 90, 160, "tier T1-kin = self-action OPEN loop  |  NOT closed loop: the cameras are the human's recorded frames, nothing re-renders the scene", 14, "#ffd9b3")
    boxes = [("HOW IT RUNS", "#0072B2",
              "Every 2 s (m = 10 steps of 0.2 s) the model REPLANS.\n\n"
              "At each replan its ego state (speed, history, odometry, the route countdown) is rebuilt from what it has itself EXECUTED so far, not from the logged human motion.\n\n"
              "The floor P10 (the programme's kinematic prior, which ignores the scene) is run the same way, in its own loop, on the same windows.\n\n"
              "Score: ADE6 of each executed 6 s track against the human's logged path."),
             ("THE TWO CONFIGURATIONS", "#009E73",
              "CAUSAL:  the camera frames are the human's logged frames.  The speed bucket is built from the past 4 s of measured speed only (logged up to the window start, the model's own executed speed after it): no future row, no label.\n\n"
              "STRICT:  as CAUSAL, and the cameras are frozen: the FIRST frame of the window is held for all 6 s.\n\n"
              "Arm HACC (the HACC post-map on the M2b checkpoints), two training seeds, s0 and s1."),
             ("WHAT YOU WILL SEE", "#D55E00",
              "Seven windows, chosen by a rule fixed before any T1 number per window was read (next card).\n\n"
              "For each window: the camera frame (logged for CAUSAL, frozen for STRICT), and a bird's-eye view of the human's path (GT), the model's executed track and the floor's executed track, growing in time, replan ticks marked.\n\n"
              "Then the distribution over all 2,399 windows, three result cards (CAUSAL, STRICT, the open lateral gap), and the gap that is still open.")]
    for i, (ttl, col, body) in enumerate(boxes):
        x = 70 + i * 600
        ax = axpx(fig, x, 230, 560, 620, W, H)
        ax.set_facecolor("#18202a"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_edgecolor(col); s_.set_linewidth(2.6)
        tx(fig, x + 20, 246, ttl, 19, col, "bold")
        tx(fig, x + 20, 292, wrap(body, 40), 15.4, "#e8ecf2", ls=1.36)
    tx(fig, 70, 904, wrap(EST, 215), 12, "#aeb7c4")
    tx(fig, 70, 960, wrap("Baseline: P10 is the programme's own kinematic floor, not the state of the art; no external benchmark is shown.  Evidence class of every number in this chapter: MEASURED (ours), from the float64 dumps and the four RESULT files named on each card.", 215), 12, "#aeb7c4")
    return render4(fig)


# ================================================================================================ 7b. the selection rule
def t1k_rule():
    fig = new_fig()
    header(fig, "7  T1-kin: which windows are shown, and why", tier=TIER_K, sub="rule fixed BEFORE any T1 number per window was read  |  SELECTION_V4.md, select_scenes.py (it loads only g, ws, clip_index)")
    sj = V._SEL
    rule = ("THE RULE\n\n"
            "1.  The four windows of chapters 3-6 (JUNCTION #1131, CURVE #654, NO-TURN #763, FAILURE #2158), if they are in the T1 grid.  All four are.\n\n"
            f"2.  Three more windows drawn with numpy.random.default_rng({sj['seed']}) from the T1-grid windows whose GT mean speed is at least 5 m/s "
            f"({sj['n_ge5']:,} of {sj['n_windows']:,} windows; {sj['n_candidates']:,} candidates after excluding the four of rule 1).  "
            "GT mean speed = mean over the 30 knots of |g[k] - g[k-1]| / 0.2 s.\n\n"
            "No window is picked, swapped or dropped for how the model looks on it.  Where the floor is better, the caption says so and the window stays.")
    tx(fig, 24, 100, wrap(rule, 71), 16.5, INK, ls=1.38)
    tx(fig, 940, 100, "The seven windows, in the order shown", 17, INK, "bold")
    hdr = f"{'#':<3}{'tag':<16}{'eval window':>12}{'clip':>10}{'t (s)':>8}{'GT mean speed':>16}"
    tx(fig, 940, 140, hdr, 13, MUTED, mono=True)
    for i, s in enumerate(V.SCN):
        row = f"{i + 1:<3}{s['tag']:<16}{'#' + str(s['grid_row']):>12}{s['clip'][:8]:>10}{0.2 * s['ws']:>8.1f}{s['gt_mean_speed']:>12.1f} m/s"
        tx(fig, 940, 172 + i * 34, row, 13, INK, "bold", mono=True)
    tx(fig, 940, 430, wrap("draw positions in the candidate list: " + ", ".join(str(p) for p in sj["draw_positions_in_candidates"]) +
                           ".  Eval window = row of the T1 grid, which equals the window id of chapters 3-6.", 78), 12, MUTED)
    tx(fig, 24, 560, "Why a rule, and why these windows are NOT a sample of the typical window", 16, INK, "bold")
    tx(fig, 24, 596, wrap("The four windows of chapters 3-6 were chosen there as the median of their stratum and the worst of all windows.  The three draws are random among windows that move (at least 5 m/s), "
                          "so that the executed tracks have a length to compare.  Slow and standing windows are in the evaluation (and in the distribution card), just not in the draw.  "
                          "A window is an example; the distribution card and the result cards are the evidence.", 150), 14, INK, ls=1.35)
    caption(fig, "Each window is shown twice, side by side: the CAUSAL run (logged frames) and the STRICT run (first frame frozen), both at training seed 0 in the picture.  "
                 "The per-window ADE6 of both seeds and both runs is in the table and the caption at the end of each window.", size=16, y0=934, h=106, n=150)
    footer(fig, "Tier T1-kin: self-action OPEN loop, NOT closed loop.  " + EST)
    return render4(fig)


# ================================================================================================ 7c. one window
def _scene_ext(i):
    arrs = [V.trk("causal", 0, i, "gt"), V.trk("causal", 0, i, "model"), V.trk("strict", 0, i, "model"), V.trk("causal", 0, i, "floor")]
    return fit_extent(arrs)


def _bev_run(fig, i, run, f, box, ext):
    s = V.SCN[i]
    ax = bev_axes(fig, box, ext, title=f"{RUN_LAB[run]}: {RUN_DESC[run]} (seed 0)")
    gt, md, fl = V.trk(run, 0, i, "gt"), V.trk(run, 0, i, "model"), V.trk(run, 0, i, "floor")
    plan_t1(ax, gt, 30, "GT", C_GT, 3.4, "-", "GT: the human's path (the label)", 3, ms=5, full=True)
    plan_t1(ax, fl, f, "FLOOR", C_FLOOR, 2.4, "--", "P10 floor, executed in its own loop", 5, ms=4)
    plan_t1(ax, md, f, "MODEL", C_MODEL, 3.6, "-", "HACC, executed in its own loop", 8, ms=4.6, annotate=True)
    ax.plot([], [], "o", ms=11, mfc="none", mec=INK, mew=2.0, label="replan tick (2 s, 4 s)")
    legend(ax, "lower left", 8.6)
    return ax


def plan_t1(ax, xy, n, role, color, lw, ls, label, z, ms=4.5, full=False, annotate=False):
    """draw the first n knots of xy [30, 2] (forward, left) at 0.2 s knots; dots every 1 s; a ring at each 2 s replan tick that has been reached; a big dot at the current end.
    The line carries gid=role so CONTROLS_V4 can read the plotted arrays back from the figure."""
    xy = np.asarray(xy, float)
    X, Y = -xy[:n, 1], xy[:n, 0]
    ax.plot(X, Y, ls, color=color, lw=lw, label=label, zorder=z, solid_capstyle="round", gid=role)
    k = [i for i in range(4, n, 5)]
    if k and role != "FLOOR":
        ax.plot(X[k], Y[k], "o", color=color, ms=ms, zorder=z)
    if n >= 1 and not full:
        ax.plot([X[-1]], [Y[-1]], "o", color=color, ms=ms + 3.5, mec="white", mew=1.0, zorder=z + 1)
    for t_i in V.TICK_IDX:
        if n > t_i and not full:
            ax.plot([X[t_i]], [Y[t_i]], "o", ms=13, mfc="none", mec=color, mew=2.2, zorder=z + 2)
            if annotate:
                ax.annotate(f"{0.2 * (t_i + 1):.0f} s", xy=(X[t_i], Y[t_i]), xytext=(11, -3), textcoords="offset points", fontsize=9, color=color, fontweight="bold")


def t1k_frame(i, f):
    s = V.SCN[i]
    tag = s["tag"]
    lab = f"{tag}  (window #{s['grid_row']} of chapters 3-6)" if s["source"] == "v3" else f"{tag}  (random draw, rng {V._SEL['seed']})"
    fig = new_fig()
    header(fig, f"7  T1-kin: the model on its own motion  |  window {i + 1} of 7: {lab}", tier=TIER_K,
           sub=f"arm HACC (M2b seed-0 checkpoint in the picture) vs the P10 floor, each in its own loop  |  clip {s['clip'][:8]}  |  eval window #{s['grid_row']}  |  t = {0.2 * s['ws']:.1f} s  |  GT mean speed {s['gt_mean_speed']:.1f} m/s  |  replan every 2 s")
    tick = f % 10 == 0 and f < 30
    camera(fig, (24, 112, 640, 256), V.frame_for(i, "causal", f), title="CAUSAL run: the LOGGED camera frame at the current time",
           sub=f"frame at +{0.2 * f:.1f} s  |  the model reads the logged frame at each replan tick (+0.0, +2.0, +4.0 s)" + ("   <- a replan tick: the model reads THIS frame" if tick else ""),
           ec=C_MODEL if tick else INK)
    camera(fig, (24, 452, 640, 256), V.frame_for(i, "strict", f), title="STRICT run: the FIRST frame, held for all 6 s",
           sub="frame at +0.0 s, held: the model never sees a later one", ec=INK)
    ext = _scene_ext(i)
    _bev_run(fig, i, "causal", f, (714, 100, 566, 826), ext)
    _bev_run(fig, i, "strict", f, (1322, 100, 568, 826), ext)
    nt = int(f >= 10) + int(f >= 20)
    tx(fig, 24, 726, f"time: +{0.2 * f:.1f} s of 6.0 s   |   replans done: {nt} of 2", 12.5, INK, "bold")
    tx(fig, 24, 748, wrap("ring = a 2 s replan tick: the model rebuilds its state from its own executed motion there.  The BEV is forward-up, ego frame at +0.0 s; dots every 1 s.", 82), 10.5, MUTED)
    if f == 30:
        tx(fig, 24, 792, "ADE6 of this window (m)  -  per-window value, T1-kin, no interval", 11.5, INK, "bold")
        tx(fig, 24, 814, f"{'':<8}{'':<17}{'seed 0':>8}{'seed 1':>8}", 11.5, MUTED, mono=True)
        for r, run in enumerate(("causal", "strict")):
            for q, who in enumerate(("model", "floor")):
                nm = "HACC (model)" if who == "model" else "P10 floor"
                vals = [V.ade_reg(run, sd, i, who) for sd in V.SEEDS4]
                tx(fig, 24, 836 + (2 * r + q) * 21, f"{RUN_LAB[run] if q == 0 else '':<8}{nm:<17}{vals[0]:>8}{vals[1]:>8}", 11.8, C_MODEL if who == "model" else C_FLOOR, "bold", mono=True)
    else:
        tx(fig, 24, 792, "the window's ADE6 appears when the 6 s are complete", 11.5, MUTED)
    if f < 30:
        cap = (f"Window {tag}, running: the human's path (black), the model's executed track (orange) and the floor's executed track (grey, dashed), both in their own loop, at +{0.2 * f:.1f} s.  "
               "The replan ticks at +2 s and +4 s are the ringed points.  CAUSAL reads the logged frames; STRICT reads only the first frame.")
    else:
        parts, nb = [], 0
        for run in ("causal", "strict"):
            seg = []
            for sd in V.SEEDS4:
                a, b, d = V.ade_reg(run, sd, i, "model"), V.ade_reg(run, sd, i, "floor"), V.dlt_reg(run, sd, i)
                nb += int(V.ade6(run, sd, i, "model") < V.ade6(run, sd, i, "floor"))
                seg.append(f"s{sd}: HACC {a} vs P10 {b} ({d})")
            parts.append(f"{RUN_LAB[run]} " + "; ".join(seg))
        verdict = (f"HACC is better than P10 in {nb} of 4 (run, seed) cells" + ("." if nb == 4 else f"; P10 is better in the other {4 - nb} (kept as it is)."))
        cap = (f"Window {tag}: ADE6 (m), T1-kin, per-window value, no interval.  " + "  |  ".join(parts) + ".  " + verdict + "  One window is an example.")
    caption(fig, cap, size=15.4, y0=934, h=106, n=156)
    footer(fig, "Tracks: float64 dumps t1kc_run/ (CAUSAL) and t1ks_run/ (STRICT), dump_HACC_s{0,1}_m10_*/ep*.npz, keys g (GT), cl (HACC executed), fl (P10 executed), plotted unchanged (CONTROLS_V4).  "
                "Frames: v2ep cache step = window start + 0.2 s per step.  Selection rule: SELECTION_V4.md.  Tier T1-kin, NOT closed loop.")
    return render4(fig)


# ================================================================================================ 7d. the distribution
def _bars(ax, d, col_lo, col_hi, lim=6.0, nb=48):
    edges = np.linspace(-lim, lim, nb + 1)
    c = 0.5 * (edges[:-1] + edges[1:])
    h, _ = np.histogram(np.clip(d, -lim + 1e-9, lim - 1e-9), bins=edges)
    ax.bar(c[c < 0], h[c < 0], width=edges[1] - edges[0], color=col_lo, ec="white", lw=0.4)
    ax.bar(c[c >= 0], h[c >= 0], width=edges[1] - edges[0], color=col_hi, ec="white", lw=0.4)
    return h


def t1k_hist():
    fig = new_fig()
    header(fig, "7  T1-kin: per-window HACC - P10 ADE6 over all 2,399 windows", tier=TIER_K, sub="each bar = windows; left of zero the model's executed track is closer to the human's than the floor's  |  descriptive: windows of one episode are not independent")
    pos = {("causal", 0): (60, 130), ("causal", 1): (990, 130), ("strict", 0): (60, 540), ("strict", 1): (990, 540)}
    nice = {}
    for (run, sd), (x, y) in pos.items():
        d = V.per_window_delta(run, sd)
        ax = axpx(fig, x + 30, y + 40, 780, 290, W, H)
        ax.set_facecolor(PANEL)
        h = _bars(ax, d, C_MODEL, "#9aa0aa")
        ax.axvline(0, color=INK, lw=1.4)
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, h.max() * 1.22)
        ax.set_xlabel("HACC - P10 ADE6 of the window (m), lower is better; |value| > 6 m is drawn in the end bins", fontsize=9)
        ax.set_ylabel("windows", fontsize=9)
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="y", color="#e8ebef")
        nm, nf, nt = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
        pm, pf = 100 * nm / len(d), 100 * nf / len(d)
        mean = float(d.mean())
        sm = V.mm(mean, "{:.4f}")
        L.REG.append({"kind": "hist", "scene": L.CUR["scene"], "run": run, "seed": sd, "n_model": nm, "n_floor": nf, "n_tie": nt, "mean": sm, "n": len(d)})
        ax.set_title(f"{RUN_LAB[run]} run, seed {sd}", fontsize=12, fontweight="bold", loc="left")
        ax.text(0.985, 0.97, f"HACC better: {nm:,} windows ({pm:.1f} %)\nP10 better: {nf:,} windows ({pf:.1f} %)" + (f"\ntied: {nt:,}" if nt else "") + f"\nmean over windows: {sm} m",
                transform=ax.transAxes, ha="right", va="top", fontsize=10.5, family="DejaVu Sans Mono", color=INK, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#c9ced6"))
        for i, s in enumerate(V.SCN):
            di = float(d[s["grid_row"]])
            ax.plot([np.clip(di, -5.9, 5.9)], [0], "^", ms=11, mfc=C_GT, mec="white", mew=0.8, zorder=9, clip_on=False)
            ax.text(float(np.clip(di, -5.9, 5.9)), h.max() * (0.045 + 0.06 * (i % 3)), f"{i + 1}", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=C_GT)
        nice[(run, sd)] = (nm, nf, mean)
        L.REG.append({"kind": "ref", "scene": L.CUR["scene"], "run": run, "seed": sd, "res_file": "RESULT_V8F_T1KIN_CAUSALBUCKET" if run == "causal" else "RESULT_V8F_T1KIN_STRICT",
                      "row": "ADE6, PRIMARY", "col": sd + 1, "mean": sm})
    cap = ("Per-window HACC - P10 ADE6 (m), T1-kin, both runs, both seeds.  The mean of the bars is the point estimate of the paired read (CAUSAL " +
           V.RC("RESULT_V8F_T1KIN_CAUSALBUCKET", "ADE6, PRIMARY", 1)["val"] + " / " + V.RC("RESULT_V8F_T1KIN_CAUSALBUCKET", "ADE6, PRIMARY", 2)["val"] + " m, STRICT " +
           V.RC("RESULT_V8F_T1KIN_STRICT", "ADE6, PRIMARY", 1)["val"] + " / " + V.RC("RESULT_V8F_T1KIN_STRICT", "ADE6, PRIMARY", 2)["val"] +
           " m).  HACC wins most windows and loses some; triangles 1-7 = the seven windows of this chapter (their position was not used to choose them).  "
           "Descriptive only: the decision-grade intervals (paired episode-cluster bootstrap) are on the next cards.")
    caption(fig, cap, size=15.2, y0=934, h=106, n=158)
    footer(fig, "Source: float64 dumps t1kc_run/ and t1ks_run/ (g, cl, fl; ADE6 = mean over the 30 knots of the L2 distance to the GT), all 2,399 windows, per seed and run; the floor is identical across seeds and runs (checked, max |d| = 0.0).  Tier T1-kin, NOT closed loop.")
    return render4(fig)


# ================================================================================================ 7e. result cards
def _forest(fig, box, rows, title):
    """rows: [(label, val, lo, hi)] strings with the RESULT's characters; x axis in m."""
    ax = axpx(fig, *box, W, H)
    ax.set_facecolor(PANEL)
    f = lambda s: float(s.replace("−", "-"))
    for k, (lab, v, lo, hi) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([f(lo), f(hi)], [y, y], "-", color=C_MODEL, lw=4.0, solid_capstyle="round")
        ax.plot([f(v)], [y], "o", color=C_MODEL, ms=11, mec="white", mew=1.2, zorder=5)
        ax.text(f(hi) + 0.03, y + 0.1, f"{v} [{lo}, {hi}]", fontsize=10.5, va="center", color=INK, family="DejaVu Sans Mono")
        ax.text(-1.24, y, lab, fontsize=11, va="center", ha="right", color=INK, fontweight="bold")
    ax.axvline(0, color=INK, lw=1.4)
    ax.set_xlim(-1.25, 0.55)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel("HACC - P10 ADE6 (m), lower is better   (left of zero: HACC's executed track is closer to the human's)", fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", color="#e8ebef")
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    return ax


def _word(plain):
    m = re.search(r"(better|worse|n\.s\.)\s*$", plain, re.I)
    return m.group(1) if m else ""


def _table(fig, x0, y0, name, rows, col_x=(430, 790), fs=11.4, dy=31):
    tx(fig, x0, y0, "row (family)", 11.5, MUTED, "bold")
    tx(fig, x0 + col_x[0], y0, "seed 0", 11.5, MUTED, "bold")
    tx(fig, x0 + col_x[1], y0, "seed 1", 11.5, MUTED, "bold")
    for r, (key, nth, interval) in enumerate(rows):
        y = y0 + 30 + r * dy
        lab = label_of(name, key, nth)
        tx(fig, x0, y, lab, fs, INK, "bold")
        for c in (1, 2):
            o = V.RC(name, key, c, nth, interval=False)
            word = _word(o["plain"])
            shown = o["plain"]
            if not word and "lo" in o:
                word = V.status(o["lo"], o["hi"])
                shown = o["plain"] + "  " + word
            tx(fig, x0 + col_x[c - 1], y, shown, fs - 0.4, stat_col(word), "bold", mono=True)
    return y0 + 30 + len(rows) * dy


def t1k_card(kind):
    name = "RESULT_V8F_T1KIN_CAUSALBUCKET" if kind == "causal" else "RESULT_V8F_T1KIN_STRICT"
    big = "CAUSAL" if kind == "causal" else "STRICT"
    desc = ("HACC with a causal speed bucket (the past 4 s of measured speed only), cameras = the human's logged frames" if kind == "causal"
            else "cameras frozen at the first frame + the causal speed bucket: both open channels closed")
    fig = new_fig()
    header(fig, f"7  T1-kin: result card, {big} run  |  HACC - P10 over all 2,399 windows", tier=TIER_K, sub=desc + f"  |  source {name}.md")
    prim = [V.RC(name, "ADE6, PRIMARY", c) for c in (1, 2)]
    _forest(fig, (1180, 150, 690, 190), [("seed 0", prim[0]["val"], prim[0]["lo"], prim[0]["hi"]), ("seed 1", prim[1]["val"], prim[1]["lo"], prim[1]["hi"])],
            f"ADE6, PRIMARY row: HACC - P10, T1-kin, {NBOOT_TXT}")
    if kind == "causal":
        rows = [("ADE6, PRIMARY", 0, True), ("ADE2 (m)", 0, True), ("speed MAE", 0, True), ("along-track MAE", 0, True), ("accel MAE", 0, True), ("heading MAE", 0, True), ("yaw-rate MAE", 0, True),
                ("curvature MAE", 0, True), ("cross-track MAE", 0, True)]
        yb = _table(fig, 24, 100, name, rows, col_x=(390, 760), fs=12.4, dy=40)
    else:
        rows = [("ADE6, PRIMARY", 0, True), ("ADE2 (m)", 0, False), ("speed / along-track", 0, False), ("accel MAE", 0, False), ("heading MAE", 0, False), ("yaw-rate MAE", 0, False),
                ("curvature MAE", 0, False), ("cross-track MAE", 0, False)]
        yb = _table(fig, 24, 100, name, rows, col_x=(390, 760), fs=12.4, dy=40)
    nc = V.RT(name, "NOT COMPUTED")
    tx(fig, 24, yb + 8, wrap("TACTICAL / STRATEGIC and distance keeping: " + nc + " in the T1-kin dumps (no per-tick T code, S checkpoints or lead track): a work item, not a pass.", 128), 10.8, C_BAD, "bold")
    tx(fig, 1180, 404, "Mean ADE6 of the arms", 12.5, INK, "bold")
    if kind == "causal":
        snip = V.RT(name, "Mean ADE6 at m = 10: 2.675 / 2.710 m with the causal bucket, 2.614 / 2.649 m with the logged one, and P10 3.443 m.")
        beside = V.RT(name, "HACC(causal) − HACC(logged) ADE6 is +0.061 [+0.003, +0.120] / +0.061 [+0.019, +0.103] m")
        tx(fig, 1180, 430, wrap(snip, 66), 11.5, INK, ls=1.3)
        tx(fig, 1180, 494, wrap("What the causal bucket costs against the logged (oracle) one: " + beside + " (separated, about F/2).", 66), 11.5, INK, ls=1.3)
        tx(fig, 1180, 574, wrap(V.RT(name, "v8F's T1 result can be stated on an input built from the past alone.") +
                                "  The margin does not need the future-derived bucket: with the bucket ABSENT it vanished (the model's response to a missing input), with a causal one it is kept.", 66), 11.5, INK, ls=1.3)
    else:
        snip = V.RT(name, "Mean ADE6: strict 2.716 / 2.802 m, P10 3.443 m.")
        beside = V.RT(name, "Closing the camera channel on top of the causal bucket costs +0.040 / +0.092 m of ADE6") + " " + V.RT(name, "(strict − causal, separated)") + "."
        tx(fig, 1180, 430, wrap(snip, 66), 11.5, INK, ls=1.3)
        tx(fig, 1180, 476, wrap(beside, 66), 11.5, INK, ls=1.3)
        tx(fig, 1180, 560, wrap("From the first frame alone plus its own executed motion, a past-only speed bucket and the dead-reckoned nav, HACC beats the kinematic floor, run the same way: the strictest T1 configuration this rig can produce.", 66), 11.5, INK, ls=1.3)
    if kind == "strict":
        tx(fig, 24, yb + 46, "ADE2 and the rows below ADE6 carry the point estimate and the file's sign word only: the RESULT file lists no interval for them.", 10.8, MUTED)
    tx(fig, 24, yb + 84, "What it is not", 14, C_BAD, "bold")
    nots = ["NOT closed loop: no re-perception (T2 is not provisioned); the cameras are recorded." + ("" if kind == "causal" else " The frames are frozen, not re-rendered."),
            "NOT dominance: the lateral rows curvature and yaw rate are worse than the floor's (see the table and the next card); a worse row is shown, not hidden.",
            "The baseline is the programme's own floor P10, not a published state of the art.",
            "Not a speed LIMIT: the causal bucket is a past-speed proxy."]
    tx(fig, 24, yb + 116, wrap("\n".join("- " + n for n in nots), 100), 13, INK, ls=1.38)
    caption(fig, ("Primary row, " + big + f" run, HACC - P10 ADE6 (m), T1-kin, paired episode-cluster bootstrap ({NBOOT_TXT}, 141 episodes): seed 0 " + prim[0]["val"] + " [" + prim[0]["lo"] + ", " + prim[0]["hi"] +
                  "], seed 1 " + prim[1]["val"] + " [" + prim[1]["lo"] + ", " + prim[1]["hi"] + "].  " + ("Both intervals exclude zero." if all(V.status(p["lo"], p["hi"]) == "better" for p in prim) else "NOT both intervals exclude zero.") + "  An interval from this estimator answers 'another draw of episodes'; the two training seeds are the replicate, "
                  "and both agree.  Every cell of the table is copied from the RESULT file."), size=15.2, y0=934, h=106, n=158)
    footer(fig, f"Source: {name}.md (landed, tip abc12d3f), table 'The read'.  " + EST + "  Tier T1-kin: self-action OPEN loop, NOT closed loop.  Lower is better for every row; word = sign of the interval (better / WORSE / n.s.).")
    return render4(fig)


def t1k_lat():
    LC, HL = "RESULT_V8F_T1_LATCOND", "RESULT_V8F_HACC_LAT"
    fig = new_fig()
    header(fig, "7  T1-kin: the open gap, lateral curvature at speed", tier=TIER_K, sub="HACC with the causal speed bucket - P10, m = 10, 2,399 windows  |  sources RESULT_V8F_T1_LATCOND.md, RESULT_V8F_HACC_LAT.md")
    rows = [("L2:", 0, True), ("L1:", 0, True), ("L3:", 0, True), ("heading MAE", 0, False), ("cross-track MAE", 0, False), ("canonical curvature", 0, False)]
    yb = _table(fig, 24, 100, LC, rows, col_x=(470, 800), fs=11.2)
    tx(fig, 24, yb + 6, "L2, L3: pairs whose GT speed is at least 5.2 m/s.  L1: lateral acceleration, |v w - v_G w_G|.  'canonical curvature' is the all-speed row, shown beside.", 10.4, MUTED)
    o2 = [V.RC(LC, "L2:", c) for c in (1, 2)]
    o1 = [V.RC(LC, "L1:", c) for c in (1, 2)]
    tx(fig, 24, yb + 44, "The open gap", 15, C_BAD, "bold")
    gap = (f"Curvature at GT speed of at least 5.2 m/s is WORSE than the floor's at both seeds: {o2[0]['val']} / {o2[1]['val']} 1/m.  "
           f"Lateral acceleration is NOT worse: {o1[0]['val']} / {o1[1]['val']} m/s\u00b2 (not separated).  Heading and cross-track are better.")
    tx(fig, 24, yb + 74, wrap(gap, 96), 13.2, INK, ls=1.35)
    tx(fig, 24, yb + 156, wrap(V.RT(LC, "The lateral deficit is real at speed, and it is in curvature, not in lateral acceleration or position.") + "  " +
                               V.RT(LC, "Masking the speeds where millimetres dominate shrinks the curvature excess by 30–37 %") + " (+0.030 to +0.021, +0.013 to +0.008 1/m), but a part remains at speed.", 96), 12.4, INK, ls=1.35)
    tx(fig, 24, yb + 252, wrap("Next, unblocked: " + V.RT(LC, "A root-cause diagnosis of the at-speed excess (zero-GPU) is the next unblocked step.") +
                               "  On train ticks, HACC's plan curvature from the same state is no worse than P10's, so " + V.RT(LC, "the at-speed excess arises in the LOOP (state, seams), not in a single plan") +
                               ".  The box-2 lateral training lever is blocked on box-2 compute.", 96), 12.0, INK, ls=1.35)
    # noise-floor chart: curvature MAE of the GT scored against itself with i.i.d. position noise, by GT speed band
    ax = axpx(fig, 1130, 150, 740, 420, W, H)
    ax.set_facecolor(PANEL)
    bands = ["below 2 m/s", "2-5.2 m/s", "above 11.8 m/s"]
    cols = {"2 mm": "#cfd8e3", "5 mm": "#7fa6cf", "10 mm": "#1f5fa6"}
    wd = 0.26
    for q, key in enumerate(("2 mm", "5 mm", "10 mm")):
        cells = V.row_cells(HL, key)
        vals = []
        for c in (2, 3, 4):
            V.RC(HL, key, c, interval=False)
            vals.append(float(cells[c]))
        xs = np.arange(3) + (q - 1) * wd
        ax.bar(xs, vals, width=wd, color=cols[key], ec="white", label=f"{key} position noise per 0.2 s knot")
        for x_, v_, c in zip(xs, vals, (2, 3, 4)):
            ax.text(x_, v_ * 1.12, cells[c], ha="center", va="bottom", fontsize=8.6, rotation=0, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(3e-4, 2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(bands, fontsize=10)
    ax.set_ylabel("curvature MAE of the GT against itself (1/m)", fontsize=9.5)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#e8ebef")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title("Noise floor of the canonical curvature row (instrument probe, no model arm)", fontsize=10.8, fontweight="bold", loc="left")
    tx(fig, 1130, 612, wrap("GT of the 141 eval episodes perturbed by i.i.d. Gaussian noise per 0.2 s knot and scored against itself with the reader's own row function (" + HL + ").  " +
                            V.RT(LC, "The canonical curvature row's ratio is about 230×") + " between the slow and the fast band; the lateral-acceleration row's ratio is 1.00.", 78), 11.2, INK, ls=1.32)
    cc = [V.RC(LC, "canonical curvature", c, interval=False)["val"] for c in (1, 2)]
    tx(fig, 1130, 722, wrap(f"So the canonical row ({cc[0]} / {cc[1]} 1/m at all speeds) overstates the gap; the conditioned rows (GT speed of at least 5.2 m/s) are the fair read, and they still show a real, smaller curvature excess.  "
                            "The lateral-acceleration row is well conditioned: " + V.RT(LC, "ratio 1.00") + ".", 78), 11.2, INK, ls=1.32)
    caption(fig, ("Lateral family, T1-kin, HACC with the causal bucket - P10, paired episode-cluster bootstrap (" + NBOOT_TXT + ", 141 episodes), seeds 0 / 1.  CURVATURE at GT speed >= 5.2 m/s: " + o2[0]["val"] + " / " + o2[1]["val"] +
                  " 1/m, both separated-worse: the gap is open.  LATERAL ACCELERATION: " + o1[0]["val"] + " / " + o1[1]["val"] + " m/s\u00b2, not separated."), size=15.0, y0=934, h=106, n=160)
    footer(fig, f"Sources: {LC}.md (rows L1-L3, N-1, 'What this settles'), {HL}.md (noise-floor table).  " + EST + "  Tier T1-kin: self-action OPEN loop, NOT closed loop.  Lower is better; word = sign of the interval.")
    return render4(fig)


SCENES.update({"t1k_intro": t1k_intro, "t1k_rule": t1k_rule, "t1k_frame": t1k_frame, "t1k_hist": t1k_hist, "t1k_card": t1k_card, "t1k_lat": t1k_lat})
