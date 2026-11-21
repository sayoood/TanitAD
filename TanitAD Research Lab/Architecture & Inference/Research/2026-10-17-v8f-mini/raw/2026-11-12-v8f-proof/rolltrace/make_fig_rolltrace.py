# make_fig_rolltrace.py -- dev box: figure of per-window traces, the distance to the lead over rollout time for three forms of the lead rule (RR, RLc, VD5c).
# Input: rolltrace/pod/ROLLTRACE.json (written by export_traces.py on the pod and pulled by pull_rolltrace.sh). Style as paper/v11/figs/make_fig_gaptrace.py.
# Colours: VD5c blue (slot 1), RLc orange (slot 2), RR aqua (slot 3 of the palette file); validated with validate_palette.js --mode light.
# Usage: python make_fig_rolltrace.py [--in <ROLLTRACE.json>] [--out-dir <folder>]   (default out-dir: the paper/v11/figs folder; the lead runs it on real data)
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STYLE = {"RR": (AQUA, 1.9, "RR: reader-fed form of the candidate spec"), "RLc": (ORANGE, 1.6, "RLc: label gap and slope"), "VD5c": (BLUE, 2.0, "VD5c: reader presence, camera crop gap and slope (no label)")}
ORDER = ("RR", "RLc", "VD5c")
TITLE = "What the rule does in a window: distance to the lead under three forms of the lead rule"
FOOT = ("T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s). Windows are not hand-picked: the first 6 in sha256 order among the windows where the reader-fed form has contact "
        "and the form with no label has none.\nThe lead moves as logged; no re-perception; not driving performance. A ring marks the first step at or below 0 m (contact).")
DEF_IN = os.path.join(HERE, "pod", "ROLLTRACE.json")
DEF_OUT = "F:/TanitAD-scratch/v8proof/paper/v11/figs"


def arr(v):
    return np.array([np.nan if x is None else x for x in v], float)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=DEF_IN)
    ap.add_argument("--out-dir", default=DEF_OUT)
    a = ap.parse_args(argv)
    with open(a.inp, encoding="utf-8") as f:
        rec = json.load(f)
    wins = rec["windows"][:6]
    forms = [f for f in ORDER if f in rec["forms"]]
    allv = [arr(w["forms"][f]["gap_m"]) for w in wins for f in forms]
    lo, hi = float(np.nanmin([np.nanmin(v) for v in allv] + [0.0])), float(np.nanmax([np.nanmax(v) for v in allv] + [1.0]))
    pad = 0.06 * (hi - lo)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.06, "hspace": 0.24})
    fig.patch.set_facecolor(SURFACE)
    thr = 0.075 * (hi - lo + 2 * pad)
    for ax, w in zip(axes.ravel(), wins):
        t = arr(w["t_s"])
        ax.set_facecolor(SURFACE)
        ax.axhline(0.0, color=INK2, linewidth=0.7, zorder=1)
        ends = []
        for f in forms:
            col, lw, _ = STYLE[f]
            g = arr(w["forms"][f]["gap_m"])
            ax.plot(t, g, color=col, linewidth=lw, zorder=3)
            ok = np.isfinite(g)
            hit = np.nonzero(ok & (g <= 0.0))[0]
            if len(hit):
                ax.plot([t[hit[0]]], [g[hit[0]]], marker="o", markersize=6.5, markerfacecolor=SURFACE, markeredgecolor=col, markeredgewidth=1.6, linestyle="none", zorder=5)
            if ok.any():
                j = int(np.nonzero(ok)[0][-1])
                ends.append((float(g[j]), float(t[j]), f))
        placed = []
        for y, x, f in sorted(ends):                               # direct labels only where they do not collide
            if all(abs(y - py) >= thr for py in placed):
                ax.text(x + 0.12, y, f, color=INK2, fontsize=8, va="center", ha="left")
                placed.append(y)
        ax.set_title("model seed %d, clip %s, window %d" % (w["seed"], w["clip8"], w["ws"]), loc="left", color=INK2, fontsize=8.5, pad=4)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, length=0, labelsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0.0, 7.0)
        ax.set_xticks(range(0, 7))
    for ax in axes[1]:
        ax.set_xlabel("rollout time (s)", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("distance to the lead (m)", color=INK2)
    handles = [plt.Line2D([0], [0], color=STYLE[f][0], linewidth=STYLE[f][1]) for f in forms]
    fig.legend(handles, [STYLE[f][2] for f in forms], loc="lower left", bbox_to_anchor=(0.05, 0.085), ncol=3, frameon=False, fontsize=9, labelcolor=INK2, handlelength=2.4, columnspacing=2.0)
    fig.suptitle(TITLE, x=0.05, ha="left", color=INK, fontsize=12.5, fontweight="bold", y=0.985)
    fig.text(0.05, 0.008, FOOT, ha="left", va="bottom", color=INK2, fontsize=7.4, linespacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.2)
    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "fig_rolltrace." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-ROLLTRACE DONE windows %d" % len(wins))


if __name__ == "__main__":
    main()
