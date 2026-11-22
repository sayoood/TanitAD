# make_fig_gapmargin.py -- figure of cards ADD117 (GAPMARGIN10K: a margin on the camera gap of the lead rule, tier T1-kin with the heading bound, kinematic rollout on the
# model's own actions) and ADD118 (PRESHEAD-T0: a presence head on existing tokens, tier T0, a diagnostic of the head and not driving performance).
# No number is typed here: every value is read from the pulled result files of the two cards. Top row: the dose of the margin (blue, light to dark = 0, 10, 20 per cent)
# against the label form (green, open, a reference with label values). Bottom row: the three presence forms against the bar set before the fit.
# Output: fig_gapmargin.png and fig_gapmargin.pdf beside this file (or in --out-dir).
import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES117 = os.path.join(V, "gapmargin", "pod_add117", "read", "GAPMARGIN10K_RESULT.json")
RES118 = os.path.join(V, "preshead", "pod_add118", "read", "PRESHEAD_T0_RESULT.json")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GREEN, B0, B1, B2 = "#1baf7a", "#86b4ea", "#2a78d6", "#143f7d"
SETS = (("CONV", "convention\n(no pre-roll, pinned readout)"), ("K0", "readout K, no pre-roll"))
# arm, legend text, colour, filled, vertical offset
ARMS = (("RLc", "label form (reference, label values)", GREEN, False, 0.27), ("GEc", "candidate, camera gap as read", B0, True, 0.09),
        ("A90", "candidate, gap scaled by 0.90", B1, True, -0.09), ("A80", "candidate, gap scaled by 0.80", B2, True, -0.27))
MEAS = (("U", "unsafe gap"), ("C", "contact"), ("F", "falling behind by more than 10 m"))
FORMS = (("READER", "token reader of the candidate"), ("HEAD", "presence head alone"), ("HEAD-OR-READER", "head joined with the reader (OR)"))
PM = (("M", "missed lead frames (share of the frames with a lead)"), ("X", "extra marks (share of the frames with no lead of the label)"))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=8.6)
    ax.tick_params(axis="y", colors=INK2, length=0, labelsize=9.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res117", default=RES117)
    ap.add_argument("--res118", default=RES118)
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    st = json.load(io.open(a.res117, encoding="utf-8"))["values"]["settings"]
    pv = json.load(io.open(a.res118, encoding="utf-8"))["values"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig = plt.figure(figsize=(13.2, 7.4))
    fig.patch.set_facecolor(SURFACE)
    gs_top = fig.add_gridspec(1, 3, left=0.215, right=0.985, top=0.865, bottom=0.555, wspace=0.10)
    gs_bot = fig.add_gridspec(1, 2, left=0.215, right=0.985, top=0.335, bottom=0.115, wspace=0.10)
    ys = list(range(len(SETS) - 1, -1, -1))
    axes = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    for ax, (m, mname) in zip(axes, MEAS):
        style(ax)
        xmax = 0.0
        for y, (S, _) in zip(ys, SETS):
            for arm, _, col, filled, dy in ARMS:
                x = st[S]["levels"][arm][m]
                seeds = (st[S].get("levels_per_seed") or {}).get(arm, {}).get(m) or []
                if seeds:
                    ax.plot([min(seeds), max(seeds)], [y + dy, y + dy], color=col, linewidth=2.0, solid_capstyle="round", zorder=3)
                ax.plot([x], [y + dy], marker="o", markersize=8.0, color=col, markerfacecolor=(col if filled else SURFACE), markeredgecolor=(SURFACE if filled else col),
                        markeredgewidth=1.6, zorder=4)
                xmax = max([xmax, x] + list(seeds))
                ax.text(max([x] + list(seeds)), y + dy, "   %.4f" % x, color=INK, fontsize=8.0, ha="left", va="center")
        ax.set_xlim(-0.04 * xmax - 1e-4, xmax * 1.30 + 1e-3)
        ax.set_ylim(-0.6, len(SETS) - 0.4)
        ax.set_title(mname, color=INK, fontsize=10.2, loc="left", pad=10)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    labels = []
    for S, lab in SETS:
        n = (st[S].get("counts") or {}).get("n_lead_windows")
        labels.append(lab + ("\n%s windows with a lead" % n if n is not None else ""))
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, color=INK)
    for ax in axes[1:]:
        ax.set_yticks(ys)
    fig.text(0.008, 0.975, "A margin on the camera gap of the lead rule (rollout on the model's own actions, T1-kin with the heading bound)", color=INK, fontsize=11.0, ha="left", va="top")
    for (arm, lab, col, filled, _), x0 in zip(ARMS, (0.045, 0.285, 0.505, 0.725)):
        fig.lines.append(plt.Line2D([x0], [0.925], marker="o", markersize=7.5, color=col, markerfacecolor=(col if filled else SURFACE), markeredgecolor=(SURFACE if filled else col),
                                    markeredgewidth=1.5, linestyle="none", transform=fig.transFigure))
        fig.text(x0 + 0.010, 0.925, lab, color=INK2, fontsize=8.6, ha="left", va="center")
    fig.text(0.60, 0.505, "share of the windows with a lead (unsafe gap: of its own windows).  Dot: both model seeds pooled; bar: the two model seeds.", color=INK2, fontsize=8.6,
             ha="center", va="bottom")
    fig.text(0.008, 0.435, "A presence head on the existing narrow-window tokens (reads on recorded frames, T0: a diagnostic of the head, not driving performance)", color=INK, fontsize=11.0,
             ha="left", va="top")
    yf = list(range(len(FORMS) - 1, -1, -1))
    baxes = [fig.add_subplot(gs_bot[0, i]) for i in range(2)]
    for ax, (m, mname) in zip(baxes, PM):
        style(ax)
        bar = float(pv["bar"][m])
        xmax = bar
        for y, (F, _) in zip(yf, FORMS):
            d = pv["forms"][F]["pooled"][m]
            lo, hi = d["interval"]
            ax.plot([lo, hi], [y, y], color=B1, linewidth=2.0, solid_capstyle="round", zorder=3)
            ax.plot([d["value"]], [y], marker="o", markersize=8.0, color=B1, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
            ax.text(d["value"], y + 0.20, "%.4f" % d["value"], color=INK, fontsize=8.0, ha="center", va="bottom")
            xmax = max(xmax, hi)
        ax.axvline(bar, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=2)
        left = bar > 0.6 * xmax
        ax.text(bar, len(FORMS) - 0.30, ("bar %.4f (at most) " if left else " bar %.4f (at most)") % bar, color=INK2, fontsize=8.0, ha=("right" if left else "left"), va="center")
        ax.set_xlim(-0.02 * xmax, xmax * 1.22)
        ax.set_ylim(-0.6, len(FORMS) - 0.05)
        ax.set_title(mname, color=INK, fontsize=10.2, loc="left", pad=8)
        ax.set_yticks(yf)
        ax.set_yticklabels([lab for _, lab in FORMS] if ax is baxes[0] else [], color=INK)
    n = pv["forms"]["READER"]["per_seed"][0]
    fig.text(0.008, 0.012, "Top: two 10k models; the lead moves as logged; the label form is a reference and never a candidate.  Bottom: dot = both seeds pooled, bar = interval over "
             "%s clips (%s frames per seed, %s with a lead);\nthe bar was set before the fit; the label marking is a reference, not ground truth."
             % (pv["n_clips_with_frames"], format(n["n_frames"], ","), format(n["n_L"], ",")), color=INK2, fontsize=8.0, ha="left", va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "fig_gapmargin." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-GAPMARGIN DONE")


if __name__ == "__main__":
    main()
