# make_fig_gapmarginage.py -- figure of cards ADD117 (GAPMARGIN10K) and ADD119 (GAPMARGINAGE10K): the margin on the camera gap of the lead rule at all four settings.
# Tier T1-kin with the heading bound (kinematic rollout on the model's own actions). No number is typed here: every value is read from the pulled result files of the
# two cards. Blue, light to dark = margin of 0, 10, 20 per cent; green, open = the label form (a reference with label values, never a candidate).
# Output: fig_gapmarginage.png and fig_gapmarginage.pdf beside this file (or in --out-dir).
import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

import make_fig_gapmargin as G  # noqa: E402

RES119 = os.path.join(G.V, "gapmarginage", "pod_add119", "read", "GAPMARGINAGE10K_RESULT.json")
SETS = (("CONV", "convention\n(no pre-roll, pinned readout)"), ("K0", "readout K, no pre-roll"), ("K30", "readout K, pre-roll of 30 steps (6 s)"),
        ("K60", "readout K, pre-roll of 60 steps (12 s)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res117", default=G.RES117)
    ap.add_argument("--res119", default=RES119)
    ap.add_argument("--out-dir", default=G.HERE)
    a = ap.parse_args()
    st = {}
    for p in (a.res117, a.res119):
        st.update(json.load(io.open(p, encoding="utf-8"))["values"]["settings"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": G.GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.4), sharey=True, gridspec_kw={"wspace": 0.10})
    fig.patch.set_facecolor(G.SURFACE)
    ys = list(range(len(SETS) - 1, -1, -1))
    for ax, (m, mname) in zip(axes, G.MEAS):
        G.style(ax)
        xmax = 0.0
        for y, (S, _) in zip(ys, SETS):
            for arm, _, col, filled, dy in G.ARMS:
                x = st[S]["levels"][arm][m]
                seeds = (st[S].get("levels_per_seed") or {}).get(arm, {}).get(m) or []
                if seeds:
                    ax.plot([min(seeds), max(seeds)], [y + dy, y + dy], color=col, linewidth=2.0, solid_capstyle="round", zorder=3)
                ax.plot([x], [y + dy], marker="o", markersize=8.0, color=col, markerfacecolor=(col if filled else G.SURFACE), markeredgecolor=(G.SURFACE if filled else col),
                        markeredgewidth=1.6, zorder=4)
                xmax = max([xmax, x] + list(seeds))
                ax.text(max([x] + list(seeds)), y + dy, "   %.4f" % x, color=G.INK, fontsize=8.0, ha="left", va="center")
        ax.set_xlim(-0.04 * xmax - 1e-4, xmax * 1.30 + 1e-3)
        ax.set_ylim(-0.6, len(SETS) - 0.4)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_title(mname, color=G.INK, fontsize=10.2, loc="left", pad=10)
    labels = []
    for S, lab in SETS:
        n = (st[S].get("counts") or {}).get("n_lead_windows")
        labels.append(lab + ("\n%s windows with a lead" % n if n is not None else ""))
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, color=G.INK)
    fig.text(0.008, 0.975, "A margin on the camera gap of the lead rule at four settings (rollout on the model's own actions, T1-kin with the heading bound)", color=G.INK, fontsize=11.0,
             ha="left", va="top")
    for (arm, lab, col, filled, _), x0 in zip(G.ARMS, (0.045, 0.285, 0.505, 0.725)):
        fig.lines.append(plt.Line2D([x0], [0.918], marker="o", markersize=7.5, color=col, markerfacecolor=(col if filled else G.SURFACE), markeredgecolor=(G.SURFACE if filled else col),
                                    markeredgewidth=1.5, linestyle="none", transform=fig.transFigure))
        fig.text(x0 + 0.010, 0.918, lab, color=G.INK2, fontsize=8.6, ha="left", va="center")
    fig.text(0.63, 0.085, "share of the windows with a lead (unsafe gap: of its own windows)", color=G.INK2, fontsize=8.8, ha="center", va="bottom")
    fig.text(0.008, 0.012, "Dot: both model seeds pooled; bar: the two model seeds. Two 10k models; the lead moves as logged. The label form is a reference and never a candidate.\n"
             "The setting with a pre-roll of 60 steps has few windows and decides nothing alone.", color=G.INK2, fontsize=8.0, ha="left", va="bottom")
    fig.subplots_adjust(left=0.215, right=0.985, top=0.845, bottom=0.155)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "fig_gapmarginage." + ext), dpi=170, facecolor=G.SURFACE)
    print("ZZ FIG-GAPMARGINAGE DONE")


if __name__ == "__main__":
    main()
