# make_fig_gapbins.py -- figure for row 170: where the camera head's gap error sits (tier T0, recorded frames, 3,841 rows of 141 evaluation clips).
# Reads the ZZJSON line of falltrace/pod_t0/t0_gap_bins_c.out. Writes fig_gapbins.png and fig_gapbins.pdf beside this file.
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(V, "falltrace", "pod_t0", "t0_gap_bins_c.out")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BINS = (("0_10", "0-10 m"), ("10_20", "10-20 m"), ("20_40", "20-40 m"), ("40_60", "40-60 m"))
SER = (("fit2", "one fit of the head", BLUE), ("mean5", "mean of five fits", ORANGE))


def main():
    rec = None
    for l in io.open(SRC, encoding="utf-8"):
        if l.startswith("ZZJSON "):
            rec = json.loads(l[7:])
    if rec is None:
        raise SystemExit("no ZZJSON line")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.1), facecolor=SURFACE)
    w = 0.36
    panels = [("bins", "signed_m", "Mean error of the camera gap, by labelled distance\n(camera minus label; below zero = read too short)", "metres", "%+.1f", (-7.6, 3.2)),
              ("bins", "share_short", "Share of frames read more than 5 m too short,\nby labelled distance", "share of frames", "%.2f", (0, 0.62)),
              ("fed_bins", "label_minus_fed_m", "Label minus camera gap,\nby the distance the camera head itself reports", "metres", "%+.1f", (-7.6, 3.2))]
    for ax, (grp, key, title, ylab, fmt, ylim) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for si, (s, _lab, col) in enumerate(SER):
            for bi, (b, _bl) in enumerate(BINS):
                k0 = "%s_%s" % (s, b)
                v = rec[grp][k0 if k0 in rec[grp] else k0 + ".0001"][key]      # the tool writes the upper edge of the last fed bin as 60.0001
                x = bi + (si - 0.5) * (w + 0.03)
                ax.bar(x, v, width=w, color=col, edgecolor=SURFACE, linewidth=1.2)
                ax.text(x, v + (0.02 * (ylim[1] - ylim[0]) if v >= 0 else -0.02 * (ylim[1] - ylim[0])), fmt % v, ha="center", va="bottom" if v >= 0 else "top", color=INK, fontsize=7.3)
        ax.axhline(0, color=INK2, linewidth=0.9)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(BINS)))
        ax.set_xticklabels([bl for _b, bl in BINS], color=INK, fontsize=7.8)
        ax.set_title(title, fontsize=8.4, color=INK, loc="left", pad=6)
        ax.set_ylabel(ylab, color=INK2, fontsize=7.6)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", colors=INK2, length=0, labelsize=7.4)
        ax.tick_params(axis="x", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].set_xlabel("distance to the lead in the label", color=INK2, fontsize=7.6)
    axes[1].set_xlabel("distance to the lead in the label", color=INK2, fontsize=7.6)
    axes[2].set_xlabel("distance reported by the camera head", color=INK2, fontsize=7.6)
    fig.legend(handles=[matplotlib.patches.Patch(color=c, label=l) for _s, l, c in SER], loc="upper right", bbox_to_anchor=(0.992, 0.985), frameon=False, fontsize=7.6, ncol=2, labelcolor=INK)
    fig.suptitle("The camera head reads far leads too short, and averaging fits does not remove it", x=0.012, y=0.985, ha="left", fontsize=10.4, color=INK)
    f = rec["filters"]
    m = rec["mae_m"]
    fig.text(0.012, 0.012,
             "Row 170. Recorded frames (no rollout): %s frames with a lead in the label on 141 evaluation clips; head fit on 500 clips. Mean absolute error: one fit %.2f m, mean of five fits %.2f m.\n"
             "A filter over the last 3, 5, 9 frames raises the error of one fit to %.2f, %.2f, %.2f m (median). Right panel: given its own reading the head is close to calibrated, so a map of its output\n"
             "has 1 to 2 m to gain and not the 6 m by which far leads are read short. A diagnostic of the head, not driving performance; the label gap is the reference, not ground truth of the scene."
             % (format(rec["rows"], ","), m["fit2"], m["mean5"], f["fit2_median_k3"], f["fit2_median_k5"], f["fit2_median_k9"]),
             ha="left", va="bottom", color=INK2, fontsize=7.2, linespacing=1.4)
    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.27, wspace=0.17)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_gapbins." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-GAPBINS DONE rows %d" % rec["rows"])


if __name__ == "__main__":
    main()
