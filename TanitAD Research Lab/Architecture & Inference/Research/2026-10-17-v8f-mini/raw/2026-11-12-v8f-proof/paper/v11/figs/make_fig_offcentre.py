# make_fig_offcentre.py -- figure for row 171: the gap error of the camera head by how far the lead is off the axis of the fixed windows, with the bound arm (windows on the lead) beside it.
# Tier T0, recorded frames, 3,841 rows of 141 evaluation clips. Reads the ZZJSON line of falltrace/pod_t0/t0_offcentre.out. Writes fig_offcentre.png and .pdf beside this file.
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(V, "falltrace", "pod_t0", "t0_offcentre.out")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ABINS = (("0_1", "0-1"), ("1_2", "1-2"), ("2_4", "2-4"), ("4_8", "4-8"), ("8_up", "above 8"))
GBINS = (("0_10", "0-10 m"), ("10_20", "10-20 m"), ("20_40", "20-40 m"), ("40_60", "40-60 m"))


def main():
    rec = None
    for l in io.open(SRC, encoding="utf-8"):
        if l.startswith("ZZJSON "):
            rec = json.loads(l[7:])
    if rec is None:
        raise SystemExit("no ZZJSON line")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    w = 0.36

    def style(ax, title, ylab, xlab):
        ax.set_facecolor(SURFACE)
        ax.set_title(title, fontsize=8.4, color=INK, loc="left", pad=6)
        ax.set_ylabel(ylab, color=INK2, fontsize=7.6)
        ax.set_xlabel(xlab, color=INK2, fontsize=7.6)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", colors=INK2, length=0, labelsize=7.4)
        ax.tick_params(axis="x", length=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    for ax, grp, key_d, key_b, title, ylab, ylim, fmt in (
            (axes[0], "B", "D_mae", "B_mae", "Mean absolute gap error, all distances\n(lower is better)", "metres", (0, 17.5), "%.1f"),
            (axes[1], "C", "D_signed", "B_signed", "Mean gap error for far leads (label gap 40-60 m)\n(camera minus label; below zero = read too short)", "metres", (-17.5, 2.5), "%+.1f")):
        for bi, (b, _bl) in enumerate(ABINS):
            r = rec[grp][b]
            for si, (key, col) in enumerate(((key_d, BLUE), (key_b, ORANGE))):
                v = r[key]
                x = bi + (si - 0.5) * (w + 0.03)
                ax.bar(x, v, width=w, color=col, edgecolor=SURFACE, linewidth=1.2)
                ax.text(x, v + (0.25 if v >= 0 else -0.25), fmt % v, ha="center", va="bottom" if v >= 0 else "top", color=INK, fontsize=7.3)
        ax.axhline(0, color=INK2, linewidth=0.9)
        ax.set_ylim(*ylim)
        ax.set_xticks(range(len(ABINS)))
        ax.set_xticklabels(["%s\nn = %s" % (bl, format(rec[grp][b]["n"], ",")) for b, bl in ABINS], color=INK, fontsize=7.4)
        style(ax, title, ylab, "angle of the lead off the camera axis, degrees (n = frames)")
    ax = axes[2]
    for bi, (b, _bl) in enumerate(GBINS):
        v = rec["A"][b]["az_gt8"]
        ax.bar(bi, v, width=0.56, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
        ax.text(bi, v + 0.006, "%.2f" % v, ha="center", va="bottom", color=INK, fontsize=7.3)
    ax.set_ylim(0, 0.26)
    ax.set_xticks(range(len(GBINS)))
    ax.set_xticklabels([bl for _b, bl in GBINS], color=INK, fontsize=7.6)
    style(ax, "Share of frames with the lead more than 8 degrees\noff axis (outside the narrow fixed window)", "share of frames", "distance to the lead in the label")
    fig.legend(handles=[matplotlib.patches.Patch(color=BLUE, label="fixed windows (the form with no label)"), matplotlib.patches.Patch(color=ORANGE, label="windows placed on the lead by the label (a bound)")],
               loc="upper right", bbox_to_anchor=(0.992, 0.985), frameon=False, fontsize=7.6, ncol=2, labelcolor=INK)
    fig.suptitle("Where the camera head looks decides how well it reads the gap", x=0.012, y=0.985, ha="left", fontsize=10.4, color=INK)
    fig.text(0.012, 0.012,
             "Row 171. Recorded frames (no rollout): %s frames with a lead in the label on 141 evaluation clips; one fit of the same head on 500 clips per arm. The bound arm is not admissible at inference:\n"
             "the label says where to look. It shows what a window that follows the lead could gain; card ADD107 asks whether the head can find that direction from the camera alone.\n"
             "A diagnostic of the head, not driving performance; no interval; the label gap is the reference, not ground truth of the scene." % format(rec["rows"], ","),
             ha="left", va="bottom", color=INK2, fontsize=7.2, linespacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.30, wspace=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_offcentre." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-OFFCENTRE DONE rows %d" % rec["rows"])


if __name__ == "__main__":
    main()
