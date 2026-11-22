# make_fig_lookgap.py -- figure of row 174 (card ADD108, LOOKGAP-T0): does the gap head read the gap better with its windows placed at the direction a head finds from the camera alone.
# Tier T0 (recorded frames, no rollout; a property of a perception head, not driving performance). No number is typed here: every value is read from the pulled result
# file of the card. Output: fig_lookgap.png and fig_lookgap.pdf beside this file.
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(V, "lookgap", "pod_add108", "read", "LOOKGAP_RESULT.json")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
ARMS = (("fixed windows", ("D0", "D1", "D2"), ORANGE), ("windows at the head's direction", ("P0", "P1", "P2"), BLUE), ("windows placed by the label (bound)", ("B0", "B1"), GREEN))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=8.8)
    ax.tick_params(axis="y", colors=INK2, length=0, labelsize=8.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    d = json.load(io.open(RES, encoding="utf-8"))
    v = d["values"]
    E = v["E"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"wspace": 0.22, "width_ratios": [1.0, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    # panel a: gap error of every fit, one row per arm
    style(a)
    ys = (2, 1, 0)
    for y, (lab, keys, col) in zip(ys, ARMS):
        xs = [E[k] for k in keys]
        yy = [y + (i - (len(xs) - 1) / 2.0) * 0.16 for i in range(len(xs))]
        a.scatter(xs, yy, s=95, color=col, edgecolor=SURFACE, linewidth=2.0, zorder=4)
        a.text(sum(xs) / len(xs), y + 0.34, ", ".join("%.2f" % x for x in xs), ha="center", va="bottom", color=INK2, fontsize=8.4)
    a.set_yticks(list(ys))
    a.set_yticklabels(["fixed windows\n(3 fits)", "windows at the\nhead's direction\n(3 fits)", "windows placed by\nthe label (bound,\n2 fits)"], color=INK, fontsize=9, linespacing=1.3)
    a.set_ylim(-0.6, 2.7)
    hi = max(E[k] for _l, ks, _c in ARMS for k in ks)
    a.set_xlim(0, hi * 1.18)
    a.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    a.set_xlabel("mean absolute gap error, m (%s frames with a lead, %d clips)" % (format(v["n_rows"], ","), v["n_clips"]), color=INK2, fontsize=9)
    a.set_title("Gap error of each fit", loc="left", color=INK, fontsize=10.5, pad=10)
    iv = v["iv"]
    a.text(0.0, -0.19, "error at the head's direction / error of the fixed windows\n(mean of three fits each; fit seeds 0, 1, 2 from bottom to top): %.3f [%.3f, %.3f]" % (v["R"], iv[0], iv[1]),
           transform=a.transAxes, ha="left", va="top", color=INK2, fontsize=8.6, linespacing=1.35)
    # panel b: gap error by how far the lead is off the axis
    style(b)
    bins = [x["bin"] for x in v["bins_az"]["D0"]]
    w = 0.26
    for j, (key, col, lab) in enumerate((("D0", ORANGE, "fixed windows"), ("P0", BLUE, "at the head's direction"), ("B0", GREEN, "placed by the label (bound)"))):
        for i, row in enumerate(v["bins_az"][key]):
            if row["E"] is None:
                continue
            x = i + (j - 1) * (w + 0.02)
            b.bar(x, row["E"], width=w, color=col, zorder=3, label=lab if i == 0 else None)
            b.text(x, row["E"] + 0.12, "%.1f" % row["E"], ha="center", va="bottom", color=INK2, fontsize=8.0)
    b.set_xticks(range(len(bins)))
    b.set_xticklabels(["%s\n%s frames" % (bn.replace(" deg", " degrees"), format(v["bins_az"]["D0"][i]["n"], ",")) for i, bn in enumerate(bins)], color=INK, fontsize=8.8, linespacing=1.3)
    b.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    b.set_ylabel("mean absolute gap error, m (fit seed 0 of each arm)", color=INK2, fontsize=9)
    b.set_xlabel("how far the lead in the label is off the camera axis", color=INK2, fontsize=9)
    b.set_title("Gap error by the lead's direction", loc="left", color=INK, fontsize=10.5, pad=10)
    b.legend(loc="upper left", bbox_to_anchor=(0.0, -0.215), ncol=3, frameon=False, fontsize=8.4, labelcolor=INK2, handlelength=1.1, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
    fig.text(0.012, 0.012, "Tier T0: recorded frames, no rollout; a read of one input of the lead rule, not driving performance. Row 174, card ADD108. The bound places its windows from the label and is not admissible at inference.",
             ha="left", va="bottom", color=INK2, fontsize=7.8)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.9, bottom=0.27)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_lookgap." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-LOOKGAP DONE name %s" % d.get("name"))


if __name__ == "__main__":
    main()
