# make_fig_lookat.py -- figure of row 173 (card ADD107, LOOKAT-T0): does a head on the tokens of the fixed windows find the direction of the lead with no label.
# Tier T0 (recorded frames, no rollout; a property of the head's direction output, not driving performance). No number is typed here: every value is read from the pulled result
# file of the card. Output: fig_lookat.png and fig_lookat.pdf beside this file.
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(V, "lookat", "pod_add107", "read", "LOOKAT_RESULT.json")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
SEEDS = ("s0", "s1", "s2")


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
    fx, fits = v["fixed"], v["fits"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 5.0), gridspec_kw={"wspace": 0.2, "width_ratios": [1.0, 1.3]})
    fig.patch.set_facecolor(SURFACE)
    style(a)
    rows = [("S8", "inside a 16-degree window\n(within 8 degrees)"), ("S4", "in its central half\n(within 4 degrees)"), ("S8_out", "inside, on the %s frames\nwith the lead off the axis" % format(v["n_outside"], ","))]
    for y, (key, _lab) in zip((2, 1, 0), rows):
        xs = [fits[s][key] for s in SEEDS]
        a.scatter([fx[key]], [y], s=95, color=ORANGE, edgecolor=SURFACE, linewidth=2.0, zorder=3)
        a.scatter([fits["perm"][key]], [y - 0.2], s=70, color=GREEN, edgecolor=SURFACE, linewidth=2.0, zorder=3)
        a.scatter(xs, [y] * 3, s=95, color=BLUE, edgecolor=SURFACE, linewidth=2.0, zorder=4)
        a.text(sum(xs) / 3.0, y + 0.2, ", ".join("%.3f" % x for x in sorted(xs)), ha="center", va="bottom", color=INK2, fontsize=8.4)
        if fx[key] > 0.2:
            a.text(fx[key] - 0.03, y, "%.3f" % fx[key], ha="right", va="center", color=INK2, fontsize=8.4)
        else:
            a.text(fx[key], y + 0.2, "%.3f" % fx[key], ha="center", va="bottom", color=INK2, fontsize=8.4)
        a.text(fits["perm"][key] + 0.03, y - 0.2, "%.3f" % fits["perm"][key], ha="left", va="center", color=INK2, fontsize=7.8)
    a.set_yticks([2, 1, 0])
    a.set_yticklabels([lab for _k, lab in rows], color=INK, fontsize=9, linespacing=1.3)
    a.set_ylim(-0.75, 2.6)
    a.set_xlim(-0.06, 1.06)
    a.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    a.set_xlabel("share of the %s frames with a lead" % format(v["n_rows"], ","), color=INK2, fontsize=9)
    a.set_title("How often the lead is where the window looks", loc="left", color=INK, fontsize=10.5, pad=10)
    style(b)
    bins = [x["bin"] for x in fx["by_az"]]
    w = 0.36
    for i, _bn in enumerate(bins):
        f0 = fx["by_az"][i]["S8"]
        l0 = fits["s0"]["by_az"][i]["S8"]
        b.bar(i - w / 2 - 0.01, f0, width=w, color=ORANGE, zorder=2)
        b.bar(i + w / 2 + 0.01, l0, width=w, color=BLUE, zorder=2)
        b.text(i - w / 2 - 0.01, f0 + 0.02, "%.2f" % f0, ha="center", va="bottom", color=INK2, fontsize=8.4)
        others = [fits[s]["by_az"][i]["S8"] for s in ("s1", "s2")]
        b.scatter([i + w / 2 + 0.01] * 2, others, s=26, color=INK, edgecolor=SURFACE, linewidth=1.0, zorder=4)
        b.text(i + w / 2 + 0.01, max([l0] + others) + 0.03, "%.2f" % l0, ha="center", va="bottom", color=INK2, fontsize=8.4)
    b.set_xticks(range(len(bins)))
    b.set_xticklabels(["%s\n%s frames" % (bn.replace("deg", "degrees"), format(fx["by_az"][i]["n"], ",")) for i, bn in enumerate(bins)], color=INK, fontsize=8.8, linespacing=1.5)
    b.set_ylim(0, 1.13)
    b.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    b.set_ylabel("share with the lead inside the window", color=INK2, fontsize=9)
    b.set_title("By how far the lead is off the camera axis", loc="left", color=INK, fontsize=10.5, pad=10)
    hs = [plt.Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=c, markeredgecolor=SURFACE) for c in (ORANGE, BLUE, GREEN)]
    fig.legend(hs, ["fixed window, straight ahead", "window at the head's direction (three fits)", "control: head fit on shuffled targets"], loc="upper right", bbox_to_anchor=(0.985, 0.94), ncol=3,
               frameon=False, fontsize=8.8, labelcolor=INK, handletextpad=0.3, columnspacing=1.5)
    fig.suptitle("The head finds where the lead is, from the camera alone", x=0.012, ha="left", color=INK, fontsize=12.5, fontweight="bold", y=0.985)
    fig.text(0.012, 0.012, "Tier T0 (recorded frames, no rollout): a property of the head's direction output on 141 evaluation clips, not driving performance. The head reads only the tokens of the two "
             "fixed windows; no label at inference, no new encode.\nInside: the centre of the lead's box lies within 8 degrees of the window's centre in azimuth and elevation. Right panel: bars for fit "
             "seed 0, black dots for fit seeds 1 and 2.\nThe gap reading with windows placed this way is not measured here.",
             ha="left", va="bottom", color=INK2, fontsize=7.6, linespacing=1.4)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.8, bottom=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_lookat." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-LOOKAT DONE name %s rows %d outside %d bins %s" % (d["name"], v["n_rows"], v["n_outside"], ",".join(bins)))


if __name__ == "__main__":
    main()
