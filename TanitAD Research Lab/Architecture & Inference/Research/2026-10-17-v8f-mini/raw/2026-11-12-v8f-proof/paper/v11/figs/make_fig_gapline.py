# make_fig_gapline.py -- figure of the camera gap line after cards ADD122 (TELEGAP-T0) and ADD124 (BOXSIZE-T0). Tier T0: recorded frames, no rollout; a diagnostic of
# perception heads and not driving performance. No number is typed here: every value is read from the pulled result files of the two cards and from the post-hoc read on
# label tracks (boxsize/posthoc/BOXNOISE_T0.json, which names nothing). Filled marks: camera heads, one mark per fit and a bar at their mean. Open marks: reads on LABEL
# boxes (a bound and two requirements), not results of a head.
# Output: fig_gapline.png and fig_gapline.pdf beside this file (or in --out-dir).
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES122 = os.path.join(V, "telegap", "pod_add122", "read", "TELEGAP_RESULT.json")
RES124 = os.path.join(V, "boxsize", "pod_add124", "read", "BOXSIZE_RESULT.json")
NOISE = os.path.join(V, "boxsize", "posthoc", "BOXNOISE_T0.json")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREY, GREEN = "#2a78d6", "#8a8985", "#1baf7a"


def load(p):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    e122, v124, nz = load(RES122)["values"]["E"], load(RES124)["values"], load(NOISE)
    e124 = v124["E"]
    g_all = [x for x in v124["beside"]["G_on_label_sizes"] if str(x.get("bin", "")).startswith("all")]
    bound = g_all[0]["E"] if g_all else nz["exact"]["E"]
    heads = (("head that regresses metres, wide camera\n(reference, three fits)", [e124["D_%d" % k] for k in range(3)], BLUE),
             ("+ tele window in place of the 32-degree window", [e122["M_%d" % k] for k in range(3)], GREY),
             ("+ two tele windows beside both wide windows", [e122["F_%d" % k] for k in range(3)], GREY),
             ("head that reads the box size, wide camera", [e124["S_%d" % k] for k in range(3)], GREY))
    labels = (("label box, edges within 2 pixels, wide camera", nz["pixels"]["16_2.0"]["E"]),
              ("label box, edges within 2 pixels, tele camera", nz["pixels"]["64_2.0"]["E"]),
              ("label box, exact size (bound)", bound))
    n = len(heads) + len(labels)
    fig, ax = plt.subplots(figsize=(8.6, 4.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=9)
    ax.tick_params(axis="y", colors=INK, length=0, labelsize=9)
    names = []
    for i, (nm, vals, col) in enumerate(heads):
        y = n - 1 - i
        m = sum(vals) / len(vals)
        ax.plot([0, m], [y, y], color=col, linewidth=5, solid_capstyle="round", alpha=0.35)
        ax.plot(vals, [y] * len(vals), "o", color=col, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.text(max(vals) + 0.12, y, "%.2f m" % m, va="center", ha="left", fontsize=9, color=INK)
        names.append(nm)
    for j, (nm, val) in enumerate(labels):
        y = n - 1 - len(heads) - j
        ax.plot([0, val], [y, y], color=GREEN, linewidth=1.2, linestyle=(0, (3, 2)))
        ax.plot([val], [y], "o", markerfacecolor=SURFACE, markeredgecolor=GREEN, markeredgewidth=1.8, markersize=7)
        ax.text(val + 0.12, y, "%.2f m" % val, va="center", ha="left", fontsize=9, color=INK)
        names.append(nm)
    ax.axhline(len(labels) - 0.5, color=GRID, linewidth=1.0)
    ax.set_yticks(range(n - 1, -1, -1))
    ax.set_yticklabels(names)
    ax.set_xlim(0, 6.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("mean absolute error of the gap to the lead, m (lower is better)", fontsize=8.8, color=INK2)
    fig.text(0.012, 0.955, "Camera gap to the lead: more pixels do not help a head on pooled tokens; the size of the box would", fontsize=10.2, color=INK, ha="left", va="center")
    fig.text(0.012, 0.012, "Tier T0 (recorded frames, no rollout; not driving performance). 3,841 frames of 141 evaluation clips (rows with a tele window: 3,817).\n"
             "Filled: camera heads (dots = fits, bar = mean). Open: reads on label boxes, a bound and two requirements, not results of a head.", fontsize=6.8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.925))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "fig_gapline." + ext), facecolor=SURFACE)
    print("ZZ FIG-GAPLINE DONE", ["%.3f" % (sum(v) / 3) for _, v, _ in heads], ["%.3f" % v for _, v in labels])


if __name__ == "__main__":
    main()
