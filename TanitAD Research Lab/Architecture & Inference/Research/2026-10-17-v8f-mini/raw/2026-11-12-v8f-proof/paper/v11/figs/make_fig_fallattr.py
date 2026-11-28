# make_fig_fallattr.py -- figure of row 176 (card ADD110, FALLATTR10K): where in the gap range the price in falling behind of the camera-fed lead rule sits.
# Tier T1-kin with the heading bound (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s). No number is typed here: every value is read from the
# pulled result file of the card. Output: fig_fallattr.png and fig_fallattr.pdf beside this file (or in --out-dir).
import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(V, "fallattr", "pod_add110", "read", "FALLATTR10K_RESULT.json")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ARMS = (("LA", "label gap on every lead row"), ("LN", "label gap below 20 m"), ("LM", "label gap from 20 to 40 m"), ("LF", "label gap from 40 m"))
MEAS = (("F", "falling behind by more than 10 m"), ("C", "contact"), ("U", "unsafe gap"))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=8.6)
    ax.tick_params(axis="y", colors=INK2, length=0, labelsize=9.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    d = json.load(io.open(a.res, encoding="utf-8"))
    v = d["values"]
    lv, pairs = v["levels"], v["pairs"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), sharey=True, gridspec_kw={"wspace": 0.10})
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(ARMS) - 1, -1, -1))
    for ax, (m, mname) in zip(axes, MEAS):
        style(ax)
        ref = lv["RLc"][m] - lv["GEc"][m]
        xs_all = [0.0, ref]
        for y, (x, _) in zip(ys, ARMS):
            p = pairs["%s-VS-GEc" % x][m]
            lo, hi = p["interval"]
            ax.plot([lo, hi], [y, y], color=BLUE, linewidth=2.0, solid_capstyle="round", zorder=3)
            for s in p["per_seed"]:
                ax.plot([s], [y - 0.19], marker="|", markersize=7, markeredgewidth=1.3, color=INK2, zorder=3)
            ax.plot([p["pooled"]], [y], marker="o", markersize=8.5, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
            ax.text(hi, y + 0.21, "%+.4f" % p["pooled"], color=INK, fontsize=8.6, ha="right", va="bottom")
            xs_all += [lo, hi] + list(p["per_seed"])
        ax.axvline(0.0, color=INK2, linewidth=1.0, zorder=2)
        ax.axvline(ref, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)), zorder=2)
        span = max(abs(min(xs_all)), abs(max(xs_all))) * 1.22 + 1e-4
        ax.set_xlim(-span, span)
        ax.set_ylim(-0.75, len(ARMS) - 0.25)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(mname, color=INK, fontsize=10.2, loc="left", pad=20)
        ax.text(ref, len(ARMS) - 0.30, "label form  %+.4f" % ref, color=INK2, fontsize=8.2, ha="center", va="bottom")
    fig.text(0.5875, 0.115, "difference to the camera gap on every row (share of the windows of the measure)", color=INK2, fontsize=8.8, ha="center", va="bottom")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([lab for _, lab in ARMS], color=INK)
    fig.text(0.008, 0.012, "Dot: both model seeds pooled; bar: interval of the two-level bootstrap; small strokes: each model seed. Dashed line: the rule fed by the label (presence, gap and closing), a bound and not admissible at inference.",
             color=INK2, fontsize=8.0, ha="left", va="bottom")
    fig.subplots_adjust(left=0.185, right=0.99, top=0.84, bottom=0.215)
    os.makedirs(a.out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "fig_fallattr." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-FALLATTR DONE")


if __name__ == "__main__":
    main()
