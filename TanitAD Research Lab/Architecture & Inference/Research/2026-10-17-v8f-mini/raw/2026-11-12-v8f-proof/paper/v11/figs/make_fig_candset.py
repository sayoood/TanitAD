# make_fig_candset.py -- figure of card ADD113 (CANDSET10K): the camera-fed candidate form of the lead rule at the convention and at three rollout ages under readout K,
# beside the reader-fed form and the label form. Tier T1-kin with the heading bound (kinematic rollout on the model's own actions). No number is typed here: every value is
# read from the pulled result file of the card. Output: fig_candset.png and fig_candset.pdf beside this file (or in --out-dir).
import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES = os.path.join(V, "candset", "pod_add113", "read", "CANDSET10K_RESULT.json")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
SETS = (("CONV", "convention\n(no pre-roll, pinned readout)"), ("K0", "readout K, no pre-roll"), ("K30", "readout K, pre-roll of 30 steps (6 s)"), ("K60", "readout K, pre-roll of 60 steps (12 s)"))
ARMS = (("RR", "rule fed by the present reader", ORANGE, 0.24), ("GEc", "candidate: rule fed by the camera gap, no label", BLUE, 0.0), ("RLc", "rule fed by the label (not admissible at inference)", GREEN, -0.24))
MEAS = (("C", "contact"), ("U", "unsafe gap"), ("F", "falling behind by more than 10 m"))


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
    st = d["values"]["settings"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=True, gridspec_kw={"wspace": 0.10})
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(SETS) - 1, -1, -1))
    for ax, (m, mname) in zip(axes, MEAS):
        style(ax)
        xmax = 0.0
        for y, (S, _) in zip(ys, SETS):
            for arm, _, col, dy in ARMS:
                x = st[S]["levels"][arm][m]
                seeds = (st[S].get("levels_per_seed") or {}).get(arm, {}).get(m) or []
                if seeds:
                    ax.plot([min(seeds), max(seeds)], [y + dy, y + dy], color=col, linewidth=2.0, solid_capstyle="round", zorder=3)
                ax.plot([x], [y + dy], marker="o", markersize=8.0, color=col, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4)
                xmax = max([xmax, x] + list(seeds))
                if arm == "GEc":
                    ax.text(max([x] + list(seeds)), y + dy, "   %.4f" % x, color=INK, fontsize=8.6, ha="left", va="center")
        ax.set_xlim(-0.03 * xmax - 1e-4, xmax * 1.28 + 1e-3)
        ax.set_ylim(-0.6, len(SETS) - 0.4)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(mname, color=INK, fontsize=10.2, loc="left", pad=10)
    labels = []
    for S, lab in SETS:
        n = (st[S].get("counts") or {}).get("n_lead_windows")
        labels.append(lab + ("\n%s windows with a lead" % n if n is not None else ""))
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, color=INK)
    for (arm, lab, col, _), x0 in zip(ARMS, (0.215, 0.415, 0.69)):
        fig.text(x0, 0.945, "●", color=col, fontsize=11, ha="left", va="center")
        fig.text(x0 + 0.012, 0.945, lab, color=INK2, fontsize=8.8, ha="left", va="center")
    fig.text(0.63, 0.105, "share of the windows with a lead (unsafe gap: of its own windows)", color=INK2, fontsize=8.8, ha="center", va="bottom")
    fig.text(0.008, 0.012, "Dot: both model seeds pooled; bar: the two model seeds. T1-kin with the heading bound, two 10k models; the lead moves as logged.\nThe label form uses label values and is a reference, not a candidate.",
             color=INK2, fontsize=8.0, ha="left", va="bottom")
    fig.subplots_adjust(left=0.215, right=0.985, top=0.86, bottom=0.20)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "fig_candset." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-CANDSET DONE")


if __name__ == "__main__":
    main()
