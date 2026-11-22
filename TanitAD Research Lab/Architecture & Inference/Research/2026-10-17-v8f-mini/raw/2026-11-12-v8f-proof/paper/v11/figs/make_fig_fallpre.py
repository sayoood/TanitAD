# make_fig_fallpre.py -- figure for row 169: what the rule log shows in the windows where the camera-fed lead rule falls behind and the label gap form does not.
# Reads falltrace/pod_pre/out_pre/FALLPRE.json (the second count, per-window table). Writes fig_fallpre.png and fig_fallpre.pdf beside this file. Every number drawn is recomputed from the table.
import io
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(V, "falltrace", "pod_pre", "out_pre", "FALLPRE.json")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GROUPS = [("excess", "Camera form falls behind,\nlabel form does not"),
          ("ref_only", "Label form falls behind,\ncamera form does not"),
          ("neither", "Neither falls behind"),
          ("both", "Both fall behind")]


def main():
    T = json.load(io.open(SRC, encoding="utf-8"))["windows"]
    G = {g: [r for r in T if r["group"] == g] for g, _l in GROUPS}
    n = {g: len(G[g]) for g in G}
    cut = {g: sum(r["n_cut"] for r in G[g]) for g in G}
    lab = {g: sum(r["n_lab"] for r in G[g]) for g in G}
    low = {g: sum(r["n_low"] for r in G[g]) for g in G}
    nol = {g: sum(r["n_nolab"] for r in G[g]) for g in G}
    slo = {g: sum(r["n_slope"] for r in G[g]) for g in G}
    dpre = {g: sum(r["dsum_pre"] for r in G[g]) / n[g] for g in G}
    dwin = {g: sum(r["dsum_win"] for r in G[g]) / n[g] for g in G}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.9), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1, 1, 1.45]})
    ys = list(range(len(GROUPS)))[::-1]
    share_panels = [("Camera gap more than 5 m\nshorter than the label gap", low, lab),
                    ("No lead in the label\n(the reader marks one)", nol, cut),
                    ("Closing input says closing,\nthe label gap does not close", slo, cut)]
    for ax, (title, num, den) in zip(axes[:3], share_panels):
        ax.set_facecolor(SURFACE)
        for y, (g, _l) in zip(ys, GROUPS):
            v = (num[g] / den[g]) if den[g] else 0.0
            ax.barh(y, v, height=0.56, color=BLUE, edgecolor=SURFACE, linewidth=1.2)
            ax.text(v + 0.008, y, "%.2f  (%s of %s)" % (v, format(num[g], ","), format(den[g], ",")), va="center", ha="left", color=INK, fontsize=7.6)
        ax.set_xlim(0, 0.46)
        ax.set_ylim(-0.6, len(GROUPS) - 0.4)
        ax.set_title(title, fontsize=8.4, color=INK, loc="left", pad=6)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=INK2, length=0, labelsize=7.4)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([])
        ax.set_xlabel("share of the rule's cut looks", color=INK2, fontsize=7.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticklabels(["%s\n(%s arm-windows)" % (l, format(n[g], ",")) for g, l in GROUPS], color=INK, fontsize=7.8)
    ax = axes[3]
    ax.set_facecolor(SURFACE)
    h = 0.27
    for y, (g, _l) in zip(ys, GROUPS):
        for off, val, col in ((h / 2 + 0.02, dpre[g], BLUE), (-h / 2 - 0.02, dwin[g], ORANGE)):
            ax.barh(y + off, val, height=h, color=col, edgecolor=SURFACE, linewidth=1.2)
            ax.text(val + (0.12 if val >= 0 else -0.12), y + off, "%+.2f" % val, va="center", ha="left" if val >= 0 else "right", color=INK, fontsize=7.6)
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_xlim(-3.6, 6.4)
    ax.set_ylim(-0.6, len(GROUPS) - 0.4)
    ax.set_title("Executed steps cut by the rule:\ncamera form minus label form (mean per window)", fontsize=8.4, color=INK, loc="left", pad=6)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=7.4)
    ax.tick_params(axis="y", length=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_xlabel("steps", color=INK2, fontsize=7.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.legend(bbox_to_anchor=(0.992, 0.03), handles=[matplotlib.patches.Patch(color=BLUE, label="in the pre-roll (6 s before the window)"), matplotlib.patches.Patch(color=ORANGE, label="inside the window")],
              loc="lower right", frameon=False, fontsize=7.4, labelcolor=INK, borderaxespad=0.2, handlelength=1.2)
    fig.suptitle("Where the lead rule with no label at inference falls behind and the label gap form does not: what the rule log shows", x=0.012, y=0.985, ha="left", fontsize=10.2, color=INK)
    fig.text(0.012, 0.012,
             "Row 169. Counts on banked rollouts (kinematic rollout on the model's own actions, heading bound, pre-roll of 30 steps (6 s)); two model seeds, five arms that share one reader and one head design;\n"
             "650 windows with a lead per arm. Falling behind: by more than 10 m. A cut look: the rule has a fed lead and cuts at least one executed step. Counts, no intervals: they rank candidates for the next arm\n"
             "and prove no cause. The label gap is the rule's reference, not ground truth of the scene. No re-perception.",
             ha="left", va="bottom", color=INK2, fontsize=7.2, linespacing=1.4)
    fig.subplots_adjust(left=0.158, right=0.99, top=0.80, bottom=0.28, wspace=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_fallpre." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-FALLPRE DONE groups %s cut looks %s low %s nolab %s slope %s dpre %s dwin %s" % (
        n, cut, low, nol, slo, {g: round(v, 2) for g, v in dpre.items()}, {g: round(v, 2) for g, v in dwin.items()}))


if __name__ == "__main__":
    main()
