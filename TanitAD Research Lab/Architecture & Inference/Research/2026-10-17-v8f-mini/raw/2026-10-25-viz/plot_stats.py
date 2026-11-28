"""plot_stats.py -- fig_attention_stats.png from fig_attention_stats.json + controls.json (both seeds, episode-cluster bootstrap CIs)."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from viz_lib import LV_COL, LV_CMAP, INK, MUTED, BG, PANEL  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
R = json.load(open(f"{OUT}/fig_attention_stats.json"))
C = json.load(open(f"{OUT}/controls.json"))
SEEDS = ("M2b_s0", "M2b_s1")
LV = ("S", "T", "O")
COND = ("ego", "nav", "lim", "ck", "code", "age")
HAT = {SEEDS[0]: "", SEEDS[1]: "////"}
ALPHA = {SEEDS[0]: 1.0, SEEDS[1]: 0.55}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": PANEL, "axes.edgecolor": "#c9ced6", "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.labelsize": 8.5})
fig = plt.figure(figsize=(19.2, 12.4), dpi=100, facecolor=BG)
gs = fig.add_gridspec(3, 12, left=0.045, right=0.99, top=0.915, bottom=0.06, hspace=0.62, wspace=1.2)


def val(s, lv, key, i=0):
    d = R["seeds"][s][lv][key]
    return d["mean"][i], d["lo"][i], d["hi"][i]


def err(m, lo, hi):
    return [[max(m - lo, 0)], [max(hi - m, 0)]]


def grouped(ax, getter, ncat, cat_labels, ylabel, ylim=None, uniform=None, title=""):
    w = 0.36
    x0 = np.arange(ncat)
    for si, s in enumerate(SEEDS):
        for li, lv in enumerate(LV):
            pass
    # categories on the x axis; per category 3 levels x 2 seeds
    for li, lv in enumerate(LV):
        for si, s in enumerate(SEEDS):
            ms, los, his = getter(s, lv)
            xs = x0 + (li - 1) * 0.27 + (si - 0.5) * 0.12
            ax.bar(xs, ms, width=0.115, color=LV_COL[lv], alpha=ALPHA[s], hatch=HAT[s], edgecolor="white" if si == 0 else LV_COL[lv], linewidth=0.4,
                   label=f"{lv} {s[-2:]}" if True else None)
            ax.errorbar(xs, ms, yerr=[np.maximum(np.array(ms) - np.array(los), 0), np.maximum(np.array(his) - np.array(ms), 0)], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
    if uniform is not None:
        for xi, u in zip(x0, uniform):
            ax.plot([xi - 0.42, xi + 0.42], [u, u], color=INK, lw=1.1, ls="--")
    ax.set_xticks(x0)
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.grid(axis="y", color="#e6e9ee", lw=0.6)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


# (a) cond vs scene
ax = fig.add_subplot(gs[0, 0:3])
for li, lv in enumerate(LV):
    for si, s in enumerate(SEEDS):
        m, lo, hi = val(s, lv, "cond_mass")
        y = li * 1.0 - (si - 0.5) * 0.36 * -1
        ax.barh(y, 1 - m, color=LV_COL[lv], alpha=ALPHA[s], hatch=HAT[s], height=0.32, edgecolor="white")
        ax.barh(y, m, left=1 - m, color="#4a4f57", height=0.32)
        ax.text(0.5, y, f"scene {100 * (1 - m):.1f}%   cond {100 * m:.1f}%  [{100 * lo:.1f}, {100 * hi:.1f}]", ha="center", va="center", fontsize=7.8, color="white" if m < 0.4 else "white", fontweight="bold")
    ax.text(-0.02, li, lv, ha="right", va="center", fontsize=12, fontweight="bold", color=LV_COL[lv])
ax.set_xlim(0, 1)
ax.set_ylim(2.6, -0.6)
ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_title("A. where attention goes: scene vs cond tokens")
ax.text(0, -0.26, "dark = cond tokens (ego, nav, lim, ck, code, age); solid = seed 0, hatched = seed 1; [95% CI over episodes]", transform=ax.transAxes, fontsize=7.4, color=MUTED)

# (b),(c) cond-token mass tables
for pi, s in enumerate(SEEDS):
    ax = fig.add_subplot(gs[0, 3 + 2 * pi:5 + 2 * pi])
    M_ = np.full((3, 6), np.nan)
    for li, lv in enumerate(LV):
        nm = R["cond_names"][lv]
        for j, n in enumerate(nm):
            M_[li, COND.index(n)] = 100 * R["seeds"][s][lv]["cond_token_mass"]["mean"][j]
    im = ax.imshow(np.ma.masked_invalid(M_), cmap="Greys", vmin=0, vmax=60, aspect="auto")
    for li in range(3):
        for j in range(6):
            if np.isfinite(M_[li, j]):
                ax.text(j, li, f"{M_[li, j]:.1f}", ha="center", va="center", fontsize=9, color="white" if M_[li, j] > 30 else INK, fontweight="bold")
            else:
                ax.text(j, li, "-", ha="center", va="center", fontsize=9, color="#b0b6bf")
    ax.set_xticks(range(6))
    ax.set_xticklabels(COND)
    ax.set_yticks(range(3))
    ax.set_yticklabels(LV)
    for t_, lv in zip(ax.get_yticklabels(), LV):
        t_.set_color(LV_COL[lv])
        t_.set_fontweight("bold")
    ax.set_title(f"{'B' if pi == 0 else 'C'}. cond-token share, seed {pi} (%)")

# (d) effective tokens
ax = fig.add_subplot(gs[0, 7:9])
for li, lv in enumerate(LV):
    for si, s in enumerate(SEEDS):
        m, lo, hi = val(s, lv, "effective_tokens_exp_entropy")
        ax.bar(li + (si - 0.5) * 0.36, m, width=0.34, color=LV_COL[lv], alpha=ALPHA[s], hatch=HAT[s], edgecolor="white")
        ax.errorbar(li + (si - 0.5) * 0.36, m, yerr=err(m, lo, hi), fmt="none", ecolor=INK, capsize=2)
        ax.text(li + (si - 0.5) * 0.36, hi + 0.6, f"{m:.1f}", ha="center", fontsize=8)
ax.set_xticks(range(3))
ax.set_xticklabels(LV)
ax.set_ylabel("exp(entropy) of scene attention\n(of 160 tokens; uniform = 160)")
ax.set_title("D. effective tokens")
ax.grid(axis="y", color="#e6e9ee", lw=0.6)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)

# (e) head peakedness text-table
ax = fig.add_subplot(gs[0, 9:12])
ax.axis("off")
ax.set_title("E. head x query rows are near one-hot")
lines = ["level_layer   max-w  rows>.9  argmax/16"]
for s in SEEDS:
    lines.append(f"-- {s}")
    for k, d in R["head_peakedness"][s].items():
        lines.append(f"{k:<10s}   {d['mean_max_weight_per_head_query']:.2f}   {100 * d['frac_rows_max_gt_0.9']:3.0f}%     {d['mean_distinct_argmax_tokens_of_16']:.1f}")
ax.text(0.0, 0.95, "\n".join(lines), family="DejaVu Sans Mono", fontsize=8.3, va="top", transform=ax.transAxes, linespacing=1.45)
ax.text(0.0, -0.02, "max-w = mean over windows, heads, queries of the largest weight;\nrows>.9 = share of (head,query) rows with max weight > 0.9;\nargmax/16 = distinct argmax tokens among the 16 rows.", transform=ax.transAxes, fontsize=7.2, color=MUTED, va="top")

# (f) per-history-frame share of scene mass
ax = fig.add_subplot(gs[1, 0:4])
grouped(ax, lambda s, lv: tuple([R["seeds"][s][lv]["frame_share_of_scene"][k][:4] for k in ("mean", "lo", "hi")]), 4,
        ["oldest\n(t-3)", "t-2", "t-1", "tick frame\n(newest)"], "share of the SCENE attention", (0, 1.0), uniform=[0.25] * 4,
        title="F. recency: attention per history frame")
ax.text(0.01, 0.97, "dashed = uniform (25%)", transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top")
# (g) per-row share
ax = fig.add_subplot(gs[1, 4:8])
grouped(ax, lambda s, lv: tuple([R["seeds"][s][lv]["row_share_of_scene"][k][:4] for k in ("mean", "lo", "hi")]), 4,
        ["row 0\nfar / sky", "row 1\nhorizon", "row 2\nroad ahead", "row 3\nnear / bonnet"], "share of the SCENE attention", (0, 0.8), uniform=[0.25] * 4,
        title="G. far vs near: attention per token row (all 4 frames)")
# (h) randomisation control
ax = fig.add_subplot(gs[1, 8:12])
for li, lv in enumerate(LV):
    d = C["randomisation"][lv]["spearman_all_160_tokens"]
    d2 = C["randomisation"][lv]["spearman_tick_frame_40_tokens"]
    for k, (dd, off, lab) in enumerate(((d, -0.18, "160 scene tokens"), (d2, 0.18, "40 tick-frame tokens"))):
        m, lo, hi = dd["mean"][0], dd["lo"][0], dd["hi"][0]
        ax.bar(li + off, m, width=0.32, color=LV_COL[lv], alpha=1.0 if k == 0 else 0.55, edgecolor="white", label=lab if li == 0 else None)
        ax.errorbar(li + off, m, yerr=err(m, lo, hi), fmt="none", ecolor=INK, capsize=2)
        ax.text(li + off, (hi if m >= 0 else lo) + (0.02 if m >= 0 else -0.045), f"{m:+.2f}", ha="center", fontsize=8)
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylim(-0.6, 1.0)
ax.set_xticks(range(3))
ax.set_xticklabels([f"{lv} reader\nrandomised" for lv in LV])
ax.set_ylabel("Spearman rho, trained vs randomised map")
ax.set_title("H. control: randomise ONE Reader (Adebayo 2018)")
ax.legend(fontsize=7.5, loc="upper right", frameon=False)
ax.text(0.01, 0.03, "rho ~ 0 => the map depends on the trained weights. The other two levels' attention changed by exactly\nmax|d| = 0 (downstream inputs frozen at trained values).  Seed 0, 2,399 windows, CI over episodes.",
        transform=ax.transAxes, fontsize=7.2, color=MUTED, va="bottom")
for a_ in (ax,):
    for s_ in ("top", "right"):
        a_.spines[s_].set_visible(False)

# draw as one wide axis per level with the two seeds stacked
for li, lv in enumerate(LV):
    sub = gs[2, li * 4:li * 4 + 4].subgridspec(2, 1, hspace=0.35)
    for pi, s in enumerate(SEEDS):
        ax = fig.add_subplot(sub[pi])
        m = np.array(R["seeds"][s][lv]["mean_tick_frame_map_4x10"])
        ax.imshow(m, cmap=LV_CMAP[lv], vmin=0, aspect="auto")
        for r in range(4):
            for c in range(10):
                ax.text(c, r, f"{100 * m[r, c]:.1f}", ha="center", va="center", fontsize=6.8, color="white" if m[r, c] > 0.6 * m.max() else INK)
        ax.set_xticks([])
        ax.set_yticks(range(4))
        ax.set_yticklabels(["far", "", "", "near"], fontsize=7)
        ax.set_ylabel(s.replace("M2b_", ""), fontsize=8, rotation=0, labelpad=14, va="center")
        if pi == 0:
            ax.set_title(f"I. {lv}: tick-frame token map, mean (% of the tick frame's attention)", color=LV_COL[lv], fontsize=9.5)

fig.text(0.012, 0.972, "v8F-mini hierarchy attention statistics  |  arm M2b, seeds 0 and 1  |  all 2,399 eval windows of 141 episodes  |  T0 open loop  |  MEASURED",
         fontsize=13.5, fontweight="bold", va="center")
fig.text(0.012, 0.948, "Aggregation: mean over 4 heads, 4 queries and the Reader's layers. CI = episode-cluster bootstrap (answers 'another draw of EPISODES' only; the two training seeds are the replicate for 'another training run').  "
         "Attention mass is not an explanation.", fontsize=8.6, color=MUTED, va="center")
fig.savefig(f"{OUT}/fig_attention_stats.png", dpi=100, facecolor=BG)
print("wrote fig_attention_stats.png")
