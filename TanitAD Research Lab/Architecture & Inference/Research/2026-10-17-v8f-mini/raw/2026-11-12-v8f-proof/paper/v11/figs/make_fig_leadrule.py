# make_fig_leadrule.py -- figure of the lead rule study (rows 158 to 165): contact rate, unsafe rate and falling behind per form of the rule, from the landed result files.
# No number is typed here: every value is read from the pulled result records of cards ADD101, ADD102 and (when present) ADD103. Output: fig_leadrule.png and fig_leadrule.pdf beside this file.
# Tier of every value: T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s); models UE_s0 and UE_s1; a study of a rule input, not driving performance.
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
F101 = os.path.join(V, "croprule", "pod_add101", "read", "CROPRULE10K_RESULT.json")
F102 = os.path.join(V, "visrule", "pod_add102", "read", "VISRULE10K_RESULT.json")
F103 = os.path.join(V, "visrep", "pod_add103", "read", "VISREP10K_RESULT.json")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 of the validated default palette: no label at inference / label at inference


def jl(p):
    return json.load(io.open(p, encoding="utf-8"))


def main():
    v101, v102 = jl(F101)["values"], jl(F102)["values"]
    lv, ls = dict(v101["levels"]), dict(v101["levels_per_seed"])
    for k in ("VD5", "VD5c"):
        lv[k], ls[k] = v102["levels"][k], v102["levels_per_seed"][k]
    have103 = os.path.isfile(F103)
    if have103:
        v103 = jl(F103)["values"]
        for k in ("VD5cx", "VRc", "VD1c"):
            lv[k], ls[k] = v103["levels"][k], v103["levels_per_seed"][k]
    # (key, label, group): group 0 = no label at inference, 1 = the label enters at inference (bounds and references)
    forms = [("RR", "Reader presence, reader gap (candidate spec)", 0),
             ("VD5", "Reader presence, crop gap", 0)]
    if have103:
        forms += [("VRc", "Reader presence, reader gap + slope", 0), ("VD1c", "Reader presence, crop gap + slope, one-frame head", 0),
                  ("VD5cx", "Reader presence, crop gap + slope, other fit seed", 0)]
    forms += [("VD5c", "Reader presence, crop gap + slope (row 164)", 0),
              ("K100", "Label marks the lead, reader gap + slope", 1),
              ("KD5", "Label marks the lead, crop gap + slope", 1),
              ("KB5", "Label marks the lead, label-centred crop + slope", 1),
              ("RLc", "Label gap + slope", 1)]
    for k, _l, _g in forms:
        if k not in lv:
            sys.exit("no level for " + k)
    h = 0.5 * (lv["RR"]["C"] - lv["RLc"]["C"])
    panels = [("C", "Contact rate"), ("U", "Unsafe rate"), ("F", "Falling behind by more than 10 m")]
    n = len(forms)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 0.46 * n + 2.3), sharey=True, gridspec_kw={"width_ratios": [1.25, 1.0, 1.0], "wspace": 0.2})
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(n))[::-1]
    for ax, (c, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        vals = [lv[k][c] for k, _l, _g in forms]
        cols = [BLUE if g == 0 else ORANGE for _k, _l, g in forms]
        ax.barh(ys, vals, height=0.52, color=cols, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        for y, (k, _l, _g) in zip(ys, forms):
            per = ls[k][c]
            ax.scatter(per, [y] * len(per), s=26, facecolor=SURFACE, edgecolor=INK, linewidth=1.0, zorder=3)
        xmax = max(max(vals), max(max(ls[k][c]) for k, _l, _g in forms))
        ax.set_xlim(0, xmax * 1.22)
        for y, v in zip(ys, vals):
            ax.text(max(v, max(ls[forms[n - 1 - y][0]][c])) + xmax * 0.025, y, "%.4f" % v, va="center", ha="left", color=INK, fontsize=9)
        if c == "C":
            ax.axvline(h, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            ax.text(h, n - 0.42, " half the gap (%.4f)" % h, color=INK2, fontsize=8.5, va="bottom", ha="left")
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, pad=14)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=INK2, length=0, labelsize=8.5)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(-0.6, n - 0.2)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([l for _k, l, _g in forms], color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE),
               plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK, markersize=5.5)]
    fig.legend(handles, ["no label at inference", "the label enters at inference (bound or reference)", "one model seed"], loc="lower left", bbox_to_anchor=(0.01, 0.045), ncol=3,
               frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.2, columnspacing=1.8)
    fig.suptitle("Lead rule: what the rule is fed, and what it costs", x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.text(0.012, 0.012, "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, 650 windows with a lead.\n"
             "A study of a rule input; no re-perception. Half the gap: half of the contact rate of the candidate spec's form minus that of the label gap. Crop gap: the fixed-window head on 500 train clips.",
             ha="left", va="bottom", color=INK2, fontsize=7.4, linespacing=1.4)
    fig.subplots_adjust(left=0.30, right=0.985, top=0.86, bottom=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_leadrule." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-LEADRULE DONE forms %d add103 %s H %.4f" % (n, have103, h))


if __name__ == "__main__":
    main()
