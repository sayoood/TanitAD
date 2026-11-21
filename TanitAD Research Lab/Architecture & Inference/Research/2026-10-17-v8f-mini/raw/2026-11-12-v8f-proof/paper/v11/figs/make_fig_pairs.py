# make_fig_pairs.py -- figure of the paired differences of the lead rule forms with no label at inference (rows 164 and later): contact against the reader-fed form of the candidate spec,
# contact against the label gap form (with half the gap), and falling behind against the reader-fed form. Point = pooled paired difference, line = its interval
# (two-level bootstrap over model seeds and evaluation episodes, 2,000 draws). No number is typed here: every value is read from the pulled result files of cards ADD102, ADD103 and, when
# present, ADD104 and ADD105. Output: fig_pairs.png and fig_pairs.pdf beside this file.
# Tier of every value: T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s); models UE_s0 and UE_s1; a study of a rule input, not driving performance.
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = [("ADD102", os.path.join(V, "visrule", "pod_add102", "read", "VISRULE10K_RESULT.json")),
       ("ADD103", os.path.join(V, "visrep", "pod_add103", "read", "VISREP10K_RESULT.json")),
       ("ADD104", os.path.join(V, "visseed", "pod_add104", "read", "VISSEED10K_RESULT.json")),
       ("ADD105", os.path.join(V, "visens", "pod_add105", "read", "VISENS10K_RESULT.json"))]
# (arm key, label); an arm is drawn when a result file on disk has its pairs
ARMS = [("VD5c", "Crop gap + slope, first fit (row 164)"),
        ("VD5cx", "Crop gap + slope, second fit (row 165)"),
        ("F2", "Crop gap + slope, new fit 2"),
        ("F3", "Crop gap + slope, new fit 3"),
        ("F4", "Crop gap + slope, new fit 4"),
        ("E5c", "Crop gap + slope, mean of five fits"),
        ("VD1c", "Crop gap + slope, one-frame head"),
        ("VRc", "Reader gap + slope (no new encode)")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = "#2a78d6"


def main():
    pairs, levels, used = {}, {}, []
    for card, p in SRC:
        if not os.path.isfile(p):
            continue
        v = json.load(io.open(p, encoding="utf-8"))["values"]
        used.append(card)
        for k, d in v["pairs"].items():
            pairs.setdefault(k, d)
        for k, d in v["levels"].items():
            levels.setdefault(k, d)
    h = 0.5 * (levels["RR"]["C"] - levels["RLc"]["C"])
    arms = [(k, l) for k, l in ARMS if (k + "-VS-RR") in pairs and (k + "-VS-RLc") in pairs]
    panels = [("-VS-RR", "C", "Contact rate against the spec's form", None),
              ("-VS-RLc", "C", "Contact rate against the label gap form", h),
              ("-VS-RR", "F", "Falling behind against the spec's form", None)]
    n = len(arms)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 0.5 * n + 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(n))[::-1]
    for ax, (suf, c, title, line) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        los, his = [], []
        for y, (k, _l) in zip(ys, arms):
            d = pairs[k + suf][c]
            lo, hi = d["interval"]
            los.append(lo)
            his.append(hi)
            ax.plot([lo, hi], [y, y], color=BLUE, linewidth=2.0, solid_capstyle="round", zorder=2)
            ax.scatter([d["pooled"]], [y], s=46, color=BLUE, edgecolor=SURFACE, linewidth=1.6, zorder=3)
            ax.text(0.5 * (lo + hi), y + 0.2, "%+.4f [%+.4f, %+.4f]" % (d["pooled"], lo, hi), va="bottom", ha="center", color=INK2, fontsize=7.6, zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2})
        span = max(max(his), line or 0.0, 0.0) - min(min(los), 0.0)
        ax.set_xlim(min(min(los), 0.0) - 0.08 * span, max(max(his), line or 0.0, 0.0) + 0.08 * span)
        ax.axvline(0.0, color=INK, linewidth=1.0, zorder=1)
        if line is not None:
            ax.axvline(line, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            ax.text(line, -0.6, " half the gap (%.4f)" % line, color=INK2, fontsize=8, va="bottom", ha="left")
        ax.set_title(title, loc="left", color=INK, fontsize=10, pad=10)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=INK2, length=0, labelsize=8.5)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(-0.7, n - 0.3)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([l for _k, l in arms], color=INK)
    fig.suptitle("Lead rule with no label at inference: paired differences and their intervals", x=0.01, ha="left", color=INK, fontsize=12.5, fontweight="bold", y=0.995)
    fig.text(0.012, 0.012, "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, 650 windows with a lead. Point: pooled paired difference; "
             "line: interval (two-level bootstrap over model seeds and episodes).\nLeft of zero in the first panel: fewer contacts than the reader-fed form of the candidate spec. Second panel: a line that ends at or "
             "left of the dashed line lies within half the gap to the label gap. Falling behind: by more than 10 m. A study of a rule input; no re-perception.",
             ha="left", va="bottom", color=INK2, fontsize=7.4, linespacing=1.4)
    fig.subplots_adjust(left=0.235, right=0.985, top=0.86, bottom=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_pairs." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-PAIRS DONE arms %s cards %s H %.4f" % (",".join(k for k, _l in arms), ",".join(used), h))


if __name__ == "__main__":
    main()
