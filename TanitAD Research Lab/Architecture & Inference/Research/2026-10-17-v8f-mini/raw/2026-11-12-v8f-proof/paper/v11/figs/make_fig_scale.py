# make_fig_scale.py -- two figures of row 172 (card ADD106, CROPSCALE10K): what more fit clips give the camera head for the gap of the lead.
# fig_scale_t0: tier T0 (recorded frames, no rollout; a property of the head's gap output, not driving performance): the gap error of every fit by fit size, and the error by bin of the label gap
#   for fit seed 2 at both sizes.
# fig_scale_pairs: tier T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1: paired differences of the lead rule fed by each fit,
#   against the reader-fed form and against the label gap form. Point = pooled paired difference, line = its interval (two-level bootstrap over model seeds and evaluation episodes, 2,000 draws).
# No number is typed here: every value is read from the pulled result files of cards ADD104, ADD105 and ADD106 and from the fetched output of falltrace/t0_gap_bins_scale.py.
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
R106 = os.path.join(V, "cropscale", "pod_add106", "read", "CROPSCALE10K_RESULT.json")
R104 = os.path.join(V, "visseed", "pod_add104", "read", "VISSEED10K_RESULT.json")
R105 = os.path.join(V, "visens", "pod_add105", "read", "VISENS10K_RESULT.json")
BINS = os.path.join(V, "falltrace", "pod_t0", "t0_gap_bins_scale.out")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SMALL = "fit on 450 clips"
LARGE = "fit on 1,629 clips"


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=8.5)
    ax.tick_params(axis="y", colors=INK2, length=0, labelsize=8.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def legend(fig, y):
    hs = [plt.Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor=c, markeredgecolor=SURFACE) for c in (BLUE, ORANGE)]
    fig.legend(hs, [SMALL, LARGE], loc="upper right", bbox_to_anchor=(0.985, y), ncol=2, frameon=False, fontsize=9, labelcolor=INK, handletextpad=0.3, columnspacing=1.6)


def fig_t0(v):
    t0 = v["t0"]
    err = t0["errors_m"]
    small = [k for k in ("O0", "O1", "M2", "M3", "M4") if k in err]
    large = [k for k in ("L2", "L3", "L4") if k in err]
    bins = None
    for line in io.open(BINS, encoding="utf-8"):
        if line.startswith("ZZJSON "):
            bins = json.loads(line[7:])["bins"]
    if bins is None:
        raise SystemExit("no ZZJSON line in " + BINS)
    keys = [k[3:] for k in bins if k.startswith("M2_")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.9), gridspec_kw={"wspace": 0.2, "width_ratios": [1.0, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    style(a)
    for y, (names, col) in zip((1, 0), ((small, BLUE), (large, ORANGE))):
        xs = sorted(err[k] for k in names)
        a.scatter(xs, [y] * len(xs), s=95, color=col, edgecolor=SURFACE, linewidth=2.0, zorder=3)
        a.text(sum(xs) / len(xs), y + 0.2, "%s fits: %s m" % (len(xs), ", ".join("%.2f" % x for x in xs)), ha="center", va="bottom", color=INK2, fontsize=8.4)
    a.set_yticks([1, 0])
    a.set_yticklabels(["450 clips", "1,629 clips"], color=INK, fontsize=9.5)
    a.set_ylim(-0.6, 1.7)
    lo = min(err.values())
    hi = max(err.values())
    a.set_xlim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
    a.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    a.set_xlabel("mean absolute gap error on %s frames with a lead (m)" % format(t0["n_rows"], ","), color=INK2, fontsize=9)
    a.set_title("Every fit of the head: error of the gap reading", loc="left", color=INK, fontsize=10.5, pad=10)
    a.text(0.98, 0.06, "ratio of the mean errors, large to small:\n%.3f, interval [%.3f, %.3f]" % (t0["ratio"], t0["interval"][0], t0["interval"][1]), transform=a.transAxes, color=INK2, fontsize=8.6,
           ha="right", va="bottom", linespacing=1.4)
    style(b)
    w = 0.36
    for i, k in enumerate(keys):
        for off, name, col in ((-w / 2 - 0.01, "M2", BLUE), (w / 2 + 0.01, "L2", ORANGE)):
            d = bins[name + "_" + k]
            b.bar(i + off, d["mae_m"], width=w, color=col, zorder=2)
            b.text(i + off, d["mae_m"] + 0.12, "%.2f" % d["mae_m"], ha="center", va="bottom", color=INK2, fontsize=8.2)
    b.set_xticks(range(len(keys)))
    b.set_xticklabels(["%s m\n%+.2f / %+.2f m" % (k.replace("_", " to "), bins["M2_" + k]["signed_m"], bins["L2_" + k]["signed_m"]) for k in keys], color=INK, fontsize=8.8, linespacing=1.5)
    b.tick_params(axis="x", pad=6)
    b.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    b.axhline(0.0, color=INK, linewidth=0.8, zorder=1)
    b.set_ylim(0, max(bins["M2_" + k]["mae_m"] for k in keys) * 1.16)
    b.set_ylabel("mean absolute gap error (m)", color=INK2, fontsize=9)
    b.set_title("By distance of the lead in the label (fit seed 2 at both sizes)", loc="left", color=INK, fontsize=10.5, pad=10)
    fig.suptitle("More fit clips: the camera head reads the gap to the lead better", x=0.012, ha="left", color=INK, fontsize=12.5, fontweight="bold", y=0.985)
    legend(fig, 0.995)
    fig.text(0.012, 0.012, "Tier T0 (recorded frames, no rollout): a property of the head's gap output on 141 evaluation clips, not driving performance. Interval of the ratio: cluster bootstrap over the clips, 2,000 draws.\n"
             "Second line under each bin: signed error, head minus label, for 450 clips / for 1,629 clips; a negative value is a lead read nearer than it is. Far leads stay read short at both sizes.",
             ha="left", va="bottom", color=INK2, fontsize=7.6, linespacing=1.4)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.84, bottom=0.24)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_scale_t0." + ext), dpi=170, facecolor=SURFACE)
    return len(small), len(large), keys


def fig_pairs(v):
    pairs, levels = {}, {}
    for p in (R106, R104, R105):
        if not os.path.isfile(p):
            continue
        x = json.load(io.open(p, encoding="utf-8"))["values"]
        for k, d in x["pairs"].items():
            pairs.setdefault(k, d)
        for k, d in x["levels"].items():
            levels.setdefault(k, d)
    h = 0.5 * (levels["RR"]["C"] - levels["RLc"]["C"])
    want = [("F2", "450 clips, fit seed 2", BLUE), ("F3", "450 clips, fit seed 3", BLUE), ("F4", "450 clips, fit seed 4", BLUE), ("G2", "1,629 clips, fit seed 2", ORANGE),
            ("G3", "1,629 clips, fit seed 3", ORANGE), ("G4", "1,629 clips, fit seed 4", ORANGE), ("GEc", "1,629 clips, mean of the three", ORANGE)]
    arms = [a for a in want if (a[0] + "-VS-RR") in pairs and (a[0] + "-VS-RLc") in pairs]
    panels = [("-VS-RR", "C", "Contact rate against the reader-fed form", None),
              ("-VS-RLc", "C", "Contact rate against the label gap form", h),
              ("-VS-RLc", "F", "Falling behind against the label gap form", None)]
    n = len(arms)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 0.52 * n + 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(n))[::-1]
    for ax, (suf, c, title, line) in zip(axes, panels):
        style(ax)
        los, his = [], []
        for y, (k, _l, col) in zip(ys, arms):
            d = pairs[k + suf][c]
            lo, hi = d["interval"]
            los.append(lo)
            his.append(hi)
            ax.plot([lo, hi], [y, y], color=col, linewidth=2.0, solid_capstyle="round", zorder=2)
            ax.scatter([d["pooled"]], [y], s=46, color=col, edgecolor=SURFACE, linewidth=1.6, zorder=3)
            ax.text(0.5 * (lo + hi), y + 0.2, "%+.4f [%+.4f, %+.4f]" % (d["pooled"], lo, hi), va="bottom", ha="center", color=INK2, fontsize=7.6, zorder=5,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2})
        top = max(max(his), line or 0.0, 0.0)
        bot = min(min(los), 0.0)
        ax.set_xlim(bot - 0.1 * (top - bot), top + 0.1 * (top - bot))
        ax.axvline(0.0, color=INK, linewidth=1.0, zorder=1)
        if line is not None:
            ax.axvline(line, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            ax.text(line, -0.62, "half the gap (%.4f) " % line, color=INK2, fontsize=8, va="bottom", ha="right")
        ax.set_title(title, loc="left", color=INK, fontsize=10, pad=10)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_ylim(-0.7, n - 0.3)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([l for _k, l, _c in arms], color=INK)
    fig.suptitle("Lead rule fed by the camera head, by fit size: paired differences and their intervals", x=0.012, ha="left", color=INK, fontsize=12.5, fontweight="bold", y=0.985)
    legend(fig, 0.995)
    fig.text(0.012, 0.012, "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s), models UE_s0 and UE_s1, 650 windows with a lead. Point: pooled paired difference; "
             "line: interval (two-level bootstrap over model seeds and episodes).\nFirst panel, left of zero: fewer contacts than the reader-fed form. Second panel: a line that ends left of the dashed line lies "
             "within half the gap to the label gap form. Third panel, right of zero: falls behind\nby more than 10 m more often than the label gap form. A study of a rule input; no re-perception.",
             ha="left", va="bottom", color=INK2, fontsize=7.4, linespacing=1.4)
    fig.subplots_adjust(left=0.165, right=0.985, top=0.855, bottom=0.175)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_scale_pairs." + ext), dpi=170, facecolor=SURFACE)
    return [a[0] for a in arms], h


def main():
    v = json.load(io.open(R106, encoding="utf-8"))["values"]
    ns, nl, keys = fig_t0(v)
    arms, h = fig_pairs(v)
    print("ZZ FIG-SCALE DONE t0 fits %d+%d bins %s; pairs arms %s H %.4f" % (ns, nl, ",".join(keys), ",".join(arms), h))


if __name__ == "__main__":
    main()
