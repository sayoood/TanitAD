# make_fig_gaptrace.py -- figure: the gap to the lead over time on evaluation clips, as the label has it, as the present reader reads it and as the fixed-window head of card ADD99
# (D-H5, fit seed 0) reads it. Tier T0 (a probe on logged frames; a diagnostic of a reader). Input: the two per-frame dumps pulled from the fork pod into figdata/ (derived values, no pixels).
# Clips are not hand-picked: the first N_CLIPS evaluation clips in the order of the sha256 hex of the clip id among those with at least MIN_LEAD frames with a label lead.
import hashlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
N_CLIPS, MIN_LEAD = 6, 60


def main():
    zh = np.load(os.path.join(HERE, "figdata", "head_D_H5_s0.npz"), allow_pickle=False)
    zp = np.load(os.path.join(HERE, "figdata", "present_s0.npz"), allow_pickle=False)
    clips = [str(c) for c in zh["clips"]]
    assert clips == [str(c) for c in zp["clips"]] and np.array_equal(zh["cid"], zp["cid"]) and np.array_equal(zh["frame"], zp["frame"])
    dt = float(zh["dt_s"])
    cid, fr = np.asarray(zh["cid"]), np.asarray(zh["frame"])
    lab, pres_l = np.asarray(zh["lab_gap"], float), np.asarray(zh["lab_present"]).astype(bool)
    gh, gp = np.asarray(zh["gap"], float), np.asarray(zp["gap"], float)
    order = sorted(range(len(clips)), key=lambda i: hashlib.sha256(clips[i].encode("utf-8")).hexdigest())
    pick = [i for i in order if int((pres_l & (cid == i)).sum()) >= MIN_LEAD][:N_CLIPS]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.4), sharex=True, sharey=True, gridspec_kw={"wspace": 0.06, "hspace": 0.22})
    fig.patch.set_facecolor(SURFACE)
    sd_h, sd_p = [], []
    for ax, i in zip(axes.ravel(), pick):
        m = cid == i
        t = fr[m] * dt
        o = np.argsort(t)
        t, L, H, P, ok = t[o], lab[m][o], gh[m][o], gp[m][o], pres_l[m][o]
        Lm = np.where(ok, L, np.nan)
        Hm, Pm = np.where(ok, H, np.nan), np.where(ok, P, np.nan)
        ax.set_facecolor(SURFACE)
        ax.plot(t, Pm, color=ORANGE, linewidth=1.6, zorder=2)
        ax.plot(t, Hm, color=BLUE, linewidth=2.0, zorder=3)
        ax.plot(t, Lm, color=INK, linewidth=1.4, linestyle=(0, (5, 2)), zorder=4)
        both = ok & np.isfinite(H) & np.isfinite(P)
        if both.any():
            sd_h.append(float(np.std(H[both] - L[both])))
            sd_p.append(float(np.std(P[both] - L[both])))
        ax.set_title("clip %s" % clips[i][:8], loc="left", color=INK2, fontsize=8.5, pad=4)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, length=0, labelsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylim(0, 62)
    for ax in axes[1]:
        ax.set_xlabel("time in the clip (s)", color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("gap to the lead (m)", color=INK2)
    handles = [plt.Line2D([0], [0], color=INK, linewidth=1.4, linestyle=(0, (5, 2))), plt.Line2D([0], [0], color=BLUE, linewidth=2.0), plt.Line2D([0], [0], color=ORANGE, linewidth=1.6)]
    fig.legend(handles, ["label", "fixed-window head (new read, image only)", "present reader"], loc="lower left", bbox_to_anchor=(0.05, 0.035), ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2, handlelength=2.4, columnspacing=2.0)
    fig.suptitle("The gap to the lead as read from the camera: new fixed-window read against the present reader", x=0.05, ha="left", color=INK, fontsize=12.5, fontweight="bold", y=0.985)
    fig.text(0.05, 0.008, "T0 (a probe on logged frames; a diagnostic of a reader, not driving performance). Evaluation clips, frames with a lead in the label; fit seed 0 of the head, fit on 500 other clips.\n"
             "Clips are not hand-picked: the first %d in the order of the sha256 of the clip id with at least %d lead frames. A jump of the label is a change of the lead in the label." % (N_CLIPS, MIN_LEAD),
             ha="left", va="bottom", color=INK2, fontsize=7.4, linespacing=1.4)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.9, bottom=0.15)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(HERE, "fig_gaptrace." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-GAPTRACE DONE clips %s | error sd per clip head %s present %s" % (",".join(clips[i][:8] for i in pick), ["%.1f" % x for x in sd_h], ["%.1f" % x for x in sd_p]))


if __name__ == "__main__":
    main()
