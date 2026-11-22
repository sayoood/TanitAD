# make_fig_presattr.py -- figure of cards ADD114 (PRESATTR10K: readout K with no pre-roll and with a pre-roll of 30 steps) and ADD115 (PRESATTRC10K: the convention):
# the two-by-two of the two inputs of the lead rule, presence (label or camera reader) and gap (label or camera). Tier T1-kin with the heading bound (kinematic rollout
# on the model's own actions). No number is typed here: every value is read from the pulled result files of the two cards. Colour = where the presence comes from,
# filled or open = where the gap comes from. Output: fig_presattr.png and fig_presattr.pdf beside this file (or in --out-dir).
import argparse
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RES114 = os.path.join(V, "presattr", "pod_add114", "read", "PRESATTR10K_RESULT.json")
RES115 = os.path.join(V, "presattrc", "pod_add115", "read", "PRESATTRC10K_RESULT.json")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREEN = "#2a78d6", "#1baf7a"
SETS = (("CONV", "convention\n(no pre-roll, pinned readout)"), ("K0", "readout K, no pre-roll"), ("K30", "readout K, pre-roll of 30 steps (6 s)"))
# arm, legend text, colour (presence source), filled (gap from the camera), vertical offset
ARMS = (("RLc", "presence label, gap label (reference)", GREEN, False, 0.27), ("PL", "presence label, gap camera", GREEN, True, 0.09),
        ("LA", "presence camera, gap label", BLUE, False, -0.09), ("GEc", "presence camera, gap camera (candidate, no label)", BLUE, True, -0.27))
MEAS = (("U", "unsafe gap"), ("C", "contact"), ("F", "falling behind by more than 10 m"))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, length=0, labelsize=8.6)
    ax.tick_params(axis="y", colors=INK2, length=0, labelsize=9.0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res114", default=RES114)
    ap.add_argument("--res115", default=RES115)
    ap.add_argument("--out-dir", default=HERE)
    a = ap.parse_args()
    st = {}
    for p in (a.res115, a.res114):
        if os.path.isfile(p):
            st.update(json.load(io.open(p, encoding="utf-8"))["values"]["settings"])
    sets = [s for s in SETS if s[0] in st]
    if not sets:
        raise SystemExit("ZZ FIG-PRESATTR STOP: no result file")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": GRID, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 1.6 + 1.15 * len(sets)), sharey=True, gridspec_kw={"wspace": 0.10})
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(sets) - 1, -1, -1))
    for ax, (m, mname) in zip(axes, MEAS):
        style(ax)
        xmax = 0.0
        for y, (S, _) in zip(ys, sets):
            for arm, _, col, filled, dy in ARMS:
                x = st[S]["levels"][arm][m]
                seeds = (st[S].get("levels_per_seed") or {}).get(arm, {}).get(m) or []
                if seeds:
                    ax.plot([min(seeds), max(seeds)], [y + dy, y + dy], color=col, linewidth=2.0, solid_capstyle="round", zorder=3)
                ax.plot([x], [y + dy], marker="o", markersize=8.0, color=col, markerfacecolor=(col if filled else SURFACE), markeredgecolor=(SURFACE if filled else col),
                        markeredgewidth=1.6, zorder=4)
                xmax = max([xmax, x] + list(seeds))
                ax.text(max([x] + list(seeds)), y + dy, "   %.4f" % x, color=INK, fontsize=8.0, ha="left", va="center")
        ax.set_xlim(-0.04 * xmax - 1e-4, xmax * 1.30 + 1e-3)
        ax.set_ylim(-0.6, len(sets) - 0.4)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_title(mname, color=INK, fontsize=10.2, loc="left", pad=10)
    labels = []
    for S, lab in sets:
        n = (st[S].get("counts") or {}).get("n_lead_windows")
        labels.append(lab + ("\n%s windows with a lead" % n if n is not None else ""))
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, color=INK)
    top = 1.0 - 0.20 / fig.get_figheight()
    for (arm, lab, col, filled, _), x0 in zip(ARMS, (0.045, 0.265, 0.435, 0.605)):
        fig.lines.append(plt.Line2D([x0], [top], marker="o", markersize=7.5, color=col, markerfacecolor=(col if filled else SURFACE), markeredgecolor=(SURFACE if filled else col),
                                    markeredgewidth=1.5, linestyle="none", transform=fig.transFigure))
        fig.text(x0 + 0.010, top, lab, color=INK2, fontsize=8.6, ha="left", va="center")
    fig.text(0.63, 0.50 / fig.get_figheight(), "share of the windows with a lead (unsafe gap: of its own windows)", color=INK2, fontsize=8.8, ha="center", va="bottom")
    fig.text(0.008, 0.012, "Dot: both model seeds pooled; bar: the two model seeds. T1-kin with the heading bound, two 10k models; the lead moves as logged.\n"
             "Cells with a label input are attribution arms, never candidates.", color=INK2, fontsize=8.0, ha="left", va="bottom")
    fig.subplots_adjust(left=0.215, right=0.985, top=1.0 - 0.72 / fig.get_figheight(), bottom=0.95 / fig.get_figheight())
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out_dir, "fig_presattr." + ext), dpi=170, facecolor=SURFACE)
    print("ZZ FIG-PRESATTR DONE settings %s" % ",".join(s[0] for s in sets))


if __name__ == "__main__":
    main()
