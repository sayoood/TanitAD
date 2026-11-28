"""fig_listen_grid.py -- the PREREG_V8F_LISTEN figure in its committed form, drawn from READ.json only.

Rows S, T-lat, T-lon, O; columns scene, ego, nav, limit, S checkpoints, T code, tick ages. Colour = n(L,X) = delta(L,X) / delta(L,all), clipped to
[0, 1] (one sequential hue, light = ignores, dark = listens; monotone in grey-scale, so it survives print). Cell text = the raw delta with its
unit; '*' = the cell LISTENS (delta > 0, separated at BOTH seeds, |delta| above the donor-seed difference); a heavy frame = the row's largest n.
Hatched grey = not an input of that layer. One panel per training seed; the four Q-NAV bars (ADE6 change, donor nav injected at S / T / O / all
three) sit beside, both seeds, with their 95 % paired episode-cluster intervals.

    python fig_listen_grid.py [--read READ.json] [--out fig_listen_grid.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROWS = ("S", "T-lat", "T-lon", "O")
COLS = ("scene", "ego", "nav", "limit", "ck", "code", "age")
COL_LABEL = {"scene": "scene", "ego": "ego", "nav": "nav", "limit": "limit", "ck": "S check-\npoints", "code": "T code", "age": "tick ages"}
ROW_LABEL = {"S": "S\ncheckpoint err @ 50 m\n(m)", "T-lat": "T-lat\nlateral macro-F1\n(F1 drop)", "T-lon": "T-lon\nlongitudinal macro-F1\n(F1 drop)", "O": "O\nADE6\n(m)"}
UNIT = {"S": "m", "T-lat": "F1", "T-lon": "F1", "O": "m"}
QNAV_LABEL = {"nav@S": "S only", "nav@T": "T only", "nav@O": "O only", "nav@STO": "S+T+O"}
INK, INK2, GRID = "#1b1b1b", "#555555", "#9a9a9a"
CMAP = plt.get_cmap("Blues")
SEED_FILL = {"s0": "#08519c", "s1": "#9ecae1"}


def fmt(row, d):
    return f"{d:+.2f}" if UNIT[row] == "m" else f"{d:+.3f}"


def heat(ax, rec, listens, title, show_rows=True):
    ax.set_xlim(0, len(COLS))
    ax.set_ylim(len(ROWS), 0)
    for i, row in enumerate(ROWS):
        ent = rec["rows"][row]
        allc = ent["cells"]["all"]
        n_ok = bool(ent.get("all_defined"))
        best = max((c["n"], j) for j, col in enumerate(COLS) if (c := ent["cells"][col]) is not None and c.get("n") is not None)[1] if n_ok else None
        for j, col in enumerate(COLS):
            c = ent["cells"][col]
            if c is None:                                                  # not an input of this layer
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor="#e4e4e4", edgecolor=GRID, hatch="////", lw=0.6))
                continue
            nc = c.get("n_clipped")
            face = CMAP(0.04 + 0.96 * nc) if nc is not None else "white"
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", lw=1.2))
            dark = nc is not None and nc > 0.55
            tc = "white" if dark else INK
            star = "*" if listens[row][col] else ""
            ax.text(j + 0.5, i + 0.40, f"{fmt(row, c['delta'])}{star}", ha="center", va="center", fontsize=8.7, color=tc,
                    fontweight="bold" if listens[row][col] else "normal")
            ax.text(j + 0.5, i + 0.66, f"[{c['lo']:.2f}, {c['hi']:.2f}]", ha="center", va="center", fontsize=6.4, color=tc)
            ax.text(j + 0.5, i + 0.86, f"n = {c['n']:.2f}" if c.get("n") is not None else "n = NA", ha="center", va="center", fontsize=6.8,
                    color=tc)
        if best is not None:
            ax.add_patch(Rectangle((best + 0.03, i + 0.03), 0.94, 0.94, facecolor="none", edgecolor=INK, lw=2.4))
        ax.text(len(COLS) + 0.08, i + 0.5, f"all: {fmt(row, allc['delta'])}" + ("" if n_ok else "\n(n undefined)"), ha="left", va="center",
                fontsize=7.6, color=INK2)
    ax.set_xticks([j + 0.5 for j in range(len(COLS))])
    ax.set_xticklabels([COL_LABEL[c] for c in COLS], fontsize=8.5, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(ROWS))])
    ax.set_yticklabels([ROW_LABEL[r] for r in ROWS] if show_rows else [""] * len(ROWS), fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=10.5, color=INK, pad=34, loc="left")


def qnav(ax, grid):
    arms = ("nav@S", "nav@T", "nav@O", "nav@STO")
    w = 0.36
    lo = min(min(grid[s]["qnav"][a]["lo"] for a in arms) for s in ("s0", "s1"))
    hi = max(max(grid[s]["qnav"][a]["hi"] for a in arms) for s in ("s0", "s1"))
    for k, (s, lab) in enumerate((("s0", "seed 0"), ("s1", "seed 1"))):
        q = grid[s]["qnav"]
        xs = [i + (k - 0.5) * w for i in range(len(arms))]
        ys = [q[a]["delta"] for a in arms]
        err = [[max(0.0, q[a]["delta"] - q[a]["lo"]) for a in arms], [max(0.0, q[a]["hi"] - q[a]["delta"]) for a in arms]]
        ax.bar(xs, ys, width=w * 0.92, color=SEED_FILL[s], edgecolor=INK if s == "s1" else SEED_FILL[s], lw=0.6, label=lab, zorder=2)
        ax.errorbar(xs, ys, yerr=err, fmt="none", ecolor=INK, elinewidth=1.0, capsize=2.5, zorder=3)
        for x, y, a in zip(xs, ys, arms):
            ax.text(x, max(y, q[a]["hi"]) + 0.015 * (hi - lo + 1e-9), f"{y:+.2f}", ha="center", va="bottom", fontsize=7.2, color=INK, rotation=90)
    pad = 0.12 * (max(hi, 0.0) - min(lo, 0.0) + 1e-9)
    ax.set_ylim(min(lo, 0.0) - pad, max(hi, 0.0) + 5.5 * pad)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([QNAV_LABEL[a] for a in arms], fontsize=8.5, color=INK)
    ax.set_ylabel("ADE6 change vs clean (m)", fontsize=8.5, color=INK)
    ax.set_xlabel("donor NAV injected at", fontsize=8.5, color=INK)
    ax.tick_params(labelsize=8, colors=INK2)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    d = {s: grid[s]["qnav"]["D_T_minus_O"] for s in ("s0", "s1")}
    ax.set_title("Q-NAV: does nav reach the trajectory\nthrough T, or around it?", fontsize=10.5, color=INK, loc="left", pad=8)
    ax.text(0.0, -0.17, "D = ADE6(nav at T only)\n    - ADE6(nav at O only), paired:\n" + "\n".join(
        f"  seed {s[1]}: {d[s]['delta']:+.2f} m [{d[s]['lo']:+.2f}, {d[s]['hi']:+.2f}]" + (", separated" if d[s]["separated"] else ", not separated")
        for s in ("s0", "s1")), transform=ax.transAxes, fontsize=7.6, color=INK, va="top")
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def main(argv=None):
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--read", default=str(here / "READ.json"))
    ap.add_argument("--out", default=str(here / "fig_listen_grid.png"))
    a = ap.parse_args(argv)
    R = json.load(open(a.read, encoding="utf-8"))
    grid, listens = R["grid"], R["analysis"]["listens"]
    fig = plt.figure(figsize=(17.5, 6.4), dpi=200)
    gs = fig.add_gridspec(1, 5, width_ratios=[7, 7, 0.22, 0.9, 3.9], wspace=0.32, left=0.12, right=0.985, top=0.80, bottom=0.27)
    for k, s in enumerate(("s0", "s1")):
        ax = fig.add_subplot(gs[0, k])
        heat(ax, grid[s], listens, f"M2b seed {s[1]}", show_rows=(k == 0))
    cax = fig.add_subplot(gs[0, 2])
    sm = matplotlib.cm.ScalarMappable(cmap=CMAP, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("n = delta(L,X) / delta(L,all), clipped to [0, 1]", fontsize=8.5, color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK2)
    qnav(fig.add_subplot(gs[0, 4]), grid)
    o, q = R["analysis"]["outcome"]["headline"], R["analysis"]["qnav"]["headline"]
    fig.suptitle(f"Who listens to what: one input of ONE layer swapped for a donor window's (other episode)   |   outcome: {o}   |   Q-NAV: {q}",
                 x=0.02, y=0.975, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.02, 0.115,
             "Cell text = raw delta (S, O: metres; T: macro-F1 drop), its 95 % paired episode-cluster interval, then n.   * = listens (delta > 0, separated at BOTH seeds,\n"
             "|delta| above the donor-seed difference).   Heavy frame = the row's largest n; * is a sign-and-separation test with no size threshold, read size from delta and n.   Hatched grey = not an input of that layer.   'all' = every input of the layer\n"
             "from the donor (O's kinematic prior excepted).   Zero training, CPU, T1-class open loop, DIAGNOSTIC.   Estimator: paired episode-cluster bootstrap, B = 2,000,\n"
             "141 episodes; S and T rows 136-139 with labels (answers 'another draw of EPISODES'; the two training seeds are shown side by side for run-to-run variance).   PREREG_V8F_LISTEN, tip 3fdb1b61.",
             fontsize=7.6, color=INK2, ha="left", va="top")
    fig.savefig(a.out, dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
