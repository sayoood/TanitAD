"""v3_draw.py -- drawing helpers of the v3 video (copied from viz2/v2_draw.py; HACC colour / label, header text and the drawn-text log added) (matplotlib / Agg, CPU).  Style reused from v1 (`viz/out/viz_lib.py`): BEV forward-up with plot x = -y_left,
range rings, ego triangle, sample-and-hold diamonds for S checkpoints, Okabe-Ito colours."""
from __future__ import annotations

import os
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

sys.path.insert(0, "C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/viz/out")
from viz_lib import axpx, INK, MUTED, BG, PANEL, LV_COL  # noqa: E402,F401

W, H = 1920, 1080
CMD_COL = {"FOLLOW": "#009E73", "TURN_L": "#0072B2", "TURN_R": "#D55E00"}
CMD_LAB = {"FOLLOW": "FOLLOW", "TURN_L": "TURN LEFT", "TURN_R": "TURN RIGHT"}
PLAN_COL = {"GT": "#1b1f24", "P10": "#8a8f98", "B": "#E69F00", "H": "#0072B2", "HS": "#009E73", "HACC": "#D55E00"}
PLAN_LAB = {"GT": "GT: what the human drove", "P10": "P10: kinematic prior (no scene)", "B": "B: M2b's own plan", "H": "H: B shrunk toward P10 (HBLEND)",
            "HS": "HS: per-mode shrink + smoothness (HSPEC stage 2)", "HACC": "HACC: P + w (tau - P), accel-fitted weights (NEW)"}
DARK = "#10141a"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c9ced6", "text.color": INK})


def new_fig(bg=BG):
    return plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=bg)


def tx(fig, x, y, s, fs=12, color=INK, weight="normal", ha="left", va="top", mono=False, ls=1.25, alpha=1.0, **kw):
    """text at PIXEL position (x, y from the top-left)."""
    return fig.text(x / W, 1 - y / H, s, fontsize=fs, color=color, fontweight=weight, ha=ha, va=va, linespacing=ls, alpha=alpha,
                    family="DejaVu Sans Mono" if mono else "DejaVu Sans", **kw)


def wrap(s, n):
    return "\n".join(textwrap.fill(p, n) for p in s.split("\n"))


def panel_box(fig, x, y, w, h, ec="#c9ced6", lw=1.2, fc=PANEL):
    ax = axpx(fig, x, y, w, h, W, H)
    ax.set_facecolor(fc)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(ec)
        s.set_linewidth(lw)
    return ax


def header(fig, seg, tier="T1-class open loop", sub=None):
    tx(fig, 14, 6, "TanitAD v8F-mini  |  evidence video v5 (HACC + T1-kin + DAgger)  |  numbers read from banked artifacts, paths in VIDEO_NOTES_V5.md  |  CPU, dev box", 10, MUTED)
    tx(fig, W - 14, 6, tier, 10.5, "#b30000", "bold", ha="right")
    tx(fig, 14, 22, seg, 17.5, INK, "bold")
    if sub:
        tx(fig, 14, 47, sub, 10.5, MUTED)


def caption(fig, s, size=17, y0=942, h=100, n=118, sub=None):
    panel_box(fig, 10, y0, W - 20, h, ec="#9aa3af", lw=1.4)
    tx(fig, 26, y0 + 10, wrap(s, n), size, INK, ls=1.3)
    if sub:
        tx(fig, 26, y0 + h - 22, sub, 10, MUTED)


def footer(fig, s):
    tx(fig, 14, 1050, wrap(s, 262), 8.6, MUTED, ls=1.2)


def chips(fig, x, y, injected, labels=("S", "T", "O"), fs=13):
    tx(fig, x, y + 4, "nav command injected at:", 11, MUTED)
    for i, lv in enumerate(labels):
        on = lv in injected
        fig.text((x + 222 + i * 44) / W, 1 - (y + 14) / H, lv, fontsize=fs, fontweight="bold", color="white" if on else "#9aa3af", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.32", fc=DARK if on else "white", ec=DARK if on else "#c9ced6", lw=1.2))


# ------------------------------------------------------------------------------------------------ BEV
def fit_extent(arrs, extra_lat=0.0):
    """arrs: list of [n, >=2] (forward, left) arrays -> (ymax, lat)."""
    a = np.concatenate([np.asarray(x, float)[:, :2] for x in arrs if x is not None])
    a = a[np.isfinite(a).all(1)]
    return float(a[:, 0].max()) * 1.07 + 4.0, float(np.abs(a[:, 1]).max()) * 1.12 + 2.5 + extra_lat


def bev_axes(fig, box, ext, title=None, fs_tick=7.5):
    x, y, w, h = box
    ax = axpx(fig, x, y, w, h, W, H)
    ymax, lat = ext
    rng = max(ymax + 6.0, 2.0 * lat * h / w)
    ax.set_facecolor(PANEL)
    ax.set_ylim(-6.0, -6.0 + rng)
    ax.set_xlim(-rng * w / h / 2, rng * w / h / 2)
    ax.set_aspect("equal", adjustable="box")
    for rad in (25, 50, 75, 100, 150):
        if rad < rng - 6:
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, ec="#d5d9e0", lw=0.7, ls=":"))
            ax.text(0, rad, f"{rad} m", fontsize=fs_tick, color=MUTED, ha="center", va="bottom")
    ax.tick_params(labelsize=fs_tick, colors=MUTED, length=2)
    ax.set_xlabel("lateral (m):  left <-    -> right", fontsize=8.5, color=MUTED, labelpad=1)
    ax.set_ylabel("forward (m), ego frame at NOW", fontsize=8.5, color=MUTED, labelpad=1)
    if title:
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=4)
    ax.add_patch(Polygon([[0, 2.2], [-1.1, -1.8], [1.1, -1.8]], closed=True, fc=INK, ec="white", zorder=8))
    return ax


def plan(ax, xy, p=1.0, color=INK, lw=2.6, ls="-", label=None, z=4, dots=True, end=True, ms=4.5, alpha=1.0):
    """xy [30, 2] (forward, left) at 0.2 s knots; draws the first fraction p; dots every 1 s; a big dot at the last drawn point."""
    xy = np.asarray(xy, float)
    n = int(np.clip(round(p * 29) + 1, 2, 30))
    X, Y = -xy[:n, 1], xy[:n, 0]
    ax.plot(X, Y, ls, color=color, lw=lw, label=label, zorder=z, alpha=alpha, solid_capstyle="round")
    if dots:
        k = [i for i in range(4, n, 5)]
        ax.plot(X[k], Y[k], "o", color=color, ms=ms, zorder=z, alpha=alpha)
    if end and n >= 2:
        ax.plot([X[-1]], [Y[-1]], "o", color=color, ms=ms + 3.5, mec="white", mew=1.0, zorder=z + 1, alpha=alpha)


def ckpts(ax, ck, color, marker="D", filled=True, ms=9, label=None, z=6, arrows=True, alpha=1.0):
    """ck [3, 3] (x forward, y left, psi) in NOW."""
    ck = np.asarray(ck, float)
    for k in range(3):
        x, y, psi = ck[k]
        ax.plot(-y, x, marker, ms=ms, mfc=color if filled else "white", mec=color if filled else color, mew=1.8 if not filled else 1.0, zorder=z,
                label=label if k == 0 else None, alpha=alpha, clip_on=True)
        if arrows:
            ax.annotate("", xy=(-(y + 3.6 * np.sin(psi)), x + 3.6 * np.cos(psi)), xytext=(-y, x), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, alpha=alpha), zorder=z)


def legend(ax, loc="upper left", fs=9.2, ncol=1):
    h, l = ax.get_legend_handles_labels()
    if h:
        ax.legend(h, l, loc=loc, fontsize=fs, framealpha=0.94, borderpad=0.45, labelspacing=0.35, ncol=ncol, handlelength=2.0)


def camera(fig, box, img, title=None, sub=None, ec=INK):
    x, y, w, h = box
    ax = axpx(fig, x, y, w, h, W, H)
    ax.imshow(img, extent=(0, 640, 256, 0), interpolation="nearest", aspect="auto")
    ax.set_xlim(0, 640)
    ax.set_ylim(256, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(ec)
        s.set_linewidth(1.6)
    if title:
        tx(fig, x, y - 4, title, 10.5, INK, "bold", va="bottom")
    if sub:
        tx(fig, x, y + h + 3, sub, 9, MUTED)
    return ax


LAST_TEXTS: list = []


def _drawn_texts(fig):
    out = [t.get_text() for t in fig.texts if t.get_text()]
    for ax in fig.axes:
        out += [t.get_text() for t in ax.texts if t.get_text()]
        out += [x for x in (ax.get_title(loc="left"), ax.get_title(), ax.get_xlabel(), ax.get_ylabel()) if x]
        lg = ax.get_legend()
        if lg is not None:
            out += [t.get_text() for t in lg.get_texts()]
            if lg.get_title().get_text():
                out.append(lg.get_title().get_text())
    return out


def render(fig):
    LAST_TEXTS[:] = _drawn_texts(fig)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def shade(v, vmax, base=(0.86, 0.92, 0.98), hi=(0.10, 0.36, 0.66)):
    t = float(np.clip(v / vmax, 0, 1))
    return tuple(b + (h_ - b) * t for b, h_ in zip(base, hi))
