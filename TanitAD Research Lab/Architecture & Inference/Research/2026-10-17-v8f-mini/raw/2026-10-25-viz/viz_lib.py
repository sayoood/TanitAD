"""viz_lib.py -- drawing helpers shared by the key-frame figure and the video (matplotlib, Agg, CPU).

Colour is a LEVEL IDENTITY, never a value scale shared across levels: S = reddish purple, T = blue, O = orange (Okabe-Ito, colour-blind safe).
Each level's heat overlay uses its own ramp; opacity is proportional to  cell / (max cell of that panel)  -- the panel states the absolute max.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

LV_COL = {"S": "#CC79A7", "T": "#0072B2", "O": "#E69F00"}
LV_RAMP = {"S": ["#fbd3ec", "#e040a0", "#7a0a5c"], "T": ["#c9ecff", "#1e8cff", "#082a8a"], "O": ["#fff0b3", "#ff9a14", "#b53009"]}
LV_CMAP = {k: LinearSegmentedColormap.from_list(f"lv{k}", v) for k, v in LV_RAMP.items()}
LV_TITLE = {"S": "S  strategic  (ticks 1 Hz)", "T": "T  tactical  (ticks 2 Hz)", "O": "O  operative  (every step, NOW)"}
LAT5 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHG_L", "LANE_CHG_R")
LON7 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECEL", "ACCEL", "KEEP")
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT", "SPEED_BAND", "CORRIDOR_OFFSET",
          "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET", "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT",
          "TRAFFIC_LIGHT_REACT_RED", "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")
CONS_FIELDS = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg", "turn_r_arc_m", "turn_exit_x_m",
               "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m", "lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s",
               "stop_x_m", "stop_y_m", "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s", "v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")
CIDX = {n: i for i, n in enumerate(CONS_FIELDS)}
COND_NAMES = {"S": ("ego", "nav", "lim"), "T": ("ego", "nav", "lim", "ck"), "O": ("ego", "nav", "lim", "ck", "code", "age")}
INK, MUTED, BG, PANEL = "#1b1f24", "#5b6470", "#f6f7f9", "#ffffff"
FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "axes.edgecolor": "#c9ced6", "text.color": INK})


def axpx(fig, x, y, w, h, W=1920, H=1080, **kw):
    """axes from PIXEL box (x, y from the TOP-LEFT)."""
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def draw_overlay(ax, frame, cell, level, n_annot=3, total_mass=1.0):
    """frame [256, 640, 3] uint8; cell [4, 10] = attention mass of the TICK frame's 40 tokens (each cell = a 64x64 px block)."""
    ax.imshow(frame, extent=(0, 640, 256, 0), interpolation="nearest", aspect="auto")
    mx = float(cell.max()) if cell.max() > 0 else 1.0
    v = cell / mx
    rgba = LV_CMAP[level](v)
    rgba[..., 3] = 0.12 + 0.70 * v ** 0.85
    ax.imshow(rgba, extent=(0, 640, 256, 0), interpolation="nearest", aspect="auto")
    for k in range(1, 10):
        ax.plot([64 * k, 64 * k], [0, 256], color="white", lw=0.4, alpha=0.30)
    for k in range(1, 4):
        ax.plot([0, 640], [64 * k, 64 * k], color="white", lw=0.4, alpha=0.30)
    top = np.argsort(cell.ravel())[::-1][:n_annot]
    for tk in top:
        r, c = divmod(int(tk), 10)
        ax.text(64 * c + 32, 64 * r + 32, f"{100 * cell[r, c] / total_mass:.1f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=LV_COL[level], ec="none", alpha=0.85))
    ax.set_xlim(0, 640)
    ax.set_ylim(256, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(LV_COL[level])
        s.set_linewidth(2.2)


def bev_extent(rows_gt, rows_pred, rows_ck):
    xs = np.concatenate([np.ravel(rows_gt[..., 0]), np.ravel(rows_pred[..., 0]), np.ravel(rows_ck[..., 0])])
    ys = np.concatenate([np.ravel(rows_gt[..., 1]), np.ravel(rows_pred[..., 1]), np.ravel(rows_ck[..., 1])])
    ymax = float(np.nanmax(xs)) * 1.08 + 5
    lat = max(12.0, float(np.nanmax(np.abs(ys))) * 1.15 + 3)
    return ymax, lat


def draw_bev(ax, r, ext, show_oracle=True):
    """forward = up, LEFT of the car = left of the plot  (plot x = -y_left).  r: dict with gt/traj30/traj30_oracle/ck_now."""
    ymax, lat = ext
    ax.set_facecolor(PANEL)
    for rad in (25, 50, 75, 100, 150):
        if rad < ymax:
            ax.add_patch(plt.Circle((0, 0), rad, fill=False, ec="#d5d9e0", lw=0.6, ls=":"))
            ax.text(0, rad, f"{rad} m", fontsize=7, color=MUTED, ha="center", va="bottom")
    ax.plot(-r["gt"][:, 1], r["gt"][:, 0], "-", color="#1b1f24", lw=2.6, label="GT future path (6 s)", zorder=3)
    ax.plot(-r["gt"][4::5, 1], r["gt"][4::5, 0], "o", color="#1b1f24", ms=4, zorder=3)
    if show_oracle and r.get("traj30_oracle") is not None and np.isfinite(r["traj30_oracle"]).all():
        ax.plot(-r["traj30_oracle"][:, 1], r["traj30_oracle"][:, 0], "--", color="#009E73", lw=2.0, label="T0 ORACLE (GT code), diagnostic only", zorder=4)
    ax.plot(-r["traj30"][:, 1], r["traj30"][:, 0], "-", color=LV_COL["O"], lw=3.0, label="O prediction (from T's own code)", zorder=5)
    ax.plot(-r["traj30"][4::5, 1], r["traj30"][4::5, 0], "o", color=LV_COL["O"], ms=5, zorder=5)
    ck = r["ck_now"]
    for k in range(3):
        x, y, psi = float(ck[k, 0]), float(ck[k, 1]), float(ck[k, 2])
        ax.plot(-y, x, "D", color=LV_COL["S"], ms=9, mec="white", mew=1.0, zorder=6, label="S checkpoints 30/50/80 m" if k == 0 else None)
        ax.annotate("", xy=(-(y + 4.0 * np.sin(psi)), x + 4.0 * np.cos(psi)), xytext=(-y, x),
                    arrowprops=dict(arrowstyle="-|>", color=LV_COL["S"], lw=1.4), zorder=6)
    ax.add_patch(Polygon([[0, 2.2], [-1.1, -1.8], [1.1, -1.8]], closed=True, fc=INK, ec="white", zorder=7))
    ax.set_xlim(-lat, lat)
    ax.set_ylim(-6, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("lateral (m)   left <-   -> right", fontsize=8, color=MUTED)
    ax.set_ylabel("forward (m), ego frame at NOW", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92, ncol=1, borderpad=0.4)


def hbars(ax, names, vals, color, gt_idx=None, fmt="{:.0%}", xmax=1.0, fs=8.5):
    n = len(names)
    y = np.arange(n)[::-1]
    ax.barh(y, vals, color=color, height=0.62, zorder=2)
    for yi, nm, v in zip(y, names, vals):
        ax.text(-0.01 * xmax, yi, nm, ha="right", va="center", fontsize=fs, color=INK)
        if v > 0.82 * xmax:
            ax.text(v - 0.012 * xmax, yi, fmt.format(v), ha="right", va="center", fontsize=fs - 0.5, color="white", fontweight="bold", zorder=4)
        else:
            ax.text(v + 0.012 * xmax, yi, fmt.format(v), ha="left", va="center", fontsize=fs - 0.5, color=INK, zorder=4)
    if gt_idx is not None and 0 <= gt_idx < n:
        ax.barh(y[gt_idx], xmax, height=0.62, fc="none", ec="#1b1f24", lw=1.6, zorder=3)
        ax.text(xmax * 1.015, y[gt_idx], "GT", ha="left", va="center", fontsize=7.5, color="#1b1f24", fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.4)
    ax.axis("off")


def draw_code(fig, box, r, W=1920, H=1080):
    """T's tactical code: lat 5-way, lon 7-way, top-4 goals, key constraints."""
    x, y, w, h = box
    bg = axpx(fig, x, y, w, h, W, H)
    bg.set_facecolor(PANEL)
    bg.set_xticks([])
    bg.set_yticks([])
    for s in bg.spines.values():
        s.set_edgecolor(LV_COL["T"])
        s.set_linewidth(1.6)
    bg.text(0.02, 0.985, "T's tactical code  (the HELD output O reads)", transform=bg.transAxes, fontsize=10.5, fontweight="bold", color=LV_COL["T"], va="top")
    bg.text(0.02, 0.95, "bars = T's prediction;  black box = GT label (diagnostic only)", transform=bg.transAxes, fontsize=7.5, color=MUTED, va="top")
    L = 0.30
    a1 = axpx(fig, x + w * L, y + 62, w * (1 - L) - 52, 100, W, H)
    hbars(a1, LAT5, r["lat_p"], LV_COL["T"], gt_idx=r["gt_lat_cls"] if r.get("gt_lat_cls", -1) >= 0 else None)
    bg.text(0.02, 1 - 52 / h, "lateral decision", transform=bg.transAxes, fontsize=8.5, color=MUTED, va="center", fontweight="bold")
    a2 = axpx(fig, x + w * L, y + 196, w * (1 - L) - 52, 140, W, H)
    hbars(a2, LON7, r["lon_p"], LV_COL["T"], gt_idx=r["gt_lon_cls"] if r.get("gt_lon_cls", -1) >= 0 else None)
    bg.text(0.02, 1 - 184 / h, "longitudinal decision", transform=bg.transAxes, fontsize=8.5, color=MUTED, va="center", fontweight="bold")
    top = np.argsort(r["goal_prob"])[::-1][:4]
    a3 = axpx(fig, x + w * 0.40, y + 378, w * 0.60 - 52, 82, W, H)
    hbars(a3, [GOAL22[i].replace("TRAFFIC_LIGHT_REACT", "TL") for i in top], r["goal_prob"][top], LV_COL["T"], fs=8)
    bg.text(0.02, 1 - 366 / h, "top goals  (sigmoid)", transform=bg.transAxes, fontsize=8.5, color=MUTED, va="center", fontweight="bold")
    cp, gc = r["cons_pred"], r["gt_cons"]
    lines = []
    for nm, lab, u in (("lon_v_target_ms", "target speed", "m/s"), ("lead_gap_m", "lead gap", "m"),
                       ("turn_d_start_m", "turn starts in", "m"), ("turn_dyaw_deg", "turn heading change", "deg")):
        i = CIDX[nm]
        g = gc[i]
        lines.append(f"{lab:<20s} {cp[i]:7.1f} {u:<4s}" + (f"   (GT {g:.1f})" if np.isfinite(g) else "   (GT absent)"))
    bg.text(0.02, 1 - 478 / h, "key constraints  (T regression  vs  GT)", transform=bg.transAxes, fontsize=8.5, color=MUTED, va="center", fontweight="bold")
    bg.text(0.03, 1 - 496 / h, "\n".join(lines), transform=bg.transAxes, fontsize=8.6, family="DejaVu Sans Mono", va="top", color=INK, linespacing=1.4)


def draw_cond_bars(fig, boxes, r, W=1920, H=1080):
    """per level: where the attention goes, split into cond tokens | scene tick-frame | scene older frames, with the UNIFORM-attention reference."""
    for (x, y, w, h), lv in zip(boxes, ("S", "T", "O")):
        names = list(COND_NAMES[lv]) + ["scene: tick frame", "scene: 3 older frames"]
        cond, sc = r[f"cond_{lv}"], r[f"scene_{lv}"]
        vals = np.array(list(cond) + [sc[-1].sum(), sc[:-1].sum()])
        nmem = len(cond) + 160
        unif = np.array([1 / nmem] * len(cond) + [40 / nmem, 120 / nmem])
        bg = axpx(fig, x, y, w, h, W, H)
        bg.set_facecolor(PANEL)
        bg.set_xticks([])
        bg.set_yticks([])
        for s in bg.spines.values():
            s.set_edgecolor(LV_COL[lv])
            s.set_linewidth(1.4)
        p = sc.reshape(-1)
        p = p / max(p.sum(), 1e-12)
        eff = float(np.exp(-(p * np.log(np.maximum(p, 1e-30))).sum()))
        bg.text(0.015, 0.97, f"{lv}: attention mass by source", transform=bg.transAxes, fontsize=9.5, fontweight="bold", color=LV_COL[lv], va="top")
        bg.text(0.985, 0.97, f"scene focus: {eff:.0f} effective tokens / 160", transform=bg.transAxes, fontsize=7.8, color=MUTED, va="top", ha="right")
        ax = axpx(fig, x + w * 0.30, y + 26, w * 0.60, h - 32, W, H)
        n = len(names)
        yy = np.arange(n)[::-1]
        ax.barh(yy, vals, color=LV_COL[lv], height=0.62, zorder=2)
        ax.scatter(unif, yy, marker="|", s=170, color=INK, zorder=3, linewidths=1.6)
        for yi, nm, v in zip(yy, names, vals):
            ax.text(-0.01, yi, nm, ha="right", va="center", fontsize=8, color=INK, transform=ax.get_yaxis_transform())
            ax.text(min(v, 0.98) + 0.012, yi, f"{100 * v:.1f}%", ha="left", va="center", fontsize=7.8, color=INK)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.6, n - 0.4)
        ax.axis("off")
    # legend for the black tick
    fig.text(boxes[0][0] / W + 0.005, 1 - (boxes[0][1] - 6) / H, "black tick | = share if attention were UNIFORM over that level's memory", fontsize=7.3, color=MUTED, va="bottom")


def fmt_age(a):
    return f"{0.2 * a:.1f} s"


def style_title(ax, text, color):
    ax.text(0.0, 1.0, text, transform=ax.transAxes, fontsize=10.5, fontweight="bold", color=color, va="bottom", ha="left")
