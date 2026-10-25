"""v2_scenes.py -- every scene (card or animated frame) of the v2 video.  Each scene is a function returning an RGB uint8 [1080, 1920, 3]."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2_lib import *  # noqa: E402,F401,F403
from v2_draw import *  # noqa: E402,F401,F403
from matplotlib.patches import Polygon  # noqa: E402

CF = np.load(f"{OUT}/data/cf_windows.npz")
PK = np.load(f"{OUT}/data/plans_pack.npz")
TB = json.load(open(f"{OUT}/data/tables.json"))
EW = json.load(open(f"{OUT}/data/cf_evalwide.json"))
SFX = json.load(open(f"{SC}/sfresh/sfresh_extra.json"))
S0, S1 = SEEDS
WLAB = {"JUNCTION": "JUNCTION  (median of its stratum)", "CURVE": "CURVE  (median of its stratum)", "NO-TURN": "NO-TURN  (median of its stratum)",
        "FAILURE(worst)": "FAILURE CASE  (worst of all 2,399 windows)"}
WSHORT = {"JUNCTION": "JUNCTION", "CURVE": "CURVE", "NO-TURN": "NO-TURN", "FAILURE(worst)": "FAILURE (worst)"}
CFG_LAB = {"a_all": "(a) ALL three layers", "b_O": "(b) O only", "c_T": "(c) T only", "d_S": "(d) S only"}
DESC = ("(Only the nav input of the named layer(s) is replaced; everything else is the window's own; the change flows down S -> T -> O through the model's own chain.)")
TIER_CF = "T1-class open loop  |  INTERVENTION on the model's input: not driving"


NL = chr(10)


def cfd(seed, name, key):
    return CF[f"{seed}/{name}/{key}"]


def fr(name, j):
    return v2ep_frame(WINDOWS[name]["clip"], int(j))


def own_nav_str(name, seed=S0):
    v = cfd(seed, name, "nav_own_now")
    k = int(np.argmax(v[:3]))
    if v[4] < 0.5:
        return ["FOLLOW", "TURN_L", "TURN_R"][k] + " (no distance given)" if v[:3].sum() > 0 else "no nav (all zero)"
    d = 100 * v[3]
    return f"{['FOLLOW', 'TURN_L', 'TURN_R'][k]}, next turn in " + (">= 300 m" if d >= 299 else f"{d:.0f} m")


def side(y):
    return f"{abs(y):.1f} m {'left' if y > 0 else 'right'}"


def ext_for(name, seed=S0, mode="all"):
    """mode 'plans': GT + every counterfactual plan (+ B); 'all': also every S checkpoint; 'planset': GT + the 5 compared plans; 'strat': GT, GT ckpts, held / fresh ckpts."""
    if mode == "planset":
        return fit_extent([cfd(seed, name, "gt")] + [PK[f"{seed}/{name}/{a}"] for a in ("B", "P10", "H", "HS")])
    if mode == "strat":
        return fit_extent([cfd(seed, name, "gt"), cfd(seed, name, "rc"), cfd(seed, name, "own_ckpt_now"), cfd(seed, name, "fresh_ckpt_now")])
    arrs = [cfd(seed, name, "gt"), PK[f"{seed}/{name}/B"]]
    for c in CONFIGS:
        for cm in CMDS:
            arrs.append(cfd(seed, name, f"cf/{c}/{cm}/traj30"))
            if mode == "all":
                arrs.append(cfd(seed, name, f"cf/{c}/{cm}/ckpt_now"))
    return fit_extent(arrs)


def lat_txt(p, short=True):
    k = int(np.argmax(p))
    nm = LAT5[k]
    return f"{nm} {100 * p[k]:.0f}%"


def common_info(name):
    sel = WINDOWS[name]
    return f"window #{sel['window']}  |  clip {sel['clip'][:8]}  |  t = {0.2 * sel['t']:.1f} s"


# ================================================================================================ 1. title
def title_card():
    fig = new_fig(DARK)
    tx(fig, 90, 64, "TanitAD v8F-mini", 40, "white", "bold")
    tx(fig, 90, 124, "what each layer depends on, and what the planner outputs", 25, "#e8ecf2")
    tx(fig, 90, 176, "arm M2b, two training seeds (s0, s1)  |  CPU re-run of the banked checkpoints  |  T1-class open loop; interventions are diagnostics, not driving", 13.5, "#aeb7c4")
    # the three layers
    cols = {"S": LV_COL["S"], "T": LV_COL["T"], "O": LV_COL["O"]}
    items = [("S", "strategic layer", "route checkpoints at 30 / 50 / 80 m", "ticks ~1 Hz"), ("T", "tactical layer", "lateral / longitudinal decision + constraints (its CODE)", "ticks ~2 Hz"),
             ("O", "operative layer", "6 s trajectory, 10 Hz", "runs at every step")]
    for i, (lv, nm, what, hz) in enumerate(items):
        x = 90 + i * 590
        ax = axpx(fig, x, 250, 520, 190, W, H)
        ax.set_facecolor("#18202a")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(cols[lv]); s.set_linewidth(3)
        tx(fig, x + 22, 266, lv, 46, cols[lv], "bold")
        tx(fig, x + 100, 276, nm, 22, "white", "bold")
        tx(fig, x + 100, 312, hz, 15, "#aeb7c4")
        tx(fig, x + 22, 372, wrap(what, 40), 16, "#e8ecf2")
        if i < 2:
            fig.text((x + 520 + 35) / W, 1 - 345 / H, "->", fontsize=30, color="#aeb7c4", ha="center", va="center", fontweight="bold")
    tx(fig, 90, 470, "Every layer also reads the NAV command (FOLLOW / TURN_L / TURN_R and the distance to the next turn), ego motion and the camera tokens.", 15, "#aeb7c4")
    tx(fig, 90, 540, "Three questions", 24, "white", "bold")
    qs = ["1   Who listens to what?  Which input does each layer actually depend on (causal, one input swapped at a time)?",
          "2   Does the trajectory follow the nav command, and through which layer?  (nav injected at one layer at a time)",
          "3   What does the improved planner output look like against the human path, the kinematic prior and the old plan?"]
    for i, q in enumerate(qs):
        tx(fig, 90, 590 + i * 56, q, 18.5, "#e8ecf2")
    tx(fig, 90, 790, wrap("Four windows were declared before anything was rendered (SELECTION_V2.md): JUNCTION #1131, CURVE #654, NO-TURN #763, FAILURE #2158.  Every panel is seed 0; seed 1 sits in a summary card.", 150), 13.5, "#aeb7c4")
    tx(fig, 90, 850, "Tiers: T1-class open loop = the model reads its own held outputs, nothing it does feeds back into the scene.  Intervention = an input is replaced; it shows dependence, not driving.", 13.5, "#aeb7c4")
    tx(fig, 90, 1010, "Estimators: paired / single-arm episode-cluster bootstrap, B = 2,000 over 141 episodes (answers 'another draw of EPISODES'); training variance = the two seeds side by side; inference is deterministic.", 11.5, "#7f8a99")
    return render(fig)


# ================================================================================================ 2. who listens to what
_LG = None
LISTEN_NOTE = ("LISTEN grid (zero training, CPU): one input of ONE layer swapped for a donor window of another episode; the layer's own output is re-read.  "
               "Tier T1-class open loop, DIAGNOSTIC.  Paired episode-cluster bootstrap (B = 2,000, 141 episodes); the two training seeds side by side.  Source: listen/LISTEN_RESULT.md")
# display-coordinate geometry of fig_listen_grid.png (displayed 2000 x 731 = original / 1.75)
GRID_Y0, GRID_DY = 147.0, 96.5
LISTEN_ROW = {"S": 0, "T-lat": 1, "T-lon": 2, "O": 3}
LISTEN_COL = {"scene": 0, "ego": 1, "nav": 2, "limit": 3, "S ck": 4, "T code": 5, "tick": 6}


def _grid_x(seed, col):
    x0 = 240.0 + (595.0 if seed == 1 else 0.0)
    return x0 + 72.3 * col, x0 + 72.3 * (col + 1)


def listen_frame(step: int):
    from PIL import Image
    global _LG
    if _LG is None:
        _LG = Image.open(f"{SC}/listen/fig_listen_grid.png").convert("RGB")
    fig = new_fig()
    header(fig, "2  Who listens to what: one input of ONE layer swapped for a donor's", tier="T1-class open loop  |  diagnostic interventions")
    box = (0, 58, 1500, 600)                                   # display coords
    s = 1.75
    crop = _LG.crop((int(box[0] * s), int(box[1] * s), int(box[2] * s), int(box[3] * s)))
    sc = 1900.0 / crop.width
    wpx, hpx = 1900, int(crop.height * sc)
    ax = axpx(fig, 10, 66, wpx, hpx, W, H)
    ax.imshow(np.asarray(crop), extent=(box[0], box[2], box[3], box[1]), interpolation="lanczos", aspect="auto")
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[3], box[1]); ax.axis("off")
    hl = {1: ("S", "#CC79A7"), 2: ("T-lat", "#0072B2"), 3: ("O", "#E69F00")}
    if step in hl:
        row, col = hl[step]
        r = LISTEN_ROW[row]
        y0, y1 = GRID_Y0 + GRID_DY * r, GRID_Y0 + GRID_DY * (r + 1)
        # dim everything but the highlighted row (both seeds)
        for (ya, yb) in ((box[1], y0), (y1, box[3])):
            ax.add_patch(Rectangle((box[0], ya), box[2] - box[0], yb - ya, fc="white", ec="none", alpha=0.62, zorder=5))
        ax.add_patch(Rectangle((20, y0), 1400, y1 - y0, fc="none", ec=col, lw=4.0, zorder=6))
        if step == 2:                                          # the scene cell of T-lat, both seeds: the one that reads ~0
            for sd in (0, 1):
                xa, xb = _grid_x(sd, 0)
                ax.add_patch(Rectangle((xa, y0), xb - xa, y1 - y0, fc="none", ec="#b30000", lw=3.2, ls="--", zorder=7))
    if step == 4:                                              # the nav path: S x nav, T-lat x (nav, S ck), O x (S ck, T code) solid; O x nav (about zero) dashed red
        rows_ = {"S": 0, "T": 1, "O": 3}
        cells_ = [("S", 2, "#CC79A7", "-"), ("T", 2, "#0072B2", "-"), ("T", 4, "#0072B2", "-"), ("O", 4, "#E69F00", "-"), ("O", 5, "#E69F00", "-"), ("O", 2, "#b30000", "--")]
        for sd in (0, 1):
            for rw, cl, colr, lsx in cells_:
                xa, xb = _grid_x(sd, cl)
                ya = GRID_Y0 + rows_[rw] * GRID_DY
                ax.add_patch(Rectangle((xa + 1, ya + 2), xb - xa - 2, GRID_DY - 4, fc="none", ec=colr, lw=3.4, ls=lsx, zorder=7))
    legend_txt = "Cell = layer's output change when that input is swapped (S, O: metres; T: macro-F1 drop).   Heavy frame = the row's largest ratio n.   * = listens at both seeds.   Hatched = not an input."
    tx(fig, 14, 66 + hpx + 2, legend_txt, 10, MUTED)
    caps = {
        0: ("HOW TO READ IT.  Take ONE input of ONE layer (scene, ego, nav, speed limit, S's checkpoints, T's code, tick ages), swap it for the value from a window of ANOTHER episode, "
            "and measure how much that layer's own output gets worse.  A big cell = the layer depends on that input.  Left: training seed 0.  Right: seed 1.", ""),
        1: ("S (strategic) listens to NAV and the scene.  Swapping nav moves S's 50 m checkpoint by +4.20 m (seed 0) / +3.99 m (seed 1), swapping the scene by +3.21 / +3.63 m; "
            "swapping every input: +8.06 / +8.38 m.", "n = 0.52 / 0.48 for nav; paired CI s0 [+2.99, +5.52], s1 [+2.81, +5.18], separated at both seeds."),
        2: ("T's LATERAL decision listens to S's CHECKPOINTS (macro-F1 drop +0.391 / +0.406) and to nav (+0.173 / +0.216), not to the image: the scene cell is -0.003 / +0.010, not separated.",
            "T puts ~97-100 % of its attention on scene tokens, yet the scene cell reads ~0: attention is not dependence.  T's longitudinal decision does read ego (+0.247 / +0.261) and the scene (+0.185 / +0.187)."),
        3: ("O (trajectory) listens to EGO (ADE6 +5.70 / +4.72 m) and to T's CODE (+5.40 / +4.13 m), then S's checkpoints (+0.87 / +1.89 m).  Its own nav token barely matters: +0.015 / +0.011 m.",
            "n > 1 for O: the sum of single swaps exceeds the all-input swap (+1.39 / +1.32 m), because a donor token next to own tokens creates disagreement; read it with care."),
        4: ("So nav reaches the trajectory through S's checkpoints and through T, not through O's own nav token.", "Frames: purple = S x nav;  blue = T x nav and T x S checkpoints;  orange = O x S checkpoints and O x T code;  red dashed = O x nav, about zero.  Next: the donor-nav intervention at one layer at a time."),
    }
    c, sub = caps[step]
    caption(fig, c, size=17.5, y0=840, h=170, n=132, sub=sub or None)
    footer(fig, LISTEN_NOTE)
    return render(fig)


def qnav_frame(p=1.0):
    fig = new_fig()
    header(fig, "2  Who listens to what: does nav reach the plan through T, or around it?", tier="T1-class open loop  |  diagnostic interventions")
    ax = axpx(fig, 90, 150, 1000, 620, W, H)
    ax.set_facecolor(PANEL)
    arms = ["S only", "T only", "O only", "S + T + O"]
    v0 = [0.726, 0.476, 0.015, 1.174]
    c0 = [(0.426, 1.081), (0.261, 0.703), (0.001, 0.030), (0.759, 1.634)]
    v1 = [0.821, 0.521, 0.011, 1.180]
    c1 = [(0.509, 1.172), (0.282, 0.812), (-0.002, 0.024), (0.769, 1.619)]
    xs = np.arange(4)
    for off, vv, cc, col, lab in ((-0.19, v0, c0, "#08519c", "seed 0"), (0.19, v1, c1, "#9ecae1", "seed 1")):
        ax.bar(xs + off, np.array(vv) * p, 0.36, color=col, ec="#1b1f24", lw=0.8, label=lab, zorder=3)
        for i in range(4):
            if p > 0.99:
                ax.errorbar(xs[i] + off, vv[i], yerr=[[vv[i] - cc[i][0]], [cc[i][1] - vv[i]]], color="#1b1f24", capsize=4, lw=1.4, zorder=4)
                ax.text(xs[i] + off, cc[i][1] + 0.04, f"+{vv[i]:.2f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(arms, fontsize=14)
    ax.set_ylim(0, 1.95)
    ax.set_ylabel("ADE6 change vs clean (m)", fontsize=12)
    ax.grid(axis="y", color="#e1e5ea", zorder=0)
    ax.legend(loc="upper left", fontsize=12, title="donor NAV injected at", title_fontsize=11)
    ax.tick_params(labelsize=11)
    tx(fig, 1130, 160, "Q-NAV, the committed contrast", 18, INK, "bold")
    tx(fig, 1130, 200, wrap("D = ADE6(nav at T only) - ADE6(nav at O only), paired:\n\n"
                            "seed 0:  +0.461 m  [+0.254, +0.687]   separated\n"
                            "seed 1:  +0.510 m  [+0.273, +0.793]   separated\n\n"
                            "Outcome table: THROUGH T.", 52), 14.5, INK, mono=False)
    tx(fig, 1130, 420, wrap("Beside it (descriptive): the S-only arm is LARGER than the T-only arm, so the route also passes through S's checkpoints, which O reads directly.  "
                            "Nav at O only moves the plan by +0.015 / +0.011 m: O's own nav token is almost unused.", 56), 14, INK)
    caption(fig, "The donor's nav injected at ONE layer: at T only the plan moves +0.48 / +0.52 m, at O only +0.015 / +0.011 m, at S only +0.73 / +0.82 m, at all three +1.17 / +1.18 m.  "
                 "Nav reaches the trajectory through S and T; O's own nav input adds almost nothing.  Next: the same question with a synthetic command, on four windows.",
            size=17, y0=860, h=150, n=130)
    footer(fig, "LISTEN_RESULT.md section 4 (listen/READ.json): paired episode-cluster bootstrap (B = 2,000, 141 episodes; another draw of EPISODES); the two training seeds side by side.  Donor inputs are interventions, not driving.  Tier T1-class open loop, DIAGNOSTIC.")
    return render(fig)


# ================================================================================================ 3. nav counterfactual
def _plan_stats(seed, name, cfg, cmd):
    pl = cfd(seed, name, f"cf/{cfg}/{cmd}/traj30")
    ck = cfd(seed, name, f"cf/{cfg}/{cmd}/ckpt_now")
    lat = cfd(seed, name, f"cf/{cfg}/{cmd}/lat_p")
    return pl, ck, lat


def method_card():
    fig = new_fig()
    header(fig, "3  Nav counterfactual: what happens to the plan when ONLY the nav input changes?", tier=TIER_CF)
    tx(fig, 60, 100, "The experiment", 22, INK, "bold")
    tx(fig, 60, 142, wrap("For each of the four pre-declared windows the nav input is REPLACED by a synthetic command and the model is run through its own chain S -> T -> O.\n"
                          "Commands:  FOLLOW,  TURN LEFT,  TURN RIGHT, each with d_next = 30 m and 'arguments valid' = 1.\n"
                          "Everything else is the window's own: ego motion, speed limit, camera tokens, held ages.", 125), 16.5, INK, ls=1.5)
    # diagram: three layer boxes, four injection arrangements
    rows = [("(a)", ("S", "T", "O"), "all three layers"), ("(b)", ("O",), "O only"), ("(c)", ("T",), "T only"), ("(d)", ("S",), "S only")]
    for i, (lab, inj, nm) in enumerate(rows):
        y = 380 + i * 112
        tx(fig, 80, y + 22, f"{lab}  nav injected at {nm}", 17, INK, "bold")
        for j, lv in enumerate(("S", "T", "O")):
            on = lv in inj
            x = 560 + j * 230
            ax = axpx(fig, x, y, 170, 80, W, H)
            ax.set_facecolor(LV_COL[lv] if on else "white")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(LV_COL[lv]); s.set_linewidth(2.4)
            tx(fig, x + 85, y + 40, f"{lv}: " + ("COMMAND" if on else "own nav"), 13.5, "white" if on else INK, "bold", ha="center", va="center")
            if j < 2:
                fig.text((x + 200) / W, 1 - (y + 40) / H, "->", fontsize=18, color=MUTED, ha="center", va="center")
    tx(fig, 1260, 392, wrap("Read for each case:\n  - the 6 s plan (dots every 1 s)\n  - S's three checkpoints\n  - T's lateral decision (argmax)\n\n"
                            "Headline number D6 = how far the TURN plans end (at 6 s) from the FOLLOW plan, in metres, averaged over TURN LEFT and TURN RIGHT.", 46), 15, INK, ls=1.4)
    caption(fig, "INTERVENTION on the model's input, open loop; not driving.  A TURN command on a straight road is something the road does not offer: the experiment shows what the model DEPENDS on, not what is correct.",
            size=17, y0=900, h=110, n=130)
    footer(fig, "SELECTION_V2.md section 2 (pre-declared before any render).  Command vectors in the model's nav encoding [FOLLOW, TURN_L, TURN_R one-hot, d_next/100, args_valid].  Layers are run through the model's own run_s / run_t / run_o (tip 9a6551d5).")
    return render(fig)


def cf_all(name, k, p, seed=S0):
    """(a) all three layers; commands < k are complete, command k is swept with fraction p, later ones absent.  k = 3: all complete."""
    sel = WINDOWS[name]
    ext = ext_for(name)
    fig = new_fig()
    header(fig, f"3  Nav counterfactual  |  {WLAB[name]}  |  (a) command injected at ALL three layers", tier=TIER_CF, sub=f"arm M2b, seed 0  |  {common_info(name)}")
    t, a_s = sel["t"], int(cfd(seed, name, "a_s"))
    camera(fig, (24, 112, 640, 256), fr(name, t), title="front camera at NOW (cylindrical 256 x 640)")
    # info
    info = (f"window's OWN nav input:  {own_nav_str(name)}\n"
            f"speed {cfd(seed, name, 'kin_v'):.1f} m/s ({3.6 * cfd(seed, name, 'kin_v'):.0f} km/h)   held ages: T {0.2 * int(cfd(seed, name, 'a_t')):.1f} s, S {0.2 * a_s:.1f} s\n"
            f"banked ADE6 of the window's own plan B: {PK[f'{seed}/{name}/ade6_B']:.2f} m")
    tx(fig, 24, 384, info, 12.2, INK, ls=1.45)
    # zoom on the plans only (the S checkpoints stretch the main panel's scale)
    zx = axpx(fig, 24, 500, 330, 420, W, H)
    zext = ext_for(name, mode="plans")
    ymax_, lat_ = zext
    rng = max(ymax_ + 6.0, 2.0 * lat_ * 420 / 330)
    zx.set_ylim(-6, -6 + rng); zx.set_xlim(-rng * 330 / 420 / 2, rng * 330 / 420 / 2); zx.set_aspect("equal", adjustable="box")
    zx.set_facecolor(PANEL); zx.tick_params(labelsize=7, colors=MUTED, length=2)
    zx.set_title("zoom: the 6 s plans only", fontsize=10.5, fontweight="bold", loc="left", pad=3)
    plan(zx, cfd(seed, name, "gt"), 1.0, color="#1b1f24", lw=2.6, z=3, end=False, ms=3.5)
    plan(zx, PK[f"{seed}/{name}/B"], 1.0, color="#6b7280", lw=1.5, ls="--", z=3, dots=False, end=False)
    for i, cm in enumerate(CMDS):
        if i > k:
            break
        plan(zx, cfd(seed, name, f"cf/a_all/{cm}/traj30"), p if i == k else 1.0, color=CMD_COL[cm], lw=2.8, z=5 + i, ms=3.5)
    zx.add_patch(Polygon([[0, 1.6], [-0.9, -1.3], [0.9, -1.3]], closed=True, fc=INK, ec="white", zorder=8))
    zx.plot([], [], "-", color="#1b1f24", lw=2.4, label="GT"); zx.plot([], [], "--", color="#6b7280", label="B (own nav)")
    for cm_ in CMDS:
        zx.plot([], [], "-", color=CMD_COL[cm_], lw=2.8, label=CMD_LAB[cm_])
    legend(zx, "upper left", 7.4)
    tx(fig, 366, 500, "How to read", 12.5, INK, "bold")
    tx(fig, 366, 530, wrap("Each COLOURED line is the 6 s plan the model outputs when its nav input is that command (dots every 1 s, big dot = 6 s).\n\n"
                           "Diamonds with arrows (right panel): S's three route checkpoints at 30 / 50 / 80 m, same colour as the command.\n\n"
                           "Black = the path the human drove (GT).  Dashed grey = the plan under the window's own nav (B).\n\n"
                           "Left: the same plans zoomed in, because S's far checkpoints stretch the scale of the big panel.", 33), 11.3, INK, ls=1.4)
    chips(fig, 700, 66, ("S", "T", "O"))
    ax = bev_axes(fig, (690, 100, 700, 826), ext)
    plan(ax, cfd(seed, name, "gt"), 1.0, color="#1b1f24", lw=3.0, label="GT: the human's 6 s path", z=3, end=False)
    plan(ax, PK[f"{seed}/{name}/B"], 1.0, color="#6b7280", lw=1.8, ls="--", label="B: plan under the window's OWN nav", z=3, dots=False, end=False)
    shown = 0
    for i, cm in enumerate(CMDS):
        if i > k:
            break
        pp = p if i == k else 1.0
        pl, ck, lat = _plan_stats(seed, name, "a_all", cm)
        plan(ax, pl, pp, color=CMD_COL[cm], lw=3.4, label=f"plan, nav = {CMD_LAB[cm]}", z=5 + i)
        if pp >= 0.99:
            ckpts(ax, ck, CMD_COL[cm], marker="D", ms=8.5, label=f"S checkpoints, nav = {CMD_LAB[cm]}", z=7)
    legend(ax, "upper left", 8.8)
    # right column cards
    tx(fig, 1410, 70, "result per command (seed 0)", 12.5, INK, "bold")
    f6 = cfd(seed, name, "cf/a_all/FOLLOW/traj30")[29]
    for i, cm in enumerate(CMDS):
        y0 = 100 + i * 205
        done = i < k or (i == k and p >= 0.99) or k >= 3
        panel_box(fig, 1410, y0, 500, 195, ec=CMD_COL[cm] if done else "#d5d9e0", lw=2.2 if done else 1.0)
        tx(fig, 1424, y0 + 8, f"nav = {CMD_LAB[cm]}  (d_next 30 m)", 14, CMD_COL[cm], "bold")
        if not done:
            tx(fig, 1424, y0 + 50, "...", 18, "#9aa3af")
            continue
        pl, ck, lat = _plan_stats(seed, name, "a_all", cm)
        d6 = float(np.linalg.norm(pl[29] - f6))
        lines = (f"T lateral decision:  {lat_txt(lat)}\n"
                 f"S checkpoint @80 m:  {ck[2, 0]:.0f} m ahead, {side(ck[2, 1])}\n"
                 f"plan at 6 s:  {pl[29, 0]:.0f} m ahead, {side(pl[29, 1])}\n"
                 + ("(reference for D6)" if cm == "FOLLOW" else f"6 s plan moves {d6:.1f} m vs FOLLOW"))
        tx(fig, 1424, y0 + 44, lines, 12.6, INK, ls=1.5)
    if k >= 3:
        tx(fig, 1410, 722, f"D6 = {cfd(seed, name, 'cf/a_all/D6'):.1f} m  (mean of TURN LEFT / RIGHT\nvs FOLLOW, at 6 s)\nleft-right lateral spread at 6 s = {cfd(seed, name, 'cf/a_all/lat_sep_LR'):.1f} m",
           13.2, "#08519c", "bold", ls=1.5)
    if k >= 3:
        cap = (f"The plan follows the command: with nav injected at all three layers, TURN LEFT and TURN RIGHT end {cfd(seed, name, 'cf/a_all/D6'):.1f} m (mean) from the FOLLOW plan at 6 s, "
               f"and T's lateral decision follows the command (LANE_KEEP / TURN_L / TURN_R).  Seed 1: {cfd(S1, name, 'cf/a_all/D6'):.1f} m (summary card).")
    else:
        cap = "Same window, same camera, same everything: only the nav input changes.  The plan swings with the command."
    caption(fig, cap, size=17, y0=942, h=92, n=130)
    footer(fig, TIER_CF + ".  " + DESC + "  Seed 0 shown; seed 1 in the summary card.")
    return render(fig)


def _bev_cell(fig, box, name, cfg, p, seed=S0, fs=8.2, legend_on=True):
    ax = bev_axes(fig, box, ext_for(name, mode="plans"), fs_tick=7)
    plan(ax, cfd(seed, name, "gt"), 1.0, color="#1b1f24", lw=2.4, label="GT (human)", z=3, end=False, ms=3.5)
    plan(ax, PK[f"{seed}/{name}/B"], 1.0, color="#6b7280", lw=1.5, ls="--", label="B (window's own nav)", z=3, dots=False, end=False)
    for i, cm in enumerate(CMDS):
        pl, ck, lat = _plan_stats(seed, name, cfg, cm)
        plan(ax, pl, p, color=CMD_COL[cm], lw=3.0, label=f"nav = {CMD_LAB[cm]}", z=5 + i, ms=3.8)
        if p >= 0.99:
            ckpts(ax, ck, CMD_COL[cm], marker="D", ms=7, z=7, arrows=False)
    ax.plot([], [], "D", color="#6b7280", ms=6, label="S checkpoints (if in view)")
    if legend_on:
        legend(ax, "upper left", fs)
    return ax


def _cell_numbers(fig, x, y, name, cfg, seed=S0, fs=12, compact=False):
    d6, sep = cfd(seed, name, f"cf/{cfg}/D6"), cfd(seed, name, f"cf/{cfg}/lat_sep_LR")
    tx(fig, x, y, f"D6 = {d6:.1f} m   L-R spread = {sep:.1f} m" if compact else f"D6 = {d6:.1f} m     L-R lateral spread = {sep:.1f} m", fs + (1.8 if compact else 1.6), "#08519c", "bold")
    rows = []
    for cm in CMDS:
        _, ck, lat = _plan_stats(seed, name, cfg, cm)
        if compact:
            sd = "R" if ck[2, 1] < 0 else "L"
            rows.append(f"{CMD_LAB[cm].split()[-1] if cm != 'FOLLOW' else 'FOLLOW':<7s} T: {lat_txt(lat):<15s} S@80m: {abs(ck[2, 1]):.1f}{sd}")
        else:
            rows.append(f"nav {CMD_LAB[cm]:<11s}-> T lateral: {lat_txt(lat):<17s} S 80 m ckpt: {side(ck[2, 1])}")
    tx(fig, x, y + (30 if compact else 28), NL.join(rows), fs - (1.0 if compact else 1.2), INK, mono=True, ls=1.4)


def _sub_title(fig, x, y, cfg, extra=""):
    tx(fig, x + 4, y, CFG_LAB[cfg] + extra, 14, INK, "bold")


def cf_bcd(name, p, seed=S0, emphasize=("b_O", "c_T")):
    sel = WINDOWS[name]
    fig = new_fig()
    header(fig, f"3  Nav counterfactual  |  {WSHORT[name]}  |  nav at O only  vs  T only  vs  S only", tier=TIER_CF, sub=f"arm M2b, seed 0  |  {common_info(name)}")
    xs = (14, 650, 1286)
    for i, cfg in enumerate(("b_O", "c_T", "d_S")):
        x = xs[i]
        _sub_title(fig, x, 70, cfg, "   (for completeness)" if cfg == "d_S" else "")
        chips(fig, x + 8, 98, CONFIGS[cfg], fs=12)
        ax = _bev_cell(fig, (x, 126, 620, 640), name, cfg, p)
        if cfg in emphasize and p >= 0.99:
            for s_ in ax.spines.values():
                s_.set_edgecolor("#1b1f24"); s_.set_linewidth(2.4)
        if p >= 0.99:
            _cell_numbers(fig, x + 4, 798, name, cfg, fs=11.2)
    if p >= 0.99:
        g = lambda c, k: cfd(seed, name, f"cf/{c}/{k}")
        cap = (f"Nav at O ONLY: D6 {g('b_O', 'D6'):.1f} m, left-right lateral spread {g('b_O', 'lat_sep_LR'):.1f} m (T and S still read the window's own nav, so the code and the checkpoints O reads do not change).  "
               f"Nav at T ONLY: D6 {g('c_T', 'D6'):.1f} m, spread {g('c_T', 'lat_sep_LR'):.1f} m.  Nav at S ONLY: D6 {g('d_S', 'D6'):.1f} m, spread {g('d_S', 'lat_sep_LR'):.1f} m.  Seed 1 in the summary card.")
    else:
        cap = "Same window, three single injection points: O only, T only, S only."
    caption(fig, cap, size=16.5, y0=942, h=92, n=136)
    footer(fig, TIER_CF + ".  " + DESC)
    return render(fig)


def cf_four(name, p, seed=S0):
    sel = WINDOWS[name]
    fig = new_fig()
    header(fig, f"3  Nav counterfactual  |  {WLAB[name]}", tier=TIER_CF, sub=f"arm M2b, seed 0  |  {common_info(name)}")
    camera(fig, (14, 100, 320, 128), fr(name, sel["t"]), title="camera at NOW")
    tx(fig, 350, 102, f"window's OWN nav input:  {own_nav_str(name)}" + NL +
       f"speed {cfd(seed, name, 'kin_v'):.1f} m/s   banked ADE6 of B {PK[f'{seed}/{name}/ade6_B']:.2f} m (s0) / {PK[f'{S1}/{name}/ade6_B']:.2f} m (s1)" + NL +
       "commands: FOLLOW (green), TURN LEFT (blue), TURN RIGHT (vermillion), d_next = 30 m;  black = GT;  dashed grey = B", 12.5, INK, ls=1.45)
    for i, cfg in enumerate(CONFIGS):
        x = 14 + i * 474
        _sub_title(fig, x, 244, cfg)
        chips(fig, x + 4, 270, CONFIGS[cfg], fs=11)
        _bev_cell(fig, (x, 296, 462, 480), name, cfg, p, fs=7.2)
        if p >= 0.99:
            _cell_numbers(fig, x + 2, 812, name, cfg, fs=10.2, compact=True)
    g = lambda c, k: cfd(seed, name, f"cf/{c}/{k}")
    cap = (f"The four injection points, D6 (m): all layers {g('a_all', 'D6'):.1f}, O only {g('b_O', 'D6'):.1f}, T only {g('c_T', 'D6'):.1f}, S only {g('d_S', 'D6'):.1f} (seed 0); "
           f"left-right lateral spread at 6 s: {g('a_all', 'lat_sep_LR'):.1f} / {g('b_O', 'lat_sep_LR'):.1f} / {g('c_T', 'lat_sep_LR'):.1f} / {g('d_S', 'lat_sep_LR'):.1f} m.")
    if name == "FAILURE(worst)":
        cap += "  This is the window with the worst own-code error of all 2,399 (16.9 m); it is kept as it is."
    caption(fig, cap, size=16.5, y0=942, h=92, n=136)
    footer(fig, TIER_CF + ".  " + DESC)
    return render(fig)


def cf_summary():
    fig = new_fig()
    header(fig, "3  Nav counterfactual: how far each command moves the 6 s plan, per injection point, both seeds", tier=TIER_CF)
    cols = ["a_all", "b_O", "c_T", "d_S"]
    x0, y0, cw, rh = 330, 130, 380, 118
    for j, c in enumerate(cols):
        panel_box(fig, x0 + j * cw, y0, cw - 8, 70, ec="#9aa3af", fc="#eef1f5")
        tx(fig, x0 + j * cw + (cw - 8) / 2, y0 + 35, CFG_LAB[c], 16, INK, "bold", ha="center", va="center")
    tx(fig, 24, y0 + 28, "D6 in metres  (top: seed 0  |  seed 1)\nL-R lateral spread (below)", 11.5, MUTED, ls=1.35)
    rowsl = list(KEYS) + ["ALL"]
    for i, name in enumerate(rowsl):
        y = y0 + 80 + i * rh
        lab = WSHORT.get(name, "") if name != "ALL" else "ALL 2,399 windows"
        tx(fig, 24, y + rh / 2 - 8, lab, 15.5, INK, "bold", va="center")
        if name == "ALL":
            tx(fig, 24, y + rh / 2 + 20, "mean [95 % CI]", 11, MUTED, va="center")
        for j, c in enumerate(cols):
            if name != "ALL":
                a0, a1 = cfd(S0, name, f"cf/{c}/D6"), cfd(S1, name, f"cf/{c}/D6")
                b0, b1 = cfd(S0, name, f"cf/{c}/lat_sep_LR"), cfd(S1, name, f"cf/{c}/lat_sep_LR")
                t1, t2 = f"{a0:.1f}   |   {a1:.1f}", f"L-R {b0:.1f}   |   {b1:.1f}"
            else:
                e0, e1 = EW["seeds"][S0][c], EW["seeds"][S1][c]
                t1 = f"{e0['D6']['mean']:.1f}   |   {e1['D6']['mean']:.1f}"
                t2 = f"[{e0['D6']['lo']:.1f}, {e0['D6']['hi']:.1f}]  |  [{e1['D6']['lo']:.1f}, {e1['D6']['hi']:.1f}]"
                a0 = e0["D6"]["mean"]
            ax = panel_box(fig, x0 + j * cw, y, cw - 8, rh - 8, ec="#c9ced6", fc=shade(a0, 28.0))
            lum = 0.299 * ax.get_facecolor()[0] + 0.587 * ax.get_facecolor()[1] + 0.114 * ax.get_facecolor()[2]
            col = "white" if lum < 0.55 else INK
            tx(fig, x0 + j * cw + (cw - 8) / 2, y + 34, t1, 21, col, "bold", ha="center", va="center")
            tx(fig, x0 + j * cw + (cw - 8) / 2, y + 74, t2, 12.5 if name != "ALL" else 12, col, ha="center", va="center")
    f0 = EW["seeds"][S0]["a_all"]["T_lat_argmax_follows_cmd"]
    f1 = EW["seeds"][S1]["a_all"]["T_lat_argmax_follows_cmd"]
    tx(fig, 24, 826, wrap("D6 = mean over {TURN LEFT, TURN RIGHT} of the distance at 6 s between that plan and the FOLLOW plan, same injection point.  "
                          "Eval-wide, with nav at all three layers, T's lateral argmax equals the command in "
                          f"{100 * f0['FOLLOW']:.0f} / {100 * f0['TURN_L']:.0f} / {100 * f0['TURN_R']:.0f} % of windows (FOLLOW / TURN_L / TURN_R, seed 0) and {100 * f1['FOLLOW']:.0f} / {100 * f1['TURN_L']:.0f} / {100 * f1['TURN_R']:.0f} % (seed 1).", 215), 12, MUTED, ls=1.4)
    miss = [(WSHORT[n], sd[-1], cm) for n in KEYS for sd in SEEDS for k, cm in enumerate(CMDS) if int(np.argmax(cfd(sd, n, f"cf/a_all/{cm}/lat_p"))) != k]
    flips = [(WSHORT[n], sd[-1]) for n in KEYS for sd in SEEDS if cfd(sd, n, "cf/c_T/D6") < cfd(sd, n, "cf/b_O/D6")]
    nfol = 24 - len(miss)
    cap = ("Eval-wide (bottom row), both seeds: O only (about 1 m) < T only (about 4 m) < all three layers ~ S only (about 10-11 m).  The command moves the trajectory through S's checkpoints and T's code; O's own nav token is nearly inert.  "
           f"On the four windows (examples), T's lateral decision follows the command (nav at all three layers) in {nfol} of 24 cases (4 windows x 2 seeds x 3 commands)"
           + ("; the exceptions: " + ", ".join(f"{w} s{s} {c}" for w, s, c in miss) if miss else "") + "."
           + (f"  Single windows can flip the O-only / T-only order: T only < O only on {', '.join(f'{w} seed {s}' for w, s in flips)}." if flips else ""))
    caption(fig, cap, size=16, y0=890, h=135, n=150)
    footer(fig, "Artifacts: viz2/data/cf_windows.npz (4 windows, both seeds), viz2/data/cf_evalwide.json (2,399 windows; episode-cluster bootstrap B = 2,000, 141 episodes: another draw of EPISODES; the two training seeds side by side; inference deterministic).  "
                "Tier T1-class open loop; intervention on the model's input, not driving.")
    return render(fig)


# ================================================================================================ 4. better plans
def _horizon_inset(fig, box, name, seed):
    ax = axpx(fig, *box, W, H)
    ax.set_facecolor(PANEL)
    t = 0.2 * (np.arange(30) + 1)
    for arm in ("P10", "B", "H", "HS"):
        ax.plot(t, PK[f"{seed}/{name}/ade_h_{arm}"], "--" if arm == "P10" else "-", color=PLAN_COL[arm], lw=2.4, label=arm)
    ax.set_xlabel("horizon (s)", fontsize=9.5)
    ax.set_ylabel("distance to GT (m)", fontsize=9.5)
    ax.set_xlim(0.2, 6.0)
    ax.set_ylim(bottom=0)
    ax.grid(color="#e8ebef")
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper left", fontsize=9.5, ncol=2, framealpha=0.95)
    ax.set_title("error of each plan vs GT, by horizon (this window)", fontsize=10.5, fontweight="bold", loc="left")


def plans_frame(name, p, seed=S0):
    sel = WINDOWS[name]
    fig = new_fig()
    header(fig, f"4  Better plans  |  {WLAB[name]}", tier="T1-class open loop  |  own held outputs, not closed loop", sub=f"arm M2b, seed 0  |  {common_info(name)}")
    camera(fig, (24, 112, 640, 256), fr(name, sel["t"]), title="front camera at NOW")
    _horizon_inset(fig, (64, 420, 560, 290), name, seed)
    # per-window ADE6 table, both seeds
    tx(fig, 24, 752, "ADE6 of this window (m)", 12, INK, "bold")
    arms = ("P10", "B", "H", "HS")
    tx(fig, 24, 778, "plan         seed 0     seed 1", 11.5, MUTED, mono=True)
    for i, a in enumerate(arms):
        tx(fig, 24, 802 + i * 26, f"{a:<10s}{PK[f'{S0}/{name}/ade6_{a}']:>8.2f}   {PK[f'{S1}/{name}/ade6_{a}']:>8.2f}", 13, PLAN_COL[a] if a != "P10" else "#5b6470", "bold", mono=True)
    ext = ext_for(name, mode="planset")
    ax = bev_axes(fig, (690, 100, 720, 826), ext)
    plan(ax, cfd(seed, name, "gt"), 1.0, color=PLAN_COL["GT"], lw=3.4, label=PLAN_LAB["GT"], z=3, end=False, ms=5)
    for i, a in enumerate(("P10", "B", "H", "HS")):
        plan(ax, PK[f"{seed}/{name}/{a}"], p, color=PLAN_COL[a], lw=2.8 if a != "P10" else 2.2, ls="--" if a == "P10" else "-", label=PLAN_LAB[a], z=4 + i, dots=(a != "P10"), ms=4)
    legend(ax, "upper left", 8.6)
    tx(fig, 1430, 100, "what each line is", 13.5, INK, "bold")
    expl = ("GT\nthe human's path (the label)\n\nP10\nfrozen kinematic prior: extrapolates ego motion, does not look at the scene\n\n"
            "B\nM2b's own plan = P10 + a learned residual (what the banked eval scores)\n\nH\nB with the residual shrunk toward P10 by fitted per-step weights (60 numbers)\n\n"
            "HS\nH with the residual rebuilt from 12 smooth modes, each shrunk, plus a smoothness penalty")
    tx(fig, 1430, 136, wrap(expl, 34), 12, INK, ls=1.38)
    hb = {sd: PK[f"{sd}/{name}/ade6_H"] - PK[f"{sd}/{name}/ade6_B"] for sd in SEEDS}
    cap = (f"Banked plans of this window.  ADE6 (m), seed 0: P10 {PK[f'{S0}/{name}/ade6_P10']:.2f}, B {PK[f'{S0}/{name}/ade6_B']:.2f}, H {PK[f'{S0}/{name}/ade6_H']:.2f}, HS {PK[f'{S0}/{name}/ade6_HS']:.2f}; "
           f"seed 1: {PK[f'{S1}/{name}/ade6_P10']:.2f} / {PK[f'{S1}/{name}/ade6_B']:.2f} / {PK[f'{S1}/{name}/ade6_H']:.2f} / {PK[f'{S1}/{name}/ade6_HS']:.2f}.  "
           f"H minus B on this window: {hb[S0]:+.2f} m (s0), {hb[S1]:+.2f} m (s1)" + ("" if (hb[S0] < 0 and hb[S1] < 0) else " (H is not better than B everywhere: the table that follows is the evidence)") + ".")
    caption(fig, cap, size=16.5, y0=942, h=92, n=138)
    footer(fig, "Plans: hblend/score_M2b_s*_perwindow.npz (tau = B, P = P10, H) and hspec/score_hs2_M2b_s*_perwindow.npz (HS), indexed by eval window; GT = the eval's own context.  Open loop: the plan is scored against the logged human path; nothing feeds back.")
    return render(fig)


def _cell2(v):
    return f"{v['mean']:.2f}"


def plans_table_card():
    fig = new_fig()
    header(fig, "4  Better plans: ADE by horizon over ALL 2,399 eval windows (141 episodes)", tier="T1-class open loop  |  banked evals")
    hz = [1, 2, 3, 4, 5, 6]
    x0, cw = 380, 252
    y0 = 100
    for j, h in enumerate(hz):
        panel_box(fig, x0 + j * cw, y0, cw - 6, 46, ec="#9aa3af", fc="#eef1f5")
        tx(fig, x0 + j * cw + (cw - 6) / 2, y0 + 23, f"ADE @ {h} s  (m)", 15, INK, "bold", ha="center", va="center")
    rh = 62
    row = 0
    for sd, lab in ((S0, "seed 0"), (S1, "seed 1")):
        t = TB["hblend_hspec"][sd]["ade"]
        tx(fig, 24, y0 + 62 + row * rh + 4, lab, 19, INK, "bold")
        for arm in ("B", "H", "HS", "P10"):
            y = y0 + 54 + row * rh
            tx(fig, 150, y + 26, {"B": "B  (own plan)", "H": "H  (HBLEND)", "HS": "HS (HSPEC st.2)", "P10": "P10 (prior)"}[arm], 15, PLAN_COL[arm] if arm != "P10" else "#5b6470", "bold", va="center")
            for j, h in enumerate(hz):
                v = t[arm][f"ade{h}"]
                best = min(TB["hblend_hspec"][sd]["ade"][a_][f"ade{h}"]["mean"] for a_ in ("B", "H", "HS", "P10"))
                panel_box(fig, x0 + j * cw, y, cw - 6, rh - 5, ec="#e1e5ea", fc="#e1f2e8" if abs(v["mean"] - best) < 1e-9 else PANEL)
                tx(fig, x0 + j * cw + (cw - 6) / 2, y + 17, f"{v['mean']:.3f}", 17, INK, "bold", ha="center", va="center")
                tx(fig, x0 + j * cw + (cw - 6) / 2, y + 42, f"[{v['lo']:.3f}, {v['hi']:.3f}]", 10.2, MUTED, ha="center", va="center")
            row += 1
        row += 0.4
    pa = {sd: TB["hblend_hspec"][sd]["paired_ade6"] for sd in SEEDS}
    tx(fig, 24, 770, wrap(
        f"Paired ADE6 deltas (m, paired episode-cluster bootstrap):  H - B  s0 {pa[S0]['H-B']['mean']:+.3f} [{pa[S0]['H-B']['lo']:+.3f}, {pa[S0]['H-B']['hi']:+.3f}],  s1 {pa[S1]['H-B']['mean']:+.3f} [{pa[S1]['H-B']['lo']:+.3f}, {pa[S1]['H-B']['hi']:+.3f}];   "
        f"H - P10  s0 {pa[S0]['H-P10']['mean']:+.3f} [{pa[S0]['H-P10']['lo']:+.3f}, {pa[S0]['H-P10']['hi']:+.3f}],  s1 {pa[S1]['H-P10']['mean']:+.3f} [{pa[S1]['H-P10']['lo']:+.3f}, {pa[S1]['H-P10']['hi']:+.3f}];   "
        f"HS - B  s0 {pa[S0]['HS-B']['mean']:+.3f} [{pa[S0]['HS-B']['lo']:+.3f}, {pa[S0]['HS-B']['hi']:+.3f}],  s1 {pa[S1]['HS-B']['mean']:+.3f} [{pa[S1]['HS-B']['lo']:+.3f}, {pa[S1]['HS-B']['hi']:+.3f}].  All separated from 0.", 190), 12.5, INK, ls=1.45)
    ad = {sd: TB["hblend_hspec"][sd]["ade"] for sd in SEEDS}
    hb = all(ad[sd]["H"][f"ade{h}"]["mean"] < ad[sd]["B"][f"ade{h}"]["mean"] for sd in SEEDS for h in range(1, 7))
    hp = all(ad[sd]["H"][f"ade{h}"]["mean"] < ad[sd]["P10"][f"ade{h}"]["mean"] for sd in SEEDS for h in range(1, 7))
    sb = all(ad[sd]["HS"][f"ade{h}"]["mean"] < ad[sd]["B"][f"ade{h}"]["mean"] for sd in SEEDS for h in range(1, 7))
    sp_from = {sd: min(h for h in range(1, 7) if all(ad[sd]["HS"][f"ade{k}"]["mean"] < ad[sd]["P10"][f"ade{k}"]["mean"] for k in range(h, 7))) for sd in SEEDS}
    cap = (("H has a lower mean ADE than B and than the kinematic prior P10 at every horizon 1-6 s, at both seeds; " if hb and hp else "") +
           ("HS is below B at every horizon" if sb else "") + f" and below P10 from {max(sp_from.values())} s on (at seed 0 it is level with P10 at 1 s).  "
           f"H minus B at 6 s: seed 0 {pa[S0]['H-B']['mean']:+.2f} m, seed 1 {pa[S1]['H-B']['mean']:+.2f} m (B's own ADE6 differs by 0.12 m between the seeds).  Green = lowest mean in that cell.  Cell = mean and 95 % interval.")
    caption(fig, cap, size=16.5, y0=850, h=150, n=145)
    footer(fig, "Source: hblend/score_M2b_s{0,1}.json::single (B, H, P10) and hspec/score_hs2_M2b_s{0,1}.json::single (HS, stage 2).  Estimator: single-arm / paired episode-cluster bootstrap, B = 2,000, 141 episodes, seed 0 (taniteval/ci.py): another draw of EPISODES; "
                "the two training seeds side by side (|ADE6 s0 - s1| = 0.121 m for B); inference deterministic.  Tier T1-class open loop (own held outputs; not closed loop).")
    return render(fig)
def plans_gap_card():
    fig = new_fig()
    header(fig, "4  Better plans: what is NOT yet better", tier="T1-class open loop  |  banked evals")
    t0, t1 = TB["hblend_hspec"][S0], TB["hblend_hspec"][S1]

    def pr(d):
        return f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]" + (" separated" if d["separated"] else " not separated")
    # left: acceleration
    panel_box(fig, 24, 100, 930, 700, ec="#0072B2", lw=2.0)
    tx(fig, 44, 114, "1   Acceleration is still worse than the kinematic prior", 21, "#0072B2", "bold")
    ac0, ac1 = t0["accel"], t1["accel"]
    tx(fig, 44, 170, wrap(f"Longitudinal acceleration error (accel MAE, m/s^2; lower is better)\n\n"
                          f"             seed 0     seed 1\n"
                          f"B            {ac0['single']['B']['mean']:.3f}      {ac1['single']['B']['mean']:.3f}\n"
                          f"H            {ac0['single']['H']['mean']:.3f}      {ac1['single']['H']['mean']:.3f}\n"
                          f"HS           {ac0['single']['HS']['mean']:.3f}      {ac1['single']['HS']['mean']:.3f}\n"
                          f"P10 (prior)  {ac0['single']['P10']['mean']:.3f}      {ac1['single']['P10']['mean']:.3f}", 70), 16, INK, mono=True, ls=1.45)
    tx(fig, 44, 450, wrap(f"H - P10:   s0 {pr(ac0['H-P10'])};   s1 {pr(ac1['H-P10'])}\n"
                          f"HS - P10:  s0 {pr(ac0['HS-P10'])};   s1 {pr(ac1['HS-P10'])}\n\n"
                          "H and HS are far better than B (H - B: s0 %+.3f, s1 %+.3f), but at seed 0 neither reaches the prior's smoothness; HS closes most of the gap "
                          "(+0.104 -> +0.026 at s0) and is not separated from P10 at seed 1." % (ac0["H-B"]["mean"], ac1["H-B"]["mean"]), 78), 13.2, INK, ls=1.45)
    # right: consistency
    panel_box(fig, 966, 100, 930, 700, ec="#D55E00", lw=2.0)
    tx(fig, 986, 114, "2   The consistency cost", 21, "#D55E00", "bold")
    c0, c1 = t0["consistency_any"], t1["consistency_any"]
    tx(fig, 986, 170, wrap("CONSISTENCY(a): share of windows whose plan violates a GT-derived constraint (speed limit, stop, turn, lead gap; lower is better)\n\n"
                           "             seed 0     seed 1\n"
                           f"B            {c0['rate']['B']:.3f}      {c1['rate']['B']:.3f}\n"
                           f"H            {c0['rate']['H']:.3f}      {c1['rate']['H']:.3f}\n"
                           f"HS           {c0['rate']['HS']:.3f}      {c1['rate']['HS']:.3f}\n"
                           f"P10 (prior)  {c0['rate']['P10']:.3f}      {c1['rate']['P10']:.3f}", 68), 15, INK, mono=True, ls=1.4)
    tx(fig, 986, 470, wrap(f"H - B:   s0 {pr(c0['H-B'])};   s1 {pr(c1['H-B'])}\nHS - B:  s0 {pr(c0['HS-B'])};   s1 {pr(c1['HS-B'])}\n\n"
                           "Shrinking B's residual toward the prior gives back part of what B learned from its consistency penalties: ADE and acceleration improve, "
                           "constraint violations get WORSE (separated at both seeds), though still no worse than P10.  HS does not change that (HS - H not separated).", 76), 13.2, INK, ls=1.45)
    caption(fig, "Honest status: the position gain is real and survives both seeds; the plan is still rougher than the prior in acceleration (seed 0), and the gain is paid for in constraint violations.  "
                 "The committed next step is a second-difference penalty in the training loss, which needs a launch gate and box-2 time.",
            size=16.5, y0=850, h=130, n=140)
    footer(fig, "Source: hblend/score_M2b_s{0,1}.json (paired_H_minus_P10, consistency_beside), hspec/score_hs2_M2b_s{0,1}.json (paired, consistency); paired episode-cluster bootstrap, B = 2,000, 141 episodes (another draw of EPISODES); two seeds side by side.  "
                "Tier T1-class open loop (own held outputs).  Quoted as the source documents state them.")
    return render(fig)


# ================================================================================================ 5. strategic layer at its own tick
def s_frame(name, p, seed=S0):
    sel = WINDOWS[name]
    t, a_s = sel["t"], int(cfd(seed, name, "a_s"))
    fig = new_fig()
    header(fig, f"5  Strategic layer at its own tick  |  {WLAB[name]}", tier="T1-class open loop  |  DIAGNOSTIC: fresh ages are not the deployed schedule", sub=f"arm M2b, seed 0  |  {common_info(name)}")
    camera(fig, (24, 112, 640, 256), fr(name, t - a_s), title=f"the frame S last read (t - {a_s} steps = {0.2 * a_s:.1f} s ago)", ec=LV_COL["S"])
    camera(fig, (24, 400, 640, 256), fr(name, t), title="the frame at NOW")
    held, fresh, gt, ok = cfd(seed, name, "own_ckpt_now"), cfd(seed, name, "fresh_ckpt_now"), cfd(seed, name, "rc"), cfd(seed, name, "rc_ok")
    travel = float(np.hypot(*(cfd(seed, name, "pose_now")[:2] - cfd(seed, name, "pose_s")[:2])))
    eh = np.linalg.norm(held[:, :2] - gt[:, :2], axis=-1)
    ef = np.linalg.norm(fresh[:, :2] - gt[:, :2], axis=-1)
    tx(fig, 24, 676, f"held tick: S ticked {0.2 * a_s:.1f} s ago, the ego has driven {travel:.1f} m since", 12.5, LV_COL["S"], "bold")
    tx(fig, 24, 706, "position error vs GT checkpoint (m)         30 m    50 m    80 m", 11.2, MUTED, mono=True)
    tx(fig, 24, 730, f"S held tick (banked eval)              {eh[0]:6.1f}  {eh[1]:6.1f}  {eh[2]:6.1f}", 12.2, LV_COL["S"], "bold", mono=True)
    tx(fig, 24, 756, f"S fresh (ticked at NOW, diagnostic)     {ef[0]:6.1f}  {ef[1]:6.1f}  {ef[2]:6.1f}", 12.2, "#7a0a5c", "bold", mono=True)
    tx(fig, 24, 790, "GT checkpoint = the v9 route-centre anchor at 30 / 50 / 80 m arc length (a label).", 10.2, MUTED)
    tx(fig, 24, 810, "Part of the held-vs-fresh gap is DEFINITIONAL: the checkpoint is predicted in the frame of S's tick,\nthe label lives in the NOW frame, so the metric pays for the distance driven since the tick.", 10.2, MUTED, ls=1.35)
    ext = ext_for(name, mode="strat")
    ax = bev_axes(fig, (724, 100, 756, 826), ext)
    plan(ax, cfd(seed, name, "gt"), 1.0, color="#1b1f24", lw=2.2, label="GT 6 s path", z=2, end=False, dots=False)
    gck = gt.copy()
    for k in range(3):
        if ok[k]:
            ax.plot(-gck[k, 1], gck[k, 0], "*", ms=17, mfc="#1b1f24", mec="white", mew=1.0, zorder=8, label="GT checkpoint (30/50/80 m)" if k == 0 else None)
    ckpts(ax, held, LV_COL["S"], marker="D", filled=True, ms=10, label=f"S, held tick ({0.2 * a_s:.1f} s old)", z=6)
    ckpts(ax, fresh, "#7a0a5c", marker="D", filled=False, ms=11, label="S, fresh (diagnostic)", z=7)
    for k in range(3):
        if ok[k]:
            ax.plot([-gck[k, 1], -held[k, 1]], [gck[k, 0], held[k, 0]], "-", color=LV_COL["S"], lw=1.2, alpha=0.6, zorder=4)
            ax.plot([-gck[k, 1], -fresh[k, 1]], [gck[k, 0], fresh[k, 0]], "-", color="#7a0a5c", lw=1.2, alpha=0.6, zorder=4)
    legend(ax, "upper left", 9.0)
    tx(fig, 1500, 100, "what to look at", 13.5, INK, "bold")
    tx(fig, 1500, 136, wrap("Filled purple diamonds: S's checkpoints as the banked eval scores them (S ticks about once a second, the plan is read at NOW).\n\n"
                            "Open diamonds: the same S, ticked at NOW.\n\n"
                            "Stars: the GT checkpoints.\n\nThin lines join each checkpoint to its GT star.", 30), 12.2, INK, ls=1.4)
    closer = [bool(ef[k] < eh[k]) for k in range(3)]
    cap = (f"Held tick vs fresh vs GT.  S's last read is {0.2 * a_s:.1f} s old and the ego has moved {travel:.1f} m since.  Position error of the 30 / 50 / 80 m checkpoints (seed 0): held {eh[0]:.1f} / {eh[1]:.1f} / {eh[2]:.1f} m, fresh {ef[0]:.1f} / {ef[1]:.1f} / {ef[2]:.1f} m"
           + ("; fresh is closer at all three." if all(closer) else ("; in THIS window fresh is closer at " + (", ".join(str(m) for m, c in zip((30, 50, 80), closer) if c) or "none") + " m only (one window is an example; the eval-wide card is the evidence)."))
           + "  Seed 1: closing card of this segment.")
    caption(fig, cap, size=16, y0=938, h=102, n=146)
    footer(fig, "Held = the banked eval's cycle ages (--ages cycle); fresh = --ages fresh (a_T = a_S = 0).  GT = X['rc'], the v9 RC-A labels in the NOW frame.  CPU re-runs of the banked M2b_s0; held / fresh checkpoints match the banked evals to <= 2e-5 (CONTROLS_V2.json).")
    return render(fig)


def s_card():
    fig = new_fig()
    header(fig, "5  Strategic layer at its own tick: the eval-wide numbers", tier="T1-class open loop  |  DIAGNOSTIC: fresh ages are not the deployed schedule")
    sf = TB["sfresh"]
    lv = {sd: sf[sd]["levels_pos_err_m"] for sd in ("s0", "s1")}
    # left table: level at 50 m
    def lvl(sd, k):
        d = lv[sd][k]
        d = d.get("@50", d.get("50", d)) if isinstance(d, dict) else d
        return d
    panel_box(fig, 24, 100, 900, 520, ec=LV_COL["S"], lw=2.0)
    tx(fig, 44, 114, "S checkpoint position error at 50 m (m), all windows", 18, LV_COL["S"], "bold")
    rows = [("banked eval ages (S held, 0-1.2 s old)", "cycle_banked"), ("S fresh (ticks at NOW, diagnostic)", "fresh"), ("nav-only kinematic baseline", "navonly_baseline")]
    tx(fig, 470, 168, "seed 0 [95 % CI]", 12.5, MUTED, mono=True)
    tx(fig, 700, 168, "seed 1 [95 % CI]", 12.5, MUTED, mono=True)
    for i, (lab, key) in enumerate(rows):
        v0, v1 = lvl("s0", key), lvl("s1", key)
        def f(v):
            return f"{v['mean']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" if isinstance(v, dict) and "mean" in v else str(v)
        tx(fig, 44, 200 + i * 62, lab, 13.5, INK, "bold")
        tx(fig, 470, 200 + i * 62, f(v0), 13.5, INK, mono=True)
        tx(fig, 700, 200 + i * 62, f(v1), 13.5, INK, mono=True)
    ql0, ql1 = sf["s0"]["Q_LAYER_50m"], sf["s1"]["Q_LAYER_50m"]
    bl0, bl1 = sf["s0"]["banked_cycle_minus_navonly_50"], sf["s1"]["banked_cycle_minus_navonly_50"]
    tx(fig, 44, 400, "paired differences at 50 m (m):", 14, INK, "bold")
    tx(fig, 44, 436, wrap(f"fresh S  -  nav-only baseline:   s0 {ql0['delta']:+.2f} [{ql0['lo']:+.2f}, {ql0['hi']:+.2f}]     s1 {ql1['delta']:+.2f} [{ql1['lo']:+.2f}, {ql1['hi']:+.2f}]   (S beats the baseline, separated)\n"
                          f"banked cycle S  -  nav-only baseline:   s0 {bl0['delta']:+.2f}     s1 {bl1['delta']:+.2f}   (S held at the banked ages loses to the baseline)", 100), 13, INK, ls=1.5)
    # right: the two windows, both seeds
    panel_box(fig, 944, 100, 952, 520, ec="#9aa3af", lw=1.4)
    tx(fig, 964, 114, "The two windows of this segment, 50 m checkpoint error (m)", 16, INK, "bold")
    tx(fig, 964, 160, "window                    seed   held    fresh   (S age, ego travel)", 12.5, MUTED, mono=True)
    y = 192
    for name in ("JUNCTION", "NO-TURN"):
        for sd in SEEDS:
            held, fresh, gt = cfd(sd, name, "own_ckpt_now"), cfd(sd, name, "fresh_ckpt_now"), cfd(sd, name, "rc")
            eh, ef = np.linalg.norm(held[1, :2] - gt[1, :2]), np.linalg.norm(fresh[1, :2] - gt[1, :2])
            trv = float(np.hypot(*(cfd(sd, name, "pose_now")[:2] - cfd(sd, name, "pose_s")[:2])))
            tx(fig, 964, y, f"{WSHORT[name]:<20s}  s{sd[-1]}   {eh:5.1f}   {ef:5.1f}    ({0.2 * int(cfd(sd, name, 'a_s')):.1f} s, {trv:.1f} m)", 13, INK, mono=True)
            y += 34
        y += 14
    tx(fig, 964, y - 8, "Two windows are examples; the eval-wide numbers (left, and the chart below) are the evidence.", 11.5, MUTED)
    cx = axpx(fig, 1010, 440, 840, 160, W, H)
    cx.set_facecolor(PANEL)
    for sd, ls_, lab in (("s0", "-", "seed 0"), ("s1", "--", "seed 1")):
        prof = [SFX[sd]["age_profile_50m"][str(i)] for i in range(7)]
        ag = [r_["age_s"] for r_ in prof]
        cx.plot(ag, [r_["cycle_banked_S_pos_err_50m"] for r_ in prof], ls_, color=LV_COL["S"], lw=2.6, marker="o", ms=4, label=f"held (banked eval), {lab}")
        cx.plot(ag, [r_["fresh_S_pos_err_50m"] for r_ in prof], ls_, color="#7a0a5c", lw=2.6, marker="s", ms=4, label=f"fresh (ticked at NOW), {lab}")
    cx.set_xlabel("age of S's last tick (s)", fontsize=9)
    cx.set_ylabel("50 m error (m)", fontsize=9)
    cx.grid(color="#e8ebef")
    cx.tick_params(labelsize=8.5)
    cx.legend(loc="upper left", fontsize=8.2, ncol=2, framealpha=0.95)
    cx.set_title("S's 50 m error by the age of its last tick (all windows; error of the held checkpoint grows with the distance driven)", fontsize=9.2, loc="left")
    qs0, qs1 = sf["s0"]["Q_STALE_50m"], sf["s1"]["Q_STALE_50m"]
    caption(fig, f"S at its OWN tick is accurate: its 50 m checkpoint error falls from 8.0 m to 2.3 m (seed 0; 7.9 -> 2.2 m seed 1) and beats the nav-only baseline by {-ql0['delta']:.2f} m [{-ql0['hi']:.2f}, {-ql0['lo']:.2f}] / {-ql1['delta']:.2f} m "
                 f"[{-ql1['hi']:.2f}, {-ql1['lo']:.2f}].  The banked eval scores S at held ages against NOW-frame labels, which is partly definitional: it pays for the distance driven since the tick.  "
                 f"Deployed behaviour is NOT shown; the deployed-schedule cost (fresh - held: {qs0['delta']:+.2f} / {qs1['delta']:+.2f} m) is the next strategic lever.",
            size=16, y0=720, h=170, n=146)
    tx(fig, 24, 910, wrap("Strategic FAMILY: PAUSED by the PI ruling of 2026-09-19 (PI_DECISION_QUEUE.md item 25).  S appears here only as a diagnostic of its checkpoints; this is not a strategic-family pass and does not lift the pause.", 175), 12.5, "#b30000", "bold")
    footer(fig, "Source: sfresh/SFRESH_RESULT.md and sfresh/sfresh_numbers.json (PREREG_V8F_SFRESH, tip 583cfc02).  Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes (another draw of EPISODES); training seeds side by side; deterministic inference.  "
                "n = 2,319 windows with a valid 50 m label.  Tier T1-class open loop, DIAGNOSTIC.")
    return render(fig)


# ================================================================================================ 6. closing
def closing_card():
    fig = new_fig(DARK)
    tx(fig, 70, 50, "What this video measured, and what it did not", 32, "white", "bold")
    boxes = [
        ("MEASURED", "#009E73", "CPU re-runs of the banked M2b checkpoints, training seeds 0 and 1, on the 2,399-window / 141-episode eval.\n\n"
                               "Four windows declared before any rendering; every panel seed 0, seed 1 in the cards.\n\n"
                               "Who listens to what (LISTEN grid); nav counterfactual at S / T / O (4 windows + all 2,399); B / H / HS / P10 plans (banked per-window arrays); S held vs fresh vs GT (SFRESH)."),
        ("TIER AND ESTIMATOR", "#0072B2", "T1-class open loop: the model reads its own held outputs; nothing it does feeds back.\n\n"
                                         "Interventions (donor or synthetic nav) are DIAGNOSTICS of dependence, not driving.\n\n"
                                         "Intervals = episode-cluster bootstrap (B = 2,000): they answer 'another draw of EPISODES' only.  Training variance = the two seeds side by side.  Inference is deterministic."),
        ("NOT SHOWN", "#D55E00", "Closed-loop driving.  External benchmarks (NAVSIM and others).\n\n"
                                "Whether a synthetic command is right for the road: a TURN command on a straight road is an intervention.\n\n"
                                "The STRATEGIC family result: paused by the PI ruling of 2026-09-19 (item 25).\n\n"
                                "Still open: acceleration vs the prior at seed 0; the consistency cost of H / HS; the deployed S schedule."),
    ]
    for i, (ttl, col, body) in enumerate(boxes):
        x = 70 + i * 600
        ax = axpx(fig, x, 130, 560, 560, W, H)
        ax.set_facecolor("#18202a"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(col); s.set_linewidth(2.6)
        tx(fig, x + 20, 146, ttl, 20, col, "bold")
        tx(fig, x + 20, 190, wrap(body, 41), 15.5, "#e8ecf2", ls=1.4)
    tx(fig, 70, 722, "Artifacts (dev box, nothing staged or pushed):", 15, "white", "bold")
    paths = ("scratchpad/viz2/:  v8f_evidence_v2.mp4,  SELECTION_V2.md (pre-declaration),  CONTROLS_V2.json,  VIDEO_NOTES.md (every number with its artifact path),  qa_*.png,  data/ (cf_windows.npz, cf_evalwide.json, plans_pack.npz, tables.json),  *.py\n"
             "inputs: listen/LISTEN_RESULT.md + fig_listen_grid.png,  hblend/ + hspec/ (per-window plans),  sfresh/ (S held vs fresh),  viz/ (v1 frames, checkpoints, controls),  model code tip 9a6551d5")
    tx(fig, 70, 756, wrap(paths, 215), 11, "#aeb7c4", ls=1.5)
    tx(fig, 70, 1010, "Registered claims are not updated by this video: it presents banked results and CPU re-reads; landing and the GOALS_AND_CLAIMS register are the Master Mind's integration step.", 11.5, "#7f8a99")
    return render(fig)


SCENES = {k: v for k, v in globals().items() if callable(v) and not k.startswith("_") and k in (
    "title_card", "listen_frame", "qnav_frame", "method_card", "cf_all", "cf_bcd", "cf_four", "cf_summary", "plans_frame", "plans_table_card", "plans_gap_card", "s_frame", "s_card", "closing_card")}
