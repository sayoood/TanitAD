"""v3_scenes.py -- every scene (card or animated frame) of the v3 video (v2_scenes.py + HACC on the plans, the new dominance segment, strategic family IN FORCE).
Each scene is a function returning an RGB uint8 [1080, 1920, 3]."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_lib import *  # noqa: E402,F401,F403
from v3_draw import *  # noqa: E402,F401,F403
from matplotlib.patches import Polygon  # noqa: E402

CF = np.load(f"{OUT}/data/cf_windows.npz")
PK = np.load(f"{OUT}/data/plans_pack3.npz")
TB = json.load(open(f"{OUT}/data/tables.json"))
EW = json.load(open(f"{OUT}/data/cf_evalwide.json"))
SFX = json.load(open(f"{SC}/sfresh/sfresh_extra.json"))
PLANS5 = ("P10", "B", "H", "HS", "HACC")
S0, S1 = SEEDS
WLAB = {"JUNCTION": "JUNCTION  (median of its stratum)", "CURVE": "CURVE  (median of its stratum)", "NO-TURN": "NO-TURN  (median of its stratum)",
        "FAILURE(worst)": "FAILURE CASE  (worst of all 2,399 windows)"}
WSHORT = {"JUNCTION": "JUNCTION", "CURVE": "CURVE", "NO-TURN": "NO-TURN", "FAILURE(worst)": "FAILURE (worst)"}
CFG_LAB = {"a_all": "(a) ALL three layers", "b_O": "(b) O only", "c_T": "(c) T only", "d_S": "(d) S only"}
DESC = ("(Only the nav input of the named layer(s) is replaced; everything else is the window's own; the change flows down S -> T -> O through the model's own chain.  "
        "Plans in this segment are the model's own plan B, BEFORE the HACC post-map.)")
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
        return fit_extent([cfd(seed, name, "gt")] + [PK[f"{seed}/{name}/{a}"] for a in ("B", "P10", "H", "HS", "HACC")])
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
    tx(fig, 90, 124, "what each layer depends on, what the planner outputs, and the HACC post-map", 25, "#e8ecf2")
    tx(fig, 90, 176, "arm M2b, two training seeds (s0, s1)  |  banked checkpoints and banked per-window plans, CPU re-read  |  T1-class open loop (NOT closed loop); interventions are diagnostics", 13.5, "#aeb7c4")
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
    tx(fig, 90, 530, "Five questions", 24, "white", "bold")   # v4: was "Four questions"
    qs = ["1   Who listens to what?  Which input does each layer actually depend on (causal, one input swapped at a time)?",
          "2   Does the trajectory follow the nav command, and through which layer?  (nav injected at one layer at a time)",
          "3   What does the planner's output look like against the human path, the kinematic prior P10 and the earlier blends?",
          "4   NEW in v3:  does HACC, a zero-training per-step blend of the model's plan and P10, dominate P10 at both training seeds, and where does the gain come from?",
          "5   NEW in v4 (chapter 7):  T1-kin, the model run on its OWN executed motion, replanning every 2 s (self-action OPEN loop, NOT closed loop): does HACC still beat the floor run the same way, and which gap stays open?"]
    for i, q in enumerate(qs):                                                  # v4: five questions; y positions compacted to make room (v3: 578 + 50 i)
        tx(fig, 90, (578, 622, 666, 710, 774)[i], wrap(q, 118), 18.5, "#e8ecf2" if i < 3 else "#ffd9b3", "normal" if i < 3 else "bold")
    tx(fig, 90, 842, wrap("Four windows were declared before anything was rendered (SELECTION_V2.md, unchanged): JUNCTION #1131, CURVE #654, NO-TURN #763, FAILURE #2158.  Every window panel is seed 0; seed 1 sits in a summary card.", 150), 13.5, "#aeb7c4")
    tx(fig, 90, 894, wrap("Tier of chapters 1-6: T1-class open loop (own held outputs; NOT closed loop).  The model reads its own held outputs; nothing it does feeds back into the scene.  Interventions replace an input: they show dependence, not skill.", 150), 13.5, "#aeb7c4")
    tx(fig, 90, 948, "Baseline: P10 is the programme's own kinematic floor (it extrapolates ego motion and ignores the scene).  It is not the state of the art, and no external benchmark is shown.", 13.5, "#ffd9b3")
    tx(fig, 90, 1010, "Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes (answers 'another draw of EPISODES' only); training variance = the two seeds side by side; inference is deterministic.", 11.5, "#7f8a99")
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
    footer(fig, "SELECTION_V2.md section 2 (pre-declared before any render).  Plans in this segment are the model's own plan B, BEFORE the HACC post-map.  Command vectors in the model's nav encoding [FOLLOW, TURN_L, TURN_R one-hot, d_next/100, args_valid].  Layers are run through the model's own run_s / run_t / run_o (tip 9a6551d5).")
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
    footer(fig, "Artifacts: viz3/data/cf_windows.npz (a byte-identical copy of viz2's; 4 windows, both seeds), viz3/data/cf_evalwide.json (a copy; 2,399 windows; episode-cluster bootstrap B = 2,000, 141 episodes: another draw of EPISODES; the two training seeds side by side; inference deterministic).  "
                "Tier T1-class open loop; intervention on the model's input, not driving.")
    return render(fig)


# ================================================================================================ 4. better plans (HACC drawn beside GT / P10 / B / H, HS kept thin)
TIER_PL = "T1-class open loop (own held outputs; NOT closed loop)"
PLAN_ORDER = ("P10", "B", "H", "HS", "HACC")
SD = {S0: "s0", S1: "s1"}


def _horizon_inset(fig, box, name, seed):
    ax = axpx(fig, *box, W, H)
    ax.set_facecolor(PANEL)
    t = 0.2 * (np.arange(30) + 1)
    for arm in PLAN_ORDER:
        kw = dict(color=PLAN_COL[arm], label=arm)
        if arm == "P10":
            kw.update(ls="--", lw=2.2)
        elif arm == "HS":
            kw.update(ls=":", lw=1.6, alpha=0.75)
        elif arm == "HACC":
            kw.update(ls="-", lw=3.2, zorder=6)
        else:
            kw.update(ls="-", lw=2.3)
        ax.plot(t, PK[f"{seed}/{name}/ade_h_{arm}"], **kw)
    ax.set_xlabel("horizon (s)", fontsize=9.5)
    ax.set_ylabel("distance to GT (m)", fontsize=9.5)
    ax.set_xlim(0.2, 6.0)
    ax.set_ylim(bottom=0)
    ax.grid(color="#e8ebef")
    ax.tick_params(labelsize=9)
    ax.legend(loc="upper left", fontsize=9.5, ncol=3, framealpha=0.95)
    ax.set_title("error of each plan vs GT, by horizon (this window)", fontsize=10.5, fontweight="bold", loc="left")


def plans_frame(name, p, seed=S0):
    sel = WINDOWS[name]
    fig = new_fig()
    header(fig, f"4  Better plans, now with HACC  |  {WLAB[name]}", tier=TIER_PL, sub=f"arm M2b, seed 0  |  {common_info(name)}")
    camera(fig, (24, 112, 640, 256), fr(name, sel["t"]), title="front camera at NOW")
    _horizon_inset(fig, (64, 420, 560, 290), name, seed)
    # per-window ADE6 table, both seeds
    tx(fig, 24, 758, "ADE6 of this window (m)", 12, INK, "bold")
    tx(fig, 24, 782, "plan         seed 0     seed 1", 11.5, MUTED, mono=True)
    for i, a in enumerate(PLAN_ORDER):
        tx(fig, 24, 804 + i * 23, f"{a:<10s}{PK[f'{S0}/{name}/ade6_{a}']:>8.2f}   {PK[f'{S1}/{name}/ade6_{a}']:>8.2f}", 12.6, PLAN_COL[a] if a != "P10" else "#5b6470", "bold", mono=True)
    ext = ext_for(name, mode="planset")
    ax = bev_axes(fig, (690, 100, 720, 826), ext)
    plan(ax, cfd(seed, name, "gt"), 1.0, color=PLAN_COL["GT"], lw=3.4, label=PLAN_LAB["GT"], z=3, end=False, ms=5)
    for i, a in enumerate(PLAN_ORDER):
        if a == "HS":
            plan(ax, PK[f"{seed}/{name}/{a}"], p, color=PLAN_COL[a], lw=1.7, ls=":", label=PLAN_LAB[a], z=4, dots=False, end=False, alpha=0.8)
        elif a == "HACC":
            plan(ax, PK[f"{seed}/{name}/{a}"], p, color=PLAN_COL[a], lw=3.6, label=PLAN_LAB[a], z=9, dots=True, ms=4.6)
        else:
            plan(ax, PK[f"{seed}/{name}/{a}"], p, color=PLAN_COL[a], lw=2.8 if a != "P10" else 2.2, ls="--" if a == "P10" else "-", label=PLAN_LAB[a], z=4 + i, dots=(a != "P10"), ms=4)
    legend(ax, "upper left", 8.6)
    tx(fig, 1430, 100, "what each line is", 13.5, INK, "bold")
    expl = ("GT\nthe human's path (the label)\n\nP10\nfrozen kinematic prior: extrapolates ego motion, does not look at the scene\n\n"
            "B\nM2b's own plan = P10 + a learned residual (what the banked eval scores)\n\nH\nB with the residual shrunk toward P10 by 30 x 2 fitted per-step weights (monotone)\n\n"
            "HS (thin, dotted)\nH with the residual rebuilt from 12 smooth modes: same ADE as H\n\n"
            "HACC (thick, new)\nthe same blend P + w (tau - P) as H, but w is fitted with an acceleration term and WITHOUT the monotone constraint; zero training")
    tx(fig, 1430, 136, wrap(expl, 34), 12, INK, ls=1.34)
    a6 = lambda sd, a: PK[f"{sd}/{name}/ade6_{a}"]
    dH = {sd: a6(sd, "HACC") - a6(sd, "H") for sd in SEEDS}
    dP = {sd: a6(sd, "HACC") - a6(sd, "P10") for sd in SEEDS}
    cap = (f"Banked plans of this window.  ADE6 (m), seed 0: P10 {a6(S0, 'P10'):.2f}, B {a6(S0, 'B'):.2f}, H {a6(S0, 'H'):.2f}, HS {a6(S0, 'HS'):.2f}, HACC {a6(S0, 'HACC'):.2f}; "
           f"seed 1: {a6(S1, 'P10'):.2f} / {a6(S1, 'B'):.2f} / {a6(S1, 'H'):.2f} / {a6(S1, 'HS'):.2f} / {a6(S1, 'HACC'):.2f}.  "
           f"HACC minus P10 on this window: {dP[S0]:+.2f} m (s0), {dP[S1]:+.2f} m (s1); HACC minus H: {dH[S0]:+.2f} / {dH[S1]:+.2f} m."
           + ("" if (dP[S0] < 0 and dP[S1] < 0) else "  HACC is NOT better than P10 on this window at " + " and ".join(f"seed {sd[-1]}" for sd in SEEDS if dP[sd] >= 0) + " (kept as it is).")
           + "  One window is an example; the eval-wide rows follow.")
    caption(fig, cap, size=15.6, y0=934, h=106, n=152)
    footer(fig, "Plans: hblend/score_M2b_s*_perwindow.npz (tau = B, P = P10, H), hspec/score_hs2_M2b_s*_perwindow.npz (HS), hacc/score_M2b_s*_perwindow.npz (HACC); GT = the eval's own context.  "
                "HACC = P + w (tau - P) from hacc/weights_HACC_M2b_s*.json (control in CONTROLS_V3.json).  Open loop: the plan is scored against the logged human path; nothing feeds back.")
    return render(fig)


def _ade_ptr(seed, arm, h, f):
    if arm == "HS":
        return "tables_v2", f"hblend_hspec/{seed}/ade/HS/ade{h}/{f}"
    return f"hacc_score_{SD[seed]}", f"single/{arm}/ade{h}/{f}"


def plans_table_card():
    fig = new_fig()
    header(fig, "4  Better plans: ADE by horizon over ALL 2,399 eval windows (141 episodes), with HACC", tier="T1-class open loop (own held outputs; NOT closed loop)")
    hz = [1, 2, 3, 4, 5, 6]
    x0, cw = 380, 252
    y0 = 96
    for j, h in enumerate(hz):
        panel_box(fig, x0 + j * cw, y0, cw - 6, 44, ec="#9aa3af", fc="#eef1f5")
        tx(fig, x0 + j * cw + (cw - 6) / 2, y0 + 22, f"ADE @ {h} s  (m)", 15, INK, "bold", ha="center", va="center")
    rh = 54
    row = 0
    arms = ("B", "H", "HS", "HACC", "P10")
    labs = {"B": "B  (own plan)", "H": "H  (HBLEND)", "HS": "HS (HSPEC st.2)", "HACC": "HACC (new)", "P10": "P10 (prior)"}
    for sd, lab in ((S0, "seed 0"), (S1, "seed 1")):
        tx(fig, 24, y0 + 52 + row * rh + 4, lab, 19, INK, "bold")
        for arm in arms:
            y = y0 + 48 + row * rh
            col = PLAN_COL[arm] if arm != "P10" else "#5b6470"
            tx(fig, 150, y + 24, labs[arm], 15, col, "bold", va="center")
            for j, h in enumerate(hz):
                best = min(NV(*_ade_ptr(sd, a_, h, "mean")) for a_ in arms)
                v = NV(*_ade_ptr(sd, arm, h, "mean"))
                panel_box(fig, x0 + j * cw, y, cw - 6, rh - 4, ec="#e1e5ea", fc="#e1f2e8" if abs(v - best) < 1e-9 else PANEL)
                tx(fig, x0 + j * cw + (cw - 6) / 2, y + 15, N(*_ade_ptr(sd, arm, h, "mean"), "{:.3f}"), 16.5, INK, "bold", ha="center", va="center")
                tx(fig, x0 + j * cw + (cw - 6) / 2, y + 37, f"[{N(*_ade_ptr(sd, arm, h, 'lo'), '{:.3f}')}, {N(*_ade_ptr(sd, arm, h, 'hi'), '{:.3f}')}]", 10.2, MUTED, ha="center", va="center")
            row += 1
        row += 0.35
    # paired ADE6 lines (HACC from the HACC score files; H-B, H-P10, HS-B from the v2 tables)
    def pr(src, ptr):
        m, lo, hi = (N(src, f"{ptr}/{k}", "{:+.3f}") for k in ("mean", "lo", "hi"))
        sep = bool(getp(J(src), f"{ptr}/separated"))
        return f"{m} [{lo}, {hi}]" + ("" if sep else " (n.s.)")

    def pr_v2(sd, pair):
        t = TB["hblend_hspec"][sd]["paired_ade6"][pair]
        return f"{t['mean']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]" + ("" if t["separated"] else " (n.s.)")
    ln = []
    ln.append("Paired ADE6 deltas (m, paired episode-cluster bootstrap; (n.s.) = interval includes 0; all others separated from 0):")
    ln.append("  H - B     s0 " + pr_v2(S0, "H-B") + "    s1 " + pr_v2(S1, "H-B") + "          H - P10    s0 " + pr_v2(S0, "H-P10") + "    s1 " + pr_v2(S1, "H-P10"))
    for lab, key in (("HACC - P10", "HACC_minus_P10"), ("HACC - H  ", "HACC_minus_H"), ("HACC - B  ", "HACC_minus_B")):
        ln.append(f"  {lab}  s0 " + pr("hacc_score_s0", f"paired/{key}/ade6") + "    s1 " + pr("hacc_score_s1", f"paired/{key}/ade6"))
    tx(fig, 24, 746, NL.join(ln), 12.2, INK, ls=1.42, mono=True)
    ad = {sd: {a: {h: NV(*_ade_ptr(sd, a, h, "mean")) for h in range(1, 7)} for a in arms} for sd in SEEDS}
    hp = all(ad[sd]["HACC"][h] < ad[sd]["P10"][h] for sd in SEEDS for h in range(1, 7))
    hb = all(ad[sd]["HACC"][h] < ad[sd]["B"][h] for sd in SEEDS for h in range(1, 7))
    dH = {sd: NV(f"hacc_score_{SD[sd]}", "paired/HACC_minus_H/ade6/mean") for sd in SEEDS}
    mx15 = max(abs(NV(f"hacc_score_{SD[sd]}", f"paired/HACC_minus_H/ade{h}/mean")) for sd in SEEDS for h in range(1, 6))
    cap = (("HACC has a lower mean ADE than the kinematic prior P10 at every horizon 1-6 s, at both seeds" if hp else "HACC is NOT below P10 at every horizon") + ("; it is also below B at every horizon.  " if hb else ".  ")
           + f"It is within {mx15:.3f} m of H at 1-5 s and sits slightly above it at 6 s: ADE6 HACC minus H {dH[S0]:+.3f} m (s0), {dH[S1]:+.3f} m (s1; separated, see the caveats card).  "
           "Green = lowest mean in that cell.  Cell = mean and 95 % interval.")
    caption(fig, cap, size=16.5, y0=900, h=110, n=145)
    footer(fig, "Source: hacc/score_M2b_s{0,1}.json::single (B, H, HACC, P10; the B / H / P10 values equal hblend/score_M2b_s*.json, control in CONTROLS_V3.json) and hspec/score_hs2_M2b_s{0,1}.json::single (HS, via viz3/data/tables.json).  "
                "Estimator: single-arm / paired episode-cluster bootstrap, B = 2,000, 141 episodes, seed 0 (taniteval/ci.py): another draw of EPISODES; the two training seeds side by side; inference deterministic.  Tier T1-class open loop (own held outputs; NOT closed loop).")
    return render(fig)


def plans_gap_card():
    fig = new_fig()
    header(fig, "4  Better plans: what HACC closes, and what is still NOT better", tier="T1-class open loop (own held outputs; NOT closed loop)")
    hs = {S0: "hacc_score_s0", S1: "hacc_score_s1"}

    def pr(sd, pair, metric, d=4):
        src = hs[sd]
        m, lo, hi = (N(src, f"paired/{pair}/{metric}/{k}", f"{{:+.{d}f}}") for k in ("mean", "lo", "hi"))
        return f"{m} [{lo}, {hi}]" + (" sep." if getp(J(src), f"paired/{pair}/{metric}/separated") else " n.s.")

    def cs(sd, pair, metric="viol_any"):
        src = hs[sd]
        m, lo, hi = (N(src, f"consistency/paired/{pair}/{metric}/{k}", "{:+.4f}") for k in ("mean", "lo", "hi"))
        return f"{m} [{lo}, {hi}]" + (" sep." if getp(J(src), f"consistency/paired/{pair}/{metric}/separated") else " n.s.")
    # left: closed
    panel_box(fig, 24, 96, 930, 744, ec="#0072B2", lw=2.0)
    tx(fig, 44, 110, "1   Accel and curvature gap to P10: closed", 20, "#0072B2", "bold")
    ac = lambda sd, a: N(hs[sd], f"single/{a}/fw_lon_accel_mae/mean", "{:.3f}")
    cv = lambda sd, a: N(hs[sd], f"single/{a}/fw_lat_curvature_mae_1pm/mean", "{:.4f}")
    tx(fig, 44, 156, wrap("accel MAE (m/s^2)                curvature MAE (1/m)\n"
                          "             seed 0   seed 1    seed 0   seed 1\n"
                          + NL.join(f"{lab:<10s}   {ac(S0, a)}    {ac(S1, a)}     {cv(S0, a)}   {cv(S1, a)}" for lab, a in (("B", "B"), ("H", "H"), ("HACC", "HACC"), ("P10 prior", "P10"))), 86), 15, INK, mono=True, ls=1.45)
    tx(fig, 44, 352, wrap("Paired (episode-cluster bootstrap, lower is better; sep. = interval excludes 0):\n\n"
                          "accel MAE, HACC - P10 (m/s^2):\n"
                          f"   s0 {pr(S0, 'HACC_minus_P10', 'fw_lon_accel_mae')}\n   s1 {pr(S1, 'HACC_minus_P10', 'fw_lon_accel_mae')}\n"
                          "accel MAE, HACC - H:\n"
                          f"   s0 {pr(S0, 'HACC_minus_H', 'fw_lon_accel_mae')}\n   s1 {pr(S1, 'HACC_minus_H', 'fw_lon_accel_mae')}\n"
                          "curvature MAE, HACC - P10 (1/m):\n"
                          f"   s0 {pr(S0, 'HACC_minus_P10', 'fw_lat_curvature_mae_1pm', 5)}\n   s1 {pr(S1, 'HACC_minus_P10', 'fw_lat_curvature_mae_1pm', 5)}\n"
                          "curvature MAE, HACC - H:\n"
                          f"   s0 {pr(S0, 'HACC_minus_H', 'fw_lat_curvature_mae_1pm', 5)}\n   s1 {pr(S1, 'HACC_minus_H', 'fw_lat_curvature_mae_1pm', 5)}", 86), 12.4, INK, mono=True, ls=1.38)
    tx(fig, 44, 770, wrap("Before HACC, H was separated-WORSE than P10 on the accel row (accel MAE, H - P10: "
                          f"{N('tables_v2', 'hblend_hspec/M2b_s0/accel/H-P10/mean', '{:+.3f}')} s0, {N('tables_v2', 'hblend_hspec/M2b_s1/accel/H-P10/mean', '{:+.3f}')} s1 m/s^2).  HACC is separated-better at both seeds.", 82), 13, INK, ls=1.4)
    # right: costs
    panel_box(fig, 966, 96, 930, 744, ec="#D55E00", lw=2.0)
    tx(fig, 986, 110, "2   What is NOT better, and what it costs", 20, "#D55E00", "bold")
    tx(fig, 986, 156, wrap("(a) vs the model's own plan B: the consistency cost remains.\n"
                           "CONSISTENCY(a) 'any' (share of windows violating a GT-derived constraint; lower is better):\n"
                           f"   HACC - B   s0 {cs(S0, 'HACC_minus_B')}\n              s1 {cs(S1, 'HACC_minus_B')}\n"
                           f"   HACC - P10 s0 {cs(S0, 'HACC_minus_P10')}\n              s1 {cs(S1, 'HACC_minus_P10')}\n\n"
                           "(b) vs H: a small loss on three rows.\n"
                           f"   cross-track MAE (m)  HACC - H   s0 {pr(S0, 'HACC_minus_H', 'fw_lat_cross_mae')}\n                                   s1 {pr(S1, 'HACC_minus_H', 'fw_lat_cross_mae')}\n"
                           f"   ADE6 (m)             HACC - H   s0 {pr(S0, 'HACC_minus_H', 'ade6')}\n                                   s1 {pr(S1, 'HACC_minus_H', 'ade6')}\n"
                           f"   FDE6 (m)             HACC - H   s0 {pr(S0, 'HACC_minus_H', 'fde6')}\n                                   s1 {pr(S1, 'HACC_minus_H', 'fde6')}\n\n"
                           "(c) vs P10: distance keeping (headway, time-gap) is NOT separated at either seed (next segment).\n"
                           "(d) at 5.2-5.8 s the accel error is still ABOVE P10 (HACC - P10 "
                           f"{N('hacc_accel', 'per_seed/M2b_s0/bands/5.2-5.8 s (cells 26-29)/HACC_minus_P10', '{:+.3f}')} s0, {N('hacc_accel', 'per_seed/M2b_s1/bands/5.2-5.8 s (cells 26-29)/HACC_minus_P10', '{:+.3f}')} s1 m/s^2; next segment).", 84), 11.8, INK, mono=True, ls=1.36)
    caption(fig, "Honest status: HACC meets the dominance read against the prior at both seeds (no row separated-worse; next segment) and closes the acceleration and curvature gap that H left.  "
                 "What it gives back is small against H (cross-track, FDE6, and at seed 1 ADE6) and the consistency cost against B stays.  Distance keeping is not separated from P10.",
            size=16.5, y0=866, h=130, n=140)
    footer(fig, "Source: hacc/score_M2b_s{0,1}.json (paired.HACC_minus_{P10,H}, consistency.paired.HACC_minus_{B,P10}, single); hacc/accel_profile.json (late-horizon band).  Paired episode-cluster bootstrap, B = 2,000, 141 episodes (another draw of EPISODES); two seeds side by side.  "
                "Tier T1-class open loop (own held outputs; NOT closed loop).  H - P10 (before HACC) from hspec/RESULT_V8F_HSPEC.md via viz3/data/tables.json.")
    return render(fig)


# ================================================================================================ 5. Dominance over the kinematic prior (NEW)
DOM_ROWS = [  # (label, READ.json key, decimals)
    ("ADE2 (m)", "ADE2", 4), ("ADE6 (m)", "ADE6", 4), ("accel MAE (m/s^2)", "accel MAE", 4), ("curvature MAE (1/m)", "curvature MAE", 5),
    ("speed MAE (m/s)", "speed MAE", 4), ("along MAE (m)", "along MAE", 4), ("cross-track MAE (m)", "cross-track MAE", 4), ("heading MAE (deg)", "heading MAE", 4),
    ("yaw-rate MAE (deg/s)", "yaw-rate MAE", 4), ("lateral decision agreement*", "plan-derived lateral decision agreement", 4),
    ("longitudinal decision agreement*", "plan-derived longitudinal decision agreement", 4), ("CONSISTENCY(a) 'any' (violation share)", "CONSISTENCY(a) 'any'", 4),
    ("DK headway min (m)", "DK headway min (m)", 4), ("DK time-gap min (s)", "DK time-gap min (s)", 4), ("DK LEAD-VIOL at-risk share", "DK LEAD-VIOL at-risk share", 4)]
HIGHER_BETTER = {"plan-derived lateral decision agreement", "plan-derived longitudinal decision agreement", "DK headway min (m)", "DK time-gap min (s)"}
GREEN, GREY = "#d8f0e3", "#e6e8ec"


def _pctr(seed, key, f):
    return f"per_seed/{seed}/PRIMARY_DOM/rows/{key}/{f}"


def dom_table(p=1.0):
    fig = new_fig()
    header(fig, "5  Dominance over the kinematic prior P10: HACC - P10 on the dominance row, both seeds", tier=TIER_PL, sub="eval-141: all 2,399 windows, 141 episodes  |  paired episode-cluster bootstrap, B = 2,000  |  outcome read from hacc/READ.json")
    ax = axpx(fig, 0, 0, W, H, W, H)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    X0, XR, XS = 24, 520, (690, 1290)
    y0, rh = 92, 45
    ax.add_patch(Rectangle((14, y0), W - 28, 40, fc="#eef1f5", ec="#9aa3af", lw=1.0))
    ax.text(X0, y0 + 20, "metric  (lower is better unless marked)", fontsize=12, fontweight="bold", va="center", color=INK)
    ax.text(XR, y0 + 20, "requirement", fontsize=12, fontweight="bold", va="center", color=INK)
    for sd, bx in zip(SEEDS, XS):
        ax.text(bx, y0 + 20, f"seed {sd[-1]}:  HACC - P10, mean [95 % CI]", fontsize=12, fontweight="bold", va="center", color=INK)
    nshow = int(np.ceil(p * len(DOM_ROWS) - 1e-9))
    for i, (lab, key, dec) in enumerate(DOM_ROWS):
        if i >= nshow:
            break
        y = y0 + 44 + i * rh
        ax.add_patch(Rectangle((14, y), W - 28, rh - 3, fc="#ffffff" if i % 2 == 0 else "#f6f7f9", ec="#e1e5ea", lw=0.8))
        hb = " (up = better)" if key in HIGHER_BETTER else ""
        ax.text(X0, y + (rh - 3) / 2, lab + hb, fontsize=11.6, va="center", color=INK, fontweight="bold" if i < 4 else "normal")
        ax.text(XR, y + (rh - 3) / 2, getp(J("hacc_read"), _pctr("M2b_s0", key, "need")), fontsize=10.8, va="center", color=MUTED)
        scl = max(abs(NV("hacc_read", _pctr(sd, key, f))) for sd in SEEDS for f in ("lo", "hi"))
        for sd, bx in zip(SEEDS, XS):
            m, lo, hi = (N("hacc_read", _pctr(sd, key, f), f"{{:+.{dec}f}}") for f in ("mean", "lo", "hi"))
            sep = bool(getp(J("hacc_read"), _pctr(sd, key, "separated")))
            better = bool(getp(J("hacc_read"), _pctr(sd, key, "better")))
            worse = bool(getp(J("hacc_read"), _pctr(sd, key, "worse")))
            ax.text(bx, y + (rh - 3) / 2, f"{m} [{lo}, {hi}]", fontsize=11.2, va="center", color=INK, family="DejaVu Sans Mono")
            chip = ("better, sep." if (sep and better) else ("WORSE, sep." if (sep and worse) else "not sep."))
            ax.add_patch(Rectangle((bx + 300, y + 8), 96, rh - 19, fc=(GREEN if (sep and better) else ("#f6c9c9" if (sep and worse) else GREY)), ec="#9aa3af", lw=0.8))
            ax.text(bx + 348, y + (rh - 3) / 2, chip, fontsize=9.8, va="center", ha="center", color=INK, fontweight="bold")
            # CI bar: zero at the centre, the row's own scale shared by the two seeds
            cx, hw = bx + 506, 84
            fx = lambda v: cx + hw * v / scl
            col = "#009E73" if (sep and better) else "#8a8f98"
            ax.plot([cx, cx], [y + 6, y + rh - 9], color="#1b1f24", lw=1.0)
            ax.plot([fx(NV("hacc_read", _pctr(sd, key, "lo"))), fx(NV("hacc_read", _pctr(sd, key, "hi")))], [y + (rh - 3) / 2] * 2, color=col, lw=3.4, solid_capstyle="butt")
            ax.plot([fx(NV("hacc_read", _pctr(sd, key, "mean")))], [y + (rh - 3) / 2], "o", color=col, ms=6, mec="white")
    if p >= 0.99:
        nb = lambda sd, a, f: N("hacc_dk", f"pairs/HACC_{SD[sd]}__minus__P10/{a}/{f}", "{:.0f}")
        nsb = {sd: (nb(sd, "dk/headway_min_m", "n_used"), nb(sd, "dk/headway_min_m", "n_episodes"), nb(sd, "dk/time_gap_min_s", "n_used"), nb(sd, "dk/time_gap_min_s", "n_episodes"),
                    nb(sd, "lead_viol/at_risk", "n_at_risk"), nb(sd, "lead_viol/at_risk", "n_episodes")) for sd in SEEDS}
        yb = y0 + 44 + len(DOM_ROWS) * rh + 6
        tx(fig, 24, yb, wrap("* decision agreements are derived from the plan.  Bar = 95 % CI around 0 (black line), scaled per row; green = excludes 0 on the better side.  The DK rows use only windows with a lead agent in frame (bootstrap over those episodes): "
                        f"seed 0 headway n = {nsb[S0][0]} windows / {nsb[S0][1]} episodes, time-gap {nsb[S0][2]} / {nsb[S0][3]}, LEAD-VIOL at-risk {nsb[S0][4]} / {nsb[S0][5]};  "
                        f"seed 1 {nsb[S1][0]} / {nsb[S1][1]}, {nsb[S1][2]} / {nsb[S1][3]}, {nsb[S1][4]} / {nsb[S1][5]}.", 205), 10, MUTED, ls=1.3)
        rows = getp(J("hacc_read"), "per_seed/M2b_s0/PRIMARY_DOM/rows").keys()
        nsep = {sd: sum(1 for k in rows if getp(J("hacc_read"), _pctr(sd, k, "separated")) and getp(J("hacc_read"), _pctr(sd, k, "better"))) for sd in SEEDS}
        nworse = sum(1 for sd in SEEDS for k in rows if getp(J("hacc_read"), _pctr(sd, k, "worse")))
        caption(fig, f"HACC - P10, T1-class open loop, both seeds: separated-better on all four primary rows (ADE2, ADE6, accel MAE, curvature MAE) and on {nsep[S0]} of 15 rows at seed 0, {nsep[S1]} of 15 at seed 1.  "
                     f"Not separated: DK headway and time-gap at both seeds (and LEAD-VIOL at seed 1).  Separated-worse rows: {nworse}.  Outcome of the pre-registered read: DOMINATES P10 "
                     "(the programme's own kinematic floor; not the state of the art).", size=16, y0=898, h=112, n=148)
    footer(fig, "Source: hacc/READ.json (per_seed.<seed>.PRIMARY_DOM.rows; the same numbers are in RESULT_V8F_HACC.md section 4, cross-checked in CONTROLS_V3.json), hacc/dk_read.json (n).  "
                "Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes, seed 0 (taniteval/ci.py): another draw of EPISODES only; training variance = the two seeds side by side; inference deterministic.  "
                "Tier T1-class open loop (own held outputs; NOT closed loop).")
    return render(fig)


BANDS = [("0.8-1.6 s (cells 4-8)", "0.8-1.6 s"), ("2.2-2.6 s (cells 11-13)", "2.2-2.6 s"), ("2.8-5.0 s (cells 14-25)", "2.8-5.0 s"), ("5.2-5.8 s (cells 26-29)", "5.2-5.8 s")]
BAND_T = [(0.8, 1.6), (2.2, 2.6), (5.2, 5.8)]


def dom_accel(p=1.0):
    fig = new_fig()
    header(fig, "5  Dominance: where the acceleration gap went (per-step accel error of HACC / H / P10)", tier=TIER_PL, sub="eval-141, all 2,399 windows  |  unpaired per-cell means (descriptive, no bar)  |  accel error = |a_plan - a_GT| on the 0.2 s knots")
    ap = lambda sd: f"per_seed/{sd}/"
    for k, (sd, bx) in enumerate(zip(SEEDS, (90, 1030))):
        ax = axpx(fig, bx, 100, 840, 420, W, H)
        ax.set_facecolor(PANEL)
        t = np.array(getp(J("hacc_accel"), ap(sd) + "t_s"))
        n = max(2, int(round(p * len(t))))
        for (a, b) in BAND_T:
            ax.axvspan(a - 0.1, b + 0.1, color="#fff2cc", zorder=0)
        for arm, kw in (("P10", dict(color="#8a8f98", ls="--", lw=2.2, label="P10 (prior)")), ("H", dict(color=PLAN_COL["H"], lw=2.4, label="H (HBLEND)")),
                        ("HACC", dict(color=PLAN_COL["HACC"], lw=3.4, label="HACC", zorder=6))):
            y = np.array(getp(J("hacc_accel"), ap(sd) + f"cells/{arm}"))
            ax.plot(t[:n], y[:n], **kw)
        ax.set_xlim(0.1, 5.9)
        ax.set_ylim(0.0, 1.6)
        ax.set_xlabel("time into the plan (s)  (accel cell k at t = 0.2 k s)", fontsize=9.5)
        ax.set_ylabel("mean |a_plan - a_GT|  (m/s^2)", fontsize=9.5)
        ax.grid(color="#e8ebef")
        ax.tick_params(labelsize=9)
        ax.set_title(f"seed {sd[-1]}   (yellow = the three bands where H sat above P10)", fontsize=10.8, fontweight="bold", loc="left")
        ax.legend(loc="upper left", fontsize=10, ncol=3, framealpha=0.95)
        if p >= 0.99:
            # band table under the chart
            hdr = "band             HACC    H     P10   HACC-P10   H-P10"
            lines = [hdr]
            for key, lab in BANDS:
                g = lambda a: N("hacc_accel", ap(sd) + f"bands/{key}/{a}", "{:.3f}")
                gm = lambda a: N("hacc_accel", ap(sd) + f"bands/{key}/{a}", "{:+.3f}")
                lines.append(f"{lab:<14s}  {g('HACC')}  {g('H')}  {g('P10')}   {gm('HACC_minus_P10')}    {gm('H_minus_P10')}")
            g = lambda a: N("hacc_accel", ap(sd) + f"all_cells_mean/{a}", "{:.3f}")
            dd = lambda a: ND("hacc_accel", ap(sd) + f"all_cells_mean/{a}", ap(sd) + "all_cells_mean/P10", "{:+.3f}")
            lines.append(f"{'all 29 cells':<14s}  {g('HACC')}  {g('H')}  {g('P10')}   {dd('HACC')}    {dd('H')}")
            tx(fig, bx, 598, NL.join(lines), 12.6, INK, mono=True, ls=1.45)
    if p >= 0.99:
        bd = lambda sd, key, a: N("hacc_accel", f"per_seed/{sd}/bands/{key}/{a}", "{:+.3f}")
        k1, k2, k3 = BANDS[0][0], BANDS[1][0], BANDS[3][0]
        caption(fig, f"Where the bumps went (m/s^2, seed 0 / seed 1, minus P10).  2.2-2.6 s: H {bd(S0, k2, 'H_minus_P10')} / {bd(S1, k2, 'H_minus_P10')} -> HACC {bd(S0, k2, 'HACC_minus_P10')} / {bd(S1, k2, 'HACC_minus_P10')} (gone).  "
                     f"0.8-1.6 s: H {bd(S0, k1, 'H_minus_P10')} / {bd(S1, k1, 'H_minus_P10')} -> HACC {bd(S0, k1, 'HACC_minus_P10')} / {bd(S1, k1, 'HACC_minus_P10')} (gone).  "
                     f"5.2-5.8 s: H {bd(S0, k3, 'H_minus_P10')} / {bd(S1, k3, 'H_minus_P10')} -> HACC {bd(S0, k3, 'HACC_minus_P10')} / {bd(S1, k3, 'HACC_minus_P10')}: reduced, NOT gone; HACC is still above P10 in the last second of the plan.",
                size=16, y0=872, h=130, n=148)
    footer(fig, "Source: hacc/accel_profile.json (= score_<seed>.json accel_profile); the instrument is tied to the banked fw_lon_accel_mae by control 7 of RESULT_V8F_HACC.md (9.5e-7 / 4.8e-7).  Unpaired cell means over 2,399 windows: a description of where the error sits, "
                "not an interval.  The paired accel MAE rows are in the dominance table.  Tier T1-class open loop (own held outputs; NOT closed loop).")
    return render(fig)


def dom_mech():
    fig = new_fig()
    header(fig, "5  Dominance: the mechanism (one lever: how the per-step blend weight is fitted)", tier=TIER_PL, sub="HACC = P + w (tau - P)  |  w is 30 x 2 numbers per seed, a fixed function of the step index  |  zero network training")
    cells = [(sd, ax_i) for ax_i in (0, 1) for sd in SEEDS]
    t = 0.2 * (np.arange(30) + 1)
    for (sd, ax_i), (bx, by) in zip(cells, ((84, 106), (500, 106), (84, 506), (500, 506))):
        ax = axpx(fig, bx, by, 340, 270, W, H)
        ax.set_facecolor(PANEL)
        key = ("w_along_by_step", "w_cross_by_step")[ax_i]
        s_ = SD[sd]
        wh, wc = np.array(getp(J(f"hblend_w_{s_}"), key)), np.array(getp(J(f"hacc_w_{s_}"), key))
        ax.axvspan(2.0, 2.8, color="#fff2cc", zorder=0)
        ax.text(2.4, 0.03, "steps 10-14", ha="center", fontsize=8, color=MUTED)
        ax.plot(t, wh, color=PLAN_COL["H"], lw=2.4, label="H (position fit + isotonic)")
        ax.plot(t, wc, color=PLAN_COL["HACC"], lw=3.0, label="HACC (+ accel term, no isotonic)")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0.2, 6.0)
        ax.grid(color="#e8ebef")
        ax.tick_params(labelsize=8.5)
        ax.set_xlabel("step time (s)", fontsize=9)
        ax.set_ylabel("blend weight w  (0 = P10, 1 = B)", fontsize=9)
        ax.set_title(f"{('ALONG (forward)', 'CROSS (lateral)')[ax_i]} axis, seed {sd[-1]}", fontsize=10.5, fontweight="bold", loc="left")
        if ax_i == 0 and sd == S0:
            ax.legend(loc="upper left", fontsize=8.6, framealpha=0.95)
    x1 = 870
    panel_box(fig, x1 + 20, 100, 1000 - 20 - 14, 700, ec="#D55E00", lw=2.0)
    tx(fig, x1 + 40, 114, "What changed", 20, "#D55E00", "bold")
    wa = lambda s_, who, i: N(f"{who}_w_{s_}", f"w_along_by_step/{i}", "{:.2f}")
    r4 = lambda sd, a: N("aux3", f"{sd}/{a}_by_step/3", "{:.3f}")
    nm = lambda s_, key: int(np.sum(np.diff(np.array(getp(J(f"hacc_w_{s_}"), key))) < -1e-9))
    ml = lambda sd: N(f"hacc_read", f"per_seed/{sd}/lam_star", "{:,.0f}")
    txt = ("H (HBLEND):  w fitted on POSITION error only, then forced to be non-decreasing in time.\n\n"
           "HACC:  w fitted on position error PLUS lambda x the second-difference (acceleration) error, bounded to [0, 1], NO monotone constraint.  "
           f"lambda is picked by a train-only rule (lambda* = {ml('M2b_s0')} seed 0, {ml('M2b_s1')} seed 1).\n\n"
           "Both: one weight per plan step and axis (30 x 2 per model), fitted on TRAINING windows only; at inference w depends on the step index alone and P is the deployed prior, "
           "so there is no scene read, no situation-classifier output, no label.\n\n"
           "A DESCRIPTIVE reading (not an ablation: the accel term and the dropped monotone constraint changed together):\n"
           f"  along weight, steps 10 -> 14 (2.0 -> 2.8 s), seed 0:  H {wa('s0', 'hblend', 9)} -> {wa('s0', 'hblend', 13)},  HACC {wa('s0', 'hacc', 9)} -> {wa('s0', 'hacc', 13)}\n"
           f"  seed 1:  H {wa('s1', 'hblend', 9)} -> {wa('s1', 'hblend', 13)},  HACC {wa('s1', 'hacc', 9)} -> {wa('s1', 'hacc', 13)}.\n"
           "H's steep ramp sits where its accel error peaked (yellow band); HACC ramps more gently and is not monotone elsewhere "
           f"(steps where w decreases: along {nm('s0', 'w_along_by_step')} / {nm('s1', 'w_along_by_step')}, cross {nm('s0', 'w_cross_by_step')} / {nm('s1', 'w_cross_by_step')}, seed 0 / seed 1).\n\n"
           f"First steps of the cross axis: HACC's w jumps 1 / 0 / 1 / 0 because mean |tau - P| there is only {r4('M2b_s0', 'cross')} m (seed 0, step 4): the weight is barely identified there and moves the plan by that much on average at most.")
    tx(fig, x1 + 40, 154, wrap(txt, 84), 13.4, INK, ls=1.36)
    caption(fig, "The mechanism in one sentence: HACC is the same one-line blend as H, but its per-step weight is fitted with an acceleration term in the objective and without the monotone constraint, "
                 "so it can ramp gently where the model's plan turns and dip where it is rough.  Written as a prediction before any number was seen (PREREG_V8F_HACC); the result is what dominates P10, the explanation is descriptive.",
            size=16, y0=872, h=130, n=148)
    footer(fig, "Source: hacc/weights_HACC_M2b_s{0,1}.json, hblend/weights_M2b_s{0,1}.json (w_along_by_step, w_cross_by_step), hacc/READ.json (lambda*), viz3/data/aux3.json (mean |tau - P| from hblend/score_*_perwindow.npz).  "
                "Tier T1-class open loop (own held outputs; NOT closed loop).  Weights were fitted on the 9,582 training windows of HBLEND's fit set; no eval window enters a fit or a selection.")
    return render(fig)


def dom_caveats():
    fig = new_fig()
    header(fig, "5  Dominance: what travels with this result", tier=TIER_PL, sub="read these before quoting the dominance row")
    kp = lambda sd, f, d=4: N("hacc_read", f"per_seed/{sd}/PRIMARY_KEEP/{f}", f"{{:+.{d}f}}")
    bar = N("hacc_read", "KEEP_bar_m", "{:+.5f}")
    Fm = N("hacc_read", "F_exact_m", "{:.4f}")
    margin = ND("hacc_read", "KEEP_bar_m", "per_seed/M2b_s1/PRIMARY_KEEP/hi", "{:.4f}")
    dlt1 = N("hacc_read", "per_seed/M2b_s1/PRIMARY_KEEP/delta", "{:.4f}")
    items = [
        ("1", "Tier.", "T1-class open loop: the model's own held outputs scored against the logged human path.  NOT closed loop; nothing here shows the plan being followed by a vehicle."),
        ("2", "Baseline.", "P10 is the programme's kinematic floor (it extrapolates ego motion and ignores the scene).  Beating it is a statement about our own floor, not about the state of the art; no external benchmark is shown."),
        ("3", "Estimator.", "Paired episode-cluster bootstrap, B = 2,000, 141 episodes: it answers 'would another draw of EPISODES say this?' only.  Training variance is read only from the two seeds side by side; "
                            f"F = |ADE6(s0) - ADE6(s1)| of B = {Fm} m is the seed floor.  Inference is deterministic, so there is no inference-seed floor."),
        ("4", "Seed 1's KEEP margin is thin.", f"PRIMARY-KEEP asks that HACC stays within F/2 = {bar} m of H on ADE6.  Seed 0: HACC - H {kp('M2b_s0', 'delta')} [{kp('M2b_s0', 'lo')}, {kp('M2b_s0', 'hi')}] (not separated).  "
                                                  f"Seed 1: {kp('M2b_s1', 'delta')} [{kp('M2b_s1', 'lo')}, {kp('M2b_s1', 'hi')}]: HACC is SEPARATED-WORSE than H by {dlt1} m ADE6, and passes the bar by only {margin} m."),
        ("5", "Distance keeping.", "Headway and time-gap are not separated from P10 at either seed (LEAD-VIOL is separated at seed 0 only).  Nothing is separated-worse."),
        ("6", "Only the plan moves.", "HACC touches the 6 s plan only.  T's decision heads and S's checkpoints are the banked B arrays, so HACC - H is exactly 0 on the strategic block.  "
                                      "The STRATEGIC family is IN FORCE for v8F (PI-V8F-2026-10-17; R72 / R73): S at its deployed (held) schedule still loses to the nav-only baseline at 50 m (next segment)."),
        ("7", "Not done here.", "HACC is not yet folded into forward_infer (its own landing, with tests), and the next tier (D-V8F-T1KIN-1) needs a PI go for the pooled eval cache."),
    ]
    y = 100
    for num, ttl, body in items:
        hh = 40 + 27 * (len(wrap(body, 160).split(NL)))
        panel_box(fig, 24, y, W - 48, hh, ec="#c9ced6", lw=1.0)
        tx(fig, 40, y + 8, num, 20, "#D55E00", "bold")
        tx(fig, 80, y + 9, ttl, 14.5, INK, "bold")
        tx(fig, 80, y + 36, wrap(body, 160), 14.0, INK, ls=1.38)
        y += hh + 9
    caption(fig, "Dominance over P10 is measured at both training seeds on the pre-registered read.  It is a T1-class open-loop result against our own floor; the cost against H is small but real at seed 1, "
                 "and the strategic layer is a separate open item that HACC does not touch.", size=16.5, y0=y + 6, h=88, n=146)
    footer(fig, "Source: hacc/READ.json (PRIMARY_KEEP, KEEP_bar_m, F_exact_m), RESULT_V8F_HACC.md sections 0, 4 and 8, PREREG_V8F_HACC.md (committed next step).  Tier T1-class open loop (own held outputs; NOT closed loop).")
    return render(fig)


# ================================================================================================ 6. strategic layer at its own tick
def s_frame(name, p, seed=S0):
    sel = WINDOWS[name]
    t, a_s = sel["t"], int(cfd(seed, name, "a_s"))
    fig = new_fig()
    header(fig, f"6  Strategic layer at its own tick  |  {WLAB[name]}", tier="T1-class open loop  |  DIAGNOSTIC: fresh ages are not the deployed schedule", sub=f"arm M2b, seed 0  |  {common_info(name)}")
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
    footer(fig, "Held = the banked eval's cycle ages (--ages cycle); fresh = --ages fresh (a_T = a_S = 0).  GT = X['rc'], the v9 RC-A labels in the NOW frame.  CPU re-runs of the banked M2b_s0; held / fresh checkpoints match the banked evals to <= 2e-5 (CONTROLS_V2.json).  STRATEGIC family: IN FORCE for v8F (PI-V8F-2026-10-17); HACC does not touch S.")
    return render(fig)


def s_card():
    fig = new_fig()
    header(fig, "6  Strategic layer at its own tick: the eval-wide numbers", tier="T1-class open loop  |  DIAGNOSTIC: fresh ages are not the deployed schedule")
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
    tx(fig, 24, 910, wrap("STRATEGIC family: IN FORCE for v8F (PI-V8F-2026-10-17; R72 / R73).  S's checkpoint error above IS that family's metric: at the deployed (held) schedule S loses to the nav-only baseline at 50 m; ticked fresh it beats it.  "
                          "HACC (previous segment) touches the 6 s plan only: S's checkpoints are the banked B arrays, so none of these numbers move with HACC.", 175), 12.5, "#b30000", "bold")
    footer(fig, "Source: sfresh/SFRESH_RESULT.md and sfresh/sfresh_numbers.json (PREREG_V8F_SFRESH, tip 583cfc02).  Estimator: paired episode-cluster bootstrap, B = 2,000, 141 episodes (another draw of EPISODES); training seeds side by side; deterministic inference.  "
                "n = 2,319 windows with a valid 50 m label.  Tier T1-class open loop, DIAGNOSTIC.")
    return render(fig)


# ================================================================================================ 7. closing
def closing_card():
    import v4_lib as V4                                   # v4: the closing card is rewritten for the T1-kin chapter (v3's 'any tier above this one: not run' is no longer true)
    fig = new_fig(DARK)
    tx(fig, 70, 50, "What this video measured, and what it did not", 32, "white", "bold")
    sb = lambda sd: N("tables_v2", f"sfresh/{sd}/banked_cycle_minus_navonly_50/delta", "{:+.2f}")
    kc = [V4.RC("RESULT_V8F_T1KIN_CAUSALBUCKET", "ADE6, PRIMARY", c)["val"] for c in (1, 2)]
    ks = [V4.RC("RESULT_V8F_T1KIN_STRICT", "ADE6, PRIMARY", c)["val"] for c in (1, 2)]
    l2 = [V4.RC("RESULT_V8F_T1_LATCOND", "L2:", c)["val"] for c in (1, 2)]
    l1 = [V4.RC("RESULT_V8F_T1_LATCOND", "L1:", c)["val"] for c in (1, 2)]
    boxes = [
        ("MEASURED", "#009E73", "Banked M2b checkpoints and banked per-window plans, training seeds 0 and 1, on the 2,399-window / 141-episode eval (CPU re-read).\n\n"
                               "Four windows declared before any rendering (SELECTION_V2.md, unchanged); every window panel is seed 0, seed 1 sits in the cards.\n\n"
                               "Who listens to what (LISTEN); nav counterfactual at S / T / O; B / H / HS / HACC / P10 plans; HACC - P10 on the dominance row (RESULT_V8F_HACC); S held vs fresh vs GT (SFRESH).\n\n"
                               f"NEW, chapter 7, T1-kin: HACC run on its own executed motion, minus the P10 floor run the same way, ADE6: CAUSAL {kc[0]} / {kc[1]} m, STRICT {ks[0]} / {ks[1]} m (seed 0 / seed 1, separated)."),
        ("TIER, ESTIMATOR, BASELINE", "#0072B2", "Chapters 1-6: T1-class open loop (own held outputs; NOT closed loop): the model reads its own held outputs; nothing it does feeds back.  Interventions are diagnostics.\n\n"
                                                 "Chapter 7: T1-kin, self-action OPEN loop: the model replans every 2 s from its own executed motion; the cameras are the human's recorded frames (CAUSAL) or frozen at the first frame (STRICT).  NOT closed loop.\n\n"
                                                 "Intervals = paired episode-cluster bootstrap, B = 2,000, 141 episodes: 'another draw of EPISODES' only.  Training variance = the two seeds side by side.\n\n"
                                                 "P10 is the programme's own kinematic floor, not the state of the art."),
        ("NOT SHOWN, STILL OPEN", "#D55E00", "Re-perception (T2): not provisioned.  External benchmarks: none.  T1-kin has no tactical or strategic rows (NOT COMPUTED).\n\n"
                                            f"OPEN GAP, T1-kin lateral: curvature at GT speed of at least 5.2 m/s is worse than the floor's ({l2[0]} / {l2[1]} 1/m, separated); lateral acceleration is not ({l1[0]} / {l1[1]} m/s²).\n\n"
                                            "STRATEGIC family: IN FORCE for v8F (PI-V8F-2026-10-17; R72 / R73).  S at the deployed (held) schedule loses to the nav-only baseline at 50 m "
                                            f"({sb('s0')} / {sb('s1')} m); HACC does not touch S.\n\n"
                                            "Open on HACC: the consistency cost against B; seed 1's thin KEEP margin against H; distance keeping not separated from P10; accel above P10 at 5.2-5.8 s."),
    ]
    for i, (ttl, col, body) in enumerate(boxes):
        x = 70 + i * 600
        ax = axpx(fig, x, 130, 560, 600, W, H)
        ax.set_facecolor("#18202a"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_edgecolor(col); s_.set_linewidth(2.6)
        tx(fig, x + 20, 146, ttl, 19, col, "bold")
        tx(fig, x + 20, 188, wrap(body, 46), 13.4, "#e8ecf2", ls=1.34)
    tx(fig, 70, 758, "Artifacts (dev box, nothing staged or pushed):", 15, "white", "bold")
    paths = ("scratchpad/viz4/:  v8f_evidence_v4.mp4,  CONTROLS_V4.json,  VIDEO_NOTES_V4.md (every number with its artifact path, tier and estimator),  SELECTION_V4.md,  qa_*.png,  FILES.txt,  data/,  *.py\n"
             "inputs (read-only):  t1kc_run/ + t1ks_run/ (float64 T1-kin dumps),  viz4/data/results/ (the four RESULT files, commit abc12d3f),  hacc/ (READ.json, RESULT_V8F_HACC.md, score_*, weights_HACC_*),  viz2/,  hblend/ + hspec/,  sfresh/,  listen/,  viz/ (frames, checkpoints)")
    tx(fig, 70, 792, wrap(paths, 215), 11, "#aeb7c4", ls=1.5)
    tx(fig, 70, 1010, "Registered claims are not updated by this video: it presents banked results and CPU re-reads; landing and the GOALS_AND_CLAIMS register are the Master Mind's integration step.", 11.5, "#7f8a99")
    return render(fig)


SCENES = {k: v for k, v in globals().items() if callable(v) and not k.startswith("_") and k in (
    "title_card", "listen_frame", "qnav_frame", "method_card", "cf_all", "cf_bcd", "cf_four", "cf_summary", "plans_frame", "plans_table_card", "plans_gap_card",
    "dom_table", "dom_accel", "dom_mech", "dom_caveats", "s_frame", "s_card", "closing_card")}
