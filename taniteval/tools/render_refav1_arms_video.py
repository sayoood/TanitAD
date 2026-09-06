#!/usr/bin/env python3
"""render_refav1_arms_video.py — FOUR refav1 planner arms, ONE scene, side by side.

⛔ **READ `taniteval/tools/RENDER_REFAV1_VIDEO.md` FIRST.** This tool is its
sibling and reuses its camera model, its decode and its refusals. The difference
is the QUESTION:

  * `render_refav1_video.py` draws ONE arm against the trivial floor, to make
    visible that the shipped planner emits nothing.
  * **this file draws SEVERAL arms on the SAME window**, because the 2026-09-05
    cost-geometry panel's finding is a TRADE-OFF between arms, and a trade-off
    cannot be seen one arm at a time.

⭐ **THE ONE THING THIS REEL EXISTS TO MAKE VISIBLE** (MEASURED, this panel,
n = 40 windows / 8 episodes; `…/2026-09-05-refav1-cost-geometry/raw/arms/`,
computed here by `taniteval.four_families.lateral` — the EVAL'S OWN function):

    arm                          curvature MAE 1/m   ADE m   reading
    wk15   (W_KAPPA 15.11)            0.030982       0.8934  best in the programme
    best   (+ Kamm cap mu 0.7)        0.031281       0.8838  -22 % vs the floor
    ha0    THE STRAIGHT FLOOR         0.040083       0.9251  a plan that never steers
    ccos_argmax (W_KAPPA 0)           0.055369       1.3272  WORSE THAN STRAIGHT
    ha0_ext  the ECHO control         0.077298       0.8772  holds a MEASURED kappa

⇒ ⛔ **THE UNPENALISED ARM TRACKS THE ROAD WORSE THAN A PERFECTLY STRAIGHT
PLAN**, and ⭐ **`wk15` beats that straight floor by 23 %.** That is a picture,
not a table: on a window where the road does NOT bend, the magenta arm swings
0.08 1/m of curvature into a straight road while the orange and violet ones lie
exactly on the white straight line — and so does the green ground truth.

⛔ **THIS REEL SUPERSEDES ITS OWN FIRST CUT, AND THE CORRECTION IS THE FLOOR.**
The first render drew **`ha0_ext`** — the ECHO control, which holds the MEASURED
curvature and reaches **8.21 m** of lateral swing — as the white "floor", and
called `ccos_argmax` *"the arm that turns"*. Both readings are refuted
(`M74`/`M75`/`M77`/`M79`, `Project Steering/Decisions/2026-09-06-mm-decisions.md`):
the turn-recall gate `|dyaw| > 0.15` rad is **UNREACHABLE at this panel's speeds
and the HUMAN fails it 3 of 9**, so "it turns" had to be re-read on curvature
MAE — a metric the human passes by construction — against the only floor that
makes the sentence checkable: **`ha0`, the perfectly straight plan** (MEASURED
max |y| **0.000000 m** over all 40 windows). ⇒ **A comparison is only as good as
the floor it is drawn against, and the floor must be the one the CLAIM names.**

WHAT THIS TOOL DOES AND DOES NOT DO
===================================
It draws pixels. It **invents no number and runs no model**: every trajectory,
every head argmax and every planner provenance field is read out of dumps written
by `taniteval/tools/refav1_arm.py`, the tool that produced the scored records.
The only quantities computed here are (a) per-window ADE, a mean of norms over
arrays the dump carries, (b) the per-window feasibility flags, delegated to
`tanitad.refs.feasible_decode.assert_feasible` — the SCORER'S OWN envelope — and
(c) the per-window **curvature MAE**, delegated to
`taniteval.four_families.lateral` — the EVAL'S OWN function. Neither (b) nor (c)
is re-typed here: a re-typed rule drifts from the one the banked numbers were
computed under, and the frame would then contradict the record it illustrates.

⭐ **AND (c) CARRIES ITS OWN CONTROL.** `--expect-curv-mae` re-computes the
PANEL aggregate for each drawn series and REFUSES if it misses the banked value
in `raw/four_family_all.txt`. MEASURED: the reel's own instrument reproduces
`wk15` 0.030982 / `best` 0.031281 / `ha0` 0.040083 / `ccos_argmax` 0.055369 /
`ha0_ext` 0.077298 exactly. A per-frame number that cannot reproduce the banked
panel is a different metric wearing its name.

⭐ **THE ARMS ARE VERIFIED TO SHARE EVERYTHING BUT THE PLANNER.** Before a frame
is drawn, this tool asserts that every rendered dump carries bit-identical
`ws`, `g`, `v0`, `ha0_ext`, `wm_mse_*` and decision-head argmaxes, and REFUSES
otherwise. That is what licenses drawing them on one image and saying the
difference is the cost geometry: if the scenes or the labels differed, the
picture would be comparing two things at once.

THE REFUSALS INHERITED FROM THE SIBLING (each one has cost this programme hours)
===============================================================================
**(a) The codec is read from the `codec` FIELD, never from the buffer's NAME.**
The buffer is called `jpeg_buf` and the codec says `png`. Decoding by the name
raises — or, on a pre-allocated output, leaves zeros and exits 0.

**(b) The projection is CYLINDRICAL, and it is the clip's own.** `u = W/2 +
f_ref * atan2(y_right, x_fwd)`; the pinhole formula on this raster implies a
92.6 deg field where the rig is `camera_front_wide_120fov` (120 deg). Extrinsics
are MEASURED per clip. The horizon predicted from those extrinsics is drawn as
the frame's own FALSIFIER.

**(c) The integrator is NOT re-implemented.** Every arm's trajectory is read from
the dump. `--gt-control` re-rolls the RECORDED actions through the programme's
own `refav1_arm.paths_from_controls` -> `refa_v1_plan.unicycle_paths` and asserts
the result lands on the dumped ground truth.

**(d) Frames are bridged LOCALLY and the plan is HELD, never re-invented.**
Between replans the last plan is carried into the current ego frame by a rigid
SE(2) transform of the RECORDED poses. Past the plan's own horizon there is no
action left, so the frame is SKIPPED and COUNTED.

**(e) Every arm is OPEN LOOP** (PI ruling 2026-09-02). The model controls
nothing; the ego data keeps arriving from the recording.

⭐ SLOW MOTION IS APPLIED BY A MEASURED CRITERION, AND SAID ON THE FRAME
=======================================================================
Windows whose v7.2 lateral label is `TURN_L` / `TURN_R` are held `--slowmo`x
longer, because they are the windows the finding is about and 2.0 s at 10 fps is
too fast to read four lines apart. The criterion is the LABEL, fixed before any
arm ran — never "the frames where the arms differ most", which would be choosing
the evidence by the answer. Every slowed frame prints `SLOW MOTION` and the
factor, so no viewer can mistake it for the vehicle's real speed.

USAGE
=====
    python taniteval/tools/render_refav1_arms_video.py \\
        --arms-dir  <dir of dump_<arm>/ >  --arms ccos_argmax,best,wk15 \\
        --arm-colours ccos_argmax=magenta,best=violet,wk15=orange \\
        --episodes  <dir of *.v2ep.pt>     --extrinsics extrinsics.json \\
        --expect-curv-mae ha0=0.040083,wk15=0.030982,ccos_argmax=0.055369 \\
        --out reel.mp4 --fps 10 --slowmo 6 --gt-control

Verify the result with `taniteval/tools/verify_mp4.py`, which DECODES IT BACK.
ASCII only in every print(): this box is cp1252.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_by_path(name: str, path: str):
    if not os.path.exists(path):
        sys.exit(f"[arms-reel] required sibling {path} is missing")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


#: ⭐ ONE camera model and ONE set of draw helpers in the programme, imported —
#: never re-implemented beside the tool that uses them.
rv = _load_by_path("render_refcv3_for_arms",
                   os.path.join(_HERE, "render_refcv3_video.py"))
ra = _load_by_path("render_refav1_for_arms",
                   os.path.join(_HERE, "render_refav1_video.py"))
#: ⭐ THE ARM TOOL ITSELF, for its integrator wrapper `paths_from_controls`
#: (-> `refa_v1_plan.unicycle_paths`). The ground-truth control must exercise
#: the SAME integrator the dumps were rolled through; a second copy of a
#: unicycle beside the one that produced the numbers is how a control ends up
#: validating itself instead of the harness.
arm_tool = _load_by_path("refav1_arm_for_arms",
                         os.path.join(_HERE, "refav1_arm.py"))

import torch                                                       # noqa: E402
from PIL import Image, ImageDraw                                   # noqa: E402

from tanitad.models.v6 import (tactical_lat_actions,               # noqa: E402
                               tactical_lon_actions_v)
from tanitad.refs.refb import NAV_COMMANDS, ROUTE_CLASSES          # noqa: E402
from tanitad.refs import feasible_decode as FD                     # noqa: E402
from tanitad.viz_standard import VizElement, check_frame           # noqa: E402


def _import_four_families():
    """`taniteval.four_families`, past the NAMESPACE-PACKAGE SHADOW.

    ⚠ The repo's OUTER `taniteval/` carries no `__init__.py`, so an `import
    taniteval` resolved from the repo root binds a NAMESPACE package that has no
    `four_families` in it — and the failure reads `cannot import name
    'four_families' from 'taniteval' (unknown location)`, which looks exactly
    like a missing module rather than a shadow. The REAL package is the INNER
    `taniteval/taniteval/`, whose parent is this file's own grandparent."""
    parent = os.path.dirname(_HERE)                       # <repo>/taniteval
    if parent not in sys.path:
        sys.path.insert(0, parent)
    for k in [k for k in list(sys.modules)
              if k == "taniteval" or k.startswith("taniteval.")]:
        if getattr(sys.modules[k], "__file__", None) is None:
            del sys.modules[k]                            # drop the shadow
    from taniteval import four_families as _ff
    if not hasattr(_ff, "lateral"):
        sys.exit("[arms-reel] imported a taniteval without `lateral` — the "
                 "namespace shadow is still in front of the real package")
    return _ff


FF = _import_four_families()

CylProjector = rv.CylProjector
polyline, densify, font, fit, wrap = rv.polyline, rv.densify, rv.font, rv.fit, rv.wrap
load_extrinsics = rv.load_extrinsics
episode_payload, frame_decoder, se2_carry = (ra.episode_payload, ra.frame_decoder,
                                             ra.se2_carry)

DT_CACHE = 0.2          # one cache step / one operative tick
DT_FRAME = 0.1          # one raw frame
CAM_UP = 2              # 256x640 -> 512x1280
WM_DT = 0.2             # the world-model diagnostic tick

# --------------------------------------------------------------------------- #
# COLOUR SEMANTICS — one meaning per colour, in EVERY panel                    #
# --------------------------------------------------------------------------- #
C_GT = (110, 231, 138)          # GROUND TRUTH — green, everywhere, always
#: ⛔ THE WHITE IS `ha0`, THE PERFECTLY STRAIGHT PLAN — NOT `ha0_ext`, AND THE
#: SWAP IS THE WHOLE POINT OF THIS CUT. The claim the reel now makes is
#: *"`ccos_argmax` tracks the road WORSE THAN A STRAIGHT LINE"*, and that
#: sentence is only CHECKABLE with the straight line drawn. `ha0` is
#: `a = 0, kappa = 0` at the measured v0 (MEASURED max |y| 0.000000 m over all
#: 40 windows); `ha0_ext` holds the MEASURED (a0, kappa0) and swings up to
#: 8.21 m, so it is a floor for ADE and NOT for road-tracking.
C_FLOOR = (236, 240, 246)       # the STRAIGHT FLOOR ha0 — white, wide, under
C_ECHO = (120, 130, 148)        # the ECHO control ha0_ext — thin, dim, beneath
C_GIVEN = (240, 190, 90)        # a GIVEN INPUT (the nav token) — amber
#: the arm palette, in the order arms are drawn. Four hues no viewer confuses,
#: none of them green (ground truth), white (the floor) or amber (a given input).
#: ARM_COLOURS = [(244, 114, 182),      # magenta
#: (56, 189, 248),  # cyan ... -- see the tuple below. The orange is
#: (249, 115, 22), NOT the sibling reel's (255, 158, 61): that one sits an L1
#: distance of 76 from C_GIVEN (240, 190, 90), the programme's "this is an
#: INPUT, not a decision" amber, and a viewer scanning one frame cannot be
#: asked to hold a 76-unit distinction. Pinned by
#: `stack/tests/test_render_refav1_arms.py`, which requires > 90 from every
#: reserved colour and FAILED on the original palette.
ARM_COLOURS = [(244, 114, 182),      # magenta
               (56, 189, 248),       # cyan
               (249, 115, 22),       # orange (see the note above)
               (167, 139, 250),      # violet
               (163, 230, 53),       # lime (a 5th, if ever asked for)
               (45, 212, 191)]       # teal (a 6th)
#: ⭐ THE SAME SIX, BY NAME, so `--arm-colours` can pin a hue to an ARM rather
#: than to its position in `--arms`. A reel whose colours move when the arm
#: ORDER moves cannot be compared with the cut before it, and the previous cut
#: drew `ccos_argmax` magenta / `wk15` orange / `best` violet at positions
#: 1 / 3 / 4. Dropping one arm would have silently recoloured the other three.
NAMED_COLOURS = {"magenta": (244, 114, 182), "cyan": (56, 189, 248),
                 "orange": (249, 115, 22), "violet": (167, 139, 250),
                 "lime": (163, 230, 53), "teal": (45, 212, 191)}
C_BG = (9, 12, 17)
C_PANEL = (16, 21, 29)
C_GRID = (34, 42, 53)
C_FG = (233, 238, 245)
C_DIM = (140, 152, 168)
C_WARN = (245, 180, 90)
C_OK = (110, 231, 138)
C_BAD = (248, 113, 113)

# --------------------------------------------------------------------------- #
# LAYOUT — 1920 x 1080                                                         #
# --------------------------------------------------------------------------- #
PAD = 12
W_LEFT, H_CAM = 1280, 512
W_RIGHT = 604
W_TOT = PAD + W_LEFT + PAD + W_RIGHT + PAD          # 1920
Y_BAN, H_BAN = 0, 72
Y_CAM = Y_BAN + H_BAN + 6                            # 78
Y_MID = Y_CAM + H_CAM + 12                           # 602
H_MID = 288
Y_BOT = Y_MID + H_MID + 12                           # 902
#: ⛔ H_BOT IS SET SO H_TOT LANDS ON 1122 -- THE SIBLING REEL'S CANVAS -- AND
#: THAT IS NOT COSMETIC. `draw_card` is IMPORTED from `render_refav1_video.py`
#: and paints at ITS `W_TOT x H_TOT`. MEASURED by `verify_mp4.py`: with a
#: 1080-tall frame and a 1122-tall card, ffmpeg's image demuxer took the FIRST
#: frame's size (the intro card) and rescaled every clip frame into it -- the
#: whole reel stretched vertically by 3.9 %, and every exit code said success.
#: `emit()` now asserts the size of every frame it writes, so a future layout
#: change fails loudly instead of silently distorting the delivery.
H_BOT = 208
H_TOT = Y_BOT + H_BOT + PAD                          # 1122
X_RIGHT = PAD + W_LEFT + PAD                         # 1304

PLAN_SOURCE_NAMES = ("cem", "baseline:cv", "baseline:hold_v0",
                     "baseline:proposal", "baseline:decel_1.5")
#: the v7.2 lateral tokens that mean "the road turns here". The slow-motion
#: criterion, fixed by the LABEL rather than by where the arms happen to differ.
TURN_TOKENS = ("TURN_L", "TURN_R")


def _p(*a):
    print(*a, flush=True)


#: the colours no arm may wear, because each already MEANS something on every
#: frame of every reel in this programme.
RESERVED_COLOURS = (("ground truth", C_GT), ("floor ha0", C_FLOOR),
                    ("echo ha0_ext", C_ECHO), ("given input", C_GIVEN))
#: the minimum L1 distance an arm colour must keep from each of them. A viewer
#: scanning ONE frame cannot be asked to hold a smaller distinction; MEASURED,
#: the sibling reel's orange (255, 158, 61) sits L1 76 from the given-input
#: amber and had to be moved.
MIN_COLOUR_L1 = 90


def resolve_arm_colours(order, spec=None):
    """-> {arm: rgb}. Positional by default; pinned by name when `spec` is given.

    ⛔ IT REFUSES three ways, and each has cost something somewhere: an unknown
    arm or colour name (a typo silently drawing the default palette), two arms
    on one hue (two lines a viewer reads as one), and any arm within
    :data:`MIN_COLOUR_L1` of a RESERVED colour (an arm read as the thing it is
    being compared against). The rule is EXECUTED here rather than only asserted
    in a test, so `--arm-colours` cannot walk around it."""
    colours = {n: ARM_COLOURS[i] for i, n in enumerate(order)}
    if spec:
        want = {}
        for tok in str(spec).split(","):
            nm, _, cn = tok.strip().partition("=")
            if nm not in order:
                sys.exit(f"[arms-reel] --arm-colours names {nm!r}, not in --arms")
            if cn not in NAMED_COLOURS:
                sys.exit(f"[arms-reel] --arm-colours: {cn!r} is not one of "
                         f"{', '.join(sorted(NAMED_COLOURS))}")
            want[nm] = NAMED_COLOURS[cn]
        if len(set(map(tuple, want.values()))) != len(want):
            sys.exit("[arms-reel] --arm-colours assigns one hue to two arms")
        colours.update(want)
    if len(set(map(tuple, colours.values()))) != len(colours):
        sys.exit("[arms-reel] two arms would be drawn in the same colour")
    for n, c in colours.items():
        for rn, rc in RESERVED_COLOURS:
            dist = sum(abs(int(x) - int(y)) for x, y in zip(c, rc))
            if dist <= MIN_COLOUR_L1:
                sys.exit(f"[arms-reel] arm {n}'s colour {tuple(c)} sits L1 "
                         f"{dist} from the RESERVED {rn} colour {tuple(rc)}. "
                         f"Refusing a palette a viewer cannot read.")
    return colours


# =========================================================================== #
# THE DUMPS — read once, and VERIFIED to share everything but the planner      #
# =========================================================================== #
#: read from every dump and asserted BIT-IDENTICAL across arms. If any of these
#: moved, the arms would not be on the same scene and the whole overlay would be
#: comparing two things at once.
#: ⛔ `ha0` JOINS THE SHARED SET, and it is not decoration: it is the floor the
#: reel's central claim is measured against, so it must be asserted BIT-IDENTICAL
#: across arms exactly as the scene and the echo control are. A floor that
#: differed between arms would make the comparison meaningless in the one panel
#: that carries the argument.
SHARED_KEYS = ("ws", "v0", "g", "ha0", "ha0_ext")
SHARED_DEC = ("wm_mse_model", "wm_mse_const", "wm_mse_zero", "wm_tgt_std",
              "lat_label", "lon_label", "route_label", "nav_cmd", "nav_valid",
              "lat_pred_nav_true", "lon_pred_nav_true", "route_pred_nav_true",
              # the echo control's COMMANDED (a, kappa) — so the floor rows in
              # the arms table quote the same quantity the arm rows do
              # (commanded curvature), never a path-derived one beside it.
              "ha0_ext_controls")
#: read per arm — these are what the cost geometry actually moves.
ARM_KEYS = ("cl",)
ARM_DEC = ("cl_controls", "plan_source_cl", "plan_cost_cl", "plan_neval_cl",
           "goal_lat_cl", "goal_lon_cl", "goal_source_cl",
           "basecost_cv_cl", "basecost_hold_v0_cl", "basecost_proposal_cl",
           "basecost_decel_1.5_cl", "finecost_plan_cl")


def read_arm(arms_dir: str, arm: str) -> dict:
    """One arm's dump: its manifest and every array, per episode name."""
    D = os.path.join(arms_dir, f"dump_{arm}")
    man_p = os.path.join(D, "manifest.json")
    if not os.path.exists(man_p):
        sys.exit(f"[arms-reel] no manifest at {man_p}")
    with open(man_p, encoding="utf-8") as fh:
        man = json.load(fh)
    eps = sorted(glob.glob(os.path.join(D, "ep*.npz")))
    if not eps:
        sys.exit(f"[arms-reel] no ep*.npz under {D}")
    out = {}
    for p in eps:
        dec_p = os.path.join(D, "decisions", os.path.basename(p))
        if not os.path.exists(dec_p):
            sys.exit(f"[arms-reel] {p} has no decisions/ sibling — the reel needs "
                     f"the decision arrays and refuses to draw a blank panel")
        d, dp = np.load(p), np.load(dec_p)
        fi = int(os.path.basename(p)[2:5])
        name = next((e.get("name") for e in man.get("episodes", [])
                     if int(e.get("file_index", -1)) == fi), None)
        if name is None:
            sys.exit(f"[arms-reel] manifest has no episode for {p}")
        rec = {k: np.asarray(d[k]) for k in SHARED_KEYS + ARM_KEYS}
        rec.update({k: np.asarray(dp[k]) for k in SHARED_DEC + ARM_DEC})
        # ⛔ a window order that differed between arms would silently pair a
        # magenta plan with a cyan plan from a DIFFERENT moment.
        order = np.argsort(rec["ws"])
        for k in list(rec):
            rec[k] = rec[k][order]
        out[name] = rec
    return {"manifest": man, "eps": out, "dir": os.path.abspath(D)}


def verify_shared(arms: dict, order: list[str]) -> dict:
    """⛔ REFUSE unless every arm carries the SAME scene, floor, labels and
    world model. This is the assertion that licenses one overlay.

    ⭐ It is a POSITIVE assertion with its own control: alongside the equality
    check it reports how many arms actually DIFFER on `cl` — which must be
    non-zero, or the "arms" are one arm under several names and the reel shows
    nothing. Equality that comes from reading the same bytes twice is not
    evidence, so both halves are printed."""
    ref = order[0]
    rep = {"shared_keys_checked": [], "arms_differing_on_cl": 0, "n_windows": 0}
    for name, r0 in arms[ref]["eps"].items():
        rep["n_windows"] += int(r0["ws"].shape[0])
        for arm in order[1:]:
            r1 = arms[arm]["eps"].get(name)
            if r1 is None:
                sys.exit(f"[arms-reel] arm {arm} has no episode {name} — the "
                         f"arms are not on the same panel")
            for k in SHARED_KEYS + SHARED_DEC:
                if not np.array_equal(r0[k], r1[k]):
                    sys.exit(
                        f"[arms-reel] REFUSING: {arm} differs from {ref} on the "
                        f"SHARED array {k!r} in episode {name}. These arms are "
                        f"not on the same scene/labels/world-model, so drawing "
                        f"them on one image would attribute a scene difference "
                        f"to the cost geometry.")
    rep["shared_keys_checked"] = list(SHARED_KEYS + SHARED_DEC)
    for arm in order[1:]:
        diff = any(not np.array_equal(arms[ref]["eps"][n]["cl"],
                                      arms[arm]["eps"][n]["cl"])
                   for n in arms[ref]["eps"])
        rep["arms_differing_on_cl"] += int(diff)
    if order[1:] and rep["arms_differing_on_cl"] == 0:
        sys.exit("[arms-reel] REFUSING: every arm emits the SAME cl trajectory. "
                 "The reel would draw one line under four names.")
    return rep


# =========================================================================== #
# FEASIBILITY — the SCORER'S OWN envelope, per window, never re-derived        #
# =========================================================================== #
def window_feasibility(path_xy: np.ndarray, dt: float = DT_CACHE) -> dict:
    """One window's path -> the scorer's own flags.

    ⛔ Delegated to `tanitad.refs.feasible_decode.assert_feasible`, the module
    the fan scorer itself uses (A_MAX 4.0, KAPPA_MAX 0.2, MU_KAMM 0.7). A
    friction-circle rule re-typed here could drift from the one the numbers in
    `RESULT.md` were computed under, and the frame would then contradict the
    record it claims to illustrate.

    ⚠️ The ego origin is PREPENDED so the first step's speed is recoverable —
    exactly as `raw/feas_audit.py` does it; without it the first step has no
    incoming velocity and `kappa = a_lat / v^2` is undefined there."""
    t = torch.from_numpy(np.asarray(path_xy, dtype=np.float64))[None]
    t = torch.cat([torch.zeros(1, 1, 2, dtype=t.dtype), t], dim=1)
    r = FD.assert_feasible(t, dt=dt)
    return {"kamm_over": float(r.get("kamm_over_rate", 0.0)) > 0.0,
            "envelope_over": float(r["envelope_rate"]) > 0.0,
            "peak_g": float(r["peak_g_max"]),
            "max_kappa": float(r["max_abs_kappa"])}


# =========================================================================== #
# CURVATURE — THE EVAL'S OWN REDUCTION, CALLED, NEVER RE-TYPED                 #
# =========================================================================== #
#: the band inside which a per-window curvature error is a TIE with the floor.
#: It is the arms table's own display precision, so the colour can never
#: contradict the digits printed next to it.
CURV_TIE = 1e-5


def curv_colour(v: float, floor: float):
    """green BELOW the straight floor, red ABOVE it, neutral on a TIE."""
    if v != v or floor != floor:              # NaN
        return C_FG
    if v < floor - CURV_TIE:
        return C_OK
    if v > floor + CURV_TIE:
        return C_BAD
    return C_FG

def curvature_mae(pred, gt, dt: float = DT_CACHE):
    """-> (per-window array, POOLED float, n valid steps).

    ⛔ THE POOLED VALUE IS A RATIO OF SUMS OVER VALID STEPS, WHICH IS NOT THE
    MEAN OF THE PER-WINDOW MEANS, and the difference is large enough to miss the
    banked number. MEASURED on this panel: `ccos_argmax` reads **0.055369**
    pooled — the value `raw/four_family_all.txt` prints — and **0.056726** as a
    mean of window means. `four_families.lateral` reduces it the first way
    (`_masked`, and `_lateral_ci` says so in its own docstring), so this
    function does too and the `--expect-curv-mae` control checks it.

    ⚠ The mask is `pair_valid`, of shape [n, H-1], NOT the per-step `valid` of
    shape [n, H]: curvature is a PAIR quantity and masking it with `valid`
    mis-aligns rather than being merely imprecise."""
    P = FF._seq_geometry(torch.as_tensor(np.asarray(pred)).float(), dt)
    G = FF._seq_geometry(torch.as_tensor(np.asarray(gt)).float(), dt)
    both = P["pair_valid"] & G["pair_valid"]
    err = (P["curvature"] - G["curvature"]).abs()
    if err.shape != both.shape:
        sys.exit(f"[arms-reel] curvature {tuple(err.shape)} vs mask "
                 f"{tuple(both.shape)} — refusing to reduce a mis-aligned mask")
    cnt = both.sum(1)
    per = torch.where(cnt > 0, (err * both).sum(1) / cnt.clamp_min(1),
                      torch.full_like(cnt, float("nan"), dtype=err.dtype))
    pooled, n = FF._masked(err, both)
    #: ⭐ A SELF-CHECK THAT COSTS NOTHING: the step-count-weighted mean of the
    #: per-window values MUST reproduce the pooled value. If it does not, the two
    #: numbers on the frame (per window) and in the banner (per panel) are
    #: different statistics and one of them is lying.
    if n:
        c = cnt.numpy().astype(np.float64)
        p = per.numpy().astype(np.float64)
        recon = float(np.nansum(np.where(c > 0, p, 0.0) * c) / c.sum())
        if abs(recon - pooled) > 1e-6:
            sys.exit(f"[arms-reel] curvature reduction is inconsistent: "
                     f"per-window recombines to {recon:.9f}, pooled {pooled:.9f}")
    return per.numpy().astype(np.float64), float(pooled), int(n)


def curvature_control(panel: dict, expect: dict) -> dict:
    """⛔ REFUSE unless the reel's own instrument reproduces the BANKED panel.

    ⭐ This is the control that makes the reel's central sentence checkable. A
    per-frame curvature number that cannot reproduce `raw/four_family_all.txt`
    is a different metric wearing the same name, and the frame would then
    contradict the record it claims to illustrate."""
    rep = {"expected": dict(expect), "got": {}, "tol": 1.5e-6, "checked": 0}
    if not expect:
        _p("[curv  ] ⚠ NO --expect-curv-mae given — the panel aggregate is "
           "UNCHECKED against the banked table. Pass it.")
        rep["unchecked"] = True
        return rep
    bad = []
    for k, want in expect.items():
        if k not in panel:
            sys.exit(f"[arms-reel] --expect-curv-mae names {k!r}, which is not "
                     f"a drawn series ({', '.join(sorted(panel))})")
        got = float(panel[k]["pooled"])
        rep["got"][k] = got
        rep["checked"] += 1
        ok = abs(got - want) <= rep["tol"]
        _p(f"[curv  ] CONTROL {k:12s} banked {want:.6f}  measured {got:.6f}  "
           f"n_steps {panel[k]['n_steps']:4d}  {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(k)
    if bad:
        sys.exit(f"[arms-reel] REFUSING to render: the curvature instrument "
                 f"misses the banked panel on {bad}. Every curvature number the "
                 f"reel would print is inadmissible.")
    rep["pass"] = True
    return rep


# =========================================================================== #
# PANELS                                                                      #
# =========================================================================== #
def draw_bev(size, series, x_lo, x_hi, note, F, ego_v):
    """The metric BEV — calibration-independent, so it carries the comparison
    even on a clip whose extrinsics are missing.

    ``series`` is a list of ``(path, colour, width, dashed)`` drawn IN ORDER,
    so the floor (wide, first) shows as a halo under any arm that coincides with
    it. Where a plan IS the floor a viewer must read "these are the same path",
    not "one of them is missing".

    ⛔ THE X RANGE STARTS BEHIND THE EGO, AND IT MUST. A plan is emitted once
    per scored window and then HELD while the vehicle drives along it, so by the
    end of the window its first waypoints are BEHIND the current ego origin. A
    panel that started at x = 0 would silently clip them off the bottom edge and
    the plan would appear to shrink from its own tail — which reads as a model
    that revises its plan, and it does not. The range is also fixed for the
    WHOLE window (see the caller), so the grid does not jump between frames and
    two moments can be compared by eye."""
    w, h = size
    im = Image.new("RGB", (w, h), C_PANEL)
    d = ImageDraw.Draw(im)
    m = 44
    x_lo, x_hi = float(x_lo), float(max(x_hi, x_lo + 10.0))
    y_half = max((x_hi - x_lo) * (w - 2 * m) / (2.0 * (h - 2 * m)), 5.0)

    def P(x, y):
        return (m + (w - 2 * m) * (0.5 - y / (2 * y_half)),
                (h - m) - (h - 2 * m) * (x - x_lo) / (x_hi - x_lo))

    g0 = int(math.ceil(x_lo / 5.0) * 5)
    for gx in range(g0, int(x_hi) + 1, 5):
        py = P(gx, 0.0)[1]
        d.line([(m, py), (w - m, py)], fill=C_GRID if gx else (66, 78, 94),
               width=1)
        d.text((m + 3, py - 13), f"{gx} m", fill=C_DIM, font=F["micro"])
    for gy in range(-int(y_half), int(y_half) + 1, 5):
        px = m + (w - 2 * m) * (0.5 - gy / (2 * y_half))
        d.line([(px, m), (px, h - m)], fill=C_GRID, width=1)
    d.text((w // 2 - 74, h - m + 5), "+y LEFT   <->   -y RIGHT",
           fill=C_DIM, font=F["micro"])

    for path, col, wd, dash in series:
        if path is None:
            continue
        pts = [P(px, py) for px, py in densify(path, 96)]
        if dash:
            for i in range(0, len(pts) - 1, 6):
                d.line(pts[i:i + 4], fill=col, width=wd)
        else:
            d.line(pts, fill=col, width=wd, joint="curve")
        for px, py in path:
            q = P(px, py)
            r = max(3, wd // 2)
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=col)

    e = P(0.0, 0.0)
    d.polygon([(e[0], e[1] - 11), (e[0] - 7, e[1] + 7), (e[0] + 7, e[1] + 7)],
              fill=C_FG)
    d.text((m, m + 2), f"METRIC BEV - ego frame, {x_lo:.0f} to {x_hi:.0f} m",
           fill=C_FG, font=F["small"])
    for j, ln in enumerate(wrap(d, note, F["micro"], w - 2 * m)[:2]):
        d.text((m, 6 + 17 * j), ln, fill=C_WARN, font=F["micro"])
    d.text((w - m - 110, m + 4), f"v0 {ego_v:5.2f} m/s", fill=C_DIM,
           font=F["micro"])
    return im


def draw_wm(size, model, const, zero, F, mark_s=None):
    """The world-model strip: `wm_mse_model` against the PERSISTENCE and ZERO
    controls, over the diagnostic's own 30 steps = 6.0 s.

    ⭐ THIS IS THE PANEL THAT SHOWS WHAT IS WORKING. The planner's failure is
    everywhere else on the frame; the predictor beats persistence monotonically
    past the first tick, and a reel that only showed the planner would leave a
    viewer with the wrong impression of the model as a whole.

    ⛔ It is a T0 diagnostic — a teacher-forced world-model error — and the
    panel says so, because a T0 number presented beside trajectories reads as
    driving performance and is not (EVAL_DOCTRINE)."""
    w, h = size
    im = Image.new("RGB", (w, h), C_PANEL)
    d = ImageDraw.Draw(im)
    m_l, m_r, m_t, m_b = 62, 16, 46, 34
    n = int(len(model))
    ts = (np.arange(n) + 1) * WM_DT
    lo = 0.0
    hi = float(max(np.max(model), np.max(const), np.max(zero))) * 1.08 or 1.0

    def P(t_s, y):
        return (m_l + (w - m_l - m_r) * (t_s - WM_DT) / max(ts[-1] - WM_DT, 1e-9),
                (h - m_b) - (h - m_t - m_b) * (y - lo) / max(hi - lo, 1e-9))

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = lo + frac * (hi - lo)
        py = P(ts[0], y)[1]
        d.line([(m_l, py), (w - m_r, py)], fill=C_GRID, width=1)
        d.text((6, py - 8), f"{y:5.3f}", fill=C_DIM, font=F["micro"])
    for t_s in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        if t_s > ts[-1] + 1e-9:
            continue
        px = P(t_s, lo)[0]
        d.line([(px, m_t), (px, h - m_b)], fill=C_GRID, width=1)
        d.text((px - 10, h - m_b + 6), f"{t_s:.0f}s", fill=C_DIM, font=F["micro"])

    for arr, col, wd, lab in ((zero, (96, 108, 126), 2, "zero"),
                              (const, (170, 182, 200), 3, "persistence"),
                              (np.asarray(model), C_OK, 4, "MODEL")):
        d.line([P(t, float(v)) for t, v in zip(ts, arr)], fill=col, width=wd,
               joint="curve")
    if mark_s is not None and ts[0] - 1e-9 <= mark_s <= ts[-1] + 1e-9:
        px = P(float(mark_s), lo)[0]
        d.line([(px, m_t), (px, h - m_b)], fill=C_WARN, width=2)

    d.text((10, 6), "WORLD MODEL (T0 diagnostic, NOT driving performance)",
           fill=C_FG, font=F["small"])
    #: ⚠️ SIGN, STATED: `advantage` is persistence MINUS model, so POSITIVE
    #: means the model predicts BETTER than simply holding the last frame. The
    #: opposite convention reads identically on the page and inverts the finding.
    adv0 = float(const[0] - model[0])
    advN = float(const[-1] - model[-1])
    d.text((10, 25),
           fit(d, f"beats persistence by (POSITIVE = model wins): "
                  f"{adv0:+.4f} at 0.2 s  ->  {advN:+.4f} at {ts[-1]:.1f} s",
               F["micro"], w - 20),
           fill=C_OK if advN > 0 else C_BAD, font=F["micro"])
    lx = w - m_r - 210
    for i, (lab, col) in enumerate((("MODEL", C_OK), ("persistence", (170, 182, 200)),
                                    ("zero", (96, 108, 126)))):
        d.line([(lx, m_t + 8 + 16 * i), (lx + 22, m_t + 8 + 16 * i)],
               fill=col, width=4)
        d.text((lx + 28, m_t + 8 + 16 * i - 7), lab, fill=col, font=F["micro"])
    return im


def draw_arms_table(size, rows, F, hdr, foot=None):
    """One row per arm AND one per FLOOR: what the cost geometry bought.

    Every column is either read from the dump or computed here from arrays the
    dump carries; the friction column is the SCORER'S OWN flag and the curvature
    column is the EVAL'S OWN reduction (:func:`curvature_mae`).

    ⭐ THE FLOOR ROWS ARE IN THE SAME TABLE ON PURPOSE. The finding this reel
    exists to show is a COMPARISON WITH A FLOOR — *"the unpenalised arm tracks
    the road worse than a plan that never steers"* — and a floor quoted in a
    caption while the arms sit in a table is a claim the viewer has to take on
    trust. `is_floor` only changes the label, never the columns: every row's
    `max|k|` is the COMMANDED curvature and every row's curvature MAE is
    computed by the same function, so the rows are comparable by construction.

    ⚠ `foot` carries the PANEL aggregate. A per-window number can always be
    cherry-picked by pausing; the panel line is on every frame so it cannot be."""
    w, h = size
    im = Image.new("RGB", (w, h), C_PANEL)
    d = ImageDraw.Draw(im)
    for j, ln in enumerate(wrap(d, hdr, F["small"], w - 28)[:2]):
        d.text((14, 8 + 18 * j), ln, fill=C_FG, font=F["small"])
    cols = [(14, "arm / floor"), (190, "cost geometry"), (424, "ADE m"),
            (506, "curv MAE 1/m"), (614, "max|k| 1/m"), (712, "a m/s2"),
            (786, "plan source"), (920, "peak lat g"), (1030, "friction circle")]
    y = 46
    for x, t in cols:
        d.text((x, y), t, fill=C_DIM, font=F["micro"])
    y += 20
    d.line([(14, y), (w - 14, y)], fill=C_GRID, width=1)
    y += 6
    for r in rows:
        col = r["colour"]
        if r.get("is_floor"):
            d.rectangle([14, y + 7, 30, y + 13], fill=col)
        else:
            d.rectangle([14, y + 5, 30, y + 15], fill=col)
        d.text((38, y), r["arm"], fill=col, font=F["small"])
        d.text((190, y + 1), fit(d, r["geom"], F["micro"], 226), fill=C_DIM,
               font=F["micro"])
        d.text((424, y + 1), r["ade"], fill=C_FG, font=F["mono"])
        d.text((506, y + 1), r["curv"], fill=r.get("curv_col", C_FG),
               font=F["mono"])
        d.text((614, y + 1), r["kap"], fill=C_FG, font=F["mono"])
        d.text((712, y + 1), r["acc"], fill=C_FG, font=F["mono"])
        d.text((786, y + 1), fit(d, r["src"], F["micro"], 128), fill=C_DIM,
               font=F["micro"])
        d.text((920, y + 1), r["pg"], fill=r["pg_col"], font=F["mono"])
        d.text((1030, y + 1), r["feas"], fill=r["feas_col"], font=F["micro"])
        y += 24
    if foot:
        y += 4
        d.line([(14, y), (w - 14, y)], fill=C_GRID, width=1)
        y += 6
        for ln in wrap(d, foot, F["micro"], w - 28)[:3]:
            d.text((14, y), ln, fill=C_WARN, font=F["micro"])
            y += 17
    return im


def draw_decisions(size, D, F):
    """TACTICAL + STRATEGIC, by NAME, against the v7.2 labels.

    ⭐ These are SHARED by every arm on the frame — same checkpoint, same
    conditioning — and the panel says so. Only the PLANNER differs between the
    coloured lines above, and a viewer who thinks the heads differ too would
    attribute the trajectory spread to the wrong stage.

    ⛔ The dump stores argmaxes, not logits, so this shows what the head DECIDED
    and never a probability bar it does not have."""
    w, h = size
    im = Image.new("RGB", (w, h), C_PANEL)
    d = ImageDraw.Draw(im)
    d.text((14, 8), "THE MODEL'S DECISIONS - shared by every arm above (one "
                    "checkpoint, one conditioning); only the PLANNER differs",
           fill=C_FG, font=F["small"])

    def cell(x, y, title, pred, lab, classes, extra=None):   # noqa: ARG001
        d.text((x, y), title, fill=C_DIM, font=F["micro"])
        ls = classes[lab] if 0 <= lab < len(classes) else "-- (outside the v7.2 band)"
        ps = classes[pred] if 0 <= pred < len(classes) else f"?{pred}"
        ok = (0 <= lab < len(classes)) and (pred == lab)
        col = C_OK if ok else (C_BAD if 0 <= lab < len(classes) else C_DIM)
        d.text((x, y + 21), "GT   " + ls, fill=C_GT if 0 <= lab < len(classes)
               else C_DIM, font=F["body"])
        d.text((x, y + 45), "pred " + ps, fill=col, font=F["body"])
        if 0 <= lab < len(classes):
            d.text((x, y + 69), "MATCH" if ok else "MISS", fill=col, font=F["small"])
        if extra:
            d.text((x, y + 93), extra, fill=C_DIM, font=F["micro"])

    y0 = 36
    cell(14, y0, "TACTICAL - lateral (v7.0 vocab)", D["lat_pred"], D["lat_lab"],
         D["LAT"])
    cell(400, y0, "TACTICAL - longitudinal", D["lon_pred"], D["lon_lab"], D["LON"])
    cell(786, y0, "STRATEGIC - route", D["rt_pred"], D["rt_lab"], D["ROUTE"])
    x = 1160
    d.text((x, y0), "NAV COMMAND - a GIVEN INPUT, not a prediction",
           fill=C_DIM, font=F["micro"])
    nav = D["nav"]
    navs = D["NAV"][nav] if 0 <= nav < len(D["NAV"]) else f"?{nav}"
    d.text((x, y0 + 21), navs + ("" if D["nav_valid"] else "  (nav_valid FALSE)"),
           fill=C_GIVEN, font=F["body"])
    d.text((x, y0 + 48), fit(d, "the planner's decoded goal token per arm:",
                             F["micro"], w - x - 20), fill=C_DIM, font=F["micro"])
    yy = y0 + 66
    for g in D["goals"][:4]:
        d.text((x, yy), fit(d, g["text"], F["micro"], w - x - 20), fill=g["colour"],
               font=F["micro"])
        yy += 16
    d.text((14, h - 16),
           fit(d, "T0/T1 are CONDITIONING labels, not loop labels: every arm on "
                  "this frame is OPEN LOOP - the model controls nothing and the "
                  "ego data keeps arriving from the recording (PI ruling "
                  "2026-09-02).", F["micro"], w - 28),
           fill=C_DIM, font=F["micro"])
    return im


# =========================================================================== #
# THE GROUND-TRUTH CONTROL — the frame that decides whether anything is true   #
# =========================================================================== #
def gt_control(arms, order, episodes_dir, extr, F, out_png: str,
               v_min: float = 2.0, med_tol_m: float = 0.25,
               min_ratio: float = 2.0) -> dict:
    """THE CONTROL THAT MUST PASS BEFORE ANY OTHER FRAME IS BELIEVED.

    Take the RECORDED actions out of each clip's own `*.v2ep.pt`, feed them to
    the planner's integrator AS IF THEY WERE A PLAN, and require the result to
    land on the dumped ground truth. If it does not, the projection or the
    integrator is wrong and nothing else in the reel can be trusted.

    IT HAS TWO ARMS, AND THE SECOND IS WHY IT IS A CONTROL AT ALL.
    `v2ep actions[:, 0]` is a road-wheel STEER ANGLE `arctan(L*kappa)` at
    L = 2.9 m (RETRACTION_LOG #15). Read as `steer` it must reproduce `g`; read
    as `kappa` -- the LEGACY reading the dumps were rolled under -- it must NOT,
    by a known ~2.9x over-rotation. A control with only the passing arm cannot
    distinguish "the integrator is right" from "the tolerance is loose", so the
    criterion requires BOTH a small deviation under `steer` AND a much larger
    one under `kappa`. MEASURED on the p4 panel: 0.1552 m against 0.8727 m, a
    5.62x separation (8.57x over the turning windows alone).

    THE SPEED EXCLUSION IS THE PROGRAMME'S OWN, AND IT IS REPORTED, NOT HIDDEN.
    Below ~2 m/s the unicycle is ill-conditioned -- `kappa = a_lat / v^2` blows
    up and a 0.2 s speed difference is most of the speed -- so
    `raw/feas_audit.py` already restricts this rig's feasibility control to
    `v0 >= 2 m/s` for exactly this reason (at `vmin = 0` even the GROUND TRUTH
    reads max|kappa| 31.4 and its whole block is reported inadmissible). The
    same block is used here and the excluded windows are COUNTED AND PRINTED
    rather than dropped: an exclusion a reader cannot see is a moved goalpost.

    NOTHING IS RE-IMPLEMENTED: the actions are built by the loader's own
    `RefAV1Windows._kin_actions` and integrated by
    `refav1_arm.paths_from_controls` -> `refa_v1_plan.unicycle_paths` -- the
    same integrator every arm in the dump was rolled through.
    """
    from tanitad.data.refav1_loader import RefAV1Windows
    arm0 = arms[order[0]]
    grid = arm0["manifest"]["grid"]
    k = int(grid["horizon_k"])
    dt = float(grid["dt_s"])

    class _DtOnly:                     # the loader is read ONLY for its dt --
        dt = float(grid["dt_s"])       # `hold_ext_controls`' own loader=None path

    per = []                           # every window, both readings
    best = None                        # the frame to draw: most lateral, admissible
    for name, rec in arm0["eps"].items():
        pay = episode_payload(episodes_dir, name)
        v = pay["poses"][:, 3].float()
        kap = pay["actions"][:, 0].float()
        for i, t in enumerate(rec["ws"].astype(int)):
            ctrl = RefAV1Windows._kin_actions(_DtOnly(), v, kap, int(t), k)
            g = rec["g"][i].astype(np.float64)
            v0 = float(rec["v0"][i])
            paths, dev = {}, {}
            for units in ("steer", "kappa"):
                p = arm_tool.paths_from_controls(ctrl, v0, dt, k,
                                                 action_units=units)
                paths[units] = np.asarray(p[0].detach().cpu().numpy(),
                                          dtype=np.float64)
                e = np.linalg.norm(paths[units] - g, axis=1)
                dev[units] = (float(e.mean()), float(e.max()))
            lat = float(np.abs(g[:, 1]).max())
            per.append({"clip": name, "t_cache": int(t), "v0_mps": round(v0, 3),
                        "gt_lat_extent_m": round(lat, 3),
                        "ade_steer_m": round(dev["steer"][0], 4),
                        "max_steer_m": round(dev["steer"][1], 4),
                        "ade_kappa_m": round(dev["kappa"][0], 4),
                        "max_kappa_m": round(dev["kappa"][1], 4),
                        "admissible": bool(v0 >= v_min)})
            if v0 >= v_min and (best is None or lat > best["lat"]):
                best = {"name": name, "t": int(t), "i": i, "lat": lat, "v0": v0,
                        "g": g, "steer": paths["steer"], "kappa": paths["kappa"],
                        "pay": pay}
    if best is None:
        sys.exit(f"[arms-reel] the control has NO window at v0 >= {v_min} m/s -- "
                 f"it cannot be run on this panel, and a control that cannot run "
                 f"is not a control that passed")

    adm = [r for r in per if r["admissible"]]
    exc = [r for r in per if not r["admissible"]]
    s_ade = np.array([r["ade_steer_m"] for r in adm])
    k_ade = np.array([r["ade_kappa_m"] for r in adm])
    med, mx = float(np.median(s_ade)), float(s_ade.max())
    ratio = float(k_ade.mean() / max(s_ade.mean(), 1e-9))
    ok_dev = bool(med <= med_tol_m)
    ok_sep = bool(ratio >= min_ratio)
    ok = bool(ok_dev and ok_sep)

    # ---- draw it, so the control is EVIDENCE and not a log line ------------ #
    pay = best["pay"]
    fr = dict(pay["frame"])
    cid = str(pay.get("clip_id", best["name"]))
    proj = CylProjector(fr, extr.get(cid))
    get_frame, codec, magic = frame_decoder(pay)
    im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
    d = ImageDraw.Draw(im)
    d.text((PAD + 6, 12), "GROUND-TRUTH CONTROL - the RECORDED actions, "
                          "integrated as if they were a plan", fill=C_FG,
           font=F["h1"])
    d.text((PAD + 6, 50),
           fit(d, f"clip {cid[:8]}  -  window t = {best['t'] * dt:.1f} s, v0 "
                  f"{best['v0']:.2f} m/s  -  the most lateral ADMISSIBLE window "
                  f"in the panel  -  actions from RefAV1Windows._kin_actions, "
                  f"integrated by refav1_arm.paths_from_controls -> "
                  f"refa_v1_plan.unicycle_paths: the SAME integrator every arm "
                  f"in the dump was rolled through",
               F["micro"], W_TOT - 2 * PAD - 12), fill=C_DIM, font=F["micro"])
    img = get_frame(2 * best["t"]).permute(1, 2, 0).numpy()
    # ⛔ into the PANE, never onto the canvas — see the main loop's note: the
    # projector's one-raster margin otherwise draws the control's own
    # trajectories straight across the verdict text underneath it.
    cam = Image.fromarray(img).resize(
        (fr["width"] * CAM_UP, fr["height"] * CAM_UP), Image.NEAREST)
    dc = ImageDraw.Draw(cam)

    def proj_to(path):
        return proj(densify(path, 96), up=CAM_UP)

    hz = proj(np.array([[4000.0, 0.0]]), up=CAM_UP)
    if hz and hz[0] is not None:
        yy = hz[0][1]
        dc.line([(0, yy), (W_LEFT, yy)], fill=(90, 100, 118), width=1)
        dc.text((8, yy - 18), "horizon predicted from this clip's own "
                "extrinsics - if it is not on the skyline the projection is "
                "wrong", fill=(140, 150, 168), font=F["micro"])
    polyline(dc, proj_to(best["g"]), C_GT, 13)                  # GT, wide, under
    polyline(dc, proj_to(best["steer"]), (56, 189, 248), 4)     # must land ON it
    polyline(dc, proj_to(best["kappa"]), (244, 114, 182), 3)    # must NOT
    dc.rectangle([0, H_CAM - 30, W_LEFT, H_CAM - 1], fill=(10, 13, 19))
    lx = 12
    for lab, col in (("GROUND TRUTH (the dump's own g)", C_GT),
                     ("recorded actions as STEER - must lie on it", (56, 189, 248)),
                     ("the SAME actions read as KAPPA", (244, 114, 182))):
        dc.line([(lx, H_CAM - 18), (lx + 24, H_CAM - 18)], fill=col, width=5)
        dc.text((lx + 30, H_CAM - 26), lab, fill=col, font=F["micro"])
        lx += 40 + int(dc.textlength(lab, font=F["micro"]))
    im.paste(cam, (PAD, Y_CAM))
    span = float(min(60.0, 10.0 * math.ceil((max(6.0, float(np.abs(np.concatenate(
        [best["g"], best["steer"], best["kappa"]])).max())) * 1.15) / 10.0)))
    im.paste(draw_bev((W_RIGHT, H_CAM),
                      [(best["g"], C_GT, 13, False),
                       (best["steer"], (56, 189, 248), 4, False),
                       (best["kappa"], (244, 114, 182), 3, True)],
                      -5.0, span,
                      "GREEN = the dump's own GT. CYAN must lie on it. MAGENTA "
                      "is the same actions under the legacy unit reading.",
                      F, best["v0"]), (X_RIGHT, Y_CAM))
    yy = Y_MID + 4
    lines = [
        (f"EVERY ONE OF THE {len(per)} WINDOWS OF THE PANEL, under BOTH unit "
         f"readings:", C_FG),
        (f"ADMISSIBLE BLOCK   v0 >= {v_min:.0f} m/s, n = {len(adm)}   (the "
         f"exclusion raw/feas_audit.py already uses on this rig: below it "
         f"kappa = a_lat/v^2 is ill-conditioned and even the GROUND TRUTH reads "
         f"max|kappa| 31.4)", C_DIM),
        (f"    STEER reading    ADE mean {s_ade.mean():.4f} m    median "
         f"{med:.4f} m    p90 {float(np.percentile(s_ade, 90)):.4f} m    max "
         f"{mx:.4f} m       {'PASS' if ok_dev else 'FAIL'}  "
         f"(median must be <= {med_tol_m:.2f} m)", C_OK if ok_dev else C_BAD),
        (f"    KAPPA reading    ADE mean {k_ade.mean():.4f} m    median "
         f"{float(np.median(k_ade)):.4f} m       separation {ratio:.2f}x       "
         f"{'PASS' if ok_sep else 'FAIL'}  (must be >= {min_ratio:.1f}x, or the "
         f"control cannot tell a right integrator from a loose tolerance)",
         C_OK if ok_sep else C_BAD),
        (f"EXCLUDED BLOCK     v0 < {v_min:.0f} m/s, n = {len(exc)} - COUNTED AND "
         f"REPORTED, never dropped"
         + (f"   (steer ADE mean {np.mean([r['ade_steer_m'] for r in exc]):.4f} m "
            f"there, on near-stationary windows)" if exc else ""), C_WARN),
        (f"FOR SCALE: the best planner arm on this panel scores ADE 0.8838 m. "
         f"The recorded actions through this same integrator score "
         f"{s_ade.mean():.4f} m - {0.8838 / max(s_ade.mean(), 1e-9):.1f}x better "
         f"- so the geometry this reel draws is not what limits the arms.",
         C_DIM),
        (f"codec field {codec!r}, magic {magic!r} - read from the FIELD, never "
         f"from the buffer's name.  {proj.label()}", C_DIM),
        (f"VERDICT: {'CONTROL PASSES' if ok else 'CONTROL FAILS'} - "
         + ("the projection and the integrator are trustworthy, and every other "
            "frame in this reel rests on them" if ok else
            "NOTHING ELSE IN THIS REEL CAN BE TRUSTED"),
         C_OK if ok else C_BAD)]
    for txt, col in lines:
        for ln in (wrap(d, txt, F["body"], W_TOT - 2 * PAD - 24) or [""]):
            d.text((PAD + 12, yy), ln, fill=col, font=F["body"])
            yy += 25
    im.save(out_png)
    return {"n_windows": len(per), "v_min_mps": v_min,
            "n_admissible": len(adm), "n_excluded": len(exc),
            "steer_ade_mean_m": round(float(s_ade.mean()), 4),
            "steer_ade_median_m": round(med, 4),
            "steer_ade_max_m": round(mx, 4),
            "kappa_ade_mean_m": round(float(k_ade.mean()), 4),
            "separation_x": round(ratio, 3),
            "median_tolerance_m": med_tol_m, "min_separation_x": min_ratio,
            "pass_deviation": ok_dev, "pass_separation": ok_sep, "pass": ok,
            "drawn_window": {"clip": best["name"], "t_cache": best["t"],
                             "v0_mps": round(best["v0"], 3)},
            "per_window": per, "png": os.path.abspath(out_png)}


# =========================================================================== #
# CONTENT ASSERTIONS — a black reel reports success identically to a good one  #
# =========================================================================== #
def assert_frame_content(im: Image.Image, paths, where: str, stats: dict):
    """⛔ ASSERT ON CONTENT, NEVER ON SIZE OR EXIT CODE.

    A renderer that decoded nothing writes a full-size, perfectly valid, BLACK
    frame and exits 0 — the E-DETECT-1 failure in a video costume. So every
    frame is checked for a plausible non-zero mean, and every overlaid path for
    finiteness and a sane magnitude, before it is written."""
    a = np.asarray(im.convert("L"), dtype=np.float64)
    mu = float(a.mean())
    if not (2.0 <= mu <= 250.0):
        sys.exit(f"[arms-reel] frame {where} has mean luminance {mu:.3f} — a "
                 f"black or blown frame. REFUSING to write a reel whose frames "
                 f"carry no image.")
    for tag, p in paths:
        if p is None:
            continue
        if not np.all(np.isfinite(p)):
            sys.exit(f"[arms-reel] frame {where}: path {tag} is not finite")
        if np.abs(p).max() > 500.0:
            sys.exit(f"[arms-reel] frame {where}: path {tag} reaches "
                     f"{np.abs(p).max():.1f} m — implausible for a 2.0 s plan; "
                     f"refusing rather than drawing it off-panel")
    stats["lum_sum"] += mu
    stats["lum_n"] += 1
    stats["lum_min"] = min(stats["lum_min"], mu)
    stats["lum_max"] = max(stats["lum_max"], mu)


# =========================================================================== #
# MAIN                                                                        #
# =========================================================================== #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms-dir", required=True,
                    help="the dir holding dump_<arm>/ per arm")
    ap.add_argument("--arms", required=True,
                    help="comma-separated arm names, in draw order (max 6)")
    ap.add_argument("--episodes", required=True, help="a dir of *.v2ep.pt")
    ap.add_argument("--extrinsics", default=None,
                    help="clip_id -> MEASURED extrinsic (pai_extrinsics_table.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--slowmo", type=int, default=5,
                    help="hold LATERALLY-DEMANDING windows this many times "
                         "longer (1 = off). See --slowmo-lat-m.")
    ap.add_argument("--slowmo-lat-m", type=float, default=1.0,
                    help="a window is laterally demanding if its v7.2 lateral "
                         "label is a turn OR its GROUND-TRUTH lateral extent "
                         "reaches this many metres. BOTH halves are properties "
                         "of the GROUND TRUTH alone, fixed before any arm is "
                         "read -- never 'the windows where the arms differ "
                         "most', which would be choosing the evidence by the "
                         "answer. MEASURED on the p4 panel the two halves "
                         "select 4 and 9 windows and 13 of 40 in union, with a "
                         "clean gap in the GT extent between 0.83 m and 2.02 m.")
    ap.add_argument("--base-hold", type=int, default=2,
                    help="hold EVERY other frame this many times. Real time is "
                         "10 Hz, so 1 is real speed and 2 is half speed; the "
                         "factor is printed on every frame it applies to.")
    ap.add_argument("--expect-step", type=int, default=None)
    ap.add_argument("--span-m", type=float, default=50.0)
    ap.add_argument("--max-plan-age", type=float, default=1.0,
                    help="stop animating a window once the plan is this old "
                         "(s). Past it the remaining plan is a stub of under "
                         "0.6 s that lands under the camera's own nose and "
                         "projects to nothing; the frames are SKIPPED and "
                         "COUNTED, never drawn as a shrinking plan.")
    ap.add_argument("--title-frames", type=int, default=16)
    ap.add_argument("--card-frames", type=int, default=80)
    ap.add_argument("--cards", default=None, help="JSON with intro/outro cards")
    ap.add_argument("--notes", default=None, help="JSON clip_id -> one-line note")
    ap.add_argument("--gt-control", action="store_true",
                    help="render and ASSERT the ground-truth integrator control")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-small", action="store_true")
    ap.add_argument("--crf", type=int, default=20,
                    help="x264 quality for the FULL copy. \u26a0 30 MiB is a "
                         "DELIVERY GATE, not a preference (verify_mp4.py: a "
                         "38.74 MiB reel was REFUSED by the PI's channel), and "
                         "at 10 fps a longer reel crosses it. Raising this by 1 "
                         "is the cheapest way back under the line; the small "
                         "copy is always re-encoded from the ORIGINAL frames.")
    ap.add_argument("--arm-colours", default=None,
                    help="pin a hue to an ARM rather than to its position in "
                         "--arms, e.g. ccos_argmax=magenta,wk15=orange,"
                         "best=violet. Without it the colours move when the arm "
                         "ORDER moves, and two cuts of the same reel stop being "
                         "comparable. Names: " + ", ".join(sorted(NAMED_COLOURS)))
    ap.add_argument("--expect-curv-mae", default=None,
                    help="THE CONTENT CONTROL for the curvature story, e.g. "
                         "ha0=0.040083,wk15=0.030982,ccos_argmax=0.055369. Each "
                         "named series' PANEL aggregate is recomputed here by "
                         "taniteval.four_families and must reproduce the banked "
                         "value in raw/four_family_all.txt, or the render is "
                         "REFUSED. A per-frame curvature that cannot reproduce "
                         "the banked panel is a different metric wearing its name.")
    ap.add_argument("--slowmo-turn", type=int, default=0,
                    help="hold a window whose v7.2 lateral LABEL is a turn this "
                         "many times longer (0 = use --slowmo). The turn windows "
                         "are where the curvature story lives, and they are "
                         "selected by the GROUND-TRUTH LABEL alone -- never by "
                         "where the arms happen to differ.")
    ap.add_argument("--no-echo", action="store_true",
                    help="drop the ha0_ext echo control from the overlay. The "
                         "FLOOR this reel argues against is ha0 (perfectly "
                         "straight); ha0_ext is drawn thin and dim beneath it as "
                         "the ADE floor, and can be dropped if it clutters.")
    a = ap.parse_args(argv)

    order = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not order:
        sys.exit("[arms-reel] --arms named nothing")
    if len(order) > len(ARM_COLOURS):
        sys.exit(f"[arms-reel] {len(order)} arms exceeds the {len(ARM_COLOURS)} "
                 f"distinguishable colours this reel defines. More lines than a "
                 f"viewer can tell apart is not more information.")
    arms = {n: read_arm(a.arms_dir, n) for n in order}
    colours = resolve_arm_colours(order, a.arm_colours)
    shared = verify_shared(arms, order)
    _p(f"[verify] {len(order)} arms share {len(shared['shared_keys_checked'])} "
       f"arrays bit-identically over {shared['n_windows']} windows; "
       f"{shared['arms_differing_on_cl']}/{len(order)-1} differ from "
       f"{order[0]} on the plan (must be > 0)")

    man = arms[order[0]]["manifest"]
    step = int(man["model"].get("step", -1))
    if a.expect_step is not None and step != a.expect_step:
        sys.exit(f"[arms-reel] dump is step {step}, --expect-step {a.expect_step}")
    ckpt = man["model"].get("ckpt", "?")
    load_rep = man["model"].get("state_dict_load", {})
    strict = not load_rep.get("missing_keys") and not load_rep.get("unexpected_keys")
    grid = man.get("grid", {})
    k_h = int(grid.get("horizon_k", 10))
    horizon_s = k_h * DT_CACHE
    plan_cfg = man.get("plan_cfg", {})

    #: the arm's cost geometry, read from ITS OWN manifest — never inherited
    #: from a table, because the geometry is the independent variable here.
    geom = {}
    for n in order:
        c = arms[n]["manifest"]["cost"]
        gr = arms[n]["manifest"].get("goal_rule", {})
        bits = [f"{c['metric']} W_KAPPA {c['weights']['W_KAPPA']:g}"]
        if gr.get("kamm_mu") is not None:
            bits.append(f"Kamm cap mu={gr['kamm_mu']}")
        if gr.get("seed_kappa_ladder"):
            bits.append("seed ladder")
        geom[n] = " + ".join(bits)
        _p(f"[arm   ] {n:14s} {geom[n]}")

    names = list(arms[order[0]]["eps"].keys())
    notes = {}
    if a.notes:
        with open(a.notes, encoding="utf-8") as fh:
            notes = json.load(fh)
    extr = load_extrinsics(a.extrinsics, names)
    _p(f"[calib ] per-clip extrinsics for {len(extr)}/{len(names)} clips")

    # ---- CURVATURE: per window for the frames, POOLED for the control ----- #
    #: ⭐ THE ONE NUMBER THIS CUT EXISTS TO PUT ON SCREEN. The previous cut drew
    #: `ha0_ext` as "the floor" and called `ccos_argmax` "the arm that turns";
    #: both readings are refuted (M74/M75/M77/M79). The reachable metric — one
    #: the human passes BY CONSTRUCTION, unlike the |dyaw| > 0.15 turn gate she
    #: fails 3 of 9 — is curvature MAE against a PERFECTLY STRAIGHT floor.
    echo_on = not a.no_echo
    ref_eps = arms[order[0]]["eps"]
    floors = ["ha0"] + (["ha0_ext"] if echo_on else [])
    series_of = {n: {e: arms[n]["eps"][e]["cl"] for e in names} for n in order}
    for fl in floors:
        series_of[fl] = {e: ref_eps[e][fl] for e in names}
    G_by_ep = {e: ref_eps[e]["g"] for e in names}
    G_all = np.concatenate([G_by_ep[e] for e in names], 0)

    curv_pw, curv_panel = {}, {}
    for k, by_ep in series_of.items():
        curv_pw[k] = {e: curvature_mae(by_ep[e], G_by_ep[e])[0] for e in names}
        _, pooled, nst = curvature_mae(
            np.concatenate([by_ep[e] for e in names], 0), G_all)
        curv_panel[k] = {"pooled": pooled, "n_steps": nst,
                         "n_windows": int(G_all.shape[0])}
    expect = {}
    if a.expect_curv_mae:
        for tok in a.expect_curv_mae.split(","):
            kk, _, vv = tok.strip().partition("=")
            expect[kk] = float(vv)
    curv_rep = curvature_control(curv_panel, expect)
    rank = sorted(curv_panel, key=lambda k: curv_panel[k]["pooled"])
    curv_foot = ("PANEL, n=" + str(int(G_all.shape[0])) + " windows / "
                 + str(len(names)) + " episodes, curvature MAE 1/m pooled over "
                 "valid steps (taniteval.four_families): "
                 + "  <  ".join(f"{k} {curv_panel[k]['pooled']:.6f}" for k in rank)
                 + "   ->   an arm ABOVE ha0 tracks the road WORSE THAN A PLAN "
                   "THAT NEVER STEERS.")
    _p("[curv  ] " + curv_foot)

    F = {"h1": font(30, True), "h2": font(21, True), "body": font(18),
         "small": font(16, True), "micro": font(14), "mono": font(15)}
    LAT = list(tactical_lat_actions("v7.0"))
    LON = list(tactical_lon_actions_v("v7.0"))
    ROUTE, NAVN = list(ROUTE_CLASSES), list(NAV_COMMANDS)
    turn_ids = {i for i, s in enumerate(LAT) if s in TURN_TOKENS}
    #: ⭐ THE TURN WINDOWS ARE ENUMERATED UP FRONT AND NAMED ON THE FRAME. They
    #: are selected by the v7.2 GROUND-TRUTH LABEL alone, before any arm is read,
    #: so "more screen time for the turns" cannot become "more screen time where
    #: my arm wins". Each frame prints which turn window it is, out of how many.
    turn_windows = [(e, int(wi), LAT[int(l)])
                    for e in names
                    for wi, l in enumerate(ref_eps[e]["lat_label"].tolist())
                    if 0 <= int(l) < len(LAT) and LAT[int(l)] in TURN_TOKENS]
    turn_rank = {(e, wi): (i + 1, len(turn_windows))
                 for i, (e, wi, _) in enumerate(turn_windows)}
    _p(f"[turns ] {len(turn_windows)} GT-LABELLED turn windows: "
       + ", ".join(f"{e[:8]}#{wi} {tok}" for e, wi, tok in turn_windows))

    frames_dir = a.frames_dir or (os.path.splitext(a.out)[0] + "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    for old in glob.glob(os.path.join(frames_dir, "f_*.png")):
        os.remove(old)

    # ---- THE CONTROL, FIRST: nothing else is worth rendering if it fails --- #
    control = None
    if a.gt_control:
        control = gt_control(arms, order, a.episodes, extr, F,
                             os.path.splitext(a.out)[0] + "_GT_CONTROL.png")
        _p(f"[control] n={control['n_windows']} windows, admissible "
           f"{control['n_admissible']} at v0>={control['v_min_mps']:.0f} m/s "
           f"(excluded {control['n_excluded']}); STEER ADE mean "
           f"{control['steer_ade_mean_m']:.4f} median "
           f"{control['steer_ade_median_m']:.4f} max {control['steer_ade_max_m']:.4f} m; "
           f"KAPPA ADE mean {control['kappa_ade_mean_m']:.4f} m = "
           f"{control['separation_x']:.2f}x separation -> "
           f"{'PASS' if control['pass'] else 'FAIL'}")
        _p(f"[control] {control['png']}")
        if not control["pass"]:
            sys.exit("[arms-reel] REFUSING to render: the ground-truth "
                     "integrator/projection control FAILED. Every trajectory in "
                     "the reel would be untrustworthy.")

    cards = {}
    if a.cards:
        with open(a.cards, encoding="utf-8") as fh:
            cards = json.load(fh)
    card_foot = (f"refav1 step {step}  ·  {'STRICT load' if strict else 'NON-STRICT LOAD'}"
                 f"  ·  arms {', '.join(order)}  ·  {arms[order[0]]['dir']}")

    #: ⛔ CHECKED BEFORE A SINGLE FRAME IS DRAWN, not after 2,073 of them.
    if (ra.W_TOT, ra.H_TOT) != (W_TOT, H_TOT):
        sys.exit(f"[arms-reel] this reel's canvas is {(W_TOT, H_TOT)} but the "
                 f"imported card renderer paints {(ra.W_TOT, ra.H_TOT)}. Mixed "
                 f"frame sizes are silently rescaled by ffmpeg. REFUSING "
                 f"rather than shipping a stretched reel.")

    n_out = 0
    index: list[dict] = []
    stats = dict(frames=0, clips=0, scored=0, slowed=0, skipped_stale=0,
                 cam_on=0, cam_off=0, lum_sum=0.0, lum_n=0, lum_min=1e9,
                 lum_max=-1e9, decoded=0, cam_stub=0,
                 #: ⚠ `slowed` counts every frame held MORE THAN ONCE, which with
                 #: --base-hold 2 is every clip frame and therefore says nothing.
                 #: `turn_frames` is the number that answers "did the turns get
                 #: more screen time", so it is counted separately.
                 turn_windows=0, turn_frames=0)
    t_start = time.time()

    def emit(im, meta):
        nonlocal n_out
        # ⛔ EVERY frame, cards included, must be the SAME size — see the H_BOT
        # note. A mixed-size sequence is silently rescaled by ffmpeg.
        if im.size != (W_TOT, H_TOT):
            sys.exit(f"[arms-reel] frame {n_out} is {im.size}, not "
                     f"{(W_TOT, H_TOT)}. A mixed-size image sequence is "
                     f"SILENTLY rescaled by ffmpeg's demuxer to the FIRST "
                     f"frame's size — the reel would ship distorted and every "
                     f"exit code would still read 0. REFUSING.")
        im.save(os.path.join(frames_dir, f"f_{n_out:06d}.png"))
        index.append(dict(meta, frame=n_out))
        n_out += 1

    if cards.get("intro"):
        im = ra.draw_card(cards["intro"], F, card_foot)
        for _ in range(a.card_frames):
            emit(im, {"kind": "card", "card": "intro"})

    for ci, name in enumerate(names):
        pay = episode_payload(a.episodes, name)
        get_frame, codec, magic = frame_decoder(pay)
        poses = pay["poses"].numpy().astype(np.float64)
        fr = dict(pay["frame"])
        cid = str(pay.get("clip_id", name))
        try:
            proj = CylProjector(fr, extr.get(cid))
        except rv.ProjectionRefused as e:
            _p(f"[clip  ] {cid}: {e} — camera overlay off")
            proj = None
        cam_on = bool(proj is not None and proj.enabled)
        stats["cam_on" if cam_on else "cam_off"] += 1
        ref = arms[order[0]]["eps"][name]
        ws = ref["ws"].astype(int)
        note = notes.get(cid, "")
        _p(f"[clip {ci+1}/{len(names)}] {cid} codec={codec} windows={len(ws)} "
           f"cam={'ON' if cam_on else 'OFF'}")

        for _ in range(a.title_frames):
            im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
            d = ImageDraw.Draw(im)
            d.text((PAD + 40, 300), f"CLIP {ci + 1} of {len(names)}", fill=C_DIM,
                   font=F["h2"])
            d.text((PAD + 40, 342), cid, fill=C_FG, font=F["h1"])
            for j, ln in enumerate(wrap(d, note, F["h2"], W_TOT - 160)):
                d.text((PAD + 40, 400 + 34 * j), ln, fill=C_WARN, font=F["h2"])
            d.text((PAD + 40, 520),
                   f"refav1 step {step}  ·  OPEN LOOP  ·  {len(ws)} scored "
                   f"windows  ·  {len(order)} planner arms on the same scene",
                   fill=C_DIM, font=F["body"])
            emit(im, {"kind": "title", "clip_id": cid, "clip_order": ci})
            if a.max_frames and n_out >= a.max_frames:
                break
        if a.max_frames and n_out >= a.max_frames:
            break

        for wi in range(len(ws)):
            t_w = int(ws[wi])
            lat_lab = int(ref["lat_label"][wi])
            gt_lat = float(np.abs(ref["g"][wi][:, 1]).max())
            by_label = lat_lab in turn_ids
            by_geom = gt_lat >= float(a.slowmo_lat_m)
            is_turn = bool(by_label or by_geom)
            #: ⭐ A LABELLED TURN GETS ITS OWN, LONGER HOLD. The curvature story
            #: is decided on the windows where the road actually bends, and at
            #: 10 fps a 2.0 s plan is 20 frames -- long enough to miss.
            hold = max(1, (a.slowmo_turn or a.slowmo) if by_label
                       else (a.slowmo if is_turn else a.base_hold))
            trank = turn_rank.get((name, wi))
            age_cap = min(float(a.max_plan_age), horizon_s - DT_FRAME)
            f_last = min(2 * t_w + int(round(age_cap / DT_FRAME)),
                         poses.shape[0] - 1)
            stats["scored"] += 1
            if by_label:
                stats["turn_windows"] += 1

            # ⭐ THE BEV RANGE IS FIXED FOR THE WHOLE WINDOW, and it is
            # MEASURED from the paths this window will actually draw — at its
            # FIRST frame (the largest forward reach) and at its LAST (the
            # largest recession behind the ego). A range recomputed per frame
            # makes the grid crawl and two moments incomparable by eye; a range
            # guessed from v0 * age wastes half the panel on empty road, which
            # is how a 1 m lateral difference ends up invisible.
            _paths0 = [ref["g"][wi]] + [ref[fl][wi] for fl in floors] + \
                      [arms[n]["eps"][name]["cl"][wi] for n in order]
            _pf, _pl = poses[2 * t_w], poses[min(f_last, poses.shape[0] - 1)]
            _pathsL = [se2_carry(Q, _pf, _pl) for Q in _paths0]
            _hi = max(float(max(Q[:, 0].max() for Q in _paths0)), 6.0)
            _lo = min(float(min(Q[:, 0].min() for Q in _pathsL)), 0.0)
            bev_hi = float(min(a.span_m, 5.0 * math.ceil((_hi * 1.12) / 5.0)))
            bev_lo = float(-5.0 * math.ceil((abs(_lo) + 3.0) / 5.0))

            # ---- per-arm, per-window quantities: read, or the scorer's own -- #
            rows, goal_of = [], {}
            #: the straight floor's own curvature error on THIS window — the bar
            #: every arm row is coloured against, so the comparison is IN the
            #: table rather than in a caption the viewer has to remember.
            c_floor = float(curv_pw["ha0"][name][wi])
            for arm in order:
                r = arms[arm]["eps"][name]
                cl, ctrl = r["cl"][wi], r["cl_controls"][wi]
                fe = window_feasibility(cl)
                ade = float(np.linalg.norm(cl - r["g"][wi], axis=1).mean())
                cvw = float(curv_pw[arm][name][wi])
                si = int(r["plan_source_cl"][wi])
                src = (PLAN_SOURCE_NAMES[si] if 0 <= si < len(PLAN_SOURCE_NAMES)
                       else f"idx{si}")
                rows.append(dict(
                    arm=arm, colour=colours[arm], geom=geom[arm],
                    ade=f"{ade:6.3f}",
                    curv="   n/a " if np.isnan(cvw) else f"{cvw:8.5f}",
                    #: ⚠ A TIE IS NOT A FAILURE. On a window where the road
                    #: does not bend, an arm that lies exactly on the straight
                    #: floor has the floor's error, and coloring that RED would
                    #: report the correct behaviour as a defect. The band is the
                    #: DISPLAY precision (%8.5f), so the colour never contradicts
                    #: the number printed beside it.
                    curv_col=curv_colour(cvw, c_floor),
                    kap=f"{np.abs(ctrl[:, 1]).max():7.4f}",
                    acc=f"{ctrl[0, 0]:+6.2f}", src=src,
                    pg=f"{fe['peak_g']:6.3f}",
                    pg_col=C_BAD if fe["kamm_over"] else C_FG,
                    feas="OUTSIDE (mu=0.7)" if fe["kamm_over"] else "inside",
                    feas_col=C_BAD if fe["kamm_over"] else C_OK,
                    _ade=ade, _kamm=fe["kamm_over"], _curv=cvw))
                gl, gn = int(r["goal_lat_cl"][wi]), int(r["goal_lon_cl"][wi])
                goal_of[arm] = (f"{LAT[gl] if 0 <= gl < len(LAT) else '--'}"
                                f" / {LON[gn] if 0 <= gn < len(LON) else '--'}")
            #: ⛔ THE FLOOR ROWS. `ha0` is the bar the reel's claim names; its
            #: commanded curvature is EXACTLY 0 by construction, so its `max|k|`
            #: is a constant and not a measurement. `ha0_ext` carries its own
            #: COMMANDED controls in the dump, so both floor rows quote the same
            #: quantity the arm rows do.
            for fl in floors:
                fp = ref_eps[name][fl][wi]
                ffe = window_feasibility(fp)
                fade = float(np.linalg.norm(fp - ref_eps[name]["g"][wi],
                                            axis=1).mean())
                fcv = float(curv_pw[fl][name][wi])
                if fl == "ha0":
                    fkap, facc = " 0.0000", " +0.00"
                    fgeom = "FLOOR: do nothing (a=0, kappa=0 at v0)"
                    fcol, fsrc = C_FLOOR, "no planner"
                else:
                    fc = ref_eps[name]["ha0_ext_controls"][wi]
                    fkap = f"{np.abs(np.atleast_2d(fc)[:, 1]).max():7.4f}"
                    facc = f"{np.atleast_2d(fc)[0, 0]:+6.2f}"
                    fgeom = "ECHO: hold the MEASURED (a0, kappa0)"
                    fcol, fsrc = C_ECHO, "no planner"
                rows.append(dict(
                    arm=fl, colour=fcol, geom=fgeom, is_floor=True,
                    ade=f"{fade:6.3f}",
                    curv="   n/a " if np.isnan(fcv) else f"{fcv:8.5f}",
                    curv_col=C_FG, kap=fkap, acc=facc, src=fsrc,
                    pg=f"{ffe['peak_g']:6.3f}",
                    pg_col=C_BAD if ffe["kamm_over"] else C_FG,
                    feas="OUTSIDE (mu=0.7)" if ffe["kamm_over"] else "inside",
                    feas_col=C_BAD if ffe["kamm_over"] else C_OK,
                    _ade=fade, _kamm=ffe["kamm_over"], _curv=fcv))

            for f in range(2 * t_w, f_last + 1):
                plan_age_s = (f - 2 * t_w) * DT_FRAME
                if plan_age_s > age_cap + 1e-9:
                    stats["skipped_stale"] += 1
                    continue
                p_from, p_to = poses[2 * t_w], poses[f]
                carry = lambda P: se2_carry(P, p_from, p_to)      # noqa: E731
                g_c = carry(ref["g"][wi])
                fl_c = carry(ref["ha0"][wi])          # THE STRAIGHT FLOOR
                ec_c = carry(ref["ha0_ext"][wi]) if echo_on else None
                arm_c = {n: carry(arms[n]["eps"][name]["cl"][wi]) for n in order}

                im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
                d = ImageDraw.Draw(im)
                elements = []

                # ---- banner
                d.rectangle([0, Y_BAN, W_TOT, Y_BAN + H_BAN], fill=C_PANEL)
                d.text((PAD + 6, Y_BAN + 6),
                       f"refav1  ·  step {step}  ·  OPEN LOOP  ·  clip {cid[:8]}"
                       f"  ·  t = {t_w * DT_CACHE:5.1f} s  ·  window {wi+1}/{len(ws)}"
                       f"  ·  plan age {plan_age_s:.1f} s",
                       fill=C_FG, font=F["h2"])
                d.text((PAD + 6, Y_BAN + 40),
                       fit(d, f"{len(order)} planner arms, ONE checkpoint, ONE "
                              f"scene - they differ ONLY in the cost geometry."
                              f"  ·  {'STRICT load' if strict else 'NON-STRICT LOAD'}"
                              f"  ·  iCEM {{samples {plan_cfg.get('n_samples')}, "
                              f"iters {plan_cfg.get('n_iters')}}}"
                              f"  ·  every line is READ from a banked dump - no "
                              f"model runs in this renderer",
                           F["micro"], W_TOT - 2 * PAD - 600),
                       fill=C_DIM, font=F["micro"])
                #: ⭐ THE TURN BADGE IS ITS OWN LINE AND IT NAMES THE WINDOW.
                #: "give the turns more screen time" is only honest if the viewer
                #: can SEE which windows those are and count them; the badge says
                #: which labelled turn this is, out of how many on the panel.
                right = []
                if trank:
                    right.append((f"GT {LAT[lat_lab]}  -  LABELLED TURN WINDOW "
                                  f"{trank[0]} of {trank[1]}", C_WARN, F["h2"]))
                if hold > 1:
                    why = ([] if not by_label else [f"GT {LAT[lat_lab]}"]) + \
                          ([] if not by_geom else [f"GT swings {gt_lat:.1f} m"])
                    tag = (f"SLOW MOTION 1/{hold}"
                           + (f"  -  {' + '.join(why)}" if why else ""))
                    right.append((tag, C_DIM if trank else C_WARN,
                                  F["small"] if trank else F["h2"]))
                for _j, (_t, _c, _f) in enumerate(right[:2]):
                    d.text((W_TOT - 12 - int(d.textlength(_t, font=_f)),
                            Y_BAN + 6 + 30 * _j), _t, fill=_c, font=_f)

                # ---- camera
                if cam_on:
                    img = get_frame(f).permute(1, 2, 0).numpy()
                    stats["decoded"] += 1
                    # ⛔ THE OVERLAY IS DRAWN INTO THE PANE, NOT ONTO THE CANVAS.
                    # `CylProjector` deliberately keeps a point up to ONE RASTER
                    # outside the image (`-h <= v <= 2h`) so a polyline heading
                    # just off-frame still connects — but on a shared canvas
                    # those coordinates land in the ARMS TABLE and the DECISIONS
                    # panel, and a trajectory drawn across a table of numbers
                    # reads as a defect in the numbers. Drawing into a pane-sized
                    # layer makes escaping structurally impossible, rather than
                    # relying on every projected point being in range.
                    cam = Image.fromarray(img).resize(
                        (fr["width"] * CAM_UP, fr["height"] * CAM_UP),
                        Image.NEAREST)            # NEAREST: no pixel invented
                    dc = ImageDraw.Draw(cam)

                    def proj_to(path):
                        return proj(densify(path, 96), up=CAM_UP)

                    hz = proj(np.array([[4000.0, 0.0]]), up=CAM_UP)
                    if hz and hz[0] is not None:
                        yy = hz[0][1]
                        dc.line([(0, yy), (W_LEFT, yy)],
                                fill=(90, 100, 118), width=1)
                        dc.text((8, yy - 18),
                                "horizon predicted from this clip's own "
                                "extrinsics - if it is not on the skyline the "
                                "projection is wrong", fill=(140, 150, 168),
                                font=F["micro"])
                    # ⛔ DRAW ORDER IS LOAD-BEARING, AND IT COST A RENDER TO
                    # GET RIGHT. The FLOOR goes down first and WIDE, so an arm
                    # that coincides with it shows the white as a halo and the
                    # viewer reads "these are the same path" rather than "one is
                    # missing". But the GROUND TRUTH must go ON TOP of the
                    # floor: drawn underneath a 13 px white line it disappeared
                    # entirely on every straight window, and a reel whose
                    # reference line is invisible is worse than no reel.
                    polyline(dc, proj_to(fl_c), C_FLOOR, 13)   # STRAIGHT floor
                    if ec_c is not None:
                        polyline(dc, proj_to(ec_c), C_ECHO, 3)  # echo, thin
                    polyline(dc, proj_to(g_c), C_GT, 8)        # GT, never hidden
                    n_vis = {}
                    for n in order:
                        pts = proj_to(arm_c[n])
                        n_vis[n] = sum(1 for q in pts if q is not None)
                        polyline(dc, pts, colours[n], 4)
                    # ⛔ AN EMPTY CAMERA PANEL MUST SAY WHY. The projector DROPS
                    # a point nearer than 0.5 m radially rather than clamping
                    # it, and the camera sits ~2 m FORWARD of the vehicle
                    # origin — so a short remaining plan at low speed lands
                    # under the hood and nothing is drawn. Silence there reads
                    # as "the model predicted nothing", which is a different
                    # claim entirely.
                    if max(n_vis.values() or [0]) < 2:
                        stats["cam_stub"] += 1
                        dc.text((14, H_CAM - 62),
                                fit(dc, f"the remaining "
                                        f"{max(0.0, horizon_s - plan_age_s):.1f} s "
                                        f"of every plan projects BELOW this "
                                        f"image - the camera sits ~2 m forward "
                                        f"of the vehicle origin and the "
                                        f"projector DROPS a point rather than "
                                        f"clamping it. The BEV carries this "
                                        f"frame.", F["micro"], W_LEFT - 28),
                                fill=C_WARN, font=F["micro"])
                    dc.rectangle([0, H_CAM - 30, W_LEFT, H_CAM - 1],
                                 fill=(10, 13, 19))
                    lx, ly = 12, H_CAM - 26
                    for lab, col in ([("GROUND TRUTH (the human)", C_GT),
                                      ("ha0 STRAIGHT FLOOR (never steers)",
                                       C_FLOOR)]
                                     + ([("ha0_ext echo control", C_ECHO)]
                                        if echo_on else [])
                                     + [(n, colours[n]) for n in order]):
                        dc.line([(lx, ly + 8), (lx + 24, ly + 8)], fill=col, width=5)
                        dc.text((lx + 30, ly), lab, fill=col, font=F["micro"])
                        lx += 34 + int(dc.textlength(lab, font=F["micro"]))
                    im.paste(cam, (PAD, Y_CAM))
                    elements.append(VizElement(
                        "camera", "present", kind="derived",
                        value=f"GT + ha0 STRAIGHT floor"
                              + (" + ha0_ext echo" if echo_on else "")
                              + f" + {len(order)} arms projected, "
                                f"plan age {plan_age_s:.1f} s",
                        source="CylProjector(v2ep['frame'], per-clip "
                               "sensor_extrinsics) on the dumps' g/ha0/ha0_ext/cl, "
                               f"drawn on the v2ep PNG at raw index {f}"))
                else:
                    d.rectangle([PAD, Y_CAM, PAD + W_LEFT, Y_CAM + H_CAM],
                                fill=C_PANEL)
                    for j, ln in enumerate(wrap(
                            d, "CAMERA OVERLAY DISABLED - no MEASURED per-clip "
                               "extrinsics. The BEV carries the comparison; a "
                               "constant camera height is wrong by metres exactly "
                               "where the trajectory is.", F["body"], W_LEFT - 60)):
                        d.text((PAD + 30, Y_CAM + 60 + 28 * j), ln, fill=C_WARN,
                               font=F["body"])
                    elements.append(VizElement(
                        "camera", "unavailable",
                        reason="no MEASURED per-clip extrinsic; refusing to "
                               "approximate with a constant camera height"))
                d.rectangle([PAD, Y_CAM, PAD + W_LEFT, Y_CAM + H_CAM],
                            outline=C_GRID, width=1)

                # ---- BEV
                arm_rows = [r for r in rows if not r.get("is_floor")]
                n_out_circle = sum(1 for r in arm_rows if r["_kamm"])
                #: ⛔ THE NOTE REPORTS CURVATURE, NOT ADE, AND IT NAMES THE
                #: FLOOR. "best ADE here" was the previous cut's headline and it
                #: is the wrong question: the floor wins ADE on this panel while
                #: tracking the road worse, which is exactly the confusion this
                #: re-render exists to remove.
                _cv = [r for r in arm_rows if not np.isnan(r["_curv"])]
                _bc = min(_cv, key=lambda r: r["_curv"]) if _cv else None
                note = (("closest to the human's CURVATURE: "
                         f"{_bc['arm']} {_bc['_curv']:.5f}"
                         if _bc else "curvature undefined here (ds too small)")
                        + (f"  ·  STRAIGHT FLOOR ha0 {c_floor:.5f}"
                           if not np.isnan(c_floor) else "")
                        + (f"  ·  {n_out_circle}/{len(arm_rows)} arms LEAVE the "
                           f"friction circle" if n_out_circle else
                           "  ·  all arms inside the friction circle"))
                bev = draw_bev(
                    (W_RIGHT, H_CAM),
                    [(fl_c, C_FLOOR, 13, True)]
                    + ([(ec_c, C_ECHO, 3, False)] if ec_c is not None else [])
                    + [(g_c, C_GT, 8, False)]
                    + [(arm_c[n], colours[n], 4, False) for n in order],
                    bev_lo, bev_hi, note, F, float(ref["v0"][wi]))
                im.paste(bev, (X_RIGHT, Y_CAM))
                elements.append(VizElement(
                    "bev", "present", kind="derived",
                    value=f"GT / ha0 STRAIGHT floor"
                          + (" / ha0_ext echo" if echo_on else "")
                          + f" / {len(order)} arms, "
                            f"x {bev_lo:.0f} to {bev_hi:.0f} m",
                    source="the dumps' own arrays, carried into the current ego "
                           "frame by a rigid SE(2) transform of the RECORDED poses"))

                # ---- arms table (the ADE slot: this reel's ADE is PER ARM)
                im.paste(draw_arms_table(
                    (W_LEFT, H_MID), rows, F,
                    "THE PLANNER ARMS on THIS window, against the FLOORS - ADE "
                    "over the K=10 waypoints; curv MAE is |plan curvature - the "
                    "human's| (green = better than the straight floor, red = "
                    "worse); the friction column is the SCORER'S OWN "
                    "assert_feasible (mu 0.7)", foot=curv_foot), (PAD, Y_MID))
                elements.append(VizElement(
                    "ade", "present", kind="derived",
                    value="; ".join(f"{r['arm']} {r['ade'].strip()} m "
                                    f"curv {r['curv'].strip()}" for r in rows),
                    source="mean ||cl - g||_2 over the K=10 waypoints of this "
                           "window at dt 0.2 s, from each arm's own ep*.npz; the "
                           "curvature column is taniteval.four_families "
                           "(_seq_geometry dh/ds, masked by pair_valid), the "
                           "EVAL'S OWN reduction, controlled against the banked "
                           "panel by --expect-curv-mae; the friction column is "
                           "tanitad.refs.feasible_decode.assert_feasible "
                           "(A_MAX 4.0, KAPPA_MAX 0.2, mu 0.7)"))

                # ---- world model
                #: ⚠️ NOT declared as a viz element: `world_model` is not in the
                #: standard's KNOWN_ELEMENTS and this renderer does not widen a
                #: shared contract to suit itself. It is titled on the frame as
                #: the T0 diagnostic it is, and recorded in the frame index.
                im.paste(draw_wm((W_RIGHT, H_MID), ref["wm_mse_model"][wi],
                                 ref["wm_mse_const"][wi], ref["wm_mse_zero"][wi],
                                 F, mark_s=plan_age_s or None), (X_RIGHT, Y_MID))

                # ---- decisions
                # ⭐ COLLAPSE IDENTICAL DECODED GOALS. The arms share one
                # tactical head, so on most windows every arm decodes the SAME
                # goal token and four identical lines would suggest four
                # decisions. One line per DISTINCT goal, naming the arms that
                # hold it, shows at a glance whether the goal or the cost is
                # what moved.
                by_goal: dict[str, list[str]] = {}
                for n in order:
                    by_goal.setdefault(goal_of[n], []).append(n)
                goal_lines = [{"colour": (C_FG if len(v) == len(order)
                                          else colours[v[0]]),
                               "text": (f"all {len(order)} arms: {g}"
                                        if len(v) == len(order)
                                        else f"{', '.join(v)}: {g}")}
                              for g, v in by_goal.items()]
                im.paste(draw_decisions((W_TOT - 2 * PAD, H_BOT), {
                    "LAT": LAT, "LON": LON, "ROUTE": ROUTE, "NAV": NAVN,
                    "lat_pred": int(ref["lat_pred_nav_true"][wi]),
                    "lat_lab": lat_lab,
                    "lon_pred": int(ref["lon_pred_nav_true"][wi]),
                    "lon_lab": int(ref["lon_label"][wi]),
                    "rt_pred": int(ref["route_pred_nav_true"][wi]),
                    "rt_lab": int(ref["route_label"][wi]),
                    "nav": int(ref["nav_cmd"][wi]),
                    "nav_valid": bool(ref["nav_valid"][wi]),
                    "goals": goal_lines}, F), (PAD, Y_BOT))
                lat_p, lon_p = (int(ref["lat_pred_nav_true"][wi]),
                                int(ref["lon_pred_nav_true"][wi]))
                rt_p = int(ref["route_pred_nav_true"][wi])
                nv = int(ref["nav_cmd"][wi])
                #: ⛔ the head argmaxes go in the PREDICTION slots and the nav
                #: token goes in its OWN slot. A GT-derived input rendered where
                #: a reader can only read it as the model's decision is the
                #: nav-echo defect, and `check_frame` refuses it.
                elements.append(VizElement(
                    "tactical", "present", kind="model_output",
                    value=f"lat {LAT[lat_p] if 0 <= lat_p < len(LAT) else lat_p}"
                          f" / lon {LON[lon_p] if 0 <= lon_p < len(LON) else lon_p}",
                    source="decisions/ep*.npz lat_pred_nav_true / "
                           "lon_pred_nav_true (argmax, bit-identical across "
                           "every arm on this frame)",
                    conditioned_on=("nav_cmd",)))
                elements.append(VizElement(
                    "strategic", "present", kind="model_output",
                    value=(ROUTE[rt_p] if 0 <= rt_p < len(ROUTE) else str(rt_p)),
                    source="decisions/ep*.npz route_pred_nav_true (argmax)",
                    conditioned_on=("nav_cmd",)))
                elements.append(VizElement(
                    "strategic_input", "present", kind="given_input",
                    value=(NAVN[nv] if 0 <= nv < len(NAVN) else str(nv)),
                    source="decisions/ep*.npz nav_cmd — the v7.2 navigator "
                           "command the heads were conditioned on"))
                elements.append(VizElement(
                    "tactical_gt", "present", kind="gt_label",
                    value=f"lat {LAT[lat_lab] if 0 <= lat_lab < len(LAT) else lat_lab}"
                          f" / lon {LON[int(ref['lon_label'][wi])] if 0 <= int(ref['lon_label'][wi]) < len(LON) else int(ref['lon_label'][wi])}",
                    source="decisions/ep*.npz lat_label / lon_label (v7.2)"))

                check_frame(elements, where=f"render_refav1_arms:{cid}:{f}")
                assert_frame_content(
                    im, [("g", g_c), ("ha0", fl_c), ("ha0_ext", ec_c)]
                    + [(n, arm_c[n]) for n in order],
                    f"{cid}:{f}", stats)

                meta = {"kind": "clip", "clip_id": cid, "clip_order": ci,
                        "raw_frame": f, "t_cache": t_w,
                        "t_s": round(t_w * DT_CACHE, 2),
                        "plan_age_s": round(plan_age_s, 2),
                        "window": wi, "gt_lat_token": LAT[lat_lab]
                        if 0 <= lat_lab < len(LAT) else str(lat_lab),
                        "slowmo": hold, "slowmo_by_label": by_label,
                        "slowmo_by_geometry": by_geom, "camera": bool(cam_on),
                        "gt_lat_extent_m": round(float(np.abs(g_c[:, 1]).max()), 3),
                        "v0_mps": round(float(ref["v0"][wi]), 3),
                        "turn_window": None if not trank else
                        {"rank": trank[0], "of": trank[1], "token": LAT[lat_lab]},
                        "arms": {r["arm"]: {
                            "ade_m": round(r["_ade"], 4),
                            "curv_mae_1pm": (None if np.isnan(r["_curv"])
                                             else round(r["_curv"], 6)),
                            "is_floor": bool(r.get("is_floor")),
                            "kamm_over": r["_kamm"]} for r in rows}}
                for _ in range(hold):
                    emit(im, meta)
                    stats["frames"] += 1
                    if by_label:
                        stats["turn_frames"] += 1
                    if hold > 1:
                        stats["slowed"] += 1
                if a.max_frames and n_out >= a.max_frames:
                    break
            if a.max_frames and n_out >= a.max_frames:
                break
        stats["clips"] += 1
        if a.max_frames and n_out >= a.max_frames:
            break

    if cards.get("outro") and not (a.max_frames and n_out >= a.max_frames):
        im = ra.draw_card(cards["outro"], F, card_foot)
        for _ in range(a.card_frames):
            emit(im, {"kind": "card", "card": "outro"})

    if stats["frames"] == 0:
        sys.exit("[arms-reel] no clip frames were produced — refusing to encode "
                 "a reel of title cards")
    #: ⛔ divided by the number of frames ACTUALLY CHECKED, not by the number
    #: emitted: slow-motion duplicates the same image, and dividing by the
    #: emitted count would report a mean below every sample it averaged — a
    #: statistic that cannot be right is worse than none.
    stats["lum_mean"] = stats["lum_sum"] / max(stats["lum_n"], 1)
    stats.pop("lum_sum")
    idx = os.path.splitext(a.out)[0] + "_frame_index.json"
    with open(idx, "w", encoding="utf-8") as fh:
        json.dump({"tool": "render_refav1_arms_video.py", "step": step,
                   "fps": a.fps, "n_frames": n_out, "ckpt": ckpt,
                   "arms": order, "arm_geometry": geom,
                   "arm_colours": {k: list(v) for k, v in colours.items()},
                   "floors_drawn": floors,
                   "arms_dir": os.path.abspath(a.arms_dir),
                   "shared_verification": shared, "gt_control": control,
                   "curvature_panel": curv_panel,
                   "curvature_control": curv_rep,
                   "curvature_footer": curv_foot,
                   "turn_windows": [{"episode": e, "window": w, "token": t}
                                    for e, w, t in turn_windows],
                   "stats": stats, "frames": index}, fh, indent=1)
    _p(f"[index ] {idx}")
    _p(f"[frames] {n_out} written in {time.time() - t_start:.0f} s  "
       f"(scored windows {stats['scored']}, slow-motion frames {stats['slowed']}, "
       f"decoded camera frames {stats['decoded']}, frames past the plan-age cap "
       f"skipped {stats['skipped_stale']}, frames whose plans project under the "
       f"hood {stats['cam_stub']})")
    _p(f"[turns ] {stats['turn_windows']} GT-labelled turn windows drawn over "
       f"{stats['turn_frames']} frames "
       f"({100.0 * stats['turn_frames'] / max(stats['frames'], 1):.1f} % of the "
       f"clip footage, from {100.0 * stats['turn_windows'] / max(stats['scored'], 1):.1f} % "
       f"of the windows)")
    _p(f"[content] mean luminance {stats['lum_mean']:.2f} "
       f"(min {stats['lum_min']:.2f}, max {stats['lum_max']:.2f}) — asserted "
       f"non-black on EVERY frame")

    ff = "ffmpeg"
    pat = os.path.join(frames_dir, "f_%06d.png")
    jobs = [([ff, "-y", "-v", "error", "-framerate", str(a.fps), "-i", pat,
              "-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf),
              "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], "full")]
    if not a.no_small:
        small = os.path.splitext(a.out)[0] + "_small.mp4"
        # ⚠️ RE-ENCODED FROM THE ORIGINAL FRAMES, never transcoded from `full`.
        jobs.append(([ff, "-y", "-v", "error", "-framerate", str(a.fps), "-i", pat,
                      "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "slow",
                      "-crf", "30", "-pix_fmt", "yuv420p", "-movflags",
                      "+faststart", small], "small"))
    for cmd, tag in jobs:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if r.returncode != 0:
            sys.exit(f"[arms-reel] ffmpeg ({tag}) failed: {r.stderr[:800]}")
        _p(f"[encode] {tag}: {cmd[-1]}  {os.path.getsize(cmd[-1]):,} B")

    if not a.keep_frames:
        keep = {0, n_out // 2, n_out - 1}
        keep |= {int(r["frame"]) for r in index if r.get("kind") == "title"}
        for i, p in enumerate(sorted(glob.glob(os.path.join(frames_dir, "f_*.png")))):
            if i not in keep:
                os.remove(p)
    _p("[done  ] now verify by DECODING BACK: python taniteval/tools/verify_mp4.py "
       + " ".join(c[-1] for c, _ in jobs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
