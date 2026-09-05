"""THE NAV-COMPLIANCE METRIC — does the model's BEHAVIOUR follow the nav command?

PI, 2026-09-05, verbatim: *"we are not evaluating the nav command itself, we are
evaluating the fact that the model is following the nav command in consistency
to the strategic goals."*

WHY THE OLD METRIC COULD NOT FAIL, AND WHY THIS ONE CAN
========================================================
The strategic ROUTE metric scored refcv3 at 1.0000 because its label is an
exact bijection of the fed nav token (``D-REFAV1-ROUTE-LABEL-IS-THE-NAV``,
141/141). A head that reproduces its own input scores perfectly; a head with
zero route skill scores perfectly. **A metric that reads the same value for a
working model and a broken one is not a measurement.** Three things make this
module different, and each is a place the number can come out LOW:

1. **It scores BEHAVIOUR, not a label.** The readouts are the GEOMETRY of the
   emitted plan (heading change and lateral displacement at the plan's last
   slot), the DIRECTION of the strategic goal ``g_str``, and the manoeuvre
   CLASS of the selected anchor. None of them is a function the label was
   built from; a model can set the wrong goal, pick the wrong anchor, or drive
   straight through a commanded turn, and each shows up as its own number.
2. **The command it is scored against is always the window's TRUE command,
   while the command FED to the model is intervened on.** Under the nav-SHUFFLE
   and nav-FLIP arms the fed token and the true command disagree by
   construction, so a model that follows the token scores LOW against the
   truth, a model that ignores the token scores the SAME under every arm, and
   a model that reads the scene scores HIGH whatever token it is fed. Those
   are three DIFFERENT readings, committed in advance (:func:`verdict`).
3. **Its tolerance comes from the corpus, never from the model.** The lateral
   threshold is the 90th percentile of the GT heading change over the plan
   horizon on FOLLOW-token windows — what "not turning" looks like on this
   road network, curves included. A model cannot move it.

⛔ ADMISSIBLE ONLY WITH ITS CONTROLS. A compliance rate without the nav-shuffle
and nav-zero arms on the SAME windows is not evidence (BACKLOG R39; the
criteria registry refuses it): compliance must DROP under shuffle if the model
follows nav, and if it does not drop the model ignores nav and any high
compliance is coincidence with the scene. ⚠️ The converse limit is stated too:
this metric alone CANNOT distinguish a model that follows the token from a
model that echoes it blindly — a canned turn in the fed direction reads 1.0
here. That is why it is one row of the four binding families and never a
score on its own: the echo of a token drives into the kerb, and the LATERAL /
LONGITUDINAL families see that.

THE STRATA — why "did it turn?" is the wrong question on most windows
=====================================================================
A nav token is per CLIP (``t0_constant`` semantics, ``v7_labels.NavEmitter``),
so a window whose turn is 25 s away carries the same ``NAV_TURN_R`` as one
whose turn starts now. Penalising the first for not turning within a 6 s plan
would reward premature turning. Every informative window is therefore placed
in exactly one stratum from the GROUND TRUTH and the label's turn timing:

    imminent   the GT itself shows the commanded turn within the plan horizon
               -> the plan must turn the commanded way   (compliance)
    deferred   the turn is still ahead but not within the horizon
               -> the plan must HOLD, and g_str should already point the way
               (this is the "consistency with the strategic goals" reading:
               strategic level committed, tactical level correctly waiting)
    stale      the commanded turn already ENDED before the window origin —
               the constant token no longer describes the future; EXCLUDED
    conflict   the GT turns the OTHER way inside the horizon (a curve before
               the junction); EXCLUDED and counted
    ambiguous  the turn is in progress at t0 but too little of it remains to
               clear the threshold; EXCLUDED and counted

``n`` is reported for every stratum, and a stratum below :data:`MIN_N_WINDOWS`
/ :data:`MIN_N_EPISODES` is CANNOT-RULE, never a number.

ESTIMATOR: episode-cluster bootstrap for every rate, PAIRED for every delta
(``taniteval.ci``). ``overlapping_holdout_se`` is never called.

ONLY numpy (plus ``taniteval.ci``). No torch, no model, no paths — so the
metric is unit-testable against synthetic arms whose reading is KNOWN.
"""
from __future__ import annotations

import math

import numpy as np

from taniteval import ci as _ci

__all__ = [
    "NAV_FOLLOW", "NAV_LEFT", "NAV_RIGHT", "NAV_NAMES", "STRAIGHT", "SIGN",
    "STRATA", "READOUTS", "MIN_N_WINDOWS", "MIN_N_EPISODES", "DEFAULT_THETA_Q",
    "wrap_to_pi", "heading_change", "lateral_displacement", "bearing_angle",
    "flip_nav", "derive_threshold", "classify_lateral", "assign_strata",
    "known_value_controls", "compliance_report",
]

#: ``refb.NAV_COMMANDS[:3]`` — pinned by ``tests/test_nav_compliance.py``
#: against the real tuple so a silent re-ordering cannot land a command on the
#: wrong row. ``straight`` (index 3) is never emitted by the labeller.
NAV_FOLLOW, NAV_LEFT, NAV_RIGHT = 0, 1, 2
NAV_NAMES = ("follow", "left", "right")
#: the lateral CLASS space is nav-aligned on purpose: ``cls == cmd`` is
#: compliance and ``cls == STRAIGHT`` is holding, with no mapping table.
STRAIGHT = NAV_FOLLOW
#: +y = left, CCW-positive yaw (the repo's ``_ego`` convention).
SIGN = {NAV_LEFT: 1.0, NAV_RIGHT: -1.0}
STRATA = ("imminent", "deferred", "stale", "conflict", "ambiguous", "follow",
          "unlabeled")
#: the behavioural readouts, in the order the report prints them.
READOUTS = ("plan", "anchor", "g_str", "core_lat", "tac_lat")
#: below these a stratum is CANNOT-RULE (the O6 lesson: a gate that cannot
#: decide must say so, never read as a pass).
MIN_N_WINDOWS = 20
MIN_N_EPISODES = 5
DEFAULT_THETA_Q = 0.90
#: a path step shorter than this is a stopped/crawling vehicle with no tangent
#: (``four_families.MIN_DS_MPS * dt`` at the 0.5 s slot spacing).
MIN_DS_M = 0.05
#: a goal bearing shorter than this is undefined (``refc_v3`` clamps at 1e-6).
MIN_BEARING_NORM = 1e-6


# --------------------------------------------------------------------------- #
# geometry                                                                     #
# --------------------------------------------------------------------------- #
def wrap_to_pi(a):
    return (np.asarray(a, dtype=np.float64) + math.pi) % (2 * math.pi) - math.pi


def heading_change(P, min_ds_m: float = MIN_DS_M) -> np.ndarray:
    """``[N, S, 2]`` ego-frame path -> heading change at the LAST slot (rad).

    The path tangent of the final moving segment, with the origin prepended as
    step 0 — the same construction as ``four_families.maneuver_kinematics``
    (pinned by test), including its stationary rule: the heading is HELD at
    the last step that moved, and a path that never moved reads 0.0 (it did
    not turn). In the window-origin frame ``yaw(t0) == 0`` so the tangent IS
    the heading change; +rad = left.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 3 or P.shape[-1] != 2:
        raise ValueError(f"path must be [N, S, 2], got {P.shape}")
    n, s = P.shape[:2]
    p = np.concatenate([np.zeros((n, 1, 2)), P], axis=1)
    d = p[:, 1:] - p[:, :-1]
    ds = np.linalg.norm(d, axis=-1)
    valid = ds > min_ds_m
    head = np.arctan2(d[..., 1], d[..., 0])
    last = np.where(valid, np.arange(s)[None, :], -1).max(axis=1)
    out = np.where(last >= 0, head[np.arange(n), np.clip(last, 0, None)], 0.0)
    return wrap_to_pi(out)


def lateral_displacement(P) -> np.ndarray:
    """``[N, S, 2]`` -> the last slot's lateral offset (m, +left)."""
    P = np.asarray(P, dtype=np.float64)
    return P[:, -1, 1].copy()


def bearing_angle(g) -> np.ndarray:
    """``[N, >=2]`` (x, y, ...) -> atan2(y, x) (rad, +left); NaN where the
    vector is too short to have a direction."""
    g = np.asarray(g, dtype=np.float64)
    nrm = np.linalg.norm(g[:, :2], axis=-1)
    ang = np.arctan2(g[:, 1], g[:, 0])
    return np.where(nrm > MIN_BEARING_NORM, ang, np.nan)


def flip_nav(cmd) -> np.ndarray:
    """left <-> right; follow (and anything else) unchanged. The sharpest
    per-window intervention: every informative window changes."""
    c = np.asarray(cmd, dtype=np.int64).copy()
    out = c.copy()
    out[c == NAV_LEFT] = NAV_RIGHT
    out[c == NAV_RIGHT] = NAV_LEFT
    return out


# --------------------------------------------------------------------------- #
# threshold + classes + strata                                                 #
# --------------------------------------------------------------------------- #
def derive_threshold(values, mask, q: float = DEFAULT_THETA_Q,
                     what: str = "gt heading change at the plan horizon") -> dict:
    """``theta`` = the ``q`` quantile of ``|values[mask]|`` — the CORPUS's own
    "not turning" envelope, derived from the GROUND TRUTH on follow-token
    windows and never from the model. Fails loud on an empty reference set: a
    threshold of NaN would classify everything as straight and read as a
    model that never turns."""
    v = np.abs(np.asarray(values, dtype=np.float64))
    m = np.asarray(mask, dtype=bool) & np.isfinite(v)
    if m.sum() == 0:
        raise ValueError(f"derive_threshold({what}): no finite reference value "
                         f"under the mask — the corpus-derived tolerance cannot "
                         f"be computed (an empty follow set)")
    theta = float(np.quantile(v[m], q))
    return {"theta": theta, "q": float(q), "n_ref": int(m.sum()),
            "ref_median": float(np.median(v[m])),
            "source": f"quantile({q}) of |{what}| over follow-token windows "
                      f"(GT only; the model cannot move it)"}


def classify_lateral(angle, theta: float) -> np.ndarray:
    """rad -> nav-aligned class: > theta LEFT, < -theta RIGHT, else STRAIGHT.
    NaN (no direction) -> -1, which equals no command and never complies."""
    a = np.asarray(angle, dtype=np.float64)
    cls = np.full(a.shape, STRAIGHT, dtype=np.int64)
    cls[a > theta] = NAV_LEFT
    cls[a < -theta] = NAV_RIGHT
    cls[~np.isfinite(a)] = -1
    return cls


def assign_strata(cmd, cmd_valid, gt_cls, turn_start_rel_s, turn_end_rel_s,
                  horizon_s: float) -> np.ndarray:
    """One stratum per window (module docstring). ``turn_*_rel_s`` are the
    commanded turn's start/end in seconds RELATIVE TO THE WINDOW ORIGIN, from
    the label's ``nav_command.args.time_s`` and the matching manoeuvre's
    ``t_end_s``; NaN where the label carries no timing."""
    cmd = np.asarray(cmd, dtype=np.int64)
    valid = np.asarray(cmd_valid, dtype=bool)
    gt = np.asarray(gt_cls, dtype=np.int64)
    ts = np.asarray(turn_start_rel_s, dtype=np.float64)
    te = np.asarray(turn_end_rel_s, dtype=np.float64)
    out = np.full(cmd.shape, "unlabeled", dtype=object)
    inf = valid & ((cmd == NAV_LEFT) | (cmd == NAV_RIGHT))
    out[valid & (cmd == NAV_FOLLOW)] = "follow"
    other = np.where(cmd == NAV_LEFT, NAV_RIGHT, NAV_LEFT)
    ended = np.isfinite(te) & (te <= 0.0)
    ahead = np.isfinite(ts) & (ts > 0.0)
    out[inf & ended] = "stale"
    live = inf & ~ended
    out[live & (gt == cmd)] = "imminent"
    out[live & (gt == other)] = "conflict"
    hold = live & (gt == STRAIGHT)
    out[hold & ahead] = "deferred"
    # the turn has begun (start <= 0 < end) but the GT over the horizon does
    # not clear the threshold: too little of it is left to score
    out[hold & ~ahead] = "ambiguous"
    out[live & (gt < 0)] = "ambiguous"
    return out.astype(str)


# --------------------------------------------------------------------------- #
# estimator wrappers                                                           #
# --------------------------------------------------------------------------- #
def _unavailable(reason: str, n: int = 0) -> dict:
    return {"status": "UNAVAILABLE", "reason": reason, "n": int(n)}


def _rate(ok, eid, n_boot, seed) -> dict:
    ok = np.asarray(ok, dtype=np.float64)
    if ok.size == 0:
        return _unavailable("no window in this cell", 0)
    r = _ci.episode_cluster_bootstrap(ok, list(eid), n_boot=n_boot, seed=seed)
    r["rate"] = r.pop("mean")
    return r


def _paired(a, b, eid, n_boot, seed) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return _unavailable("no window in this cell", 0)
    return _ci.paired_episode_cluster_bootstrap(a, b, list(eid), n_boot=n_boot,
                                                seed=seed)


def _powered(n_win: int, eids) -> tuple[bool, str]:
    n_ep = len(set(eids))
    if n_win < MIN_N_WINDOWS or n_ep < MIN_N_EPISODES:
        return False, (f"CANNOT-RULE: n_windows {n_win} < {MIN_N_WINDOWS} or "
                       f"n_episodes {n_ep} < {MIN_N_EPISODES}")
    return True, "powered"


# --------------------------------------------------------------------------- #
# the report                                                                   #
# --------------------------------------------------------------------------- #
def _readout_classes(arm: dict, theta_plan: float, theta_gstr: float) -> dict:
    """The five nav-aligned class arrays of one arm (None when the arm did not
    bank that readout — an ABSENT readout is reported, never zero-filled)."""
    out = {}
    out["plan"] = (classify_lateral(arm["plan_dpsi"], theta_plan)
                   if arm.get("plan_dpsi") is not None else None)
    out["anchor"] = (classify_lateral(arm["anchor_dpsi"], theta_plan)
                     if arm.get("anchor_dpsi") is not None else None)
    out["g_str"] = (classify_lateral(arm["gstr_angle"], theta_gstr)
                    if arm.get("gstr_angle") is not None else None)
    for k in ("core_lat", "tac_lat"):
        v = arm.get(k)
        out[k] = None if v is None else np.asarray(v, dtype=np.int64)
    return out


def _ok(cls, cmd, stratum: str) -> np.ndarray:
    """Per-window compliance of a class array against the TRUE command in a
    stratum: turn the commanded way when imminent, hold when deferred."""
    cls = np.asarray(cls, dtype=np.int64)
    cmd = np.asarray(cmd, dtype=np.int64)
    if stratum == "imminent":
        return (cls == cmd).astype(np.float64)
    if stratum == "deferred":
        return (cls == STRAIGHT).astype(np.float64)
    raise ValueError(f"no compliance rule for stratum {stratum!r}")


def _gstr_ok(cls, cmd) -> np.ndarray:
    """g_str compliance is direction-only in BOTH scoring strata: the strategic
    goal should point the commanded way whether the turn is now or later."""
    return (np.asarray(cls) == np.asarray(cmd)).astype(np.float64)


def known_value_controls(win: dict, theta_plan: float, strata: np.ndarray,
                         true_arm: str, n_boot: int, seed: int) -> dict:
    """⭐ CONTROLS THAT MUST READ A KNOWN VALUE — the harness check.

    Three synthetic arms are built from the GT and the commands ALONE and
    scored through exactly the same code path as the model:

      const_straight  a plan that never turns: imminent compliance EXACTLY 0.0,
                      deferred hold EXACTLY 1.0, token-following 0.0.
      nav_template    a canned turn in the FED direction (the pure token echo):
                      imminent compliance 1.0 under the true token, 0.0 under
                      the flip, token-following 1.0 — and deferred hold 0.0,
                      because an echo turns early.
      gt_echo         a plan equal to the GT whatever the token: imminent
                      compliance 1.0 under EVERY arm and every paired delta
                      EXACTLY 0.0 — the reading a scene-coincident model gives,
                      which the verdict must call IGNORES_NAV, never
                      FOLLOWS_NAV.

    If any of these misreads, THE HARNESS IS WRONG, NOT THE MODEL, and the
    report says so in ``harness_ok``.
    """
    cmd = np.asarray(win["cmd"], dtype=np.int64)
    eid = np.asarray(win["eid"])
    gt = np.asarray(win["gt_dpsi"], dtype=np.float64)
    arms = win["arms"]
    fed_true = np.asarray(arms[true_arm]["fed"], dtype=np.int64)
    fed_flip = flip_nav(fed_true)
    imm = strata == "imminent"
    dfd = strata == "deferred"
    sgn = np.where(cmd == NAV_LEFT, 1.0, np.where(cmd == NAV_RIGHT, -1.0, 0.0))
    sgn_fed = np.where(fed_true == NAV_LEFT, 1.0,
                       np.where(fed_true == NAV_RIGHT, -1.0, 0.0))
    sgn_flip = np.where(fed_flip == NAV_LEFT, 1.0,
                        np.where(fed_flip == NAV_RIGHT, -1.0, 0.0))
    big = theta_plan + 1.0
    checks, ok_all = {}, True

    def rec(name, cond, detail):
        nonlocal ok_all
        ok_all &= bool(cond)
        checks[name] = {"pass": bool(cond), **detail}

    if imm.sum() == 0 and dfd.sum() == 0:
        return {"harness_ok": None,
                "reason": "no imminent or deferred window — controls cannot run",
                "checks": {}}
    cs = classify_lateral(np.zeros_like(gt), theta_plan)
    tmpl_t = classify_lateral(sgn_fed * big, theta_plan)
    tmpl_f = classify_lateral(sgn_flip * big, theta_plan)
    gte = classify_lateral(gt, theta_plan)
    if imm.sum():
        r = float(_ok(cs[imm], cmd[imm], "imminent").mean())
        rec("const_straight.imminent_compliance_is_0", r == 0.0, {"read": r, "want": 0.0})
        r = float(_ok(tmpl_t[imm], cmd[imm], "imminent").mean())
        rec("nav_template.imminent_compliance_under_true_is_1", r == 1.0, {"read": r, "want": 1.0})
        r = float(_ok(tmpl_f[imm], cmd[imm], "imminent").mean())
        rec("nav_template.imminent_compliance_under_flip_is_0", r == 0.0, {"read": r, "want": 0.0})
        r = float((tmpl_f[imm] == fed_flip[imm]).mean())
        rec("nav_template.token_following_under_flip_is_1", r == 1.0, {"read": r, "want": 1.0})
        r = float((cs[imm] == fed_flip[imm]).mean())
        rec("const_straight.token_following_under_flip_is_0", r == 0.0, {"read": r, "want": 0.0})
        r = float(_ok(gte[imm], cmd[imm], "imminent").mean())
        rec("gt_echo.imminent_compliance_is_1", r == 1.0, {"read": r, "want": 1.0})
        e = eid[imm]
        a = _ok(gte[imm], cmd[imm], "imminent")
        d = _paired(a, a, e, n_boot, seed)
        rec("gt_echo.paired_delta_true_minus_flip_is_exactly_0",
            d.get("delta") == 0.0 and d.get("lo") == 0.0 and d.get("hi") == 0.0
            and not d.get("separated"),
            {"read": {k: d.get(k) for k in ("delta", "lo", "hi", "separated")},
             "want": {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": False}})
    if dfd.sum():
        r = float(_ok(cs[dfd], cmd[dfd], "deferred").mean())
        rec("const_straight.deferred_hold_is_1", r == 1.0, {"read": r, "want": 1.0})
        r = float(_ok(tmpl_t[dfd], cmd[dfd], "deferred").mean())
        rec("nav_template.deferred_hold_is_0", r == 0.0, {"read": r, "want": 0.0})
    return {"harness_ok": bool(ok_all), "checks": checks,
            "_reads": ("every control is built from GT + commands only and "
                       "scored through the model's own code path; a failed "
                       "check means the HARNESS is wrong, not the model")}


def verdict(plan_true: dict, ctrl: dict, n_zero_arm: bool) -> dict:
    """The pre-registered decision on the PLAN readout, imminent stratum.

    FOLLOWS_NAV   the paired drop under the shuffle (changed subset) is
                  separated > 0; if the shuffle is CANNOT-RULE, the flip
                  decides (same intervention, every window changed).
    IGNORES_NAV   the deciding delta's CI includes zero (or is <= 0). Any
                  compliance the true-nav arm shows is then scene-driven or
                  coincidental — the nav-zero arm says how much is scene.
    CANNOT_RULE   the stratum is under-powered on every deciding control.
    """
    def _sep_pos(d):
        return isinstance(d, dict) and d.get("status") is None \
            and bool(d.get("separated")) and float(d.get("delta", 0)) > 0

    def _readable(d):
        return isinstance(d, dict) and d.get("status") is None

    shuf = ((ctrl.get("nav_shuffle") or {}).get("plan") or {}).get("imminent")
    flip = ((ctrl.get("nav_flip") or {}).get("plan") or {}).get("imminent")
    zero = ((ctrl.get("nav_zero") or {}).get("plan") or {}).get("imminent")
    decider, d = None, None
    if _readable(shuf):
        decider, d = "nav_shuffle (changed subset)", shuf
    elif _readable(flip):
        decider, d = "nav_flip (shuffle CANNOT-RULE)", flip
    if d is None:
        return {"nav_effect": "CANNOT_RULE", "decided_by": None,
                "reason": "no powered intervention control on the imminent "
                          "stratum", "rule": verdict.__doc__}
    out = {"decided_by": decider,
           "delta_true_minus_control": {k: d.get(k) for k in
                                        ("delta", "lo", "hi", "separated",
                                         "n_windows", "n_episodes")},
           "rule": verdict.__doc__}
    if _sep_pos(d):
        out["nav_effect"] = "FOLLOWS_NAV"
        out["reason"] = ("plan compliance against the TRUE command drops when "
                         "the fed token is intervened on — the plan is "
                         "reading the token")
    else:
        out["nav_effect"] = "IGNORES_NAV"
        rt = plan_true.get("rate")
        out["reason"] = (f"no separated drop under {decider}; the true-nav "
                         f"compliance rate {rt} is therefore scene-driven or "
                         f"coincidental, not nav-following")
    if _readable(zero):
        out["nav_zero_delta"] = {k: zero.get(k) for k in
                                 ("delta", "lo", "hi", "separated")}
    elif n_zero_arm:
        out["nav_zero_delta"] = zero
    return out


def compliance_report(win: dict, *, horizon_s: float = 6.0,
                      n_boot: int = 2000, seed: int = 0,
                      theta_q: float = DEFAULT_THETA_Q,
                      true_arm: str = "nav_true", tier: str = "T1",
                      arms_meta: dict | None = None,
                      theta_ref: dict | None = None) -> dict:
    """The whole nav-compliance block from per-window arrays.

    ``win``:
      eid [N] str · cmd [N] int (TRUE command) · cmd_valid [N] bool ·
      turn_start_rel_s / turn_end_rel_s [N] (NaN = no timing) ·
      gt_dpsi [N] rad (GT heading change over the plan horizon, same
      construction as the plan) · optional gt_far_bearing [N] rad ·
      optional str_goal_dir [N] int (the 30 s strategic label's direction) ·
      arms: {name: {fed [N] int (-1 = no token), plan_dpsi [N], plan_y [N],
                    anchor_dpsi [N]|None, gstr_angle [N]|None,
                    core_lat [N] int|None, tac_lat [N] int|None}}
    The ``true_arm`` is the model under its real token; every other arm is an
    intervention on the SAME windows (shuffle / zero / flip / an ablation).

    ``theta_ref`` (optional): ``{"gt_dpsi": [M], "gt_far_bearing": [M]|None}``
    — GT values on FOLLOW windows from the WHOLE corpus, so the tolerance does
    not depend on which clips the model happened to be rolled on (a probe that
    forwards only turn clips would otherwise derive it from a handful of
    follow windows). Its size is recorded in the threshold block.
    """
    eid = np.asarray(win["eid"]).astype(str)
    cmd = np.asarray(win["cmd"], dtype=np.int64)
    valid = np.asarray(win["cmd_valid"], dtype=bool)
    gt = np.asarray(win["gt_dpsi"], dtype=np.float64)
    n = int(cmd.size)
    arms = win["arms"]
    if true_arm not in arms:
        raise ValueError(f"true arm {true_arm!r} not in arms {sorted(arms)}")
    follow = valid & (cmd == NAV_FOLLOW)
    # ---- the corpus-derived tolerance, from GT on follow windows only ------
    if theta_ref is not None and theta_ref.get("gt_dpsi") is not None:
        ref = np.asarray(theta_ref["gt_dpsi"], dtype=np.float64)
        thr = derive_threshold(ref, np.ones(ref.size, bool), theta_q,
                               what="gt heading change at the plan horizon "
                                    "(corpus-wide follow reference)")
        gfb = theta_ref.get("gt_far_bearing")
        gfb_mask = (np.ones(np.asarray(gfb).size, bool)
                    if gfb is not None else None)
    else:
        thr = derive_threshold(gt, follow, theta_q)
        gfb = win.get("gt_far_bearing")
        gfb_mask = follow
    theta = thr["theta"]
    if gfb is not None and np.isfinite(np.asarray(gfb, float)[gfb_mask]).any():
        thr_g = derive_threshold(np.asarray(gfb, float), gfb_mask, theta_q,
                                 what="gt bearing to the far future point")
    else:
        thr_g = dict(thr, source="fallback: the plan threshold (no far-bearing "
                                 "reference supplied)")
    theta_g = thr_g["theta"]
    gt_cls = classify_lateral(gt, theta)
    strata = assign_strata(cmd, valid, gt_cls,
                           win.get("turn_start_rel_s", np.full(n, np.nan)),
                           win.get("turn_end_rel_s", np.full(n, np.nan)),
                           horizon_s)
    counts = {s: int((strata == s).sum()) for s in STRATA}
    eps = {s: int(len(set(eid[strata == s]))) for s in STRATA}
    # label-timing cross-check on the geometric stratification
    ts = np.asarray(win.get("turn_start_rel_s", np.full(n, np.nan)), float)
    imm = strata == "imminent"
    xcheck = {"imminent_with_label_start_beyond_horizon":
              int((imm & np.isfinite(ts) & (ts > horizon_s)).sum()),
              "_reads": "windows the GT calls imminent while the label's turn "
                        "start lies beyond the plan horizon — a curve of the "
                        "commanded sign before the junction, or label timing "
                        "drift; counted, not dropped"}

    classes = {a: _readout_classes(arms[a], theta, theta_g) for a in arms}
    fed = {a: np.asarray(arms[a]["fed"], dtype=np.int64) for a in arms}

    readouts: dict = {}
    for r in READOUTS:
        readouts[r] = {}
        for a in arms:
            cls = classes[a][r]
            if cls is None:
                readouts[r][a] = _unavailable(f"arm {a!r} banked no {r} readout")
                continue
            blk = {}
            for s in ("imminent", "deferred"):
                m = strata == s
                if m.sum() == 0:
                    blk[s] = _unavailable(f"no {s} window", 0)
                    continue
                powered, why = _powered(int(m.sum()), eid[m])
                ok = (_gstr_ok(cls[m], cmd[m]) if r == "g_str"
                      else _ok(cls[m], cmd[m], s))
                cell = _rate(ok, eid[m], n_boot, seed)
                cell["powered"] = powered
                cell["rule"] = ("g_str points the commanded way" if r == "g_str"
                                else "turns the commanded way" if s == "imminent"
                                else "holds (no premature turn)")
                if not powered:
                    cell["status_note"] = why
                blk[s] = cell
            # threshold-free continuous statistic on the plan readout
            if r == "plan" and imm.sum():
                sgn = np.where(cmd == NAV_LEFT, 1.0,
                               np.where(cmd == NAV_RIGHT, -1.0, 0.0))
                tow = sgn[imm] * np.asarray(arms[a]["plan_dpsi"], float)[imm]
                keep = np.isfinite(tow)
                blk["signed_dpsi_toward_cmd_rad"] = (
                    _ci.episode_cluster_bootstrap(tow[keep], list(eid[imm][keep]),
                                                  n_boot=n_boot, seed=seed)
                    if keep.sum() else _unavailable("no finite plan heading", 0))
            readouts[r][a] = blk

    # ---- the interventions, PAIRED against the true arm --------------------
    controls: dict = {}
    for a in arms:
        if a == true_arm:
            continue
        cblk = {"_intervention": (arms_meta or {}).get(a, {}).get("what", a)}
        changed = fed[a] != fed[true_arm]
        no_token = (fed[a] < 0).all()
        for r in READOUTS:
            ct, ca = classes[true_arm][r], classes[a][r]
            if ct is None or ca is None:
                cblk[r] = _unavailable(f"readout {r!r} absent on one arm")
                continue
            rb = {}
            for s in ("imminent", "deferred"):
                m = strata == s
                # the shuffle has power only where the token CHANGED; the
                # zero and flip arms change every informative window
                if not no_token:
                    m = m & changed
                if m.sum() == 0:
                    rb[s] = _unavailable(f"no {s} window with a changed token", 0)
                    continue
                powered, why = _powered(int(m.sum()), eid[m])
                if r == "g_str":
                    ot, oa = _gstr_ok(ct[m], cmd[m]), _gstr_ok(ca[m], cmd[m])
                else:
                    ot, oa = _ok(ct[m], cmd[m], s), _ok(ca[m], cmd[m], s)
                if not powered:
                    rb[s] = _unavailable(why, int(m.sum()))
                    continue
                d = _paired(ot, oa, eid[m], n_boot, seed)
                d["direction"] = f"{true_arm} - {a}"
                d["rate_true"] = round(float(ot.mean()), 4)
                d["rate_arm"] = round(float(oa.mean()), 4)
                if not no_token and s == "imminent":
                    d["token_following_rate"] = round(
                        float((ca[m] == fed[a][m]).mean()), 4)
                    d["_token_following_is"] = ("fraction of changed imminent "
                                                "windows where the arm's class "
                                                "equals the FED (wrong) token")
                rb[s] = d
            cblk[r] = rb
        cblk["n_changed_tokens"] = int(changed.sum()) if not no_token else None
        controls[a] = cblk

    # ---- consistency with the strategic goal (the seam reading) ------------
    consistency: dict = {}
    ct = classes[true_arm]
    for s in ("imminent", "deferred"):
        m = strata == s
        if ct["plan"] is None or ct["g_str"] is None or m.sum() == 0:
            consistency[s] = _unavailable("needs both the plan and g_str "
                                          "readouts on the true arm", 0)
            continue
        p_ok = _ok(ct["plan"][m], cmd[m], s) > 0.5
        g_ok = _gstr_ok(ct["g_str"][m], cmd[m]) > 0.5
        e = eid[m]
        cells = {
            "both_ok": p_ok & g_ok,
            "goal_ok_plan_wrong__SEAM_FAILURE": g_ok & ~p_ok,
            "goal_wrong_plan_ok__BYPASS": ~g_ok & p_ok,
            "neither": ~g_ok & ~p_ok,
        }
        consistency[s] = {k: _rate(v.astype(float), e, n_boot, seed)
                          for k, v in cells.items()}
        consistency[s]["n"] = int(m.sum())
        consistency[s]["_reads"] = (
            "SEAM_FAILURE = the strategic level set the right goal and the "
            "plan did not follow it (a hierarchy defect, NOT a nav defect); "
            "BYPASS = the plan complied without the goal pointing the way "
            "(nav or scene reached the plan around the strategic level)")
    # the 30 s strategic label, when supplied: g_str vs the label's direction
    sgd = win.get("str_goal_dir")
    if sgd is not None and ct["g_str"] is not None:
        sgd = np.asarray(sgd, dtype=np.int64)
        m = valid & (sgd != STRAIGHT) & (strata != "stale")
        consistency["g_str_vs_30s_label"] = (
            dict(_rate((ct["g_str"][m] == sgd[m]).astype(float), eid[m],
                       n_boot, seed), n=int(m.sum()),
                 _is="g_str direction == the v7.2 g_str token's direction "
                     "(TURN_LEFT/RIGHT_FOLLOW_ROUTE) on windows carrying one")
            if m.sum() else _unavailable("no window carries a directional "
                                         "30 s strategic label", 0))

    kv = known_value_controls(win, theta, strata, true_arm, n_boot, seed)
    plan_true_imm = readouts["plan"].get(true_arm, {}).get("imminent", {})
    vd = verdict(plan_true_imm if isinstance(plan_true_imm, dict) else {},
                 controls, any((fed[a] < 0).all() for a in arms if a != true_arm))
    if kv.get("harness_ok") is False:
        vd = {"nav_effect": "HARNESS_BROKEN",
              "reason": "a known-value control misread; see known_value_controls",
              "rule": verdict.__doc__}

    return {
        "_is": ("nav-COMPLIANCE: does the BEHAVIOUR (plan geometry, strategic "
                "goal direction, selected anchor class, declared lateral "
                "decisions) follow the window's TRUE nav command, scored per "
                "stratum, with the FED token intervened on in the control arms"),
        "tier": tier,
        "estimator": "episode_cluster_bootstrap (rates) / "
                     "paired_episode_cluster_bootstrap (deltas)",
        "n_windows": n, "n_episodes": int(len(set(eid))),
        "horizon_s": float(horizon_s),
        "threshold": {"plan_and_anchor_rad": thr, "g_str_rad": thr_g,
                      "plan_theta_deg": round(math.degrees(theta), 3),
                      "g_str_theta_deg": round(math.degrees(theta_g), 3)},
        "strata": {"n_windows": counts, "n_episodes": eps,
                   "scored": ["imminent", "deferred"],
                   "excluded": ["stale", "conflict", "ambiguous"],
                   "label_timing_crosscheck": xcheck,
                   "min_n": {"windows": MIN_N_WINDOWS, "episodes": MIN_N_EPISODES}},
        "true_arm": true_arm,
        "arms": sorted(arms),
        "readouts": readouts,
        "controls": controls,
        "consistency": consistency,
        "known_value_controls": kv,
        "verdict": vd,
        "_falsifiability": [
            "the readouts are behaviour (geometry / goal direction / anchor "
            "class), not a label reproduced from the fed input",
            "compliance is scored against the TRUE command while the FED token "
            "is shuffled / zeroed / flipped on the same windows, so a "
            "token-following model, a nav-blind model and a scene-reading model "
            "give three DIFFERENT readings (verdict rule)",
            "the tolerance is derived from GT on follow windows; the model "
            "cannot move it",
            "three synthetic controls must read known values exactly or the "
            "block is HARNESS_BROKEN",
            "⚠️ the metric alone cannot separate token-following from "
            "token-echo — a canned turn in the fed direction reads 1.0 — so it "
            "is one row beside the four binding families, never a score",
        ],
    }
