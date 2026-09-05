"""NAV-COMPLIANCE — does the model's BEHAVIOUR follow the route command, and
does it STOP following it when the command is taken away?

PI rulings this instrument implements (2026-09-04/05, verbatim):

  * "we are not evaluating the nav command itself, we are evaluating the fact
    that the model is following the nav command in consistency to the
    strategic goals."
  * "Use the 30 s label as the strategic supervision; keep nav as a
    conditioning input."

WHY THE OLD METRIC COULD NOT BE FALSIFIED (MEASURED 2026-09-04,
``GOALS_AND_CLAIMS.md`` D-REFAV1-ROUTE-LABEL-IS-THE-NAV; refcv3 ARM json
``strategic._echo_caveat``): *route accuracy* compared the route HEAD's argmax
to a route LABEL that is an exact bijection of the nav token fed at inference
(141/141). A model that copies its input scores 1.0000 and NO behaviour of the
model can lower that score, so the number carries no information about
driving — it is ``f(nav)`` predicting ``f(nav)``. The existing echo test
compares the PREDICTION to the input and cannot see this, because the defect
is that the LABEL is the input.

WHAT THIS ONE SCORES INSTEAD — three BEHAVIOURAL readouts, never a head:

  ``plan``    the emitted 6 s path's terminal heading (with end-bearing and
              lateral end as secondary signals)
  ``gstr``    the strategic goal bearing ``g_str = (cos, sin, dist_pref)``
  ``anchor``  the SELECTED vocabulary element's geometry — the selection
              surface, which is decided at the t=0 confidence and never sees
              the refined fan (implementation audit, commit 7c67b3d)

against the COMMANDED side, on INFORMATIVE windows only — the true command is
``left``/``right`` AND the commanded manoeuvre lies inside the readout's
horizon — with ``n`` (windows AND episodes) always reported.

WHY IT IS FALSIFIABLE WHERE THE OLD ONE WAS NOT — the INTERVENTION PAIR is
the metric; the rate alone is not:

  ``nav_true``      the rate C(true). It CAN be coincidence with the scene.
  ``nav_shuffled``  the SAME windows with the token permuted across windows:
                    if the model follows the command, C(true) must DROP, and
                    on the changed subset the behaviour must follow the FED
                    (wrong) command.
  ``nav_zero``      nav withheld (``nav_cmd=None``, the E13 injection is
                    skipped): the robustness ablation.

A high C(true) that does NOT drop under shuffle is coincidence with GT — the
scene or the ego state already implied the turn — and the score then carries
no nav-following information. A path is not a token: it cannot be produced by
copying the input, so the echo that manufactured 1.0000 has no counterpart.

Plan-compliance and g_str-compliance are reported SEPARATELY on the SAME
windows, because they fail differently: right goal + wrong path is a SEAM
failure (E4/E7/E9), not a nav failure; wrong goal + right path means nav
reaches the operative layer without the decision layers (the exact inversion
D-REFCV4-NAV-WIRING measured in refcv3).

CONTROLS THAT MUST READ KNOWN VALUES (the 2026-08-22 lesson: three of four
estimator failures were caught ONLY because a control read the same value as
the thing being measured):

  ``ha0``        the constant-velocity straight line: compliance 0.0000 EXACTLY
  ``ha``,        nav-blind ego extrapolations: delta(true - shuffled) 0.0000
  ``ha0_ext``    EXACTLY, because they never see nav — and their C(true) is the
                 COINCIDENCE FLOOR the model must clear
  ``gt``         the recorded future itself on the informative windows: reads
                 the label / window / tolerance consistency (≈ 1 expected)
  ``always_cmd`` a synthetic policy that executes the FED command: C(true) is
                 1.0 exactly and its shuffle delta is the CEILING any follower
                 can reach on these windows

Estimator: ``taniteval.ci`` — episode-cluster bootstrap for rates, the PAIRED
form for every delta, ``n_boot`` 2000, cluster = episode. Never
``overlapping_holdout_se``.

Sign convention (``refb_labels`` docstring, ``kinematic_goal_extrapolation``):
CCW-positive yaw, +y = LEFT; ``left`` ⇒ positive heading change / positive
lateral end / positive curvature.
"""
from __future__ import annotations

import json
import math
import os
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "NAV_FOLLOW", "NAV_LEFT", "NAV_RIGHT", "NAV_STRAIGHT", "DT", "PLAN_HORIZON_S",
    "STRATEGIC_REACH_M", "WHEELBASE_M", "N_MIN_WINDOWS", "N_MIN_EPISODES",
    "CONDITIONS", "commanded_side", "terminal_heading", "end_bearing",
    "lateral_end", "wrap_pi", "ego_frame", "kinematic_heading",
    "informative_mask", "derive_tolerance", "complies", "compliance_arm",
    "control_arm", "nav_compliance_report", "verdict", "seam_reading",
    "load_labels", "join_windows_to_labels", "time_base_control",
    "from_refcv3_dump", "FALSIFIABILITY", "flip_nav", "assign_strata",
    "unavailable_block", "REQUIRED_CONDITIONS", "STRATA", "kinematic_end_signals",
]

#: ``tanitad.refs.refc.NAV_COMMANDS`` = ("follow", "left", "right", "straight")
#: and ``refb_labels.NAV_FOLLOW, NAV_LEFT, NAV_RIGHT, NAV_STRAIGHT = range(4)``.
#: Pinned against both by ``tests/test_nav_compliance.py`` — a local copy on
#: purpose (this module must run on a box without ``tanitad``), but never a
#: silent one.
NAV_FOLLOW, NAV_LEFT, NAV_RIGHT, NAV_STRAIGHT = 0, 1, 2, 3
#: 10 Hz — the corpus tick (``V3_HORIZONS`` are steps of 0.1 s).
DT: float = 0.1
#: the plan's farthest slot: ``max(V3_HORIZONS) * DT`` = 6.0 s.
PLAN_HORIZON_S: float = 6.0
#: the strategic label's farthest arc-length anchor (``tanitad/data/lan.py``,
#: ``(20, 40, 80, 160) m``): a window whose commanded turn starts beyond it
#: has a legitimately STRAIGHT strategic bearing and is not informative for
#: ``g_str``.
STRATEGIC_REACH_M: float = 160.0
#: ``physicalai.py:621`` legacy const2p9 steer encoding, inverted the same way
#: ``ego_state_at_t0`` inverts it.
WHEELBASE_M: float = 2.9
#: below these the block is UNPOWERED, never "no effect".
N_MIN_WINDOWS: int = 30
N_MIN_EPISODES: int = 5
#: the conditionings, in the order the harness rolls them. ``nav_flipped``
#: (left <-> right, ``--with-navflip``) is OPTIONAL and the sharpest: it
#: disagrees with the truth on EVERY informative window, where the shuffle feeds
#: ``follow`` on most of them. Harvested from the predecessor's draft
#: (``flip_nav``); the two REQUIRED controls stay shuffle + zero (BACKLOG R39).
CONDITIONS: tuple[str, ...] = ("nav_true", "nav_shuffled", "nav_zero", "nav_flipped")
REQUIRED_CONDITIONS: tuple[str, ...] = ("nav_true", "nav_shuffled", "nav_zero")
#: the strata a left/right window falls into, from the label's turn timing and
#: the GT (predecessor's draft, kept): the PRIMARY is ``imminent``; ``deferred``
#: is where "consistency with the strategic goals" is read.
STRATA: tuple[str, ...] = ("imminent", "deferred", "stale", "conflict",
                           "ambiguous", "follow", "unlabeled")
#: a plan whose last segment is shorter than this is STALLED and carries no
#: heading — read as 0 (no turn), counted, never NaN-propagated.
STALL_SEGMENT_M: float = 0.05

FALSIFIABILITY = (
    "The old strategic metric scored a HEAD against a LABEL that is a bijection "
    "of the nav token fed at inference (141/141 MEASURED): copying the input "
    "scores 1.0000 and no behaviour can lower it. This metric scores the "
    "BEHAVIOUR (emitted path / selected anchor / goal bearing) against the "
    "command, and it is decided by the INTERVENTION PAIR, not by the rate: "
    "compliance with the TRUE command must DROP under nav-SHUFFLE and nav-ZERO "
    "on the same windows (paired episode-cluster bootstrap), and on the changed "
    "subset the behaviour must follow the FED command. A rate that does not "
    "drop is coincidence with the scene, and the nav-blind controls (ha0_ext) "
    "measure exactly that floor on the same windows. Nothing here can be "
    "satisfied by echoing a token, because a path is not a token.")


# --------------------------------------------------------------------------- #
# geometry                                                                     #
# --------------------------------------------------------------------------- #
def wrap_pi(a):
    a = np.asarray(a, dtype=np.float64)
    return np.arctan2(np.sin(a), np.cos(a))


def commanded_side(nav) -> np.ndarray:
    """``+1`` left, ``-1`` right, ``0`` follow/straight/unknown. ``[N]``."""
    nav = np.asarray(nav, dtype=np.int64)
    side = np.zeros(nav.shape, dtype=np.int64)
    side[nav == NAV_LEFT] = 1
    side[nav == NAV_RIGHT] = -1
    return side


def flip_nav(nav) -> np.ndarray:
    """left <-> right; follow / straight / anything else unchanged. The
    intervention that disagrees with the truth on EVERY commanded window."""
    nav = np.asarray(nav, dtype=np.int64).copy()
    left, right = nav == NAV_LEFT, nav == NAV_RIGHT
    nav[left], nav[right] = NAV_RIGHT, NAV_LEFT
    return nav


def assign_strata(nav_true, nav_valid, t_now_s, t_start_s, t_end_s, gt_signal, *,
                  tau: float, horizon_s: float = PLAN_HORIZON_S,
                  min_overlap_s: float = 1.0) -> np.ndarray:
    """One stratum per window (``STRATA``), from the label's turn timing and
    the GT signal (signed heading change over the plan horizon):

      imminent   commanded turn overlaps the plan horizon AND the GT executes
                 it (commanded sign, |GT| >= tau)         -> the PRIMARY set
      conflict   overlaps, but the GT turns the OTHER way >= tau (a curve
                 before the junction)                      -> excluded, counted
      ambiguous  overlaps, but the GT clears no threshold  -> excluded, counted
      deferred   the turn starts BEYOND the plan horizon   -> plan must HOLD,
                 g_str should already point the way (the consistency reading)
      stale      the commanded turn ended before t_now     -> excluded, counted
      follow     a valid follow token
      unlabeled  no record / invalid token / no commanded-turn timing
    """
    nav_true = np.asarray(nav_true, dtype=np.int64)
    valid = np.asarray(nav_valid, dtype=bool)
    tn = np.asarray(t_now_s, dtype=np.float64)
    ts = np.asarray(t_start_s, dtype=np.float64)
    te = np.asarray(t_end_s, dtype=np.float64)
    g = np.asarray(gt_signal, dtype=np.float64)
    side = commanded_side(nav_true)
    out = np.full(nav_true.shape, "unlabeled", dtype=object)
    out[valid & (nav_true == NAV_FOLLOW)] = "follow"
    lr = valid & (side != 0) & np.isfinite(ts) & np.isfinite(te)
    with np.errstate(invalid="ignore"):
        overlap = np.minimum(te, tn + horizon_s) - np.maximum(ts, tn)
        gt_ok = (np.sign(g) == np.sign(side)) & (np.abs(g) >= tau)
        gt_other = (np.sign(g) == -np.sign(side)) & (np.abs(g) >= tau)
    out[lr & (te <= tn)] = "stale"
    inh = lr & (te > tn) & (overlap >= min_overlap_s)
    out[inh & gt_ok] = "imminent"
    out[inh & ~gt_ok & gt_other] = "conflict"
    out[inh & ~gt_ok & ~gt_other] = "ambiguous"
    out[inh & ~np.isfinite(g)] = "ambiguous"
    out[lr & (te > tn) & (ts > tn + horizon_s)] = "deferred"
    out[lr & (te > tn) & ~(ts > tn + horizon_s) & ~inh] = "ambiguous"
    return out.astype(str)


def terminal_heading(path, *, stall_m: float = STALL_SEGMENT_M) -> np.ndarray:
    """Tangent direction of the LAST segment of an ego-frame path ``[N, S, 2]``
    → ``[N]`` radians (0 = straight ahead, + = left). A stalled last segment
    (``< stall_m``) reads 0.0 — no heading is carried by a standstill."""
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 3 or p.shape[-1] != 2 or p.shape[1] < 2:
        raise ValueError(f"path must be [N, S>=2, 2], got {p.shape}")
    d = p[:, -1] - p[:, -2]
    seg = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 1], d[:, 0])
    th[seg < stall_m] = 0.0
    return th


def end_bearing(path, *, stall_m: float = STALL_SEGMENT_M) -> np.ndarray:
    """Bearing of the END POINT from the origin, ``[N]`` radians. The signal
    ``_lan_anchor_prior`` scores anchors on (``refc.py``: *terminal bearing*)."""
    p = np.asarray(path, dtype=np.float64)
    e = p[:, -1]
    r = np.hypot(e[:, 0], e[:, 1])
    b = np.arctan2(e[:, 1], e[:, 0])
    b[r < stall_m] = 0.0
    return b


def lateral_end(path) -> np.ndarray:
    """Lateral displacement of the end point, metres, + = left. ``[N]``."""
    return np.asarray(path, dtype=np.float64)[:, -1, 1].copy()


def ego_frame(future_xy, pose_last) -> np.ndarray:
    """World ``[N, K, 2]`` positions → ego frame of ``pose_last`` ``[N, 4]``
    (x, y, yaw, v): rotate the displacement by ``-yaw``; +x forward, +y left —
    ``refb_labels.waypoint_targets``' convention, re-implemented in numpy and
    pinned by test against a hand-computed rotation."""
    f = np.asarray(future_xy, dtype=np.float64)
    pl = np.asarray(pose_last, dtype=np.float64)
    d = f - pl[:, None, :2]
    c, s = np.cos(-pl[:, 2])[:, None], np.sin(-pl[:, 2])[:, None]
    return np.stack([d[..., 0] * c - d[..., 1] * s,
                     d[..., 0] * s + d[..., 1] * c], axis=-1)


def _kinematic_arc(v0, a0, horizon_s: float) -> np.ndarray:
    """Arc length after ``horizon_s`` under constant ``a0`` from ``v0`` — the
    numpy twin of ``echo_gate._np_extrapolate``'s arc (same stop clamp: a
    decelerating vehicle STOPS rather than reversing)."""
    v0 = np.asarray(v0, dtype=np.float64)
    a0 = np.asarray(a0, dtype=np.float64)
    t = np.full_like(v0, float(horizon_s))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stop = np.where(a0 < 0, -v0 / a0, np.inf)
    t = np.minimum(t, np.where(np.isfinite(t_stop), t_stop, t))
    t = np.maximum(t, 0.0)
    return np.maximum(v0 * t + 0.5 * a0 * t * t, 0.0)


def kinematic_heading(v0, a0, k0, horizon_s: float = PLAN_HORIZON_S) -> np.ndarray:
    """Heading change after ``horizon_s`` under constant ``a0`` and constant
    curvature ``k0`` from speed ``v0`` = ``k0 * arc``."""
    return np.asarray(k0, dtype=np.float64) * _kinematic_arc(v0, a0, horizon_s)


def kinematic_end_signals(v0, a0, k0, horizon_s: float = PLAN_HORIZON_S) -> dict:
    """The THREE plan signals of the constant-(a, kappa) extrapolation, so a
    nav-blind control can be scored with the SAME signal and tolerance as the
    readout it floors: ``plan`` (terminal heading), ``plan_bearing`` (end
    bearing), ``plan_lateral`` (lateral end, m). Same geometry as
    ``echo_gate._np_extrapolate`` (``x = sin(ks)/k``, ``y = (1-cos ks)/k``)."""
    k0 = np.asarray(k0, dtype=np.float64)
    s = _kinematic_arc(v0, a0, horizon_s)
    th = k0 * s
    small = np.abs(k0) < 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(small, s, np.sin(th) / k0)
        y = np.where(small, 0.5 * k0 * s * s, (1.0 - np.cos(th)) / k0)
    return {"plan": th, "plan_bearing": np.arctan2(y, x), "plan_lateral": y}


# --------------------------------------------------------------------------- #
# the informative windows and the corpus-derived tolerance                     #
# --------------------------------------------------------------------------- #
def informative_mask(nav_true, nav_valid, t_now_s, t_start_s, t_end_s, *,
                     horizon_s: float = PLAN_HORIZON_S,
                     min_overlap_s: float = 1.0) -> np.ndarray:
    """Windows where FOLLOWING THE COMMAND SHOWS in a readout of reach
    ``horizon_s``: the true command is left/right, the token is valid, and the
    commanded manoeuvre ``[t_start, t_end]`` overlaps ``[t_now, t_now +
    horizon]`` by at least ``min_overlap_s``. NaN timing ⇒ not informative."""
    side = commanded_side(nav_true) != 0
    valid = np.asarray(nav_valid, dtype=bool)
    tn = np.asarray(t_now_s, dtype=np.float64)
    ts = np.asarray(t_start_s, dtype=np.float64)
    te = np.asarray(t_end_s, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        overlap = np.minimum(te, tn + horizon_s) - np.maximum(ts, tn)
    ok = np.isfinite(overlap) & (overlap >= min_overlap_s)
    return side & valid & ok


def derive_tolerance(pos, neg, *, grid: Sequence[float] | None = None,
                     unit: str = "rad") -> dict:
    """The corpus-derived dead-band ``tau``: the magnitude that best separates
    the GT signal on INFORMATIVE windows (``pos``, a turn is being executed)
    from FOLLOW windows (``neg``) — Youden's J over a grid. Derived from the
    LABEL and the GT only, never from a model output, and reported with the
    percentiles so the choice is auditable. ``pos``/``neg`` are SIGNED; the
    magnitude is what is thresholded."""
    p = np.abs(np.asarray(pos, dtype=np.float64))
    n = np.abs(np.asarray(neg, dtype=np.float64))
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    if p.size == 0 or n.size == 0:
        return {"tau": None, "unit": unit, "n_pos": int(p.size), "n_neg": int(n.size),
                "status": "UNAVAILABLE",
                "reason": "need both informative and follow windows with a GT signal"}
    if grid is None:
        hi = float(np.nanmax(np.concatenate([p, n])))
        grid = np.linspace(0.0, hi, 401)[1:]
    best = None
    for tau in grid:
        tpr = float((p >= tau).mean())
        fpr = float((n >= tau).mean())
        j = tpr - fpr
        if best is None or j > best["youden_j"] + 1e-12:
            best = {"tau": float(tau), "youden_j": round(j, 4),
                    "tpr_informative": round(tpr, 4), "fpr_follow": round(fpr, 4)}
    best.update({"unit": unit, "n_pos": int(p.size), "n_neg": int(n.size),
                 "p10_informative": round(float(np.percentile(p, 10)), 4),
                 "p50_informative": round(float(np.percentile(p, 50)), 4),
                 "p50_follow": round(float(np.percentile(n, 50)), 4),
                 "p90_follow": round(float(np.percentile(n, 90)), 4),
                 "status": "OK",
                 "rule": ("argmax over the grid of TPR(informative |GT| >= tau) - "
                          "FPR(follow |GT| >= tau); derived from GT + label only")})
    return best


def complies(signal, side, tau: float) -> np.ndarray:
    """1.0 where the SIGNED signal has the commanded sign AND a magnitude of at
    least ``tau``; 0.0 otherwise (including where ``side == 0``). ``[N]``."""
    s = np.asarray(signal, dtype=np.float64)
    sd = np.asarray(side, dtype=np.float64)
    ok = (np.sign(s) == np.sign(sd)) & (np.abs(s) >= float(tau)) & (sd != 0)
    return ok.astype(np.float64)


# --------------------------------------------------------------------------- #
# estimator                                                                    #
# --------------------------------------------------------------------------- #
def _ci():
    """``taniteval.ci`` — the ONLY decision-grade estimator (CLAUDE.md). Relative
    import first (this file lives INSIDE the package), then the absolute one for
    a caller that loaded this module by path."""
    try:
        from . import ci as _c            # type: ignore
        return _c
    except ImportError:
        from taniteval import ci as _c    # type: ignore
        return _c


def _rate(vals, eid, n_boot, seed) -> dict:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {"mean": None, "n_windows": 0, "n_episodes": 0,
                "status": "UNAVAILABLE", "reason": "no windows"}
    return _ci().episode_cluster_bootstrap(vals, list(eid), n_boot=n_boot, seed=seed)


def _paired(a, b, eid, n_boot, seed) -> dict:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return {"delta": None, "separated": None, "n_windows": 0,
                "status": "UNAVAILABLE", "reason": "no windows"}
    return _ci().paired_episode_cluster_bootstrap(a, b, list(eid), n_boot=n_boot,
                                                  seed=seed)


# --------------------------------------------------------------------------- #
# one readout, three conditionings                                             #
# --------------------------------------------------------------------------- #
def compliance_arm(signal_by_cond: Mapping[str, np.ndarray],
                   nav_fed_by_cond: Mapping[str, np.ndarray],
                   nav_true, informative, eid, *, tau: float,
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """The block for ONE behavioural readout.

    ``signal_by_cond[c]`` ``[N]`` signed signal of the model under conditioning
    ``c`` (``nav_true`` required; ``nav_shuffled`` / ``nav_zero`` optional and
    REFUSED with a reason when absent — a block without them is not
    admissible). ``nav_fed_by_cond[c]`` ``[N]`` the token the model was fed
    under ``c`` (``nav_zero`` ⇒ all FOLLOW). ``informative`` ``[N]`` bool.
    """
    nav_true = np.asarray(nav_true, dtype=np.int64)
    inf = np.asarray(informative, dtype=bool)
    eid = np.asarray(eid)
    side_true = commanded_side(nav_true)
    n_inf, n_ep = int(inf.sum()), int(len(set(eid[inf].tolist())))
    out: dict = {"n_informative_windows": n_inf, "n_informative_episodes": n_ep,
                 "tau": float(tau), "estimator": "episode_cluster_bootstrap / paired",
                 "n_boot": int(n_boot), "seed": int(seed), "conditionings": {}}
    if "nav_true" not in signal_by_cond:
        raise ValueError("compliance_arm needs the nav_true readout")
    powered = n_inf >= N_MIN_WINDOWS and n_ep >= N_MIN_EPISODES
    out["powered"] = bool(powered)
    if not powered:
        out["status"] = "UNPOWERED"
        out["reason"] = (f"{n_inf} informative windows / {n_ep} episodes < "
                         f"{N_MIN_WINDOWS} / {N_MIN_EPISODES}: nothing below is a "
                         f"finding about the model")
    comp: dict[str, np.ndarray] = {}
    for c in CONDITIONS:
        if c not in signal_by_cond or signal_by_cond[c] is None:
            if c not in REQUIRED_CONDITIONS:
                continue                       # nav_flipped is optional
            out["conditionings"][c] = {
                "status": "UNAVAILABLE", "n": n_inf,
                "reason": f"the {c} conditioning was not rolled — ⛔ the block is "
                          f"INADMISSIBLE without nav_shuffled AND nav_zero"}
            continue
        sig = np.asarray(signal_by_cond[c], dtype=np.float64)
        comp[c] = complies(sig, side_true, tau)
        out["conditionings"][c] = {
            "compliance_with_TRUE_command": _rate(comp[c][inf], eid[inf], n_boot, seed),
            "n": n_inf}
    # ---- the paired deltas — the metric itself --------------------------- #
    for c in ("nav_shuffled", "nav_zero", "nav_flipped"):
        key = f"paired_true_minus_{c.split('_')[1]}"
        if c in comp:
            out[key] = _paired(comp["nav_true"][inf], comp[c][inf], eid[inf],
                               n_boot, seed)
        elif c in REQUIRED_CONDITIONS:
            out[key] = {"status": "UNAVAILABLE", "n": n_inf,
                        "reason": f"{c} not rolled"}
    # ---- the flip: EVERY informative window is fed the opposite turn -------- #
    if "nav_flipped" in comp and "nav_flipped" in nav_fed_by_cond:
        fed_f = np.asarray(nav_fed_by_cond["nav_flipped"], dtype=np.int64)
        sig_f = np.asarray(signal_by_cond["nav_flipped"], dtype=np.float64)
        out["flipped"] = {
            "n": n_inf,
            "follows_FED_command": _rate(complies(sig_f, commanded_side(fed_f), tau)[inf],
                                         eid[inf], n_boot, seed),
            "follows_TRUE_command": _rate(comp["nav_flipped"][inf], eid[inf], n_boot, seed),
            "_reading": ("under the flip the fed and true sides are opposite on every "
                         "informative window: a follower reads follows_FED high / "
                         "follows_TRUE low; a scene-driven arm keeps follows_TRUE at "
                         "its nav_true rate; a blind token-echo would read exactly 1/0")}
    # ---- the changed subset under shuffle: does the behaviour follow the FED
    # (wrong) command? This is the direction that separates 'follows nav' from
    # 'coincides with the scene'.
    if "nav_shuffled" in comp and "nav_shuffled" in nav_fed_by_cond:
        fed = np.asarray(nav_fed_by_cond["nav_shuffled"], dtype=np.int64)
        side_fed = commanded_side(fed)
        changed = inf & (fed != nav_true)
        fed_turn = changed & (side_fed != 0)          # fed a left/right ≠ true
        sig_s = np.asarray(signal_by_cond["nav_shuffled"], dtype=np.float64)
        blk = {"n_changed": int(changed.sum()),
               "n_changed_fed_a_turn": int(fed_turn.sum()),
               "n_changed_fed_follow": int((changed & (side_fed == 0)).sum())}
        if fed_turn.sum():
            blk["follows_FED_command"] = _rate(
                complies(sig_s, side_fed, tau)[fed_turn], eid[fed_turn], n_boot, seed)
            blk["follows_TRUE_command"] = _rate(
                comp["nav_shuffled"][fed_turn], eid[fed_turn], n_boot, seed)
            blk["_reading"] = (
                "on windows fed a DIFFERENT turn than the truth, FED and TRUE "
                "compliance are mutually exclusive by construction: a follower "
                "reads follows_FED high / follows_TRUE low; a scene-driven arm "
                "reads follows_TRUE ≈ its nav_true rate regardless of the token")
        else:
            blk["status"] = "UNAVAILABLE"
            blk["reason"] = "the permutation fed no different turn on any informative window"
        out["changed_subset"] = blk
    # ---- the ceiling: a synthetic policy that ALWAYS executes the fed command
    if "nav_shuffled" in nav_fed_by_cond:
        fed = np.asarray(nav_fed_by_cond["nav_shuffled"], dtype=np.int64)
        same = (commanded_side(fed) == side_true).astype(np.float64)
        out["always_commanded_ceiling"] = {
            "compliance_nav_true": 1.0,
            "compliance_nav_shuffled": round(float(same[inf].mean()), 4) if n_inf else None,
            "delta_true_minus_shuffled": (round(float(1.0 - same[inf].mean()), 4)
                                          if n_inf else None),
            "_is": ("a policy that turns exactly as commanded scores 1.0 under "
                    "the true token and, under the shuffle, only where the fed "
                    "side equals the true side — its delta is the LARGEST shuffle "
                    "drop any follower can show on these windows")}
    out["_reads"] = (
        "compliance_with_TRUE_command under nav_true is NOT the result; the "
        "paired_true_minus_shuffled and paired_true_minus_zero deltas are. A "
        "rate that does not drop under either intervention is coincidence with "
        "the scene, however high it reads.")
    return out


def control_arm(signal, nav_true, informative, eid, *, tau: float,
                nav_fed_shuffled=None, n_boot: int = 2000, seed: int = 0,
                expect: str = "nav_blind") -> dict:
    """A NAV-BLIND reference (``ha0``/``ha``/``ha0_ext``/``gt``) on the same
    windows. Its shuffle delta is IDENTICALLY zero — asserted, and written as
    the known value it must read — and its C(true) is the coincidence floor."""
    nav_true = np.asarray(nav_true, dtype=np.int64)
    inf = np.asarray(informative, dtype=bool)
    eid = np.asarray(eid)
    c_true = complies(np.asarray(signal, dtype=np.float64),
                      commanded_side(nav_true), tau)
    out = {"compliance_with_TRUE_command": _rate(c_true[inf], eid[inf], n_boot, seed),
           "n": int(inf.sum()), "expect": expect}
    # nav-blind ⇒ the same signal under every conditioning ⇒ the paired delta is
    # exactly 0 and the bootstrap is degenerate BY CONSTRUCTION. Say so instead
    # of printing a degenerate interval that reads like evidence.
    out["delta_true_minus_shuffled"] = 0.0
    out["delta_true_minus_zero"] = 0.0
    out["_known_value"] = ("nav-blind by construction: both deltas are identically "
                           "0.0000. A non-zero value here means the CONTROL saw nav "
                           "and the panel is broken.")
    if expect == "zero":
        v = out["compliance_with_TRUE_command"].get("mean")
        out["reads_known_value"] = (v == 0.0)
        out["_known_value"] += (" This control is the straight line: compliance "
                                "must read 0.0000 EXACTLY.")
    return out


# --------------------------------------------------------------------------- #
# verdicts                                                                     #
# --------------------------------------------------------------------------- #
def verdict(arm: dict, floor: dict | None = None, *, min_delta: float = 0.0) -> dict:
    """Read ONE readout's block. ``floor`` is the ``ha0_ext`` control block."""
    if not arm.get("powered", False):
        return {"verdict": "UNPOWERED", "reason": arm.get("reason")}
    d_s = arm.get("paired_true_minus_shuffled") or {}
    d_z = arm.get("paired_true_minus_zero") or {}
    c_t = ((arm.get("conditionings") or {}).get("nav_true") or {}).get(
        "compliance_with_TRUE_command") or {}
    if d_s.get("delta") is None:
        return {"verdict": "INADMISSIBLE", "reason": "no nav_shuffled conditioning"}
    follows_s = bool(d_s.get("separated")) and float(d_s["delta"]) >= min_delta and float(d_s["delta"]) > 0
    follows_z = bool(d_z.get("separated")) and (d_z.get("delta") or 0) > 0
    anti = bool(d_s.get("separated")) and float(d_s["delta"]) < 0
    above_floor = None
    if floor is not None:
        f = (floor.get("compliance_with_TRUE_command") or {}).get("mean")
        m = c_t.get("mean")
        if f is not None and m is not None:
            # a rate CI against a point floor: the floor is deterministic given the
            # windows, so 'above' is the model's CI lower bound clearing it.
            above_floor = bool(c_t.get("lo", m) > f)
    if anti:
        v, why = "ANTI_COMPLIANT", "compliance with the TRUE command is HIGHER under the shuffle"
    elif follows_s and follows_z:
        v, why = "FOLLOWS_NAV", "compliance drops separated under BOTH shuffle and zero"
    elif follows_s:
        v, why = "FOLLOWS_NAV_SHUFFLE_ONLY", ("drops under shuffle but not under zero: "
                                             "nav_cmd=None collapses to the follow "
                                             "embedding, a lower bound on nav dependence")
    elif above_floor:
        v, why = "NAV_BLIND_COINCIDENT", ("the rate is above the nav-blind floor but does "
                                         "NOT drop under the interventions: the turn is "
                                         "read from the scene/ego, not from the command")
    elif above_floor is False:
        v, why = "NAV_BLIND_AT_FLOOR", ("neither follows the command nor clears the "
                                       "ego-extrapolation floor on these windows")
    else:
        v, why = "NAV_BLIND", "no separated drop under either intervention"
    return {"verdict": v, "reason": why,
            "delta_shuffle": d_s.get("delta"), "delta_shuffle_ci": [d_s.get("lo"), d_s.get("hi")],
            "delta_zero": d_z.get("delta"), "delta_zero_ci": [d_z.get("lo"), d_z.get("hi")],
            "rate_nav_true": c_t.get("mean"), "above_nav_blind_floor": above_floor,
            "min_delta_required": min_delta}


def seam_reading(v_plan: dict, v_gstr: dict, v_anchor: dict | None = None) -> dict:
    """Localise a failure: right goal + wrong path is a SEAM failure."""
    p, g = v_plan.get("verdict"), v_gstr.get("verdict")
    a = (v_anchor or {}).get("verdict")
    fol = {"FOLLOWS_NAV", "FOLLOWS_NAV_SHUFFLE_ONLY"}
    if p == "UNPOWERED" or g == "UNPOWERED":
        r = "UNPOWERED"
    elif g in fol and p in fol:
        r = "GOAL_AND_PATH_FOLLOW"
    elif g in fol and p not in fol:
        r = "SEAM_FAILURE_RIGHT_GOAL_WRONG_PATH"
    elif p in fol and g not in fol:
        r = "PATH_FOLLOWS_WITHOUT_GOAL"
    else:
        r = "NEITHER_FOLLOWS"
    out = {"reading": r, "plan": p, "gstr": g, "anchor": a}
    out["_reads"] = {
        "GOAL_AND_PATH_FOLLOW": "the command reaches the strategic goal AND the emitted path",
        "SEAM_FAILURE_RIGHT_GOAL_WRONG_PATH": ("g_str follows the command but the path does "
                                              "not: the E4/E7/E9 seam does not carry the "
                                              "goal into behaviour — a hierarchy defect, "
                                              "not a nav defect"),
        "PATH_FOLLOWS_WITHOUT_GOAL": ("the path follows nav while g_str does not: nav "
                                      "reaches the operative layer (measurement encoder "
                                      "→ decoder) without the strategic decision — the "
                                      "inversion D-REFCV4-NAV-WIRING measured in refcv3"),
        "NEITHER_FOLLOWS": "no readout follows the command on these windows",
        "UNPOWERED": "too few informative windows to read the seam",
    }[r]
    if a is not None and p in fol and a not in fol:
        out["selection_note"] = ("the emitted path follows but the SELECTED anchor does "
                                 "not: the compliance comes from the refinement offset, "
                                 "not from selection (selection is decided at t=0)")
    return out


# --------------------------------------------------------------------------- #
# labels                                                                       #
# --------------------------------------------------------------------------- #
def load_labels(path: str) -> dict:
    """``clip_id -> {nav_token, side, t0_s, turns: [(t_start_rel, t_end_rel,
    dyaw_deg, is_turn)]}`` from the v7.2 jsonl.gz. Times in
    ``manoeuvre_sequence`` are RELATIVE TO ``t0_s`` (``s2_geom_emit_v7.
    manoeuvre_sequence``: ``round(i / hz, 1)`` with ``i`` indexing from
    ``key``) — ``time_base_control`` VERIFIES that reading on the GT rather
    than trusting this docstring."""
    import gzip
    out = {}
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            nav = (r.get("nav_command") or {})
            tok = str(nav.get("token") or "NAV_FOLLOW_ROAD")
            side = 1 if tok.endswith("_L") else (-1 if tok.endswith("_R") else 0)
            turns = [(float(m["t_start_s"]), float(m["t_end_s"]),
                      float(m["dyaw_deg"]), bool(m.get("is_turn", False)))
                     for m in (r.get("manoeuvre_sequence") or [])]
            out[str(r["clip_id"])] = {
                "nav_token": tok, "side": side, "t0_s": float(r.get("t0_s", 8.0)),
                "nav_time_rel_s": (nav.get("args") or {}).get("time_s"),
                "nav_distance_m": (nav.get("args") or {}).get("distance_m"),
                "turns": turns}
    return out


def _commanded_turn(rec: dict):
    """The manoeuvre the nav token names: the FIRST entry of the sequence
    (``nav_command``: ``nxt = seq[0]``) when it is a turn on the commanded side;
    else the first ``is_turn`` entry on that side. ``None`` if none."""
    side = rec["side"]
    if side == 0 or not rec["turns"]:
        return None
    first = rec["turns"][0]
    if first[3] and np.sign(first[2]) == side:
        return first
    for t in rec["turns"]:
        if t[3] and np.sign(t[2]) == side:
            return t
    return None


def join_windows_to_labels(clip_of_window: Sequence[str], t_now_s, labels: dict,
                           *, time_base: str = "relative") -> dict:
    """Per-window commanded-turn timing. ``time_base`` = ``relative`` (turn
    times are ``t0_s + t_rel``) or ``absolute``. Returns arrays ``[N]``:
    ``t_start_s``, ``t_end_s``, ``dyaw_deg`` (NaN where no commanded turn),
    ``side`` and ``has_record``."""
    n = len(clip_of_window)
    ts = np.full(n, np.nan)
    te = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    side = np.zeros(n, dtype=np.int64)
    has = np.zeros(n, dtype=bool)
    for i, cid in enumerate(clip_of_window):
        rec = labels.get(str(cid))
        if rec is None:
            continue
        has[i] = True
        side[i] = rec["side"]
        ct = _commanded_turn(rec)
        if ct is None:
            continue
        off = rec["t0_s"] if time_base == "relative" else 0.0
        ts[i], te[i], dy[i] = off + ct[0], off + ct[1], ct[2]
    return {"t_start_s": ts, "t_end_s": te, "dyaw_deg": dy, "side": side,
            "has_record": has, "time_base": time_base}


def time_base_control(ep_yaw_by_clip: Mapping[str, np.ndarray], labels: dict, *,
                      dt: float = DT, raw_offset_frames: int = 0) -> dict:
    """⭐ THE CONTROL THAT DECIDES THE TIME BASE ON THE GT, NOT ON A DOCSTRING.

    For every left/right clip with a commanded turn, integrate the recorded yaw
    over the turn's span under BOTH hypotheses (relative to ``t0_s`` /
    absolute clip time) and count on how many clips the measured heading change
    reproduces the label's ``dyaw_deg`` (same sign, ≥ half the magnitude). The
    hypothesis that reproduces more wins; if neither reproduces on ≥ 50 % of
    the clips the join is broken and the metric is UNPOWERED.

    ``ep_yaw_by_clip[clip]`` is the PROVIDER's yaw series ``[T]`` (provider
    row j = raw frame ``j + raw_offset_frames``)."""
    res = {"relative": {"n": 0, "match": 0}, "absolute": {"n": 0, "match": 0}}
    per_clip = []
    for cid, rec in labels.items():
        if rec["side"] == 0 or cid not in ep_yaw_by_clip:
            continue
        ct = _commanded_turn(rec)
        if ct is None:
            continue
        yaw = np.unwrap(np.asarray(ep_yaw_by_clip[cid], dtype=np.float64))
        T = yaw.shape[0]
        row = {"clip": cid, "dyaw_deg": ct[2]}
        for hyp, off in (("relative", rec["t0_s"]), ("absolute", 0.0)):
            a = int(round((off + ct[0]) / dt)) - raw_offset_frames
            b = int(round((off + ct[1]) / dt)) - raw_offset_frames
            a, b = max(0, min(a, T - 1)), max(0, min(b, T - 1))
            if b <= a:
                row[hyp] = None
                continue
            d = math.degrees(yaw[b] - yaw[a])
            ok = (np.sign(d) == np.sign(ct[2])) and (abs(d) >= 0.5 * abs(ct[2]))
            row[hyp] = round(d, 2)
            res[hyp]["n"] += 1
            res[hyp]["match"] += int(ok)
        per_clip.append(row)
    for hyp in res:
        n = res[hyp]["n"]
        res[hyp]["frac"] = round(res[hyp]["match"] / n, 4) if n else None
    fr, fa = res["relative"]["frac"] or 0.0, res["absolute"]["frac"] or 0.0
    if max(fr, fa) < 0.5:
        chosen, status = None, "BROKEN"
    else:
        chosen, status = ("relative" if fr >= fa else "absolute"), "OK"
    return {"status": status, "chosen": chosen, "hypotheses": res,
            "n_clips_tested": len(per_clip), "per_clip": per_clip[:60],
            "_reads": ("the label's manoeuvre times are integrated against the "
                       "recorded yaw under both time bases; the winner reproduces "
                       "dyaw_deg on more clips. BROKEN ⇒ every informative mask "
                       "below is void")}


# --------------------------------------------------------------------------- #
# the full report                                                              #
# --------------------------------------------------------------------------- #
def nav_compliance_report(*, readouts: Mapping[str, Mapping[str, np.ndarray]],
                          nav_true, nav_valid, nav_fed_by_cond: Mapping[str, np.ndarray],
                          eid, t_now_s, t_start_s, t_end_s, arc_to_turn_m=None,
                          gt_signal: Mapping[str, np.ndarray] | None = None,
                          controls: Mapping[str, np.ndarray] | None = None,
                          fan: Mapping[str, np.ndarray] | None = None,
                          tolerance: Mapping[str, float] | None = None,
                          n_boot: int = 2000, seed: int = 0, tier: str = "T1",
                          tag: str | None = None, min_delta: float = 0.0) -> dict:
    """Assemble the block. ``readouts`` = ``{"plan": {cond: heading[N]}, "gstr":
    {cond: bearing[N]}, "anchor": {cond: heading[N]}}`` (plus optional
    ``plan_bearing``, ``plan_lateral``). ``gt_signal`` = ``{"plan": gt_heading
    [N], "gstr": gt_bearing[N]}`` for the tolerance derivation and the GT
    control. ``controls`` = ``{"ha0": ..., "ha": ..., "ha0_ext": ...}`` signed
    headings ``[N]``. ``fan`` = ``{cond: {"heading": [N, A], "keep": [N, A]}}``.
    """
    nav_true = np.asarray(nav_true, dtype=np.int64)
    nav_valid = np.asarray(nav_valid, dtype=bool)
    eid = np.asarray(eid)
    N = int(nav_true.shape[0])
    inf_plan = informative_mask(nav_true, nav_valid, t_now_s, t_start_s, t_end_s,
                                horizon_s=PLAN_HORIZON_S)
    # strategic reach: the turn starts within the LAN label's farthest anchor
    # (arc-length) — if the arc distance is unknown, fall back to the 30 s label
    # horizon in time.
    side_ok = (commanded_side(nav_true) != 0) & nav_valid
    ts = np.asarray(t_start_s, dtype=np.float64)
    te = np.asarray(t_end_s, dtype=np.float64)
    tn = np.asarray(t_now_s, dtype=np.float64)
    if arc_to_turn_m is not None:
        arc = np.asarray(arc_to_turn_m, dtype=np.float64)
        with np.errstate(invalid="ignore"):
            inf_gstr = side_ok & np.isfinite(arc) & (arc <= STRATEGIC_REACH_M) & (te > tn)
        gstr_rule = f"commanded turn starts within {STRATEGIC_REACH_M:.0f} m of arc length (LAN's farthest anchor) and has not ended"
    else:
        with np.errstate(invalid="ignore"):
            inf_gstr = side_ok & np.isfinite(ts) & ((ts - tn) <= 30.0) & (te > tn)
        gstr_rule = "commanded turn starts within 30 s (STRATEGIC_S upper bound) and has not ended"
    follow = (nav_true == NAV_FOLLOW) & nav_valid
    tol = dict(tolerance or {})
    tol_meta: dict = {}
    if gt_signal is not None:
        for key, mask in (("plan", inf_plan), ("gstr", inf_gstr)):
            if key in gt_signal and key not in tol:
                g = np.asarray(gt_signal[key], dtype=np.float64)
                d = derive_tolerance(g[mask], g[follow])
                tol_meta[key] = d
                if d.get("tau") is not None:
                    tol[key] = float(d["tau"])
    # documented fallbacks, stated: the emitter's own turn threshold (15 deg,
    # `s2_geom_emit_v7.is_turn`) for headings, 5 deg for the goal bearing.
    tol.setdefault("plan", math.radians(15.0))
    tol.setdefault("gstr", math.radians(5.0))
    tol.setdefault("anchor", tol["plan"])
    tol.setdefault("plan_bearing", tol["gstr"])
    tol.setdefault("plan_lateral", 1.0)
    masks = {"plan": inf_plan, "gstr": inf_gstr, "anchor": inf_plan,
             "plan_bearing": inf_plan, "plan_lateral": inf_plan}
    out: dict = {
        "tier": tier, "tag": tag, "n_windows": N,
        "n_episodes": int(len(set(eid.tolist()))),
        "n_nav_valid": int(nav_valid.sum()),
        "n_true_left": int(((nav_true == NAV_LEFT) & nav_valid).sum()),
        "n_true_right": int(((nav_true == NAV_RIGHT) & nav_valid).sum()),
        "n_follow": int(follow.sum()),
        "informative": {
            "plan": {"n_windows": int(inf_plan.sum()),
                     "n_episodes": int(len(set(eid[inf_plan].tolist()))),
                     "rule": (f"true command left/right, token valid, commanded turn "
                              f"overlaps [t_now, t_now + {PLAN_HORIZON_S:.0f} s] by ≥ 1 s")},
            "gstr": {"n_windows": int(inf_gstr.sum()),
                     "n_episodes": int(len(set(eid[inf_gstr].tolist()))),
                     "rule": gstr_rule},
        },
        "tolerance": {"values": {k: round(float(v), 6) for k, v in tol.items()},
                      "units": {"plan": "rad (terminal heading)", "gstr": "rad (bearing)",
                                "anchor": "rad", "plan_bearing": "rad", "plan_lateral": "m"},
                      "derived": tol_meta,
                      "rule": ("Youden-J over the GT signal on informative vs follow "
                               "windows (GT + label only, never a model output); "
                               "fallbacks 15 deg (is_turn's MIN dyaw) / 5 deg stated")},
        "readouts": {}, "controls": {}, "verdicts": {},
        "falsifiability": FALSIFIABILITY,
    }
    for name, sig in readouts.items():
        if name not in masks:
            continue
        out["readouts"][name] = compliance_arm(
            sig, nav_fed_by_cond, nav_true, masks[name], eid, tau=tol[name],
            n_boot=n_boot, seed=seed)
    # ---- nav-blind controls on the PLAN windows, PER READOUT ---------------- #
    # A control is scored with the SAME signal and tolerance as the readout it
    # floors — a heading-signal floor quoted against a bearing or a lateral
    # readout is the `df`/`step_s` scope error. `controls[name]` is either a
    # heading array (scored as the `plan` signal) or a dict of the three plan
    # signals (`kinematic_end_signals`). The top level of each block is the
    # PLAN control (backward-compatible); `per_readout` carries the others. The
    # `anchor` readout shares the plan signal and tau. g_str has NO kinematic
    # floor: no ego extrapolation reaches the strategic horizon.
    ctl = dict(controls or {})
    for cname, sig in ctl.items():
        sigs = dict(sig) if isinstance(sig, Mapping) else {"plan": sig}
        expect = "zero" if cname == "ha0" else "nav_blind"
        if "plan" not in sigs:
            continue
        blk = control_arm(sigs["plan"], nav_true, inf_plan, eid, tau=tol["plan"],
                          n_boot=n_boot, seed=seed, expect=expect)
        blk["per_readout"] = {}
        for rname in ("plan_bearing", "plan_lateral"):
            if rname in sigs and rname in readouts:
                blk["per_readout"][rname] = control_arm(
                    sigs[rname], nav_true, inf_plan, eid, tau=tol[rname],
                    n_boot=n_boot, seed=seed, expect=expect)
        blk["_scored_with"] = "the plan (terminal-heading) signal and tau; per_readout blocks use each readout's own signal and tau"
        out["controls"][cname] = blk
    if gt_signal is not None and "plan" in gt_signal:
        out["controls"]["gt"] = control_arm(
            gt_signal["plan"], nav_true, inf_plan, eid, tau=tol["plan"],
            n_boot=n_boot, seed=seed, expect="label_consistency")
        out["controls"]["gt"]["_known_value"] = (
            "the recorded future on the informative windows: ≈ 1 by construction "
            "of the tolerance; a low value means the window selection or the "
            "time base is wrong, and the block is then not about the model")
    # ---- fan coverage: could the selection have complied? ----------------- #
    if fan:
        cov = {}
        for c, f in fan.items():
            h = np.asarray(f["heading"], dtype=np.float64)
            k = np.asarray(f["keep"], dtype=bool) if f.get("keep") is not None \
                else np.ones_like(h, dtype=bool)
            side = commanded_side(nav_true)
            ok = (np.sign(h) == np.sign(side)[:, None]) & (np.abs(h) >= tol["anchor"]) & k
            any_ok = ok.any(axis=1).astype(np.float64)
            cov[c] = {"fan_has_compliant_candidate": _rate(any_ok[inf_plan], eid[inf_plan],
                                                           n_boot, seed),
                      "compliant_candidates_mean": round(float(ok[inf_plan].sum(1).mean()), 3)
                      if inf_plan.any() else None,
                      "_reads": ("< 1.0 means the vocabulary (after the reach mask) "
                                 "cannot express the commanded turn on some windows — "
                                 "a coverage limit, not a selection failure")}
        out["fan_coverage"] = cov
    # ---- strata census + the DEFERRED consistency block --------------------- #
    # (predecessor's draft, kept): the PRIMARY above is label-timing informative;
    # the census says how it decomposes against the GT, and `deferred` is where
    # the PI's "in consistency to the strategic goals" is read — the plan must
    # HOLD while g_str already points the commanded way.
    if gt_signal is not None and "plan" in gt_signal:
        strata = assign_strata(nav_true, nav_valid, t_now_s, t_start_s, t_end_s,
                               gt_signal["plan"], tau=tol["plan"])
        census = {s: int((strata == s).sum()) for s in STRATA}
        out["strata"] = {"census": census,
                         "primary_is": "label-timing informative (see informative.plan.rule); "
                                       "imminent = that set gated by the GT executing the "
                                       "commanded turn; conflict/ambiguous/stale are its "
                                       "complement and are COUNTED, never scored"}
        deferred = strata == "deferred"
        if arc_to_turn_m is not None:
            arc = np.asarray(arc_to_turn_m, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                deferred = deferred & np.isfinite(arc) & (arc <= STRATEGIC_REACH_M)
        n_def, n_def_ep = int(deferred.sum()), int(len(set(eid[deferred].tolist())))
        cons: dict = {"n_windows": n_def, "n_episodes": n_def_ep,
                      "rule": (f"commanded turn starts beyond the {PLAN_HORIZON_S:.0f} s plan "
                               f"but within {STRATEGIC_REACH_M:.0f} m: the plan must HOLD "
                               f"(|terminal heading| < tau_plan) while g_str points the "
                               f"commanded way (|bearing| >= tau_gstr, right sign)"),
                      "powered": bool(n_def >= N_MIN_WINDOWS and n_def_ep >= N_MIN_EPISODES),
                      "conditionings": {}}
        side_true = commanded_side(nav_true)
        if "plan" in readouts and "gstr" in readouts and n_def:
            hold_c, point_c = {}, {}
            for c in CONDITIONS:
                if c not in readouts["plan"] or c not in readouts["gstr"]:
                    continue
                hold = (np.abs(np.asarray(readouts["plan"][c], dtype=np.float64))
                        < tol["plan"]).astype(np.float64)
                point = complies(readouts["gstr"][c], side_true, tol["gstr"])
                hold_c[c], point_c[c] = hold, point
                cons["conditionings"][c] = {
                    "plan_holds": _rate(hold[deferred], eid[deferred], n_boot, seed),
                    "gstr_points_commanded_way": _rate(point[deferred], eid[deferred],
                                                       n_boot, seed),
                    "consistent_hold_and_point": _rate((hold * point)[deferred],
                                                       eid[deferred], n_boot, seed)}
            for c in ("nav_shuffled", "nav_zero", "nav_flipped"):
                if c in point_c and "nav_true" in point_c:
                    cons[f"gstr_paired_true_minus_{c.split('_')[1]}"] = _paired(
                        point_c["nav_true"][deferred], point_c[c][deferred],
                        eid[deferred], n_boot, seed)
            cons["_reads"] = ("gstr_points is intervention-decided like the primary; "
                              "plan_holds is expected INVARIANT under the interventions "
                              "for a well-behaved follower (holding is right whatever "
                              "the token says while the turn is beyond the plan)")
        out["consistency_deferred"] = cons
    # ---- verdicts --------------------------------------------------------- #
    ext = out["controls"].get("ha0_ext")
    for name in out["readouts"]:
        if name in ("plan", "anchor"):
            floor = ext
        elif name in ("plan_bearing", "plan_lateral"):
            floor = ((ext or {}).get("per_readout") or {}).get(name)
        else:                                  # gstr: no kinematic floor exists
            floor = None
        out["verdicts"][name] = verdict(out["readouts"][name], floor, min_delta=min_delta)
        if name == "gstr":
            out["verdicts"][name]["_floor_note"] = (
                "no nav-blind kinematic floor at the strategic horizon: a "
                "constant-curvature extrapolation to 160 m is not a control. The "
                "follow-window rate of the same readout is the coincidence reference")
    if "plan" in out["verdicts"] and "gstr" in out["verdicts"]:
        out["seam"] = seam_reading(out["verdicts"]["plan"], out["verdicts"]["gstr"],
                                   out["verdicts"].get("anchor"))
    ha0 = out["controls"].get("ha0")
    out["panel_ok"] = bool(ha0 is None or ha0.get("reads_known_value", True))
    if not out["panel_ok"]:
        out["panel_broken_reason"] = ("the straight-line control did not read 0.0000 "
                                      "compliance: no row in this block is admissible")
    return out


def unavailable_block(reason: str, n: int = 0, **extra) -> dict:
    """An honest refusal in the shape the criteria checker reads: the status
    object sits AT each registered key (the nav_true rate and the two paired
    deltas), so ``tools/criteria_check.py`` classifies all three
    ``strat.nav_compliance*`` criteria REFUSED-with-a-reason (work items)
    rather than ABSENT (a silent omission)."""
    st = {"status": "UNAVAILABLE", "reason": reason, "n": int(n)}
    return {"status": "UNAVAILABLE", "reason": reason, "n": int(n),
            "readouts": {r: {"conditionings": {"nav_true": {
                                 "compliance_with_TRUE_command": dict(st)}},
                             "paired_true_minus_shuffled": dict(st),
                             "paired_true_minus_zero": dict(st)}
                         for r in ("plan", "gstr", "anchor")},
            **extra}


# --------------------------------------------------------------------------- #
# the refcv3_arm dump reader                                                   #
# --------------------------------------------------------------------------- #
def _load_sidecar(dump_dir: str) -> tuple[list[dict], dict]:
    import glob
    files = sorted(glob.glob(os.path.join(dump_dir, "decisions", "ep*.npz")))
    if not files:
        raise FileNotFoundError(f"no decisions/ep*.npz under {dump_dir}")
    man_path = os.path.join(dump_dir, "manifest.json")
    with open(man_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    eps = []
    for f in files:
        with np.load(f) as d:
            eps.append({k: d[k] for k in d.files} | {"_file": os.path.basename(f)})
    return eps, manifest


def resolve_labels_path(labels_path, corpus):
    """The labels blob's PATH, from any of the three manifest shapes we have written.

    ⛔ **`corpus["labels"]` IS A DICT ON EVERY DUMP `refcv3_arm.py` HAS EVER
    WRITTEN.** Its manifest literal read ``{"labels": a.labels, **join}`` and
    ``join`` carries its OWN ``"labels"`` key holding the provenance BLOCK; the
    splat comes last, so the dict overwrote the path. Reading it as a path raised
    ``TypeError: os.path.exists(dict)``, which ``from_refcv3_dump``'s caller
    caught as a REFUSAL — so the entire STRATEGIC nav-compliance family read
    UNAVAILABLE on **every arm**, silently, while the other three families
    reported normally. MEASURED 2026-09-05 on the RL panel: 3/3 strategic
    nav-compliance rows lost on 4,823 windows of compute already paid for.

    ⇒ Accept all three shapes, so banked dumps are readable **without a
    re-roll**: the explicit ``labels_path`` key (written from 2026-09-05), the
    provenance dict's ``.path``, and a bare string (the pre-``join`` era).
    An explicit argument always wins.
    """
    if labels_path is None:
        labels_path = corpus.get("labels_path") or corpus.get("labels")
    if isinstance(labels_path, dict):
        labels_path = labels_path.get("path")
    return labels_path


def from_refcv3_dump(dump_dir: str, labels_path: str | None = None, *,
                     n_boot: int = 2000, seed: int = 0, tag: str | None = None,
                     min_delta: float = 0.0) -> dict:
    """Build the block from a ``refcv3_arm.py`` dump whose sidecar carries the
    nav-compliance keys (``plan_full_*``, ``gstr_*``, ``sel_bank_*``,
    ``fan_term_heading_*``, ``reach_keep_*``, ``gt_future_ext``, ``pose_last``,
    ``ego_t0``, ``ep_poses``). Dumps written before those keys existed get an
    UNAVAILABLE block with the reason, never a number."""
    eps, manifest = _load_sidecar(dump_dir)
    need = ("plan_full_nav_true", "gstr_nav_true", "gt_future_ext", "pose_last",
            "ego_t0", "ep_poses", "nav_cmd", "nav_cmd_shuf", "nav_valid", "ws")
    missing = [k for k in need if k not in eps[0]]
    n_all = int(sum(int(e["ws"].shape[0]) for e in eps))
    if missing:
        return unavailable_block(
            f"the dump's sidecar predates the nav-compliance keys (missing "
            f"{missing}); re-roll with the current refcv3_arm.py — nothing here "
            f"can be computed post hoc", n_all)
    labels_path = resolve_labels_path(labels_path, manifest.get("corpus") or {})
    if not labels_path or not os.path.exists(labels_path):
        return unavailable_block(
            f"labels blob not found at {labels_path!r}; pass --labels", n_all)
    labels = load_labels(labels_path)
    ep_man = {e["file_index"]: e for e in manifest.get("episodes", [])}
    raw_off = int((((manifest.get("corpus") or {}).get("frames") or {})
                   .get("provider_to_raw_frame_offset", 0)))
    # ---- flatten ------------------------------------------------------------ #
    cols: dict[str, list] = {}
    clip_of_window, eid = [], []
    ep_yaw_by_clip = {}
    for e in eps:
        fi = int(str(e["_file"]).replace("ep", "").replace(".npz", ""))
        cid = str(ep_man.get(fi, {}).get("clip_id", f"file{fi}"))
        n = int(e["ws"].shape[0])
        clip_of_window += [cid] * n
        eid += [e["_file"]] * n
        ep_yaw_by_clip[cid] = np.asarray(e["ep_poses"])[:, 2]
        for k, v in e.items():
            if k.startswith("_") or k == "ep_poses":
                continue
            v = np.asarray(v)
            if v.shape[0] == n:
                cols.setdefault(k, []).append(v)
    A = {k: np.concatenate(v) for k, v in cols.items()}
    N = len(eid)
    eid = np.asarray(eid)
    nav_true = A["nav_cmd"].astype(np.int64)
    nav_shuf = A["nav_cmd_shuf"].astype(np.int64)
    nav_valid = A["nav_valid"].astype(bool)
    t_now = (A["ws"].astype(np.float64) + raw_off) * DT
    # ---- the time base, decided on the GT ----------------------------------- #
    tb = time_base_control(ep_yaw_by_clip, labels, dt=DT, raw_offset_frames=raw_off)
    if tb["status"] != "OK":
        return unavailable_block(
            "the label time base could not be reproduced on the GT yaw under "
            "either hypothesis — the join is broken and no informative mask exists",
            N, time_base_control=tb)
    join = join_windows_to_labels(clip_of_window, t_now, labels, time_base=tb["chosen"])
    # ---- arc length from NOW to the commanded turn (for the g_str reach) ---- #
    arc = np.full(N, np.nan)
    pl = A["pose_last"].astype(np.float64)
    gt = A["gt_future_ext"].astype(np.float64)              # [N, 60, 4] world
    gtv = A["gt_future_valid_ext"].astype(bool) if "gt_future_valid_ext" in A \
        else np.ones(gt.shape[:2], dtype=bool)
    for i in range(N):
        if not np.isfinite(join["t_start_s"][i]):
            continue
        k = int(round((join["t_start_s"][i] - t_now[i]) / DT))
        if k <= 0:
            arc[i] = 0.0
        elif k <= gt.shape[1]:
            xy = np.concatenate([pl[i:i + 1, :2], gt[i, :k, :2]], 0)
            arc[i] = float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum())
        else:
            # beyond the 6 s future we hold in the sidecar: extrapolate at v0
            xy = np.concatenate([pl[i:i + 1, :2], gt[i, :, :2]], 0)
            arc[i] = float(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])).sum()
                           + pl[i, 3] * (k - gt.shape[1]) * DT)
    # ---- GT signals ---------------------------------------------------------- #
    gt_full = gtv[:, -1]                                     # 6 s GT not end-clamped
    gt_dpsi = wrap_pi(gt[:, -1, 2] - pl[:, 2])
    gt_dpsi[~gt_full] = np.nan
    gt_ego = ego_frame(gt[:, :, :2], pl)                     # [N, 60, 2]
    gt_bear = end_bearing(gt_ego)
    gt_bear[~gt_full] = np.nan
    # ---- readouts ------------------------------------------------------------ #
    readouts: dict = {"plan": {}, "gstr": {}, "anchor": {}, "plan_bearing": {},
                      "plan_lateral": {}}
    fan: dict = {}
    nav_fed = {"nav_true": nav_true, "nav_shuffled": nav_shuf,
               "nav_zero": np.zeros(N, dtype=np.int64)}
    if "nav_cmd_flip" in A:
        nav_fed["nav_flipped"] = A["nav_cmd_flip"].astype(np.int64)
    for c in CONDITIONS:
        if f"plan_full_{c}" in A:
            p = A[f"plan_full_{c}"].astype(np.float64)
            readouts["plan"][c] = terminal_heading(p)
            readouts["plan_bearing"][c] = end_bearing(p)
            readouts["plan_lateral"][c] = lateral_end(p)
        if f"gstr_{c}" in A:
            g = A[f"gstr_{c}"].astype(np.float64)
            readouts["gstr"][c] = np.arctan2(g[:, 1], g[:, 0])
        if f"sel_bank_{c}" in A:
            readouts["anchor"][c] = terminal_heading(A[f"sel_bank_{c}"].astype(np.float64))
        if f"fan_term_heading_{c}" in A:
            fan[c] = {"heading": A[f"fan_term_heading_{c}"],
                      "keep": A.get(f"reach_keep_{c}")}
    readouts = {k: v for k, v in readouts.items() if v}
    # ---- nav-blind controls -------------------------------------------------- #
    ego = A["ego_t0"].astype(np.float64)                     # (v0, a_long, yaw_rate, curvature)
    zeros = np.zeros(N)
    controls = {"ha0": {"plan": zeros, "plan_bearing": zeros, "plan_lateral": zeros},
                "ha0_ext": kinematic_end_signals(ego[:, 0], ego[:, 1], ego[:, 3])}
    if "ha_controls" in A:
        hc = A["ha_controls"].astype(np.float64)             # [N, n, 2] (a, chan-1)
        units = ((manifest.get("action_units") or {}).get("recorded") or "steer")
        k_ha = (np.tan(hc[:, 0, 1]) / WHEELBASE_M) if units == "steer" else hc[:, 0, 1]
        controls["ha"] = kinematic_end_signals(ego[:, 0], hc[:, 0, 0], k_ha)
    rep = nav_compliance_report(
        readouts=readouts, nav_true=nav_true, nav_valid=nav_valid,
        nav_fed_by_cond=nav_fed, eid=eid, t_now_s=t_now,
        t_start_s=join["t_start_s"], t_end_s=join["t_end_s"], arc_to_turn_m=arc,
        gt_signal={"plan": gt_dpsi, "gstr": gt_bear}, controls=controls,
        fan=fan or None, n_boot=n_boot, seed=seed, tag=tag, min_delta=min_delta)
    rep["time_base_control"] = tb
    rep["join"] = {"labels": labels_path, "n_windows_with_record": int(join["has_record"].sum()),
                   "n_windows_with_commanded_turn": int(np.isfinite(join["t_start_s"]).sum()),
                   "n_gt_6s_end_clamped": int((~gt_full).sum()),
                   "provider_to_raw_frame_offset": raw_off,
                   "t_now_rule": "t_now = (ws + provider_to_raw_frame_offset) * 0.1 s"}
    rep["dump_dir"] = dump_dir
    rep["ckpt"] = (manifest.get("model") or {}).get("ckpt")
    rep["step"] = (manifest.get("model") or {}).get("step")
    rep["ego_state_fed"] = manifest.get("ego_state_fed")
    return rep
