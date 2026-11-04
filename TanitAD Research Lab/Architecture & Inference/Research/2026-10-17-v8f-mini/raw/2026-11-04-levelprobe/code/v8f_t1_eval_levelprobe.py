#!/usr/bin/env python3
"""v8F-mini T1-kin: a SELF-ACTION rollout on the refav1 trunk-probe grid (design: ``DESIGN_T1_V8F.md``).

TIER, in the vocabulary of ``Project Steering/EVAL_DOCTRINE.md`` (PI amendment 2026-09-02): **T1 = self-action OPEN loop.** The
model's kinematic state is its OWN executed motion, rebuilt at every replan; the camera stream is NOT re-rendered (that is
T2, not provisioned). So this tool never says "closed loop" and supports only open-loop trajectory-quality claims, not
"the model drives". Every output carries ``scene``, ``nav``, ``limit`` and ``replan_steps`` beside the numbers, because
those four decide how strict the T1 is (DESIGN section 4).

WHAT IS REPLACED by the model's own executed motion (everything the model reads that is built from the ego track):
``p10f`` (8 kinematic features), ``base`` (kd_x path), ``pose_now / pose_t / pose_s`` (odometry), and -- by default -- the
route countdown in ``nav``. WHAT STAYS LOGGED: ``feats`` / ``frame_ok`` (camera), ``limit`` / ``limit_ok`` (a declared
oracle, D-SPEEDBAND-EGO-ECHO-1), the sample-and-hold ages (a schedule). The model is handed ONLY the keys it reads
(``MODEL_KEYS``); a logged ego row after t0 does not exist in the per-window view (NaN), so it cannot be read.

NO MODEL CHANGE: ``V8FMini.forward_infer`` is called as is; no new input, no new weight.

The scoring is ``taniteval/tools/t1_eval.py::analyze`` (four families, S-curve, lag, paired episode-cluster bootstrap),
fed per-episode dumps in its schema -- not re-implemented.

``--dump-ticks DIR`` (PREREG_V8F_HACC_LAT, carried from the hlat build): one row per (window, tick) of the MODEL's rollout -- O's plan before the post-map (tau), the prior (P), the plan the
model issued, the measured ego speed v0 at the tick, the logged future in the tick's EXECUTED frame (GT, "rejoin the human's path") and the window / episode / clip ids.
``--dump-heads`` (needs ``--dump-ticks``; T1-HEADS) adds, to the SAME rows, what the S head and the T head emitted in the very ``forward_infer`` call that produced the plan -- exactly the
tensors ``v8f_mini_eval.py``'s T0 eval scores (``checkpoint_metrics`` / ``tactical_metrics`` / ``self_consistency``): ``s_ckpt`` (+ its NOW-frame and window-origin-frame forms),
``t_goal_prob``, ``t_lat_logits``, ``t_lon_logits``, ``t_cons_pred``, and the poses that frame them. Both flags default OFF; absent, the tool's output is unchanged bit for bit
(control C-T, and ``tests/test_v8f_t1_heads.py``). The P10 floor has no heads and is never recorded.

``--smooth {off,sg72}`` (SMOOTH-T1, default off): ``sg72`` applies PREREG_V8F_SMOOTH's operator, verbatim, to the model's plan at EVERY replan, before the executor:
``S(plan) = P + SG(plan - P)`` with ``P`` = the prior the model computed in the very ``forward_infer`` call (``out["p10"]``, the P10 at the EXECUTED state) and SG = a
Savitzky-Golay filter along time, per coordinate, window 7, order 2, ``mode="interp"`` (``scipy.signal.savgol_filter``). The smoothed plan is what the executor commits, what ``ol``
(the tick-0 plan) holds and what ``--dump-ticks`` calls ``plan``; with ``--dump-ticks`` the ticks file also carries ``plan_raw`` (the model's plan before S) and ``smooth``.
The P10 floor is never smoothed. Absent the flag (or ``off``) the output is unchanged bit for bit (``tests/test_v8f_t1_smooth.py``).

``--feasible {off,box}`` (KAPPA-T1, H-KAPPA-T1-1, default off): ``box`` projects the plan, AFTER ``--smooth``, onto the envelope of
``tanitad.refs.feasible_decode.project_feasible`` verbatim (box only: ``mu=None``, ``a_max`` = ``A_MAX_MPS2`` 4.0, ``kappa_max`` = ``KAPPA_MAX_1PM`` 0.2, the module's
default margin and stop epsilons, ``clamp_entry=False``) on the origin-prepended plan at the plan's own grid ``dt`` = ``DT`` (0.2 s), at EVERY replan, before the executor.
The projected plan is what the executor commits, what ``ol`` holds and what ``--dump-ticks`` calls ``plan``; the ticks file also carries ``plan_raw`` (the model's plan) and
``plan_s`` (the plan after ``--smooth``, before the projection). The P10 floor is never projected. Absent the flag (or ``off``) the output is unchanged bit for bit
(``tests/test_v8f_t1_feasible.py``).

``--feasible {kbox,abox,kabox}`` (FEAS2-T1, H-FEAS2-T1-1) split ``box`` into its two clamps, same place (after ``--smooth``, every replan, before the executor), same ``dt``, ``mu=None``:
``kbox`` = ``project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, clamp_entry=False)`` (the curvature clamp alone); ``abox`` = ``project_feasible(kappa_max=inf, a_max=A_MAX_MPS2,
clamp_entry=True, v0=the tick's measured ego speed batch["p10f"][:, -1, 0])`` (the accel clamp and the entry-speed bound alone); ``kabox`` = ``kbox`` then ``abox``. ``box`` is unchanged
bit for bit from KAPPA-T1; ``off`` is unchanged bit for bit (``tests/test_v8f_t1_feas2.py``).

``--histswap {off,logged}`` (HISTSWAP, H-HISTSWAP-1; this file is ``v8f_t1_eval_histswap.py``): ``logged`` hands the model, at EVERY tick, its ego-history input from the LOGGED clip arrays (rows <= the tick's j;
every key ``ds.item`` builds from the ego rows -- ``p10f``, ``base``, ``kin3``, ``pose``, ``pose_now`` / ``pose_t`` / ``pose_s`` -- and the causal speed bucket) instead of from the executed ones; the plan is
still committed to the EXECUTED state, and ``nav`` (dead-reckoned from the executed motion), the camera stream, the ages, ``--smooth`` and the P10 floor are untouched. It is an ORACLE diagnostic (the logged
history after t0 is the human's, not deployable). Under it the ticks file's ``v0`` stays the EXECUTED measured speed (the quantity ``feas2`` / ``kappa`` readers define) and ``v0_in`` is the speed the network read;
the S head's ``s_pose`` / ``t_pose`` are the logged poses it read. ``off`` (default) and the flag absent are bit-identical to the FEAS2 tool; it refuses to combine with ``--feasible`` (``tests/test_v8f_t1_histswap.py``).

``--histswap {cv_exec,cv_logged}`` (HISTSWAP-LF, H-HISTSWAP-LF-1; this file is ``v8f_t1_eval_histswap_lf.py``; ``off`` and ``logged`` are the HISTSWAP tool's behaviour bit for bit): the model's ego-history
input at EVERY tick is a CONSTANT-TWIST history -- speed ``v`` and yaw rate ``omega`` held over every earlier row (zero acceleration, no trend), anchored at the EXECUTED pose of the tick -- built through
the programme's one spelling (``yaw_kin_from_track`` / ``ego_channels_native`` / ``kin_features`` / ``kdx_base`` on a synthetic native track) for the same keys ``logged`` swaps (``p10f``, ``base``, ``kin3``,
``pose``, ``pose_now`` / ``pose_t`` / ``pose_s``, the causal speed bucket); ``nav`` (dead-reckoned), the camera stream, the ages, ``--smooth`` and the P10 floor are untouched, and the plan is committed to the
EXECUTED state. ``cv_exec`` (HCV, deployable): ``v`` = the EXECUTED measured speed at the tick. ``cv_logged`` (HLVL, ORACLE, level only): ``v`` = the LOGGED measured speed at the tick (the human's level, no
trend). ``omega`` is the EXECUTED yaw rate in both. Needs ``--limit causal``; refuses ``--feasible``. The ticks file's ``v0`` stays the EXECUTED speed, ``v0_in`` is the speed the network read.

Modes:  run  (rollout + dumps + analysis)   |   analyze  (re-score existing dumps)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
from taniteval import floors_kin as fk  # noqa: E402
from tanitad.refs.refcv6_max_speed import speed_max_bin  # noqa: E402  (the sidecar's own containing-window rule)

DT = D.DT                   # 0.2 s: the cache step
H = D.HORIZON                # 30 steps = 6 s
EPS_M = 1e-3                 # a tangent shorter than 1 mm per 0.2 s interval holds the previous heading (plan_kin_t's convention)
#: the ONLY physical bound the executor applies: no executed 0.2 s step is longer than V_ENV_MS * 0.2 s (60 m/s = 216 km/h, beyond
#: any road speed in the corpus). It exists so that a diverging window stays finite instead of overflowing to NaN and corrupting a
#: whole run; it is COUNTED (``WindowState.clipped``) and reported, never silent. A sane plan never reaches it.
V_ENV_MS = 60.0
#: the allowlist: the model is handed EXACTLY the keys forward_infer reads. ``act`` and ``pose`` are in INFERENCE_INPUT_KEYS but
#: forward_infer never reads them (``act`` only inside ``loss``); dropping them makes a future read a loud KeyError.
NO_READ = frozenset({"act", "pose"})
MODEL_KEYS = frozenset(M.INFERENCE_INPUT_KEYS) - NO_READ
NAV_MODES = ("deadreckon", "logged")
SCENE_MODES = ("logged", "frozen")
LIMIT_MODES = ("logged", "off", "causal", "causal_all")
#: PREREG_V8F_T1KIN_CAUSALBUCKET: ``causal`` reads the max MEASURED ego speed over the last CAUSAL_STEPS + 1 rows [j - 20, j] (4 s; logged
#: up to t0, own executed after), ``causal_all`` over [0, j]; both snapped by ``speed_max_bin``. Rows after j are NaN by WindowState's
#: construction, and the slice ends at j anyway.
CAUSAL_STEPS = 20
CAUSAL_STATS = {}            # per run: ticks, ticks with no finite past, agreement with the logged bucket, 4 x 4 confusion (logged x causal)
SCHEDULES = ("periodic", "fresh")
#: SMOOTH-T1 (PREREG_V8F_SMOOTH, sha256 a2f19e48...): the operator is fixed there and is NOT tuned here -- window 7 steps (1.4 s), polynomial order 2, ``mode="interp"``, along time, per coordinate.
SMOOTH_MODES = ("off", "sg72")
#: KAPPA-T1 (PREREG H-KAPPA-T1-1, gate G-KAPPA outcome (a)): the envelope is feasible_decode's, NOT tuned here.
FEASIBLE_MODES = ("off", "box", "kbox", "abox", "kabox")
#: FEAS2-T1: the header's operator text per mode (``box`` is KAPPA-T1's, verbatim).
FEASIBLE_OPERATORS = {
    "box": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=A_MAX_MPS2, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)",
    "kbox": "tanitad.refs.feasible_decode.project_feasible(origin + plan, dt=DT, a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False)",
    "abox": "tanitad.refs.feasible_decode.project_feasible(origin + plan, v0=p10f[:, -1, 0], dt=DT, a_max=A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)",
    "kabox": "abox(kbox(plan)): kbox = project_feasible(a_max=inf, kappa_max=KAPPA_MAX_1PM, mu=None, clamp_entry=False), then abox = project_feasible(v0=p10f[:, -1, 0], a_max=A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)",
}
SG_WINDOW, SG_ORDER = 7, 2
#: HISTSWAP (PREREG_HISTSWAP_FROZEN.md): ``logged`` = the model's ego-history input is the LOGGED history at the tick; ``off`` = the executed one (FEAS2's behaviour, bit for bit).
HISTSWAP_MODES = ("off", "logged", "cv_exec", "cv_logged", "cv_t0", "cv_pred")
#: LEVELPROBE (PREREG_LEVELPROBE.md): a per-window CONSTANT level held at every tick (the speed is not re-read from the executed / logged track at the tick).
LEVEL_MODES = ("cv_t0", "cv_pred")
#: HISTSWAP-LF (PREREG_HISTSWAP_LF_FROZEN.md): a constant-twist history at the executed (HCV) or the logged (HLVL) current speed.
CV_MODES = ("cv_exec", "cv_logged", "cv_t0", "cv_pred")
#: rows [j - CV_ROWS, j] of the item builder's view are rebuilt: ``ds.item`` reads j - (hmax - 1) .. j, j - a_s (a_s <= 6) and the causal bucket j - CAUSAL_STEPS (20) .. j; ``cv_view`` asserts the bound.
CV_ROWS = 32


def hist_mode_of(x) -> str:
    """The history mode of a ``hist_swap`` argument: False / None / "off" -> off; True / "logged" -> logged (HISTSWAP's bool contract); "cv_exec" / "cv_logged" -> itself; anything else raises."""
    if x is None or x is False or x == "off":
        return "off"
    if x is True or x == "logged":
        return "logged"
    if x in CV_MODES:
        return str(x)
    raise ValueError(f"hist_swap must be one of {HISTSWAP_MODES} (or a bool), got {x!r}")
TIER_DOCTRINE = ("EVAL_DOCTRINE.md, PI amendment 2026-09-02: T0 and T1 are both OPEN LOOP; T2 is the only closed loop and is not "
                 "provisioned. T1 here = the ego state is the model's OWN executed motion; the camera stream is logged. "
                 "It supports open-loop trajectory-quality claims only, never 'drives'.")


# ------------------------------------------------------------------------------------------------ geometry
def plan_to_world(plan, pose) -> np.ndarray:
    """``plan`` [N, 2] in the ego frame at ``pose`` = (x, y, yaw, ...) -> world [N, 2]."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    plan = np.asarray(plan, np.float64)
    return np.stack([pose[0] + c * plan[:, 0] - s * plan[:, 1], pose[1] + s * plan[:, 0] + c * plan[:, 1]], 1)


def hermite_frames(k_prev, knots, m: int, psi0: float):
    """Fill the 10 Hz raw frames of ``m`` executed cache steps from the plan's 0.2 s knots (DESIGN section 3, step 5).

    ``k_prev`` [2] = the previous executed knot (step j-1); ``knots`` [m+1.., 2] with ``knots[0]`` = the current knot (step j) and
    ``knots[i]`` = the plan's knot j+i. Cubic Hermite with Catmull-Rom tangents (a backward difference at the plan's last knot).
    Returns ``(xy [2m, 2], psi [2m], v [2m])`` for raw frames 2j+1 .. 2j+2m: the odd frame of interval i is the Hermite midpoint, the
    even frame the knot itself. Heading = tangent angle (held while the tangent is < ``EPS_M``); speed = |tangent| / 0.2 s."""
    K = np.vstack([np.asarray(k_prev, np.float64)[None], np.asarray(knots, np.float64)])      # K[i] = knot j+i-1
    nk = len(K)
    if nk < m + 2:
        raise ValueError(f"need at least m+1={m + 1} knots, got {nk - 1}")

    def tang(i):
        return 0.5 * (K[i + 1] - K[i - 1]) if i + 1 < nk else K[i] - K[i - 1]

    xy, psi, v = [], [], []
    cur = float(psi0)
    for i in range(1, m + 1):                                                    # interval K[i] -> K[i + 1]
        t0, t1 = tang(i), tang(i + 1)
        mid = 0.5 * (K[i] + K[i + 1]) + (t0 - t1) / 8.0
        dmid = 1.5 * (K[i + 1] - K[i]) - 0.25 * (t0 + t1)
        for pos, d in ((mid, dmid), (K[i + 1], t1)):
            nrm = float(np.hypot(d[0], d[1]))
            if nrm >= EPS_M:
                cur = math.atan2(d[1], d[0])
            xy.append(pos)
            psi.append(cur)
            v.append(nrm / DT)
    return np.asarray(xy), np.asarray(psi), np.asarray(v)


def envelope(plan):
    """Bound the plan's 0.2 s steps by ``V_ENV_MS * DT`` (direction kept) and zero a non-finite step. Returns ``(plan, hit)``.
    The origin is prepended, so step 0 is |plan[0]| itself. A plan inside the envelope is returned UNCHANGED (bit-identical)."""
    plan = np.asarray(plan, np.float64)
    steps = np.diff(np.vstack([np.zeros((1, 2)), plan]), axis=0)
    bad = ~np.isfinite(steps).all(1)
    nrm = np.linalg.norm(np.where(bad[:, None], 0.0, steps), axis=1)
    over = nrm > V_ENV_MS * DT
    if not (bad.any() or over.any()):
        return plan, False
    steps = np.where(bad[:, None], 0.0, steps)
    steps = steps * np.where(over, V_ENV_MS * DT / np.maximum(nrm, 1e-12), 1.0)[:, None]
    return np.cumsum(steps, axis=0), True


def exec_in_ego(track, t0: int) -> np.ndarray:
    """The executed cache-step knots t0+1 .. t0+30 in the t0 ego frame, by the SAME function that builds the GT (``gt_waypoints``)."""
    return fk.gt_waypoints(np.asarray(track, np.float64)[0::2], np.array([t0]), np.arange(1, H + 1))[0]


def ages_for(step: int, schedule: str, k_t: int = 3, k_s: int = 6):
    """Sample-and-hold ages (cache steps) at ``step`` = steps since t0. periodic: T ticks every ``k_t`` steps, S every ``k_s``
    (defaults 3 / 6: inside the trained support max_age_t=2, max_age_s=6 and nested as the trainer requires a_s >= a_t)."""
    if schedule == "fresh":
        return 0, 0
    if k_s % k_t:
        raise ValueError("k_s must be a multiple of k_t (S ticks no later than T)")
    return step % k_t, step % k_s


# ------------------------------------------------------------------------------------------------ the per-window state
def cv_motion_track(pose_j, v: float, omega: float, n_back: int) -> np.ndarray:
    """``[2 * n_back + 1, 4]`` native (0.1 s) rows (x, y, yaw, v) of the CONSTANT-TWIST history that ENDS at ``pose_j`` = (x, y, yaw): speed ``v`` and yaw rate ``omega`` (rad/s) held on every earlier
    row (the exact circular arc; a straight line when |omega| < 1e-9). The last row is ``pose_j`` with ``v``; the earlier rows are the arc travelled backwards."""
    n = 2 * int(n_back) + 1
    x0, y0, p0 = float(pose_j[0]), float(pose_j[1]), float(pose_j[2])
    tau = (np.arange(n, dtype=np.float64) - (n - 1)) * D.DT_NATIVE                  # seconds before the tick, <= 0
    psi = p0 + omega * tau
    if abs(omega) < 1e-9:
        x, y = x0 + v * tau * math.cos(p0), y0 + v * tau * math.sin(p0)
    else:
        x, y = x0 + (v / omega) * (np.sin(psi) - math.sin(p0)), y0 + (v / omega) * (math.cos(p0) - np.cos(psi))
    return np.stack([x, y, psi, np.full(n, float(v))], 1)


def channels_from_track(sl) -> dict:
    """The rows ``WindowState._finish`` writes for steps 1 .. n of a native slice ``sl`` of ``2 n + 1`` rows, by the SAME calls: ``kin3``, ``ch1`` (omega), ``p10f``, ``base``, ``pose`` (4 columns), ``a``, ``v``.
    (Re-spelled here so that ``_finish`` stays byte-identical to the HISTSWAP tool; ``tests/test_v8f_t1_histswap_lf.py`` pins the two to equal exactly on a real executed slice.)"""
    sl = np.asarray(sl, np.float64)
    kin, om = D.yaw_kin_from_track(sl)
    ch = D.ego_channels_native(sl)[0::2]
    n = len(kin) - 1
    memo = {}
    base = []
    for s in range(1, n + 1):
        key = kin[s].tobytes()
        if key not in memo:
            memo[key] = D.kdx_base(kin[s:s + 1], D.KIN_UNITS["yaw"])[0]
        base.append(memo[key])
    return {"kin3": kin[1:], "ch1": np.nan_to_num(om[1:], nan=0.0), "p10f": np.stack([fk.kin_features(kin[s:s + 1])[0].astype(np.float32) for s in range(1, n + 1)]),
            "base": np.stack(base), "pose": sl[0::2][1:], "a": ch[1:, 1], "v": sl[0::2][1:, 3]}


class WindowState:
    """One window's view of its clip, with the ego arrays logged UP TO t0 and own-executed after.

    The constructor copies ONLY rows <= t0 of the logged ego arrays and of the native track (the future rows are NaN), so this
    object cannot leak the logged ego after t0 whatever the item builder does. ``d`` is a ClipStore-shaped dict that
    ``V8FMiniWindows.item`` reads unchanged. The logged track is passed to ``commit_logged`` only by the CONTROL executor."""

    EGO = ("pose", "kin3", "p10f", "base", "a", "ch1", "v")

    def __init__(self, base: dict, track, nav_rows, ci: int, t0: int, nav_mode: str = "deadreckon", hist_swap=False):
        if nav_mode not in NAV_MODES:
            raise ValueError(f"nav_mode must be one of {NAV_MODES}")
        T = len(base["pose"])
        if not bool(base["gt_ok"][t0]):
            raise ValueError("window has no complete 6 s future (gt_ok False)")
        self.ci, self.t0, self.T, self.j, self.nav_mode = int(ci), int(t0), T, int(t0), nav_mode
        tr = np.full(np.asarray(track).shape, np.nan, np.float64)
        tr[:2 * t0 + 1] = np.asarray(track, np.float64)[:2 * t0 + 1]
        self.track = tr
        d = dict(base)                                                   # shallow: feats / gt / s_h stay shared and logged
        for k in self.EGO:
            arr = np.array(base[k], copy=True)
            arr[t0 + 1:] = np.nan
            d[k] = arr
        self.nav = np.array(nav_rows, np.float32, copy=True)
        self.nav0 = self.nav[t0].copy()
        if nav_mode == "deadreckon":
            self.nav[t0 + 1:] = np.nan
        self.arc = {t0: 0.0}
        self.d = d
        self.scene0 = None
        self.clipped = 0                  # ticks whose plan had a step beyond the envelope (or a non-finite value)
        self.hist_mode = hist_mode_of(hist_swap)
        self.hist_swap = self.hist_mode != "off"
        self.level = None                                                 # LEVELPROBE: the per-window constant level of cv_pred (set by make_states); cv_t0 reads the logged clip at t0
        self.base_log = base                                              # the LOGGED clip (read-only, never written): the source of the logged speed level (cv_logged) and of ``logged``
        self.d_in = base if self.hist_mode == "logged" else d             # the dict the ITEM BUILDER reads: the executed view ``d`` (off), the LOGGED clip ``base`` (logged), or, under cv_*, the per-tick :meth:`cv_view`

    # -- committing executed motion ------------------------------------------------------------------------------
    def commit_plan(self, plan_ego, m: int):
        """Execute the first ``m`` knots of ``plan_ego`` [30, 2] (ego frame at NOW), perfect tracking."""
        j = self.j
        pose = self.d["pose"][j]
        plan_ego, hit = envelope(plan_ego)
        self.clipped += int(hit)
        W = plan_to_world(plan_ego, pose)
        knots = np.vstack([pose[None, :2], W])
        xy, psi, v = hermite_frames(self.track[2 * (j - 1), :2], knots, m, float(pose[2]))
        sl = slice(2 * j + 1, 2 * (j + m) + 1)
        self.track[sl, 0:2], self.track[sl, 2], self.track[sl, 3] = xy, psi, v
        self._finish(j, j + m)

    def commit_logged(self, logged_track, m: int):
        """CONTROL executor: the 'own' motion IS the logged motion (DESIGN C1). Never used by a result."""
        j = self.j
        sl = slice(2 * j + 1, 2 * (j + m) + 1)
        self.track[sl] = np.asarray(logged_track, np.float64)[sl]
        self._finish(j, j + m)

    def _finish(self, j_old: int, j_new: int):
        sl = self.track[2 * j_old: 2 * j_new + 1]
        kin, om = D.yaw_kin_from_track(sl)                              # the ONE spelling, on the appended slice
        ch = D.ego_channels_native(sl)[0::2]
        d = self.d
        for s in range(1, j_new - j_old + 1):
            j = j_old + s
            d["kin3"][j] = kin[s]
            d["ch1"][j] = np.nan_to_num(om[s], nan=0.0)
            d["p10f"][j] = fk.kin_features(kin[s:s + 1])[0].astype(np.float32)
            d["base"][j] = D.kdx_base(kin[s:s + 1], D.KIN_UNITS["yaw"])[0]
            d["pose"][j] = self.track[2 * j]
            d["a"][j] = ch[s, 1]
            d["v"][j] = self.track[2 * j, 3]
            self.arc[j] = self.arc[j - 1] + float(np.linalg.norm(d["pose"][j, :2] - d["pose"][j - 1, :2]))
            if self.nav_mode == "deadreckon":
                row = self.nav0.copy()
                if row[4] > 0:                                           # route progress from OWN motion; token / valid held from t0
                    row[3] = np.clip(row[3] - self.arc[j] / 100.0, 0.0, 3.0)
                self.nav[j] = row
        self.j = j_new

    def cv_view(self, limit: str = "causal") -> dict:
        """HISTSWAP-LF: the dict the item builder reads at this tick under ``cv_exec`` / ``cv_logged`` -- a copy of the EXECUTED view ``d`` whose rows [j - CV_ROWS, j] of every ego key are the CONSTANT-TWIST
        history at speed ``v`` (cv_exec: the executed ``p10f[j][0]``; cv_logged: the LOGGED ``p10f[j][0]``) and yaw rate ``omega`` = the executed ``ch1[j]``, anchored at the executed pose, built by
        :func:`channels_from_track` on :func:`cv_motion_track`. Nothing is written to ``d`` or to the logged clip."""
        if self.hist_mode not in CV_MODES:
            raise RuntimeError("cv_view is defined for cv_exec / cv_logged only")
        if limit == "causal_all":
            raise ValueError("REFUSED: cv_exec / cv_logged are defined with --limit causal (the bucket reads rows [j - 20, j])")
        j, d = self.j, self.d
        if self.hist_mode == "cv_t0":
            v = float(self.base_log["p10f"][self.t0][0])                  # the LOGGED measured speed at t0, held constant
        elif self.hist_mode == "cv_pred":
            if self.level is None or not (math.isfinite(float(self.level)) and float(self.level) >= 0.0):
                raise ValueError(f"cv_pred: window (clip_index {self.ci}, ws {self.t0}) has no valid level ({self.level!r})")
            v = float(self.level)
        else:
            src = d if self.hist_mode == "cv_exec" else self.base_log
            v = float(src["p10f"][j][0])
        om = float(np.nan_to_num(d["ch1"][j], nan=0.0))
        pj = np.asarray(d["pose"][j], np.float64)
        if not (math.isfinite(v) and bool(np.isfinite(pj[:3]).all())):
            raise ValueError(f"{self.hist_mode}: the current speed / pose is not finite at step {j} (v={v!r})")
        lo = max(1, j - CV_ROWS)
        assert CV_ROWS >= max(CAUSAL_STEPS, 9, 6), "CV_ROWS must cover every row the item builder reads"
        ch = channels_from_track(cv_motion_track(pj, v, om, j - (lo - 1)))
        out = dict(d)
        for k in self.EGO:
            out[k] = np.array(d[k], copy=True)
        out["kin3"][lo:j + 1], out["ch1"][lo:j + 1], out["p10f"][lo:j + 1], out["base"][lo:j + 1] = ch["kin3"], ch["ch1"], ch["p10f"], ch["base"]
        out["pose"][lo:j + 1], out["a"][lo:j + 1] = ch["pose"], ch["a"]
        out["pose"][j] = d["pose"][j]                                     # the anchor IS the executed pose (the cameras, the nav and every target live in this frame)
        out["v"][max(0, j - CV_ROWS):j + 1] = v                           # the causal bucket reads d["v"][max(0, j - 20): j + 1]
        return out

    def executed_ego(self) -> np.ndarray:
        return exec_in_ego(self.track, self.t0)


class _OneClipStore:
    """A ClipStore stand-in that serves the dict of whichever window is being built."""

    def __init__(self):
        self.d = None

    def get(self, clip):
        return self.d


def build_item(ds, proxy, st: WindowState, tick: int, a_t: int, a_s: int, scene: str, limit: str, stats: dict | None = None) -> dict:
    if st.hist_mode in CV_MODES:                                         # HISTSWAP-LF: the view is rebuilt at every tick from the tick's executed (cv_exec) / logged (cv_logged) speed
        st.d_in = st.cv_view(limit)
    proxy.d = st.d_in
    it = ds.item(st.ci, st.j, a_t, a_s)
    idc = np.clip(np.arange(st.j - ds.hmax + 1, st.j + 1), 0, st.T - 1)
    it["nav"] = st.nav[idc].astype(np.float32)
    if scene == "frozen":
        if tick == 0:
            st.scene0 = (it["feats"], it["frame_ok"])
        it["feats"], it["frame_ok"] = st.scene0
    if limit == "off":
        it["limit"], it["limit_ok"] = np.zeros(4, np.float32), False
    elif limit in ("causal", "causal_all"):
        logged = np.asarray(it["limit"], np.float32).copy() if bool(it["limit_ok"]) else None
        it["limit"], it["limit_ok"] = causal_bucket(st.d_in["v"], st.j, CAUSAL_STEPS if limit == "causal" else None)
        if stats is not None and st.d_in is not st.d:                    # HISTSWAP only: how often the swapped bucket differs from the executed-history bucket
            ex, ex_ok = causal_bucket(st.d["v"], st.j, CAUSAL_STEPS if limit == "causal" else None)
            stats["swap_ticks"] = stats.get("swap_ticks", 0) + 1
            stats["swap_bucket_differs"] = stats.get("swap_bucket_differs", 0) + int(bool(ex_ok) != bool(it["limit_ok"]) or (bool(ex_ok) and int(ex.argmax()) != int(it["limit"].argmax())))
        if stats is not None:
            stats["ticks"] = stats.get("ticks", 0) + 1
            if not it["limit_ok"]:
                stats["no_finite_past"] = stats.get("no_finite_past", 0) + 1
            elif logged is not None:
                cm = stats.setdefault("confusion_logged_x_causal", [[0] * 4 for _ in range(4)])
                cm[int(logged.argmax())][int(it["limit"].argmax())] += 1
    return it


def causal_bucket(v, j: int, steps: int | None):
    """The PREREG_V8F_T1KIN_CAUSALBUCKET bucket at step ``j``: ``speed_max_bin`` of the max finite speed over rows [j - steps, j] (or
    [0, j] when ``steps`` is None). Reads NOTHING after row j. No finite row -> (zeros, False), the ``--limit off`` format."""
    lo = 0 if steps is None else max(0, j - steps)
    w = np.asarray(v[lo:j + 1], np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return np.zeros(4, np.float32), False
    i, _ = speed_max_bin(float(max(w.max(), 0.0)))
    oh = np.zeros(4, np.float32)
    oh[i] = 1.0
    return oh, True


# ------------------------------------------------------------------------------------------------ planners
def smooth_plan(plan, prior, mode: str = "off"):
    """PREREG_V8F_SMOOTH's operator ``S(plan) = P + SG(plan - P)`` on a batch of plans [B, 30, 2] and their prior ``P`` [B, 30, 2] (the SAME arithmetic as
    ``v8f_smooth_read.smooth``: ``base + savgol_filter(pred - base, 7, 2, axis=1, mode="interp")``, evaluated in float64). ``off`` returns ``plan`` itself (the same object)."""
    if mode == "off":
        return plan
    if mode != "sg72":
        raise ValueError(f"--smooth must be one of {SMOOTH_MODES}, got {mode!r}")
    from scipy.signal import savgol_filter
    plan64, p64 = np.asarray(plan, np.float64), np.asarray(prior, np.float64)
    if plan64.shape != p64.shape or plan64.ndim != 3 or plan64.shape[1] < SG_WINDOW:
        raise ValueError(f"smooth_plan: plan {plan64.shape} and prior {p64.shape} must match and be [B, T>={SG_WINDOW}, 2]")
    return (p64 + savgol_filter(plan64 - p64, SG_WINDOW, SG_ORDER, axis=1, mode="interp")).astype(np.asarray(plan).dtype)


def feasible_plan(plan, mode: str = "off", v0=None):
    """The feasible-decode projection on a batch of plans [B, T, 2] (ego frame, knots 1..T; the origin is implied): ``project_feasible`` of ``tanitad.refs.feasible_decode`` on the
    origin-prepended path, float64, ``dt=DT``, ``mu=None``, returned in the plan's dtype without the origin. ``box`` = both clamps (KAPPA-T1), ``clamp_entry=False``; ``kbox`` = the
    curvature clamp alone (``a_max=inf``); ``abox`` = the accel clamp alone (``kappa_max=inf``) WITH the entry-speed bound ``clamp_entry=True`` (needs ``v0`` [B], the tick's measured ego
    speed); ``kabox`` = ``kbox`` then ``abox``. ``off`` returns ``plan`` itself (the same object)."""
    if mode == "off":
        return plan
    if mode not in FEASIBLE_MODES:
        raise ValueError(f"--feasible must be one of {FEASIBLE_MODES}, got {mode!r}")
    import torch
    from tanitad.refs import feasible_decode as FD
    p64 = np.asarray(plan, np.float64)
    if p64.ndim != 3 or p64.shape[-1] != 2 or p64.shape[1] < 3:
        raise ValueError(f"feasible_plan: plan {p64.shape} must be [B, T>=3, 2]")
    inf = float("inf")
    full = torch.from_numpy(np.concatenate([np.zeros_like(p64[:, :1]), p64], 1))
    if mode in ("abox", "kabox"):
        if v0 is None or np.asarray(v0).shape != (p64.shape[0],):
            raise ValueError(f"feasible_plan: mode {mode!r} needs v0 of shape [B]={p64.shape[0]}, got {None if v0 is None else np.asarray(v0).shape}")
        v0t = torch.from_numpy(np.asarray(v0, np.float64))
    if mode == "box":
        out = FD.project_feasible(full, dt=DT, a_max=FD.A_MAX_MPS2, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)
    elif mode in ("kbox", "kabox"):
        out = FD.project_feasible(full, dt=DT, a_max=inf, kappa_max=FD.KAPPA_MAX_1PM, mu=None, clamp_entry=False)
    if mode in ("abox", "kabox"):
        out = FD.project_feasible(full if mode == "abox" else out, v0=v0t, dt=DT, a_max=FD.A_MAX_MPS2, kappa_max=inf, mu=None, clamp_entry=True)
    return out[:, 1:].numpy().astype(np.asarray(plan).dtype)


#: the S / T head outputs ``--dump-heads`` keeps from ``forward_infer`` (the names ``v8f_mini_eval.cmd_run`` collects into ``outs``); float32, as the T0 eval stores them
HEAD_KEYS = ("ckpt", "goal_prob", "lat_logits", "lon_logits", "cons_pred")


def model_planner(model, device, keep_aux: bool = False, keep_heads: bool = False, smooth: str = "off", feasible: str = "off"):
    """The model's plan. ``keep_aux`` (``--dump-ticks``) additionally keeps, per call, O's plan before the post-map (``tau``), the prior (``P``) and the gate flags;
    ``keep_heads`` (``--dump-heads``) keeps :data:`HEAD_KEYS` of the SAME ``forward_infer`` call (never a second call). Neither changes any output.
    ``smooth`` (``--smooth``): ``sg72`` returns :func:`smooth_plan` of the model's plan over its own prior ``out["p10"]`` (the pre-S plan is kept as ``plan.plan_raw``); ``off`` returns the plan untouched."""
    import torch

    def plan(batch):
        with torch.no_grad():
            out = model.forward_infer(batch)
        plan.keys_read = set(out["_keys_read"])
        if keep_aux:
            plan.aux = {"tau": out.get("traj30_o", out["traj30"]).double().cpu().numpy(), "P": out["p10"].double().cpu().numpy(),
                        "fired": None if "hacc_gate_fired" not in out else out["hacc_gate_fired"].cpu().numpy()}
        if keep_heads:
            plan.heads = {k: out[k].float().cpu().numpy() for k in HEAD_KEYS}
        raw = out["traj30"].float().cpu().numpy()
        if smooth == "off" and feasible == "off":
            return raw
        plan.plan_raw = raw
        if feasible != "off":                                            # KAPPA-T1: the projection AFTER S (S is the identity when --smooth off)
            plan.plan_s = smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
            v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()          # FEAS2-T1: the measured ego speed the prior reads (== TickRecorder's ``v0``); used by abox / kabox only
            return feasible_plan(plan.plan_s, feasible, v0=v0)
        return smooth_plan(raw, out["p10"].float().cpu().numpy(), smooth)
    plan.keys_read = set()
    plan.aux = None
    plan.heads = None
    plan.plan_raw = None
    plan.plan_s = None
    return plan


def floor_planner(model, device):
    """The P10 floor driven the same way: the prior alone, the very call ``run_o`` makes (``v8f_mini.py:579``)."""
    import torch

    def plan(batch):
        with torch.no_grad():
            return model.prior(batch["p10f"][:, -1].float(), batch["base"].float()).float().cpu().numpy()
    return plan


# ------------------------------------------------------------------------------------------------ the tick recorder (--dump-ticks, --dump-heads)
def gt_in_frame(pose_grid, pose_now, j: int):
    """The logged future ``pose_grid[j+1 .. j+30]`` in the frame of ``pose_now`` = (x, y, yaw, ...): the SAME arithmetic as ``floors_kin.gt_waypoints`` / ``v8f_mini_data.gt_future`` (rotation by
    ``-yaw``), with the origin pose replaced by the EXECUTED pose. At tick 0 the executed pose IS the logged pose, so this equals the store's ``gt[t0]`` (control C-T2). ``ok`` False (NaN GT) when the logged clip
    ends before ``j + 30`` (the same ``t <= T - 31`` rule as ``gt_future``)."""
    pg = np.asarray(pose_grid, np.float64)
    if j + H > len(pg) - 1:
        return np.full((H, 2), np.nan), False
    d = pg[j + 1:j + 1 + H, :2] - np.asarray(pose_now, np.float64)[None, :2]
    c, s = np.cos(-pose_now[2]), np.sin(-pose_now[2])
    return np.stack([d[:, 0] * c - d[:, 1] * s, d[:, 0] * s + d[:, 1] * c], axis=-1), True


class TickRecorder:
    """The ``hook`` of :func:`rollout` that keeps the tick samples. ``hook(n, batch, plan, states)`` runs BEFORE the plan is committed, so ``st.j`` and ``st.d['pose'][st.j]`` are the tick's executed state.

    ``heads=True`` (``--dump-heads``) adds the S / T head outputs of the tick's own ``forward_infer`` call (``planner.heads``) under :attr:`HEAD_FIELDS`:
      * ``s_ckpt``      [N, 3, 3] f32  S's checkpoints (x, y, psi) at the 30 / 50 / 80 m arcs, RAW, in S's TICK frame = the executed pose at step ``j - age_s`` (``s_pose``); what ``forward_infer`` emits
      * ``s_ckpt_now``  [N, 3, 3] f32  the same in the NOW frame (the executed pose at the tick, ``pose``[:3]) by ``transform_ckpt`` -- the frame ``v8f_mini_eval._ckpt_to_now`` scores in
      * ``s_ckpt_w0``   [N, 3, 3] f32  the same in the WINDOW-ORIGIN frame (the logged pose at ``t0`` = ``pose_w0``): the one frame every tick of a window shares, whatever the executed path did
      * ``s_pose``      [N, 3]    f32  the world pose (x, y, yaw) of S's tick frame, as the batch gave it to the model
      * ``t_goal_prob`` [N, 22]   f32  T's goal tokens (sigmoid), ``v8f_mini.GOAL22`` order     * ``t_lat_logits`` [N, 5] f32  * ``t_lon_logits`` [N, 7] f32  * ``t_cons_pred`` [N, C] f32 (de-standardised)
      * ``t_pose``      [N, 3]    f32  the world pose of T's tick frame (``j - age_t``)
      * ``pose_w0``     [N, 3]    f64  the window-origin pose (x, y, yaw): ``st.d['pose'][t0][:3]`` (rows <= t0 are logged, never overwritten)
    T and S are the model's SAMPLE-AND-HOLD heads: their outputs at tick ``j`` belong to the frames ``j - age_t`` / ``j - age_s`` (``age_t`` / ``age_s`` are in the dump)."""

    FIELDS = ("tau", "P", "plan", "gt", "gt_ok", "v0", "pose", "fired", "win", "ci", "t0", "tick", "j", "age_t", "age_s")
    RAW_HEAD_FIELDS = ("s_ckpt", "s_pose", "t_goal_prob", "t_lat_logits", "t_lon_logits", "t_cons_pred", "t_pose", "pose_w0")      # recorded per tick
    HEAD_FIELDS = RAW_HEAD_FIELDS + ("s_ckpt_now", "s_ckpt_w0")                                                                      # + the two frames derived at save()

    def __init__(self, planner, store, J, heads: bool = False, smooth: str = "off", feasible: str = "off", histswap: str = "off"):
        self.planner, self.store, self.J, self.heads, self.smooth, self.feasible, self.histswap = planner, store, J, bool(heads), smooth, feasible, histswap
        self.rows = {k: [] for k in self.FIELDS + (self.RAW_HEAD_FIELDS if self.heads else ()) + (("plan_raw",) if (smooth != "off" or feasible != "off") else ())
                     + (("plan_s",) if feasible != "off" else ()) + (("v0_in",) if histswap != "off" else ())}
        self.clip = []
        self.base = 0                                                    # the index of the chunk's first window in the run's window list

    def __call__(self, n, batch, plan, states):
        aux = self.planner.aux
        if aux is None:
            raise RuntimeError("--dump-ticks needs a planner built with keep_aux=True")
        hd = self.planner.heads
        if self.heads and hd is None:
            raise RuntimeError("--dump-heads needs a planner built with keep_heads=True")
        v0 = batch["p10f"][:, -1, 0].double().cpu().numpy()             # the measured ego speed the prior (and the gate) read: exactly p10f[:, -1, 0]
        v0_in = v0
        if self.histswap != "off":                                       # HISTSWAP: v0 stays the EXECUTED measured speed (what Y's v0 is); v0_in is the speed the network read
            v0 = np.array([float(np.nan_to_num(st.d["p10f"][st.j][0], nan=0.0)) for st in states], np.float64)
        if self.heads:
            pose_s, pose_t = batch["pose_s"].float().cpu().numpy(), batch["pose_t"].float().cpu().numpy()
        for b, st in enumerate(states):
            clip = str(self.J["clip"][st.ci])
            pose_now = np.array(st.d["pose"][st.j], np.float64)
            gt, ok = gt_in_frame(self.store.get(clip)["pose"], pose_now, st.j)
            r = self.rows
            r["tau"].append(aux["tau"][b]); r["P"].append(aux["P"][b]); r["plan"].append(np.asarray(plan[b], np.float64))
            if self.smooth != "off" or self.feasible != "off":          # only then: the pre-S plan beside the issued (smoothed) one
                r["plan_raw"].append(np.asarray(self.planner.plan_raw[b], np.float64))
            if self.feasible != "off":                                   # KAPPA-T1: the plan after S, before the projection
                r["plan_s"].append(np.asarray(self.planner.plan_s[b], np.float64))
            r["gt"].append(gt); r["gt_ok"].append(ok); r["v0"].append(v0[b]); r["pose"].append(pose_now[:4])
            if self.histswap != "off":
                r["v0_in"].append(v0_in[b])
            r["fired"].append(False if aux["fired"] is None else bool(aux["fired"][b]))
            r["win"].append(self.base + b); r["ci"].append(st.ci); r["t0"].append(st.t0); r["tick"].append(n); r["j"].append(st.j)
            r["age_t"].append(int(batch["a_t"][b])); r["age_s"].append(int(batch["a_s"][b]))
            if self.heads:
                r["s_ckpt"].append(hd["ckpt"][b])
                r["s_pose"].append(pose_s[b])
                r["t_goal_prob"].append(hd["goal_prob"][b])
                r["t_lat_logits"].append(hd["lat_logits"][b])
                r["t_lon_logits"].append(hd["lon_logits"][b])
                r["t_cons_pred"].append(hd["cons_pred"][b])
                r["t_pose"].append(pose_t[b])
                r["pose_w0"].append(np.array(st.d["pose"][st.t0][:3], np.float64))
            self.clip.append(clip)

    def _ckpt_frames(self, out):
        """``s_ckpt_now`` / ``s_ckpt_w0``: the raw checkpoints re-expressed by the model's own ``transform_ckpt`` (float32, as ``_forward`` does for O)."""
        import torch
        ck, ps = torch.as_tensor(out["s_ckpt"]), torch.as_tensor(out["s_pose"])
        now = torch.as_tensor(out["pose"][:, :3].astype(np.float32))
        w0 = torch.as_tensor(out["pose_w0"].astype(np.float32))
        return M.transform_ckpt(ck, ps, now).numpy(), M.transform_ckpt(ck, ps, w0).numpy()

    def save(self, path, extra: dict | None = None):
        out = {k: np.asarray(v) for k, v in self.rows.items()}
        if self.heads and len(out["s_ckpt"]):
            out["s_ckpt_now"], out["s_ckpt_w0"] = self._ckpt_frames(out)
        out["clip"] = np.asarray(self.clip)
        for k, v in (extra or {}).items():
            out[k] = np.asarray(v)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **out)
        return out


# ------------------------------------------------------------------------------------------------ the rollout
def rollout(planner, states, ds, proxy, *, m: int = 1, schedule: str = "periodic", k_t: int = 3, k_s: int = 6,
            scene: str = "logged", limit: str = "logged", device="cpu", executor: str = "plan", logged_tracks=None, hook=None):
    """Roll every window of ``states`` for ``H // m`` ticks. Returns ``plan0`` [B, 30, 2] (the tick-0 plans) and the list of
    per-window executed paths in the t0 ego frame. ``hook(n, batch, plan, states)`` is the test seam."""
    import torch
    if H % m:
        raise ValueError(f"--replan-steps {m} must divide {H}")
    plan0 = None
    for n in range(H // m):
        step = n * m
        a_t, a_s = ages_for(step, schedule, k_t, k_s)
        stats = CAUSAL_STATS if hasattr(planner, "keys_read") else None          # the MODEL's ticks only (the floor reads no bucket)
        items = [build_item(ds, proxy, st, n, a_t, a_s, scene, limit, stats) for st in states]
        full = D.collate(items)
        batch = {k: full[k].to(device) for k in MODEL_KEYS}
        plan = planner(batch)
        if plan0 is None:
            plan0 = plan.copy()
        if hook is not None:
            hook(n, batch, plan, states)
        for st, pl in zip(states, plan):
            if executor == "logged":
                st.commit_logged(logged_tracks[st.ci], m)
            else:
                st.commit_plan(pl.astype(np.float64), m)
    return plan0, [st.executed_ego() for st in states]


def make_states(store, tracks, J, wins, nav_mode, hist_swap=False, levels=None):
    out = []
    for ci, t0, _, _ in wins:
        clip = str(J["clip"][ci])
        base = store.get(clip)
        o, n = int(J["off"][ci]), int(J["n"][ci])
        st = WindowState(base, tracks[clip], J["nav"][o:o + n], ci, t0, nav_mode, hist_swap)
        if st.hist_mode == "cv_pred":
            if levels is None or (int(ci), int(t0)) not in levels:
                raise ValueError(f"REFUSED: cv_pred has no level for window (clip_index {ci}, ws {t0})")
            st.level = float(levels[(int(ci), int(t0))])
        out.append(st)
    return out


def load_levels(path):
    """``--level-file`` npz -> {(clip_index, ws): level}.  Refuses duplicate windows and non-finite / negative levels."""
    z = np.load(path, allow_pickle=False)
    ci, ws, lv = np.asarray(z["clip_index"], np.int64), np.asarray(z["ws"], np.int64), np.asarray(z["level"], np.float64)
    if not (len(ci) == len(ws) == len(lv)) or not np.isfinite(lv).all() or (lv < 0).any():
        raise SystemExit(f"REFUSED: {path}: lengths differ, or a level is non-finite or negative")
    out = {(int(a), int(b)): float(c) for a, b, c in zip(ci, ws, lv)}
    if len(out) != len(lv):
        raise SystemExit(f"REFUSED: {path}: duplicate (clip_index, ws) rows")
    return out


# ------------------------------------------------------------------------------------------------ scoring
def load_t1_eval():
    path = os.path.join(REPO, "taniteval", "tools", "t1_eval.py")
    spec = importlib.util.spec_from_file_location("t1_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_dumps(out_dir, wins, arms: dict, g, v0):
    """One ``ep*.npz`` per episode in ``t1_eval``'s schema; ``arms`` = {name: [N, 30, 2]}."""
    os.makedirs(out_dir, exist_ok=True)
    cis = np.array([w[0] for w in wins])
    ts = np.array([w[1] for w in wins])
    files = []
    for ci in dict.fromkeys(cis.tolist()):
        m = cis == ci
        f = os.path.join(out_dir, f"ep{ci:04d}.npz")
        np.savez(f, g=g[m], ws=ts[m], eid=cis[m], clip_index=cis[m], v0=v0[m], **{k: v[m] for k, v in arms.items()})
        files.append(f)
    return files


def analyze_dumps(files, header: dict, n_boot: int):
    t1 = load_t1_eval()
    res = t1.analyze(files, tiers={"fl": "T1", "ha0": "T1"}, n_boot=n_boot, dt=DT,
                     paired=[("ol", "cl", "paired_t1_minus_open"), ("fl", "cl", "paired_model_minus_floor_t1")])
    res["_tier_doctrine"] = TIER_DOCTRINE                                # t1_eval still prints the pre-amendment 'action-closed loop'
    res["t1v8f"] = header
    return res


# ------------------------------------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("run")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--tracks", required=True, help="native_tracks.pt (yaw mode needs the 10 Hz track)")
    p.add_argument("--eval-cache", default="C:/Users/Admin/refav1_hj_cache")
    p.add_argument("--eval-grid", required=True)
    p.add_argument("--eval-join", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--replan-steps", type=int, default=1, help="m: cache steps (0.2 s) executed per plan; must divide 30. m=30 == the one-shot open loop")
    p.add_argument("--schedule", choices=SCHEDULES, default="periodic")
    p.add_argument("--nav", choices=NAV_MODES, default="deadreckon")
    p.add_argument("--scene", choices=SCENE_MODES, default="logged")
    p.add_argument("--limit", choices=LIMIT_MODES, default="logged")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lru", type=int, default=32)
    p.add_argument("--max-windows", type=int, default=0, help="smoke: evenly-spaced subset of the grid (0 = all)")
    p.add_argument("--level-file", default="", help="LEVELPROBE: npz (clip_index, ws, level) with one row per window; required by, and only allowed with, --histswap cv_pred")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--dump-ticks", default="", help="PREREG_V8F_HACC_LAT: a directory; writes ticks_<tag>.npz = one row per (window, tick) of the model's rollout (tau, P, plan, v0, GT in the executed frame, ids)")
    p.add_argument("--feasible", choices=FEASIBLE_MODES, default="off", help="KAPPA-T1 / FEAS2-T1: feasible_decode.project_feasible on the plan AFTER --smooth, at every replan, before the executor (dt=DT, mu=None): box = A_MAX 4.0 and KAPPA_MAX 0.2, clamp_entry=False; kbox = KAPPA_MAX 0.2 only (a_max=inf); abox = A_MAX 4.0 only (kappa_max=inf) with clamp_entry=True and v0 = the tick's measured ego speed; kabox = kbox then abox; off (default) = unchanged bit for bit")
    p.add_argument("--smooth", choices=SMOOTH_MODES, default="off", help="SMOOTH-T1: sg72 = PREREG_V8F_SMOOTH's S(plan) = P + SG(plan - P) (window 7, order 2, interp) on the model's plan at every replan, before the executor; off (default) = unchanged bit for bit")
    p.add_argument("--histswap", choices=HISTSWAP_MODES, default="off", help="HISTSWAP-LF: cv_exec = a constant-twist (constant-velocity) history at the EXECUTED current speed (deployable); cv_logged = the same at the LOGGED current speed (ORACLE, level only); both need --limit causal. HISTSWAP (H-HISTSWAP-1): logged = the model's ego-history input at every tick (p10f, base, kin3, pose*, the causal speed bucket) is read from the LOGGED clip arrays instead of the executed ones; the plan is still committed to the executed state; nav, scene, ages, --smooth and the P10 floor untouched. ORACLE diagnostic, not deployable. off (default) = unchanged bit for bit; refuses to combine with --feasible")
    p.add_argument("--dump-heads", action="store_true", help="T1-HEADS (needs --dump-ticks): add the S head and T head outputs of each tick's forward_infer call to those rows (s_ckpt*, t_goal_prob, t_lat_logits, t_lon_logits, t_cons_pred + poses)")
    p = sub.add_parser("analyze")
    p.add_argument("--dump-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=2000)
    return ap


def cmd_run(a):
    import torch
    if a.dump_heads and not a.dump_ticks:
        raise SystemExit("REFUSED: --dump-heads adds columns to the --dump-ticks file; it needs --dump-ticks DIR (a flag that silently writes nothing is worse than none)")
    if a.histswap != "off" and a.feasible != "off":
        raise SystemExit("REFUSED: --histswap logged is not defined together with --feasible (abox / kabox read the tick's input speed; the card has no such arm)")
    if a.histswap in CV_MODES and a.limit != "causal":
        raise SystemExit(f"REFUSED: --histswap {a.histswap} is defined with --limit causal only (got {a.limit})")
    if (a.histswap == "cv_pred") != bool(a.level_file):
        raise SystemExit("REFUSED: --histswap cv_pred needs --level-file, and --level-file is allowed with cv_pred only")
    LEVELS = load_levels(a.level_file) if a.level_file else None
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    if cfg.kin_mode != "yaw":
        raise SystemExit("REFUSED: T1-kin rebuilds the ego state from the executed pose track; a --kin-mode steer checkpoint reads the "
                         "recorded steering angle (PI ruling D-PI-YAWRATE-1) and has no own-motion equivalent.")
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    dev = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    tracks = D.load_native_tracks(a.tracks)
    J = D.load_join(a.eval_join)
    store = D.ClipStore(a.eval_cache, "eval", lru=a.lru, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(a.eval_grid, J)
    if a.max_windows and a.max_windows < len(wins):
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, a.max_windows).round().astype(int)]
    proxy = _OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s,
                          window_start=cfg.window_start)     # PREREG_V8F_HIST Amendment 1: the held-age clamp origin is the CHECKPOINT's (-1 = absent = hist - 1)
    kw = dict(m=a.replan_steps, schedule=a.schedule, scene=a.scene, limit=a.limit, device=dev)
    cl, ol, fl, ha0, g, v0, keys = [], [], [], [], [], [], set()
    n_clip = {"cl": 0, "fl": 0}
    rec = None
    t_start = time.time()
    for b0 in range(0, len(wins), a.batch):
        chunk = wins[b0:b0 + a.batch]
        mp = model_planner(model, dev, keep_aux=bool(a.dump_ticks), keep_heads=bool(a.dump_heads), smooth=a.smooth, feasible=a.feasible)
        st_m = make_states(store, tracks, J, chunk, a.nav, hist_swap=a.histswap, levels=LEVELS)     # the P10 floor's states below are never swapped
        if a.dump_ticks:
            rec = rec or TickRecorder(mp, store, J, heads=a.dump_heads, smooth=a.smooth, feasible=a.feasible, histswap=a.histswap)
            rec.planner, rec.base = mp, b0
            plan0, exe = rollout(mp, st_m, ds, proxy, hook=rec, **kw)
        else:
            plan0, exe = rollout(mp, st_m, ds, proxy, **kw)
        keys |= mp.keys_read
        st_f = make_states(store, tracks, J, chunk, a.nav)
        _, exe_f = rollout(floor_planner(model, dev), st_f, ds, proxy, **kw)
        n_clip["cl"] += sum(s.clipped > 0 for s in st_m)
        n_clip["fl"] += sum(s.clipped > 0 for s in st_f)
        cl += exe
        fl += exe_f
        ol.append(plan0)
        for ci, t0, _, _ in chunk:
            d = store.get(str(J["clip"][ci]))
            g.append(d["gt"][t0])
            v0.append(d["kin3"][t0][0])                                  # the ego speed AT t0 (logged history, allowed)
        print(f"[{b0 + len(chunk)}/{len(wins)}] {time.time() - t_start:.0f}s", flush=True)
    v0 = np.asarray(v0, np.float64)
    arms = {"cl": np.asarray(cl), "ol": np.concatenate(ol).astype(np.float64), "fl": np.asarray(fl),
            "ha0": fk.ha0_paths(v0, DT, H)}
    gt = np.asarray(g, np.float64)
    files = write_dumps(os.path.join(a.out_dir, f"dump_{a.tag}"), wins, arms, gt, v0)
    header = {"tag": a.tag, "ckpt": a.ckpt, "kin_mode": cfg.kin_mode, "tier": "T1 (self-action OPEN loop; ego state = own executed motion)",
              "replan_steps": a.replan_steps, "replan_hz": 5.0 / a.replan_steps, "schedule": a.schedule, "scene": a.scene, "nav": a.nav,
              "limit": a.limit, "n_windows": len(wins), "model_keys_read": sorted(keys),
              "windows_with_envelope_clip": n_clip, "envelope_ms": V_ENV_MS,
              "strict": {"kinematics": True, "route": a.nav == "deadreckon", "camera": a.scene == "frozen", "speed_bucket": a.limit != "logged"},
              "open_channels": ["camera stream (logged frames after t0 carry the human's ego-motion)" if a.scene == "logged" else "",
                                "route countdown from the logged human position" if a.nav == "logged" else "",
                                "speed bucket from the human's realised future max speed (D-SPEEDBAND-EGO-ECHO-1)" if a.limit == "logged" else ""],
              "acts": {k: fk.acts_check(v, DT) for k, v in arms.items() if k in ("cl", "fl", "ol")}}
    header["open_channels"] = [c for c in header["open_channels"] if c]
    if a.limit in ("causal", "causal_all"):                              # only then: --limit logged / off outputs stay byte-identical (C-L1)
        header["causal_bucket"] = {"rule": "speed_max_bin(max finite measured ego speed over rows "
                                           + (f"[j - {CAUSAL_STEPS}, j])" if a.limit == "causal" else "[0, j])"),
                                   "stats_model_ticks": CAUSAL_STATS}
    if a.smooth != "off":                                                # only then: --smooth off / absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        header["smooth"] = {"mode": a.smooth, "operator": f"P + savgol_filter(plan - P, {SG_WINDOW}, {SG_ORDER}, axis=1, mode='interp')", "prereg": "PREREG_V8F_SMOOTH (sha256 a2f19e48...)",
                            "applied": "to the model's plan at every replan, before the executor; the P10 floor is not smoothed"}
    if a.feasible != "off":                                              # only then: --feasible off / absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        header["feasible"] = {"mode": a.feasible, "operator": FEASIBLE_OPERATORS[a.feasible],
                              "prereg": "dagland/kappa_t1/PREREG_FROZEN.md (H-KAPPA-T1-1)" if a.feasible == "box" else "dagland/feas2_t1/PREREG_FROZEN.md (H-FEAS2-T1-1)",
                              "applied": "to the plan after --smooth at every replan, before the executor; the P10 floor is not projected"}
    if a.histswap != "off":                                              # only then: --histswap off / absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        header["histswap"] = {"mode": a.histswap, "applied": "the model's ego-history input at every tick comes from the LOGGED clip arrays; the plan is committed to the EXECUTED state; the P10 floor, nav, scene, ages and --smooth are untouched",
                              "oracle": "yes: the logged history after t0 is the human's; NOT deployable", "prereg": "dagland/histswap/PREREG_HISTSWAP_FROZEN.md (H-HISTSWAP-1)",
                              "bucket_stats_model_ticks": {k: CAUSAL_STATS.get(k, 0) for k in ("swap_ticks", "swap_bucket_differs")}}
    if a.histswap in CV_MODES:                                           # LF: the header says what the history IS
        header["histswap"].update({"applied": "the model's ego-history input at every tick is a CONSTANT-TWIST history (speed and yaw rate held, zero acceleration) anchored at the EXECUTED pose; the plan is committed to the EXECUTED state; "
                                              "the P10 floor, nav, scene, ages and --smooth are untouched",
                                   "speed_source": "the EXECUTED measured speed at the tick (deployable)" if a.histswap == "cv_exec" else "the LOGGED measured speed at the tick (ORACLE, level only: the human's speed, no trend)",
                                   "oracle": "no" if a.histswap == "cv_exec" else "yes: the logged speed after t0 is the human's; NOT deployable",
                                   "prereg": "dagland/histswap_lf/PREREG_HISTSWAP_LF_FROZEN.md (H-HISTSWAP-LF-1)"})
    if a.histswap in LEVEL_MODES:                                        # LEVELPROBE: the header says where the constant level comes from
        header["histswap"].update({"speed_source": ("the LOGGED measured speed at the window's t0, held constant at every tick (ORACLE, level at t0 only)" if a.histswap == "cv_t0"
                                                    else "a per-window level read from --level-file (vision-derived), held constant at every tick"),
                                   "oracle": "yes: the logged speed at t0 is the human's; NOT deployable" if a.histswap == "cv_t0" else "no: the level file is the vision probe's prediction from the t0 frame",
                                   "prereg": "dagland/levelprobe/PREREG_LEVELPROBE.md (H-LEVELPROBE-1)"})
        if a.level_file:
            header["histswap"]["level_file"] = {"path": a.level_file, "sha256": hashlib.sha256(open(a.level_file, "rb").read()).hexdigest(), "n_windows": len(LEVELS)}
    if a.dump_ticks:                                                     # only then: flags absent => the header (and the result JSON) is byte-identical to the pre-flag tool's
        tick_path = os.path.join(a.dump_ticks, f"ticks_{a.tag}.npz")
        rec.save(tick_path, {"split": "eval", "tag": a.tag, "ckpt": a.ckpt, "replan_steps": a.replan_steps, "n_windows": len(wins), "clip_names": np.asarray(J["clip"]),
                             **({"smooth": a.smooth} if a.smooth != "off" else {}), **({"feasible": a.feasible} if a.feasible != "off" else {}),
                             **({"histswap": a.histswap} if a.histswap != "off" else {})})
        header["ticks_file"] = tick_path
        header["ticks_heads"] = bool(a.dump_heads)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.no_analyze:
        json.dump(header, open(os.path.join(a.out_dir, f"t1v8f_{a.tag}.json"), "w"), indent=1)
        return 0
    res = analyze_dumps(files, header, a.n_boot)
    out = os.path.join(a.out_dir, f"t1v8f_{a.tag}.json")
    json.dump(res, open(out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote", out)
    return 0


def cmd_analyze(a):
    files = sorted(os.path.join(a.dump_dir, f) for f in os.listdir(a.dump_dir) if f.startswith("ep") and f.endswith(".npz"))
    res = analyze_dumps(files, {"note": "re-scored from existing dumps; run header not available"}, a.n_boot)
    json.dump(res, open(a.out, "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return 0


def main(argv=None):
    a = build_parser().parse_args(argv)
    return {"run": cmd_run, "analyze": cmd_analyze}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
