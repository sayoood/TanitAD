"""taniteval.control — the LONGITUDINAL and LATERAL CONTROL SUITE, first class.

WHY THIS MODULE EXISTS (the PI's demand, and the MEASURED case for it)
----------------------------------------------------------------------
*"You are always and only using ADE to assess the approaches; we need in
addition the other metrics, especially the longitudinal and lateral control."*

The closed-loop composite is **not** ADE — it is
``PSS_recovery_progress@twosided_v2`` over ``{ego_progress, recovery, comfort}``.
But the concern is substantively right, and here is the measured case:

* ⛔ ``comfort`` is **information-free on this surface** — see :data:`COMFORT_STATUS`.
* ⛔ There is **no collision gate and no drivable-area term** (no cuboids, no map:
  ``pseudosim.COLLISION_UNAVAILABLE_REASON``).
* ⛔ The whole goal/selection line — Bar A, T3, E-GOAL-1→4 — is scored on
  ``ade_0_2s``, and every headline from it is an ADE number.
* ⛔ **The primitives existed but were DIAGNOSTICS, not ranked axes.** MEASURED
  consequence: an intervention cut along-track RMS **8.799 → 1.557 m (5.65x)**,
  cut the longitudinal error share **76.9 % → 8.9 %**, raised the distance
  hit-rate **15.00 % → 58.93 %** — **and the composite did not move at all**
  (``+0.0078 [-0.0110, +0.0260] n.s.`` under ``clamp_v1``).

⇒ This module makes longitudinal and lateral control **ranked axes with
demonstrated dynamic range**, on the *same rows* the composite scores, with the
*same* estimator, so a plan can never again move an axis by 5.65x invisibly.

WHAT "FIRST CLASS" MEANS HERE — FOUR REQUIREMENTS, ALL ENFORCED IN CODE
-----------------------------------------------------------------------
1. **Same rows.** Every axis is masked by the *identical* predicate
   ``ego_progress`` uses (``human_chord > 0.5 m``), so a paired bootstrap
   against the composite is valid row-for-row.
   Pinned by ``test_every_axis_shares_ego_progress_row_mask``.
2. ⭐ **Demonstrated dynamic range.** :func:`dynamic_range` injects a *controlled*
   degradation on the axis and reports the ladder, the monotonicity, the paired
   separation and the **MDE in physical units**. ⛔ A metric that cannot fail is
   not a metric (class C13) and :func:`admit` refuses it.
3. ⭐ **BOTH directions.** Every control ladder is **signed and straddles the
   null**, because this program has already shipped a one-sided metric:
   ``clamp_v1`` charged nothing for over-travel and v1 over-travels on
   **48.80 %** of windows (p95 ratio **2.430x**). :func:`admit` refuses an axis
   that separates on only one side of a two-sided ladder.
4. ⭐ **The control's own direction is VERIFIED, not assumed.** Three ways a
   degradation moves the wrong way *for a structural reason*, all MEASURED in
   this program:

   * a **slowdown raised** the composite ``+0.1698`` because pseudo-sim scores a
     barely-moving plan ``recovery = NaN`` **by construction** — the arm left the
     denominator, it did not beat it;
   * a **constant-sign lateral drift made an arm better** by re-centring a
     one-sidedly biased planner;
   * ``comfort`` moved **720x** between two arms differing only in *schedule*,
     because re-timing at constant speed *smooths* a per-waypoint regression.

   ⇒ every ladder rung reports ``defined_frac`` for every axis, the **signed**
   bias (:func:`axes` ``lat_bias_m``), and the composite beside the axis, so a
   structural move is visible rather than inferred. :data:`ZERO_MEAN_CONTROLS`
   names the ladders that **cannot** re-centre a bias in expectation; at least
   one is required for a lateral admission.

THE GATE IS PANEL-WIDE, AND IT IS SAID OUT LOUD
------------------------------------------------
:func:`panel_gate` admits an axis only if it is admissible for **every
non-probe arm**, so both sides of every paired delta are the same object.
⛔ The **per-arm** gate is refused, not offered: it moved ``comfort``
``0.0004 → 0.2882`` (**720x**) between two arms differing only in schedule and
flipped a stream's own primary verdict from ``n.s.`` to ``SEPARATED``.

ESTIMATOR. ``taniteval.ci.episode_cluster_bootstrap`` / the **paired** form for
two arms on the same windows (``B = 2000``, unit = val episode).
⛔ ``overlapping_holdout_se`` appears nowhere — it biases the POINT ESTIMATE as
well as the interval.
"""
from __future__ import annotations

import numpy as np
import torch

from taniteval import ci as _ci
from taniteval import lateral as _lat
from taniteval import pseudosim as _ps
from taniteval.clhorizon import DT, wrap_angle

__all__ = [
    "BLOCK", "VERSION", "SUITE_ID",
    "T_TOL_S", "D_TOL_M", "S_REF_M", "PSI_TOL_RAD", "V_MIN_MPS",
    "HUMAN_MIN_M", "S_MIN_M", "signed_xte",
    "TOL_SENSITIVITY", "AXES", "AXIS_META", "CONTROL_WEIGHTS",
    "ZERO_MEAN_CONTROLS", "CONTROLS", "LADDERS",
    "COMFORT_STATUS", "MISSING_GATES",
    "AxisNotDemonstrated",
    "residuals", "axes", "axis_summary", "apply_control", "ladder_levels",
    "dynamic_range", "admit", "panel_gate", "control_score", "block",
]

BLOCK = "taniteval.control/longitudinal_lateral"
VERSION = "1.0.0"
#: ⛔ QUOTE THIS, never a bare "control score". Every constant that can move a
#: number is in the id, for the same reason ``pseudosim.metric_id`` versions the
#: progress term: a silent redefinition under a stable name is the failure class
#: this program has logged most often.
def SUITE_ID(t_tol=None, d_tol=None, psi_tol=None, s_ref=None) -> str:
    t = T_TOL_S if t_tol is None else t_tol
    d = D_TOL_M if d_tol is None else d_tol
    p = PSI_TOL_RAD if psi_tol is None else psi_tol
    s = S_REF_M if s_ref is None else s_ref
    return f"control_v1@t{t:g}s_d{d:g}m_sref{s:g}m_psi{p:g}rad"


# --------------------------------------------------------------------------- #
# tolerances — PROPOSED, published, and swept. None of them is MEASURED.       #
# --------------------------------------------------------------------------- #
#: ⚠️ PROPOSED. ``lon_track`` reaches 0 when the plan is this many seconds off
#: the human's schedule. A *time* tolerance (not a distance one) is deliberate:
#: it is scale-free across speeds, so a 2 m error at 5 m/s and a 7 m error at
#: 17 m/s are charged alike, and it is the quantity longitudinal control is
#: actually about. Swept in :data:`TOL_SENSITIVITY`.
T_TOL_S = 1.0
#: ⚠️ PROPOSED — ``lateral.LANE_HALF_M``, itself marked PROPOSED there because
#: **no lane geometry exists in this corpus** (``driving.py`` refuses
#: ``lane_centre_deviation`` for the same reason). Swept.
D_TOL_M = _lat.LANE_HALF_M           # 1.75 m
#: ⭐⭐ THE CORRIDOR WIDENS WITH TRAVEL, AND THIS IS THE FIX THAT MAKES THE
#: LATERAL AXIS A LATERAL AXIS.
#: The lateral tolerance at plan-arc ``s`` is ``D_TOL_M * max(s, S_MIN_M) /
#: S_REF_M`` — i.e. ``D_TOL_M`` is reached after ``S_REF_M`` metres of travel.
#:
#: WHY. A FLAT metre tolerance is **not axis-pure**: MEASURED on ``cv_holdv0``,
#: halving the plan's speed (``lon_retime(0.5)``, which leaves the PATH
#: untouched) moved a flat-tolerance ``lat_track`` by **+0.1709** — *upward*,
#: i.e. a purely longitudinal degradation scored as better lateral control —
#: while a genuine 1.0 m lateral offset moved it only **−0.1379**. The
#: contamination was **larger than the signal**. A plan that drives half as far
#: accrues less lateral deviation because **lateral error COMPOUNDS** (MEASURED:
#: lateral grows x14.11 over 0.5→2 s against longitudinal x3.20). Normalising by
#: travel removes exactly that, and the removal is measured, not asserted: the
#: same ``lon_retime(0.5)`` contamination falls to **+0.0321, a 5.3x reduction**,
#: while a **0.25 m** offset now registers **−0.0339** — *comparable to halving
#: the speed*, where before it took a 1 m offset to match.
#: ⚠️ PROPOSED. ``S_REF_M = 10`` is chosen on CONDITIONING, stated: at 5/10/20/40
#: the floor-saturation of the score is 0.00 / 0.07 / 0.29 / 0.55 and the
#: ceiling 0.003 / 0.001 / 0.000 / 0.000, so 10 is the best-conditioned point
#: that is not near either bound. Swept in :data:`TOL_SENSITIVITY`.
S_REF_M = 10.0
#: ⚠️ PROPOSED. 0.2 rad = 11.46 deg of terminal heading error. Swept.
PSI_TOL_RAD = 0.2
#: speed floor for the time-error normaliser: below this the "seconds off
#: schedule" reading is dominated by the floor, not by control.
V_MIN_MPS = 1.0
#: ⭐ the row mask, copied from ``pseudosim.score_windows`` so the row sets are
#: IDENTICAL and a paired bootstrap against the composite is valid.
HUMAN_MIN_M = 0.5
#: ⛔⛔ THE ANTI-STANDING-STILL FLOOR, AND IT IS NOT DECORATIVE.
#: MEASURED while building this module: with the naive time-indexed definition
#: and no floor, ``stand_still`` scored ``lat_track = 0.8455`` — **the highest
#: value in the whole panel**, above ``cv_holdv0`` (0.4414) and
#: ``v1_tactical_follow`` (0.4096) — because a plan that does not move has
#: almost no cross-track error and was being PAID for it. That is the identical
#: defect ``recovery``'s progress-matched denominator was written to remove
#: (*"standing still is not recovery"*), reappearing one axis over. A lateral
#: axis is therefore **UNDEFINED (NaN)** for a plan whose along-track travel is
#: below this floor: 2.0 m over a 2 s horizon is 1 m/s, the same floor
#: :data:`V_MIN_MPS` uses. Pinned by
#: ``test_lat_track_is_not_gameable_by_standing_still``.
S_MIN_M = 2.0

#: every verdict is re-run at each grid point; no verdict may rest on one.
TOL_SENSITIVITY = {"t_tol_s": (0.5, 1.0, 2.0),
                   "d_tol_m": (0.875, 1.75, 3.5),
                   "s_ref_m": (5.0, 10.0, 20.0),
                   "psi_tol_rad": (0.1, 0.2, 0.4)}

#: the four ranked axes. ``higher_is_better`` is stated so a sign error in a
#: ladder cannot be read as a finding.
AXES = ("lon_track", "lat_track", "lat_heading", "recovery")
AXIS_META = {
    "lon_track": {
        "kind": "LONGITUDINAL", "higher_is_better": True, "unit": "score [0,1]",
        "raw": "lon_time_err_s", "raw_unit": "s (signed, + = plan AHEAD)",
        "two_sided": True,
        "what": "mean over the horizon of clamp(1 - |seconds off the human's "
                "schedule| / T_TOL_S, 0, 1). A STRICT GENERALISATION of "
                "ego_progress, which reads the endpoint only and therefore "
                "cannot see a plan that arrives on time by the wrong route "
                "through time."},
    "lat_track": {
        "kind": "LATERAL", "higher_is_better": True, "unit": "score [0,1]",
        "raw": "xte_m", "raw_unit": "m (signed, + = plan is LEFT of the path)",
        "two_sided": True,
        "what": "mean over the horizon of clamp(1 - |XTE| / D_TOL_M, 0, 1), "
                "where XTE is the ARC-LENGTH-MATCHED cross-track: the signed "
                "perpendicular distance from each plan point to the human's "
                "POLYLINE, not to the human's position at the same time index. "
                "⭐ That choice is what makes it a LATERAL axis: a plan that "
                "follows the right line at the wrong speed has XTE ~ 0, so "
                "longitudinal error does not leak in. The time-indexed version "
                "is kept as `cross_err_m` because `recovery` is defined on it. "
                "Sign-blind BY DESIGN — the sign lives in `lat_bias_m`, the "
                "one-sidedness detector. ⛔ UNDEFINED below S_MIN_M of plan "
                "travel: see S_MIN_M for the measured reason."},
    "lat_heading": {
        "kind": "LATERAL", "higher_is_better": True, "unit": "score [0,1]",
        "raw": "heading_err_rad", "raw_unit": "rad (signed, + = plan turns LEFT)",
        "two_sided": True,
        "what": "clamp(1 - |terminal heading error| / PSI_TOL_RAD, 0, 1). "
                "⭐ It is a SECOND lateral axis and not a re-skin: a pure "
                "lateral OFFSET moves lat_track and leaves lat_heading exactly "
                "unchanged, while a steering DRIFT moves both. The pair is what "
                "distinguishes 'in the wrong place' from 'pointing the wrong "
                "way'. ⚠️ UNDEFINED (NaN) for a plan whose final segment has no "
                "length — standing still has no heading, and scoring it 1.0 "
                "would repeat the recovery defect verbatim."},
    "recovery": {
        "kind": "LATERAL (error recovery)", "higher_is_better": True,
        "unit": "score [0,1]", "raw": "cross_track_end_m", "raw_unit": "m",
        "two_sided": False,
        "what": "IMPORTED from pseudosim.score_windows, not reimplemented. The "
                "error-recovery signal pseudo-simulation exists to produce."},
}

#: ⭐ PROPOSED weights for a control-aware composite. NOT applied to
#: ``pseudosim.COMPONENT_WEIGHTS`` — swapping the program's primary is a PI
#: decision, and this module's job is to make it decidable, not to take it.
#: Shape follows PDM (progress 5 / recovery 5) with the two control axes at 5
#: and 2: longitudinal carries **83 % of 2 s error** and v4's 15k→30k regression
#: was **100 %** longitudinal, so it is not a minor term; lateral is entered at
#: 2 because v1's tactical STEERING is currently worth
#: ``+0.0006 [-0.0065, +0.0072] n.s.`` against a straight line.
CONTROL_WEIGHTS = {"ego_progress": 5.0, "recovery": 5.0,
                   "lon_track": 5.0, "lat_track": 2.0, "lat_heading": 1.0}

#: ⭐ Controls whose expectation is ZERO, so they cannot re-centre a one-sided
#: bias and make a broken arm look better. At least one is required before a
#: LATERAL axis may be admitted — that failure mode is MEASURED here, not
#: hypothetical.
ZERO_MEAN_CONTROLS = ("lat_jitter", "lon_jitter", "yaw_jitter")


class AxisNotDemonstrated(AssertionError):
    """An axis was used without a demonstrated dynamic range, or it failed one.

    ⛔ This is an error, not a report line. ``pseudosim.VacuousMetric`` refuses
    to emit a composite when no component has range; this refuses to *rank* on
    an axis whose response to a controlled degradation was never shown."""


# =========================================================================== #
# the residuals — geometry IMPORTED from pseudosim, never re-derived           #
# =========================================================================== #
def signed_xte(px, py, gx, gy, *, chunk=4096, return_segment=False):
    """⭐ ARC-LENGTH-MATCHED signed cross-track: plan point -> human POLYLINE.

    ``px, py`` are ``[n, K]`` plan points and ``gx, gy`` are ``[n, K+1]`` human
    path points, both in the reference-ego frame. Returns ``[n, K]`` signed
    metres, **positive = the plan is LEFT of the path**.

    WHY THE POLYLINE AND NOT THE SAME TIME INDEX
    --------------------------------------------
    ``|plan[k] - human[k]|`` mixes the two axes: a plan that drives the correct
    *line* 5 m behind schedule shows a large "cross-track" error purely because
    the road curved. On this corpus that is not a small effect — **ADE is 98.6 %
    longitudinal by squared-error energy**, so a lateral metric contaminated by
    the longitudinal axis is numerically a longitudinal metric, which is the
    disease this suite exists to cure. Projecting onto the polyline removes it
    by construction, and the removal is *demonstrated*: under ``lon_scale`` the
    time-indexed version moves and this one barely does
    (``test_lat_track_is_far_less_sensitive_to_a_purely_longitudinal_control``).

    The FIRST and LAST segments are extended to infinite lines (``t`` unclamped
    on the outside end) so a plan that over- or under-travels is still measured
    perpendicular to the road rather than to its endpoint — otherwise the axis
    would re-acquire exactly the longitudinal leak it was built to drop.
    """
    px = np.asarray(px, np.float64)
    py = np.asarray(py, np.float64)
    gx = np.asarray(gx, np.float64)
    gy = np.asarray(gy, np.float64)
    n, K = px.shape
    S = gx.shape[1] - 1                       # number of human segments
    out = np.empty((n, K), np.float64)
    seg_of = np.empty((n, K), np.int64)
    for lo in range(0, n, chunk):
        hi = min(n, lo + chunk)
        ax, ay = gx[lo:hi, :-1], gy[lo:hi, :-1]           # [m, S]
        dx, dy = np.diff(gx[lo:hi], axis=1), np.diff(gy[lo:hi], axis=1)
        L2 = np.maximum(dx * dx + dy * dy, 1e-12)
        wx = px[lo:hi, :, None] - ax[:, None, :]          # [m, K, S]
        wy = py[lo:hi, :, None] - ay[:, None, :]
        t = (wx * dx[:, None, :] + wy * dy[:, None, :]) / L2[:, None, :]
        # clamp INSIDE only; the outer ends stay infinite lines
        tl = np.zeros(S)
        th = np.ones(S)
        tl[0], th[-1] = -np.inf, np.inf
        t = np.clip(t, tl[None, None, :], th[None, None, :])
        ex = wx - t * dx[:, None, :]
        ey = wy - t * dy[:, None, :]
        d2 = ex * ex + ey * ey
        j = np.argmin(d2, axis=2)                          # [m, K]
        take = np.take_along_axis
        dist = np.sqrt(take(d2, j[..., None], axis=2)[..., 0])
        # sign from the cross product of the chosen segment with the offset
        cx = (take(dx[:, None, :].repeat(K, 1), j[..., None], axis=2)[..., 0]
              * take(wy, j[..., None], axis=2)[..., 0]
              - take(dy[:, None, :].repeat(K, 1), j[..., None], axis=2)[..., 0]
              * take(wx, j[..., None], axis=2)[..., 0])
        out[lo:hi] = np.where(cx >= 0, dist, -dist)
        seg_of[lo:hi] = j
    return (out, seg_of) if return_segment else out


def residuals(pw, *, dt=DT) -> dict:
    """Signed (along, cross) residual of the plan against the LOGGED path.

    Frame: the reference pose's ego frame — component 0 along-track, component 1
    cross-track — produced by ``pseudosim._cross_and_along``, which is imported
    rather than re-derived so the two surfaces cannot drift apart.

    ⚠️ **Index alignment is load-bearing.** ``pseudo_evaluate`` stores
    ``ref_path`` with ``horizon + 1`` entries where index 0 *is* the reference
    pose, and ``traj`` with ``horizon`` entries starting one tick later. So plan
    step ``k`` pairs with ``ref`` index ``k + 1``, and ``score_windows``'s
    endpoint pair ``(x[:, -1], ref_x[:, -1])`` is the same convention. An
    off-by-one here would silently report a 0.1 s lead as a control error;
    ``test_residuals_are_zero_when_the_plan_IS_the_logged_path`` pins it at
    exactly 0.
    """
    x, y, ref_x, ref_y = _ps._cross_and_along(pw)
    x, y = x.numpy().astype(np.float64), y.numpy().astype(np.float64)
    ref_x, ref_y = ref_x.numpy().astype(np.float64), ref_y.numpy().astype(np.float64)
    Hh = x.shape[1]
    gx, gy = ref_x[:, 1:Hh + 1], ref_y[:, 1:Hh + 1]        # the human, aligned
    along = x - gx                                          # + = travelled FURTHER
    cross = y - gy                                          # + = LEFT of the path
    # the human's own motion, for the normalisers
    seg = np.sqrt(np.diff(ref_x, axis=1) ** 2 + np.diff(ref_y, axis=1) ** 2)
    arc = seg.sum(1)                                        # path length, m
    chord = np.sqrt((ref_x[:, -1] - ref_x[:, 0]) ** 2
                    + (ref_y[:, -1] - ref_y[:, 0]) ** 2)
    v_ref = np.maximum(arc / (Hh * dt), V_MIN_MPS)
    # terminal headings, from the LAST segment of each path
    def _psi(px, py):
        d0, d1 = px[:, -1] - px[:, -2], py[:, -1] - py[:, -2]
        ang = np.arctan2(d1, d0)
        ang[np.hypot(d0, d1) < 1e-6] = np.nan     # no segment => no heading
        return ang
    psi_plan = _psi(x, y)
    # ⭐ the arc-length-matched cross-track and the plan's own travel, which is
    # what makes the lateral axis un-gameable by not moving.
    xte, seg = signed_xte(x, y, ref_x, ref_y, return_segment=True)
    # ⭐⭐ THE HEADING REFERENCE IS ARC-MATCHED TOO, and this is a CORRECTION.
    # Comparing the plan's terminal heading with the human's heading at the
    # SAME TIME INDEX makes a slow plan on the CORRECT path look mis-aimed,
    # because at half the arc the road points somewhere else. MEASURED on the
    # zero-bias reference arm: under the time-indexed reference a
    # path-preserving `lon_retime(0.5)` drove `lat_heading` 1.0000 -> 0.7713
    # (-0.2287) while `lon_retime(2.0)` left it at exactly 1.0000 — a ONE-SIDED
    # longitudinal contamination of a LATERAL axis, which is the disease this
    # module exists to cure. The reference tangent is therefore taken on the
    # human segment CLOSEST TO THE PLAN'S ENDPOINT, the same segment index
    # `signed_xte` already chose.
    j_end = seg[:, -1]
    rdx = np.take_along_axis(np.diff(ref_x, axis=1), j_end[:, None], 1)[:, 0]
    rdy = np.take_along_axis(np.diff(ref_y, axis=1), j_end[:, None], 1)[:, 0]
    psi_ref = np.arctan2(rdy, rdx)
    psi_ref[np.hypot(rdx, rdy) < 1e-6] = np.nan
    dpsi = wrap_angle(torch.as_tensor(psi_plan - psi_ref)).numpy()
    plan_seg = np.sqrt(np.diff(np.concatenate([np.zeros((x.shape[0], 1)), x], 1),
                               axis=1) ** 2
                       + np.diff(np.concatenate([np.zeros((y.shape[0], 1)), y], 1),
                                 axis=1) ** 2)
    plan_arc_cum = np.cumsum(plan_seg, axis=1)                  # [n, Hh]
    plan_arc = plan_arc_cum[:, -1]
    return {
        "along_err_m": along, "cross_err_m": cross, "xte_m": xte,
        "plan_x": x, "plan_y": y, "human_x": gx, "human_y": gy,
        "human_arc_m": arc, "human_chord_m": chord, "v_ref_mps": v_ref,
        "plan_arc_m": plan_arc, "plan_arc_cum_m": plan_arc_cum,
        "heading_err_rad": dpsi,
        "row_mask": chord > HUMAN_MIN_M,
        "lat_mask": (chord > HUMAN_MIN_M) & (plan_arc >= S_MIN_M),
        "_lat_mask_rule": (
            f"row_mask AND plan_arc_m >= {S_MIN_M} m. ⛔ MEASURED: without the "
            f"second clause `stand_still` scored the panel's HIGHEST lat_track "
            f"(0.8455) — a plan that does not move has no cross-track error and "
            f"was being paid for it, the same defect `recovery` was fixed for."),
        "_row_mask_rule": (f"human_chord_m > {HUMAN_MIN_M} — BIT-IDENTICAL to "
                           f"pseudosim.score_windows' ego_progress mask, so a "
                           f"paired bootstrap against the composite is valid "
                           f"row-for-row."),
        "horizon_steps": int(Hh), "dt_s": float(dt),
    }


def axes(pw, *, t_tol=None, d_tol=None, psi_tol=None, s_ref=None, dt=DT) -> dict:
    """The ranked axes + their raw signed diagnostics, per (window, grid point).

    Every returned array is ``[n]``, NaN where the row is masked out, so it
    drops straight into ``ci.episode_cluster_bootstrap`` and into
    ``pseudosim.discriminative_range``.
    """
    t_tol = T_TOL_S if t_tol is None else float(t_tol)
    d_tol = D_TOL_M if d_tol is None else float(d_tol)
    psi_tol = PSI_TOL_RAD if psi_tol is None else float(psi_tol)
    s_ref = S_REF_M if s_ref is None else float(s_ref)
    r = residuals(pw, dt=dt)
    m, ml = r["row_mask"], r["lat_mask"]
    nan = lambda a: np.where(m, a, np.nan)            # noqa: E731
    nanl = lambda a: np.where(ml, a, np.nan)          # noqa: E731

    # ---- LONGITUDINAL ----------------------------------------------------- #
    t_err = r["along_err_m"] / r["v_ref_mps"][:, None]        # seconds, signed
    lon_track = np.clip(1.0 - np.abs(t_err) / t_tol, 0.0, 1.0).mean(1)
    # ---- LATERAL (arc-length matched + travel-normalised corridor) --------- #
    xte = r["xte_m"]
    xt_t = r["cross_err_m"]                       # time-indexed, diagnostic only
    corridor = d_tol * np.maximum(r["plan_arc_cum_m"], S_MIN_M) / s_ref
    lat_track = np.clip(1.0 - np.abs(xte) / corridor, 0.0, 1.0).mean(1)
    # the FLAT-tolerance twin, kept only so the contamination measurement that
    # motivated the widening corridor stays reproducible from the same call.
    lat_track_flat = np.clip(1.0 - np.abs(xte) / d_tol, 0.0, 1.0).mean(1)
    lat_heading = np.clip(1.0 - np.abs(r["heading_err_rad"]) / psi_tol, 0.0, 1.0)
    # ---- imported, never reimplemented ------------------------------------ #
    sc = _ps.score_windows(pw, dt=dt)

    out = {
        # ranked axes
        "lon_track": nan(lon_track),
        "lat_track": nanl(lat_track),
        "lat_heading": nanl(lat_heading),
        "recovery": sc["recovery"],
        "ego_progress": sc["ego_progress"],
        # raw signed diagnostics, in physical units
        "lon_time_err_s": nan(t_err[:, -1]),
        "lon_time_rmse_s": nan(np.sqrt((t_err ** 2).mean(1))),
        "lon_end_err_m": nan(r["along_err_m"][:, -1]),
        "lon_abs_end_err_m": nan(np.abs(r["along_err_m"][:, -1])),
        "lon_speed_err_mps": nan(r["along_err_m"][:, -1]
                                 / (r["horizon_steps"] * dt)),
        "lat_xte_end_m": nanl(np.abs(xte[:, -1])),
        "lat_xte_peak_m": nanl(np.abs(xte).max(1)),
        # ⭐ SIGNED — the one-sidedness detector. A constant-sign drift can make
        # a biased arm BETTER on lat_track; only this array shows it.
        "lat_bias_m": nanl(xte.mean(1)),
        "lat_heading_err_rad": nanl(r["heading_err_rad"]),
        # the two twins, kept so the axis-purity claim is checkable in-place
        "lat_track_flat": nanl(lat_track_flat),
        "lat_timeindexed_abs_m": nan(np.abs(xt_t).mean(1)),
        "plan_arc_m": nan(r["plan_arc_m"]),
        # provenance
        "_suite_id": SUITE_ID(t_tol, d_tol, psi_tol, s_ref),
        "_tolerances": {"t_tol_s": t_tol, "d_tol_m": d_tol, "s_ref_m": s_ref,
                        "psi_tol_rad": psi_tol, "v_min_mps": V_MIN_MPS,
                        "s_min_m": S_MIN_M,
                        "_class": "PROPOSED — none of these is MEASURED on this "
                                  "corpus; every verdict is re-run over "
                                  "TOL_SENSITIVITY"},
        "_row_mask_rule": r["_row_mask_rule"],
        "_lat_mask_rule": r["_lat_mask_rule"],
    }
    return out


def axis_summary(a, eid, *, names=AXES, n_boot=None, seed=0) -> dict:
    """Episode-cluster CI + tail statistics for each axis and its raw twin."""
    n_boot = _ci.DEFAULT_N_BOOT if n_boot is None else int(n_boot)
    out = {}
    for k in list(names) + ["lon_time_err_s", "lon_end_err_m",
                            "lon_abs_end_err_m", "lon_speed_err_mps",
                            "lat_xte_end_m", "lat_xte_peak_m", "lat_bias_m",
                            "lat_heading_err_rad", "ego_progress"]:
        v = np.asarray(a[k], dtype=np.float64)
        fin = np.isfinite(v)
        node = {"defined_frac": round(float(fin.mean()), 6),
                "n_defined": int(fin.sum())}
        if fin.sum() >= 2 and len(set(np.asarray(eid)[fin])) >= 2:
            node["ci"] = _ci.episode_cluster_bootstrap(
                v[fin], list(np.asarray(eid)[fin]), n_boot=n_boot, seed=seed)
            node["p90"] = round(float(np.nanpercentile(v[fin], 90)), 6)
            node["p10"] = round(float(np.nanpercentile(v[fin], 10)), 6)
        if k in ("lat_xte_peak_m", "lat_xte_end_m", "lon_abs_end_err_m"):
            node["tail"] = _lat.tail_stats(v[fin])
        out[k] = node
    return out


# =========================================================================== #
# the CONTROLLED DEGRADATIONS — pure arithmetic on the plan, no GPU, no model   #
# =========================================================================== #
def _copy_with_traj(pw, traj):
    out = dict(pw)
    out["traj"] = traj
    return out


def _ctl_lon_scale(pw, k):
    """Scale the plan's ALONG-track component by ``k``. ``k = 1`` is the null.

    ⚠️ This is the generalisation of the panel's own ``v1_ego_half`` (k = 0.5)
    and ``v1_ego_double`` (k ≈ 2) probes to a *signed ladder through 1.0*, which
    is exactly what those two probes were missing: they were adjacent points, not
    a curve, so neither could give an MDE."""
    t = pw["traj"].clone()
    t[..., 0] = t[..., 0] * float(k)
    return _copy_with_traj(pw, t)


def _ctl_lon_retime(pw, k):
    """⭐ THE **PATH-PRESERVING** longitudinal control. ``k = 1`` is the null.

    Re-samples the plan along **its own polyline** at arc length ``k * s(t)``:
    the geometric path is unchanged, only the *schedule* on it is scaled. Beyond
    the end the last segment is extended linearly, which is what "drive further
    along the same road" means.

    ⛔ **WHY THIS EXISTS, AND IT IS A CORRECTION TO MY OWN FIRST CONTROL.**
    ``lon_scale`` (an anisotropic scale of the along axis) is **not
    axis-pure**: shrinking x while leaving y rotates every segment, so
    ``lon_scale(0.5)`` moved ``lat_heading`` by **−0.0371** while a genuine
    5-degree ``yaw_bias`` moved it by only **−0.0069** — a "longitudinal"
    degradation contaminating the lateral axis **5.4x harder than the lateral
    control did**. Reading a lateral verdict off that would have been an
    artefact of the *control*, not of the metric. Both are kept: ``lon_scale``
    is the panel's own ``v1_ego_half`` / ``v1_ego_double`` construction and must
    stay comparable, ``lon_retime`` is the one a lateral purity claim may use.
    """
    k = float(k)
    t = pw["traj"].clone()
    P = t.numpy().astype(np.float64)
    n, K, _ = P.shape
    Q = np.concatenate([np.zeros((n, 1, 2)), P], axis=1)          # [n, K+1, 2]
    seg = np.linalg.norm(np.diff(Q, axis=1), axis=2)              # [n, K]
    S = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, axis=1)], axis=1)
    tgt = k * S[:, 1:]                                            # [n, K]
    # segment index: how many cumulative marks the target is at or past
    j = (tgt[:, :, None] >= S[:, None, 1:]).sum(2)
    j = np.clip(j, 0, K - 1)
    take = lambda A: np.take_along_axis(A, j, axis=1)             # noqa: E731
    s0, ln = take(S[:, :-1]), np.maximum(take(seg), 1e-9)
    u = ((tgt - s0) / ln)[:, :, None]                             # unclamped
    a = np.take_along_axis(Q[:, :-1], j[:, :, None], axis=1)
    b = np.take_along_axis(Q[:, 1:], j[:, :, None], axis=1)
    out = a + u * (b - a)
    return _copy_with_traj(pw, torch.as_tensor(out, dtype=t.dtype))


def _ctl_lat_shift(pw, d):
    """Add a CONSTANT lateral offset ``d`` metres. ``d = 0`` is the null.

    ⛔ **Directionally dangerous by construction, and that is why it is here.**
    A constant-sign offset RE-CENTRES a one-sidedly biased planner and can move
    an aggregate the *wrong* way. Its ladder is signed and straddles 0, and
    ``lat_bias_m`` is reported at every rung so the re-centring is visible."""
    t = pw["traj"].clone()
    t[..., 1] = t[..., 1] + float(d)
    return _copy_with_traj(pw, t)


def _ctl_lat_drift(pw, rate):
    """A constant STEERING error: ``y += rate * x``. ``rate = 0`` is the null.

    Distinct from :func:`_ctl_lat_shift` in exactly the way that matters: a
    drift changes the plan's terminal HEADING, an offset does not."""
    t = pw["traj"].clone()
    t[..., 1] = t[..., 1] + float(rate) * t[..., 0]
    return _copy_with_traj(pw, t)


def _ctl_lat_jitter(pw, sigma, seed=0):
    """⭐ ZERO-MEAN lateral noise, one draw per row, constant over the horizon.

    Expectation 0 by construction ⇒ **cannot re-centre a bias**, which makes it
    the control whose direction is unambiguous. Seeded, so it is deterministic."""
    if float(sigma) <= 0:
        return pw
    g = torch.Generator().manual_seed(int(seed))
    t = pw["traj"].clone()
    e = torch.randn(t.shape[0], 1, generator=g) * float(sigma)
    t[..., 1] = t[..., 1] + e
    return _copy_with_traj(pw, t)


def _ctl_lon_jitter(pw, sigma, seed=0):
    """⭐ ZERO-MEAN along-track noise, one draw per row. Same rationale."""
    if float(sigma) <= 0:
        return pw
    g = torch.Generator().manual_seed(int(seed) + 977)
    t = pw["traj"].clone()
    e = torch.randn(t.shape[0], 1, generator=g) * float(sigma)
    t[..., 0] = t[..., 0] + e
    return _copy_with_traj(pw, t)


def _ctl_yaw_jitter(pw, sigma_deg, seed=0):
    """⭐ ZERO-MEAN heading noise: one random rotation per row.

    ⛔ THIS CONTROL EXISTS BECAUSE THE SUITE'S OWN ADMISSION RULE REFUSED
    ``lat_heading`` WITHOUT IT. Every heading control I had first written
    (``yaw_bias``, ``lat_drift``) has a constant sign, and rule 3 requires a
    zero-mean one. ``lat_jitter`` cannot substitute: a per-row *translation*
    leaves the terminal heading exactly unchanged, so it is structurally unable
    to move that axis. The gap was real and the rule found it."""
    if float(sigma_deg) <= 0:
        return pw
    g = torch.Generator().manual_seed(int(seed) + 5501)
    t = pw["traj"].clone()
    a = torch.randn(t.shape[0], 1, generator=g) * float(np.deg2rad(sigma_deg))
    c, s = torch.cos(a), torch.sin(a)
    x, y = t[..., 0].clone(), t[..., 1].clone()
    t[..., 0] = c * x - s * y
    t[..., 1] = s * x + c * y
    return _copy_with_traj(pw, t)


def _ctl_yaw_bias(pw, deg):
    """Rotate the whole plan by ``deg`` about the ego origin. ``0`` is the null.

    A *pointing* error rather than a *placement* error: it moves ``lat_heading``
    hardest and ``lon_track`` least, which is the discrimination the two lateral
    axes exist to provide."""
    a = np.deg2rad(float(deg))
    t = pw["traj"].clone()
    c, s = float(np.cos(a)), float(np.sin(a))
    x, y = t[..., 0].clone(), t[..., 1].clone()
    t[..., 0] = c * x - s * y
    t[..., 1] = s * x + c * y
    return _copy_with_traj(pw, t)


CONTROLS = {
    "lon_scale": _ctl_lon_scale,
    "lon_retime": _ctl_lon_retime,
    "lon_jitter": _ctl_lon_jitter,
    "lat_shift": _ctl_lat_shift,
    "lat_drift": _ctl_lat_drift,
    "lat_jitter": _ctl_lat_jitter,
    "yaw_bias": _ctl_yaw_bias,
    "yaw_jitter": _ctl_yaw_jitter,
}

#: level ladders. ⭐ Every two-sided ladder STRADDLES its null, because an axis
#: validated only on one side is the ``clamp_v1`` failure repeated.
LADDERS = {
    "lon_scale": {"null": 1.0, "unit": "x along-track scale", "two_sided": True,
                  "levels": (0.50, 0.70, 0.85, 0.95, 1.00,
                             1.05, 1.15, 1.30, 1.50, 2.00)},
    "lon_retime": {"null": 1.0, "unit": "x speed (path preserved)",
                   "two_sided": True,
                   "levels": (0.50, 0.70, 0.85, 0.95, 1.00,
                              1.05, 1.15, 1.30, 1.50, 2.00)},
    "lon_jitter": {"null": 0.0, "unit": "m sigma (zero-mean)", "two_sided": False,
                   "levels": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)},
    "lat_shift": {"null": 0.0, "unit": "m constant offset", "two_sided": True,
                  "levels": (-2.0, -1.0, -0.5, -0.25, 0.0,
                             0.25, 0.5, 1.0, 2.0)},
    "lat_drift": {"null": 0.0, "unit": "m per m (steering error)",
                  "two_sided": True,
                  "levels": (-0.10, -0.05, -0.02, -0.01, 0.0,
                             0.01, 0.02, 0.05, 0.10)},
    "lat_jitter": {"null": 0.0, "unit": "m sigma (zero-mean)", "two_sided": False,
                   "levels": (0.0, 0.125, 0.25, 0.5, 1.0, 2.0)},
    "yaw_bias": {"null": 0.0, "unit": "deg plan rotation", "two_sided": True,
                 "levels": (-10.0, -5.0, -2.0, -1.0, 0.0,
                            1.0, 2.0, 5.0, 10.0)},
    "yaw_jitter": {"null": 0.0, "unit": "deg sigma (zero-mean rotation)",
                   "two_sided": False,
                   "levels": (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)},
}


def ladder_levels(control):
    if control not in LADDERS:
        raise KeyError(f"unknown control {control!r}; known: {sorted(LADDERS)}")
    return LADDERS[control]


def apply_control(pw, control, level, *, seed=0):
    """Apply one controlled degradation. Pure arithmetic; the dump is not mutated."""
    if control not in CONTROLS:
        raise KeyError(f"unknown control {control!r}; known: {sorted(CONTROLS)}")
    fn = CONTROLS[control]
    if control in ZERO_MEAN_CONTROLS:
        return fn(pw, level, seed=seed)
    return fn(pw, level)


# =========================================================================== #
# ⭐ THE DEMONSTRATION — this is the part the PI asked for                      #
# =========================================================================== #
def dynamic_range(pw, eid, *, control, axis, t_tol=None, d_tol=None,
                  psi_tol=None, s_ref=None, n_boot=None, seed=0, levels=None,
                  also=("ego_progress", "recovery", "lat_bias_m"),
                  composite_term=_ps.PROGRESS_TERM_DEFAULT) -> dict:
    """Inject a controlled degradation and MEASURE whether ``axis`` separates.

    Returns, for each rung of the ladder:

    * the axis value and its **paired** delta against the null rung
      (``paired_episode_cluster_bootstrap``, same rows, same episodes);
    * ``defined_frac`` for the axis — ⭐ the structural-artefact detector. A
      *slowdown* once raised the composite ``+0.1698`` because a barely-moving
      plan scores ``recovery = NaN`` **by construction**: the arm left the
      denominator rather than beating it. That is invisible in a mean and
      obvious in ``defined_frac``.
    * the **composite** at the same rung, so an axis that moves while the
      composite does not (or vice versa) is a reported fact, not a surprise;
    * ``lat_bias_m``, the signed lateral mean, so a re-centring control is visible.

    and, over the whole ladder:

    * ``monotone_*`` — is the response monotone away from the null, per side;
    * ⭐ ``mde_*`` — the **smallest |level − null| that separates**, in the
      ladder's own physical unit. This is the number a plan step must be costed
      against: ⛔ *a step whose MDE exceeds the effect it must detect is not a
      step.*
    """
    n_boot = _ci.DEFAULT_N_BOOT if n_boot is None else int(n_boot)
    spec = ladder_levels(control)
    lv = tuple(spec["levels"] if levels is None else levels)
    null = float(spec["null"])
    if null not in [float(x) for x in lv]:
        raise ValueError(f"ladder for {control!r} must contain its null {null}")
    eid = np.asarray([str(x) for x in eid])

    def _score(p):
        a = axes(p, t_tol=t_tol, d_tol=d_tol, psi_tol=psi_tol, s_ref=s_ref)
        sc = {k: np.asarray(a[k], float) for k in (axis,) + tuple(also)}
        comps = {"ego_progress": np.asarray(a["ego_progress"], float),
                 "recovery": np.asarray(a["recovery"], float)}
        num = np.zeros(comps["ego_progress"].shape)
        den = np.zeros_like(num)
        for nm, wt in (("ego_progress", 5.0), ("recovery", 5.0)):
            v = comps[nm]
            fin = np.isfinite(v)
            num = num + np.where(fin, v * wt, 0.0)
            den = den + np.where(fin, wt, 0.0)
        sc["_composite"] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        return sc

    base = _score(pw)
    rungs = []
    for L in lv:
        s = _score(apply_control(pw, control, L, seed=seed))
        v = s[axis]
        fin = np.isfinite(v)
        row = {"level": float(L),
               "axis_mean": (round(float(np.nanmean(v)), 6)
                             if fin.any() else None),
               "axis_defined_frac": round(float(fin.mean()), 6)}
        for extra in ("_composite",) + tuple(also):
            e = s[extra]
            row[extra if extra != "_composite" else "composite_progress_recovery"] = (
                round(float(np.nanmean(e)), 6) if np.isfinite(e).any() else None)
            if extra == "recovery":
                row["recovery_defined_frac"] = round(
                    float(np.isfinite(e).mean()), 6)
        if float(L) != null:
            m = np.isfinite(v) & np.isfinite(base[axis])
            row["paired_vs_null"] = (
                _ci.paired_episode_cluster_bootstrap(
                    v[m], base[axis][m], list(eid[m]), n_boot=n_boot, seed=seed)
                if m.sum() >= 2 and len(set(eid[m])) >= 2 else None)
            row["n_rows_paired"] = int(m.sum())
        rungs.append(row)

    # ---- monotonicity + MDE, per side ------------------------------------- #
    def _side(sign):
        s = [r for r in rungs if (r["level"] - null) * sign > 0]
        s.sort(key=lambda r: abs(r["level"] - null))
        return s

    out = {"control": control, "axis": axis, "unit": spec["unit"],
           "null_level": null, "two_sided_ladder": bool(spec["two_sided"]),
           "zero_mean_control": control in ZERO_MEAN_CONTROLS,
           "higher_is_better": AXIS_META.get(axis, {}).get("higher_is_better"),
           "n_boot": n_boot, "estimator": "paired_episode_cluster_bootstrap",
           "refused_estimator": "overlapping_holdout_se",
           "rungs": rungs}
    for tag, sign in (("up", +1.0), ("down", -1.0)):
        s = _side(sign)
        if not s:
            out[f"mde_{tag}"] = None
            out[f"monotone_{tag}"] = None
            out[f"separates_{tag}"] = None
            continue
        vals = [r["axis_mean"] for r in s if r["axis_mean"] is not None]
        base_v = next((r["axis_mean"] for r in rungs
                       if r["level"] == null), None)
        seq = ([base_v] + vals) if base_v is not None else vals
        # "higher is better" ⇒ a degradation must make the axis go DOWN
        out[f"monotone_{tag}"] = bool(
            all(b <= a + 1e-12 for a, b in zip(seq, seq[1:]))) if len(seq) > 1 \
            else None
        sep = [r for r in s if (r.get("paired_vs_null") or {}).get("separated")]
        out[f"separates_{tag}"] = bool(sep)
        out[f"mde_{tag}"] = (round(abs(sep[0]["level"] - null), 6)
                             if sep else None)
        # ⭐ MONOTONICITY FROM THE MDE OUTWARD — the clause :func:`admit` uses.
        # Inside the MDE the response is BY DEFINITION not separated from zero,
        # so its sign is not readable and a sign flip there is noise plus the
        # arm's own bias, not a property of the metric. MEASURED: on
        # `cv_holdv0` — which drives STRAIGHT while the logged road curves — a
        # 0.01 m/m drift partially CORRECTS the arm on some rows, so the
        # full-ladder monotonicity flips at the first rung while every rung
        # beyond the MDE is clean. Both are reported; only this one gates.
        beyond = [r["axis_mean"] for r in s
                  if out[f"mde_{tag}"] is not None
                  and abs(r["level"] - null) >= out[f"mde_{tag}"] - 1e-12
                  and r["axis_mean"] is not None]
        out[f"monotone_beyond_mde_{tag}"] = (
            bool(all(b <= a + 1e-12 for a, b in zip(beyond, beyond[1:])))
            if len(beyond) > 1 else (True if beyond else None))
        out[f"mde_{tag}_note"] = (
            "smallest |level - null| whose PAIRED episode-cluster bootstrap CI "
            f"excludes 0 (B={n_boot}, unit = val episode). None = the axis did "
            "NOT separate at any rung on this side — the ladder bounds the MDE "
            "from below at the largest rung tested.")
    out["both_directions_separate"] = (
        None if not spec["two_sided"]
        else bool(out.get("separates_up") and out.get("separates_down")))
    return out


def admit(demos, *, require_zero_mean=True) -> dict:
    """⛔ THE ADMISSION RULE. An axis may rank only if its range was DEMONSTRATED.

    ``demos`` is ``[dynamic_range(...) node, …]`` for ONE axis. The rules, each
    with the failure it prevents:

    1. **It must separate at all** — otherwise it is a C13 instrument
       (structurally unable to report the answer it is cited for).
    2. **On a two-sided ladder it must separate on BOTH sides.** ``clamp_v1``
       passed every adversary it had (``stand_still``, ``v4_blind``,
       ``v1_ego_half`` — all too *slow*) and was blind above ratio 1.0 for its
       whole life.
    3. **At least one separating control must be ZERO-MEAN** (``require_zero_mean``)
       — a constant-sign control can move an aggregate by re-centring a bias
       rather than by degrading anything, which has happened here.
    4. **Its response must be monotone away from the null** on every side that
       separates. A non-monotone response means the ladder is measuring
       something other than the injected quantity.

    Raises :class:`AxisNotDemonstrated` when the axis is not admissible, so the
    refusal cannot be downgraded to a report line.
    """
    if not demos:
        raise AxisNotDemonstrated("no dynamic-range demonstration was supplied")
    axis = demos[0]["axis"]
    sep_any, both_ok, zm_ok, mono_ok, reasons = False, True, False, True, []
    for d in demos:
        s_up, s_dn = d.get("separates_up"), d.get("separates_down")
        if s_up or s_dn:
            sep_any = True
            if d.get("zero_mean_control"):
                zm_ok = True
        if d["two_sided_ladder"] and (s_up or s_dn) and not (s_up and s_dn):
            both_ok = False
            reasons.append(
                f"{d['control']}: separates on ONE side only "
                f"(up={s_up}, down={s_dn}) — a one-sided validation is exactly "
                f"how clamp_v1 shipped blind above ratio 1.0")
        for tag in ("up", "down"):
            key = (f"monotone_beyond_mde_{tag}"
                   if f"monotone_beyond_mde_{tag}" in d
                   else f"monotone_{tag}")
            if d.get(f"separates_{tag}") and d.get(key) is False:
                mono_ok = False
                reasons.append(
                    f"{d['control']}: non-monotone on the {tag} side, BEYOND "
                    f"its own MDE — inside the MDE a sign flip is not readable "
                    f"and is not counted")
    if not sep_any:
        reasons.append("no control separated at any rung — the axis cannot fail")
    if require_zero_mean and not zm_ok:
        reasons.append(
            "no ZERO-MEAN control separated; every separating control has a "
            "constant sign and could be re-centring a bias rather than "
            "degrading the axis (MEASURED failure mode on this surface)")
    ok = bool(sep_any and both_ok and mono_ok
              and (zm_ok or not require_zero_mean))
    node = {"axis": axis, "admissible": ok, "n_demonstrations": len(demos),
            "controls": [d["control"] for d in demos],
            "mde": {d["control"]: {"up": d.get("mde_up"),
                                   "down": d.get("mde_down"),
                                   "unit": d["unit"]} for d in demos},
            "rules": ["separates at all", "separates on BOTH sides of every "
                      "two-sided ladder", "at least one ZERO-MEAN control "
                      "separates",
                      "monotone away from the null FROM THE MDE OUTWARD "
                      "(inside the MDE the response is not separated from "
                      "zero, so its sign is not readable)"],
            "reasons": reasons}
    if not ok:
        raise AxisNotDemonstrated(
            f"axis {axis!r} is NOT admissible: {'; '.join(reasons)}. "
            f"A metric that cannot fail — or that can only fail in the "
            f"direction it was built to detect — is not a metric.")
    return node


# =========================================================================== #
# the PANEL-WIDE gate and the control-aware composite                          #
# =========================================================================== #
def panel_gate(by_arm, *, probes=(), names=AXES, ceil_frac_max=None,
               range_min=None) -> dict:
    """⛔ PANEL-WIDE, and it says so. An axis is admitted only if it clears
    ``pseudosim.discriminative_range`` for **every non-probe arm**.

    The per-arm gate is REFUSED, not offered: it moved ``comfort``
    ``0.0004 → 0.2882`` (**720x**) between two arms differing only in schedule
    and flipped a stream's own primary from ``n.s.`` to ``SEPARATED``. A
    sensitivity that can only be quoted when it agrees with the author is not a
    sensitivity.
    """
    kw = {}
    if ceil_frac_max is not None:
        kw["ceil_frac_max"] = ceil_frac_max
    if range_min is not None:
        kw["range_min"] = range_min
    scores = {a: {k: np.asarray(v[k], float) for k in names}
              for a, v in by_arm.items()}
    per_arm = {a: _ps.discriminative_range(scores[a], by_arm=scores, **kw)
               for a in scores}
    gate_arms = [a for a in scores if a not in probes]
    admitted, why = {}, {}
    for k in names:
        bad = [a for a in gate_arms if not per_arm[a].get(k, {}).get("admissible")]
        admitted[k] = not bad
        why[k] = ("admissible for every non-probe arm" if not bad else
                  "INADMISSIBLE for " + ", ".join(sorted(bad))
                  + " -> dropped from EVERY arm")
    return {"gate": "PANEL-WIDE",
            "gate_refused": ("per-arm — it moved comfort 0.0004 -> 0.2882 "
                             "(720x) between two arms differing only in "
                             "schedule and flipped a primary verdict"),
            "probes_excluded": sorted(set(probes) & set(scores)),
            "admitted": {k: v for k, v in admitted.items() if v},
            "dropped": {k: why[k] for k, v in admitted.items() if not v},
            "per_arm": {a: {k: per_arm[a].get(k, {}).get("admissible")
                            for k in names} for a in scores},
            "detail": per_arm}


def control_score(a, admitted, *, weights=None, demonstrated=None) -> dict:
    """A control-aware composite over the ADMITTED, DEMONSTRATED axes only.

    ⛔ Not a Driving Score, for the same reason ``pseudosim.composite`` is not:
    **there is no collision gate and no drivable-area term** on this surface
    (:data:`MISSING_GATES`). It scores *progress, recovery and control*.

    ⛔ It also refuses to include an axis whose dynamic range was never
    demonstrated: pass ``demonstrated`` (the set of axis names that cleared
    :func:`admit`) and anything outside it is dropped with a reason.
    """
    w = dict(CONTROL_WEIGHTS if weights is None else weights)
    use, dropped = {}, {}
    for k, wt in w.items():
        if wt == 0:
            dropped[k] = "weight 0"
        elif k not in a:
            dropped[k] = "not computed"
        elif not admitted.get(k, k in ("ego_progress", "recovery")):
            dropped[k] = "dropped by the PANEL-WIDE range gate"
        elif demonstrated is not None and k not in demonstrated \
                and k not in ("ego_progress", "recovery"):
            dropped[k] = ("no demonstrated dynamic range — refused by "
                          "control.admit")
        else:
            use[k] = wt
    if not use:
        raise _ps.VacuousMetric(
            f"REFUSING TO EMIT a control score: no axis is both admitted and "
            f"demonstrated. Dropped: {dropped}")
    ref = np.asarray(a[next(iter(use))], float)
    num, den = np.zeros_like(ref), np.zeros_like(ref)
    for k, wt in use.items():
        v = np.asarray(a[k], float)
        fin = np.isfinite(v)
        num = num + np.where(fin, v * wt, 0.0)
        den = den + np.where(fin, wt, 0.0)
    return {"name": f"CONTROL_{SUITE_ID()}",
            "weights_admitted": use, "weights_dropped": dropped,
            "_not_a_driving_score": MISSING_GATES["_not_a_driving_score"],
            "value": np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)}


# =========================================================================== #
# ⛔ the two REFUSALS this suite is obliged to state, not paper over            #
# =========================================================================== #
#: MEASURED, and it is why ``comfort`` may not carry weight. Filled from the
#: audit in ``…/incoming/2026-07-28-closedloop-control-suite/raw/comfort_audit.json``.
COMFORT_STATUS = (
    "⛔ INFORMATION-FREE ON THIS SURFACE, AND THE REASON IS NOT THE PLANS. "
    "`comfort` is the AND of four bounds on the plan's finite differences. "
    "MEASURED here: the HUMAN'S OWN LOGGED PATH violates the identical bounds "
    "on the overwhelming majority of the same windows, so the term does not "
    "separate good driving from bad — it separates 10 Hz differentiation noise "
    "from smoothness. Between two arms that differ ONLY in schedule it moves "
    "720x (0.0004 -> 0.2882), because re-timing at constant speed SMOOTHS a "
    "per-waypoint regression. It is dropped by the panel-wide gate for every "
    "arm and under every progress term. ⇒ its weight is 0.0 and the composite "
    "is TWO-term, not three. The measurement is kept and reported as "
    "`plan_smoothness_flag`, which is what it actually is.")

MISSING_GATES = {
    "no_collision": _ps.COLLISION_UNAVAILABLE_REASON,
    "no_drivable_area": (
        "IMPOSSIBLE, not merely missing. PhysicalAI-AV has no map, lane graph, "
        "junction annotation, roundabout label, traffic-light feature or "
        "route/goal signal — the card says verbatim 'we do not include open "
        "maps data', and `obstacle.offline`'s enum over 87,481 cuboids is 10 "
        "classes, ALL DYNAMIC AGENTS. Settled at five independent probes. "
        "`egomotion` also carries no lat/lon/GNSS, so OSM map-matching on our "
        "traces is impossible. ⇒ DAC / lane-keeping / driving-direction / "
        "traffic-light compliance cannot be computed here and are NOT faked."),
    "_not_a_driving_score": (
        "⛔ With no collision gate and no drivable-area term, NOTHING computed "
        "on this surface is a Driving Score and none of it may be compared to "
        "a PDMS number. It scores PROGRESS, RECOVERY and CONTROL."),
    "_what_would_unblock_the_collision_gate": (
        "`obstacle.offline` covers 97.4438 % of the corpus with boxes in the "
        "rig frame. TWO blockers, both mechanical: (a) the cached `episode_id` "
        "is int.from_bytes(clip_id[:4]) and COLLIDES — 242 clip_index rows map "
        "onto the 40 val episode_ids — so episode->clip identity is not "
        "resolvable from the cache alone; (b) the matching chunks are not "
        "downloaded. Neither is a research problem."),
}


# =========================================================================== #
# THE BLOCK                                                                    #
# =========================================================================== #
def block(pw, *, arm="unknown", eid=None, t_tol=None, d_tol=None, psi_tol=None,
          s_ref=None, n_boot=None, seed=0, sensitivity=False) -> dict:
    """Full control block for one arm's pseudo-simulation dump."""
    n_boot = _ci.DEFAULT_N_BOOT if n_boot is None else int(n_boot)
    eid = list(pw["eid"]) if eid is None else list(eid)
    a = axes(pw, t_tol=t_tol, d_tol=d_tol, psi_tol=psi_tol, s_ref=s_ref)
    out = {
        "block": BLOCK, "version": VERSION, "arm": arm,
        "suite_id": a["_suite_id"], "tolerances": a["_tolerances"],
        "axis_meta": AXIS_META,
        "n_rows": int(len(eid)), "n_episodes": int(len(set(eid))),
        "row_mask_rule": a["_row_mask_rule"],
        "estimator": (f"taniteval.ci.episode_cluster_bootstrap (B={n_boot}, "
                      f"unit = val episode); paired form for two arms"),
        "refused_estimator": ("overlapping_holdout_se — it biases the POINT "
                              "ESTIMATE as well as the interval"),
        "traffic_mode": _ps.TRAFFIC_MODE_LOG_REPLAY,
        "comfort_status": COMFORT_STATUS,
        "missing_gates": MISSING_GATES,
        "axes": axis_summary(a, eid, n_boot=n_boot, seed=seed),
    }
    if sensitivity:
        sens = {}
        for tt in TOL_SENSITIVITY["t_tol_s"]:
            sens[f"t_tol_s={tt:g}"] = round(float(np.nanmean(
                axes(pw, t_tol=tt)["lon_track"])), 6)
        for dd in TOL_SENSITIVITY["d_tol_m"]:
            sens[f"d_tol_m={dd:g}"] = round(float(np.nanmean(
                axes(pw, d_tol=dd)["lat_track"])), 6)
        for sr in TOL_SENSITIVITY["s_ref_m"]:
            sens[f"s_ref_m={sr:g}"] = round(float(np.nanmean(
                axes(pw, s_ref=sr)["lat_track"])), 6)
        for pp in TOL_SENSITIVITY["psi_tol_rad"]:
            sens[f"psi_tol_rad={pp:g}"] = round(float(np.nanmean(
                axes(pw, psi_tol=pp)["lat_heading"])), 6)
        out["tolerance_sensitivity"] = sens
    return out
