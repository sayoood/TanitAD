"""taniteval.pseudosim — PSEUDO-SIMULATION: the protocol that makes our
closed-loop numbers a MEASUREMENT instead of an EXTRAPOLATION.

THE DEFECT THIS MODULE REMOVES (MEASURED, 2026-07-26/27)
--------------------------------------------------------
The sequential closed loop (:mod:`taniteval.clhorizon`) lets the ego walk
arbitrarily far from the logged pose, and the further it walks the less the
substrate can honestly re-render. Out-of-envelope window fractions:
``K=20 -> 12.3 %``, ``K=60 -> 50.7 %``, ``K=185 -> 90.2 %``. The **last horizon
at which the closed loop is a pure MEASUREMENT is 0.4 s (K=4)**, and
``GATE_PROTOCOL`` §0.3 refuses ``K <= 20`` — so **no admissible gate horizon is a
measurement**. Widening cannot rescue it: at yaw ``= inf`` the *lateral* clause
alone still leaves 3.75 % of K=20 windows outside, and ``MEASUREMENT`` requires
zero.

**The cause is ACCUMULATION**, not fidelity. Pseudo-simulation (NAVSIM v2,
*Pseudo-Simulation for Autonomous Driving*, arXiv 2506.04218, CoRL 2025) removes
the mechanism: it **pre-generates a BOUNDED GRID of perturbed observations
before evaluation and never rolls out sequentially**. Deviation is therefore
**CHOSEN, not accumulated**, and **cannot leave the validated envelope by
construction**.

    ``PUBLISHED`` Their protocol reports ``R^2 = 0.80 (r = 0.89), n = 83
    planners`` against nuPlan closed-loop, vs ``R^2 = 0.70 (r = 0.83)`` for the
    best open-loop method. We re-implement the PROTOCOL on our own data and our
    own warp; **no NAVSIM code and no NAVSIM data is vendored** (their code is
    Apache-2.0 but their data is CC-BY-NC-SA).

WHAT THE GRID'S AXES ARE, AND WHY ONE OF THEM IS REFUSED
--------------------------------------------------------
=========================  =========================================  ==========
axis                       substrate                                  status
=========================  =========================================  ==========
**heading** ``dpsi``       ``H = K R K^-1`` — a pure camera rotation   **USED**
                           is exact for ARBITRARY scene depth
                           (``max|dH| = 0.000e+00``, 30 conditions)
**longitudinal** ``dlon``  index offset along the logged path —        **USED**
                           REAL frames, zero synthesis
**lateral** ``dlat``       ground-plane homography under a             ⛔ **REFUSED**
                           FLAT-ROAD assumption
=========================  =========================================  ==========

⛔ **The lateral axis is refused on MEASURED geometry**, not on taste
(``…/incoming/2026-07-27-pseudo-simulation/artifacts/lat_warp_fidelity.json``):
the flat-road warp's relative displacement error is **exactly
``height_above_road / h_cam``** — independent of depth, of ``|dlat|`` and of
focal length. At the camera height (1.50 m) it is **100 %**: a sedan roof that
should move 35.47 px moves **1.18 px**. Above 1.50 m the applied displacement is
**SIGN-INVERTED** (truck roof ``-1.667x``, building ``-4.0x``), and **exactly
50 % of a 256-row frame at pitch 0 lies above the horizon**, where the ground
plane has no preimage at all. At ``|dlat| = 2.0 m`` only **28.3 %** of in-frame
scene points meet the pre-registered ``rel_err < 0.25`` bar against a required
95 %. ⇒ **outcome L-BAD; a 1-D warped axis that is a MEASUREMENT beats a 2-D one
that is an extrapolation.** :class:`GridSpec` therefore raises
:class:`LateralAxisRefused` unless a caller passes an explicit override *and* a
written reason.

WHAT IS ENFORCED HERE (the assertion is the whole point)
---------------------------------------------------------
* :func:`assert_grid_in_envelope` runs on **every** grid, inside
  :func:`pseudo_evaluate`, **before any model is touched**. It is a HARD
  failure (:class:`EnvelopeViolation`), never a report line.
* It **can** fail, and the value that makes it fail is published in its own
  output (``falsifier``): any ``|dpsi| > 12.0 deg`` or ``|dlat| > 3.0 m``. The
  test suite exercises exactly that input.
* ⚠️ **Every emitted node carries ``traffic_mode``.** Our AlpaSim ``trafficsim``
  is disabled (``skip: true``) ⇒ **LITERAL REPLAY**, so every published TanitAD
  closed-loop number ran against non-reactive replayed traffic. That was
  nowhere on record until 2026-07-27. It is on record in every result this
  module emits. *(Reactive traffic is NOT the blocker: NAVSIM v2 itself uses
  rule-based agents on road centrelines, and our own CATK probe measured NOT
  reactive — 155 agents moved 4 mm when a car braked to a dead stop.)*
* :func:`composite` **REFUSES TO EMIT** when no weighted component clears the
  discriminative-range gate (:class:`VacuousMetric`). Comfort is saturated at
  ``>= 99.9 %`` in the published cross-benchmark study — *"essentially zero
  discriminative information"* — and this program has shipped three vacuous
  diagnostics already.
* ⛔ Collision / TTC are **NOT COMPUTABLE** on the current val cache and are
  therefore emitted as ``None`` with a reason, **never as a constant**. See
  :data:`COLLISION_UNAVAILABLE_REASON`.

ESTIMATOR. ``taniteval.ci.episode_cluster_bootstrap`` (``B=2000``), unit = val
episode, paired form for two arms on the same windows.
``overlapping_holdout_se`` appears nowhere — it biases the POINT ESTIMATE as
well as the interval.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from taniteval import ci as _ci
from taniteval import ood as _ood
from taniteval.clhorizon import (DT, LOOKAHEAD_STEP, W as WINDOW,  # noqa: F401
                                 sampling_homography, warp_batch, wrap_angle)
from taniteval.ood import ENV_LAT_MAX, ENV_YAW_MAX

__all__ = [
    "TRAFFIC_MODE_LOG_REPLAY", "TRAFFIC_MODE_NOTE", "PROTOCOL",
    "LATERAL_REFUSAL", "COLLISION_UNAVAILABLE_REASON",
    "EnvelopeViolation", "LateralAxisRefused", "VacuousMetric",
    "GridSpec", "default_grid", "assert_grid_in_envelope",
    "proximity_weights", "pseudo_evaluate", "score_windows",
    "discriminative_range", "composite", "emit",
    "COMFORT_LIMITS", "COMPONENT_WEIGHTS", "CEIL_FRAC_MAX", "RANGE_MIN",
]

# --------------------------------------------------------------------------- #
# provenance strings that must ride along with every number                    #
# --------------------------------------------------------------------------- #
TRAFFIC_MODE_LOG_REPLAY = "log_replay_nonreactive"
TRAFFIC_MODE_NOTE = (
    "Other agents are LOGGED TRACKS REPLAYED WITHOUT REACTION. Our AlpaSim "
    "`trafficsim` is disabled (`skip: true`), i.e. literal replay — so EVERY "
    "published TanitAD closed-loop number, not just this one, ran against "
    "non-reactive traffic. Paired arm-vs-arm comparisons remain valid; absolute "
    "safety numbers are optimistic. NAVSIM v2 (the protocol being reproduced) "
    "also uses non-reactive rule-based agents on road centrelines, and our own "
    "CAT-K reactivity probe measured NOT REACTIVE (155 agents displaced 0.0044 "
    "m when a lead car braked to a dead stop). Reactive traffic is therefore "
    "NOT a precondition for this protocol.")

PROTOCOL = (
    "PSEUDO-SIMULATION. Perturbed observation states are PRE-GENERATED on a "
    "bounded grid and the planner is evaluated ONCE from each. There is no "
    "sequential rollout, so deviation is CHOSEN, not accumulated, and cannot "
    "leave the validated envelope. Re-implemented from the published protocol "
    "(NAVSIM v2 / arXiv 2506.04218, CoRL 2025) on OUR data and OUR warp; no "
    "NAVSIM code or data is vendored.")

LATERAL_REFUSAL = (
    "The LATERAL grid axis is REFUSED on MEASURED geometry. The flat-road "
    "homography's relative displacement error is exactly height_above_road / "
    "h_cam (depth-free, |dlat|-free, f-free): 100 % at the camera height, "
    "SIGN-INVERTED above it, and 50 % of the frame at pitch 0 is above the "
    "horizon where the ground plane has no preimage. At |dlat| = 2.0 m only "
    "28.3 % of in-frame points meet the pre-registered rel_err < 0.25 bar "
    "(required: 95 %). Artifact: TanitAD Research Hub/Benchmarks & Eval/"
    "Implementation/incoming/2026-07-27-pseudo-simulation/artifacts/"
    "lat_warp_fidelity.json (outcome L-BAD). The YAW axis passes the identical "
    "test at max error 0.0 px, so the test is not vacuous.")

COLLISION_UNAVAILABLE_REASON = (
    "NOT COMPUTABLE on the 40-episode val cache. The cache carries only "
    "{frames_u8, actions, poses, maneuvers, episode_id} — no agent cuboids. "
    "`obstacle.offline` (97.4438 % corpus coverage, boxes reference_frame='rig' "
    "on 100 %) would supply them, but (a) the cached `episode_id` is "
    "int.from_bytes(clip_id[:4]) and COLLIDES — 242 clip_index rows map onto the "
    "40 val episode_ids, so episode->clip identity is not resolvable from the "
    "cache alone, and (b) the matching obstacle.offline chunks are not "
    "downloaded. A constant is NOT emitted in its place: a metric that cannot "
    "fail is not a metric.")

# --------------------------------------------------------------------------- #
# the discriminative-range gate (BOOST M8 / C13 applied to METRICS)             #
# --------------------------------------------------------------------------- #
# PROPOSED thresholds, stated before any component is scored.
CEIL_FRAC_MAX = 0.95     # a component pinned at its ceiling this often is dead
RANGE_MIN = 0.05         # observed max - min below this is not a range
# nuPlan/NAVSIM-style comfort bounds. PROPOSED (their exact constants are not
# quotable from the material we verified); every one is published in the output.
COMFORT_LIMITS = {"a_lon_max_mps2": 3.0, "a_lat_max_mps2": 3.0,
                  "jerk_max_mps3": 8.0, "yaw_rate_max_radps": 0.95}
# PDM-Score weights: EP w=5, TTC w=5, Comfort w=2 (PUBLISHED, NAVSIM). TTC is
# unavailable here (no cuboids); RECOVERY is ours and carries the error-recovery
# signal pseudo-simulation exists to produce.
COMPONENT_WEIGHTS = {"ego_progress": 5.0, "recovery": 5.0, "comfort": 2.0}


class EnvelopeViolation(AssertionError):
    """A grid point lies outside the MEASURED envelope.

    This is an ERROR, not a warning. The entire claim of this protocol is that
    the out-of-envelope fraction is **0 by construction**; a grid that breaks it
    silently would reproduce the exact defect (a too-generous string surviving
    because nobody read the field beside it) in a new costume."""


class LateralAxisRefused(AssertionError):
    """The lateral axis was requested without an explicit, reasoned override."""


class VacuousMetric(AssertionError):
    """No weighted component has usable dynamic range, so no composite is emitted."""


# =========================================================================== #
# the grid                                                                     #
# =========================================================================== #
@dataclass(frozen=True)
class GridSpec:
    """A BOUNDED, PRE-GENERATED set of ego-state perturbations.

    ``dyaw_deg``   heading offsets applied by ``sampling_homography(0, dpsi)`` —
                   geometrically exact for arbitrary depth.
    ``dlon_steps`` frame-index offsets along the logged path. The observation is
                   the REAL frame window at that index: zero synthesis, and the
                   (dlat, dpsi) envelope coordinates of the point are unchanged.
    ``dlat_m``     ⛔ refused by default; see :data:`LATERAL_REFUSAL`.
    """

    dyaw_deg: tuple = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)
    dlon_steps: tuple = (-10, 0, 10)
    dlat_m: tuple = (0.0,)
    allow_lateral: bool = False
    lateral_override_reason: str = ""
    _meta: dict = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if any(abs(float(v)) > 0.0 for v in self.dlat_m):
            if not self.allow_lateral or not self.lateral_override_reason.strip():
                raise LateralAxisRefused(LATERAL_REFUSAL)

    def points(self):
        """``[(dlat_m, dyaw_deg, dlon_steps), …]`` in a deterministic order."""
        return [(float(a), float(y), int(s))
                for a in self.dlat_m for y in self.dyaw_deg
                for s in self.dlon_steps]

    @property
    def n_points(self):
        return len(self.dlat_m) * len(self.dyaw_deg) * len(self.dlon_steps)

    def describe(self):
        return {
            "dlat_m": list(self.dlat_m), "dyaw_deg": list(self.dyaw_deg),
            "dlon_steps": list(self.dlon_steps), "n_points": self.n_points,
            "lateral_axis": ("ENABLED (override: "
                             + self.lateral_override_reason + ")"
                             if self.allow_lateral and
                             any(abs(v) > 0 for v in self.dlat_m)
                             else "REFUSED"),
            "lateral_refusal": LATERAL_REFUSAL,
            "warped_axes": ["heading"],
            "unwarped_axes": ["longitudinal (real frames at an index offset)"],
        }


def default_grid(**kw):
    """The shipped grid: heading x longitudinal, lateral refused.

    Heading spans the full MEASURED envelope ``|dpsi| <= 12 deg`` — the widest
    grid that is still 0 % out of envelope. NAVSIM v2's own heading filter is
    20 deg, so **ours is NARROWER than theirs**; that is a disclosable
    limitation, not a defect: our 12 deg is measured, their 20 deg is chosen."""
    return GridSpec(**kw)


def assert_grid_in_envelope(grid, *, _path="pseudosim.grid") -> dict:
    """⭐ THE ASSERTION. 0 % out of envelope, or the run does not start.

    Returns the proof (fractions + verdict + the falsifier), and raises
    :class:`EnvelopeViolation` otherwise. Run **before** any model is loaded, so
    a bad grid costs zero GPU seconds.
    """
    pts = grid.points()
    lat = np.array([[abs(p[0])] for p in pts], dtype=float)      # [n_pts, 1]
    yaw = np.array([[abs(p[1])] for p in pts], dtype=float)
    frac = _ood.envelope_fractions(lat, yaw)
    fw = frac["frac_windows_any_step_out_of_envelope"]
    fs = frac["frac_steps_any"]
    proof = {
        "n_grid_points": len(pts),
        "max_abs_dlat_m": float(lat.max()), "max_abs_dyaw_deg": float(yaw.max()),
        "envelope": frac["envelope"],
        "EXTRAPOLATION_frac_steps_lat_over_3m": frac["frac_steps_lat_over_3m"],
        "EXTRAPOLATION_frac_steps_yaw_over_12deg": frac["frac_steps_yaw_over_12deg"],
        "EXTRAPOLATION_frac_steps_any": fs,
        "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": fw,
        "ood_peak_ratio": None,
        "ratio_is_lower_bound": bool(fs > 0.0 or fw > 0.0),
        "EXTRAPOLATION_VERDICT": _ood._verdict_string(False, fw, fs),
        "why_zero_by_construction": (
            "Deviation is the GRID, not an accumulated rollout state. The "
            "protocol has no mechanism by which a window can drift outside the "
            "envelope, because no window ever advances."),
        "falsifier": {
            "_what": "the assertion CAN fail; these are the values that fail it",
            "smallest_failing_abs_dyaw_deg": ENV_YAW_MAX + 1e-9,
            "smallest_failing_abs_dlat_m": ENV_LAT_MAX + 1e-9,
            "example": (f"GridSpec(dyaw_deg=({ENV_YAW_MAX + 0.5},)) raises "
                        f"EnvelopeViolation; it is exercised in "
                        f"taniteval/tests/test_pseudosim.py."),
        },
    }
    if fw > 0.0 or fs > 0.0:
        raise EnvelopeViolation(
            f"[{_path}] {fs:.4%} of grid points / {fw:.4%} of grid rows lie "
            f"OUTSIDE the MEASURED envelope (|dlat| <= {ENV_LAT_MAX} m, |dyaw| "
            f"<= {ENV_YAW_MAX} deg). Pseudo-simulation's entire claim is that "
            f"this fraction is 0 BY CONSTRUCTION. Shrink the grid; do not widen "
            f"the envelope — P1 proved arithmetically that widening cannot reach "
            f"0 for a sequential rollout, and for a grid it is simply a choice.")
    # belt and braces: the same consistency check the OOD guard applies
    _ood.assert_envelope_verdict_consistent(proof, _path=_path)
    return proof


def proximity_weights(pts, *, yaw_sigma_deg=8.0, lon_sigma_steps=12.0):
    """NAVSIM v2's proximity weighting, on our axes.

    *"A proximity-based weighting scheme assigns higher importance to synthetic
    observations that best match the AV's likely behaviour."* Implemented as a
    separable Gaussian kernel on the perturbation magnitude, normalised to sum 1.

    ⚠️ A weighting choice must not be able to manufacture a result, so
    :func:`emit` always reports the **unweighted** aggregate beside the weighted
    one and flags any disagreement in sign.
    """
    y = np.array([p[1] for p in pts], dtype=float)
    s = np.array([p[2] for p in pts], dtype=float)
    w = np.exp(-0.5 * (y / yaw_sigma_deg) ** 2) * \
        np.exp(-0.5 * (s / lon_sigma_steps) ** 2)
    return w / w.sum()


# =========================================================================== #
# the evaluation — ONE planner call per (window, grid point). No rollout.       #
# =========================================================================== #
def _default_frames(ep, a, b):
    fr = ep.frames[a:b] if hasattr(ep, "frames") else ep["frames_u8"][a:b]
    fr = torch.as_tensor(fr)
    return fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()


def _poses_of(ep):
    p = ep.poses if hasattr(ep, "poses") else ep["poses"]
    return torch.as_tensor(p, dtype=torch.float32)


@torch.no_grad()
def pseudo_evaluate(planner, episodes, grid, *, device="cpu", stride=8,
                    window=WINDOW, horizon=20, frames_of=None, goals=None,
                    batch=16, verbose=False) -> dict:
    """Evaluate ``planner`` at every (anchor, grid point). **No rollout.**

    For each anchor ``a`` and grid point ``(dlat, dpsi, dlon)``:

    1. take the **real** frame window ``[a + dlon, a + dlon + window)``;
    2. warp it **once** by ``sampling_homography(dlat, dpsi)``;
    3. ask the planner for a trajectory (ONE call, in the ego frame of the
       perturbed pose);
    4. record it. Nothing is fed back; step 1 is never re-entered.

    Because the observation is synthesised at the *chosen* deviation and the
    loop never advances, the deviation of every evaluated state is exactly the
    grid value — which is why :func:`assert_grid_in_envelope` can guarantee 0 %.

    Returns per-(window, grid point) arrays plus the envelope proof. Scoring is
    :func:`score_windows`; keeping them separate means any metric can be
    re-derived from the dump with **no GPU** — the arithmetic-only path whose
    absence forced five closed-loop artifacts to be re-driven in July.
    """
    proof = assert_grid_in_envelope(grid)          # BEFORE any model touch
    frames_of = frames_of or _default_frames
    pts = grid.points()
    lon_lo, lon_hi = min(p[2] for p in pts), max(p[2] for p in pts)

    rec = {k: [] for k in ("eid", "anchor", "pt_dlat", "pt_dyaw", "pt_dlon",
                           "v0", "ep_i")}
    trajs, ref_paths, ref_yaw = [], [], []
    n_calls = 0
    for ep_i, ep in enumerate(episodes):
        poses = _poses_of(ep)
        T = int(poses.shape[0])
        lo = max(0, -lon_lo)
        hi = T - window - horizon - max(0, lon_hi)
        anchors = list(range(lo, hi, stride))
        if not anchors:
            continue
        for (dlat, dyaw, dlon) in pts:
            Hm = sampling_homography(dlat, dyaw)
            for bi in range(0, len(anchors), batch):
                ch = anchors[bi:bi + batch]
                s = torch.tensor([a + dlon for a in ch])
                last = s + window - 1
                fw = torch.stack([frames_of(ep, int(x), int(x) + window)
                                  for x in s]).to(device)
                fw = warp_batch(fw, Hm[None].expand(len(ch), 3, 3).clone())
                v0 = poses[last, 3]
                g = (goals.get(ep_i, last.numpy(), device)
                     if goals is not None else None)
                tj = planner.traj(fw, v0.to(device), g)[:, :horizon].cpu().float()
                n_calls += len(ch)
                trajs.append(tj)
                # the logged path AHEAD of the observed reference pose, in world
                idx = last[:, None] + torch.arange(0, horizon + 1)[None]
                ref_paths.append(poses[idx][..., :2])
                ref_yaw.append(poses[last, 2])
                rec["eid"] += [str(ep_i)] * len(ch)
                rec["anchor"].append(torch.tensor(ch))
                rec["ep_i"].append(torch.full((len(ch),), ep_i))
                rec["v0"].append(v0)
                rec["pt_dlat"].append(torch.full((len(ch),), float(dlat)))
                rec["pt_dyaw"].append(torch.full((len(ch),), float(dyaw)))
                rec["pt_dlon"].append(torch.full((len(ch),), float(dlon)))
        if verbose:
            print(f"    [pseudosim] ep {ep_i + 1}/{len(episodes)} "
                  f"planner calls so far {n_calls}", flush=True)
    if not rec["eid"]:
        return {"_empty": True, "envelope_proof": proof,
                "traffic_mode": TRAFFIC_MODE_LOG_REPLAY}
    out = {k: (v if k == "eid" else torch.cat(v)) for k, v in rec.items()}
    out["traj"] = torch.cat(trajs)                 # [n, horizon, 2] ego frame
    out["ref_path"] = torch.cat(ref_paths)         # [n, horizon+1, 2] world
    out["ref_yaw"] = torch.cat(ref_yaw)            # [n] world heading at ref
    out["envelope_proof"] = proof
    out["grid"] = grid.describe()
    out["horizon_steps"] = int(horizon)
    out["horizon_s"] = round(horizon * DT, 2)
    out["traffic_mode"] = TRAFFIC_MODE_LOG_REPLAY
    out["traffic_mode_note"] = TRAFFIC_MODE_NOTE
    out["protocol"] = PROTOCOL
    out["planner_calls"] = int(n_calls)
    out["rollout_steps_executed"] = 0
    out["_no_accumulation"] = (
        "rollout_steps_executed == 0 BY CONSTRUCTION: the loop never advances, "
        "so deviation cannot accumulate and the envelope cannot be left.")
    return out


# =========================================================================== #
# the map-free composite                                                       #
# =========================================================================== #
def _cross_and_along(pw):
    """Plan endpoint expressed relative to the LOGGED path (map-free).

    The perturbed ego sits at the logged reference pose offset by ``(dlat,
    dpsi)``; the plan is in ITS ego frame. Both are lifted into the reference
    pose's frame, where component 0 is along-track and component 1 is
    cross-track. No map, no lane graph, no drivable-area polygon is used —
    PhysicalAI-AV has none (settled at five probes).
    """
    tj = pw["traj"]                                    # [n, Hh, 2] perturbed ego
    dpsi = torch.deg2rad(pw["pt_dyaw"])
    dlat = pw["pt_dlat"]
    c, s = torch.cos(dpsi), torch.sin(dpsi)
    # perturbed-ego -> reference-ego: rotate by dpsi, then offset by dlat on y
    x = c[:, None] * tj[..., 0] - s[:, None] * tj[..., 1]
    y = s[:, None] * tj[..., 0] + c[:, None] * tj[..., 1] + dlat[:, None]
    # the logged path in the reference-ego frame
    rp = pw["ref_path"]
    ryaw = pw["ref_yaw"]
    dx = rp[..., 0] - rp[:, :1, 0]
    dy = rp[..., 1] - rp[:, :1, 1]
    cr, sr = torch.cos(ryaw)[:, None], torch.sin(ryaw)[:, None]
    ref_x = cr * dx + sr * dy
    ref_y = -sr * dx + cr * dy
    return x, y, ref_x, ref_y


def score_windows(pw, *, comfort_limits=None, dt=DT) -> dict:
    """Per-(window, grid point) sub-scores. Pure arithmetic — **no GPU**.

    Components, each map-free and each with its discriminative range MEASURED
    (never assumed) by :func:`discriminative_range`:

    ``ego_progress``  along-track distance the plan covers, over the along-track
                      distance the human covered on the same window, clipped to
                      [0, 1]. **PUBLISHED: the strongest single predictor of
                      closed-loop Driving Score, Spearman rho = 0.83**, ahead of
                      collision rate (0.45); ADE/L2 is **-0.36, p = 0.43**.
    ``recovery``      does the plan converge back to the logged path from the
                      perturbed state? ``1 - |xt_end| / |xt_hold_matched|``
                      clipped to [0, 1]. **This is the error-recovery signal
                      pseudo-simulation exists to produce and that open-loop ADE
                      provably does not measure.** Undefined (NaN) at the
                      unperturbed grid point, by construction.

                      ⚠️ ``xt_hold_matched`` is computed from **the plan's OWN
                      along-track distance** — ``|dlat + s_along * tan(dpsi)|`` —
                      not from ``v0 * horizon``. That is not cosmetic: MEASURED
                      on the 2-episode smoke of 2026-07-27, the naive
                      ``v0``-based denominator scored the **BLIND** arm
                      **+0.597** above the sighted one, because a planner that
                      barely moves has a small cross-track error and was being
                      paid for it. **Standing still is not recovery.** With the
                      progress-matched denominator a stopped plan yields
                      ``xt_hold -> 0`` and the score is NaN (excluded), not 1.0.
                      Exercised by
                      ``test_recovery_is_not_gameable_by_standing_still``.
    ``comfort``       all of |a_lon|, |a_lat|, |jerk|, |yaw_rate| within bounds.
                      ⚠️ **PUBLISHED as saturated at >= 99.9 %** elsewhere; it is
                      admitted here only if OUR measurement gives it range.
    ``no_collision``  ⛔ ``None`` — see :data:`COLLISION_UNAVAILABLE_REASON`.
    ``ttc``           ⛔ ``None`` — same reason.
    """
    lim = dict(COMFORT_LIMITS if comfort_limits is None else comfort_limits)
    x, y, ref_x, ref_y = _cross_and_along(pw)
    n, Hh = x.shape

    # --- ego progress ------------------------------------------------------ #
    human = torch.sqrt((ref_x[:, -1] - ref_x[:, 0]) ** 2
                       + (ref_y[:, -1] - ref_y[:, 0]) ** 2)
    ego = x[:, -1]                                    # along-track, ref frame
    ratio = ego / human.clamp_min(1e-3)
    ep_score = ratio.clamp(0.0, 1.0)
    ep_score = torch.where(human > 0.5, ep_score, torch.full_like(ep_score,
                                                                 float("nan")))

    # --- recovery ---------------------------------------------------------- #
    # cross-track of the plan endpoint from the logged path endpoint
    xt_end = (y[:, -1] - ref_y[:, -1]).abs()
    dpsi = torch.deg2rad(pw["pt_dyaw"])
    # ⚠️ PROGRESS-MATCHED denominator: the drift the plan's OWN along-track
    # distance would have produced had it not steered. Using v0 * horizon
    # instead pays a planner for standing still (MEASURED: it put the BLIND arm
    # 0.597 ABOVE the sighted one on the 2026-07-27 smoke).
    s_along = x[:, -1].clamp_min(0.0)
    xt_hold = (pw["pt_dlat"] + s_along * torch.tan(dpsi)).abs()
    rc = (1.0 - xt_end / xt_hold.clamp_min(1e-6)).clamp(0.0, 1.0)
    rc = torch.where(xt_hold > 0.10, rc, torch.full_like(rc, float("nan")))
    # the naive denominator, kept only as a diagnostic so the defect stays visible
    xt_hold_v0 = (pw["pt_dlat"]
                  + pw["v0"].abs() * (Hh * dt) * torch.sin(dpsi)).abs()

    # --- comfort ----------------------------------------------------------- #
    px = torch.cat([torch.zeros(n, 1), x], 1)
    py = torch.cat([torch.zeros(n, 1), y], 1)
    vx, vy = torch.diff(px, dim=1) / dt, torch.diff(py, dim=1) / dt
    ax, ay = torch.diff(vx, dim=1) / dt, torch.diff(vy, dim=1) / dt
    jx, jy = torch.diff(ax, dim=1) / dt, torch.diff(ay, dim=1) / dt
    head = torch.atan2(vy, vx)
    yr = torch.diff(head, dim=1) / dt
    ok = ((ax.abs().amax(1) <= lim["a_lon_max_mps2"])
          & (ay.abs().amax(1) <= lim["a_lat_max_mps2"])
          & (torch.sqrt(jx ** 2 + jy ** 2).amax(1) <= lim["jerk_max_mps3"])
          & (yr.abs().amax(1) <= lim["yaw_rate_max_radps"]))
    comfort = ok.float()

    return {
        "ego_progress": ep_score.numpy(),
        "ego_progress_raw_ratio": ratio.numpy(),
        "recovery": rc.numpy(),
        "cross_track_end_m": xt_end.numpy(),
        "cross_track_hold_matched_m": xt_hold.numpy(),
        "along_track_end_m": s_along.numpy(),
        "_cross_track_hold_v0_m_DIAGNOSTIC_NOT_USED": xt_hold_v0.numpy(),
        "comfort": comfort.numpy(),
        "no_collision": None,
        "ttc": None,
        "_unavailable": {"no_collision": COLLISION_UNAVAILABLE_REASON,
                         "ttc": COLLISION_UNAVAILABLE_REASON},
        "_comfort_limits": lim,
        "_map_free": ("No map, lane graph, drivable-area polygon, traffic light "
                      "or route signal is used. PhysicalAI-AV has none — the "
                      "card says verbatim 'we do not include open maps data'. "
                      "DAC / Lane Keeping / Driving-Direction Compliance / "
                      "Traffic-Light Compliance are therefore IMPOSSIBLE here "
                      "and are not faked."),
    }


def discriminative_range(scores, *, by_arm=None, ceil_frac_max=CEIL_FRAC_MAX,
                         range_min=RANGE_MIN) -> dict:
    """⚠️ BOOST M8 / C13 applied to METRICS: state the range before adopting.

    *"Comfort saturates at >= 99.9 %, contributing essentially zero
    discriminative information"* is a dead clause in someone else's suite. A
    component is **admissible only if it is measured to have range** here.

    ``scores`` maps component name -> 1-D array (NaN allowed).
    ``by_arm`` optionally maps arm name -> {component: array}; the BETWEEN-ARM
    spread is what actually decides an adjudication, so it is reported when
    available and is what ``admissible`` keys on.
    """
    out = {"_gate": {"ceil_frac_max": ceil_frac_max, "range_min": range_min,
                     "rule": "admissible iff ceiling_frac < ceil_frac_max AND "
                             "(max - min) >= range_min. When >= 2 arms are "
                             "supplied, the between-arm spread must also be "
                             "non-zero."}}
    for name, arr in scores.items():
        if arr is None:
            out[name] = {"admissible": False, "reason": "NOT COMPUTABLE",
                         "detail": COLLISION_UNAVAILABLE_REASON}
            continue
        a = np.asarray(arr, dtype=float)
        fin = a[np.isfinite(a)]
        if fin.size == 0:
            out[name] = {"admissible": False, "reason": "no finite values",
                         "n": 0}
            continue
        ceil = float((fin >= 0.999).mean())
        floor = float((fin <= 0.001).mean())
        rng = float(fin.max() - fin.min())
        node = {
            "n": int(fin.size), "n_nan": int(a.size - fin.size),
            "min": round(float(fin.min()), 6), "max": round(float(fin.max()), 6),
            "mean": round(float(fin.mean()), 6),
            "p05": round(float(np.percentile(fin, 5)), 6),
            "p95": round(float(np.percentile(fin, 95)), 6),
            "iqr": round(float(np.percentile(fin, 75)
                               - np.percentile(fin, 25)), 6),
            "ceiling_frac_ge_0p999": round(ceil, 6),
            "floor_frac_le_0p001": round(floor, 6),
            "observed_range": round(rng, 6),
        }
        node["admissible"] = bool(ceil < ceil_frac_max and rng >= range_min)
        if not node["admissible"]:
            node["reason"] = ("SATURATED at the ceiling" if ceil >= ceil_frac_max
                              else "range below range_min")
        if by_arm and len(by_arm) >= 2:
            means = {k: float(np.nanmean(np.asarray(v[name], float)))
                     for k, v in by_arm.items()
                     if v.get(name) is not None
                     and np.isfinite(np.asarray(v[name], float)).any()}
            if len(means) >= 2:
                sp = max(means.values()) - min(means.values())
                node["between_arm_mean"] = {k: round(v, 6)
                                            for k, v in means.items()}
                node["between_arm_spread"] = round(float(sp), 6)
                if sp <= 0.0:
                    node["admissible"] = False
                    node["reason"] = "zero between-arm spread — cannot adjudicate"
        out[name] = node
    return out


def composite(scores, ranges, *, weights=None, gates=("no_collision",)) -> dict:
    """PDM-shaped composite over the **admissible** components only.

    ``PDMS = (prod gate_m) x (sum w_x s_x / sum w_x)``. Here every multiplicative
    gate is a collision term and **all of them are unavailable**, so the product
    is empty.

    ⛔ **This is therefore NOT a Driving Score and is not named one.** A
    composite with no collision gate scores *recovery and progress*, and calling
    it anything else would be the same over-claim the OOD verdict string made.

    Raises :class:`VacuousMetric` if no weighted component is admissible —
    refusing to emit is the only honest output when every clause is dead.
    """
    w = dict(COMPONENT_WEIGHTS if weights is None else weights)
    admitted, dropped = {}, {}
    for name, wt in w.items():
        r = ranges.get(name, {})
        if r.get("admissible"):
            admitted[name] = wt
        else:
            dropped[name] = r.get("reason", "not admissible")
    if not admitted:
        raise VacuousMetric(
            "REFUSING TO EMIT a composite: no weighted component cleared the "
            f"discriminative-range gate. Dropped: {dropped}. A metric that "
            "cannot fail is not a metric; this program has shipped three "
            "vacuous diagnostics and the correct output here is nothing.")
    num = np.zeros_like(np.asarray(scores[next(iter(admitted))], float))
    den = np.zeros_like(num)
    for name, wt in admitted.items():
        a = np.asarray(scores[name], float)
        m = np.isfinite(a)
        num = num + np.where(m, a * wt, 0.0)
        den = den + np.where(m, wt, 0.0)
    val = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    gate_state = {g: {"available": scores.get(g) is not None,
                      "reason": None if scores.get(g) is not None
                      else COLLISION_UNAVAILABLE_REASON} for g in gates}
    return {
        "name": "PSS_recovery_progress",
        "_not_a_driving_score": (
            "There is NO collision gate in this composite because the cuboids "
            "are not available (see gates.reason). PDMS-shaped, but it scores "
            "RECOVERY and PROGRESS only. Do not report it as a Driving Score "
            "and do not compare it to a PDMS number."),
        "formula": "(empty gate product) x (sum w_x s_x / sum w_x)",
        "weights_admitted": admitted,
        "components_dropped": dropped,
        "gates": gate_state,
        "value": val,
    }


# =========================================================================== #
# aggregation                                                                  #
# =========================================================================== #
def _boot(x, eid, n_boot, seed):
    a = np.asarray(x, dtype=float)
    m = np.isfinite(a)
    if m.sum() < 2 or len(set(np.asarray(eid)[m])) < 2:
        return None
    return _ci.episode_cluster_bootstrap(a[m], list(np.asarray(eid)[m]),
                                         n_boot=n_boot, seed=seed)


def emit(pw, *, arm="unknown", n_boot=None, seed=0, weights=None,
         by_arm_scores=None) -> dict:
    """The full result node for one arm: sub-scores, ranges, composite, CIs.

    Every node carries ``traffic_mode``, the envelope proof, the estimator and
    the refused estimator. Nothing here can be quoted without them.
    """
    n_boot = _ci.DEFAULT_N_BOOT if n_boot is None else int(n_boot)
    sc = score_windows(pw)
    eid = list(pw["eid"])
    comps = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
    comps["no_collision"] = None
    comps["ttc"] = None
    ranges = discriminative_range(comps, by_arm=by_arm_scores)

    node = {
        "arm": arm,
        "protocol": PROTOCOL,
        "traffic_mode": TRAFFIC_MODE_LOG_REPLAY,
        "traffic_mode_note": TRAFFIC_MODE_NOTE,
        "grid": pw.get("grid"),
        "envelope_proof": pw.get("envelope_proof"),
        "horizon_s": pw.get("horizon_s"),
        "n_evaluations": int(len(eid)),
        "n_episodes": int(len(set(eid))),
        "planner_calls": pw.get("planner_calls"),
        "rollout_steps_executed": pw.get("rollout_steps_executed", 0),
        "_no_accumulation": pw.get("_no_accumulation"),
        "_estimator": "taniteval.ci.episode_cluster_bootstrap "
                      f"(B={n_boot}, unit = val episode)",
        "_refused_estimator": ("overlapping_holdout_se — it biases the POINT "
                               "ESTIMATE as well as the interval"),
        "component_discriminative_range": ranges,
        "components": {},
    }
    for k, v in comps.items():
        node["components"][k] = (
            {"ci": _boot(v, eid, n_boot, seed),
             "admissible": ranges[k].get("admissible")} if v is not None
            else {"ci": None, "admissible": False,
                  "reason": COLLISION_UNAVAILABLE_REASON})
    try:
        comp = composite(comps, ranges, weights=weights)
        val = comp.pop("value")
        comp["ci"] = _boot(val, eid, n_boot, seed)
        # the weighted-vs-unweighted disagreement check
        w = proximity_weights([(0.0, float(a), float(b)) for a, b
                               in zip(pw["pt_dyaw"].numpy(),
                                      pw["pt_dlon"].numpy())])
        fin = np.isfinite(val)
        comp["proximity_weighted_mean"] = (
            round(float((val[fin] * w[fin]).sum() / w[fin].sum()), 6)
            if fin.any() else None)
        comp["unweighted_mean"] = (round(float(np.nanmean(val)), 6)
                                   if fin.any() else None)
        node["composite"] = comp
        node["_per_window_composite"] = val
    except VacuousMetric as exc:
        node["composite"] = {"REFUSED_TO_EMIT": str(exc)}
    node["_per_window"] = {k: v for k, v in sc.items()
                           if isinstance(v, np.ndarray)}
    node["_per_window"]["eid"] = np.asarray(eid)
    node["_per_window"]["pt_dyaw"] = pw["pt_dyaw"].numpy()
    node["_per_window"]["pt_dlon"] = pw["pt_dlon"].numpy()
    node["_per_window"]["pt_dlat"] = pw["pt_dlat"].numpy()
    return node
