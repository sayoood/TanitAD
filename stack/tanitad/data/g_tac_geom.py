"""`g_tac` ENGINE-A GEOMETRY DERIVATION — the hindsight tactical-goal floor.

⭐ **WHY THIS MODULE EXISTS: `g_tac` HAD ZERO PRODUCERS.** MEASURED 2026-08-22 at
three independent sites, which agree:

1. ``scripts/vlm_tac_compose.py:112-126`` calls ``tac_str_labels.compose(...)``
   and passes **neither ``vlm_goal`` nor ``ego_args``** ⇒ ``compose``'s goal
   branch (``tac_str_labels.py:669-674``) takes ``vlm_goal is None`` on every
   clip and the tactical goal is ``ABSTAIN`` **100 % of the time**.
2. ``scripts/ph1_fuse.py:794-795`` emits ``g_tac_lat``/``g_tac_lon`` as
   ``{"token": None, "unavailable_reason": _G_TAC_GAP}``, whose text says
   verbatim *"NOTHING in this fuse derives them"*.
3. ``…/Research/2026-08-19-alpamayo-screening/LABEL_PIPELINE_CONFIRMATION.md``
   §4: *"the tactical-goal layer (`g_tac`), which is 100 % VLM-dependent and is
   currently never produced at all, because `vlm_goal` is never passed."*

⛔ **AND THAT VIOLATES A BINDING SPEC RULE.** ``HIERARCHY_VOCABULARY.md`` §2:
*"Every token must be HINDSIGHT-DERIVABLE from Engine A geometry alone (VLM
enriches; geometry guarantees) — so the vocabulary works even where the VLM
abstains."* A goal layer reachable only through a 9 B VLM is the opposite of
that: it abstains exactly where the VLM does, and the VLM leg is the expensive,
partially-run, never-human-reviewed one (25.6 T4-days for the 4,522-clip target,
`LABEL_PIPELINE_CONFIRMATION.md` §5).

⇒ This module is the GEOMETRY FLOOR: the part of `g_tac` that ego poses alone
guarantee, on 100 % of clips, with every unreachable token DECLARED rather than
silently missing (:data:`REFUSED`).

--------------------------------------------------------------------------- #
ADMISSIBILITY
--------------------------------------------------------------------------- #

⛔ **Labels may use ego; inference is vision-only** (Sayed 2026-08-03). This is
the LABEL side: future poses are admissible here and nowhere else. Nothing in
this module runs at inference.

⛔ **Goal/situation disjointness** (Sayed 2026-08-03). This module **never
imports** ``tanitad.data.situations`` and takes a raw pose tensor rather than any
summary dict, so there is no channel through which a situation-classifier output
could arrive — the same load-bearing omission ``anchor_goal.py`` and
``ph0_pilot._fmt_engine_a`` make deliberately. Pinned by
``tests/test_g_tac_geom.py::test_module_has_no_situation_classifier_path``.

⚠️ **THE ECHO CONTROL THIS LABEL REQUIRES, STATED UP FRONT.** The corridor
reference is built from the ego's **PAST** curvature, and the ego's speed is fed
to the model at inference as the integrator constant. A head trained on
:func:`g_tac_geom` labels MUST therefore be run through the same **v0-shuffle
echo control** that ``tac_str_labels`` mandates for ``LON_FROM_EGO_KINEMATICS``
before any number is quotable. The *deviation* is future information the model
must predict from vision; the *reference frame* is not, and only a control
separates them.

--------------------------------------------------------------------------- #
⛔ THE LAT AXIS IS REFUTED ON THIS CORPUS — MEASURED 2026-08-22, NOT INHERITED
--------------------------------------------------------------------------- #

**This module SHIPS THE LON AXIS ONLY.** The lateral half was built, measured,
and refuted in the same session; the machinery is kept as a DIAGNOSTIC
(:func:`corridor_offset`, opt-in via ``lat_arm="refuted-diagnostic"``) so nobody
re-derives it, and it emits no token by default.

*The idea.* The naive reading of *"curvature-relative corridor frame"* on a
map-less corpus is *"the corridor is the ego's smoothed path"* — a degenerate
label (offset ~0 by construction; the shape of flagship v1's route head scoring
1.0000 as a bijection of its own input). So the reference was the
**constant-curvature continuation**: the circular arc leaving
``(x_t, y_t, yaw_t)`` with the curvature the ego carried over the **preceding**
:data:`PAST_WINDOW_S`. On synthetic paths it does exactly what it should — a
constant-radius bend reads **≤ 0.014 m** at R ∈ {50,100,200,500} m × v ∈ {5,10,30}
m/s, while the refuted ``LANE_TARGET`` gate false-positives on **12/12** of the
same bends (up to 122.7 m of "lateral displacement" on a gentle bend at 30 m/s).

*The refutation, on real poses* (400-episode local cache
``physicalai-train-14231cd29c74``, stride 5; artifacts in
``…/incoming/2026-08-22-g-tac-geometry-floor/raw/``):

1. **The control caught it.** On 1 120 windows the curvature-relative deriver
   fired at **68.84 %** and the refuted naive gate at **70.89 %** — they agree, so
   the curvature frame cancels essentially nothing on real roads. Realised
   ``|lat_offset|`` p50 = **4.38 m**, larger than a lane.
2. **Not estimator tuning.** Sweeping ``past_window_s`` over 0.5→5.0 s (a 10×
   range) moves p50 only 4.24→5.60 m and the fire rate stays **85–90 %**.
3. **Not curvature *change* either** — that was the first hypothesis and it is
   FALSIFIED: ``corr(lat_offset, Δκ·s²/2)`` = **−0.050, R² = 0.0025** (n = 1 395),
   against a shuffled control at R² = 0.0003.
4. ⭐ **It is the MODEL CLASS, and an oracle control proves it.** Give the circle
   the curvature fitted to *the very future path it is measuring* — the best any
   circular corridor can do — and the residual is still p50 **1.80 m** / p90
   **9.92 m** (n = 1 900, d = 80). Stratified by the reference sweep angle
   θ = |κ|·s, the ceiling clears the 0.75 m lane-scale bar in **exactly one band**,
   θ < 0.05 rad = **21.7 %** of the corpus (oracle p50 0.263 m) — and there the
   deployable past-referenced median is **0.752 m**, i.e. *at* the threshold, so
   it would fire on a coin flip.

⇒ **A single-arc ego-geometric corridor reference is inadmissible for
``CORRIDOR_OFFSET`` at a 6 s horizon on PhysicalAI-AV, on every band.** This is
the same wall ``LANE_TARGET`` hit — the PI adjudicated 14/18 of its labels wrong
on 2026-08-16 — but reached with a mechanism and an oracle ceiling rather than an
adjudication, and it generalises past the one gate that was tried.

⇒ **The LAT axis needs an EXTERNAL road model**, which is precisely what Engine C
was added for: ``HIERARCHY_VOCABULARY.md`` §0b, *"lane/road-surface geometry
PhysicalAI's labels never had — the closest admissible thing to the missing map."*
**The SAM3 backfill is therefore on the LAT axis's critical path, not beside it.**

--------------------------------------------------------------------------- #
WHAT THIS MODULE EMITS, AND WHAT IT REFUSES
--------------------------------------------------------------------------- #

Emitted (geometry-guaranteed):
  * LON — ``STOP_POINT(position_arc_m)`` or ``LON_UNCONSTRAINED``

Refused, each with a machine-readable reason (:data:`REFUSED`) — *"the
unrepresentable clips being DECLARED rather than COERCED"*:
  * ``CORRIDOR_OFFSET``  — refuted above; needs an external road model
  * ``ANCHOR_GOAL``      — no anchor vocabulary reaches the tactical band
  * ``EVADE_IN_CORRIDOR``/``GAP_TARGET``/``YIELD_AT``/``WAIT_FOR_ONCOMING``
                         — need an ``agent_slot``; geometry cannot NAME an agent
  * ``SPEED_BAND``       — the F-14 blocker (both named inputs unavailable, one
                           forbidden)
  * ``TRAFFIC_LIGHT_REACT`` — needs light state from the VLM/SAM3 leg

⚠️ The LAT axis therefore ABSTAINS, and does **not** emit ``LAT_UNCONSTRAINED``:
that token is a CLAIM ("this axis is unconstrained"), and we cannot distinguish it
from "constrained but unmeasurable". An abstention with a reason is the honest
record; ``LAT_UNCONSTRAINED`` would be a fabricated negative.

``STOP_POINT``'s ``reason`` slot (``sign|light|queue|hazard``) is CATEGORICAL and
not derivable from geometry: it is left **unset with mask 0** — the ``IGNORE``
discipline ``anchor_goal.py`` established, so the gap stays visible instead of
being papered over with a plausible value.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

_SCRIPTS = str(Path(__file__).resolve().parents[2] / "scripts")
if _SCRIPTS not in sys.path:                       # refb_labels lives in scripts
    sys.path.insert(0, _SCRIPTS)

from refb_labels import ego_frame, path_curvature, wrap_to_pi  # noqa: E402

from tanitad.models.v6 import (DT, GOAL_ARG_NAMES, GOAL_CAT_ARG_NAMES,  # noqa: E402
                               STOP_REASONS, TAC_BAND_S,
                               TACTICAL_GOAL_TOKENS,
                               TACTICAL_GOAL_TOKENS_LAT,
                               TACTICAL_GOAL_TOKENS_LON, goal_token_axis)

__all__ = ["GTacField", "GTacGeom", "REFUSED", "ABSTAIN",
           "PAST_WINDOW_S", "MIN_PAST_SAMPLES", "LAT_OFFSET_MIN_M",
           "MOVING_MIN_MS", "STOP_EPS_MS", "MIN_ARC_M",
           "constant_curvature_point", "reference_curvature",
           "corridor_offset", "stop_point", "g_tac_geom", "census"]

ABSTAIN = "ABSTAIN"

# --------------------------------------------------------------------------- #
# thresholds — every one named, none inline                                     #
# --------------------------------------------------------------------------- #
#: How far back the reference curvature is read. One second at 10 Hz: long
#: enough to average yaw noise, short enough that a bend entered 3 s ago does
#: not dominate "what the ego is doing now".
PAST_WINDOW_S: float = 1.0
#: Fewer past samples than this and the reference is noise — abstain instead.
MIN_PAST_SAMPLES: int = 5
#: ⚠️ **DECLARED, NOT CALIBRATED** — the same discipline as
#: ``tac_str_labels.EGO_MAGNITUDE_BANDS``. Below this the LAT axis is
#: ``LAT_UNCONSTRAINED``. A lane is ~3.5 m and a nudge ~0.5-1.0 m, so 0.75 m is
#: a plausible floor — but plausible is exactly what killed ``LANE_TARGET``.
#: ⇒ :func:`census` reports the realised ``|lat_offset|`` distribution so this
#: is set from the null, and no per-token claim is admissible until it is.
LAT_OFFSET_MIN_M: float = 0.75
#: Below this the ego is not driving and the corridor reference is meaningless
#: (yaw is noise-dominated at standstill — the same guard ``path_curvature``
#: applies per step, applied here to the window).
MOVING_MIN_MS: float = 1.0
#: At/below this the vehicle is stopped. MIRRORS ``tac_str_labels
#: .EGO_STOP_EPS_MS`` — mirrored rather than imported so a change there fires a
#: test here instead of silently redefining a label.
STOP_EPS_MS: float = 0.5
#: Arc-length floor, mirroring ``refb_labels.MIN_ARC_M`` semantics.
MIN_ARC_M: float = 1e-3
_EPS = 1e-9

#: Every `g_tac` token this module CANNOT derive, and why. ⛔ A token missing
#: from both this dict and the emitted set is a BUG, not an abstention —
#: pinned by ``test_every_token_is_either_emitted_or_refused``.
REFUSED: dict[str, str] = {
    "CORRIDOR_OFFSET":
        "REFUTED 2026-08-22 on real poses (see module docstring): a single-arc "
        "ego-geometric corridor reference is inadmissible at a 6 s horizon on "
        "PhysicalAI-AV. The ORACLE circle — curvature fitted to the very future "
        "path it measures — still leaves p50 1.80 m / p90 9.92 m residual "
        "(n=1900, d=80), and the only theta band whose ceiling clears the 0.75 m "
        "lane bar (theta<0.05 rad, 21.7 % of corpus) has a deployable median of "
        "0.752 m, i.e. AT the threshold. Not estimator noise (past-window sweep "
        "flat over 10x) and not curvature CHANGE (R^2=0.0025). Needs an EXTERNAL "
        "road model — Engine C / SAM3 drivable surface "
        "(HIERARCHY_VOCABULARY.md 0b), which puts the SAM3 backfill on this "
        "axis's critical path. Diagnostic preserved: corridor_offset(), opt-in "
        "via lat_arm='refuted-diagnostic'.",
    "ANCHOR_GOAL":
        "no anchor vocabulary in the programme reaches the tactical band: every "
        "table built by build_refc_anchors.py stops at 20 steps = 2.0 s while "
        "TAC_BAND_S is (2.0, 6.0). Emitting a 2 s 'tactical' goal would look "
        "exactly like a label. (anchor_goal.py refusal 1, inherited verbatim.)",
    "EVADE_IN_CORRIDOR":
        "needs agent_slot (GOAL_CAT_ARG_TOKENS) — geometry measures the lateral "
        "deviation but cannot NAME the obstacle evaded. Reachable once the "
        "obstacle.offline join (scripts/build_obstacle_join.py) supplies agent "
        "slots per window; until then the deviation is emitted as "
        "CORRIDOR_OFFSET, which is the honest superset.",
    "GAP_TARGET":
        "needs agent_slot — the gap being targeted is another vehicle's, and "
        "ego poses do not contain other vehicles. Requires the obstacle.offline "
        "join.",
    "YIELD_AT":
        "needs agent_slot — the gap being yielded to is another agent's. "
        "Requires the obstacle.offline join.",
    "WAIT_FOR_ONCOMING":
        "needs agent_slot AND an oncoming-direction test over other agents' "
        "tracks. Requires the obstacle.offline join.",
    "SPEED_BAND":
        "F-14 BLOCKER (v6.py:183-217): both named derivation inputs are "
        "unavailable on this corpus and one is FORBIDDEN rather than missing — "
        "sign KIND/TEXT are forbidden (RETRACTION_LOG C87; G1 closed 0/31) and "
        "no corridor/lane graph exists (dataset card: 'we do not include open "
        "maps data'). vtarget_guarded is NOT the substitute: it is hindsight ego "
        "behaviour, not a permitted speed.",
    "TRAFFIC_LIGHT_REACT":
        "needs light state (red|amber|green|none) and a light_slot — a semantic "
        "fact from the VLM/SAM3 leg. Geometry can locate a stop but cannot say "
        "a light caused it; that is exactly what STOP_POINT's unset `reason` "
        "slot records.",
}


# --------------------------------------------------------------------------- #
# the label record                                                              #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GTacField:
    """One axis of `g_tac`: a token, its args, and — when absent — WHY.

    ``args`` / ``arg_mask`` are the 8 ``GOAL_ARG_NAMES`` slots in PHYSICAL UNITS;
    ``mask=0`` means UNSET (never a fabricated 0.0 — the ``8dc5d14d`` dyaw=0 /
    dist=null row is the pinned case for why). ``cat_args`` holds only the
    categorical slots actually derived; an omitted categorical slot is the
    ``IGNORE`` discipline, not a default.
    """

    token: str
    axis: str                                        # "lat" | "lon"
    args: dict[str, float] = field(default_factory=dict)
    cat_args: dict[str, int] = field(default_factory=dict)
    reason: str = ""
    leg: str = "geometry"

    def __post_init__(self) -> None:
        if self.token == ABSTAIN and not self.reason:
            raise ValueError("an abstention MUST carry a reason — a silent "
                             "abstention is indistinguishable from a bug")
        if self.token != ABSTAIN:
            if self.token not in TACTICAL_GOAL_TOKENS and not self.token.endswith(
                    "_UNCONSTRAINED"):
                raise ValueError(f"{self.token!r} is not a g_tac token")
            if self.axis != ("lat" if self.token in TACTICAL_GOAL_TOKENS_LAT
                             else "lon"):
                raise ValueError(f"{self.token!r} is not on axis {self.axis!r}")
        for k in self.args:
            if k not in GOAL_ARG_NAMES:
                raise ValueError(f"{k!r} is not a GOAL_ARG_NAMES slot")
        for k in self.cat_args:
            if k not in GOAL_CAT_ARG_NAMES:
                raise ValueError(f"{k!r} is not a GOAL_CAT_ARG_NAMES slot")

    def arg_vector(self) -> tuple[list[float], list[int]]:
        """``(values, mask)`` over the 8 slots, in ``GOAL_ARG_NAMES`` order.

        Unset slots are ``nan`` with mask 0 — a consumer that forgets the mask
        gets a loud NaN, not a quiet zero."""
        vals = [float(self.args.get(n, float("nan"))) for n in GOAL_ARG_NAMES]
        mask = [1 if n in self.args else 0 for n in GOAL_ARG_NAMES]
        return vals, mask

    def to_dict(self) -> dict:
        vals, mask = self.arg_vector()
        return {"token": self.token, "axis": self.axis, "leg": self.leg,
                "args": dict(self.args), "arg_vector": vals, "arg_mask": mask,
                "cat_args": dict(self.cat_args), "reason": self.reason}


@dataclass(frozen=True)
class GTacGeom:
    """The factored `g_tac` label for one window: one LAT field, one LON field."""

    lat: GTacField
    lon: GTacField
    audit: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"g_tac_lat": self.lat.to_dict(), "g_tac_lon": self.lon.to_dict(),
                "audit": dict(self.audit)}


# --------------------------------------------------------------------------- #
# geometry primitives                                                           #
# --------------------------------------------------------------------------- #
def constant_curvature_point(kappa: float, s: float) -> tuple[float, float, float]:
    """Point + heading after arc length ``s`` on a circle of curvature ``kappa``.

    Ego frame of the origin (+x forward, +y LEFT, heading 0), ``kappa`` in 1/m
    with **+ = left** (the ``refb_labels.path_curvature`` convention). Returns
    ``(x, y, heading_rad)``. Degrades to the straight line as ``kappa -> 0``, so
    a straight reference needs no special case at the call site.
    """
    if not math.isfinite(kappa) or abs(kappa) < _EPS:
        return float(s), 0.0, 0.0
    r = 1.0 / kappa                                   # signed radius, + = left
    th = kappa * s
    return r * math.sin(th), r * (1.0 - math.cos(th)), th


def reference_curvature(poses: Tensor, t: int, *,
                        past_window_s: float = PAST_WINDOW_S,
                        dt: float = DT) -> tuple[float, int]:
    """Arc-length-weighted mean curvature over the ``past_window_s`` BEFORE ``t``.

    Returns ``(kappa, n_samples)``. Weighting by realised arc length rather than
    by step makes the estimate speed-invariant, which is the property that lets
    the reference cancel a bend *at any speed* — the exact failure mode that
    retired ``LANE_TARGET``.

    ⚠️ Reads only ``poses[:t+1]`` — the PAST. A reference fitted to the same
    future it is later compared against would explain that future by
    construction and the label would be a constant.
    """
    n = max(int(round(past_window_s / dt)), 1)
    lo = max(t - n, 0)
    sub = poses[lo:t + 1]
    if sub.shape[0] < MIN_PAST_SAMPLES:
        return float("nan"), int(sub.shape[0])
    kap = path_curvature(sub)                                  # [m-1], len k-1
    seg = (sub[1:, :2] - sub[:-1, :2]).norm(dim=-1)            # realised arc
    w = seg.clamp_min(0.0)
    tot = float(w.sum())
    if tot < MIN_ARC_M:                       # stood still through the window
        return float("nan"), int(sub.shape[0])
    return float((kap * w).sum() / tot), int(sub.shape[0])


def _cum_arc(poses: Tensor, t: int, t_end: int) -> float:
    """Realised arc length of the ACTUAL path from ``t`` to ``t_end`` (metres)."""
    if t_end <= t:
        return 0.0
    seg = (poses[t + 1:t_end + 1, :2] - poses[t:t_end, :2]).norm(dim=-1)
    return float(seg.sum())


def _endpoint_ego(poses: Tensor, t: int, t_end: int) -> tuple[float, float]:
    """``poses[t_end]`` position in the ego frame of ``t`` (+x fwd, +y left)."""
    dxy = (poses[t_end, :2] - poses[t, :2]).unsqueeze(0)
    e = ego_frame(dxy, poses[t, 2].unsqueeze(0))
    return float(e[0, 0]), float(e[0, 1])


# --------------------------------------------------------------------------- #
# LAT — CORRIDOR_OFFSET                                                         #
# --------------------------------------------------------------------------- #
def corridor_offset(poses: Tensor, t: int, *,
                    band_s: tuple[float, float] = TAC_BAND_S,
                    dt: float = DT,
                    past_window_s: float = PAST_WINDOW_S) -> dict:
    """Signed lateral deviation of the actual future from the constant-curvature
    continuation, measured at the band END.

    Returns a dict with ``ok``, ``lat_offset_m``, ``arc_m``, ``kappa_ref``,
    ``lat_offset_at_band_open_m`` (audit extra — lets a consumer split the
    operative 0-2 s contribution from the tactical 2-6 s one) and, when
    ``ok`` is False, ``reason``.

    The offset is taken **perpendicular to the reference tangent** at that arc
    length, not as a raw y-difference: on a bend the two differ, and only the
    perpendicular one is a lateral deviation.
    """
    T = int(poses.shape[0])
    t_end = t + int(round(band_s[1] / dt))
    t_open = t + int(round(band_s[0] / dt))
    if t_end > T - 1:
        return {"ok": False, "reason":
                f"tactical band end t+{band_s[1]}s = index {t_end} is beyond the "
                f"episode (T={T}); a clamped endpoint would label a shorter "
                f"horizon than the token claims"}

    v0 = float(poses[t, 3]) if poses.shape[1] >= 4 else float("nan")
    kappa, n_past = reference_curvature(poses, t, past_window_s=past_window_s,
                                        dt=dt)
    if not math.isfinite(kappa):
        return {"ok": False, "reason":
                f"no reference curvature: {n_past} past samples "
                f"(< {MIN_PAST_SAMPLES}) or the ego did not move through the "
                f"past window — yaw is noise-dominated at standstill"}

    arc = _cum_arc(poses, t, t_end)
    if arc < MOVING_MIN_MS * (band_s[1] - band_s[0]):
        return {"ok": False, "reason":
                f"ego travelled {arc:.2f} m over the {band_s[1]:.1f} s band "
                f"(< {MOVING_MIN_MS} m/s mean) — a corridor offset on a "
                f"stationary vehicle is noise, not intent",
                "arc_m": arc, "kappa_ref": kappa}

    def _off(t_q: int) -> float:
        s_q = _cum_arc(poses, t, t_q)
        rx, ry, rth = constant_curvature_point(kappa, s_q)
        ax, ay = _endpoint_ego(poses, t, t_q)
        # left-normal of the reference tangent at s_q
        return (ax - rx) * (-math.sin(rth)) + (ay - ry) * math.cos(rth)

    return {"ok": True, "lat_offset_m": _off(t_end), "arc_m": arc,
            "kappa_ref": kappa, "n_past": n_past, "v0_ms": v0,
            "lat_offset_at_band_open_m": _off(min(t_open, T - 1))}


# --------------------------------------------------------------------------- #
# LON — STOP_POINT                                                              #
# --------------------------------------------------------------------------- #
def stop_point(poses: Tensor, t: int, *,
               band_s: tuple[float, float] = TAC_BAND_S,
               dt: float = DT) -> dict:
    """First arc length inside the band at which the ego comes to rest and STAYS.

    Returns ``ok`` + ``position_arc_m`` when a stop EVENT occurs inside the band.
    Two honest non-events:
      * the ego is already at rest when the band opens — there is nothing to
        plan, and a `STOP_POINT` here would be the ego-speed echo (a HOLD
        predictable from v0, the family ``tac_str_labels`` flags);
      * the ego never stops — ``LON_UNCONSTRAINED``.

    ⛔ ``reason`` (sign|light|queue|hazard) is NOT derived: geometry sees the
    stop, never its cause.
    """
    T = int(poses.shape[0])
    t_end = t + int(round(band_s[1] / dt))
    if t_end > T - 1:
        return {"ok": False, "reason":
                f"tactical band end index {t_end} beyond the episode (T={T})"}
    if poses.shape[1] < 4:
        return {"ok": False, "reason": "poses carry no speed channel"}

    v = poses[t:t_end + 1, 3]
    if float(v[0]) <= STOP_EPS_MS:
        return {"ok": False, "already_stopped": True, "reason":
                f"ego already at rest at band open (v={float(v[0]):.2f} <= "
                f"{STOP_EPS_MS} m/s) — no stop EVENT inside the band; a "
                f"STOP_POINT here is predictable from v0 (ego-speed echo)"}
    stopped = (v <= STOP_EPS_MS)
    if not bool(stopped.any()):
        return {"ok": False, "reason":
                f"ego never reaches rest inside the band "
                f"(min v={float(v.min()):.2f} > {STOP_EPS_MS} m/s)"}
    k = int(torch.nonzero(stopped, as_tuple=False)[0])
    if not bool(stopped[k:].all()):
        return {"ok": False, "reason":
                "ego touches rest then moves again inside the band — a crawl, "
                "not a stop; CREEP is an ACTION (a_tac_lon), not a goal"}
    return {"ok": True, "position_arc_m": _cum_arc(poses, t, t + k),
            "stop_index_in_band": k, "v0_ms": float(v[0])}


# --------------------------------------------------------------------------- #
# the deriver                                                                   #
# --------------------------------------------------------------------------- #
def g_tac_geom(poses: Tensor, t: int, *,
               band_s: tuple[float, float] = TAC_BAND_S,
               dt: float = DT,
               lat_offset_min_m: float = LAT_OFFSET_MIN_M,
               past_window_s: float = PAST_WINDOW_S,
               lat_arm: str = "abstain") -> GTacGeom:
    """Factored `g_tac` for the window opening at ``t``, from ego poses ALONE.

    ``poses`` is ``[T, 4]`` = (x, y, yaw, v) in world/odometry frame — the
    ``refb_labels`` contract. ``t`` is the window's present.

    ``lat_arm``:
      * ``"abstain"`` (DEFAULT, and the only admissible setting) — the LAT axis
        abstains with the refutation as its reason. See the module docstring.
      * ``"refuted-diagnostic"`` — re-enables the REFUTED ``CORRIDOR_OFFSET``
        emission. ⛔ For reproducing the refutation ONLY. Labels produced this
        way are not admissible for supervision; the census tags them.
    """
    if poses.ndim != 2 or poses.shape[1] < 3:
        raise ValueError(f"poses must be [T, >=3], got {tuple(poses.shape)}")
    if not 0 <= t < poses.shape[0]:
        raise ValueError(f"t={t} out of range for T={poses.shape[0]}")
    if lat_arm not in ("abstain", "refuted-diagnostic"):
        raise ValueError(f"lat_arm must be 'abstain' or 'refuted-diagnostic', "
                         f"got {lat_arm!r}")

    co = corridor_offset(poses, t, band_s=band_s, dt=dt,
                         past_window_s=past_window_s)
    if lat_arm == "abstain":
        # ⛔ NOT `LAT_UNCONSTRAINED`: that token CLAIMS the axis is
        # unconstrained, and we cannot distinguish it from "constrained but
        # unmeasurable". A fabricated negative is still a fabrication.
        lat = GTacField(ABSTAIN, "lat", leg="none",
                        reason=REFUSED["CORRIDOR_OFFSET"].split(".")[0] + ".")
    elif not co["ok"]:
        lat = GTacField(ABSTAIN, "lat", reason=co["reason"], leg="none")
    elif abs(co["lat_offset_m"]) >= lat_offset_min_m:
        lat = GTacField("CORRIDOR_OFFSET", "lat",
                        args={"arg0": co["lat_offset_m"], "arg1": co["arc_m"],
                              "at_arc_m": co["arc_m"]})
    else:
        lat = GTacField("LAT_UNCONSTRAINED", "lat",
                        reason=f"|lat_offset| {abs(co['lat_offset_m']):.3f} m "
                               f"< {lat_offset_min_m} m — the ego tracks its "
                               f"constant-curvature continuation")

    sp = stop_point(poses, t, band_s=band_s, dt=dt)
    if sp["ok"]:
        lon = GTacField("STOP_POINT", "lon",
                        args={"arg0": sp["position_arc_m"],
                              "at_arc_m": sp["position_arc_m"]})
        # `reason` (arg1 / cat slot) deliberately UNSET — see module docstring.
    elif sp.get("already_stopped"):
        lon = GTacField(ABSTAIN, "lon", reason=sp["reason"], leg="none")
    elif "position_arc_m" not in sp and "never reaches rest" in sp.get("reason", ""):
        lon = GTacField("LON_UNCONSTRAINED", "lon", reason=sp["reason"])
    else:
        lon = GTacField(ABSTAIN, "lon", reason=sp["reason"], leg="none")

    audit = {"lat_arm": lat_arm,
             "kappa_ref": co.get("kappa_ref"), "arc_m": co.get("arc_m"),
             "lat_offset_m": co.get("lat_offset_m"),
             "lat_offset_at_band_open_m": co.get("lat_offset_at_band_open_m"),
             "v0_ms": co.get("v0_ms", sp.get("v0_ms")),
             "band_s": tuple(band_s), "lat_offset_min_m": lat_offset_min_m}
    return GTacGeom(lat=lat, lon=lon, audit=audit)


# --------------------------------------------------------------------------- #
# census — the null distribution the threshold must be set from                 #
# --------------------------------------------------------------------------- #
def census(labels) -> dict:
    """Token counts, abstention reasons, and the ``|lat_offset|`` quantiles.

    ⚠️ **The quantiles are the point.** :data:`LAT_OFFSET_MIN_M` is DECLARED, not
    calibrated; this is the measurement that must set it. A census whose
    ``CORRIDOR_OFFSET`` share is ~0 % or ~100 % says the threshold is wrong, not
    that the corpus is uniform.

    ⛔ Carries the CONSTANT-ONLY CONTROL: ``degenerate_lat``/``degenerate_lon``
    are True when one token takes every window, i.e. when the deriver is a
    constant wearing a label's clothes.
    """
    from collections import Counter

    lat_c: Counter = Counter()
    lon_c: Counter = Counter()
    reasons: Counter = Counter()
    offs: list[float] = []
    n = 0
    for lab in labels:
        n += 1
        lat_c[lab.lat.token] += 1
        lon_c[lab.lon.token] += 1
        for f in (lab.lat, lab.lon):
            if f.token == ABSTAIN:
                reasons[f.reason.split("—")[0].strip()[:80]] += 1
        o = lab.audit.get("lat_offset_m")
        if o is not None and math.isfinite(o):
            offs.append(abs(o))

    q: dict[str, float] = {}
    if offs:
        s = sorted(offs)
        for p in (50, 75, 90, 95, 99):
            q[f"p{p}"] = s[min(int(p / 100.0 * len(s)), len(s) - 1)]
        q["max"] = s[-1]
        q["mean"] = sum(s) / len(s)
    return {"n_windows": n,
            "lat": dict(lat_c), "lon": dict(lon_c),
            "abstain_reasons": dict(reasons),
            "abs_lat_offset_m": q, "n_with_offset": len(offs),
            "degenerate_lat": len(lat_c) == 1 and n > 1,
            "degenerate_lon": len(lon_c) == 1 and n > 1,
            "refused_tokens": sorted(REFUSED),
            "lat_offset_min_m": LAT_OFFSET_MIN_M,
            "emitted_tokens": sorted(
                set(TACTICAL_GOAL_TOKENS) - set(REFUSED))}
