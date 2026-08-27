#!/usr/bin/env python3
"""NavSim -> TanitEval adapter. Converts a NavSim scene/agent output into the
harness's ``win`` dict so the FOUR BINDING FAMILIES run on NavSim data.

⛔ READ THIS FIRST — THIS MODULE EMITS **TWO** BLOCKS AND THEY MUST NOT BE CONFLATED
------------------------------------------------------------------------------------
1. ``four_families`` — **OUR** instruments run on NavSim **trajectory geometry**
   (``taniteval.four_families.all_families``). LONGITUDINAL and LATERAL are
   computable here because they are derived from the ego-frame paths, not from
   anything NavSim publishes.
2. ``benchmark.navsim`` — **NAVSIM'S OWN** instrument (EPDMS/PDMS + sub-metrics).
   Four of our criteria are ABSENT from it *by construction* and are emitted as
   explicit refusals by :func:`navsim_native_family_refusals`.

Reading (1) as though it were (2) — or vice versa — would claim NavSim measures
something it does not. Every refusal below states WHICH of the two blocks it
belongs to.

THE CONTRACT THIS IS WRITTEN AGAINST (read from source, not from prose)
----------------------------------------------------------------------
The ``win`` dict, as **built by the harness itself** at
``taniteval/tools/ff_rescore.py:404-406`` and **consumed** at
``taniteval/taniteval/four_families.py:1165-1181``:

===================  =====================  =========================================
key                  shape                  source of truth
===================  =====================  =========================================
``pred_dense``       ``[N, K, 2]`` float    ``ff_rescore.py:404``; read ``four_families.py:1168``
``gt_dense``         ``[N, K, 2]`` float    ``ff_rescore.py:404``; read ``four_families.py:1169``
``pred``             ``[N, M, 2]`` float    sparse companion, ``ff_rescore.py:405``
``gt``               ``[N, M, 2]`` float    same line
``wp_steps``         list, len ``M``        MODEL-TICK contract, ``ff_rescore.py:399-403``
``dt_s``             float                  DENSE column spacing, ``four_families.py:1170``
``eid``              list len ``N``/None    bootstrap CLUSTER key, ``four_families.py:658``
``v0``               ``[N]`` float          anti-echo controls, ``ff_rescore.py:410-411``
``lead``             dict                   distance-keeping, ``four_families.py:1153-1155``
===================  =====================  =========================================

and the ``load_dump`` return shape at ``taniteval/tools/ff_rescore.py:150-152``:
``{"kind", "gt", "eid", "dt_s", "wp_steps", "source", "n_episodes", "arms": {name: (key, [N,K,2])}}``
(+ ``"v0"`` at ``:227-228``, + ``"dt_provenance"`` at ``:216``).

⛔ THE ORIGIN IS **NOT** A COLUMN. Verified, not assumed:
``taniteval/taniteval/lateral.py:140`` PREPENDS ``zeros(n, 1, 2)`` before
differencing — *"the dense path starts one tick AFTER the ego pose"* (``:136-137``)
— and ``stack/scripts/driving_diagnostic.py:105`` builds GT from
``wp_steps=(5,10,15,20)``, which has no ``k=0``. A NavSim pose array that
INCLUDES ``t=0`` therefore carries one extra leading column of zeros, and feeding
it in would shift every horizon by one tick while the zero-length first segment
is silently absorbed by ``frenet_dense``'s degenerate-tangent fallback
(``lateral.py:145-148``). :func:`poses_to_waypoints` refuses to guess: the caller
DECLARES ``origin_included`` and the claim is CHECKED against the data.

COORDINATE FRAMES
-----------------
* **Ours** — origin = ego pose at the window's LAST OBSERVED frame; ``x`` forward
  (along-track), ``y`` LEFT (cross-track). ``stack/scripts/driving_diagnostic.py:87-90``
  (``_ego``), applied at ``:101-106``; axis roles at
  ``taniteval/taniteval/four_families.py:33-40``.
* **NavSim** — poses are ``(x, y, heading)`` in the **ego frame** of the queried
  frame (``NAVSIM_PROTOCOL.md:534``), and that frame is **also x-forward /
  y-left**: the ``driving_command`` derivation interpolates a point 20 m AHEAD,
  converts it to the ego frame and tests ``y >= +2 m -> left``
  (``NAVSIM_PROTOCOL.md:506-509``). ⇒ the AXES agree; the conversion is an
  identity on the axes and the real work is (a) dropping the ``heading`` column,
  (b) the origin row, (c) the world->ego rotation when the source poses are
  absolute nuPlan world poses rather than an agent's ego-frame output.

⚠️ ``heading`` IS NOT THE SAME QUANTITY IN BOTH SYSTEMS. NavSim's third column is
**vehicle yaw**. Our LATERAL family's ``heading_mae_deg`` is the **PATH TANGENT**
``atan2(dy, dx)`` (``four_families.py:170``, and the audit states it explicitly:
*"it is the path tangent, not vehicle yaw"*). They coincide only under no
side-slip. NavSim's heading is therefore carried as an INDEPENDENT channel — it
is what makes :func:`assert_lateral_sign_convention` possible — and is never
substituted for the family's own value.

⛔ THE COLUMN TRAP — ``score``, NEVER ``pdm_score``
--------------------------------------------------
``PDMScorer._aggregate_pdm_scores`` masks Extended Comfort OUT and divides by
``5+5+2+2 = 14``; the real EPDMS (denominator **16**) is assembled downstream in
``run_pdm_score.py::compute_final_scores``. The wrong column looks perfectly
valid. :func:`read_epdms` REFUSES ``pdm_score`` by name — see
``NAVSIM_PROTOCOL.md:281-290`` and ``CRITERIA_REGISTRY.json``
``benchmarks.navsim.THE_COLUMN_TRAP``.

⛔ THE ESTIMATOR IS AN OPEN QUESTION — THIS MODULE DOES NOT CLOSE IT
-------------------------------------------------------------------
Our decision-grade interval is the **episode-cluster bootstrap**
(``taniteval/taniteval/ci.py``). It does **not** transfer: NavSim's unit is a
**scene token**, and ``docs/splits.md`` states *"NavSim splits contain
overlapping scenes"* — so scene tokens are **not independent draws**, and
clustering by nuPlan log has never been pre-registered here. Inventing either
would manufacture a decision-grade interval out of an unsettled unit.

⇒ :func:`scenes_to_win` sets ``win["eid"] = None`` **on purpose**. That is not a
loss of data (the tokens are preserved under ``win["_navsim"]["scene_tokens"]``):
it makes the harness's own guard fire, so every interval inside ``all_families``
self-refuses with a reason and an ``n`` (``four_families.py:658-664``) instead of
returning a scene-token bootstrap that would look valid. :func:`estimator_refusal`
adds the NavSim-specific reason at the top level.
⇒ ``NAVSIM_PROTOCOL.md`` §8.5 and §9 item 15. **A PI decision, not an implementation gap.**
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

# path bootstrap — same convention as taniteval/tools/*.py (ff_rescore.py:68-75),
# so this imports from any cwd with no preset PYTHONPATH.
_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/taniteval/adapters
_TE_PARENT = os.path.dirname(_HERE)                          # <repo>/taniteval
_REPO = os.path.dirname(_TE_PARENT)                          # <repo>
for _pth in (os.path.join(_REPO, "stack"),
             os.path.join(_REPO, "stack", "scripts"),
             _TE_PARENT):
    if os.path.isdir(_pth) and _pth not in sys.path:
        sys.path.insert(0, _pth)

ADAPTER = "taniteval/adapters/navsim.py"
PROTOCOL_DOC = "products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md"
DEVKIT_PIN = "autonomousvision/navsim@0a380a9 (2025-10-27)"

#: ⛔ The ONLY admissible EPDMS/PDMS column, and the one that must never be read
#: as it. ``NAVSIM_PROTOCOL.md:281-290``.
EPDMS_COLUMN = "score"
FORBIDDEN_EPDMS_COLUMN = "pdm_score"

#: ``NAVSIM_PROTOCOL.md:90-98`` (v1) and ``:125-137`` (v2); re-stated in
#: ``CRITERIA_REGISTRY.json`` ``benchmarks.navsim.variants``.
VARIANTS = {
    "PDMS_v1": {"multipliers": ("NC", "DAC"),
                "weighted": {"EP": 5, "TTC": 5, "C": 2},
                "denominator": 12,
                "note": "DDC exists in v1 code with weight 0.0 — a navtest DDC "
                        "column contributes NOTHING to PDMS."},
    "EPDMS_v2": {"multipliers": ("NC", "DAC", "DDC", "TLC"),
                 "weighted": {"EP": 5, "TTC": 5, "LK": 2, "HC": 2, "EC": 2},
                 "denominator": 16,
                 "note": "EC is injected downstream by "
                         "run_pdm_score.py::compute_final_scores; the per-frame "
                         "`pdm_score` column has it MASKED OUT and /14."},
}

#: Four mutually incomparable protocols exist — ``NAVSIM_PROTOCOL.md:549-556``.
SPLITS = ("navtest", "navhard_two_stage", "private_test_hard_two_stage",
          "warmup_two_stage", "navtrain")

#: ``resolve_tier`` (``tools/ff_rescore.py:126-141``) accepts only T0/T1; the
#: criteria registry additionally defines T2. An unstamped arm is a HARD ERROR
#: everywhere, so this module has NO DEFAULT.
TIERS = ("T0", "T1", "T2")

FRAMES = ("ego", "world")

#: NavSim's ego frame, established from ``NAVSIM_PROTOCOL.md:506-509`` + ``:534``.
NAVSIM_EGO_FRAME = {
    "pose": "(x, y, heading)",
    "x": "forward, metres",
    "y": "LEFT, metres",
    "heading": "vehicle yaw, radians, CCW positive",
    "origin": "ego pose at the queried (initial) frame of the scene",
    "established_from": (
        "NAVSIM_PROTOCOL.md:534 (submission poses are ego-frame (x,y,heading)) "
        "and :506-509 (the driving_command derivation interpolates 20 m AHEAD, "
        "converts to the ego frame and tests `y >= +2 m -> left`, which fixes "
        "+x as forward and +y as left)."),
}

#: Ours, from source.
TANITAD_WIN_FRAME = {
    "pose": "(x, y)",
    "x": "forward / ALONG-track, metres",
    "y": "LEFT / CROSS-track, metres",
    "origin": "ego pose at the window's LAST OBSERVED frame",
    "origin_is_a_column": False,
    "established_from": (
        "stack/scripts/driving_diagnostic.py:87-90 (_ego) applied at :101-106; "
        "axis roles at taniteval/taniteval/four_families.py:33-40; the origin is "
        "NOT a column because taniteval/taniteval/lateral.py:140 prepends it."),
}


# --------------------------------------------------------------------------- #
# Errors — each names the defect it prevents                                   #
# --------------------------------------------------------------------------- #
class NavSimAdapterError(Exception):
    """Base. Every subclass refuses rather than guesses."""


class PdmScoreColumnError(NavSimAdapterError):
    """⛔ ``pdm_score`` was offered where an EPDMS was required."""


class FrameConventionError(NavSimAdapterError):
    """⛔ The declared frame/origin convention disagrees with the data."""


class EstimatorNotSettledError(NavSimAdapterError):
    """⛔ A NavSim confidence interval was requested. The cluster unit is open."""


def _arr(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        raise NavSimAdapterError(f"{name} is empty — refusing to score nothing.")
    if not np.isfinite(a).all():
        raise NavSimAdapterError(
            f"{name} carries {int((~np.isfinite(a)).sum())} non-finite value(s). "
            f"⛔ NaN/inf propagate silently through every mean in "
            f"four_families; refusing rather than reporting a nan metric.")
    return a


def _wrap(a):
    """Wrap to (-pi, pi] — same convention as ``driving_diagnostic._wrap:93-95``."""
    return (np.asarray(a, dtype=np.float64) + math.pi) % (2 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# 1. Frame conversion                                                          #
# --------------------------------------------------------------------------- #
def ego_frame_from_world(xy, origin_xy, origin_yaw) -> np.ndarray:
    """World/global ``[..., 2]`` -> our ego frame. EXACT port of ``_ego``.

    ``stack/scripts/driving_diagnostic.py:87-90``::

        c, s = cos(-yaw), sin(-yaw)
        [dx*c - dy*s,  dx*s + dy*c]

    ⛔ Re-derived here rather than imported ONLY because importing
    ``driving_diagnostic`` drags in ``torch`` + the whole ``tanitad`` stack for a
    2x2 rotation; :func:`verify_frame` then checks the RESULT against the
    harness's own verifier, so the copy cannot drift undetected.
    """
    p = np.asarray(xy, dtype=np.float64)
    if p.shape[-1] != 2:
        raise FrameConventionError(
            f"xy must end in a 2-axis, got {p.shape}. If these are "
            f"(x, y, heading) poses, slice [..., :2] via poses_to_waypoints.")
    o = np.asarray(origin_xy, dtype=np.float64)
    yaw = np.asarray(origin_yaw, dtype=np.float64)
    if o.shape[-1] != 2:
        raise FrameConventionError(f"origin_xy must end in a 2-axis, got {o.shape}")
    # Insert the K axis for a PER-WINDOW origin ([N,2] against [N,K,2]). A bare
    # [2] / scalar broadcasts on its own; anything else is ambiguous and is
    # refused rather than reshaped by guesswork — silently pairing window i's
    # path with window j's origin rotates each path by ANOTHER window's yaw.
    if o.ndim == p.ndim - 1 and o.shape[0] == p.shape[0]:
        o = o[..., None, :]
    if yaw.ndim == p.ndim - 2 and yaw.ndim > 0 and yaw.shape[0] == p.shape[0]:
        yaw = yaw[..., None]
    if o.ndim not in (1, p.ndim) or yaw.ndim not in (0, p.ndim - 1):
        raise FrameConventionError(
            f"cannot align origin_xy {np.shape(origin_xy)} / origin_yaw "
            f"{np.shape(origin_yaw)} with xy {p.shape} unambiguously. Pass a "
            f"per-window [N,2] / [N] pair, or a single [2] / scalar.")
    d = p - o
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.stack([d[..., 0] * c - d[..., 1] * s,
                     d[..., 0] * s + d[..., 1] * c], axis=-1)


#: A first sample this close to the origin, on a path that clearly moved, is the
#: ``t=0`` row and not a real waypoint. Both thresholds are stated so the refusal
#: is reproducible rather than a magic number.
ORIGIN_ROW_ATOL_M = 1e-3
MOVED_FINAL_M = 1.0


def poses_to_waypoints(poses, *, frame: str, origin_included: bool,
                       origin_xy=None, origin_yaw=None,
                       name: str = "poses") -> tuple:
    """NavSim poses -> ``([N, K, 2], heading [N, K] or None, meta)``.

    ``poses`` is ``[N, P, 2|3]`` (a single scene ``[P, 2|3]`` is promoted).
    ``frame`` is ``"ego"`` (NavSim ``Trajectory.poses``) or ``"world"`` (absolute
    nuPlan poses — then ``origin_xy``/``origin_yaw`` are REQUIRED).

    ⛔ ``origin_included`` is DECLARED and then VERIFIED against the data. It has
    no default and is never inferred silently: an unnoticed leading ``t=0`` row
    shifts every horizon by one tick, and the zero-length first segment it
    creates is absorbed by ``lateral.frenet_dense``'s degenerate-tangent
    fallback (``lateral.py:145-148``) — a wrong number, not an exception.
    """
    if frame not in FRAMES:
        raise FrameConventionError(f"frame must be one of {FRAMES}, got {frame!r}")
    if not isinstance(origin_included, bool):
        raise FrameConventionError(
            "origin_included must be an explicit bool. ⛔ There is no default: "
            "whether the t=0 row is a column is a property of the PRODUCER "
            "(a NavSim agent Trajectory excludes it; the PDM simulator's state "
            "array includes it — NAVSIM_PROTOCOL.md:415-417 asserts "
            "`states.shape[1] == num_poses + 1`), and guessing shifts every "
            "horizon by one tick.")
    p = _arr(poses, name)
    if p.ndim == 2:
        p = p[None, ...]
    if p.ndim != 3 or p.shape[-1] not in (2, 3):
        raise FrameConventionError(
            f"{name} must be [N, P, 2] or [N, P, 3] ((x,y) or (x,y,heading)), "
            f"got {p.shape}")
    heading = p[..., 2].copy() if p.shape[-1] == 3 else None
    xy = p[..., :2].copy()

    if frame == "world":
        if origin_xy is None or origin_yaw is None:
            raise FrameConventionError(
                "frame='world' needs origin_xy [N,2] and origin_yaw [N] — the "
                "ego pose at the window's last observed frame "
                "(driving_diagnostic.py:103-104). Without them there is no "
                "frame to convert INTO.")
        oxy = np.asarray(origin_xy, dtype=np.float64).reshape(-1, 2)
        oyaw = np.asarray(origin_yaw, dtype=np.float64).reshape(-1)
        if oxy.shape[0] != xy.shape[0] or oyaw.shape[0] != xy.shape[0]:
            raise FrameConventionError(
                f"origin_xy {oxy.shape} / origin_yaw {oyaw.shape} do not align "
                f"with {xy.shape[0]} windows — a positional join on mismatched "
                f"rows would rotate each path by ANOTHER window's yaw.")
        xy = ego_frame_from_world(xy, oxy[:, None, :], oyaw[:, None])
        if heading is not None:
            heading = _wrap(heading - oyaw[:, None])

    # ---- the declared origin row, CHECKED ---------------------------------- #
    first = float(np.abs(xy[:, 0, :]).mean())
    last = float(np.linalg.norm(xy[:, -1, :], axis=-1).mean())
    if origin_included:
        if first > ORIGIN_ROW_ATOL_M:
            raise FrameConventionError(
                f"origin_included=True but the first sample is {first:.4f} m "
                f"from the origin (tol {ORIGIN_ROW_ATOL_M} m). Either the array "
                f"does NOT carry the t=0 row, or the declared origin is the "
                f"wrong pose. Refusing: stripping a real waypoint loses a tick "
                f"and shifts the whole horizon.")
        xy = xy[:, 1:, :]
        if heading is not None:
            heading = heading[:, 1:]
        if xy.shape[1] == 0:
            raise FrameConventionError(
                f"{name} had only the t=0 row — nothing to score after stripping it.")
    else:
        if first <= ORIGIN_ROW_ATOL_M and last > MOVED_FINAL_M:
            raise FrameConventionError(
                f"origin_included=False but the first sample sits {first:.6f} m "
                f"from the origin while the path travels {last:.2f} m — that "
                f"leading row is the t=0 state, not a waypoint. ⛔ Keeping it "
                f"shifts every horizon by one tick and its zero-length first "
                f"segment is silently absorbed by lateral.frenet_dense's "
                f"degenerate-tangent fallback (lateral.py:145-148). Declare "
                f"origin_included=True.")
    meta = {
        "n_windows": int(xy.shape[0]), "K": int(xy.shape[1]),
        "source_frame": frame,
        "origin_row_declared": bool(origin_included),
        "origin_row_stripped": bool(origin_included),
        "mean_abs_first_sample_m": round(first, 6),
        "mean_final_range_m": round(last, 4),
        "heading_channel": ("present — NavSim VEHICLE YAW, NOT the path tangent "
                            "our LATERAL family computes (four_families.py:170)"
                            if heading is not None else None),
    }
    return xy, heading, meta


# --------------------------------------------------------------------------- #
# 2. Frame VERIFICATION — the guards, and the blind spot one of them has       #
# --------------------------------------------------------------------------- #
#: Windows below these are excluded from the sign check: a straight path has no
#: turn sign to agree with, and near-zero lateral offset has no reliable sign.
SIGN_MIN_TURN_RAD = math.radians(5.0)
SIGN_MIN_LAT_M = 0.5
SIGN_MIN_AGREE = 0.90
SIGN_MIN_N = 8


def assert_lateral_sign_convention(xy, heading, *,
                                   min_turn_rad: float = SIGN_MIN_TURN_RAD,
                                   min_lat_m: float = SIGN_MIN_LAT_M,
                                   min_agree: float = SIGN_MIN_AGREE,
                                   min_n: int = SIGN_MIN_N) -> dict:
    """⭐ Catch a ``y``-SIGN FLIP, which ``lateral.assert_axis_convention`` cannot.

    ⛔ MEASURED (this module's own regression arm,
    ``test_assert_axis_convention_is_blind_to_a_y_sign_flip``): the harness's
    verifier tests that ``mean |axis0|`` DOMINATES ``mean |axis1|`` and matches
    ``v*K*dt`` (``lateral.py:183-203``). Negating ``y`` changes neither
    magnitude, so a right-handed -> left-handed flip passes it **unchanged**.
    That flip would invert every ``cross_bias_m`` sign in the LATERAL family —
    the programme's least-served axis — while every guard stayed green.

    The independent evidence is NavSim's own ``heading`` column: a left turn has
    ``+`` net yaw change AND ``+`` net lateral displacement. Requires a heading
    channel that was NOT derived from ``xy``; with none, the flip is
    **undetectable** and this returns an honest UNAVAILABLE rather than a pass.
    """
    if heading is None:
        return {"status": "UNAVAILABLE",
                "reason": ("no independent heading channel. The y-sign cannot be "
                           "checked from xy alone — a path tangent derived from "
                           "the same array flips with it, so the test would be "
                           "circular. ⛔ Supply NavSim's (x, y, heading) third "
                           "column. Without it a right-handed/left-handed flip "
                           "is UNDETECTABLE and every cross-track sign is "
                           "unverified."),
                "n": 0}
    p = _arr(xy, "xy")
    h = _arr(heading, "heading")
    if p.ndim != 3 or p.shape[-1] != 2:
        raise FrameConventionError(f"xy must be [N,K,2], got {p.shape}")
    if h.shape != p.shape[:2]:
        raise FrameConventionError(
            f"heading {h.shape} does not align with xy {p.shape[:2]}")
    dyaw = _wrap(h[:, -1] - h[:, 0])
    lat = p[:, -1, 1]
    keep = (np.abs(dyaw) >= min_turn_rad) & (np.abs(lat) >= min_lat_m)
    n = int(keep.sum())
    ev = {"n_windows": int(p.shape[0]), "n_turning_windows": n,
          "min_turn_deg": round(math.degrees(min_turn_rad), 2),
          "min_lat_m": min_lat_m,
          "_what_this_catches": ("a y-sign flip (right-handed frame). "
                                 "lateral.assert_axis_convention CANNOT: "
                                 "negating y leaves |y| unchanged."),
          "_independent_channel": "NavSim (x, y, heading) vehicle yaw"}
    if n < min_n:
        ev.update(status="UNAVAILABLE",
                  reason=(f"only {n} window(s) turn by >= "
                          f"{math.degrees(min_turn_rad):.1f} deg with >= "
                          f"{min_lat_m} m lateral offset (need {min_n}). A "
                          f"straight-only set carries NO evidence about the y "
                          f"sign — reporting it as verified would be a pass "
                          f"manufactured from an absent test."),
                  n=n)
        return ev
    agree = float((np.sign(dyaw[keep]) == np.sign(lat[keep])).mean())
    ev.update(status="OK", agreement=round(agree, 4), n=n, verified=True)
    if agree < min_agree:
        raise FrameConventionError(
            f"⛔ LATERAL SIGN CONVENTION VIOLATED: net yaw change and net "
            f"lateral displacement agree in sign on only {agree:.1%} of {n} "
            f"turning windows (need {min_agree:.0%}). Our frame is y-LEFT with "
            f"CCW-positive yaw, so a LEFT turn must give +y. A near-0 % "
            f"agreement is a y-sign flip (right-handed frame); ~50 % is noise "
            f"or a mismatched heading channel.")
    return ev


def verify_frame(gt_xy, *, dt_s: float, speed=None, heading=None,
                 tol: float = 0.35) -> dict:
    """⭐ Run BOTH frame guards and return the evidence. Raises on a violation.

    Guard 1 is the harness's OWN verifier, ``lateral.assert_axis_convention``
    (``taniteval/taniteval/lateral.py:170-205``) — seam 11 of the audit's adapter
    table, whose instruction is *"Call it in the adapter's own tests"*. It is
    called HERE, in the adapter itself, so a caller cannot skip it.
    Guard 2 is :func:`assert_lateral_sign_convention`, which covers guard 1's
    documented blind spot.
    """
    from taniteval import lateral as _lat
    axis = _lat.assert_axis_convention(np.asarray(gt_xy, dtype=np.float64),
                                       speed=speed, dt=float(dt_s), tol=tol)
    sign = assert_lateral_sign_convention(gt_xy, heading)
    return {
        "axis_convention": axis,
        "axis_convention_instrument": "taniteval.lateral.assert_axis_convention (lateral.py:170)",
        "lateral_sign": sign,
        "lateral_sign_instrument": f"{ADAPTER}::assert_lateral_sign_convention",
        "_both_needed": ("assert_axis_convention catches a TRANSPOSE ((y,x) or "
                         "swapped axes); it is blind to a y-SIGN FLIP because "
                         "|y| is unchanged by negation. The sign guard covers "
                         "exactly that gap and needs an independent heading "
                         "channel to do it."),
        "navsim_frame": NAVSIM_EGO_FRAME,
        "tanitad_frame": TANITAD_WIN_FRAME,
    }


# --------------------------------------------------------------------------- #
# 3. ⛔ THE COLUMN TRAP, encoded in code                                        #
# --------------------------------------------------------------------------- #
_COLUMN_TRAP_WHY = (
    "⛔ `pdm_score` IS NOT THE EPDMS. PDMScorer._aggregate_pdm_scores masks "
    "Extended Comfort OUT and divides by 5+5+2+2 = 14. The real EPDMS "
    "(denominator 16) is assembled downstream in "
    "run_pdm_score.py::compute_final_scores, which injects "
    "two_frame_extended_comfort into the weighted vector and recomputes "
    "`score = multiplicative_metrics_prod * (sum w*m / sum w)`. Reading "
    "`pdm_score` yields a 14-denominator, EC-free number THAT LOOKS LIKE A "
    "VALID EPDMS. Read the `score` column. "
    "(NAVSIM_PROTOCOL.md:281-290; CRITERIA_REGISTRY.json "
    "benchmarks.navsim.THE_COLUMN_TRAP.)")


def refuse_pdm_score_as_epdms(extra: str = "") -> None:
    """⛔ One place the refusal is worded, so it cannot drift between call sites."""
    raise PdmScoreColumnError(_COLUMN_TRAP_WHY + (f" {extra}" if extra else ""))


def read_epdms(row, *, variant: str = "EPDMS_v2") -> dict:
    """Read the EPDMS/PDMS from a NavSim result row. ⛔ REFUSES ``pdm_score``.

    ``row`` is a dict-like result row (a ``pandas`` row's ``.to_dict()``, or the
    parsed CSV row). Returns the value WITH its provenance, never a bare float —
    a bare float is exactly what loses the column it came from.
    """
    if variant not in VARIANTS:
        raise NavSimAdapterError(
            f"variant must be one of {sorted(VARIANTS)}, got {variant!r}. "
            f"⛔ Four mutually incomparable protocols exist "
            f"(NAVSIM_PROTOCOL.md:549-556); an unnamed variant is not a result.")
    try:
        row = dict(row)
    except Exception as e:
        raise NavSimAdapterError(
            f"row must be dict-like, got {type(row).__name__} ({e})") from e

    has_score = EPDMS_COLUMN in row and row[EPDMS_COLUMN] is not None
    has_pdm = FORBIDDEN_EPDMS_COLUMN in row and row[FORBIDDEN_EPDMS_COLUMN] is not None
    if not has_score:
        if has_pdm:
            refuse_pdm_score_as_epdms(
                f"The row offers only {FORBIDDEN_EPDMS_COLUMN!r} "
                f"(= {row[FORBIDDEN_EPDMS_COLUMN]!r}) and no {EPDMS_COLUMN!r} "
                f"column. Re-export from run_pdm_score.py::compute_final_scores; "
                f"do NOT substitute.")
        raise NavSimAdapterError(
            f"row has neither {EPDMS_COLUMN!r} nor {FORBIDDEN_EPDMS_COLUMN!r}; "
            f"keys={sorted(row)[:24]}")

    val = float(row[EPDMS_COLUMN])
    spec = VARIANTS[variant]
    out = {
        "value": val,
        "column": EPDMS_COLUMN,
        "variant": variant,
        "denominator": spec["denominator"],
        "multipliers": list(spec["multipliers"]),
        "weights": dict(spec["weighted"]),
        "_column_trap": _COLUMN_TRAP_WHY,
        "_devkit_pin": DEVKIT_PIN,
    }
    if has_pdm:
        pv = float(row[FORBIDDEN_EPDMS_COLUMN])
        out["pdm_score_column_present"] = True
        # ⛔ deliberately NOT named "epdms_*" anywhere. It is carried only so the
        # gap between the two columns is visible; a consumer that reads this key
        # has to read its label to get there.
        out["not_the_epdms__pdm_score_value"] = pv
        out["not_the_epdms__delta"] = round(val - pv, 6)
    else:
        out["pdm_score_column_present"] = False
    return out


def submetrics_from_row(row, *, variant: str = "EPDMS_v2") -> dict:
    """Every sub-metric the variant defines, or an explicit per-term absence.

    ``navsim.submetrics`` is a REQUIRED criterion (``CRITERIA_REGISTRY.json``)
    because *"a composite alone hides which term moved"*.
    """
    if variant not in VARIANTS:
        raise NavSimAdapterError(f"unknown variant {variant!r}")
    row = dict(row)
    spec = VARIANTS[variant]
    terms = list(spec["multipliers"]) + list(spec["weighted"])
    out = {"variant": variant, "denominator": spec["denominator"],
           "_note": spec["note"]}
    missing = []
    for t in terms:
        key = next((k for k in (t, t.lower()) if k in row and row[k] is not None), None)
        if key is None:
            missing.append(t)
            out[t] = {"status": "UNAVAILABLE",
                      "reason": (f"sub-metric {t!r} is not a column of the "
                                 f"supplied row. It is part of {variant} "
                                 f"(denominator {spec['denominator']}), so its "
                                 f"absence is a MISSING TERM, not a zero."),
                      "n": 0}
        else:
            out[t] = {"value": float(row[key]),
                      "role": ("multiplier" if t in spec["multipliers"]
                               else f"weighted w={spec['weighted'][t]}")}
    out["_missing_terms"] = missing
    return out


# --------------------------------------------------------------------------- #
# 4. Refusals — the estimator, and the families NavSim does not have           #
# --------------------------------------------------------------------------- #
ESTIMATOR_UNSETTLED_REASON = (
    "cluster unit unsettled — NavSim's resampling unit is a SCENE TOKEN, not an "
    "episode. taniteval.ci.episode_cluster_bootstrap resamples EPISODES, and "
    "the two are not interchangeable here: docs/splits.md states verbatim that "
    "\"NavSim splits contain overlapping scenes\", so scene tokens are NOT "
    "independent draws, and clustering by nuPlan log (the obvious alternative) "
    "has never been pre-registered for this programme. ⛔ Emitting either would "
    "manufacture a decision-grade interval from an unsettled unit — worse than "
    "no interval, because it would look valid. The point estimate below is the "
    "full_set pooled mean and is unaffected. OPEN DESIGN QUESTION FOR THE PI: "
    "NAVSIM_PROTOCOL.md sec 8.5 and sec 9 item 15.")


def estimator_refusal(n: int) -> dict:
    """The interval's honest n/a — reason + n, the harness's inline idiom.

    ``tools/criteria_check.py:131-145`` reads ``{"status", "reason", "n"}`` as
    REFUSED (a work item); the SAME object without a ``reason`` is read as
    ABSENT (a silent omission). Both fields are therefore mandatory here.
    """
    return {"status": "UNAVAILABLE",
            "reason": ESTIMATOR_UNSETTLED_REASON,
            "n": int(n),
            "n_windows_it_would_have_had": int(n),
            "estimator_if_it_were_settled": "episode_cluster_bootstrap (taniteval/ci.py)",
            "_is_a_work_item": ("⛔ A PI DECISION, not an implementation gap. The "
                                "adapter deliberately passes eid=None so the "
                                "harness's own guard (four_families.py:658-664) "
                                "fires on every interval rather than returning a "
                                "scene-token bootstrap."),
            "_forbidden": ("⛔ overlapping_holdout_se is NOT a fallback here. It "
                           "BIASES THE POINT ESTIMATE (mean-of-split-means): "
                           "MEASURED -6.67 % to +11.69 % on headline ade_0_2s "
                           "across 27 arms, bidirectional.")}


def navsim_native_family_refusals(n: int) -> dict:
    """What **NavSim's own instrument** does not measure — per family, with n.

    ⛔ This is NOT a statement about the ``four_families`` block, which runs OUR
    instruments on the trajectory geometry and CAN produce LONGITUDINAL and
    LATERAL numbers. It is the honest boundary of the BENCHMARK. Mapping table:
    ``NAVSIM_PROTOCOL.md:803-814``.
    """
    n = int(n)
    return {
        "_what_this_block_is": (
            "the criteria that NAVSIM'S OWN SCORE cannot supply, by "
            "construction. Our four_families block is computed separately from "
            "the trajectory geometry and is NOT limited by this list. Never "
            "merge the two."),
        "longitudinal_target_speed": {
            "status": "UNAVAILABLE",
            "reason": ("NavSim publishes no target-speed term. EP (ego "
                       "progress) is a normalised progress ratio and TTC is a "
                       "BINARY within-bound flag, neither of which is a speed "
                       "error. ⇒ absent from the BENCHMARK; our own "
                       "four_families.longitudinal computes it from the "
                       "trajectory instead. NAVSIM_PROTOCOL.md:807."),
            "n": n},
        "longitudinal_distance_keeping_continuous": {
            "status": "UNAVAILABLE",
            "reason": ("NavSim's TTC is a BINARY within-bound flag on a 0.9 s "
                       "projection, not a headway in metres, a time-gap in "
                       "seconds, or a continuous TTC. Our binding criterion "
                       "needs the continuous quantity, which requires a lead "
                       "block (four_families.py:1153-1155). PARTIAL coverage "
                       "only. NAVSIM_PROTOCOL.md:806."),
            "n": n},
        "lateral_heading_curvature_yaw": {
            "status": "UNAVAILABLE",
            "reason": ("NavSim has no heading, curvature or yaw-rate ERROR. Its "
                       "yaw-rate (0.95 rad/s) and yaw-accel (1.93 rad/s^2) "
                       "bounds are FEASIBILITY GATES inside comfort, and LK is "
                       "a thresholded centreline flag, not a cross-track error "
                       "in metres. ⇒ absent from the BENCHMARK; our own "
                       "four_families.lateral computes all four from the "
                       "trajectory. NAVSIM_PROTOCOL.md:809-810."),
            "n": n},
        "tactical_declared_decision": {
            "status": "UNAVAILABLE",
            "reason": ("NavSim scores the EXECUTED trajectory and never a "
                       "manoeuvre decision, so 'selected vs executed' is "
                       "unobtainable from it. ⚠️ Our four_families.tactical "
                       "block on this data is TRAJECTORY-DERIVED — both label "
                       "streams are EXECUTED manoeuvres (the arm's and the "
                       "human's), which four_families.py:828-833 states is "
                       "explicitly NOT 'selected vs executed'. Closing this "
                       "needs a tactical head. NAVSIM_PROTOCOL.md:811."),
            "n": n},
        "strategic_decision_and_route_goal": {
            "status": "UNAVAILABLE",
            "reason": ("NavSim's `driving_command` is an INPUT to the agent and "
                       "is never scored, so the benchmark reports no strategic "
                       "decision or route/goal quality at all. ⚠️ It therefore "
                       "also cannot DETECT a route-head echo — the defect "
                       "measured on flagship v1 (369/369 bijection, scored "
                       "1.0000). NAVSIM_PROTOCOL.md:812, :828-829."),
            "n": n},
        "_added_by_navsim_that_we_do_not_instrument": {
            "safety": "NC, DAC, DDC, TLC — four rule-compliance terms we have no family for",
            "comfort": "C / HC, and EC (a CROSS-FRAME consistency term we do not have)",
            "_note": ("these are a REASON TO ADOPT NavSim, recorded here so the "
                      "trade is visible in both directions. NAVSIM_PROTOCOL.md:813-814."),
        },
    }


# --------------------------------------------------------------------------- #
# 5. The win dict                                                              #
# --------------------------------------------------------------------------- #
def _sparse_view(K: int, wp_steps=None):
    """The sparse companion index/contract — EXACT port of ``ff_rescore.py:399-403``.

    Ported rather than imported because ``ff_rescore`` is a CLI module whose
    import pulls in the census/duplicate machinery; the values are pinned by
    ``test_sparse_companion_view_matches_ff_rescore``.
    """
    if wp_steps and len(wp_steps) == K:
        return list(range(K)), list(wp_steps)
    idx = sorted({max(0, int(round(K * q)) - 1) for q in (.25, .5, .75, 1.0)})
    return idx, [i + 1 for i in idx]


def scenes_to_win(pred_poses, gt_poses=None, *, frame: str,
                  origin_included: bool, dt_s: float,
                  origin_xy=None, origin_yaw=None,
                  scene_tokens=None, ego_speed_mps=None,
                  verify: bool = True, tol: float = 0.35) -> dict:
    """NavSim scenes -> the harness ``win`` dict. The core of this adapter.

    ⛔ ``win["eid"]`` is set to **None** deliberately — see the module docstring.
    The scene tokens are preserved under ``win["_navsim"]["scene_tokens"]``.

    ⛔ ``dt_s`` has NO DEFAULT. It is the spacing of the DENSE columns. NavSim's
    agent output and its simulator disagree: the submitted ``Trajectory`` is
    typically 8 poses at ``interval_length=0.5`` s, while
    ``proposal_sampling: num_poses: 40, interval_length: 0.1``
    (``NAVSIM_PROTOCOL.md:415-417``) is the SIMULATED state array. Speed scales
    as ``1/dt`` and acceleration as ``1/dt^2``: the same defect on our own corpus
    read GT ego speed **12.4565 m/s as 62.9789 m/s (x5.0559)** on 859 real
    windows (``four_families.py:1199-1205``).

    ``gt_poses`` is the HUMAN/expert future. Without it every geometry family is
    refused with its reason and its ``n`` — never silently dropped.
    """
    dt_s = float(dt_s)
    if not (dt_s > 0):
        raise NavSimAdapterError(
            f"dt_s must be > 0, got {dt_s!r}. It is the DENSE column spacing and "
            f"is never assumed: speed ~ 1/dt, accel ~ 1/dt^2.")

    pred, pred_head, pred_meta = poses_to_waypoints(
        pred_poses, frame=frame, origin_included=origin_included,
        origin_xy=origin_xy, origin_yaw=origin_yaw, name="pred_poses")
    N, K = pred.shape[0], pred.shape[1]

    gt = gt_head = None
    gt_meta = None
    if gt_poses is not None:
        gt, gt_head, gt_meta = poses_to_waypoints(
            gt_poses, frame=frame, origin_included=origin_included,
            origin_xy=origin_xy, origin_yaw=origin_yaw, name="gt_poses")
        if gt.shape != pred.shape:
            raise FrameConventionError(
                f"gt {gt.shape} != pred {pred.shape}. ⛔ Refusing to truncate: "
                f"scoring an arm's path against a differently-gridded human path "
                f"produces a plausible number, not an exception.")

    v0 = None
    if ego_speed_mps is not None:
        v0 = np.asarray(ego_speed_mps, dtype=np.float64).reshape(-1)
        if v0.shape[0] != N:
            raise NavSimAdapterError(
                f"ego_speed_mps has {v0.shape[0]} entries for {N} scenes")

    tokens = ([str(t) for t in scene_tokens] if scene_tokens is not None else None)
    if tokens is not None and len(tokens) != N:
        raise NavSimAdapterError(
            f"scene_tokens has {len(tokens)} entries for {N} scenes")

    frame_evidence = None
    if verify:
        if gt is None:
            frame_evidence = {
                "status": "UNAVAILABLE",
                "reason": ("no gt_poses supplied, and the frame verifiers are "
                           "defined on the GROUND-TRUTH path "
                           "(lateral.assert_axis_convention takes gt_dense). "
                           "⛔ Verifying the prediction instead would check the "
                           "MODEL's frame, not the corpus's, and a model that "
                           "learned a flipped frame would verify as clean."),
                "n": int(N)}
        else:
            frame_evidence = verify_frame(gt, dt_s=dt_s, speed=v0,
                                          heading=gt_head, tol=tol)

    idx, contract = _sparse_view(K)
    win = {
        # ---- REQUIRED (ff_rescore.py:404-406) ----
        "pred_dense": pred,
        "gt_dense": gt,
        "pred": pred[:, idx],
        "gt": (gt[:, idx] if gt is not None else None),
        "wp_steps": contract,
        "dt_s": dt_s,
        # ⛔ DELIBERATELY None — see the module docstring and estimator_refusal().
        "eid": None,
        # ---- OPTIONAL, each unlocks a family ----
        "v0": v0,
        "speed": v0,
        "lead": None,
        # ---- provenance ----
        "_navsim": {
            "adapter": ADAPTER,
            "protocol_doc": PROTOCOL_DOC,
            "devkit_pin": DEVKIT_PIN,
            "n_scenes": int(N),
            "horizon_steps": int(K),
            "horizon_s": round(K * dt_s, 4),
            "dt_s": dt_s,
            "dt_provenance": ("DECLARED by the caller — NavSim's agent output "
                              "(interval_length 0.5 s, 8 poses) and its "
                              "simulator state array (0.1 s, 41 states) differ; "
                              "NAVSIM_PROTOCOL.md:415-417."),
            "scene_tokens": tokens,
            "eid_is_none_because": ESTIMATOR_UNSETTLED_REASON,
            "pred_meta": pred_meta,
            "gt_meta": gt_meta,
            "frame_verification": frame_evidence,
            "navsim_heading_carried_separately": {
                "pred": pred_head is not None,
                "gt": gt_head is not None,
                "why": ("NavSim's third pose column is VEHICLE YAW. Our LATERAL "
                        "family's heading is the PATH TANGENT atan2(dy,dx) "
                        "(four_families.py:170). They coincide only under no "
                        "side-slip, so the NavSim channel is used ONLY as "
                        "independent evidence for the y-sign guard and is never "
                        "substituted for the family's own value."),
            },
            "lead_block": {
                "status": "UNAVAILABLE",
                "reason": ("no lead block built. NavSim/nuPlan DO carry the "
                           "inputs (sample_annotation boxes + ego_pose, audit "
                           "seam 6 taniteval/lead_source.py:329), so this is "
                           "buildable — but it needs the annotation stream, "
                           "which the sensor-free logs alone do not give. Until "
                           "then the distance-keeping half of LONGITUDINAL is a "
                           "WORK ITEM, not a pass."),
                "n": int(N)},
        },
    }
    if gt is None:
        win["_navsim"]["gt_absent"] = {
            "status": "UNAVAILABLE",
            "reason": ("no human/expert future trajectory supplied, so no "
                       "geometry family can be computed at all. NavSim's Scene "
                       "object carries it but is TRAINING-ONLY for the agent "
                       "(NAVSIM_PROTOCOL.md:470-474) — it must be read offline "
                       "from the log, exactly as the scorer does."),
            "n": int(N)}
    if pred_head is not None:
        win["_navsim"]["navsim_pred_heading_rad"] = pred_head
    if gt_head is not None:
        win["_navsim"]["navsim_gt_heading_rad"] = gt_head
    return win


def four_families_block(win: dict, *, tier: str, n_boot: int = 2000,
                        seed: int = 0) -> dict:
    """``all_families`` on a NavSim win, or a full block of honest refusals.

    ⛔ ``tier`` is REQUIRED and has no default — ``all_families`` emits
    ``⛔_tier_missing`` and suppresses the T0 echo warning when it is absent
    (``four_families.py:1252-1265``), and ``ff_rescore.resolve_tier:126-141``
    treats an unstamped arm as a hard error.
    """
    if tier not in TIERS:
        raise NavSimAdapterError(
            f"tier must be one of {TIERS}, got {tier!r}. ⛔ There is no default. "
            f"See tier_rationale() for how NavSim maps onto the vocabulary.")
    n = int(win["pred_dense"].shape[0])
    if win.get("gt_dense") is None:
        why = win.get("_navsim", {}).get("gt_absent", {}).get(
            "reason", "no ground-truth path supplied")
        block = {k: {"status": "UNAVAILABLE", "reason": why, "n": n, "tier": tier}
                 for k in ("longitudinal", "lateral", "tactical", "strategic")}
        block["_families_unavailable"] = ["longitudinal", "lateral",
                                          "tactical", "strategic"]
        block["_complete"] = False
        block["_rule_satisfied"] = True   # clause 5: reason + n given per family
        block["_tier"] = tier
        block["_binding_rule"] = (
            "Sayed 2026-08-02: LONGITUDINAL + LATERAL + TACTICAL + STRATEGIC in "
            "ADDITION to ADE, per-family, never pooled. A family reported "
            "UNAVAILABLE is a WORK ITEM, not a pass.")
        return block

    from taniteval import four_families as ff
    # ⛔ strategic_no_label=True is CORRECT for NavSim but for a DIFFERENT reason
    # than PhysicalAI's: NavSim HAS a map and a route, it simply never SCORES
    # them (NAVSIM_PROTOCOL.md:812). The reason is restated below so the
    # PhysicalAI-specific text inside strategic_unavailable() cannot be read as
    # the NavSim fact.
    fam = ff.all_families(win, tactical_from_traj=True,
                          strategic_no_label=True, tier=tier,
                          n_boot=n_boot, seed=seed)
    fam["strategic"]["navsim_specific_reason"] = (
        "⚠️ The generic reason above is PhysicalAI-specific (no map in the "
        "corpus). On NavSim the fact is DIFFERENT and must not be conflated: "
        "the lane graph and route EXIST, but `driving_command` is an INPUT and "
        "is never scored, and the Scene object carrying the map is TRAINING-ONLY "
        "at inference (NAVSIM_PROTOCOL.md:470-474, :812). So STRATEGIC is absent "
        "because the BENCHMARK declines to measure it, not because the data "
        "lacks it — which makes it a buildable WORK ITEM here (audit seam 7, "
        "taniteval/strategic_optionset.py:193), unlike on PhysicalAI.")
    fam["_navsim_estimator"] = estimator_refusal(n)
    return fam


def tier_rationale() -> dict:
    """Why NavSim maps onto T1, and where that mapping is genuinely open."""
    return {
        "recommended_single_stage": "T1",
        "why": ("T0 is TEACHER-FORCED — the predictor consumes the RECORDED "
                "future actions. A NavSim agent consumes NO future actions: it "
                "emits a 4 s plan from the initial observation and that plan is "
                "what gets executed by the LQR + kinematic-bicycle propagation "
                "(NAVSIM_PROTOCOL.md:424-435). Nothing recorded is fed back. ⇒ "
                "T0 is WRONG; T1 (the arm's own output is executed) is the "
                "closest member of our vocabulary."),
        "⚠️_where_it_is_imperfect": (
            "T1 in our vocabulary is an ACTION-CLOSED loop. NavSim is NOT "
            "closed: the agent is queried exactly ONCE per scene and receives no "
            "environmental feedback (NAVSIM_PROTOCOL.md:424-428). It is "
            "open-loop planning scored through a simulated ego. Calling it T1 "
            "overstates the closure; calling it T0 misstates the mechanism. "
            "State the mechanism alongside the stamp."),
        "⚠️_two_stage_is_arguably_T2": (
            "NavSim v2 Stage 2 scores the agent on PRE-RENDERED MTGS "
            "observations — a re-perception step, which is what CRITERIA_REGISTRY "
            "calls T2. ⛔ But ff_rescore.resolve_tier and t1_eval.resolve_tiers "
            "accept ONLY T0/T1 (audit section D), so a T2 stamp is refused "
            "downstream today. Unresolved; do not stamp T2 without extending "
            "those two resolvers first."),
    }


# --------------------------------------------------------------------------- #
# 6. Vision-only compliance                                                    #
# --------------------------------------------------------------------------- #
def navsim_inference_inputs(*, cameras=True, lidar=False,
                            ego_velocity=True, ego_acceleration=True,
                            ego_pose_history=True, driving_command=True,
                            claimed_abstention_from_ego: bool = False) -> dict:
    """What the agent consumed at inference — recorded HONESTLY.

    ⛔ Defaults reflect what NavSim ACTUALLY hands every agent. ``EgoStatus`` is
    always populated and *"there is no switch that removes ego status"*, nor does
    anything in the framework verify that an agent declined to read it
    (``NAVSIM_PROTOCOL.md:488-492``). NavSim publishes **no fully ego-free
    number**: even the camera-only LTF baseline consumes ``driving_command +
    ego_velocity + ego_acceleration``.
    """
    ego_used = [n for n, on in (("ego_velocity", ego_velocity),
                                ("ego_acceleration", ego_acceleration),
                                ("ego_pose_history", ego_pose_history)) if on]
    inputs = {
        "cameras": {"used": bool(cameras),
                    "spec": "8 cameras, 1920x1080 (cam_f0,l0,l1,l2,r0,r1,r2,b0)",
                    "privileged_under_vision_only_rule": False},
        "lidar": {"used": bool(lidar),
                  "spec": "merged 5-sensor point cloud, (6, n) float32",
                  "privileged_under_vision_only_rule": False,
                  "⚠️": ("a SENSOR but NOT vision — excluded under a strict "
                         "camera-only reading of the binding rule.")},
        "ego_velocity": {"used": bool(ego_velocity), "privileged": True},
        "ego_acceleration": {"used": bool(ego_acceleration), "privileged": True},
        "ego_pose_history": {"used": bool(ego_pose_history), "privileged": True,
                             "spec": "4 history frames (num_history_frames=4)"},
        "driving_command": {
            "used": bool(driving_command),
            "privileged": False,
            "dim": 4,
            "⚠️_cardinality": ("4-dim (left, forward, right, UNKNOWN) in the "
                               "devkit and docs/agents.md. The paper says 3. "
                               "IMPLEMENT 4 — NAVSIM_PROTOCOL.md:502-504."),
            "kind": ("a ROUTE/GOAL signal, not ego kinematics: derived from the "
                     "lane graph 20 m ahead, and explicitly disentangled from "
                     "obstacles and traffic signs. Admissible in principle under "
                     "the goal-input rule."),
        },
        "history_frames": 4,
    }
    vision_only = not ego_used
    return {
        "inputs": inputs,
        "ego_channels_consumed": ego_used,
        "vision_only": vision_only,
        "claimed_abstention_from_ego": bool(claimed_abstention_from_ego),
        "⛔_framework_cannot_verify": (
            "NavSim ALWAYS populates EgoStatus and has no switch to remove it; "
            "nothing in the framework checks that an agent declined to read it "
            "(NAVSIM_PROTOCOL.md:488-492). ⇒ a vision-only claim on NavSim must "
            "be enforced and evidenced ON OUR SIDE (e.g. by zeroing the channel "
            "in our own agent wrapper and banking the diff). A bare "
            "'camera-only' label on a leaderboard row is the AUTHORS' claim, "
            "never the server's (:542-543)."),
        "_no_ego_free_published_number": (
            "NavSim publishes no fully ego-free result. Design against "
            "TransFuser B1 'Goal only' = 81.8 PDMS."),
    }


def route_leak_check(status: str = "UNVERIFIED", evidence: str = "") -> dict:
    """The ``navsim.route_leak_check`` criterion — never silently 'fine'."""
    if status == "UNVERIFIED":
        return {"status": "UNAVAILABLE",
                "reason": ("whether nuPlan's `route_roadblock_ids` is itself "
                           "derived from the EXPERT'S DRIVEN PATH is UNVERIFIED "
                           "(NAVSIM_PROTOCOL.md:511-515, sec 9 item 4). The devkit "
                           "reads it straight from the nuPlan DB with no "
                           "derivation exposed. ⛔ Same family as our own rule "
                           "that a supplied route is optimistic by construction, "
                           "and as the flagship route-head echo. Settle it "
                           "BEFORE quoting a route-conditioned NavSim result."),
                "n": 0}
    return {"status": "OK", "checked": status, "evidence": evidence}


# --------------------------------------------------------------------------- #
# 7. The artifact                                                              #
# --------------------------------------------------------------------------- #
def build_artifact(win: dict, *, tier: str, variant: str = "EPDMS_v2",
                   split: str | None = None, arm: str = "unnamed",
                   epdms=None, submetrics=None,
                   inference_inputs=None, goal_source: str | None = None,
                   route_leak: dict | None = None,
                   n_boot: int = 2000, seed: int = 0) -> dict:
    """The full artifact, shaped for ``tools/criteria_check.py``.

    Key paths are the registry's (``CRITERIA_REGISTRY.json``): ``four_families.*``
    for the families, ``protocol.inference_inputs`` / ``protocol.goal_source``
    for the leak guards, ``benchmark.navsim.*`` for the five NavSim criteria,
    ``estimator.interval`` + ``tier`` + ``n_windows`` for hygiene.
    """
    if tier not in TIERS:
        raise NavSimAdapterError(f"tier must be one of {TIERS}, got {tier!r}")
    if variant not in VARIANTS:
        raise NavSimAdapterError(f"variant must be one of {sorted(VARIANTS)}")
    if split is not None and split not in SPLITS:
        raise NavSimAdapterError(
            f"split {split!r} is not one of {SPLITS}. ⛔ Four mutually "
            f"incomparable protocols exist; an unnamed split is not a result.")
    n = int(win["pred_dense"].shape[0])
    fam = four_families_block(win, tier=tier, n_boot=n_boot, seed=seed)
    ii = inference_inputs if inference_inputs is not None else navsim_inference_inputs()

    score_block = epdms if epdms is not None else {
        "status": "UNAVAILABLE",
        "reason": ("no NavSim `score` column supplied to this artifact. ⛔ It is "
                   "produced by run_pdm_score.py::compute_final_scores and "
                   "CANNOT be derived from the trajectories alone — EPDMS needs "
                   "the map, the boxes, the traffic lights and the LQR+bicycle "
                   "propagation. Read `score`, never `pdm_score`."),
        "n": 0}
    sub_block = submetrics if submetrics is not None else {
        "status": "UNAVAILABLE",
        "reason": ("no per-term sub-metrics supplied. A composite alone hides "
                   "which term moved, so the registry makes them REQUIRED."),
        "n": 0}

    return {
        "tool": ADAPTER,
        "arm": arm,
        # ⛔ tier at the TOP level: tools/criteria_check.resolve_tier reads
        # `block` FIRST and returns on any string it finds there, so a
        # `block: "taniteval.navsim/..."` value it does not recognise would
        # report this artifact as UNSTAMPED. `tier` is the path that resolves.
        "tier": tier,
        "n_windows": n,
        "n_episodes": {
            "status": "UNAVAILABLE",
            "reason": ("NavSim's unit is a scene TOKEN. Reporting a token count "
                       "as an episode count would assert the very clustering "
                       "this adapter refuses to assume. See estimator.interval."),
            "n": n},
        "n_scenes": n,
        "protocol": {
            "tier": tier,
            "tier_rationale": tier_rationale(),
            "benchmark": "NavSim",
            "variant": variant,
            "split": split,
            "protocol_doc": PROTOCOL_DOC,
            "devkit_pin": DEVKIT_PIN,
            "inference_inputs": ii,
            "vision_only": ii.get("vision_only"),
            "goal_source": goal_source or (
                "NavSim `driving_command` (4-dim one-hot), derived from the "
                "ROUTE LANE GRAPH 20 m ahead and explicitly disentangled from "
                "obstacles and traffic signs. ⚠️ Admissible under the "
                "goal/situation-disjoint rule ONLY while route_leak_check is "
                "settled: its upstream `route_roadblock_ids` provenance is "
                "UNVERIFIED."),
            "simulation_semantics": (
                "the agent is queried ONCE per scene; the plan is then fixed "
                "and propagated by an LQR + kinematic bicycle model at 10 Hz "
                "over 4 s, with NO environmental feedback. Submitted velocity "
                "and acceleration are DISCARDED — poses only "
                "(NAVSIM_PROTOCOL.md:424-440)."),
        },
        "four_families": fam,
        "benchmark": {"navsim": {
            "score": score_block,
            "variant": f"{variant} / {split or 'SPLIT-NOT-NAMED'}",
            "submetrics": sub_block,
            "ego_inputs": ii,
            "route_leak_check": route_leak or route_leak_check(),
            "families_absent_natively": navsim_native_family_refusals(n),
            "THE_COLUMN_TRAP": _COLUMN_TRAP_WHY,
        }},
        "estimator": {
            "point_estimate": "full_set pooled mean over scenes",
            "interval": estimator_refusal(n),
        },
        "_two_blocks": (
            "⛔ `four_families` is OUR instruments on NavSim TRAJECTORY "
            "GEOMETRY. `benchmark.navsim` is NAVSIM'S OWN score. They measure "
            "different things and must never be merged or averaged."),
    }


# --------------------------------------------------------------------------- #
# 8. ff_rescore-shaped dump — structurally compatible, NOT auto-wired          #
# --------------------------------------------------------------------------- #
def to_ff_dump(win: dict, label: str = "navsim") -> dict:
    """The ``load_dump`` return shape (``tools/ff_rescore.py:150-152``).

    ⛔ **NOT wired into ``ff_rescore.load_dump`` on purpose.** Adding the branch
    would make ``--dump navsim=...`` produce an episode-cluster bootstrap over
    SCENE TOKENS — exactly the interval this adapter refuses to invent, and
    ``load_dump:199-202`` hard-requires an ``eid`` per window, so the branch
    cannot be added without supplying one. ⇒ ``eid`` here is the scene-token
    list AND ``_cluster_unit_unsettled`` is True; a caller must settle the unit
    with the PI before this is fed to a bootstrap.
    """
    if win.get("gt_dense") is None:
        raise NavSimAdapterError(
            "cannot build an ff_rescore dump without gt_dense — every consumer "
            "below load_dump scores pred AGAINST gt.")
    tokens = win.get("_navsim", {}).get("scene_tokens")
    n = int(win["pred_dense"].shape[0])
    return {
        "kind": "navsim_scenes",
        "gt": np.asarray(win["gt_dense"], dtype=np.float64),
        "eid": (list(tokens) if tokens else [str(i) for i in range(n)]),
        "dt_s": float(win["dt_s"]),
        "dt_provenance": win.get("_navsim", {}).get("dt_provenance"),
        "wp_steps": None,      # dense path supplied; no sparse tick contract
        "source": PROTOCOL_DOC,
        "n_episodes": len(set(tokens)) if tokens else n,
        "v0": win.get("v0"),
        "arms": {label: (label, np.asarray(win["pred_dense"], dtype=np.float64))},
        "_cluster_unit_unsettled": True,
        "_estimator_refusal": estimator_refusal(n),
        "_not_wired": ("⛔ ff_rescore.load_dump has NOT been given a branch for "
                       "this. Wiring it would silently enable a scene-token "
                       "episode-cluster bootstrap. PI decision first."),
    }
