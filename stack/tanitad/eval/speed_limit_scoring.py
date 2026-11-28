"""Score a trajectory against a POSTED-LIMIT ceiling -- output-scored, never fed.

⭐ WHY THIS IS A SEPARATE MODULE FROM THE INPUT CHANNEL. ``tanitad.refs
.max_speed_input`` is an INPUT: it hands the planner a ceiling. This module is an
OUTPUT SCORE: it asks whether the planner's OWN proposed trajectory respected one.
They must not be the same object, because the whole admissibility argument turns
on the difference -- ``tanitad.eval.constraints``' opening rule is that *"a
constraint the planner must SATISFY cannot echo, because it is not an input."*

⛔ SO THIS FIRES WITH THE FLAG OFF, AND THAT IS THE POINT. An arm that was never
given a ceiling is still scoreable against one; that run is the HONEST BASELINE
against which a flag-ON arm must be read. Nothing here is ever fed to a model.

⛔⛔ THE VACUITY CONTROL THIS MODULE MAKES CHEAP, AND IT IS NOT OPTIONAL.
:func:`posted_limit_ceiling` will happily accept the RAW ``v_hi_ms`` as a
"ceiling". Scoring any trajectory against it is VACUOUS BY CONSTRUCTION -- the raw
value IS the max of that trajectory's own speed over the band, so
``frac_over_ceiling`` is exactly 0.0 for every clip, forever, whatever the planner
does. :func:`score_against_posted_limit` therefore returns the raw-ceiling score
BESIDE the quantized one under ``vacuity_control``, so a reader can see the metric
reading zero for a reason that has nothing to do with driving. A safety zero with
no control beside it is not a safety result.

⛔ UNITS. Every speed in and out is METRES PER SECOND; the returned block carries
``control_units = "m_s"`` and the ceiling's own provenance. A ceiling whose units
are not declared is refused upstream by
``max_speed_input.read_max_speed_field``.
"""
from __future__ import annotations

import numpy as np

from tanitad.eval import constraints as C
from tanitad.refs import max_speed_input as MSI

#: A posted limit does not vary within one 2-6 s tactical window, so the ceiling
#: is broadcast across the window's steps. Stated as a constant rather than
#: assumed, because a MAP service that returned a per-step limit would be a
#: different (better) signal and must not be silently conflated with this one.
CEILING_IS_CONSTANT_OVER_WINDOW = True


def posted_limit_ceiling(v_max_ms: float, n_steps: int, *,
                         mode: str = MSI.DEFAULT_MODE):
    """``(v_ceiling [K], ceiling_valid [K], meta)`` for one window.

    ``mode="quantized"`` snaps UP to the pinned posted-limit ladder;
    ``mode="raw"`` passes the value through and is the VACUITY CONTROL, not a
    deployable ceiling.
    """
    if mode not in MSI.MODES:
        raise ValueError(f"mode must be one of {MSI.MODES}, got {mode!r}")
    if v_max_ms is None or not np.isfinite(v_max_ms):
        return (np.full(n_steps, np.nan), np.zeros(n_steps, dtype=bool),
                {"control_units": MSI.CONTROL_UNITS, "mode": mode,
                 "status": "CENSORED", "reason": "no ceiling for this window"})
    if mode == "quantized":
        c, over = MSI.quantize_up(float(v_max_ms))
    else:
        c, over = float(v_max_ms), False
    return (np.full(n_steps, c), np.ones(n_steps, dtype=bool),
            {"control_units": MSI.CONTROL_UNITS, "mode": mode, "status": "OK",
             "ceiling_ms": c, "source_ms": float(v_max_ms),
             "source_over_top_step": bool(over),
             "steps_kmh": (list(MSI.POSTED_LIMIT_STEPS_KMH)
                           if mode == "quantized" else None),
             "constant_over_window": CEILING_IS_CONSTANT_OVER_WINDOW})


def score_against_posted_limit(v_planned, v_max_ms, *, manoeuvre_flag=None,
                               kinematic=None, kin_valid=None,
                               clearance=None, clr_valid=None) -> dict:
    """Two-sided envelope score against the POSTED-LIMIT ceiling, plus controls.

    ``v_planned`` [K] is the speed along the model's OWN proposed trajectory.
    The kinematic / clearance ceilings, when supplied, are combined elementwise
    with the posted limit exactly as ``constraints.combined_ceiling`` combines
    the other two -- a real limit is one more upper bound, not a replacement.

    Returns ``{"quantized": <report>, "vacuity_control": {"raw": <report>}, ...}``.
    ⛔ The vacuity control is ALWAYS computed. Quoting the quantized number
    without it would hide that the same instrument reads exactly 0.0 over on the
    raw ceiling for every clip, by construction.
    """
    v = np.asarray(v_planned, dtype=np.float64)
    out: dict = {"control_units": MSI.CONTROL_UNITS, "n_steps": int(v.size)}
    for tag, mode in (("quantized", "quantized"), ("raw", "raw")):
        c, ok, meta = posted_limit_ceiling(v_max_ms, v.size, mode=mode)
        if kinematic is not None and kin_valid is not None:
            c, ok = C.combined_ceiling(kinematic, kin_valid, c, ok)
        if clearance is not None and clr_valid is not None:
            c, ok = C.combined_ceiling(clearance, clr_valid, c, ok)
        rep = C.speed_envelope_report(v, v_ceiling=c, ceiling_valid=ok,
                                      manoeuvre_flag=manoeuvre_flag)
        rep["ceiling_meta"] = meta
        if tag == "quantized":
            out["quantized"] = rep
        else:
            out.setdefault("vacuity_control", {})["raw"] = rep
    out["_vacuity_note"] = (
        "the raw ceiling IS max(this trajectory's own speed over 2-6 s), so its "
        "frac_over_ceiling is 0.0 BY CONSTRUCTION and says nothing about the "
        "planner. Never quote the quantized number without it.")
    return out
