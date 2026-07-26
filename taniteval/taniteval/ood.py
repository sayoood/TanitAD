"""taniteval.ood — the OOD/EXTRAPOLATION guard, with its SATURATION declared.

⚠️ THE DEFECT THIS MODULE EXISTS TO REMOVE (MEASURED, 2026-07-26)
------------------------------------------------------------------
The closed-loop OOD guard maps a deviation ``(|dlat|, |dpsi|)`` to an ADE-ratio
through the **P1 MEASURED envelope** (``lowood_flagship_ci.json``) with
``np.interp`` — which **CLAMPS** at the envelope edge, ``|dlat| = 3.0 m`` and
``|dpsi| = 12 deg``. Beyond that edge the ratio **SATURATES**: it stops growing
no matter how far the loop drifts.

Two consequences, both structural:

1. **Every long-horizon OOD ratio this program has quoted is a LOWER BOUND, not
   a measurement.** At the flagship-v4 30 k gate the ratio read **1.2741**
   ("under 1.5") while **54.63 % of steps exceeded 3 m** and **90.24 % of
   windows left the measured envelope**. E1a's own K=185 artifact is the same
   shape: ratio 1.2664 with ``frac_steps_lat_over_3m`` **0.5281** and
   ``frac_windows_any_step_out_of_envelope`` **0.9070**.
2. **The ``ratio > 1.5`` criterion structurally CANNOT fire out of envelope** —
   it is uninformative exactly where it matters most. A guard keyed on the ratio
   alone (``ood <= 1.30``) is therefore not an envelope test at all; the constant
   1.30 has no provenance as one, being merely the observed (saturated) K=185
   value.

E1a's stated rule was always a **DISJUNCTION** (``e1a_horizon.py:28-30``):

    "any horizon whose peak OOD ratio exceeds ~1.5x, **OR whose steps leave the
     measured envelope**, is EXTRAPOLATION, not measurement."

Only the first clause was ever implemented. This module implements the whole
rule, and makes the second clause **first-class alongside the ratio** rather
than an afterthought field nobody reads.

WHAT IS ENFORCED HERE
---------------------
* ``envelope_fractions`` is computed and returned **on every call** — you cannot
  obtain a ratio from this module without also obtaining the out-of-envelope
  fractions.
* ``verdict`` implements the DISJUNCTION, reports **which clause fired and why
  the other could not**, and stamps ``ratio_is_lower_bound`` whenever any step
  is outside.
* ``assert_envelope_verdict_consistent`` runs inside ``verdict`` on every emitted
  node, so **it is impossible to emit "within the measured envelope" while a
  majority of steps are outside it.** A saturating estimator must declare its own
  saturation; this is the mechanical form of that rule.

NUMERICS ARE UNCHANGED. ``OODMap.ratio_arr`` is byte-equivalent to
``e1a_horizon.OODMap.ratio_arr`` / ``v4_corridor_cl.OODMap.ratio_arr`` — the
ratio is not "fixed", because the ratio was never the error. What changes is
that the ratio can no longer be quoted, or adjudicated on, without its
saturation.

EVIDENCE CLASS. ``ENV_LAT_MAX`` / ``ENV_YAW_MAX`` are **MEASURED** (P1 sweep,
``lowood_flagship_ci.json``, on the flagship **v1** arm — not on v4).
``RATIO_EXTRAPOLATION_X`` (1.5) is **PROPOSED** — E1a's "~1.5x".
``MAJORITY_FRAC`` (0.5) is a **REPORTING CONVENTION, PROPOSED**: it only decides
whether the string says EXTRAPOLATION or PARTIAL EXTRAPOLATION. Both underlying
fractions are always printed, so no verdict depends on it being right.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from taniteval import ci as _ci
from taniteval.corridor import ENV_LAT_MAX, ENV_YAW_MAX

__all__ = ["ENV_LAT_MAX", "ENV_YAW_MAX", "RATIO_EXTRAPOLATION_X",
           "MAJORITY_FRAC", "SATURATION_NOTE", "RULE", "OODMap",
           "envelope_fractions", "verdict", "readjudicate",
           "assert_envelope_verdict_consistent", "EnvelopeVerdictError",
           "VERDICT_MEASUREMENT", "VERDICT_PARTIAL", "VERDICT_EXTRAPOLATION"]

# PROPOSED — E1a's "~1.5x" (e1a_horizon.py:28-30). Never MEASURED as a bound.
RATIO_EXTRAPOLATION_X = 1.5
# PROPOSED REPORTING CONVENTION — decides EXTRAPOLATION vs PARTIAL only.
MAJORITY_FRAC = 0.5

VERDICT_MEASUREMENT = "MEASUREMENT — every step stayed inside the MEASURED envelope"
VERDICT_PARTIAL = ("PARTIAL EXTRAPOLATION — a minority of windows leave the "
                   "MEASURED envelope; the OOD ratio is a LOWER BOUND there")
VERDICT_EXTRAPOLATION = ("EXTRAPOLATION — NOT a measurement at this horizon")

SATURATION_NOTE = (
    "OODMap.ratio_arr interpolates with np.interp, which CLAMPS at "
    f"|dlat|={ENV_LAT_MAX} m / |dyaw|={ENV_YAW_MAX} deg. Once steps leave the "
    "envelope the ratio SATURATES, so the ratio criterion structurally cannot "
    "fire there and the reported ratio is a LOWER BOUND, not a measurement.")

RULE = ("E1a's FULL disjunction (e1a_horizon.py:28-30): peak OOD ratio > ~"
        f"{RATIO_EXTRAPOLATION_X}x OR steps leave the MEASURED envelope "
        f"(|dlat|<={ENV_LAT_MAX} m, |dyaw|<={ENV_YAW_MAX} deg). Testing only "
        "the ratio half is the defect: the ratio saturates exactly where the "
        "envelope clause fires.")


# Every verdict string ever emitted for this quantity, mapped to what it CLAIMS.
# Comparing strings would make a re-wording look like a retraction and a
# retraction look like a re-wording; only the CLASS is load-bearing.
CLASS_MEASUREMENT = "MEASUREMENT"
CLASS_PARTIAL = "PARTIAL_EXTRAPOLATION"
CLASS_EXTRAPOLATION = "EXTRAPOLATION"
CLASS_UNKNOWN = "UNKNOWN"
# The legacy ratio-only string. It asserts the envelope held, so it classifies as
# MEASUREMENT — which is exactly why it was wrong at K=185.
LEGACY_RATIO_ONLY_STRING = "within the measured envelope on average"


def verdict_class(s) -> str:
    """What a verdict string CLAIMS, independent of its wording."""
    if not s:
        return CLASS_UNKNOWN
    t = str(s).strip().upper()
    if t.startswith("PARTIAL"):
        return CLASS_PARTIAL
    if t.startswith("EXTRAPOLATION"):
        return CLASS_EXTRAPOLATION
    if t.startswith("MEASUREMENT") or "WITHIN THE MEASURED ENVELOPE" in t:
        return CLASS_MEASUREMENT
    return CLASS_UNKNOWN


class EnvelopeVerdictError(AssertionError):
    """Raised when a node claims the envelope held while the steps say otherwise.

    This is the mechanical guarantee that a saturating estimator declares its own
    saturation. It is an ERROR, not a warning: the whole failure mode being fixed
    is a too-generous string surviving because nobody read the field beside it."""


# =========================================================================== #
# the P1 envelope map                                                          #
# =========================================================================== #
class OODMap:
    """P1 MEASURED envelope ``(|dlat|, |dpsi|) -> ADE ratio``, SATURATION-AWARE.

    Numerically identical to ``e1a_horizon.OODMap`` (the ratio is unchanged, so
    no published ratio moves). The difference is that this class refuses to hand
    you a ratio without also handing you the out-of-envelope fractions —
    :meth:`ratio_and_fractions` is the intended entry point, and
    :meth:`ratio_arr` carries the clamp in its own docstring.
    """

    def __init__(self, ci_json):
        d = (ci_json if isinstance(ci_json, dict)
             else json.loads(Path(ci_json).read_text(encoding="utf-8")))
        self.base = d["baseline_real_frames"]["mean"]
        self.lat_x = np.array([r["amount"] for r in d["conditions"]["lat"]])
        self.lat_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["lat"]])
        self.yaw_x = np.array([r["amount"] for r in d["conditions"]["yaw"]])
        self.yaw_y = np.array([r["ade2s_ci"]["mean"] for r in d["conditions"]["yaw"]])
        self.source = None if isinstance(ci_json, dict) else str(ci_json)
        # The envelope the interpolation was actually MEASURED over. Anything
        # beyond it is clamped, i.e. a lower bound.
        self.lat_max_measured = float(self.lat_x.max())
        self.yaw_max_measured = float(self.yaw_x.max())

    def ratio_arr(self, lat_abs, yaw_abs_deg):
        """The P1 ADE-ratio per step. ⚠️ ``np.interp`` **CLAMPS** at
        ``|dlat| = 3.0 m`` / ``|dyaw| = 12 deg``: beyond the envelope this is a
        **LOWER BOUND**, not a measurement. Use :meth:`ratio_and_fractions`
        unless you have already established the envelope holds."""
        lat_abs = np.asarray(lat_abs, dtype=float)
        yaw_abs_deg = np.asarray(yaw_abs_deg, dtype=float)
        al = np.interp(lat_abs, self.lat_x, self.lat_y)      # CLAMPS beyond 3.0 m
        ay = np.interp(yaw_abs_deg, self.yaw_x, self.yaw_y)  # CLAMPS beyond 12 deg
        ex_l = np.clip((al - self.base) / self.base, 0.0, None)
        ex_y = np.clip((ay - self.base) / self.base, 0.0, None)
        return 1.0 + ex_l + ex_y

    def ratio_and_fractions(self, lat_abs, yaw_abs_deg):
        """``(ratio, fractions)`` — the ONLY way to get a ratio out of this class
        together with the context that says whether it means anything."""
        return (self.ratio_arr(lat_abs, yaw_abs_deg),
                envelope_fractions(lat_abs, yaw_abs_deg))


# =========================================================================== #
# clause 2 — the out-of-envelope fractions, promoted to first class            #
# =========================================================================== #
def envelope_fractions(lat_abs, yaw_abs_deg) -> dict:
    """How much of the rollout left the MEASURED envelope.

    ``lat_abs`` / ``yaw_abs_deg`` are ``[n_windows, n_steps]`` (a 1-D array is
    treated as one window). Every fraction is over the SAME denominator it
    names, so none of them can be confused for another:

    * ``frac_steps_lat_over_3m``   — steps with ``|dlat| > 3.0 m``
    * ``frac_steps_yaw_over_12deg``— steps with ``|dyaw| > 12 deg``
    * ``frac_steps_any``           — steps outside on EITHER axis
    * ``frac_windows_any_step_out_of_envelope`` — windows with >= 1 such step
    """
    lat = np.atleast_2d(np.asarray(lat_abs, dtype=float))
    yaw = np.atleast_2d(np.asarray(yaw_abs_deg, dtype=float))
    out_lat = lat > ENV_LAT_MAX
    out_yaw = yaw > ENV_YAW_MAX
    out_any = out_lat | out_yaw
    return {
        "frac_steps_lat_over_3m": round(float(out_lat.mean()), 5),
        "frac_steps_yaw_over_12deg": round(float(out_yaw.mean()), 5),
        "frac_steps_any": round(float(out_any.mean()), 5),
        "frac_windows_any_step_out_of_envelope": round(
            float(out_any.any(axis=1).mean()), 4),
        "envelope": {"lat_max_m": ENV_LAT_MAX, "yaw_max_deg": ENV_YAW_MAX,
                     "provenance": "P1 MEASURED (lowood_flagship_ci.json), on "
                                   "the flagship v1 arm — NOT on v4"},
    }


# =========================================================================== #
# the DISJUNCTION                                                              #
# =========================================================================== #
def _verdict_string(ratio_fires, frac_windows_out, frac_steps_out):
    if ratio_fires or frac_steps_out > MAJORITY_FRAC or frac_windows_out > MAJORITY_FRAC:
        return VERDICT_EXTRAPOLATION
    if frac_windows_out > 0.0 or frac_steps_out > 0.0:
        return VERDICT_PARTIAL
    return VERDICT_MEASUREMENT


def assert_envelope_verdict_consistent(node, _path="ood"):
    """Refuse a node whose verdict string contradicts its own fractions.

    Two refusals, both aimed at the exact string that survived at the 30 k gate
    ("within the measured envelope on average", emitted while 54.63 % of steps
    were outside):

    1. A **MEASUREMENT** verdict with ANY step or window outside.
    2. Any verdict that is not EXTRAPOLATION while a **majority** of steps or
       windows are outside.

    Also refuses a node that reports a ratio without declaring its saturation —
    a lower bound presented as a measurement is the defect in a new costume.
    """
    if not isinstance(node, dict):
        return node
    v = node.get("EXTRAPOLATION_VERDICT")
    if v is None:
        return node
    cls = verdict_class(v)
    fw = float(node.get("EXTRAPOLATION_frac_windows_any_step_out_of_envelope") or 0.0)
    fs = max(float(node.get("EXTRAPOLATION_frac_steps_lat_over_3m") or 0.0),
             float(node.get("EXTRAPOLATION_frac_steps_yaw_over_12deg") or 0.0),
             float(node.get("EXTRAPOLATION_frac_steps_any") or 0.0))
    if cls == CLASS_MEASUREMENT and (fw > 0.0 or fs > 0.0):
        raise EnvelopeVerdictError(
            f"[{_path}] a MEASUREMENT verdict ({v!r}) was emitted while "
            f"{fs:.2%} of steps / {fw:.2%} of windows are OUTSIDE the measured "
            f"envelope. {SATURATION_NOTE}")
    if cls != CLASS_EXTRAPOLATION and (fw > MAJORITY_FRAC or fs > MAJORITY_FRAC):
        raise EnvelopeVerdictError(
            f"[{_path}] verdict {v!r} claims the envelope substantially held "
            f"while a MAJORITY is outside it ({fs:.2%} of steps / {fw:.2%} of "
            f"windows). {SATURATION_NOTE}")
    if node.get("ood_peak_ratio") is not None and (fw > 0.0 or fs > 0.0) \
            and not node.get("ratio_is_lower_bound"):
        raise EnvelopeVerdictError(
            f"[{_path}] a peak OOD ratio is reported with steps outside the "
            f"envelope but is not stamped `ratio_is_lower_bound`. A saturating "
            f"estimator must declare its own saturation. {SATURATION_NOTE}")
    return node


def verdict(lat_abs, yaw_abs_deg, eid, ood_map, K, *, n_boot=None, seed=0,
            stratum="overall") -> dict | None:
    """The OOD block for one stratum: **both clauses of E1a's rule**, reported
    separately, with the interval from the episode-cluster bootstrap.

    Returns ``None`` when the stratum is too small to bootstrap (< 2 windows or
    < 2 episodes) — a NOT-MEASURED, which is never a pass.
    """
    lat = np.atleast_2d(np.asarray(lat_abs, dtype=float))
    yaw = np.atleast_2d(np.asarray(yaw_abs_deg, dtype=float))
    eid = list(eid)
    if lat.shape[0] < 2 or len(set(eid)) < 2:
        return None
    n_boot = _ci.DEFAULT_N_BOOT if n_boot is None else int(n_boot)
    ratio = ood_map.ratio_arr(lat, yaw)
    frac = envelope_fractions(lat, yaw)

    def _bo(x):
        return _ci.episode_cluster_bootstrap(np.asarray(x, float), eid,
                                             n_boot=n_boot, seed=seed)

    peak = _bo(ratio.max(1))
    ratio_fires = bool(peak["mean"] > RATIO_EXTRAPOLATION_X)
    fw = frac["frac_windows_any_step_out_of_envelope"]
    fs = frac["frac_steps_any"]
    env_fires = bool(fw > 0.0)
    saturated = bool(fs > 0.0 or fw > 0.0)

    node = {
        "stratum": stratum,
        "horizon_K": int(K), "horizon_s": round(int(K) * 0.1, 2),
        "n_windows": int(lat.shape[0]), "n_episodes": int(len(set(eid))),
        "ood_peak_ratio": peak,
        "ood_mean_ratio": _bo(ratio.mean(1)),
        "frac_windows_ood_peak_under_1p16": round(
            float((ratio.max(1) <= 1.16).mean()), 4),
        "frac_windows_ood_peak_under_1p5": round(
            float((ratio.max(1) <= RATIO_EXTRAPOLATION_X).mean()), 4),
        # --- clause 2, FIRST CLASS: the fractions sit beside the ratio ------ #
        "EXTRAPOLATION_frac_steps_lat_over_3m": frac["frac_steps_lat_over_3m"],
        "EXTRAPOLATION_frac_steps_yaw_over_12deg": frac["frac_steps_yaw_over_12deg"],
        "EXTRAPOLATION_frac_steps_any": frac["frac_steps_any"],
        "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": fw,
        "_envelope": frac["envelope"],
        # --- the saturation, DECLARED --------------------------------------- #
        "ratio_is_lower_bound": saturated,
        "ratio_saturated_note": SATURATION_NOTE if saturated else None,
        "criterion_1_ratio_over_1p5": {
            "fires": ratio_fires,
            "peak_ratio_mean": round(float(peak["mean"]), 4),
            "threshold": RATIO_EXTRAPOLATION_X,
            "informative": not saturated,
            "_why_it_may_be_uninformative": SATURATION_NOTE},
        "criterion_2_steps_outside_measured_envelope": {
            "fires": env_fires,
            "frac_steps_lat_over_3m": frac["frac_steps_lat_over_3m"],
            "frac_steps_yaw_over_12deg": frac["frac_steps_yaw_over_12deg"],
            "frac_steps_any": fs,
            "frac_windows_any_step_outside": fw,
            "majority_frac_convention": MAJORITY_FRAC},
        "EXTRAPOLATION_VERDICT": _verdict_string(ratio_fires, fw, fs),
        "_rule": RULE,
        "_estimator": _ci.__name__ + ".episode_cluster_bootstrap",
    }
    return assert_envelope_verdict_consistent(node, _path=f"ood[{stratum}]")


def readjudicate(node, _path="ood") -> dict:
    """Re-adjudicate an EXISTING OOD node from the fractions it already carries.

    For committed artifacts whose per-window tensors are gone: E1a's emitters
    always wrote the ``EXTRAPOLATION_*`` fractions, they were simply never used
    to decide anything. This recomputes the disjunction from those fields alone
    — no tensors, no GPU, no re-run — and records what the node said before.

    Returns a NEW dict; the input is not mutated.
    """
    out = dict(node)
    fw = float(node.get("EXTRAPOLATION_frac_windows_any_step_out_of_envelope") or 0.0)
    fs = max(float(node.get("EXTRAPOLATION_frac_steps_lat_over_3m") or 0.0),
             float(node.get("EXTRAPOLATION_frac_steps_yaw_over_12deg") or 0.0),
             float(node.get("EXTRAPOLATION_frac_steps_any") or 0.0))
    peak = node.get("ood_peak_ratio")
    peak_mean = (float(peak["mean"]) if isinstance(peak, dict) and "mean" in peak
                 else float(peak) if isinstance(peak, (int, float)) else None)
    ratio_fires = bool(peak_mean is not None and peak_mean > RATIO_EXTRAPOLATION_X)
    saturated = bool(fs > 0.0 or fw > 0.0)
    out["_verdict_before_readjudication"] = node.get("EXTRAPOLATION_VERDICT")
    out["EXTRAPOLATION_VERDICT"] = _verdict_string(ratio_fires, fw, fs)
    out["_class_before"] = verdict_class(node.get("EXTRAPOLATION_VERDICT"))
    out["_class_after"] = verdict_class(out["EXTRAPOLATION_VERDICT"])
    out["_class_changed"] = out["_class_before"] not in (out["_class_after"],
                                                         CLASS_UNKNOWN)
    out["ratio_is_lower_bound"] = saturated
    out["ratio_saturated_note"] = SATURATION_NOTE if saturated else None
    out["criterion_1_ratio_over_1p5"] = {
        "fires": ratio_fires, "peak_ratio_mean": peak_mean,
        "threshold": RATIO_EXTRAPOLATION_X, "informative": not saturated,
        "_why_it_may_be_uninformative": SATURATION_NOTE}
    out["criterion_2_steps_outside_measured_envelope"] = {
        "fires": bool(fw > 0.0), "frac_steps_any": fs,
        "frac_windows_any_step_outside": fw,
        "majority_frac_convention": MAJORITY_FRAC}
    out["_rule"] = RULE
    return assert_envelope_verdict_consistent(out, _path=_path)
