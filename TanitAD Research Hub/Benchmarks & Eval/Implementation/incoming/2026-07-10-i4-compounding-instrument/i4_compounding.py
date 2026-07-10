"""Hardened I4 / D3 compounding instrument (Benchmarks & Eval, 2026-07-10).

WHY THIS EXISTS
---------------
The D3 decomposition (`scripts/d3_decompose.py`, commits 9bbf4ca/c0b22b7) reports a single
per-horizon relative error

    rel_k = median( ||pred_k - true_k|| / ||true_k - z_t|| )                 (d3_decompose.py:81)

whose DENOMINATOR ||true_k - z_t|| is the persistence drift and GROWS with the horizon k.
An independent audit (`../../i4_horizon_normalization_audit/`, seed 20260710) showed with
synthetic ground truth that a model whose ABSOLUTE error compounds *superlinearly* still
produces a rel_k that FALLS with k, purely because the drift denominator grows faster. So
the rel_k-vs-k slope is NOT a valid readout of "does the model compound error", and cross-MODEL
rel_k comparisons are inflated up to ~2x by encoder drift-scale differences alone.

This module ships the artifact-immune companions, so no gate/D3 decision rests on the ratio slope:

  1. rel_triplet()  — the SAME rel_k, but ALWAYS returned together with the raw absolute error
     and the drift, so numerator and denominator are never conflated.
  2. compounding_ratio() — CR_k = err_rollout / err_direct at a shared horizon/target. Because
     rollout and direct predict the SAME target relative to the SAME z_t, the drift denominator
     CANCELS, so CR is denominator-free and artifact-immune. CR is the accepted world-model
     metrology for compounding (SkyJEPA 2606.23444; Robotic World Model 2501.10100): CR~=1 means
     the recursive rollout tracks the direct/teacher-forced path (no compounding); CR>1 means
     compounding; CR<1 means the rolled path is locally easier (e.g. a K-step-trained model).
  3. compounds() — a compounding VERDICT taken from the ABSOLUTE-error curve (denominator-free),
     never from the rel_k slope.

Pure numpy (accepts numpy arrays or detached torch tensors via np.asarray); no torch dependency,
so it is standalone-testable. Proposed target: fold into `stack/tanitad/eval/` and have
`d3_decompose.analyze()` emit {rel_k, abs_err_k, drift_k, CR} instead of rel_k alone.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


def _np(x):
    return np.asarray(x, dtype=np.float64)


def rel_triplet(pred, true, z_t) -> dict:
    """The I4 relative error WITH its parts exposed.

    pred, true, z_t : (N, d). Returns median rel_k, median absolute error (numerator),
    and median drift (denominator). Reporting all three makes the normalization explicit.
    """
    pred, true, z_t = _np(pred), _np(true), _np(z_t)
    num = np.linalg.norm(pred - true, axis=-1)
    den = np.maximum(np.linalg.norm(true - z_t, axis=-1), 1e-8)
    return {
        "rel_k": float(np.median(num / den)),
        "abs_err_median": float(np.median(num)),
        "drift_median": float(np.median(den)),
    }


def compounding_ratio(err_rollout, err_direct) -> float:
    """CR = median(err_rollout) / median(err_direct) at a shared horizon.

    err_rollout, err_direct : (N,) absolute errors of the recursive rollout and the direct
    (single-shot / teacher-forced) prediction of the SAME target. Denominator-free by
    construction. CR~=1 no compounding; CR>1 compounding; CR<1 rolled path locally easier.
    """
    er, ed = _np(err_rollout), _np(err_direct)
    return float(np.median(er) / max(float(np.median(ed)), 1e-12))


def cr_from_predictions(pred_rollout, pred_direct, true_k) -> float:
    """Convenience: CR from the two horizon-k latent predictions and their shared target."""
    pr, pd, tk = _np(pred_rollout), _np(pred_direct), _np(true_k)
    er = np.linalg.norm(pr - tk, axis=-1)
    ed = np.linalg.norm(pd - tk, axis=-1)
    return compounding_ratio(er, ed)


def cr_from_rel(rel_rollout: float, rel_direct: float) -> float:
    """CR reconstructed from two rel_k values that SHARE the z_t/target denominator.

    Valid ONLY when both rel values use the identical ||true_k - z_t|| (same model, same
    windows, same k) — e.g. d3_decompose's recursive_1step_x4 vs direct_k4. Then the drift
    cancels and CR = rel_rollout / rel_direct exactly.
    """
    return float(rel_direct and rel_rollout / rel_direct)


def compounds(abs_err_by_k: Mapping[int, float], tol: float = 1.0) -> dict:
    """Compounding verdict from the ABSOLUTE-error curve (denominator-free).

    abs_err_by_k : {k: median absolute error at horizon k}. Reports whether absolute error
    is monotonically non-decreasing in k (a necessary signature of compounding) and the
    growth exponent from a log-log slope. NEVER inspects rel_k. `tol` is the slope above
    which growth is called super-linear.
    """
    ks = sorted(abs_err_by_k)
    errs = [float(abs_err_by_k[k]) for k in ks]
    monotone = all(b >= a - 1e-9 for a, b in zip(errs, errs[1:]))
    slope = None
    if len(ks) >= 2 and all(e > 0 for e in errs) and ks[0] > 0:
        lk = np.log(np.asarray(ks, dtype=np.float64))
        le = np.log(np.asarray(errs, dtype=np.float64))
        slope = float(np.polyfit(lk, le, 1)[0])
    return {
        "abs_err_monotone_in_k": bool(monotone),
        "loglog_growth_exponent": slope,
        "verdict": ("compounds" if monotone and (slope is None or slope > 0) else "no-compound"),
        "superlinear": bool(slope is not None and slope > tol),
    }


__all__ = [
    "rel_triplet", "compounding_ratio", "cr_from_predictions",
    "cr_from_rel", "compounds",
]
