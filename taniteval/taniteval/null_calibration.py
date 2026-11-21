"""THE MEASURED NULL for the clip-level probe estimator — so no future read can
quote t ≈ 2 as significant.

⛔⛔ WHY THIS MODULE EXISTS. On 2026-08-25/26 this programme published, and then
retracted, three separate claims that rested on t-statistics between 2.0 and 3.1
from the probe panels (`rangeprobe_rff`-style: clip-disjoint ridge, within-clip
Pearson r, minus a time-shuffled control). Every panel already carried TWO
controls — a CONSTANT control that reads exactly 0, and a TIME-SHUFFLED control
that fixes the pairing — and **neither of them bounds the statistic's TAIL.**

⭐ THE MEASUREMENT. Rerun the identical panel with the INPUT replaced by Gaussian
noise of the same shape. A random input carries nothing, so every t it produces is
a draw from the null. Over **104 independent draws** at two column widths:

        |t| median 0.61 · p90 1.98 · p95 2.57 · p99 2.93 · MAX 3.49

⇒ **THE EFFECTIVE BAR FOR THIS PANEL FAMILY IS |t| ≈ 2.9, NOT 2.0.**

⚠️⚠️ AND DIMENSIONALITY IS NOT WHAT SETS THE TAIL — the intuition that "3 scalars
must have a tighter null than 2048 dimensions" is FALSE and was the reason given
(in E-DEC-54) for believing the action-column results were safe:

        d = 2048, 56 draws  ->  p95 2.56, max 2.93
        d = 3,    48 draws  ->  p95 2.59, max **3.49**

The heaviness comes from the ESTIMATOR — ~20 clip-level scores, a ridge refit per
fold, and a difference of two correlated means — not from the width of the input.
**It is not a t-distribution and must never be read as one.**

⚠️ A COUNTING ERROR IN THE FIRST VERSION, CORRECTED HERE. The pooled set was first
reported as **128** draws; two of the three runs shared `seed = 1000 + s` for their
first six seeds, so 24 draws were duplicated. The true independent n is **104**.
Percentiles moved in the third decimal and **no verdict changed** (action→Δspeed
0.070 → 0.067), but the n was wrong wherever it was quoted. ⇒ **A pooled statistic
must be deduplicated by SEED, not assumed independent because the runs were
separate invocations.**

PROVENANCE: `TanitAD Research Lab/Architecture & Inference/Research/
2026-08-24-action-conditioning-and-heldout/raw/egonull_measured.json` and
`…/egonull_d3.json`; generator `…/probes/egonull.py`. Registered as E-DEC-54 and
E-DEC-56 in `Project Steering/GOALS_AND_CLAIMS.md`; the retraction it forced is
C162 in `RETRACTION_LOG.md`.

USAGE — report a p-value, not a bare t:

    from taniteval.null_calibration import p_value, verdict
    p_value(4.57)          # -> 0.0    (action -> delta-yaw: survives)
    p_value(2.56)          # -> 0.067  (action -> delta-speed: inside the null)
    verdict(2.56)          # -> ('INSIDE_NULL', 0.067)

⚠️ SCOPE. This null was measured for panels with ~20 clips and the estimator named
above. A panel with a different clip count, a different fold scheme, or a different
statistic needs **its own** null — `egonull.py` measures one in minutes on CPU.
Do not import this constant into a panel it was not measured for; that is the
`df`/`free`/`step_s` scope error in a statistics costume.
"""
from __future__ import annotations

# --- d = 2048 (56 draws; supersedes an earlier 24-draw run that is its prefix) --
_D2048 = [
    -0.28, -0.16, -0.42, -0.31, 2.21, 0.28, 1.63, 0.4, 0.47, -0.41, 0.74,
    -0.29, 2.8, -0.35, -0.86, -0.78, -1.07, 1.69, 0.29, 2.93, 0, 0.22, -0.7,
    1.12, 2.56, 0.48, -0.84, 0.54, -0.75, -0.37, 0.58, -1.09, -0.33, -1.89,
    1.98, -2.57, 0.9, 0.52, -0.65, -0.11, 0.64, 0.17, 1.14, -0.1, 1.54,
    1.49, -0.12, 1.58, -0.01, 0.71, 0.4, 1.13, -0.21, -0.11, 1.18, -0.18,
]

# --- d = 3 (48 draws) — note this is the run that produced the MAX ------------
_D3 = [
    0.15, -1.6, -1.24, 0.02, -1.44, -1.04, 0.17, -0.34, -0.87, -0.07, 1.8,
    -2.54, -0.21, 0.15, -1.28, 0.32, -0.49, 2.13, 0.82, 2.61, -0.34, -1.77,
    2.85, -0.93, 0.97, 0.61, -0.78, 1.32, -1.45, 0.17, 0.08, 0.45, -0.04,
    -0.82, -1.91, 3.49, 0.61, -0.03, 0.32, -0.6, -1, 0.37, 0.22, 1.29,
    -0.46, 0.07, -0.48, 0.33,
]

NULL_DRAWS: tuple[float, ...] = tuple(_D2048 + _D3)
N_DRAWS = len(NULL_DRAWS)                                   # 104
_ABS = sorted(abs(x) for x in NULL_DRAWS)

#: the honest headline: |t| this estimator reaches from a provably-null input
NULL_MAX = _ABS[-1]                                         # 3.49


def _q(p: float) -> float:
    """Empirical quantile of |t| under the null (nearest-rank)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"quantile out of range: {p}")
    k = max(1, min(N_DRAWS, int(round(p * N_DRAWS))))
    return _ABS[k - 1]


NULL_P90 = _q(0.90)
NULL_P95 = _q(0.95)
NULL_P99 = _q(0.99)


def p_value(t: float) -> float:
    """Empirical two-sided P(|null| >= |t|).

    ⚠️ FLOORED AT 1/N, NOT ZERO. With 104 draws the smallest resolvable p-value is
    ~0.0096; reporting 0.000 would claim a precision the sample does not have.
    A caller wanting a smaller p needs more draws, not a smaller number.
    """
    k = sum(1 for x in _ABS if x >= abs(float(t)))
    return max(k / N_DRAWS, 1.0 / N_DRAWS) if k == 0 else k / N_DRAWS


def verdict(t: float, alpha: float = 0.05) -> tuple[str, float]:
    """Classify a t against the MEASURED null. Every cell is enumerated and the
    default is UNCLASSIFIED.

    ⛔ THE DEFAULT IS NOT A CLAIM. C160: an ``else`` branch that says something
    about the world is an assertion about every case the author failed to think
    of — and in the incident that earned that rule, the default branch printed the
    OPPOSITE of the table two lines above it. Here the only non-committal band is
    named as such.

    Returns ``(label, p)`` where label is one of:
      ``SURVIVES``      p < alpha and |t| > the null max — safe to quote
      ``MARGINAL``      p < alpha but |t| <= the null max — a draw this size WAS
                        observed from noise; quote the p, never the bare t
      ``INSIDE_NULL``   p >= alpha — not separable from noise
    """
    p = p_value(t)
    if p >= alpha:
        return "INSIDE_NULL", p
    if abs(float(t)) > NULL_MAX:
        return "SURVIVES", p
    return "MARGINAL", p


def describe() -> str:
    """One line a report can quote verbatim, so the constant travels with it."""
    return (f"measured null, n={N_DRAWS} draws: |t| p90 {NULL_P90:.2f} "
            f"p95 {NULL_P95:.2f} p99 {NULL_P99:.2f} max {NULL_MAX:.2f} "
            f"(E-DEC-54/56)")
