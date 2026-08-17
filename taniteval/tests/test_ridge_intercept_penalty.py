"""The ridge readout's floor must be the MEAN, not zero — pinned both ways.

WHY THIS EXISTS (2026-08-18, C92). `pc6_linear_readout.ridge_fit` builds
`X.T @ X + alpha * np.eye(d)` on a design matrix whose last column is the BIAS.
The intercept is therefore shrunk like any other coefficient, so as alpha grows
predictions collapse toward ZERO rather than toward the MEAN of y.

⇒ The readout cannot express the constant predictor it is scored against, so a
no-signal arm scores WORSE THAN A CONSTANT BY CONSTRUCTION. That biases the
FLOOR, which is why it taints "K1 FAIL" verdicts specifically: a FAIL against a
floor the model was structurally forbidden to reach is not a finding.

⚠️ TWO PROPERTIES ARE PINNED, AND THE SECOND MATTERS AS MUCH AS THE FIRST:
  1. the REPAIR works -- `intercept_col=-1` converges to the mean;
  2. the DEFAULT still reproduces the incumbent BIT-EXACTLY.
(2) exists because banked `pc6_ridge_*.json` results reproduce through the old
path and `ll1_ladder.py` asserts that reproduction to 1e-4. Silently "fixing" the
default would rewrite the meaning of every committed artifact while leaving the
filenames unchanged -- a worse failure than the bug. The banked numbers get
re-read, not mutated underneath.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_MOD = (Path(__file__).resolve().parents[2] / "TanitAD Research Hub"
        / "Architecture & Inference" / "Implementation" / "incoming"
        / "2026-08-17-probe-positive-control" / "code"
        / "pc6_linear_readout.py")


def _ridge_fit():
    """Load ONLY the function, without the module's heavy import tail."""
    if not _MOD.exists():                                   # pragma: no cover
        pytest.skip(f"module not present: {_MOD}")
    src = _MOD.read_text(encoding="utf-8")
    start = src.index("def ridge_fit")
    end = src.index("\ndef ", start + 1)
    ns = {"np": np}
    exec(compile(src[start:end], str(_MOD), "exec"), ns)     # noqa: S102
    return ns["ridge_fit"]


def _design(n=200, d=6, seed=0):
    """Pure-noise features + a ones column. y has a large, nonzero mean."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.normal(loc=50.0, scale=1.0, size=n)      # NO signal in X
    Z = np.concatenate([X, np.ones((n, 1))], axis=1)
    return Z, y


def test_the_repair_converges_to_the_MEAN_on_a_no_signal_input():
    """This is the property that was violated."""
    fit = _ridge_fit()
    Z, y = _design()
    w = fit(Z, y, alpha=1e6, intercept_col=-1)
    pred = Z @ w
    assert abs(pred.mean() - y.mean()) < 0.05, (
        f"repaired ridge must approach the mean {y.mean():.3f}, "
        f"got {pred.mean():.3f}"
    )


def test_the_incumbent_default_collapses_toward_ZERO_the_documented_defect():
    """Pin the DEFECT too -- so nobody 'discovers' it a third time."""
    fit = _ridge_fit()
    Z, y = _design()
    pred = Z @ fit(Z, y, alpha=1e6)
    assert abs(pred.mean()) < 1.0, (
        "the unrepaired path is documented to collapse toward zero; if this "
        "now approaches the mean, the DEFAULT changed and every banked "
        "pc6_ridge_*.json silently changed meaning with it"
    )
    assert abs(pred.mean() - y.mean()) > 40.0


def test_default_is_bit_exact_with_the_incumbent_formula():
    """Banked reproduction must not move."""
    fit = _ridge_fit()
    Z, y = _design(seed=3)
    for alpha in (1e-2, 1.0, 1e3):
        d = Z.shape[1]
        want = np.linalg.solve(Z.T @ Z + alpha * np.eye(d), Z.T @ y)
        assert np.allclose(fit(Z, y, alpha), want, rtol=0, atol=0), (
            f"default path diverged from the incumbent formula at alpha={alpha}"
        )


def test_repair_penalises_the_slope_but_not_the_bias():
    """The fix must be surgical: only the named column leaves the penalty."""
    fit = _ridge_fit()
    Z, y = _design(seed=7)
    w = fit(Z, y, alpha=1e6, intercept_col=-1)
    assert np.abs(w[:-1]).max() < 1e-3, "slopes must still be shrunk"
    assert abs(w[-1] - y.mean()) < 0.05, "bias must survive unshrunk"


def test_at_alpha_zero_both_paths_agree():
    """With no penalty there is nothing to exclude -- a sanity bound."""
    fit = _ridge_fit()
    Z, y = _design(seed=11)
    a = fit(Z, y, alpha=1e-9)
    b = fit(Z, y, alpha=1e-9, intercept_col=-1)
    assert np.allclose(a, b, atol=1e-5)


def test_the_defect_is_documented_where_the_next_caller_will_look():
    """A fix nobody can find gets rediscovered. Keep the warning at the code."""
    if not _MOD.exists():                                   # pragma: no cover
        pytest.skip("module not present")
    src = _MOD.read_text(encoding="utf-8")
    head = src[src.index("def ridge_fit"):][:2400]
    for token in ("intercept_col", "C92", "K1 FAIL"):
        assert token in head, f"ridge_fit docstring must name {token!r}"
