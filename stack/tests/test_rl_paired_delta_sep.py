"""`sep` means THE INTERVAL EXCLUDES ZERO -- pinned, including the asymmetric case.

⛔⛔ THE DEFECT THIS FILE PINS (MEASURED 2026-09-05,
`TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-rl-generator-collisions/RESULT.md`
section 6.9, register row `D-RL-CTRL0-SEPFLOOR-1`).

`rl_refcv3_min.paired_delta` returned::

    "sep": bool((lo > 0) == (hi > 0))

which asks *"do the two endpoints agree about being strictly positive"*. That is a different
question from *"does the interval exclude zero"*, and it is wrong on two cases -- ASYMMETRICALLY:

    lo    hi     buggy   correct
    0.0   0.0    True    False    an exactly-zero interval, read as separated
   -1.0   0.0    True    False    touches zero FROM BELOW, read as separated
    0.0   1.0    False   False    touches zero FROM ABOVE, correctly not separated

⚠️ The asymmetry is the serious half. An interval touching zero from below is called separated
while its mirror image is not, and on this rig's safety metrics NEGATIVE means BETTER -- so the
defect systematically favoured reporting IMPROVEMENTS.

MEASURED blast radius on the banked arms: **36 of `ctrl0`'s 40** `sep=True` rows -- on an arm
whose weights were BITWISE FROZEN (`weights_changed=False`) -- and **11 of `ctrl_null`'s 35**
were exact-zero rows this predicate mislabelled. The circulating figure *"a zero-information arm
separates 35 of 57 metrics"* should read **24 of 57**. The claim's substance is unaffected (those
24 are real, and `fan_peak_g_mean` drifts +0.134 on an arm carrying no reward) but the count is
not.

⭐ The separate, non-defect finding that came with it: even after this fix, `ctrl0` -- frozen
weights -- still separates **4** metrics at ~1e-8 from GPU float non-determinism alone. `sep`
carries no magnitude information, which is `H-ESTIM-SEED-1` in its purest form.

No GPU, no corpus: `paired_delta` is extracted from source and exercised on hand-built deltas, so
the intervals are exact rather than sampled.
"""

from __future__ import annotations

import importlib.util
import io
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(HERE, "scripts", "rl_refcv3_min.py")


def _paired_delta():
    """Extract `paired_delta` from the driver WITHOUT importing it.

    Importing `rl_refcv3_min` pulls in torch, taniteval and a GPU-shaped dependency chain; this
    test is about eleven characters of arithmetic and must not need any of it.
    """
    if not os.path.exists(DRIVER):
        pytest.skip("rl_refcv3_min.py not present in this tree")
    with io.open(DRIVER, encoding="utf-8") as fh:
        src = fh.read()
    start = src.index("def paired_delta(")
    end = src.index("\ndef ", start + 1)
    ns: dict = {"np": np}
    exec(compile(src[start:end], DRIVER, "exec"), ns)
    return ns["paired_delta"], src


def _call(pd, vals):
    """Per-episode deltas -> the before/after dicts `paired_delta` expects (one window each)."""
    before = {"per_window": [{"wi": i, "eid": i, "k": 0.0} for i in range(len(vals))]}
    after = {"per_window": [{"wi": i, "eid": i, "k": float(v)} for i, v in enumerate(vals)]}
    return pd(before, after, "k", reps=400, seed=0)


def test_exactly_zero_interval_is_not_separated():
    """[0, 0] does not exclude zero. This is the case that inflated every count on the rig."""
    pd, _ = _paired_delta()
    r = _call(pd, [0.0] * 16)
    assert r["delta"] == 0.0
    assert r["lo"] == 0.0 and r["hi"] == 0.0
    assert r["sep"] is False, (
        "an exactly-zero interval was reported SEPARATED -- this is the defect that made a "
        "FROZEN-WEIGHT arm read 40/57 separated"
    )


def test_clearly_nonzero_intervals_are_separated_both_signs():
    pd, _ = _paired_delta()
    assert _call(pd, [1.0] * 16)["sep"] is True
    assert _call(pd, [-1.0] * 16)["sep"] is True


def test_straddling_zero_is_not_separated():
    pd, _ = _paired_delta()
    assert _call(pd, [-1.0, 1.0] * 8)["sep"] is False


def test_the_predicate_truth_table_including_the_asymmetric_case():
    """⭐ THE ASYMMETRY IS THE SERIOUS HALF, and it needs a DIRECT test.

    ⚠ An earlier version of this test drove the asymmetry through `paired_delta` and
    PASSED ON THE BUGGY TREE -- bootstrapped percentiles essentially never land exactly on
    zero except in the all-zero case, so the `[-1, 0]` case was never constructed and the
    test was not pinning what its name claimed. Recorded rather than quietly rewritten: a
    test that cannot fail on the defect it names is decoration.

    So evaluate the PREDICATE ITSELF on synthetic endpoints, lifted from source.
    """
    _, src = _paired_delta()
    line = next(ln for ln in src.splitlines() if ln.strip().startswith('"sep": bool('))
    expr = line.strip()[len('"sep": bool('):]
    expr = expr[:expr.rindex(")")]

    table = [
        ((-2.0, -1.0), True,  "both negative"),
        ((1.0, 2.0),   True,  "both positive"),
        ((-1.0, 1.0),  False, "straddles zero"),
        ((0.0, 0.0),   False, "EXACTLY zero"),
        ((-1.0, 0.0),  False, "touches zero FROM BELOW"),
        ((0.0, 1.0),   False, "touches zero FROM ABOVE"),
    ]
    for (lo, hi), want, name in table:
        got = bool(eval(expr, {"lo": lo, "hi": hi}))
        assert got is want, (
            "sep(%g, %g) [%s] = %s, expected %s" % (lo, hi, name, got, want))

    # and the property that the two wrong rows violated: separation is invariant under negation
    for lo, hi in [(-2.0, -1.0), (1.0, 2.0), (-1.0, 1.0), (0.0, 0.0), (-1.0, 0.0), (0.0, 1.0)]:
        a = bool(eval(expr, {"lo": lo, "hi": hi}))
        b = bool(eval(expr, {"lo": -hi, "hi": -lo}))
        assert a == b, "sep is not invariant under negation at (%g, %g)" % (lo, hi)


def test_the_buggy_predicate_is_not_live_in_source():
    """A positive assertion on the CODE, not on the file.

    ⚠️ The fix's own comment deliberately QUOTES the buggy predicate so the mechanism is recorded
    beside the repair -- so a whole-file search for `(lo > 0) == (hi > 0)` matches that comment
    and fails. That is the self-match trap the monitor rule warns about, wearing a verification
    costume; it fired for real while this fix was being written. Match the full assignment.
    """
    _, src = _paired_delta()
    assert '"sep": bool((lo > 0) == (hi > 0))' not in src
    assert '"sep": bool(lo > 0 or hi < 0)' in src
