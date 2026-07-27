"""⛔ THE PRINTED INTERVAL MUST NOT CONTRADICT THE PRINTED VERDICT.

THE DEFECT (MEASURED 2026-07-27)
--------------------------------
``paired_episode_cluster_bootstrap`` decided ``separated`` on the **unrounded**
percentile bounds (``ci.py`` — ``bool(lo > 0 or hi < 0)``) while it rounded
``delta``/``lo``/``hi`` to 4 dp for publication. When two arms are bit-identical
or near-identical the true bounds are ~1e-9 of one sign, and the emitted record
read::

    {"delta": 0.0, "lo": 0.0, "hi": 0.0, "separated": true}

Found in a committed artifact:
``TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/
2026-07-27-vtband-decision/raw/legA_v5config_structural.json`` — the Leg A
structural probe, where a from-scratch model's ``sel_gate`` is **exactly 0.0** so
two goal options produce bit-identical plans.

⚠️ **Every stream in this program reads this estimator's output**, and
``heldout_gate.HeldoutGate.observe`` keys its **early-stop** on ``separated``. A
verdict printed beside a null effect is the kind of thing a reader resolves in
whichever direction they already believed.

WHAT IS AND IS NOT FIXED
-------------------------
⛔ **The statistics are unchanged.** Testing the unrounded bounds is correct;
testing rounded ones would silently redefine ``separated`` as "separated by at
least 5e-5" — an unregistered threshold that would flip real verdicts. **The
display is what lies, so only the display is fixed** (``ci._render_bounds``:
adaptive precision, marker as the fallback).

⚠️ **These tests FAIL against the pre-fix code** — that is the point. The first
one reproduces the exact contradictory rendering.
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[1], _HERE.parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from taniteval import ci as C                                      # noqa: E402


def _rendered_says_separated(r) -> bool:
    """What a READER (or a gate keying on the JSON) concludes from the numbers."""
    return bool(r["lo"] > 0 or r["hi"] < 0)


# --------------------------------------------------------------------------- #
# 1. the exact defect                                                          #
# --------------------------------------------------------------------------- #
def test_bit_identical_arms_do_not_print_separated_beside_a_zero_interval():
    """⛔ THE REGRESSION TEST. Two arms differing by ~1e-9 — Leg A's shape.

    Pre-fix this returned ``delta 0.0, lo 0.0, hi 0.0, separated True``. The
    assertion below is deliberately about **agreement**, not about which way it
    goes: whether the estimator calls this separated is a statistics question and
    is not this test's business. What the test forbids is the record *saying two
    different things at once*."""
    rng = np.random.default_rng(0)
    eid = np.repeat(np.arange(8), 20)
    base = rng.normal(5.0, 1.0, size=eid.size)
    a = base + 1e-9                       # ...differing by float-noise only
    b = base
    r = C.paired_episode_cluster_bootstrap(a, b, eid, n_boot=400, seed=0)
    assert _rendered_says_separated(r) == r["separated"], (
        f"the printed interval contradicts the printed verdict: "
        f"delta={r['delta']} [{r['lo']}, {r['hi']}] separated={r['separated']}")


def test_exactly_identical_arms_are_marked_DEGENERATE_not_just_rendered():
    """When the bounds are 0.0 to the last bit, no number can carry the verdict —
    so the record must say so in words.

    ⚠️ A marker alone would NOT have been an acceptable fix (a reader who skips
    it still sees ``0.0 [0.0, 0.0] separated``); it is the fallback for the case
    where digits genuinely cannot help."""
    eid = np.repeat(np.arange(6), 10)
    a = np.linspace(1.0, 2.0, eid.size)
    r = C.paired_episode_cluster_bootstrap(a, a.copy(), eid, n_boot=200, seed=0)
    assert r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0
    assert r["separated"] is False, r
    assert _rendered_says_separated(r) is False
    if r.get("degenerate"):
        assert "IDENTICAL" in r["degenerate_note"]


def test_a_tiny_but_REAL_separation_prints_enough_digits_to_show_it():
    """The positive case: a genuine 1e-6 effect must not render as ``0.0``.

    This is the half that "just add a marker" would have left broken."""
    rng = np.random.default_rng(7)
    eid = np.repeat(np.arange(10), 30)
    base = rng.normal(2.0, 0.5, size=eid.size)
    a, b = base + 1e-6, base
    r = C.paired_episode_cluster_bootstrap(a, b, eid, n_boot=400, seed=0)
    assert r["separated"] is True, r
    assert _rendered_says_separated(r) is True, r
    assert r["lo"] != 0.0 and r["delta"] != 0.0, r
    assert r["display_dp"] > C.DISPLAY_DP, r
    assert "RENDERING fix, not extra precision" in r["display_note"]


# --------------------------------------------------------------------------- #
# 2. the fix must be inert on every ordinary number                            #
# --------------------------------------------------------------------------- #
def test_ordinary_effect_sizes_are_still_rendered_at_exactly_4dp():
    """⚠️ The compatibility guarantee. Every published interval in the program is
    4 dp; a fix that re-rendered them all would silently invalidate cross-doc
    comparisons — and would be a far bigger change than the bug."""
    rng = np.random.default_rng(11)
    eid = np.repeat(np.arange(12), 25)
    base = rng.normal(3.0, 0.8, size=eid.size)
    for effect in (0.0, 0.01, 0.135, -0.42, 1.7):
        a = base + effect + rng.normal(0, 0.05, size=base.shape)
        b = base + rng.normal(0, 0.05, size=base.shape)
        r = C.paired_episode_cluster_bootstrap(a, b, eid, n_boot=300, seed=2)
        assert "display_dp" not in r, (effect, r)
        assert "degenerate" not in r, (effect, r)
        for k in ("delta", "lo", "hi", "ci95"):
            assert round(r[k], C.DISPLAY_DP) == r[k], (effect, k, r[k])
        assert _rendered_says_separated(r) == r["separated"], (effect, r)


def test_the_SEPARATION_TEST_ITSELF_is_untouched_by_the_rendering_fix():
    """⛔ The guard on the guard. ``separated`` must still be the UNROUNDED
    predicate — if the fix had been "round first, then test", intervals whose
    true bound is 3e-5 would flip from separated to not, which is a statistics
    change wearing a display change's clothes."""
    rng = np.random.default_rng(13)
    eid = np.repeat(np.arange(9), 22)
    base = rng.normal(1.0, 0.3, size=eid.size)
    a, b = base + 3e-5, base
    r = C.paired_episode_cluster_bootstrap(a, b, eid, n_boot=400, seed=0)
    d = np.array([float(a[s].mean() - b[s].mean())
                  for s in C._draws(*C.episode_index(eid), 400, 0)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    assert r["separated"] == bool(lo > 0 or hi < 0), (r, lo, hi)


def test_render_bounds_never_needs_more_than_MAX_DISPLAY_DP():
    """The escalation terminates, and terminating is what raises ``degenerate``."""
    assert C._render_bounds(0.05, 0.20, True) == (C.DISPLAY_DP, False)
    assert C._render_bounds(-0.30, 0.30, False) == (C.DISPLAY_DP, False)
    dp, degen = C._render_bounds(1e-9, 2e-9, True)
    assert dp > C.DISPLAY_DP and not degen
    dp, degen = C._render_bounds(0.0, 0.0, True)          # impossible to render
    assert dp == C.MAX_DISPLAY_DP and degen is True


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:                                    # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
