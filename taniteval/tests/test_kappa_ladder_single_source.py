"""⛔ THE COHERENCE-VERDICT LADDER IS SINGLE-SOURCED — pin it, and pin that nobody
restated it.

THE DEFECT THIS EXISTS TO PREVENT (found 2026-08-15 in
`…/2026-08-15-dir-yaw-gate-reread/`, fixed 2026-08-16 in
`…/2026-08-16-verdict-stable-kappa/`).

`hierarchy._gate_sensitivity` computed `verdict_stable` against a bare **κ >= 0.2**.
The verdict word the programme actually PUBLISHES — `maneuver_consistency_verdict`,
and the adjective in `MODEL_REGISTRY.md:1111` ("κ 0.6033 (SUBSTANTIAL)") and
`Paper/TANITAD_PAPER.md` — comes from a DIFFERENT ladder: **< 0.1 DECORATIVE,
< 0.4 WEAK, >= 0.4 SUBSTANTIAL**. A field named for a verdict, testing a threshold
nobody publishes, is an instrument that CANNOT FAIL when the published claim is wrong.

⭐ It had already produced a live contradiction: `hier_v1-lf19.json` emits
κ **0.253** as *"SUPPORTED … cohere"*, while `V5_FLAGSHIP_DEEP_REVIEW.md:74`
publishes the SAME κ 0.253 as **WEAK**. One number, two words, both shipped.

⭐ And it hid a real flip. MEASURED on the programme's only gate-swept panel
(`hier_v1arch_gateswept.json.xz`, 880 windows / 40 OOD-val episodes): κ runs
0.5787 → 0.2038 across the sweep = SUBSTANTIAL at 0.15/0.10/0.06/0.04 but **WEAK**
at 0.02/0.01. The published verdict flips; the old predicate reported
`verdict_stable = true` because every κ cleared 0.2.

⚠️ These tests must never be relaxed to accommodate a threshold change. If the
programme decides to publish different bands, the change belongs in
`four_families.KAPPA_VERDICT_LADDER` **and** in the docs that quote it, and this
file's expected values are updated deliberately, in the same commit.
"""
import pytest

from taniteval.four_families import (KAPPA_VERDICT_LADDER, kappa_band,
                                     kappa_verdict)


# --------------------------------------------------------------------------- #
# 1. The ladder itself                                                         #
# --------------------------------------------------------------------------- #
def test_ladder_is_exactly_the_published_bands():
    """⛔ THE PIN. These three boundaries are quoted in MODEL_REGISTRY.md and the
    paper. Changing them silently re-labels every coherence claim in the programme."""
    assert KAPPA_VERDICT_LADDER == ((0.1, "DECORATIVE"), (0.4, "WEAK"),
                                    (float("inf"), "SUBSTANTIAL"))


@pytest.mark.parametrize("k,band", [
    (-1.0, "DECORATIVE"), (0.0, "DECORATIVE"), (0.0072, "DECORATIVE"),
    (0.0999, "DECORATIVE"),
    (0.1, "WEAK"),                 # boundary is INCLUSIVE-below: <0.1 decorative
    (0.2038, "WEAK"), (0.253, "WEAK"), (0.3999, "WEAK"),
    (0.4, "SUBSTANTIAL"),          # >=0.4 is the top band
    (0.5787, "SUBSTANTIAL"), (0.6033, "SUBSTANTIAL"), (1.0, "SUBSTANTIAL"),
])
def test_band_boundaries_are_exact(k, band):
    assert kappa_band(k) == band


def test_uncomputable_kappa_is_none_not_a_band():
    """⛔ κ is UNDEFINED when one side has a single class (`_kappa` returns None).
    Mapping that to DECORATIVE would read as 'measured, and unrelated' when the
    truth is 'not computable' — the same distinction `_gate_sensitivity` already
    protects for the raw number."""
    assert kappa_band(None) is None
    assert kappa_verdict(None) is None
    assert kappa_band(float("nan")) is None
    assert kappa_band("not-a-number") is None


def test_published_verdict_strings_are_byte_identical_to_banked_artifacts():
    """The gloss travels in banked JSON; changing it breaks comparability with
    every four-family panel already on disk."""
    assert kappa_verdict(0.05) == (
        "DECORATIVE — declared manoeuvre is ~unrelated to the driven path")
    assert kappa_verdict(0.253) == "WEAK"
    assert kappa_verdict(0.6033) == "SUBSTANTIAL"


# --------------------------------------------------------------------------- #
# 2. The consumers use the SOURCE, not a copy                                  #
# --------------------------------------------------------------------------- #
def test_hierarchy_imports_the_ladder_rather_than_restating_it():
    """⭐ THE LOAD-BEARING TEST. `hierarchy` must hold the SAME objects, not equal
    copies — an equal copy is exactly what drifts."""
    from taniteval import four_families, hierarchy
    assert hierarchy.KAPPA_VERDICT_LADDER is four_families.KAPPA_VERDICT_LADDER
    assert hierarchy.kappa_band is four_families.kappa_band


#: The retired private cut. Not a published band boundary — see the module docstring.
RETIRED_PRIVATE_CUT = 0.2

#: The two functions that decide a coherence verdict from a kappa.
VERDICT_DECIDING_FUNCTIONS = ("_gate_sensitivity", "_thesis")


def _function_defs(module, names):
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(module))
    found = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [n for n in names if n not in found]
    assert not missing, f"expected functions vanished from {module.__name__}: {missing}"
    return {n: found[n] for n in names}


def test_no_bare_kappa_threshold_survives_in_hierarchy():
    """⛔ THE REGRESSION GUARD, and it is deliberately STRUCTURAL (AST), not textual.

    Two reasons a regex was wrong here, both learned the hard way:

    1. ⛔ **A text scan matches the PROSE THAT DOCUMENTS THE DEFECT.** The first
       version of this test failed on the very comments explaining that
       ``kappa >= 0.2`` was retired — the searched token appeared in the
       explanation. Same family as the polling-monitor trap in `CLAUDE.md`
       (*a filter containing the pattern it searches for matches its own echo*):
       make the searched token disjoint from the emitted one, or search a
       representation that cannot contain prose at all. An AST has no comments
       and no docstring bodies to trip over.
    2. ⛔ **The original defect had no "kappa" in it.** `_gate_sensitivity`'s line
       was ``{bool(k >= 0.2) for k in ks}`` — the identifier is ``k``. An
       identifier-matching regex would have MISSED the very bug this file exists
       to pin, and passed while the instrument was broken.

    So: walk the two verdict-deciding functions and assert (a) neither compares
    anything to the retired private cut, and (b) each actually calls
    ``kappa_band``. Unrelated float comparisons in those functions (``1e-9``
    guards, the swept gate ``g``) are untouched."""
    import ast

    from taniteval import hierarchy
    defs = _function_defs(hierarchy, VERDICT_DECIDING_FUNCTIONS)

    offenders = []
    for name, fn in defs.items():
        for node in ast.walk(fn):
            if not isinstance(node, ast.Compare):
                continue
            for operand in [node.left, *node.comparators]:
                if (isinstance(operand, ast.Constant)
                        and isinstance(operand.value, (int, float))
                        and not isinstance(operand.value, bool)
                        and float(operand.value) == RETIRED_PRIVATE_CUT):
                    offenders.append(f"{name}: {ast.unparse(node)}")
    assert offenders == [], (
        f"a verdict is being decided by the retired private cut "
        f"{RETIRED_PRIVATE_CUT} again — use four_families.kappa_band: {offenders}")

    for name, fn in defs.items():
        calls = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
        assert "kappa_band" in calls, (
            f"{name} no longer routes its verdict through kappa_band — the ladder "
            f"has been restated somewhere. calls seen: {sorted(calls)}")


def test_gate_sensitivity_stability_is_evaluated_on_the_ladder():
    """A sweep whose κ crosses 0.4 but never 0.2 is the case that separates the
    two rules. It is the SHAPE of the real 880-window panel (SUBSTANTIAL at the
    coarse gates, WEAK at the fine ones), reduced to synthetic input.

    Under the published ladder this MUST report unstable; under the retired
    `κ >= 0.2` rule it reported stable."""
    ks = [0.5787, 0.5715, 0.4796, 0.4075, 0.3065, 0.2038]   # the banked sweep
    bands = {kappa_band(k) for k in ks}
    assert bands == {"SUBSTANTIAL", "WEAK"}, bands
    assert len(bands) != 1, "the published verdict flips across this sweep"
    assert len({k >= 0.2 for k in ks}) == 1, (
        "…while the retired 0.2 rule saw no flip at all — which is the defect")


# --------------------------------------------------------------------------- #
# 3. The out-of-package duplicate must not drift                               #
# --------------------------------------------------------------------------- #
def test_v5_guard_in_stack_still_agrees_with_the_canonical_ladder():
    """`stack/scripts/v5_guard.py:216-217` re-implements the ladder because it runs
    on a pod where `taniteval` is not importable (PYTHONPATH is the stack only).
    It cannot import the constant, so it is pinned by SOURCE instead — a real
    drift detector rather than an import.

    ⚠️ If this fails, the v5 GPU-spend guard and the published panel disagree about
    what SUBSTANTIAL means. Fix the guard; do not relax this test."""
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[2]
    guard = repo / "stack" / "scripts" / "v5_guard.py"
    if not guard.is_file():                      # pragma: no cover
        pytest.skip(f"v5_guard.py not present at {guard}")
    src = guard.read_text(encoding="utf-8")
    lo, hi = KAPPA_VERDICT_LADDER[0][0], KAPPA_VERDICT_LADDER[1][0]
    assert f"kappa < {lo}" in src, (
        f"v5_guard no longer cuts DECORATIVE at {lo} — it has drifted from "
        "four_families.KAPPA_VERDICT_LADDER")
    assert f"kappa < {hi}" in src, (
        f"v5_guard no longer cuts WEAK at {hi} — it has drifted from "
        "four_families.KAPPA_VERDICT_LADDER")
    for band in ("DECORATIVE", "WEAK", "SUBSTANTIAL"):
        assert band in src, f"v5_guard lost the {band} band"
