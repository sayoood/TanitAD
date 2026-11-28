"""⛔ A GATE MAY NEVER BE DECIDED BY THE BANNED ESTIMATOR FAMILY.

The failure this pins (MEASURED 2026-08-16, JACK_IN_GATES.md): `planner_p2.py`
adjudicated ``G1_pass`` on ``_jack_paired`` — a PAIRED DELTA under
``overlapping_holdout_se``, the exact statistic the 2026-07-25 blast radius
measured at up to **x-4.15 including a SIGN FLIP** — and ``G4_pass`` on a
``_jack_scalar`` mean-of-split-means compared against a threshold that was
itself a mean-of-split-means. It sat that way for 21 days with the correct
instruction written into the file's own docstring.

⚠️ **This is an AST walk, not a regex, and that is load-bearing.**

* A regex guard matches ITS OWN COMMENTS documenting the retired rule — the
  ``pgrep -f`` / log-monitor self-match trap in a third costume. An AST walk
  cannot see a comment or a docstring, so :func:`test_docstring_mention_is_not_a_violation`
  pins that directly.
* A regex keyed on the NAME misses ``bool(_jack_paired(...)["mean"] >= 0.2)``
  and misses a laundering chain through three intermediate variables. Taint
  propagation follows the DATA, so :func:`test_inlined_call_is_caught` and
  :func:`test_laundering_chain_is_caught` are the negative controls that keep
  this test from being vacuous.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from taniteval import gate_guard as gg

REPO = Path(__file__).resolve().parents[2]

#: The enforced scope: product code whose output decides something.
#: ⭐ 2026-08-28 — ``(REPO / "taniteval", "*.py")`` was ADDED. The scope covered
#: ``taniteval/taniteval`` and ``taniteval/tools`` but not the ``taniteval/``
#: top level, where ~35 standalone drivers live — including
#: ``recompute_ci.py``, an unguarded caller of ``ci.overlapping_holdout_se``.
#: It was outside the guard by accident, not by decision. MEASURED before
#: adding: **zero** new violations, name or shape, so the extension costs
#: nothing and closes the gap. Non-recursive on purpose: ``taniteval/tests``
#: contains deliberate reproductions of the arithmetic (they are fixtures that
#: PROVE the guard fires) and policing the guard's own negative controls would
#: be the self-match trap again.
ENFORCED_ROOTS = [(REPO / "taniteval" / "taniteval", "**/*.py"),
                  (REPO / "taniteval" / "tools", "**/*.py"),
                  (REPO / "taniteval", "*.py"),
                  (REPO / "stack" / "tanitad", "**/*.py"),
                  (REPO / "stack" / "scripts", "**/*.py")]
SKIP = ("__pycache__", "/.claude/", "/experiments/")


def _scan_all(fn):
    """Run a ``gate_guard`` scanner over every ``(root, pattern)`` in scope."""
    out = []
    for root, pattern in ENFORCED_ROOTS:
        out += fn([root], pattern=pattern, skip=SKIP)
    return out


# --------------------------------------------------------------------------- #
# THE GUARD                                                                    #
# --------------------------------------------------------------------------- #
def test_no_gate_is_decided_by_the_banned_estimator():
    v = _scan_all(gg.scan_paths)
    assert not v, (
        "a GATE verdict is computed from the banned `overlapping_holdout_se` "
        "family. That estimator biases the POINT ESTIMATE (mean-of-split-means, "
        "not full_set) and on paired deltas has been measured at x-4.15 "
        "INCLUDING A SIGN FLIP — a gate decided on it can be wrong in sign.\n"
        "Replace it with ci.episode_cluster_bootstrap / "
        "ci.paired_episode_cluster_bootstrap and keep the old value beside it "
        "under a key ending `_LEGACY`.\n  " + "\n  ".join(str(x) for x in v))


# --------------------------------------------------------------------------- #
# ⭐ THE ARITHMETIC-SHAPE CENSUS — the guard that had no caller                  #
# --------------------------------------------------------------------------- #
"""``gate_guard.scan_source_shapes`` finds ``z · dispersion / sqrt(n)`` written
under ANY name — the failure mode a name-keyed rule is always one rename behind.
Its own docstring says admissibility is settled by
``tests/test_no_jack_in_gates.py::SHAPE_ALLOWLIST``.

⛔ **That allowlist did not exist, and neither did any caller.** MEASURED
2026-08-28: ``grep -rn 'scan_paths_shapes\\|scan_file_shapes'`` over the repo
returned only the definitions in ``gate_guard.py`` itself. The detector was
120 lines of correct, tested-by-nobody code — an instrument structurally unable
to report the answer it is cited for, which is the same class as ``df`` hiding
the per-pod quota and as ``_files_under`` skipping 373 of 373 files.

The census below is EXHAUSTIVE and EXACT: every shape in scope is either
allowlisted with a reason or listed as a known-open defect, and the test fails
if the detected set differs from their union in EITHER direction. So:

* a NEW site fails on the day it is written, whatever it is called;
* a known-open site that is "fixed" by renaming it into
  ``is_declared_estimator_name`` self-exemption ALSO fails, because it
  disappears from a census that requires it to be there.
"""

#: ``(relative path, function)`` -> why this ``z·sd/sqrt(n)`` is a VALID SE.
#: The inputs must be genuinely independent; that is a semantic claim no syntax
#: tree can check, which is exactly why it is written down by a human here.
SHAPE_ALLOWLIST = {
    ("stack/tanitad/eval/bakeoff.py", "mean_ci95"):
        "Aggregates over independent TRAINING SEEDS, not over holdouts of one "
        "pool. Different seeds are independent draws, so the normal-"
        "approximation SE is the right estimator. Blessed by name in "
        "gate_guard's own module note.",
}

#: ``(relative path, function)`` -> a MEASURED instance of the deprecated
#: estimator that is still live, with its escalation. These are NOT approved;
#: they are enumerated so the census is exact and so silently renaming one away
#: fails this test.
SHAPE_KNOWN_OPEN = {
    ("stack/tanitad/eval/gates.py", "run_d1"):
        "⛔ OPEN — a FOURTH unnamed clone, found 2026-08-28 while closing the "
        "other three. `run_d1` averages ADE over `split_by_episode(eid, "
        "val_frac, s)` for s in seed..seed+n_splits-1 — 8 OVERLAPPING random "
        "20 % holdouts — then decides `passed = admissible and ade < thr` on "
        "that mean-of-split-means and prints `ADE@1s=<ade>±<ade_ci95>` into "
        "the verdict string. The name guard cannot see it (no banned call "
        "name; the verdict key is `passed`, but nothing it reads is tainted "
        "by a banned CALL). ⇒ Fixing it moves every D1 verdict in the "
        "programme, so it is a PI decision, not an agent's. Escalated in "
        "products/P7-TanitEval/ESTIMATOR_CLOSEOUT.md §OPEN-1.",
}


def _shape_census():
    out = {}
    for v in _scan_all(gg.scan_paths_shapes):
        path, _ln, fname, _why = v
        rel = Path(path).resolve().relative_to(REPO).as_posix()
        out.setdefault((rel, fname), []).append(v)
    return out


def test_the_shape_detector_is_actually_RUN_over_the_enforced_scope():
    """The regression that matters most: this test existing at all.

    If ``_scan_all`` ever returns nothing because the scope collapsed (the
    ``_files_under`` skip bug did exactly that), the census would be trivially
    equal to an empty allowlist and the guard would go green on no input."""
    files = [f for root, pattern in ENFORCED_ROOTS
             for f in gg._files_under(root, pattern, SKIP)]
    assert len(files) > 200, (
        f"the shape scan saw only {len(files)} files — the scope collapsed, "
        f"and a guard with no input is not a passing guard")


def test_every_z_sd_over_sqrt_n_in_scope_is_accounted_for():
    found = set(_shape_census())
    known = set(SHAPE_ALLOWLIST) | set(SHAPE_KNOWN_OPEN)
    new = found - known
    assert not new, (
        "a NEW `z · dispersion / sqrt(n)` construction appeared in enforced "
        "scope. If its inputs are genuinely independent, add it to "
        "SHAPE_ALLOWLIST with the argument for why. If they are overlapping "
        "holdouts of one pool, it is `overlapping_holdout_se` under a new "
        "name — use taniteval.ci.episode_cluster_bootstrap instead.\n  "
        + "\n  ".join(f"{p}::{f}" for p, f in sorted(new)))
    gone = known - found
    assert not gone, (
        "a site left the census without the census being updated. A known-open "
        "defect must not vanish by RENAME — `is_declared_estimator_name` "
        "self-exemption hides the arithmetic without fixing it. If it was "
        "genuinely fixed, delete its row here in the same commit.\n  "
        + "\n  ".join(f"{p}::{f}" for p, f in sorted(gone)))


def test_the_closed_unnamed_clone_is_gone_from_the_census():
    """⭐ SITE 3, closed 2026-08-28. ``driving_diagnostic.mean_ci`` was the
    unnamed clone: the right arithmetic under a name no rule could see, with
    six live callers and NO ``estimator`` field on its output.

    It is now ``overlapping_holdout_mean_ci`` — a DECLARED reproduction, so the
    shape detector exempts it by design and the NAME guard governs it instead
    (both spellings are already in ``gg.BANNED_CALLS``). This test pins all
    three halves of that, because any one of them alone would be a fake fix."""
    assert ("stack/scripts/driving_diagnostic.py", "mean_ci") \
        not in _shape_census()
    assert gg.is_declared_estimator_name("overlapping_holdout_mean_ci")
    assert not gg.is_declared_estimator_name("mean_ci"), \
        "the innocent alias must NOT self-exempt — only a declaring name does"
    assert {"mean_ci", "overlapping_holdout_mean_ci"} <= gg.BANNED_CALLS
    import driving_diagnostic as dd
    assert dd.mean_ci is dd.overlapping_holdout_mean_ci
    node = dd.mean_ci([1.0, 2.0, 3.0])
    assert node["estimator"] == "overlapping_holdout_se"
    assert node["deprecated"] is True
    # verbatim ddof=1 arithmetic — a reproduction that changes the number is
    # not a reproduction (ci.overlapping_holdout_se is ddof=0 and would rescale
    # every published D-number by sqrt((n-1)/n)).
    assert node["ci95"] == pytest.approx(1.96 * 1.0 / 3 ** 0.5, abs=5e-5)


# --------------------------------------------------------------------------- #
# DELIBERATE REGRESSION ARM — the guard must still catch the closed defect      #
# --------------------------------------------------------------------------- #
#: ``driving_diagnostic.mean_ci`` VERBATIM as it stood until 2026-08-28.
#: A guard never shown to fail proves nothing, so the removed defect is kept
#: here as a fixture and the detector is required to find it.
PRE_CLOSEOUT_MEAN_CI = '''
def mean_ci(vals):
    """mean, 95% CI (route-resampled protocol, matching gates.run_d1)."""
    n = len(vals)
    m = sum(vals) / n
    std = (sum((v - m) ** 2 for v in vals) / max(1, n - 1)) ** 0.5
    return {"mean": round(m, 4), "ci95": round(1.96 * std / n ** 0.5, 4),
            "std": round(std, 4), "n_splits": n,
            "per_split": [round(v, 4) for v in vals]}
'''

#: The same defect after the FIRST thing an author reaches for when a one-line
#: rule complains: split it over two statements.
LAUNDERED_SE = '''
def summarise(vals):
    n = len(vals)
    sd = statistics.stdev(vals)
    se = sd / math.sqrt(n)
    return {"mean": statistics.mean(vals), "ci95": 1.96 * se}
'''

#: ...and after renaming it to something that sounds harmless.
RENAMED_CLONE = '''
def summarise_splits(vals):
    n = len(vals)
    spread = np.std(vals)
    return {"mean": float(np.mean(vals)), "ci95": 1.959964 * spread / np.sqrt(n)}
'''


@pytest.mark.parametrize("src,label", [
    (PRE_CLOSEOUT_MEAN_CI, "driving_diagnostic.mean_ci as it stood until 2026-08-28"),
    (LAUNDERED_SE, "the same SE split over two statements"),
    (RENAMED_CLONE, "the same SE under a name nobody has banned yet"),
])
def test_shape_guard_fires_on_the_reintroduced_defect(src, label):
    v = gg.scan_source_shapes(src, label)
    assert v, f"the SHAPE guard did NOT fire on: {label}"


def test_shape_guard_does_not_fire_on_a_declared_reproduction():
    """The one in-module exemption, pinned so it cannot silently widen."""
    declared = PRE_CLOSEOUT_MEAN_CI.replace("def mean_ci(",
                                            "def overlapping_holdout_mean_ci(")
    assert gg.scan_source_shapes(declared, "declared") == []


def test_planner_p2_gates_are_decision_grade():
    """The specific file this test was written for, pinned by name.

    A repo-wide scan can be silently narrowed; this cannot."""
    src = (REPO / "taniteval" / "taniteval" / "planner_p2.py").read_text(
        encoding="utf-8")
    assert not gg.scan_source(src, "planner_p2.py")
    tree = ast.parse(src)
    seen = {}
    for name, _ln, value in gg._deciding_exprs(tree):
        seen.setdefault(name, []).append(value)
    for gate in ("G1_pass", "G4_pass"):
        assert gate in seen, f"{gate} disappeared — the gate must stay visible"
    # and the decision-grade estimator must actually be reachable in the file
    calls = {n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "paired_episode_cluster_bootstrap" in calls
    assert "episode_cluster_bootstrap" in calls


def test_legacy_block_is_the_only_exemption():
    """The banned functions may still EXIST — only their verdicts are banned."""
    from taniteval import planner_p2 as P2
    assert hasattr(P2, "_jack_scalar") and hasattr(P2, "_jack_paired"), \
        "history is preserved, not deleted"
    assert P2.LEGACY_BLOCK == "legacy_overlapping_holdout_se"
    # every emitted legacy value self-labels its estimator
    assert P2.DEPRECATED_ESTIMATOR == "overlapping_holdout_se"
    assert P2.DEPRECATED_ESTIMATOR not in P2.DECISION_ESTIMATORS


# --------------------------------------------------------------------------- #
# NEGATIVE CONTROLS — the guard must FIRE on each of these                      #
# --------------------------------------------------------------------------- #
PRE_MIGRATION_G1 = '''
def analyze(col, eids, splits):
    ade = {"head": col["h"], "plan": col["p"]}
    g1_delta = _jack_paired(ade["head"], ade["plan"], eids, splits)
    return {"G1_pass": bool(g1_delta["mean"] > 0 and g1_delta["separated"])}
'''

PRE_MIGRATION_G4 = '''
def analyze(col, eids, splits):
    heldout = {k: _jack_scalar(v, eids, splits) for k, v in col.items()}
    return {"G4_pass": bool(heldout["closed_bike"]["mean"] < 1.6852)}
'''

INLINED = '''
def analyze(a, b, eids, splits):
    return {"G2_pass": bool(_jack_paired(a, b, eids, splits)["mean"] >= 0.2)}
'''

LAUNDERED = '''
def analyze(a, b, eids, splits):
    raw = _jack_scalar(a, eids, splits)
    mid = raw["mean"]
    tail = mid * 1.0
    verdict = tail < 1.6852
    return {"G4_pass": bool(verdict)}
'''

ATTRIBUTE_CALL = '''
def analyze(vals):
    se = ci.overlapping_holdout_se(vals)
    return {"gate_pass": bool(se < 0.1)}
'''

NEW_SIBLING = '''
def analyze(a, b, eids, splits):
    d = _jack_paired_v2(a, b, eids, splits)
    return {"G1_pass": bool(d["mean"] > 0)}
'''

IMPORT_ALIAS = '''
from taniteval.planner_p2 import _jack_paired as agg

def analyze(a, b, eids, splits):
    d = agg(a, b, eids, splits)
    return {"G1_pass": bool(d["mean"] > 0 and d["separated"])}
'''

SUBSCRIPT_STORE = '''
def analyze(col, eids, splits):
    heldout = {k: _jack_scalar(v, eids, splits) for k, v in col.items()}
    res = {}
    res["G4_pass"] = bool(heldout["closed_bike"]["mean"] < 1.6852)
    return res
'''


@pytest.mark.parametrize("src,label", [
    (PRE_MIGRATION_G1, "planner_p2 G1 as it stood until 2026-08-16"),
    (PRE_MIGRATION_G4, "planner_p2 G4 as it stood until 2026-08-16"),
    (INLINED, "inlined bool(_jack_paired(...)['mean'] >= 0.2)"),
    (LAUNDERED, "laundered through three intermediate variables"),
    (ATTRIBUTE_CALL, "ci.overlapping_holdout_se via attribute access"),
    (NEW_SIBLING, "a NEW _jack_* sibling nobody has added to a list yet"),
    (IMPORT_ALIAS, "the banned estimator imported under an innocent alias"),
    (SUBSCRIPT_STORE, "res['G4_pass'] = ... assigned by subscript, not literal"),
])
def test_guard_fires(src, label):
    v = gg.scan_source(src, label)
    assert v, f"guard did NOT fire on: {label}"


def test_import_alias_is_resolved():
    """A rename is not a fix — the loophole a name-keyed rule leaves open."""
    import ast as _ast
    al = gg.banned_import_aliases(_ast.parse(IMPORT_ALIAS))
    assert al == {"agg": "_jack_paired"}


# --------------------------------------------------------------------------- #
# FALSE-POSITIVE CONTROLS — the guard must NOT fire on these                    #
# --------------------------------------------------------------------------- #
DOCSTRING_MENTION = '''
def analyze(a, b, eids):
    """Compute the gate.

    ⛔ Do NOT use _jack_paired / _jack_scalar / overlapping_holdout_se here —
    they bias the point estimate. This is exactly the text a regex guard would
    match against itself.
    """
    # _jack_paired(a, b, eids, splits) was the old line; it is retired.
    d = ci.paired_episode_cluster_bootstrap(a, b, eids)
    return {"G1_pass": bool(d["delta"] > 0 and d["separated"])}
'''

MIGRATED = '''
def analyze(col, eids, splits):
    boot = {k: ci.episode_cluster_bootstrap(v, eids) for k, v in col.items()}
    heldout = {k: _jack_scalar(v, eids, splits) for k, v in col.items()}
    res = {"G4_pass": bool(boot["closed_bike"]["mean"] < 1.7318)}
    res["legacy_overlapping_holdout_se"] = {
        "G4_pass_LEGACY": bool(heldout["closed_bike"]["mean"] < 1.6852)}
    return res
'''


@pytest.mark.parametrize("src,label", [
    (DOCSTRING_MENTION, "a docstring and a comment naming the banned family"),
    (MIGRATED, "the migrated shape, legacy value kept under a _LEGACY key"),
])
def test_guard_does_not_fire(src, label):
    v = gg.scan_source(src, label)
    assert not v, f"FALSE POSITIVE on {label}: {[str(x) for x in v]}"


def test_docstring_mention_is_not_a_violation():
    """The self-match trap, pinned on its own.

    A ``grep -n '_jack_'`` over ``DOCSTRING_MENTION`` returns 2 hits. The AST
    guard returns 0. That difference IS the reason this is an AST walk."""
    assert "_jack_paired" in DOCSTRING_MENTION          # a regex would match
    assert gg.scan_source(DOCSTRING_MENTION, "d") == []  # the AST does not


def test_is_deciding_name():
    for good in ("G1_pass", "G4_pass", "gate_pass", "verdict",
                 "G4_pass_ci_separated", "planner_verdict"):
        assert gg.is_deciding_name(good), good
    for skip in ("G1_pass_LEGACY", "legacy_overlapping_holdout_se",
                 "ade2s", "n_windows", "passthrough_dim", "compass_bearing"):
        assert not gg.is_deciding_name(skip), skip
