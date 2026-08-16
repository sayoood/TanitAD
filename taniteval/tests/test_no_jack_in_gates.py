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
ENFORCED_ROOTS = [REPO / "taniteval" / "taniteval",
                  REPO / "taniteval" / "tools",
                  REPO / "stack" / "tanitad",
                  REPO / "stack" / "scripts"]
SKIP = ("__pycache__", "/.claude/", "/experiments/")


# --------------------------------------------------------------------------- #
# THE GUARD                                                                    #
# --------------------------------------------------------------------------- #
def test_no_gate_is_decided_by_the_banned_estimator():
    v = gg.scan_paths(ENFORCED_ROOTS, skip=SKIP)
    assert not v, (
        "a GATE verdict is computed from the banned `overlapping_holdout_se` "
        "family. That estimator biases the POINT ESTIMATE (mean-of-split-means, "
        "not full_set) and on paired deltas has been measured at x-4.15 "
        "INCLUDING A SIGN FLIP — a gate decided on it can be wrong in sign.\n"
        "Replace it with ci.episode_cluster_bootstrap / "
        "ci.paired_episode_cluster_bootstrap and keep the old value beside it "
        "under a key ending `_LEGACY`.\n  " + "\n  ".join(str(x) for x in v))


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
