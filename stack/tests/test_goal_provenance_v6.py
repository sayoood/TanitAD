"""⛔ THE GATE ON THE REAL v6 STACK — goal provenance, computed, per arm.

This is the half that makes the PI's 2026-08-03 ruling a MECHANISM rather than
a rule someone must remember to apply. ``goal_admissibility.py`` was written for
the same ruling and had **zero call sites for 12 days** (stale-blocker sweep
2026-08-16, item 5); a guard nothing invokes is C108, and C111 proved that costs
real exposure within a day. These tests invoke it, on the real architecture,
every time ``pytest`` runs.

WHAT IS PINNED (all MEASURED 2026-08-18 by
``stack/scripts/audit_goal_provenance.py``, artifact
``…/incoming/2026-08-18-goal-provenance-audit/goal_provenance_audit.json``):

  * every goal HEAD is a function of ``frames`` ALONE — not of ``actions``, not
    of ``v0``. This verifies BY COMPUTATION the prose claim at ``v6.py:62``
    (*"``v0`` (initial speed) enters ONLY the unicycle"*), which until now was
    only asserted;
  * ``v0`` reaches exactly ONE goal node, ``emission`` — the unicycle rollout,
    the single allowlisted non-vision edge;
  * there is NO situation-classifier node in the graph, and the probe that would
    find one FIRES on a deliberately wired leak (the per-arm positive control).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]


def _load_runner():
    """Load the runner BY PATH, without touching ``sys.path``.

    ⚠️ The obvious ``sys.path.insert(0, stack/scripts)`` is test pollution: it is
    global to the whole pytest session and would let any later
    ``import <name>`` resolve to a script instead of a package module. It is a
    live hazard here rather than a theoretical one — **``stack/scripts/
    goal_provenance.py`` already exists** (a DIFFERENT instrument: it discloses
    whether an evaluated goal was oracle-GT-future or produced-from-vision, cited
    at ``GATE_PROTOCOL.md:186``) and shares a basename with
    ``tanitad/eval/goal_provenance.py``. Loading by path keeps the two apart.
    """
    p = _STACK / "scripts" / "audit_goal_provenance.py"
    spec = importlib.util.spec_from_file_location("_audit_goal_provenance", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A = _load_runner()

from tanitad.eval.goal_provenance import (  # noqa: E402
    ProvenanceViolation, assert_information_disjoint, probe_dependency)

#: The goal HEADS — the decision path. ``emission`` is the unicycle and is
#: excluded here on purpose: it is the one node ``v0`` may legally reach.
_HEADS = ("goal_head_str", "goal_cond_tac", "goal_head_tac", "goal_cond_op",
          "goal_cond_op_lat", "goal_cond_op_lon", "goal_head_tac_lat",
          "goal_head_tac_lon")


def _inputs_for(arm: str):
    model = A._build(A.ARMS[arm])
    batch = model.synthetic_batch(2)
    run = A._runner(model, batch)
    return {r: probe_dependency(run, r, list(A._goal_nodes(model)))
            for r in A.INPUT_NODES}


@pytest.mark.parametrize("arm", ["default", "factored-goal"])
def test_no_goal_HEAD_reads_v0_or_actions_on_the_real_stack(arm):
    """⛔ THE BINDING ONE. Ego speed or past actions inside a goal HEAD would put
    non-vision state into the goal decision itself."""
    reps = _inputs_for(arm)
    for src in ("in_v0", "in_actions"):
        hits = [t for t, v in reps[src]["targets"].items()
                if v["depends"] and t in _HEADS]
        assert not hits, (
            f"{arm}: {src} reaches goal HEAD(s) {hits} — non-vision state in "
            f"the goal decision. v6.py:62 claims v0 enters ONLY the unicycle.")


@pytest.mark.parametrize("arm", ["default", "factored-goal"])
def test_the_goal_heads_ARE_vision_derived_so_the_probe_is_not_blind(arm):
    """The tautology that powers the test above: a goal head that depended on
    NOTHING would pass it vacuously. ``frames`` must move every head."""
    reps = _inputs_for(arm)
    heads = [t for t in reps["in_frames"]["targets"] if t in _HEADS]
    assert heads, "no goal head observed — the node map is wrong"
    unmoved = [t for t in heads if not reps["in_frames"]["targets"][t]["depends"]]
    assert not unmoved, (
        f"{arm}: goal heads {unmoved} do not depend on frames; the probe is "
        f"blind on this arm and its clean readings prove nothing")


def test_v0_reaches_the_unicycle_and_ONLY_the_unicycle():
    reps = _inputs_for("default")
    hit = sorted(t for t, v in reps["in_v0"]["targets"].items() if v["depends"])
    assert hit == ["goal_emission"], (
        f"v0 reaches {hit}; the allowlist authorises exactly "
        f"{sorted(t for _s, t in A.EXPECTED_NON_VISION)} (v6.py:62)")


def test_the_allowlist_stays_a_single_named_edge():
    """An allowlist is where a real violation hides. Pin its size and content so
    widening it is a deliberate, reviewed act rather than a quiet edit."""
    assert A.EXPECTED_NON_VISION == {("in_v0", "goal_emission")}


def test_there_is_NO_situation_classifier_node_in_the_v6_graph():
    """Two probes, per the standing rule — the module tree AND the forward's
    input contract. Absence at one location is not absence."""
    model = A._build({})
    names = [n for n, _ in model.named_modules()]
    assert not [n for n in names
                if "sitclf" in n or "situation" in n.lower()], names
    assert set(model.synthetic_batch(2)) == {"frames", "actions", "v0"}


@pytest.mark.parametrize("arm", ["default", "tac-goal-cond"])
def test_the_per_arm_positive_control_FIRES_on_a_detached_leak(arm):
    """⭐ C109. A clean matrix from a probe that cannot detect anything is not
    evidence. The control is wired behind ``detach()`` — the shape every
    downward goal port in ``V6Stack.forward`` already has (``v6.py:4341``)."""
    leak = A._wire_leak(A._build(A.ARMS[arm]))
    ctrl = probe_dependency(
        A._runner(leak, leak.synthetic_batch(2),
                  extra_nodes={"situation_output": "goal_head_tac.situation"}),
        "situation_output", ["goal_head_tac"])
    assert ctrl["any_dependence"] is True, (
        f"{arm}: a deliberately wired situation->goal leak was NOT detected; "
        f"every clean verdict on this arm is unpowered")


def test_the_gate_RAISES_on_the_wired_leak_panel():
    """The panel of wired-leak arms must be rejected. A gate that never fires
    is C108 — a rule with no mechanism."""
    panel = A._in_graph_audit({"default": {}})["positive_control_panel"]
    assert panel["verdict"] == "INADMISSIBLE"
    with pytest.raises(ProvenanceViolation, match="GOAL-PROVENANCE VIOLATION"):
        assert_information_disjoint(panel)


def test_the_structural_half_passes_and_scans_a_nonzero_surface():
    """The situation classifier is offline, so the in-graph probe alone would
    pass for the trivial reason that nothing is there. This is the second
    location — and it must have actually scanned something."""
    s = A._structural_audit(_STACK)
    assert s["n_model_and_trainer_sources_scanned"] > 10, s
    assert s["situation_sources_scanned"], "scanned no situation source at all"
    assert s["sources_importing_situation_path"] == []
    assert s["goal_symbols_in_situation_path"] == []
    assert s["ok"] is True
