"""Tests for tanitad/eval/goal_provenance.py — the COMPUTED admissibility gate.

The instrument is proved against the failure it was built for, not a toy: the
fixture below is the v6 shape (one shared trunk, a goal path, a situation path)
and the leak arm splices the situation output into the goal path **behind a
``detach()``**, which is how every downward port in ``V6Stack.forward`` is
already wired (``v6.py:4341`` ``_cut``). That arm is the positive control, and
:func:`test_the_gradient_probe_MISSES_what_the_forward_probe_CATCHES` is the
reason this module exists beside ``assert_isolation`` rather than inside it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from tanitad.eval.goal_provenance import (  # noqa: E402
    ProvenanceViolation, assert_information_disjoint, audit_arm, audit_arms,
    classify_edge, compare_to_gradient_probe, dependency_matrix,
    determinism_check, module_runner, probe_dependency)


# --------------------------------------------------------------------------- #
# the fixture: the v6 shape in miniature                                      #
# --------------------------------------------------------------------------- #
class _Trunk(nn.Module):
    """The shared vision encoder — v6's ``shared_encoder=True`` default."""

    def __init__(self, d=8):
        super().__init__()
        self.net = nn.Linear(d, d)

    def forward(self, x):
        return torch.tanh(self.net(x))


class _Head(nn.Module):
    def __init__(self, d=8, out=4):
        super().__init__()
        self.net = nn.Linear(d, out)

    def forward(self, x):
        return self.net(x)


class _Hier(nn.Module):
    """trunk -> {goal, situation}, with an OPTIONAL situation->goal wire.

    ``leak="detached"`` is the important arm: it carries the full signal and no
    gradient, which is exactly the shape a real violation would take in this
    codebase.
    """

    def __init__(self, d=8, leak: str | None = None):
        super().__init__()
        self.trunk = _Trunk(d)
        self.goal = _Head(d, 4)
        self.situation = _Head(d, 3)
        self.leak = leak
        self.leak_proj = nn.Linear(3, d)
        nn.init.constant_(self.leak_proj.weight, 0.5)
        nn.init.zeros_(self.leak_proj.bias)

    def forward(self, x):
        z = self.trunk(x)
        s = self.situation(z)
        zg = z
        if self.leak == "detached":
            zg = z + self.leak_proj(s.detach())
        elif self.leak == "live":
            zg = z + self.leak_proj(s)
        g = self.goal(zg)
        return {"g": g, "s": s}


def _batch(n=4, d=8, seed=0):
    torch.manual_seed(seed)
    return {"x": torch.randn(n, d)}


def _runner(model):
    model.eval()
    return module_runner(
        model, _batch(),
        nodes={"goal": "goal", "situation_output": "situation",
               "trunk": "trunk"})


ROLES = dict(goal_roles=["goal"], situation_roles=["situation_output"])


# --- 1. the probe detects a live wire --------------------------------------- #
def test_probe_detects_a_live_situation_to_goal_wire():
    r = probe_dependency(_runner(_Hier(leak="live")), "situation_output",
                         ["goal"])
    assert r["any_dependence"] is True
    assert r["targets"]["goal"]["depends"] is True
    assert r["targets"]["goal"]["max_abs"] > 0


# --- 2. ⭐ the detached wire — the case the gradient probe cannot see -------- #
def test_probe_detects_a_DETACHED_wire_because_detach_carries_full_signal():
    """``_cut()``/``detach()`` is how every downward goal port in
    ``V6Stack.forward`` is wired. A leak spliced in there is invisible to an
    autograd probe and total as an information path."""
    r = probe_dependency(_runner(_Hier(leak="detached")), "situation_output",
                         ["goal"])
    assert r["any_dependence"] is True, (
        "a detached wire carries the FULL signal — an information probe must "
        "see it even though no gradient flows")


def test_the_gradient_probe_MISSES_what_the_forward_probe_CATCHES():
    """⭐ THE DISCRIMINATOR, and the justification for a second instrument.

    On the SAME graph and batch, the forward interventional probe reports a
    path and the backward autograd probe — the shape ``assert_isolation`` uses
    — reports none. This is measured here rather than argued in the docstring.
    """
    m = _Hier(leak="detached")
    m.eval()
    rep = compare_to_gradient_probe(
        m, _batch(), _runner(m), source="situation_output",
        source_params=[(n, p) for n, p in m.named_parameters()
                       if n.startswith("situation.")],
        target_roles=["goal"], target_output_keys=["g"])
    assert rep["forward_information_path"] is True
    assert rep["backward_gradient_path"] is False, (
        "detach() severs the gradient — that is why a gradient probe cannot "
        "answer the admissibility question")
    assert rep["probes_disagree"] is True


# --- 3. the shared trunk is a COMMON ANCESTOR, not a back door -------------- #
def test_a_shared_trunk_reads_as_COMMON_ANCESTOR_not_as_a_path():
    """The ruling requires the shared-trunk disclosure. A correlational test
    cannot make this distinction; an interventional one makes it mechanically."""
    run = _runner(_Hier(leak=None))
    mat = dependency_matrix(run, ["goal", "situation_output"])
    shared = probe_dependency(run, "trunk", ["goal", "situation_output"])
    e = classify_edge(mat, "situation_output", "goal", shared)
    assert e["relation"] == "COMMON_ANCESTOR"
    assert e["direct"] is False and e["shares_ancestor"] is True
    assert "NOT a back door" in e["_reads"]


def test_the_clean_arm_measures_DISJOINT_and_discloses_the_trunk():
    m = _Hier(leak=None)
    rep = audit_arm(arm="clean", run=_runner(m), shared_input_role="trunk",
                    positive_control=probe_dependency(
                        _runner(_Hier(leak="detached")), "situation_output",
                        ["goal"]),
                    **ROLES)
    assert rep["status"] == "MEASURED" and rep["verdict"] == "DISJOINT"
    assert rep["shared_trunk"] is not None
    assert rep["shared_trunk"]["is_back_door"] is False
    assert "COMMON ANCESTOR" in rep["shared_trunk"]["_statement"]


# --- 4. the REVERSE direction, which nobody had checked --------------------- #
def test_a_DISJOINT_pair_is_not_mistaken_for_a_DEAD_one():
    """⛔ REGRESSION. The dead-node rule first judged deadness from the
    goal/situation matrix alone — but in a DISJOINT pair neither node moves the
    other BY DEFINITION, so both read as constant and the instrument reported
    its own success as an artefact (UNPOWERED instead of DISJOINT).

    Deadness may only be judged against a source that SHOULD move the node.
    """
    m = _Hier(leak=None)
    rep = audit_arm(arm="clean", run=_runner(m), shared_input_role="trunk",
                    positive_control=probe_dependency(
                        _runner(_Hier(leak="detached")), "situation_output",
                        ["goal"]), **ROLES)
    assert rep["dead_goal_nodes"] == [], (
        "the goal node IS moved by the trunk, so it is not dead")
    assert rep["verdict"] == "DISJOINT" and rep["status"] == "MEASURED"


def test_a_genuinely_constant_goal_node_IS_flagged_dead():
    """The other side of the same rule — a zero-init head, which is what the
    real v6 ``cond_tac_dyn`` is on an untrained model."""

    class _DeadGoal(_Hier):
        def __init__(self):
            super().__init__()
            nn.init.zeros_(self.goal.net.weight)
            nn.init.zeros_(self.goal.net.bias)

    rep = audit_arm(arm="zero-init", run=_runner(_DeadGoal()),
                    shared_input_role="trunk",
                    positive_control=probe_dependency(
                        _runner(_Hier(leak="detached")), "situation_output",
                        ["goal"]), **ROLES)
    assert rep["dead_goal_nodes"] == ["goal"]
    assert rep["status"] == "UNPOWERED" and rep["verdict"] is None
    assert "CONSTANT" in rep["reason"]


def test_the_reverse_direction_is_audited_and_can_fail_on_its_own():
    """goal -> situation is a violation of the SAME symmetric rule. A model
    whose situation head reads the goal must not pass."""

    class _Rev(_Hier):
        def forward(self, x):
            z = self.trunk(x)
            g = self.goal(z)
            s = self.situation(z + self.leak_proj(
                torch.nn.functional.pad(g, (0, 0))[:, :3].detach()))
            return {"g": g, "s": s}

    rep = audit_arm(arm="reverse-leak", run=_runner(_Rev()),
                    shared_input_role="trunk",
                    positive_control=probe_dependency(
                        _runner(_Hier(leak="detached")), "situation_output",
                        ["goal"]),
                    **ROLES)
    assert rep["verdict"] == "INADMISSIBLE"
    assert rep["reverse_violations"], (
        "the goal->situation direction must be reported on its own axis")
    with pytest.raises(ProvenanceViolation, match="forward information path"):
        assert_information_disjoint(rep)


# --- 5. C109 — an inert control must not license a clean verdict ------------ #
def test_an_inert_positive_control_downgrades_the_audit_to_UNPOWERED():
    """C109: a cited positive control that was inert by construction proved
    nothing. An all-clean matrix from a blind probe is not evidence."""
    inert = probe_dependency(_runner(_Hier(leak=None)), "situation_output",
                             ["goal"])
    assert inert["any_dependence"] is False
    rep = audit_arm(arm="blind", run=_runner(_Hier(leak=None)),
                    positive_control=inert, **ROLES)
    assert rep["status"] == "UNPOWERED" and rep["verdict"] is None
    assert "positive control did not fire" in rep["reason"]


def test_the_gate_REFUSES_an_unpowered_audit_by_default():
    rep = audit_arm(arm="blind", run=_runner(_Hier(leak=None)),
                    positive_control=probe_dependency(
                        _runner(_Hier(leak=None)), "situation_output",
                        ["goal"]), **ROLES)
    with pytest.raises(ProvenanceViolation, match="UNPOWERED"):
        assert_information_disjoint(rep)
    assert assert_information_disjoint(rep, allow_unpowered=True)["ok"]


# --- 6. determinism gate ---------------------------------------------------- #
def test_a_nondeterministic_forward_is_UNPOWERED_not_leaky():
    class _Noisy(_Hier):
        def forward(self, x):
            # ⚠️ the noise must be UPSTREAM of an OBSERVED node — the check
            # covers the nodes the audit gives verdicts about, and nothing else.
            z = self.trunk(x)
            return {"g": self.goal(z + torch.randn_like(z)),
                    "s": self.situation(z)}

    m = _Noisy()
    assert determinism_check(_runner(m))["deterministic"] is False
    rep = audit_arm(arm="noisy", run=_runner(m), **ROLES)
    assert rep["status"] == "UNPOWERED" and rep["verdict"] is None
    assert "non-deterministic" in rep["reason"]


def test_determinism_check_passes_on_the_real_fixture():
    d = determinism_check(_runner(_Hier(leak=None)))
    assert d["deterministic"] is True and d["max_drift"] == 0.0


# --- 7. per-arm, never per-study (C107) ------------------------------------- #
def test_one_inadmissible_arm_makes_the_PANEL_inadmissible():
    ctrl = probe_dependency(_runner(_Hier(leak="detached")),
                            "situation_output", ["goal"])
    rep = audit_arms([
        dict(arm="a-clean", run=_runner(_Hier(leak=None)),
             positive_control=ctrl, **ROLES),
        dict(arm="b-detached-leak", run=_runner(_Hier(leak="detached")),
             positive_control=ctrl, **ROLES),
    ])
    assert rep["n_arms"] == 2 and rep["verdict"] == "INADMISSIBLE"
    assert rep["inadmissible"] == ["b-detached-leak"]
    assert set(rep["arms"]) == {"a-clean", "b-detached-leak"}
    with pytest.raises(ProvenanceViolation):
        assert_information_disjoint(rep)


def test_every_arm_carries_its_OWN_verdict_row():
    ctrl = probe_dependency(_runner(_Hier(leak="detached")),
                            "situation_output", ["goal"])
    rep = audit_arms([dict(arm=f"arm{i}", run=_runner(_Hier(leak=None)),
                           positive_control=ctrl, **ROLES) for i in range(3)])
    assert rep["verdict"] == "DISJOINT" and len(rep["arms"]) == 3
    assert all(r["verdict"] == "DISJOINT" for r in rep["arms"].values())
    assert assert_information_disjoint(rep)["n_arms"] == 3


# --- 8. contract errors ----------------------------------------------------- #
def test_probe_refuses_an_unobservable_node_rather_than_reporting_clean():
    """Absence found at one location is not absence — a role that was never
    wired into the runner must RAISE, not read as disjoint."""
    with pytest.raises(KeyError, match="not observable"):
        probe_dependency(_runner(_Hier()), "situation_output", ["nope"])
    with pytest.raises(KeyError, match="not observable"):
        probe_dependency(_runner(_Hier()), "missing_src", ["goal"])


def test_the_INPUT_probe_has_its_own_positive_control():
    """⛔ REGRESSION. ``module_runner`` perturbed the input, recorded it as the
    node's value, and then ran the model on the UNPERTURBED batch — so every
    input read as unread, including the one the whole graph descends from.

    The submodule-intervention control did not catch it: a control that powers
    one code path does not power another. This is that path's own control, and
    it is a TAUTOLOGY on purpose — a head fed only ``x`` MUST depend on ``x``,
    so a False here can only be an inert probe.
    """
    m = _Hier(leak=None)
    m.eval()
    run = module_runner(m, _batch(),
                        nodes={"goal": "goal", "situation_output": "situation"},
                        input_nodes={"in_x": "x"})
    r = probe_dependency(run, "in_x", ["goal", "situation_output"])
    assert r["targets"]["goal"]["depends"] is True, (
        "the goal head descends from x by construction; False here means the "
        "perturbed input never reached the model")
    assert r["targets"]["situation_output"]["depends"] is True


def test_matrix_reports_both_directions_as_cells():
    mat = dependency_matrix(_runner(_Hier(leak="detached")),
                            ["goal", "situation_output"])
    assert set(mat["cells"]) == {"goal", "situation_output"}
    assert ["situation_output", "goal"] in mat["edges"]
    assert ["goal", "situation_output"] not in mat["edges"]
