"""PC2 — *"the hierarchy is in the scored loop"* as a code assertion.

These tests pin the property the HPP-0 audit had to establish by reading three
source files: the deployed headline surface (``rollout_decode`` under the
expert's true future actions) **cannot** express a decision at any level, and
therefore may not be reported as a hierarchy number.

The four things pinned here, each one a failure that has actually happened:

1. A path that skips a brain and CLAIMS a hierarchy **raises**
   (:class:`hierarchy_guard.HierarchyBypass`), naming the missing seam and
   whether it was absent or merely never called.
2. A path that traverses all three brains passes — so the assertion is not
   vacuously strict.
3. ``absent`` (no such module on this arch, e.g. REF-C) is distinguished from
   ``bypassed`` (module exists, never called), because the corrections differ —
   and **neither is a pass**.
4. The *decision* bypass, which is separate and was not in the review: an arm
   can traverse all three seams and still be handed the expert's actions.
"""
from __future__ import annotations

import pytest
import torch

from taniteval import hierarchy_guard as HG


# --------------------------------------------------------------------------- #
# minimal arms                                                                  #
# --------------------------------------------------------------------------- #
class _Lin(torch.nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, x, *a, **k):
        return self.lin(x)


class _Predictor(torch.nn.Module):
    def __init__(self, d=4, with_intent=True):
        super().__init__()
        self.core = torch.nn.Linear(d, d)
        self.intent_proj = _Lin(d) if with_intent else None

    def forward(self, x, intent=None):
        h = self.core(x)
        if intent is not None and self.intent_proj is not None:
            h = h + self.intent_proj(intent)
        return h


class FullArm(torch.nn.Module):
    def __init__(self, with_intent=True):
        super().__init__()
        self.strategic_policy = _Lin()
        self.tactical_policy = _Lin()
        self.predictor = _Predictor(with_intent=with_intent)


class FlatArm(torch.nn.Module):
    """REF-C shape: a predictor, no strategic/tactical brains at all."""

    def __init__(self):
        super().__init__()
        self.predictor = _Predictor()


X = torch.zeros(2, 4)


def _hierarchical_pass(m):
    ctx = m.strategic_policy(X)
    intent = m.tactical_policy(ctx)
    return m.predictor(X, intent=intent)


def _bypassing_pass(m):
    """Exactly ``metric_dynamics.rollout_decode``: predictor only, no
    intent/ctx/nav."""
    return m.predictor(X)


# --------------------------------------------------------------------------- #
# 1 + 2. the assertion discriminates                                            #
# --------------------------------------------------------------------------- #
def test_bypassing_path_claiming_hierarchy_raises():
    m = FullArm()
    with HG.HierarchyTrace(m) as tr:
        _bypassing_pass(m)
    with pytest.raises(HG.HierarchyBypass) as e:
        HG.assert_hierarchy_traversed(tr, block="taniteval.rollout/collect",
                                      claim="hierarchy")
    msg = str(e.value)
    assert "PC2 VIOLATION" in msg
    assert "strategic" in msg and "tactical" in msg
    assert "module exists, never called" in msg
    # the known-bypass registry explains the surface instead of just failing
    assert "rollout_decode" in msg


def test_traversing_path_passes():
    m = FullArm()
    with HG.HierarchyTrace(m) as tr:
        _hierarchical_pass(m)
    rep = HG.assert_hierarchy_traversed(tr, block="taniteval.hierarchy/run")
    assert rep["pc2_pass"] is True
    assert rep["hierarchy_traversed"] is True
    assert rep["missing_seams"] == []
    assert all(v > 0 for v in rep["counts"].values())


def test_counts_are_real_forward_calls():
    m = FullArm()
    with HG.HierarchyTrace(m) as tr:
        for _ in range(3):
            _hierarchical_pass(m)
    assert tr.counts["strategic"] == 3
    assert tr.counts["tactical"] == 3
    assert tr.counts["operative_intent"] == 3


def test_hooks_are_removed_on_exit():
    m = FullArm()
    with HG.HierarchyTrace(m) as tr:
        _hierarchical_pass(m)
    before = dict(tr.counts)
    _hierarchical_pass(m)                 # outside the trace: must not count
    assert tr.counts == before


# --------------------------------------------------------------------------- #
# 3. absent != bypassed, and a SKIP IS NOT A PASS                               #
# --------------------------------------------------------------------------- #
def test_absent_modules_are_reported_as_absent_and_still_fail():
    m = FlatArm()
    with HG.HierarchyTrace(m) as tr:
        _bypassing_pass(m)
    with pytest.raises(HG.HierarchyBypass) as e:
        HG.assert_hierarchy_traversed(tr, block="taniteval.refc_eval/collect",
                                      claim="hierarchy")
    assert "no such module on this arch" in str(e.value)
    rep = HG.assert_hierarchy_traversed(tr, block="x", strict=False)
    assert set(rep["absent_modules"]) == {"strategic", "tactical"}
    # the two brains cannot be "bypassed" — there is nothing to bypass; the
    # intent seam CAN be and was, and the two diagnoses stay distinguishable.
    assert rep["bypassed_seams"] == ["operative_intent"]
    assert rep["pc2_pass"] is False             # ...and it is still not a pass


def test_predictor_without_intent_proj_counts_as_absent():
    """The third seam is hooked on ``intent_proj``, not on the predictor: the
    predictor runs on every path, hierarchical or not, so hooking it would
    prove nothing."""
    m = FullArm(with_intent=False)
    with HG.HierarchyTrace(m) as tr:
        m.strategic_policy(X)
        m.tactical_policy(X)
        m.predictor(X)
    rep = HG.assert_hierarchy_traversed(tr, block="x", strict=False)
    assert rep["absent_modules"] == ["operative_intent"]
    assert rep["missing_seams"] == ["operative_intent"]
    assert rep["pc2_pass"] is False


def test_require_subset_lets_a_two_seam_claim_pass():
    """HP-3's tactical surface legitimately claims only strategic+tactical."""
    m = FullArm()
    with HG.HierarchyTrace(m) as tr:
        m.strategic_policy(X)
        m.tactical_policy(X)
    rep = HG.assert_hierarchy_traversed(
        tr, block="taniteval.strategic_probes", claim="HP-3",
        require=("strategic", "tactical"))
    assert rep["pc2_pass"] is True


# --------------------------------------------------------------------------- #
# 4. the DECISION bypass — separate, and the bigger one                         #
# --------------------------------------------------------------------------- #
def test_expert_future_actions_are_not_a_decision():
    with pytest.raises(HG.HierarchyBypass) as e:
        HG.assert_actions_are_chosen(block="taniteval.rollout/collect",
                                     actions_source="expert_future")
    assert "wm_fidelity_ade_2s" in str(e.value)


@pytest.mark.parametrize("src", ["model_chosen", "planner"])
def test_chosen_actions_pass(src):
    rep = HG.assert_actions_are_chosen(block="b", actions_source=src)
    assert rep["actions_are_chosen"] is True


def test_guarded_combines_both_checks():
    m = FullArm()
    out, rep = HG.guarded(m, lambda: _hierarchical_pass(m),
                          block="b", claim="hierarchy",
                          actions_source="planner")
    assert out is not None and rep["pc2_pass"] is True
    with pytest.raises(HG.HierarchyBypass):
        HG.guarded(m, lambda: _hierarchical_pass(m), block="b",
                   claim="hierarchy", actions_source="expert_future")


# --------------------------------------------------------------------------- #
# 5. the real callsite: rollout.collect stamps its own bypass                   #
# --------------------------------------------------------------------------- #
def test_rollout_collect_declares_itself_a_wm_fidelity_surface():
    """The headline surface must carry its own PC2 record, reading FALSE, with
    the honest metric name — non-strict so the diagnostic stays runnable."""
    from taniteval import rollout

    class _Ep:
        episode_id = "e0"
        feats = torch.zeros(64, 8)
        actions = torch.zeros(64, 2)
        poses = torch.zeros(64, 4)

    ep = _Ep()
    ep.poses[:, 0] = torch.arange(64, dtype=torch.float32)
    ep.poses[:, 3] = 10.0

    class _RolloutPredictor(torch.nn.Module):
        """``OperativePredictor``'s contract as ``rollout_decode`` uses it:
        ``predictor(win_s, win_a)[1]`` -> the 1-step latent. ``intent_proj``
        exists and is called ONLY on an intent-carrying path — which this
        surface never takes. That is the defect under test."""

        def __init__(self, d=8):
            super().__init__()
            self.core = torch.nn.Linear(d, d)
            self.intent_proj = _Lin(d)

        def forward(self, states, actions, intent=None):
            z = self.core(states[:, -1])
            if intent is not None:
                z = z + self.intent_proj(intent)
            return {1: z}

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.strategic_policy = _Lin(8)
            self.tactical_policy = _Lin(8)
            self.predictor = _RolloutPredictor(8)

        def encode_window(self, fw):
            return fw

    m = _M()
    win = rollout.collect(m, lambda z, zh: torch.zeros(z.shape[0], 3),
                          [ep], "cpu", window=8, fwd_k=20, stride=8, batch=4)
    pc2 = win["pc2"]
    assert pc2["pc2_pass"] is False
    assert pc2["honest_metric_name"] == "wm_fidelity_ade_2s"
    assert set(pc2["missing_seams"]) == {"strategic", "tactical",
                                         "operative_intent"}
    assert pc2["bypassed_seams"], "the modules exist and were never called"
    assert (pc2["decision_surface"]["actions_source"] == "expert_future"
            and pc2["decision_surface"]["actions_are_chosen"] is False)
