"""THE CAPACITY CONTROL for the goal-distance selector (V6F_PLANNER_DESIGN §5.3).

WHY THIS EXISTS. ``GoalDistanceScorer`` wins the fan with **+267** parameters and
a hard-wired ``−‖endpoint − ĝ‖`` rule. Two stories fit that observation and the
``"goal"`` arm alone cannot separate them:

  * MECHANISM — a candidate-INDEPENDENT reference has no degenerate minimiser;
  * CAPACITY  — the selector head was underpowered and any extra parameters on
    the same inputs would have done as well.

Reading the second as the first is the C6 confound verbatim. ``"mlp"`` is the
pre-registered control: **identical inputs, no distance prior, ~127x the
parameters**. §5.3 commits the refutation in advance — *if ``"mlp"`` matches or
beats ``"goal"``, SEL-1's story is wrong.*

These tests pin the three properties that make it a CONTROL rather than merely
another arm: (1) it is information-MATCHED, (2) it perturbs no pre-existing
tensor, and (3) ``"none"`` stays byte-identical so the live S-W resume is
untouched.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.models.v6 import (  # noqa: E402
    GoalDistanceScorer, MLPCandidateScorer, V6Config)


D_GOAL, N_CAND, HIDDEN = 128, 8, 256


def _n(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------- the contract --

def test_config_now_accepts_mlp_and_still_refuses_nonsense():
    assert V6Config(selector="mlp").selector == "mlp"
    assert V6Config().selector == "none"          # default unmoved
    with pytest.raises(ValueError, match="none|goal|mlp"):
        V6Config(selector="transformer")
    with pytest.raises(ValueError, match="selector_mlp_hidden"):
        V6Config(selector="mlp", selector_mlp_hidden=0)


def test_param_delta_is_measured_not_assumed():
    """⚠️ The design doc's '+41,089' was a DESIGN-TIME ESTIMATE, never realised
    in code. This test states what the implementation actually costs, so the
    number that gets published is MEASURED. Recompute, do not inherit.

    Shape: Linear(2 + d_goal, hidden) + Linear(hidden, 1) + cand_bias[N].
    """
    mlp = MLPCandidateScorer(D_GOAL, N_CAND, hidden=HIDDEN)
    goal = GoalDistanceScorer(D_GOAL, N_CAND)
    expect = ((2 + D_GOAL) * HIDDEN + HIDDEN) + (HIDDEN + 1) + N_CAND
    assert _n(mlp) == expect == 33_801
    assert _n(goal) == 267
    # the control must be decisively larger, or it is not a capacity control
    assert _n(mlp) > 50 * _n(goal)


def test_it_is_information_MATCHED_not_information_enriched():
    """The control must read the SAME inputs the goal rule reads — endpoint and
    e_g_tac. Handing it the full 60x2 path would make it an *information*
    control, and its result would no longer speak to capacity.

    Measured by construction: only the last waypoint may influence the score.
    """
    torch.manual_seed(0)
    mlp = MLPCandidateScorer(D_GOAL, N_CAND, hidden=32)
    torch.nn.init.normal_(mlp.fc2.weight, std=0.5)      # un-zero the output
    wp = torch.randn(3, N_CAND, 60, 2)
    g = torch.randn(3, D_GOAL)
    base = mlp(wp, g)["score"]
    moved = wp.clone()
    moved[:, :, :-1] += 7.0                              # everything BUT the end
    assert torch.equal(mlp(moved, g)["score"], base)
    moved2 = wp.clone()
    moved2[:, :, -1] += 7.0                              # the endpoint only
    assert not torch.allclose(mlp(moved2, g)["score"], base)
    assert mlp.fc1.in_features == 2 + D_GOAL


def test_it_emits_no_fabricated_goal_point():
    """It has no goal point. A zero-filled ``goal_point`` would be a fabricated
    field that later reads as a measurement — the same class as an interval
    quoted without its estimator."""
    out = MLPCandidateScorer(D_GOAL, N_CAND, hidden=16)(
        torch.randn(2, N_CAND, 60, 2), torch.randn(2, D_GOAL))
    assert set(out) == {"score", "mechanism"}
    assert out["mechanism"] == "mlp"
    assert "goal_point" not in out and "goal_dist" not in out


def test_score_convention_matches_the_goal_scorer():
    """Higher == better for BOTH, so the two are drop-in comparable and the
    trainer's ``sel_*`` logging (which reads only ``sel_score``) is unchanged.
    A silent sign flip here would invert the whole comparison."""
    wp, g = torch.randn(4, N_CAND, 60, 2), torch.randn(4, D_GOAL)
    a = GoalDistanceScorer(D_GOAL, N_CAND)(wp, g)["score"]
    b = MLPCandidateScorer(D_GOAL, N_CAND, hidden=16)(wp, g)["score"]
    assert a.shape == b.shape == (4, N_CAND)
    assert a.dtype == b.dtype


def test_output_layer_is_zero_init_so_a_ranking_is_LEARNED():
    """Mirrors GoalDistanceScorer's zero-init discipline: the control starts
    FLAT over the fan, so any ranking it acquires is visible as something it
    learned rather than something its initialisation handed it."""
    mlp = MLPCandidateScorer(D_GOAL, N_CAND, hidden=HIDDEN)
    s = mlp(torch.randn(5, N_CAND, 60, 2), torch.randn(5, D_GOAL))["score"]
    assert torch.equal(s, torch.zeros_like(s))
    # ...and it is still trainable: the output layer receives gradient at step 0
    s.sum().backward()
    assert mlp.fc2.weight.grad is not None
    assert float(mlp.fc2.weight.grad.abs().sum()) > 0.0


def test_it_rejects_a_malformed_fan_like_the_goal_scorer_does():
    mlp = MLPCandidateScorer(D_GOAL, N_CAND, hidden=8)
    with pytest.raises(ValueError, match=r"\[B, N, T, 2\]"):
        mlp(torch.randn(2, N_CAND, 60, 3), torch.randn(2, D_GOAL))
    with pytest.raises(ValueError, match=r"\[B, N, T, 2\]"):
        mlp(torch.randn(2, N_CAND, 60), torch.randn(2, D_GOAL))


def test_hidden_width_is_honoured():
    for h in (8, 64, 512):
        m = MLPCandidateScorer(D_GOAL, N_CAND, hidden=h)
        assert m.fc1.out_features == h
        assert _n(m) == (2 + D_GOAL) * h + h + h + 1 + N_CAND
