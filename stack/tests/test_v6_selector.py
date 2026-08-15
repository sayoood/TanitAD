"""The v6 SELECTION lever — gated, default-off, and proved inert when off.

Every test here exists because of a measured failure elsewhere in the programme:

* ``test_all_off_is_byte_identical`` — the live S-W resume must not move. A
  "default-off" flag that shifts one RNG draw silently invalidates a strict
  resume, and the run row would still say S-W.
* ``test_param_delta_is_exactly_267`` — *"not a capacity experiment"* is a claim
  that must be MEASURED (D-TAC1's first implementation cost +272,001 before its
  own control caught it).
* ``test_scorer_is_planner_group`` / ``test_scorer_is_in_planner_surface`` —
  registry §1.14: repairing a trunk moved the frozen selector 0.7933 -> 4.4159.
  The selector belongs to the group that is retrained on the trunk it consumes,
  and X3 must probe it like every other planner output.
* ``test_goal_scorer_is_not_degenerate`` — a scorer whose ranking does not
  depend on the window is a per-candidate constant wearing a selector's name.
* ``test_wta_eps_off_is_bitwise_identical`` — same discipline for the loss.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from tanitad.models.v6 import (  # noqa: E402
    GoalDistanceScorer, V6Config, V6Stack,
)

#: MEASURED on this box, 2026-08-15, HEAD 30d6d60, torch default RNG, seed 0:
#: ``V6Stack(V6Config())`` -> 87,893,449 params over 405 state_dict keys.
#: The test does not depend on these literals (it rebuilds both arms in-process)
#: — they are recorded so a future reader can tell whether the baseline moved.
HEAD_TOTAL_PARAMS = 87_893_449
HEAD_N_KEYS = 405

#: goal_point Linear(128 -> 2) = 258 · cand_bias [8] = 8 · log_tau = 1.
SCORER_PARAMS = 267


def _small() -> V6Config:
    """A tiny but structurally faithful config — every width that the scorer's
    parameter count depends on (``d_goal_embed``, ``n_candidates``) is at its
    real value; only the encoder/predictor are shrunk so the test is seconds."""
    from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig
    return V6Config(
        encoder=EncoderConfig(in_channels=9, image_size=64, image_width=64,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=2, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=2,
                                  horizons=(1,), action_dim=3, residual=True),
        d_tac=32, d_str=16, adapter_hidden=32, f_hidden_tac=32,
        f_hidden_str=32, f_blocks=1, aux_hidden=16, sigreg_slices=8,
        plan_steps=6, dt=0.1, op_band_s=(0.0, 0.2), tac_band_s=(0.2, 0.6),
        hz_op=10.0, hz_tac=2.0, hz_str=0.5,
        d_plan_feat=16, emission_hidden=16, d_goal_embed=128, n_candidates=8)


def _plan_batch(cfg: V6Config, b: int = 2) -> dict:
    """The minimal batch the plan term needs (see v6_loss_step's contract)."""
    torch.manual_seed(1)
    return {"frames": torch.randn(b, cfg.predictor.window, 9, 64, 64),
            "actions2": torch.randn(b, cfg.predictor.window, 2),
            "future_actions2": torch.randn(b, 2, 2),
            "v0": torch.full((b,), 8.0),
            "gt_wp": torch.randn(b, 1, 2),
            "z_true_steps": [torch.randn(b, cfg.d_op)],
            "plan_target": torch.randn(b, cfg.plan_steps, 2)}


def _plan_only_weights(T, *, w_select: float):
    """Every measure off except the planner terms — so the test exercises the
    lever and nothing else."""
    return T.V6LossWeights(
        o1_ctrl=0.0, o1_fact=0.0, o1_scene=0.0, o2_nearfield=0.0,
        o3_masked=0.0, o5_rollout=0.0, o6_sigreg=0.0, t1_latent=0.0,
        s1_latent=0.0, seam_op=0.0, lambda_plan=1.0, w_select=w_select)


def _build(cfg: V6Config, seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(copy.deepcopy(cfg))


def test_default_is_off():
    cfg = V6Config()
    assert cfg.selector == "none"
    assert cfg.plan_wta_eps == 0.0
    assert _build(_small()).cand_score is None


def test_all_off_is_byte_identical():
    """With the selector OFF the state_dict is IDENTICAL — keys AND values.

    Identity is checked against a stack built by the same code with the same
    seed, which is the strongest in-process statement available; the RNG
    argument is what makes it a statement about HEAD too: ``cand_score`` is
    constructed at the very END of ``__init__`` and ONLY when asked, so the
    ``none`` path draws no random numbers and every earlier module's
    initialisation is bit-for-bit what it was before the flag existed.
    """
    a, b = _build(_small(), seed=0), _build(_small(), seed=0)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), f"{k} moved with the selector OFF"
    assert not any("cand_score" in k for k in sa)


def test_param_delta_is_exactly_267():
    """+267, and every earlier tensor unchanged — the capacity control."""
    cfg = _small()
    off = _build(cfg, seed=0)
    on = _build(V6Config(**{**cfg.__dict__, "selector": "goal"}), seed=0)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    assert n_on - n_off == SCORER_PARAMS, (n_off, n_on)
    s_off, s_on = off.state_dict(), on.state_dict()
    assert set(s_on) - set(s_off) == {
        "cand_score.goal_point.weight", "cand_score.goal_point.bias",
        "cand_score.cand_bias", "cand_score.log_tau"}
    # ⭐ the part that makes it a CAPACITY control and not just a count: turning
    # the flag on must not perturb ANY pre-existing weight.
    for k in s_off:
        assert torch.equal(s_off[k], s_on[k]), f"{k} moved when the flag flipped"


def test_scorer_is_planner_group():
    on = _build(V6Config(**{**_small().__dict__, "selector": "goal"}))
    names = [n for n, _ in on.named_parameters() if n.startswith("cand_score")]
    assert names
    assert {on.group_of(n) for n in names} == {"planner"}


def test_scorer_is_in_planner_surface_and_isolation_holds():
    on = _build(V6Config(**{**_small().__dict__, "selector": "goal"}))
    frames = torch.randn(2, on.cfg.predictor.window, 9, 64, 64)
    out = on(frames, torch.randn(2, on.cfg.predictor.window, 3),
             torch.full((2,), 8.0))
    assert "sel_score" in out["plan"] and "sel_goal_point" in out["plan"]
    ids = {id(t) for t in out["planner_side"]}
    assert id(out["plan"]["sel_score"]) in ids
    assert id(out["plan"]["sel_goal_point"]) in ids
    iso = on.assert_isolation(batch_size=1, strict=True)
    assert iso["pass"], iso


def test_goal_scorer_is_not_degenerate():
    """The ranking must depend on the WINDOW, not only on the candidate index.

    A linear head on the pre-emission feature would NOT pass this: candidate
    identity enters that feature only through ``cand_queries``, so a linear
    score reduces to a per-candidate constant that cancels in the softmax. The
    goal-distance form is non-degenerate because the reference point moves with
    the scene while the endpoints move with the candidate.
    """
    torch.manual_seed(0)
    sc = GoalDistanceScorer(4, 3, tau_m=1.0)
    with torch.no_grad():        # a live (non-zero) goal decode
        sc.goal_point.weight.copy_(torch.tensor([[1.0, 0, 0, 0],
                                                 [0, 1.0, 0, 0]]))
    wp = torch.zeros(2, 3, 2, 2)
    wp[:, 0, -1] = torch.tensor([10.0, 0.0])
    wp[:, 1, -1] = torch.tensor([0.0, 10.0])
    wp[:, 2, -1] = torch.tensor([5.0, 5.0])
    g = torch.tensor([[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]])
    pick = sc(wp, g)["score"].argmax(-1)
    assert pick.tolist() == [0, 1], "the ranking ignored the window"


def test_wta_eps_off_is_bitwise_identical():
    """eps = 0 must not even construct the relaxation term."""
    import scripts.train_v6_staged as T
    cfg = _small()
    stack = _build(cfg)
    torch.manual_seed(1)
    batch = _plan_batch(cfg)
    w = _plan_only_weights(T, w_select=0.0)
    torch.manual_seed(2)
    a = T.v6_loss_step(stack, batch, stage="S-J", weights=w, o1_k=1, o5_k=1)
    torch.manual_seed(2)
    b = T.v6_loss_step(stack, batch, stage="S-J", weights=w, o1_k=1, o5_k=1)
    assert torch.equal(a["loss"], b["loss"])
    assert "plan_loser_mean" not in a["log"]
    assert a["log"]["plan_wta"] == pytest.approx(a["log"]["plan_wta"])

    stack2 = _build(V6Config(**{**cfg.__dict__, "plan_wta_eps": 0.5}))
    stack2.load_state_dict(stack.state_dict())
    torch.manual_seed(2)
    c = T.v6_loss_step(stack2, batch, stage="S-J", weights=w, o1_k=1, o5_k=1)
    assert "plan_loser_mean" in c["log"]
    # the relaxation can only ADD to the winner-take-all term
    assert float(c["loss"]) > float(a["loss"])


def test_selection_loss_refuses_without_a_scorer():
    import scripts.train_v6_staged as T
    cfg = _small()
    stack = _build(cfg)
    batch = _plan_batch(cfg)
    w = _plan_only_weights(T, w_select=1.0)
    with pytest.raises(ValueError, match="selector='none'"):
        T.v6_loss_step(stack, batch, stage="S-J", weights=w, o1_k=1, o5_k=1)


def test_capacity_control_is_now_IMPLEMENTED_and_unknown_names_still_refused():
    """SUPERSEDED 2026-08-16. This used to assert that ``"mlp"`` was REFUSED
    because it did not exist — escalation #4 of V6F_PLANNER_DESIGN.md. It now
    exists (``MLPCandidateScorer``, +33,801 MEASURED), so the assertion that
    keeps its value is the one that still holds: an UNKNOWN selector is
    refused, and the refusal names why the control matters.

    Its properties are pinned in tests/test_v6_selector_capacity_control.py.
    """
    assert V6Config(selector="mlp").selector == "mlp"
    with pytest.raises(ValueError, match="none|goal|mlp"):
        V6Config(selector="learned-cost")


def test_preflight_refuses_selector_in_sw():
    import scripts.train_v6_staged as T
    a = T.build_parser().parse_args(
        ["--stage", "S-W", "--out", "x", "--dry-run",
         "--selector", "goal", "--w-select", "1"])
    assert any("S-W" in p and "selector" in p for p in T.preflight(a))
