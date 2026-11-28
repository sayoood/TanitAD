"""E15 wiring — the goal-point edge inside ``RefCV3Model``.

⛔ THE POINT OF THIS FILE. A live 44 h training resumes through ``refc_v3.py``, so the
first obligation of an additive edge is that the DEFAULT build did not move. The second
is that the edge is BIT-INERT at init but REACHABLE after training — the exact pair
``test_T6_the_ego_edge_is_bit_inert_at_init`` asserts for E11', because zero-init buys
attributability only if the edge can actually fire once trained.

⚠️ Zero-init does NOT buy that a goal-point build equals a nav build at step 0: the two
have different parameter shapes and every subsequent RNG draw shifts. The arms are
compared by TRAINING them, never by an init identity that is false.
"""
from __future__ import annotations

import pytest
import torch

from tanitad.refs import goal_point as gp
from tanitad.refs import refc
from tanitad.refs import refc_v3 as v3


def _frames(cfg: refc.RefCConfig, b: int = 2, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    h, w = cfg.encoder.image_hw()
    return torch.rand(b, cfg.window, cfg.encoder.in_channels, h, w, generator=g)


def _smoke(inject: bool):
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.goal_point_inject = inject
    cfg.goal_point_cfg = gp.GoalPointConfig(d_goal=8)
    torch.manual_seed(0)
    return cfg, v3.RefCV3Model(cfg).eval()


def test_the_flag_defaults_OFF_so_the_live_build_did_not_move():
    assert v3.RefCV3Config().goal_point_inject is False
    assert v3.refc_v3_hier_config().goal_point_inject is False
    assert v3.refc_v3_flat_config().goal_point_inject is False


def test_v3_parity_when_off_no_gp_parameters_and_no_hook_key():
    """⛔ The parity assertion that matters for a live run: with the flag OFF there is
    no gp_* parameter, no gp_* state_dict key, and the hook emits no goal point."""
    cfg, m = _smoke(False)
    assert m.gp_head is None and m.gp_cond is None and m.gp_slot is None
    assert not [k for k in m.state_dict() if k.startswith("gp_")]
    with torch.no_grad():
        out = m(_frames(cfg.core), v0=torch.tensor([3.0, 7.0]), steps=2)
    assert out["goal_point_injected"] is False
    assert out.get("goal_point") is None
    # same-breath POSITIVE control: the ON build DOES carry them, so a passing OFF
    # assertion is not just "the attribute name is misspelled".
    cfg2, m2 = _smoke(True)
    assert [k for k in m2.state_dict() if k.startswith("gp_")]


def test_the_edge_is_BIT_INERT_at_init():
    """The zero-init projections make the ON build's emission identical to the core's,
    exactly as ``test_hier_emission_bit_inert_at_init`` asserts for the cascade."""
    cfg, m = _smoke(True)
    with torch.no_grad():
        out = m(_frames(cfg.core), v0=torch.tensor([3.0, 7.0]), steps=2)
    assert torch.equal(out["traj"], out["traj_base"])
    assert torch.equal(out["sel_idx"], out["sel_idx_base"])
    assert out["goal_point_injected"] is True
    assert out["goal_point"].shape == (2, 2)


def test_the_edge_is_REACHABLE_once_the_projections_are_trained():
    """⛔ MUTATION, and the half a zero-init check cannot give you. An edge that is
    bit-inert at init AND unreachable after training is a dead wire that reads as a
    clean ablation — the failure the ego-edge test was written to catch."""
    cfg, m = _smoke(True)
    f, v0 = _frames(cfg.core), torch.tensor([3.0, 7.0])
    with torch.no_grad():
        before = m(f, v0=v0, steps=2)
        torch.manual_seed(1)
        m.gp_cond.to_tac.weight.normal_(std=0.5)
        m.gp_cond.to_str.weight.normal_(std=0.5)
        after = m(f, v0=v0, steps=2)
    assert not torch.equal(before["z_tac"], after["z_tac"])
    assert not torch.equal(before["g_str"], after["g_str"])


def test_the_goal_point_VALUE_reaches_the_state_not_only_its_presence():
    """⛔⛔ THE E13 FAILURE MODE, tested for directly. refcv4b's nav edge moved g_str by
    0.0103 when its VALUE changed and by 0.1902 when it was REMOVED — 18.6x, i.e. a
    presence-gated bias. Here the goal point is a continuous function of ``ctx``, so the
    only way to interrogate value-sensitivity inside one forward is to perturb the head
    and confirm the downstream state follows. A build where it does not is a dead wire.
    """
    cfg, m = _smoke(True)
    f, v0 = _frames(cfg.core), torch.tensor([3.0, 7.0])
    with torch.no_grad():
        torch.manual_seed(2)
        m.gp_cond.to_tac.weight.normal_(std=0.5)
        m.gp_cond.to_str.weight.normal_(std=0.5)
        a = m(f, v0=v0, steps=2)
        m.gp_head.proj.bias.add_(torch.tensor([0.7, -0.9]))   # move the POINT only
        b = m(f, v0=v0, steps=2)
    assert not torch.equal(a["goal_point"], b["goal_point"])
    assert not torch.equal(a["g_str"], b["g_str"]), (
        "moving the goal POINT must move the strategic goal — if it does not, the edge "
        "is presence-gated exactly like E13 and the arm is not worth training")


def test_the_hook_emits_the_point_for_the_selection_seam():
    """The geometric ranking term lives in `refc.py` (a different owner's file). The
    value it needs is emitted here so that patch is one line and needs no second
    forward — PREREG.md §8 names it as the integration item."""
    cfg, m = _smoke(True)
    with torch.no_grad():
        out = m(_frames(cfg.core), v0=torch.tensor([3.0, 7.0]), steps=2)
    pt = out["goal_point"]
    bank = out["anchor_bank"]
    prior = gp.anchor_goal_prior_at_time(pt, torch.ones(pt.shape[0]), bank,
                                         slot=min(m.gp_slot, bank.shape[2] - 1),
                                         scale_m=cfg.goal_point_cfg.range_norm_m)
    assert prior.shape == bank.shape[:2]
    assert torch.isfinite(prior).all()


def test_a_config_whose_goal_time_is_not_a_model_horizon_is_REFUSED_at_build():
    cfg = v3.refc_v3_smoke_config(hier=True)
    cfg.goal_point_inject = True
    cfg.goal_point_cfg = gp.GoalPointConfig(d_goal=8, t_goal_s=3.7)
    with pytest.raises(ValueError, match="not a model horizon"):
        v3.RefCV3Model(cfg)
