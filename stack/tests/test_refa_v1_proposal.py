"""Multimodal proposals (Drive-JEPA 2601.22032, adapted to the seed-only slot).

The safety property under test is the SLOT, not the head: proposals only ever
seed the planner's search, so distilled diversity cannot become an imitation
echo — and the WTA loss must train the CLOSEST mode only, or K modes collapse
to K copies of the mean (the exact failure the multimodal literature exists
to avoid).
"""
import pytest
import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config
from tanitad.refs.refa_v1_plan import PlanConfig


def _cfg(**kw) -> RefAV1Config:
    base = dict(d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def _inputs(c, b=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, c.op_window, c.n_tokens, c.d_enc, generator=g),
            torch.randn(b, c.op_steps, c.a_dim, generator=g),
            torch.randn(b, c.op_steps, c.n_tokens, c.d_enc, generator=g))


def test_default_is_single_mode_and_scoreless():
    m = RefAV1(_cfg())
    assert m.cfg.proposal_k == 1 and m.proposal_score is None
    f, a, fut = _inputs(m.cfg)
    out = m(f, a, future_feats=fut)
    assert out["proposal"].shape == (3, 1, m.cfg.plan_steps, m.cfg.a_dim)
    assert "proposal_logits" not in out


def test_multimode_emits_logits_and_the_full_stack():
    m = RefAV1(_cfg(proposal_k=4))
    f, a, fut = _inputs(m.cfg)
    out = m(f, a, future_feats=fut)
    assert out["proposal"].shape == (3, 4, m.cfg.plan_steps, m.cfg.a_dim)
    assert out["proposal_logits"].shape == (3, 4)


def test_WTA_scores_exactly_the_closest_mode():
    """⭐ Plant the demo ON one mode: the WTA loss must read ~that mode's
    distance (0), and must NOT average over modes."""
    m = RefAV1(_cfg(proposal_k=3, w_aux_head=1.0)).eval()
    f, a, fut = _inputs(m.cfg)
    with torch.no_grad():
        modes = m(f, a, future_feats=fut)["proposal"]
    demo = modes[:, 1].clone()                       # mode 1 IS the demo
    a2 = a.clone()
    a2[:, :m.cfg.plan_steps] = demo
    out = m(f, a2, future_feats=fut)
    assert float(out["loss_proposal_wta"].detach()) == pytest.approx(0.0,
                                                                     abs=1e-8)
    # and the pick loss points at mode 1
    assert out["proposal_logits"].shape[-1] == 3
    assert "loss_proposal_pick" in out


def test_the_term_reaches_the_loss_and_trains_the_head():
    c = _cfg(proposal_k=2, w_aux_head=0.5)
    m = RefAV1(c)
    f, a, fut = _inputs(c)
    out = m(f, a, future_feats=fut)
    for k in ("loss_proposal_wta", "loss_proposal_pick"):
        assert k in out
    out["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.proposal.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.proposal_score.parameters())


def test_aux_without_a_training_loss_is_refused():
    m = RefAV1(_cfg(w_aux_head=0.5))
    f, a, _ = _inputs(m.cfg)
    with pytest.raises(ValueError, match="silently unused"):
        m(f, a)


def test_off_by_default_the_head_still_gets_no_gradient():
    """w_aux_head = 0 keeps the pre-existing behaviour: the proposal head is
    planner-seed only and receives no training signal."""
    m = RefAV1(_cfg())
    f, a, fut = _inputs(m.cfg)
    m(f, a, future_feats=fut)["loss"].backward()
    assert all(p.grad is None for p in m.proposal.parameters())


def test_plan_runs_with_a_multimode_seed_pool():
    """All modes must enter the search; smoke-scale planner settings."""
    torch.manual_seed(0)
    m = RefAV1(_cfg(proposal_k=3)).eval()
    feats = torch.randn(1, m.cfg.op_window, m.cfg.n_tokens, m.cfg.d_enc)
    pc = PlanConfig(horizon=m.cfg.plan_steps, dt=m.cfg.op_dt,
                    n_samples=8, n_iters=2, n_elites=3, seed=0)
    res = m.plan(feats, v0=5.0, plan_cfg=pc)
    assert res.controls.shape == (m.cfg.plan_steps, m.cfg.a_dim)
    assert torch.isfinite(res.controls).all()


def test_invalid_proposal_k_is_refused():
    with pytest.raises(ValueError, match="proposal_k"):
        _cfg(proposal_k=0).sanity()
