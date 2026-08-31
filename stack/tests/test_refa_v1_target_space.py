"""The collapse minimum in the adapter-space target, and the frozen escape.

⛔ THE FINDING (2026-08-31, PI question "what avoids representation collapse"):
the primary target is ``adapter(std(future))`` and the adapter is TRAINED — so
mapping everything to a constant is a GLOBAL MINIMUM of the primary loss. The
LayerNorms only raise the barrier (their affine re-opens it) and the trainer's
``adapter_std`` monitor only DETECTS the fall. Nothing removed the minimum.

``target_space="frozen"`` removes it for the primary term by predicting the
standardised DINOv3 features themselves (std has frozen, fit-once buffers): the
target's variance is pinned, so a collapsed adapter scores the target variance,
not zero. DINO-WM's own arrangement.

⭐ THE LOAD-BEARING PAIR below does not argue this — it MEASURES it, by
constructing the collapsed adapter and reading both losses.
"""
import pytest
import torch

from tanitad.refs.refa_v1 import RefAV1, RefAV1Config


def _cfg(**kw) -> RefAV1Config:
    base = dict(d_enc=16, d_state=16, n_tokens=8,
                op_dt=0.2, op_steps=30, op_layers=1, op_heads=2, op_window=2,
                tac_dt=0.6, tac_steps=10, tac_queries=4, tac_layers=1,
                str_dt=3.0, str_steps=2, str_dim=8, str_layers=1)
    base.update(kw)
    return RefAV1Config(**base)


def _collapse_the_adapter(m: RefAV1) -> None:
    """The exact failure mode: every trained path into the field zeroed."""
    with torch.no_grad():
        for p in m.adapter.parameters():
            p.zero_()
        # the residual delta heads too, so the rollout carries the constant
        for pred in (m.operative, m.tactical):
            for p in pred.head.parameters():
                p.zero_()


def _loss(m: RefAV1, seed=0) -> float:
    g = torch.Generator().manual_seed(seed)
    c = m.cfg
    f = torch.randn(2, c.op_window, c.n_tokens, c.d_enc, generator=g)
    a = torch.randn(2, c.op_steps, c.a_dim, generator=g)
    fut = torch.randn(2, c.op_steps, c.n_tokens, c.d_enc, generator=g)
    m.std.fit(torch.randn(256, c.d_enc, generator=g))
    return float(m(f, a, future_feats=fut)["loss_feat_op"].detach())


def test_THE_PAIR_adapter_mode_has_the_collapse_minimum_frozen_mode_does_not():
    """⭐⭐ Identical collapsed weights, identical data. Adapter-space primary
    goes to ~zero — the collapse is REWARDED. Frozen-space primary stays at
    the target variance — the collapse is PRICED."""
    torch.manual_seed(0)
    m_a = RefAV1(_cfg(target_space="adapter"))
    _collapse_the_adapter(m_a)
    torch.manual_seed(0)
    m_f = RefAV1(_cfg(target_space="frozen"))
    _collapse_the_adapter(m_f)
    with torch.no_grad():
        for p in m_f.to_enc.parameters():
            p.zero_()                      # even the readout conspires
    la, lf = _loss(m_a), _loss(m_f)
    assert la < 0.01, f"adapter-mode collapsed loss should be ~0, got {la}"
    assert lf > 0.5, f"frozen-mode collapsed loss must stay ~var(target), got {lf}"


def test_frozen_targets_carry_no_trained_parameter():
    """std is buffers only, fit once — asserted, since 'frozen' doing anything
    trainable on the target side would quietly rebuild the minimum."""
    m = RefAV1(_cfg(target_space="frozen"))
    assert all(not p.requires_grad for p in m.std.parameters())
    assert len(list(m.std.parameters())) == 0
    assert len(list(m.std.buffers())) == 3


def test_the_readout_exists_only_in_frozen_mode():
    assert RefAV1(_cfg()).to_enc is None
    assert RefAV1(_cfg(target_space="frozen")).to_enc is not None
    assert "to_enc.weight" not in RefAV1(_cfg()).state_dict()


def test_frozen_mode_trains_end_to_end():
    c = _cfg(target_space="frozen")
    m = RefAV1(c)
    g = torch.Generator().manual_seed(1)
    m.std.fit(torch.randn(256, c.d_enc, generator=g))
    out = m(torch.randn(2, c.op_window, c.n_tokens, c.d_enc, generator=g),
            torch.randn(2, c.op_steps, c.a_dim, generator=g),
            future_feats=torch.randn(2, c.op_steps, c.n_tokens, c.d_enc,
                                     generator=g))
    assert torch.isfinite(out["loss"].detach())
    out["loss"].backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in m.adapter.parameters())


def test_an_unknown_target_space_is_refused():
    with pytest.raises(ValueError, match="target_space"):
        _cfg(target_space="ema").sanity()
