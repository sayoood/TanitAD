"""The ego-history graft must not be a DEAD PRODUCT — one zero-init gate, never two.

MEASURED 2026-09-17 on the real `RefCModel`: `EgoHistoryConfig.zero_init_out` and
the decoder's `ego_to_cond` were **both** zero-initialised — by two authors, for
the same correct reason (*"the condition is UNCHANGED at step 0"*). Stacked they
multiply, and **3,072 parameters could never leave zero**:

    cond += ego_to_cond(ego_hist)
    d/d(ego_to_cond.W) ∝ ego_hist      = 0
    d/d(out.W)         ∝ ego_to_cond.W = 0

Only `ego_to_cond.bias` learned — a constant carrying no ego information at all.
⛔ The PI asked for ego history **by name**; it was wired, stamped, and reaching
nothing. This is the `tac_goal_tok_head` class (11,286 params, `grad_abs_sum`
exactly 0 for all 40,284 steps).

⭐ The gate ALONE is what gives bit-identity — it is the outermost factor — so the
encoder's own zero-init was redundant AND fatal.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import tanitad.refs.refc as refc
from tanitad.models.ego_history import EgoHistoryConfig, EgoHistoryEncoder


def _model():
    cfg = refc.refc_smoke_config()
    cfg.ego_history = EgoHistoryConfig(enable=True, steps=int(cfg.window))
    return refc.RefCModel(cfg), cfg


def _seam(zero_init_out: bool, steps: int = 3):
    """The seam in isolation: ``cond += gate(enc(x))``. The gate is ALWAYS
    zero-init — that is the bit-identity mechanism and it never varies. Only the
    ENCODER's own init changes, which is the whole question."""
    torch.manual_seed(0)
    cfg = EgoHistoryConfig(enable=True, steps=8, zero_init_out=zero_init_out)
    enc = EgoHistoryEncoder(cfg)
    gate = nn.Linear(cfg.out_dim, 16)
    nn.init.zeros_(gate.weight)
    nn.init.zeros_(gate.bias)
    opt = torch.optim.SGD(list(enc.parameters()) + list(gate.parameters()), lr=0.1)
    torch.manual_seed(1)
    x = torch.randn(4, cfg.steps, cfg.channels)
    y = torch.randn(4, 16)
    rows, first_contrib = [], None
    for _ in range(steps):
        opt.zero_grad()
        contrib = gate(enc(x))
        if first_contrib is None:
            first_contrib = float(contrib.detach().abs().max())
        (contrib - y).pow(2).mean().backward()
        rows.append((float(gate.weight.grad.abs().sum()),
                     float(enc.out.weight.grad.abs().sum())))
        opt.step()
    return first_contrib, rows


def test_the_built_model_is_not_a_dead_product():
    """⛔ The regression itself, on the REAL model rather than a fixture."""
    m, _ = _model()
    enc_w = m.ego_hist.out.weight
    gate_w = m.decoder.ego_to_cond.weight
    assert not (bool((enc_w == 0).all()) and bool((gate_w == 0).all())), (
        f"BOTH the encoder output ({enc_w.numel()} params) and the decoder gate "
        f"({gate_w.numel()} params) are all-zero — the graft is a dead product "
        f"and neither can ever move.")


def test_the_ONE_gate_is_kept_so_step_0_is_still_bit_identical():
    """⭐ The fix must not cost what the zero-init was FOR. The gate stays zero,
    so the condition is unchanged at step 0 and ego history remains a removable
    graft."""
    m, _ = _model()
    assert bool((m.decoder.ego_to_cond.weight == 0).all())
    assert bool((m.decoder.ego_to_cond.bias == 0).all())
    contrib, _ = _seam(zero_init_out=False)
    assert contrib == 0.0, f"the graft perturbs step 0 by {contrib}"


def test_the_config_records_what_the_model_DID_not_what_was_asked():
    """⛔ `config.json` must not say `zero_init_out: true` for a model built with
    it false — that is the stamped-but-untrue class."""
    _, cfg = _model()
    assert cfg.ego_history.zero_init_out is False
    assert cfg.ego_history.as_dict()["zero_init_out"] is False


def test_MUT_the_double_zero_init_FREEZES_BOTH_FOREVER():
    """MUTATION: restore the defect and show it is not merely slow — it is
    permanent. Three optimiser steps, every gradient EXACTLY zero."""
    contrib, rows = _seam(zero_init_out=True)
    assert contrib == 0.0
    for i, (g_gate, g_enc) in enumerate(rows):
        assert g_gate == 0.0, f"step {i}: gate grad {g_gate} — expected exactly 0"
        assert g_enc == 0.0, f"step {i}: enc grad {g_enc} — expected exactly 0"


def test_the_fixed_seam_UNLOCKS_the_chain_from_step_1():
    """⭐ The discriminating half. The gate moves immediately (its gradient is
    proportional to a now-non-zero `ego_hist`), and the ENCODER unlocks one step
    later, once the gate has left zero. A check that only asserted 'not both
    zero' would pass on a graft that still never trains."""
    _, rows = _seam(zero_init_out=False)
    assert rows[0][0] > 0.0, "the gate never moved"
    assert rows[0][1] == 0.0, "the encoder cannot have gradient while the gate is 0"
    assert rows[1][1] > 0.0, "the encoder never unlocked"
    assert rows[2][1] > rows[1][1], "the encoder's gradient is not growing"
