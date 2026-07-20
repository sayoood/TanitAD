"""Gated-intent lever (v2 lever 7, H26 fix) — CPU, synthetic.

The 4-brain OperativePredictor additively merges the tactical intent into the
action FiLM conditioning. The H26 hierarchy panel measured the ungated intent
term (``intent_proj(intent)`` norm ~31.4 at flagship-19k) COMPETING WITH the
action embedding (``act_emb`` norm ~28.3) — diluting the action conditioning
rather than gently biasing it, and engaging intent was net-harmful to the
operative. ``gated_intent`` adds a ReZero-style learnable scalar gate (init 0.1)
so the operative starts action-dominant and the intent grows only if earned.

Pins:
  (a) gated_intent=False is byte-identical to the pre-lever model — NO
      intent_gate Parameter, and the intent path reproduces the ungated
      (implicit-gate-1.0) formula EXACTLY;
  (b) gated_intent=True adds one intent_gate Parameter (requires_grad, init 0.1)
      and scales the intent contribution ~0.1x vs the ungated path;
  (c) the predictor forward runs end to end both ways;
  (d) cfg.v2_gated_intent threads through the WorldModel to the predictor,
      adding exactly the one extra state_dict key (a v1 ckpt still loads).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from tanitad.config import PredictorConfig, flagship4b_smoke_config
from tanitad.models.fourbrain import WorldModel
from tanitad.models.predictor import OperativePredictor

_CFG = dict(d_model=32, depth=2, n_heads=2, window=4, horizons=(1, 2),
            action_dim=2)
_S, _I, _B = 40, 16, 5                          # state_dim, intent_dim, batch


def _predictor(gated: bool, intent_dim: int | None = _I):
    return OperativePredictor(PredictorConfig(**_CFG), _S,
                              intent_dim=intent_dim, gated_intent=gated)


def _inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    states = torch.randn(_B, _CFG["window"], _S, generator=g)
    actions = torch.randn(_B, _CFG["window"], 2, generator=g)
    intent = torch.randn(_B, _I, generator=g)
    return states, actions, intent


def _make_film_live(pred):
    # Zero-init FiLM => intent has no effect (identity start); make it live so
    # the intent term actually reaches the output (mirrors test_flagship4b).
    for blk in pred.blocks:
        nn.init.normal_(blk.film.to_scale_shift.weight, std=0.1)


# --------------------------------------------------------------------------- #
# (a) default-off == byte-identical pre-lever behaviour                        #
# --------------------------------------------------------------------------- #
def test_gated_off_has_no_gate_param():
    torch.manual_seed(0)
    pred = _predictor(gated=False)
    assert pred.intent_gate is None
    assert "intent_gate" not in dict(pred.named_parameters())
    assert not any(k.endswith("intent_gate") for k in pred.state_dict())
    # gated_intent=True with intent_dim=None still builds no gate (guarded) —
    # a base model stays parameter-free even if the flag is flipped.
    base = _predictor(gated=True, intent_dim=None)
    assert base.intent_gate is None and base.intent_proj is None


def test_gated_off_reproduces_pre_change_formula_exactly():
    torch.manual_seed(0)
    ung, gat = _predictor(gated=False), _predictor(gated=True)
    _make_film_live(ung)
    # Copy ALL shared weights ung -> gat (the gate is not in ung's dict; it stays
    # at its own init). strict=False reports the gate as the only missing key.
    missing, unexpected = gat.load_state_dict(ung.state_dict(), strict=False)
    assert missing == ["intent_gate"] and unexpected == []
    gat.intent_gate.data.fill_(1.0)             # implicit-gate-1.0 == pre-change
    states, actions, intent = _inputs()
    with torch.no_grad():
        o_ungated = ung(states, actions, intent=intent)[1]   # ungated code path
        o_gate1 = gat(states, actions, intent=intent)[1]     # gate forced to 1.0
    assert torch.equal(o_ungated, o_gate1)      # the multiply is a no-op at 1.0


# --------------------------------------------------------------------------- #
# (b) gate present + scales the intent term                                    #
# --------------------------------------------------------------------------- #
def test_gated_on_has_gate_param_init_tenth():
    torch.manual_seed(0)
    g = _predictor(gated=True).intent_gate
    assert isinstance(g, nn.Parameter) and g.requires_grad
    assert g.shape == () and float(g.detach()) == pytest.approx(0.1, abs=1e-6)


def test_gate_scales_intent_contribution_by_a_tenth():
    torch.manual_seed(0)
    ung, gat = _predictor(gated=False), _predictor(gated=True)
    _make_film_live(ung)
    gat.load_state_dict(ung.state_dict(), strict=False)   # identical shared wts
    states, actions, intent = _inputs()
    with torch.no_grad():
        o_free = gat(states, actions, intent=None)[1]          # intent-free base
        o_ungated = ung(states, actions, intent=intent)[1]     # implicit gate 1
        gat.intent_gate.data.fill_(0.1)
        o_gated = gat(states, actions, intent=intent)[1]       # gate 0.1
    d_gated = (o_gated - o_free).norm()
    d_ungated = (o_ungated - o_free).norm()
    assert float(d_ungated) > 1e-4                             # intent reaches out
    ratio = float(d_gated / d_ungated)          # ~0.1 to first order (measured .1006)
    assert 0.08 < ratio < 0.12, f"gate did not scale the intent ~0.1x: {ratio:.4f}"


# --------------------------------------------------------------------------- #
# (c) forward runs end to end both ways                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gated", [False, True])
def test_predictor_forward_end_to_end(gated):
    torch.manual_seed(0)
    pred = _predictor(gated=gated)
    states, actions, intent = _inputs()
    out = pred(states, actions, intent=intent)
    for k in _CFG["horizons"]:
        assert out[k].shape == (_B, _S) and torch.isfinite(out[k]).all()


# --------------------------------------------------------------------------- #
# (d) cfg.v2_gated_intent threads through the WorldModel                       #
# --------------------------------------------------------------------------- #
def test_worldmodel_threads_gated_intent_flag():
    torch.manual_seed(0)
    smk = flagship4b_smoke_config()             # full 4-brain: intent_dim is set
    m_off = WorldModel(smk)
    m_on = WorldModel(dataclasses.replace(smk, v2_gated_intent=True))
    assert m_off.predictor.intent_gate is None
    assert isinstance(m_on.predictor.intent_gate, nn.Parameter)
    # exactly one extra state_dict key -> a v1 (gate-off) ckpt still loads a
    # v2-off model with no missing keys (default-off convention preserved).
    off_keys, on_keys = set(m_off.state_dict()), set(m_on.state_dict())
    assert on_keys - off_keys == {"predictor.intent_gate"}
    assert off_keys - on_keys == set()
    # both run the intent-conditioned operative forward end to end
    B, W = 2, smk.predictor.window
    with torch.no_grad():
        st = m_on.encode_window(torch.rand(B, W, 1, 64, 64))
        z = m_on.imagine(st, torch.randn(B, W, 2),
                         intent=torch.randn(B, smk.tactical_policy.d_intent))[1]
    assert z.shape == (B, m_on.state_dim) and torch.isfinite(z).all()
