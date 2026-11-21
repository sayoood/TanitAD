"""R6 (PI 2026-09-27): "keep the strategic layer off in the first experiments".

MEASURED at tip c36b6ddd (v7f_r1r4/AUDIT_R4.md s5): with S-S never trained, the tactical goal heads still read
``cond=e_g_str`` -- the UNTRAINED strategic head's embedding of the scene -- and the plan loss reaches 13
``layer_str`` tensors through it. ``--strategic-off`` hands them a ZERO conditioning instead.
Every expectation is a known value; the regression arm reproduces the tip behaviour.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from tanitad.config import EncoderConfig, PredictorConfig, ReadoutConfig  # noqa: E402
from tanitad.models.v6 import V6Config, V6Stack  # noqa: E402


def _cfg(**kw) -> V6Config:
    base = dict(
        encoder=EncoderConfig(in_channels=3, image_size=32, image_width=32,
                              patch_size=16, d_model=32, depth=1, n_heads=2),
        readout=ReadoutConfig(grid=4, d_readout=8),
        predictor=PredictorConfig(d_model=32, depth=1, n_heads=2, window=4,
                                  horizons=(1,), action_dim=3),
        d_tac=32, d_str=16, d_goal_embed=16, adapter_hidden=32,
        f_hidden_tac=32, f_hidden_str=32, d_plan_feat=16, emission_hidden=16,
        n_candidates=4, aux_hidden=16, sigreg_slices=8)
    base.update(kw)
    return V6Config(**base)


def _build(cfg: V6Config, seed: int = 0) -> V6Stack:
    torch.manual_seed(seed)
    return V6Stack(copy.deepcopy(cfg))


def _forward(s: V6Stack, seed: int = 0) -> dict:
    b = s.synthetic_batch(batch=2, seed=seed)
    return s(**{k: v for k, v in b.items()})


def _perturb_layer_str(s: V6Stack) -> int:
    n = 0
    with torch.no_grad():
        for name, p in s.named_parameters():
            if s.group_of(name) == "layer_str":
                p.add_(torch.randn_like(p) * 0.5)
                n += 1
    return n


def test_default_is_OFF():
    assert V6Config().strategic_off is False


def test_ON_the_tactical_goal_is_INDEPENDENT_of_the_strategic_layer():
    s = _build(_cfg(strategic_off=True))
    s.eval()
    with torch.no_grad():
        before = _forward(s)["g_tac"]["probs"].clone()
        assert _perturb_layer_str(s) > 0
        after = _forward(s)["g_tac"]["probs"]
    assert torch.equal(before, after)


def test_REGRESSION_ARM_OFF_the_untrained_strategic_head_moves_the_tactical_goal():
    """The tip behaviour R6 removes: perturbing ONLY layer_str changes the tactical goal posterior."""
    s = _build(_cfg())
    s.eval()
    with torch.no_grad():
        before = _forward(s)["g_tac"]["probs"].clone()
        _perturb_layer_str(s)
        after = _forward(s)["g_tac"]["probs"]
    assert not torch.equal(before, after)


def test_ON_no_gradient_from_the_tactical_goal_reaches_layer_str():
    s = _build(_cfg(strategic_off=True))
    out = _forward(s)
    loss = out["g_tac"]["probs"].pow(2).sum() + out["g_tac"]["args"].pow(2).sum()
    params = [(n, p) for n, p in s.named_parameters() if s.group_of(n) == "layer_str"]
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    reached = [n for (n, _), g in zip(params, grads) if g is not None and float(g.abs().sum()) > 0.0]
    assert reached == []


def test_REGRESSION_ARM_OFF_the_tactical_goal_DOES_backprop_into_layer_str():
    s = _build(_cfg())
    out = _forward(s)
    loss = out["g_tac"]["probs"].pow(2).sum() + out["g_tac"]["args"].pow(2).sum()
    params = [(n, p) for n, p in s.named_parameters() if s.group_of(n) == "layer_str"]
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    reached = [n for (n, _), g in zip(params, grads) if g is not None and float(g.abs().sum()) > 0.0]
    assert reached, "the tip wiring must reach layer_str, or this regression arm proves nothing"


def test_strategic_off_REFUSES_the_other_strategic_route():
    with pytest.raises(ValueError, match="tac_goal_cond"):
        _cfg(strategic_off=True, tac_goal_cond=True)


def test_OFF_forward_is_bit_identical_to_the_explicit_default():
    a, b = _build(_cfg()), _build(_cfg(strategic_off=False))
    a.eval(); b.eval()
    with torch.no_grad():
        oa, ob = _forward(a), _forward(b)
    assert torch.equal(oa["g_tac"]["probs"], ob["g_tac"]["probs"])
    assert torch.equal(oa["plan"]["waypoints"], ob["plan"]["waypoints"])


def test_the_trainer_exposes_the_flag_and_maps_it():
    import inspect
    import train_v6_staged as T                                    # noqa: E402
    src = inspect.getsource(T)
    assert 'ap.add_argument("--strategic-off", action="store_true"' in src
    assert 'strategic_off=bool(getattr(a, "strategic_off", False))' in src


def test_ON_the_FACTORED_tactical_heads_are_independent_too():
    """--goal-factored builds two more conditioned heads; the zero must reach them as well."""
    s = _build(_cfg(strategic_off=True, goal_factored=True))
    s.eval()
    with torch.no_grad():
        o = _forward(s)
        before = (o["g_tac_lat"]["probs"].clone(), o["g_tac_lon"]["probs"].clone())
        _perturb_layer_str(s)
        o = _forward(s)
    assert torch.equal(before[0], o["g_tac_lat"]["probs"]) and torch.equal(before[1], o["g_tac_lon"]["probs"])


def test_REGRESSION_ARM_OFF_the_factored_heads_follow_the_strategic_layer():
    s = _build(_cfg(goal_factored=True))
    s.eval()
    with torch.no_grad():
        o = _forward(s)
        before = o["g_tac_lat"]["probs"].clone()
        _perturb_layer_str(s)
        after = _forward(s)["g_tac_lat"]["probs"]
    assert not torch.equal(before, after)
