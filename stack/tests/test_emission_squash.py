"""The emission's bounding function — why ``tanh`` is the legacy default and
``_squash`` is what v6 uses.

MEASURED 2026-08-15 (float32, this box). The programme already knew ``tanh`` was a
trap: ``kinematic._squash``'s docstring cites ``1 - tanh(51)**2 == 0.0``. What this
module pins is that **the cliff is 5x closer than that example implies** — the
gradient is exactly zero from ``raw >= 10``, an ordinary pre-activation — and that
v6's emission no longer sits on it. v6's own S-W run logged two gradient-spike
episodes (peak ``gnorm 354 076``); that is precisely the regime that pushes a
pre-activation past 10 and leaves the head unable to learn back.

The legacy default is kept because every banked v5.8f number was measured under it
(registry §1.14). A silent change there would re-baseline published rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from tanitad.models.kinematic import _squash        # noqa: E402
from train_v58f_unicycle_head import UnicycleEmission  # noqa: E402


def test_tanh_gradient_is_EXACTLY_zero_from_raw_10_not_51():
    """The measurement that decides the default. Not 'small' — exactly 0.0."""
    dead = [r for r in range(1, 21)
            if _grad_of_tanh(float(r)) == 0.0]
    assert dead, "tanh gradient never underflowed — rerun; the premise moved"
    assert min(dead) == 10, (
        f"cliff moved to raw={min(dead)}; the docstring and the v6 config note "
        f"both quote 10 and must be updated together")
    # and just below it there is still (barely) gradient — so this is a cliff,
    # not a gradual fade the optimiser could climb back down.
    assert _grad_of_tanh(9.0) > 0.0


def _grad_of_tanh(raw: float) -> float:
    x = torch.tensor([raw], requires_grad=True)
    torch.tanh(x).backward()
    return float(x.grad.detach())


@pytest.mark.parametrize("limit", [4.0, 0.2])       # v6's a_max / kappa_max
def test_squash_is_identity_inside_the_range_and_alive_far_outside(limit):
    """The two properties that ruled out the other three candidates at once."""
    for frac in (0.5, 0.9):
        x = torch.tensor([limit * frac], requires_grad=True)
        y = _squash(x, limit)
        y.backward()
        assert torch.allclose(y.detach(), x.detach()), "not identity inside range"
        assert float(x.grad.detach()) == pytest.approx(1.0)

    x = torch.tensor([limit * 100.0], requires_grad=True)
    y = _squash(x, limit)
    y.backward()
    assert float(y.detach()) < limit, "must saturate below the limit"
    assert float(x.grad.detach()) > 0.0, (
        "gradient died at 100x the limit — that is the dead-head trap _squash exists "
        "to avoid")


def test_softsign_is_NOT_the_answer_either():
    """Guards against 'the fix is softsign', which the record shows it is not:
    plain softsign shrinks controls nowhere near their bound, which is how a
    decode lost the ability to reproduce its own anchor."""
    probe, limit = 0.04, 0.2                        # a real curvature, 20 % of bound
    softsign = probe / (1 + abs(probe) / limit)
    assert abs(softsign - probe) / probe > 0.15, "softsign shrink vanished — recheck"
    assert float(_squash(torch.tensor([probe]), limit)) == pytest.approx(probe)


def test_legacy_default_is_bit_exact_so_banked_v58f_rows_cannot_move():
    """⛔ The default must stay ``tanh``: registry §1.14's numbers were measured
    under it. This is the compatibility rail, not a preference."""
    torch.manual_seed(0)
    em = UnicycleEmission(feat_dim=8, k=4)
    assert em.squash == "tanh"

    feat = torch.randn(2, 3, 8)
    v0 = torch.full((2,), 7.0)
    a, kappa, _ = em(feat, v0)

    raw = em.net(torch.cat(
        [feat, (v0 / 10.0)[:, None, None].expand(2, 3, 1)], dim=-1)
    ).reshape(2, 3, 4, 2)
    assert torch.equal(a, em.a_max * torch.tanh(raw[..., 0]))
    assert torch.equal(kappa, em.kappa_max * torch.tanh(raw[..., 1]))


def test_squash_mode_changes_the_function_but_not_the_state_dict():
    """The whole reason this could be fixed mid-programme: an activation holds no
    parameters, so a strict resume is untouched."""
    torch.manual_seed(0)
    legacy = UnicycleEmission(feat_dim=8, k=4, squash="tanh")
    torch.manual_seed(0)
    fixed = UnicycleEmission(feat_dim=8, k=4, squash="squash")

    assert legacy.state_dict().keys() == fixed.state_dict().keys()
    for k in legacy.state_dict():
        assert legacy.state_dict()[k].shape == fixed.state_dict()[k].shape
    fixed.load_state_dict(legacy.state_dict(), strict=True)   # must not raise

    # ...and it really is a different function once the head leaves zero-init.
    with torch.no_grad():
        for m in (legacy, fixed):
            m.net[-1].weight.normal_(0, 3.0)
    fixed.load_state_dict(legacy.state_dict(), strict=True)
    feat, v0 = torch.randn(2, 3, 8), torch.full((2,), 7.0)
    assert not torch.equal(legacy(feat, v0)[0], fixed(feat, v0)[0])


def test_an_unknown_squash_name_is_refused_not_silently_defaulted():
    with pytest.raises(ValueError, match="squash must be"):
        UnicycleEmission(feat_dim=8, k=4, squash="relu")


def test_v6_config_selects_the_measured_correct_form():
    from tanitad.models.v6 import V6Config
    assert V6Config().emission_squash == "squash", (
        "v6 must not inherit the legacy tanh: its emission is in the `planner` "
        "group, untrained by S-W, so the correct form is free to adopt")
