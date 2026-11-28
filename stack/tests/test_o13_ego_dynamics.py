"""O13-EGO — pins the properties that make it a valid instrument.

⭐ THE POINT OF EACH TEST IS A CONTROL THAT MUST READ A KNOWN VALUE. This campaign
logged ELEVEN instrument defects (C151-C161) that only controls caught, including
NINE auto-verdicts that fired on the wrong quantity. O13's floor is arithmetic —
EXACTLY 1.0 — so it can be asserted rather than estimated, which is the strongest
form of control available.

⛔ The load-bearing test is `test_the_action_cannot_reach_the_readout`: E-DEC-51
measured that a head given BOTH the latent and the action learns to read the two
action scalars and ignore the 2048-d latent (latent adds -0.0065, t -0.06). O13's
entire design is that the readout never sees the action, and a refactor that
"helpfully" passes it in would silently restore O11's degeneracy while every
number still looked healthy.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# same preamble as the sibling loss tests: the trainer is a SCRIPT, not a
# package module, so `scripts/` has to be on the path explicitly.
_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from train_v6_staged import o13_ego_dynamics_loss as o13  # noqa: E402


def _targets(b: int = 64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, generator=g), torch.randn(b, generator=g) * 0.1)


def test_a_zero_prediction_scores_EXACTLY_the_no_information_floor():
    """⭐ The floor is ARITHMETIC, not estimated: standardised targets against a
    zero prediction give mean(y**2) == 1.0 exactly."""
    dv, dy = _targets()
    z = torch.zeros(len(dv), 128)
    loss, log = o13(z, dv, dy)
    assert log["o13_no_info_floor"] == 1.0
    assert loss.item() == pytest.approx(1.0, abs=1e-6)
    assert log["o13_excess"] == pytest.approx(0.0, abs=1e-6)


def test_any_CONSTANT_prediction_is_at_or_above_the_floor():
    """A constant predictor scores 1 + c**2 — it can never beat the floor, so a
    positive `o13_excess` cannot be manufactured by predicting a constant."""
    dv, dy = _targets()
    for c in (-3.0, -0.5, 0.5, 3.0):
        z = torch.full((len(dv), 64), c)
        loss, log = o13(z, dv, dy)
        assert loss.item() >= 1.0 - 1e-6, f"constant {c} beat the floor"
        assert log["o13_excess"] <= 1e-6


def test_a_latent_that_ENCODES_the_target_beats_the_floor():
    """The term must be winnable — otherwise it is a floor with no signal."""
    dv, dy = _targets()
    d = 128
    g = torch.Generator().manual_seed(1300)
    P = torch.randn(d, 2, generator=g) / math.sqrt(d)
    ys = torch.stack([dv, dy], 1)
    ys = (ys - ys.mean(0, keepdim=True)) / ys.std(0, unbiased=False, keepdim=True)
    # place the standardised target in the readout's own row space
    z = ys @ torch.linalg.pinv(P)
    loss, log = o13(z, dv, dy)
    assert loss.item() < 0.01, f"a latent carrying the target scored {loss.item()}"
    assert log["o13_excess"] > 0.9


def test_the_action_cannot_reach_the_readout():
    """⛔ THE LOAD-BEARING GUARD (E-DEC-51). The readout must be a function of the
    latent ALONE. Asserted structurally on the signature, so a refactor that adds
    an action argument fails here rather than silently reviving O11's echo."""
    import inspect
    params = list(inspect.signature(o13).parameters)
    assert params[0] == "zhat_k"
    for p in params:
        assert "act" not in p.lower(), f"O13 must not accept an action input: {p}"
    # and z_t is diagnostic-only: it must not change the differentiable loss
    dv, dy = _targets()
    z = torch.randn(len(dv), 64)
    a, _ = o13(z, dv, dy)
    b, lg = o13(z, dv, dy, z_t=torch.randn(len(dv), 64))
    assert a.item() == pytest.approx(b.item(), abs=1e-9)
    assert "o13_on_z_t" in lg and "o13_beats_passthrough" in lg


def test_the_shuffled_control_sits_at_the_floor_for_a_fitted_latent():
    """⭐ The per-step positive control. A latent that predicts the TRUE pairing
    must NOT predict a permuted one — otherwise the term is fitting something
    other than the pairing (the C159 failure: a verdict with no working control)."""
    dv, dy = _targets(b=256)
    d = 128
    g = torch.Generator().manual_seed(1300)
    P = torch.randn(d, 2, generator=g) / math.sqrt(d)
    ys = torch.stack([dv, dy], 1)
    ys = (ys - ys.mean(0, keepdim=True)) / ys.std(0, unbiased=False, keepdim=True)
    z = ys @ torch.linalg.pinv(P)
    loss, log = o13(z, dv, dy)
    assert loss.item() < 0.01
    # the shuffled control must be near the no-information value, not near 0
    assert log["o13_shuffled"] > 1.0, log["o13_shuffled"]


def test_the_frozen_readout_is_deterministic_and_seed_dependent():
    """It has no parameters and no state, so it CANNOT adapt to become easy to
    hit — that is what makes a positive `o13_excess` meaningful."""
    dv, dy = _targets()
    z = torch.randn(len(dv), 96)
    a, _ = o13(z, dv, dy, seed=1300)
    b, _ = o13(z, dv, dy, seed=1300)
    c, _ = o13(z, dv, dy, seed=7)
    assert a.item() == pytest.approx(b.item(), abs=1e-12), "readout is not frozen"
    assert abs(a.item() - c.item()) > 1e-9, "a seed change must move the target"


def test_it_is_differentiable_into_the_predicted_latent():
    """The gradient must reach zhat — that is the only path the action has."""
    dv, dy = _targets()
    z = torch.randn(len(dv), 64, requires_grad=True)
    loss, _ = o13(z, dv, dy)
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_it_refuses_a_batch_too_small_to_standardise():
    """A batch of 1 has zero target variance; standardising it would divide by a
    clamp and silently report a meaningless floor."""
    with pytest.raises(ValueError, match="batch >= 2"):
        o13(torch.zeros(1, 32), torch.zeros(1), torch.zeros(1))


def test_the_denominator_is_a_DATA_property_not_an_ARM_property():
    """⛔ C137, reintroduced as C157: a metric normalised by something the ARM
    controls makes arms incomparable. O13 standardises by the BATCH'S TRUE
    targets, so two different latents on the SAME batch share a denominator."""
    dv, dy = _targets()
    z1, z2 = torch.randn(len(dv), 64), torch.randn(len(dv), 64) * 100.0
    _, l1 = o13(z1, dv, dy)
    _, l2 = o13(z2, dv, dy)
    # identical floor regardless of the arm's latent scale
    assert l1["o13_no_info_floor"] == l2["o13_no_info_floor"] == 1.0
