"""O11-CF — the objective that cannot be minimised by ignoring the actions.

⛔ WHY THIS FILE IS NOT OPTIONAL. E-DEC-30 (MEASURED 2026-08-24, 444 windows,
3 arms, positive control passing on all three) found that replacing the ENTIRE
action tensor — a 251 % change to the input — moves the predictor's output by
1.5 %, while a 10 % nudge to the LATENT moves it 17.7 %. `nrmse`, the number the
whole Gate-B/Gate-C census ranks arms on, is unchanged to four decimals under
that shuffle. O11-CF exists to make that failure mode impossible to hide.

⭐ THE PROPERTY EVERY TEST BELOW EXISTS TO PIN: **an action-independent predictor
scores EXACTLY ln(1 + n_neg)**, and nothing it can do improves on that. This is
the C149 lesson built into a loss rather than bolted onto a panel — a control
that must read a KNOWN value exactly, not merely "look reasonable". C149 happened
because a metric had a permutation null but no constant-predictor floor; here the
floor is a closed-form constant, so a defect in the term shows up as a number
that is not ln(1+n).
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

from train_v6_staged import o11_counterfactual_action_loss  # noqa: E402


def _mk(b=8, s=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, s, generator=g)


def test_an_action_blind_predictor_scores_EXACTLY_the_no_information_floor():
    """The load-bearing property. If ẑ does not depend on the action, every
    rollout is the same tensor, every logit is equal, and the softmax is uniform
    — so the loss is ln(1+n) to floating-point exactness, not approximately."""
    z_true = _mk(seed=1)
    blind = _mk(seed=2)                      # ONE prediction, reused for all
    for n_neg in (1, 2, 3, 7):
        loss, log = o11_counterfactual_action_loss(
            blind, [blind.clone() for _ in range(n_neg)], z_true)
        assert float(loss) == pytest.approx(math.log(1 + n_neg), abs=1e-6)
        assert log["o11_no_info_floor"] == pytest.approx(math.log(1 + n_neg),
                                                         abs=1e-4)
        # ⛔ the READING must agree with the number: excess is 0 at the floor
        assert log["o11_excess"] == pytest.approx(0.0, abs=1e-5)
        assert log["o11_sep_abs"] == pytest.approx(0.0, abs=1e-5)
        # ⚠️ 1e-4, not 1e-6: `o11_chance_acc` is ROUNDED to 4 dp for the log,
        # so a tighter tolerance tests the rounding, not the property.
        assert log["o11_pick_acc"] == pytest.approx(log["o11_chance_acc"],
                                                    abs=1e-4)


def test_the_floor_MOVES_with_n_neg_so_losses_are_not_comparable_across_it():
    """⚠️ A guard against the C149 shape recurring one level up: `o11_loss` at
    n_neg=1 and n_neg=7 are not on the same scale, so an arm sweep that varies
    n_neg and compares raw loss would rank arms by their FLOOR. `o11_excess` is
    the comparable quantity and this pins that it is."""
    z_true, blind = _mk(seed=3), _mk(seed=4)
    l1 = float(o11_counterfactual_action_loss(blind, [blind.clone()], z_true)[0])
    l7 = float(o11_counterfactual_action_loss(
        blind, [blind.clone() for _ in range(7)], z_true)[0])
    assert l7 > l1                                    # purely a floor effect
    assert l1 == pytest.approx(math.log(2), abs=1e-6)
    assert l7 == pytest.approx(math.log(8), abs=1e-6)


def test_an_action_SENSITIVE_predictor_beats_the_floor():
    """The other direction: when the true-action rollout is genuinely closer to
    the observed future than the counterfactual ones, the loss falls BELOW the
    floor and `o11_excess` is positive. Without this the test above would also
    pass for a term that is constant."""
    z_true = _mk(seed=5)
    good = z_true + 0.05 * _mk(seed=6)        # close to the truth
    bad = [z_true + 1.50 * _mk(seed=7 + i) for i in range(3)]
    loss, log = o11_counterfactual_action_loss(good, bad, z_true)
    assert float(loss) < math.log(4)
    assert log["o11_excess"] > 0.0
    assert log["o11_sep_abs"] > 0.0
    assert log["o11_pick_acc"] > log["o11_chance_acc"]


def test_it_is_differentiable_into_the_prediction():
    """The term has to be able to CHANGE the predictor, not merely report on it."""
    z_true = _mk(seed=8)
    pos = (_mk(seed=9)).requires_grad_(True)
    loss, _ = o11_counterfactual_action_loss(pos, [_mk(seed=10)], z_true)
    loss.backward()
    assert pos.grad is not None
    assert float(pos.grad.abs().sum()) > 0.0


def test_it_refuses_an_empty_counterfactual_set():
    """A silent no-op here would look exactly like action-blindness."""
    with pytest.raises(ValueError, match="at least one counterfactual"):
        o11_counterfactual_action_loss(_mk(), [], _mk())


def test_a_cyclic_shift_is_a_derangement_but_randperm_is_not():
    """⛔ THE CALL SITE'S CHOICE, PINNED HERE BECAUSE ITS FAILURE IS SILENT.
    The negatives are built with `torch.roll(fa3, off)` for a non-zero `off`,
    which fixes NO element. `randperm` fixes an element with probability ~1/B,
    and a fixed point makes that row's 'counterfactual' the TRUE action —
    dragging the loss toward the floor and reading as action-blindness that is
    not there. This test documents why the call site is not written the obvious
    way."""
    b = 8
    idx = torch.arange(b)
    for off in range(1, b):
        assert int((torch.roll(idx, shifts=off, dims=0) == idx).sum()) == 0
    g = torch.Generator().manual_seed(0)
    fixed = sum(int((torch.randperm(b, generator=g) == idx).sum())
                for _ in range(200))
    assert fixed > 0, "randperm produced no fixed point in 200 draws — the "
    "premise of the call site's comment would be wrong"
