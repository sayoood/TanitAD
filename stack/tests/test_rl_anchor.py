"""The reference-policy anchor — GRPO's trust region, not an imitation loss.

⛔ THE LOAD-BEARING TEST HERE IS `test_anchor_actually_constrains_drift`.
P-RC21's P1 arm ran with no anchor and the deployed trajectory drifted +69.5 %
while the rule-based reward rose. An anchor that is merely *present* fixes
nothing; the test that matters is that turning it up REDUCES movement away from
the reference. That is a behavioural claim, so it gets a behavioural test — the
lesson of TRAIN-C3, where an existence test passed over a wrong gradient.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tanitad.rl import anchor as ANC
from tanitad.rl.config import PostTrainConfig


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 8)


# ---------------------------------------------------------------------------
# The frozen copy
# ---------------------------------------------------------------------------

def test_reference_policy_is_frozen_and_independent():
    m = Tiny()
    ref = ANC.ReferencePolicy(m)
    n = ref.assert_frozen()
    assert n == sum(p.numel() for p in m.parameters())
    # training the source must NOT move the reference
    before = ref.module.lin.weight.detach().clone()
    with torch.no_grad():
        m.lin.weight.add_(1.0)
    assert torch.allclose(before, ref.module.lin.weight), (
        "the reference moved with the source — it is a deepcopy, not a view")


def test_reference_policy_params_never_require_grad():
    ref = ANC.ReferencePolicy(Tiny())
    assert all(not p.requires_grad for p in ref.module.parameters())
    assert not ref.module.training, "the reference must be in eval mode"


def test_assert_frozen_catches_a_thawed_reference():
    """⛔ The anchor is a TARGET, never a second trainable path."""
    ref = ANC.ReferencePolicy(Tiny())
    next(ref.module.parameters()).requires_grad_(True)
    with pytest.raises(RuntimeError, match="requires grad"):
        ref.assert_frozen()


# ---------------------------------------------------------------------------
# The divergence
# ---------------------------------------------------------------------------

def test_divergence_is_zero_for_identical_trajectories():
    """And it is PER-CANDIDATE — shape [B, N], not a scalar."""
    t = torch.randn(2, 3, 5, 2)
    d = ANC.trajectory_divergence(t, t.clone())
    assert d.shape == (2, 3), "divergence must be per-candidate, for logging"
    assert torch.allclose(d, torch.zeros_like(d))


def test_l2_divergence_is_analytic():
    """A constant 3-4-5 offset: squared displacement 25 m² at every step."""
    a = torch.zeros(1, 1, 4, 2)
    b = torch.zeros(1, 1, 4, 2)
    b[..., 0], b[..., 1] = 3.0, 4.0
    assert float(ANC.trajectory_divergence(a, b, form="l2")) == pytest.approx(25.0)
    assert float(ANC.trajectory_divergence(a, b, form="l1")) == pytest.approx(7.0)


def test_divergence_grows_with_separation():
    a = torch.zeros(1, 1, 4, 2)
    near, far = a + 0.5, a + 5.0
    assert (float(ANC.trajectory_divergence(a, far))
            > float(ANC.trajectory_divergence(a, near)))


def test_a_kl_form_is_REFUSED_rather_than_approximated():
    """⛔ We do not have the decoder's step density; a 'kl' name would be a
    claim we cannot support."""
    t = torch.zeros(1, 1, 4, 2)
    with pytest.raises(ValueError, match="KL form needs the true diffusion"):
        ANC.trajectory_divergence(t, t, form="kl")


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError, match="must match"):
        ANC.trajectory_divergence(torch.zeros(1, 1, 4, 2), torch.zeros(1, 1, 5, 2))


def test_penalty_scales_with_weight_and_reports_both_parts():
    a = torch.zeros(1, 2, 4, 2)
    b = a + 1.0
    lo = ANC.anchor_penalty(a, b, w=0.1)
    hi = ANC.anchor_penalty(a, b, w=1.0)
    assert float(hi["penalty"]) == pytest.approx(10 * float(lo["penalty"]))
    assert lo["divergence"].shape == (1, 2), "per-candidate divergence for logging"


# ---------------------------------------------------------------------------
# ⛔ The behavioural test: does the anchor CONSTRAIN?
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("w_anchor", [0.0, 0.1, 10.0])
def test_anchor_actually_constrains_drift(w_anchor):
    """Optimise a free trajectory toward a far target, anchored to the origin.

    Stronger anchor ⇒ less drift. This is the property P-RC21 lacked, and it is
    checked by MEASURING movement, not by asserting the term exists.
    """
    ref = torch.zeros(1, 1, 6, 2)
    traj = nn.Parameter(torch.zeros(1, 1, 6, 2))
    opt = torch.optim.SGD([traj], lr=0.1)
    target = torch.full((1, 1, 6, 2), 10.0)
    for _ in range(200):
        pull = (traj - target).pow(2).mean()              # the "reward" pressure
        loss = pull + ANC.anchor_penalty(traj, ref, w=w_anchor)["penalty"]
        opt.zero_grad(); loss.backward(); opt.step()
    globals().setdefault("_drift", {})[w_anchor] = float(traj.detach().abs().mean())


def test_drift_is_monotonically_smaller_with_a_stronger_anchor():
    """Reads the three parametrised runs above — ordering is the claim."""
    d = globals().get("_drift", {})
    assert set(d) == {0.0, 0.1, 10.0}, "run the parametrised test first"
    assert d[0.0] > d[0.1] > d[10.0], (
        f"anchor did not constrain monotonically: {d}")
    assert d[10.0] < 0.5 * d[0.0], (
        f"a 100x anchor barely moved drift ({d}) — it is not binding")


def test_a_reference_of_an_UNCHANGED_model_has_ZERO_divergence():
    """⛔ THE CORRECTNESS PIN FOR THE ANCHOR.

    A frozen deepcopy compared against its own source, before any optimiser
    step, must diverge by EXACTLY zero. MEASURED 2026-08-29: it read
    **2.774 m²** — because the live model was in TRAIN mode (ego_dropout 0.5,
    route_dropout 0.5, stochastic diffusion) while `ReferencePolicy` forces
    eval. The anchor would have spent its entire budget penalising a MODE
    MISMATCH rather than policy drift, and the sweep would have measured the
    strength of a bug.

    A non-zero reading here means the two forwards are not comparable — check
    dropout/noise/mode before touching the divergence maths.
    """
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 4))
    m.eval()                                   # the deployed regime
    ref = ANC.ReferencePolicy(m)
    x = torch.randn(2, 4)
    live = m(x).reshape(2, 1, 2, 2)
    with torch.no_grad():
        r = ref(x).reshape(2, 1, 2, 2)
    d = ANC.trajectory_divergence(live, r)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-12), (
        f"a model vs its own frozen copy diverged by {float(d.mean()):.6f} — "
        "the two forwards are not in the same mode")


def test_a_MODE_MISMATCH_is_visible_as_nonzero_divergence():
    """The other direction: prove the pin above can actually fail."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 4))
    ref = ANC.ReferencePolicy(m)               # forced to eval
    m.train()                                  # live in TRAIN mode -> dropout on
    x = torch.randn(8, 4)
    live = m(x).reshape(8, 1, 2, 2)
    with torch.no_grad():
        r = ref(x).reshape(8, 1, 2, 2)
    d = ANC.trajectory_divergence(live, r)
    assert float(d.mean()) > 0.0, (
        "train-vs-eval must show up as divergence, or the zero-pin above "
        "proves nothing")
