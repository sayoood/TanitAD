"""V18 E3.4 ``train_stage_a`` pure-part smoke — CPU-only, no checkpoint, no
corpus, no GPU. The full path (frozen v5f trunk + v2 corpora + predictor
post-training) is pod-side and NOT exercised here.

What is pinned (the task's five groups):
  * the counterfactual SAMPLER + ENVELOPE: random draws inside their bounds
    and deterministic under a seed; every training arm (named + random)
    clamped to |a| <= A_MAX, |kappa| <= KAPPA_MAX even from out-of-envelope
    recorded actions; history (window actions before the last) untouched;
  * LOSS shapes/finiteness on mocks: a mock 1-step predictor + the real
    ``StepDisplacementReadout`` + the real ``rollout_transitions`` /
    ``decode_transitions`` — finite scalar losses, all component keys, grads
    reach the predictor, both ``ctrl_form`` branches;
  * the SUBSPACE-COMPLEMENT projection math on hand cases: residual against
    an explicit e1 basis, zero residual inside a planted subspace, basis
    orthonormality and dim clipping;
  * GATE-JSON branches incl. the NO-HARM check: all-pass, lateral-gain fail
    (the 0.27 defect), longitudinal-sign fail, lateral-sign regression fail,
    no-harm fail/boundary, missing channel -> None, p6 not computable ->
    None, longitudinal gain NOT gated, outcomes bound verbatim;
  * FROZEN-PROOF logic: ``selective_md5`` include/exclude sensitivity,
    ``frozen_proof_md5`` invariant under predictor mutation, and
    ``set_predictor_only_trainable`` flipping requires_grad on exactly the
    predictor params.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_stage_a import (LAT_CHANNELS, LON_CHANNELS,  # noqa: E402
                           NOHARM_FACTOR, OUTCOME_FAIL, OUTCOME_PASS,
                           TRAIN_ARMS, action_subspace_basis,
                           apply_random_counterfactual, build_cf_actions,
                           clamp_envelope, complement_residual,
                           frozen_proof_md5, sample_random_deltas,
                           selective_md5, set_predictor_only_trainable,
                           stage_a_gate_dict, stage_a_losses)
from stage_a_probes import kappa_of_steer, steer_of_kappa  # noqa: E402
from train_v58f_unicycle_head import A_MAX, KAPPA_MAX  # noqa: E402

B, W, H = 3, 6, 12


def _actions(steer=0.0, accel=0.0):
    aw2 = torch.zeros(B, W, 2)
    fa2 = torch.zeros(B, H, 2)
    aw2[..., 0] = steer
    aw2[..., 1] = accel
    fa2[..., 0] = steer
    fa2[..., 1] = accel
    return aw2, fa2


def _kappas(aw2, fa2):
    """All PERTURBABLE curvatures/accels: last window action + futures."""
    kap = torch.cat([kappa_of_steer(aw2[:, -1, 0]).reshape(-1),
                     kappa_of_steer(fa2[:, :, 0]).reshape(-1)])
    acc = torch.cat([aw2[:, -1, 1].reshape(-1), fa2[:, :, 1].reshape(-1)])
    return kap, acc


# ---------------------------------------------------------------------------
# 1. counterfactual sampler + envelope bounds
# ---------------------------------------------------------------------------
def test_sample_random_deltas_bounds_and_determinism():
    g1 = torch.Generator().manual_seed(7)
    dk, da = sample_random_deltas(256, g1, 0.05, 3.0)
    assert dk.shape == da.shape == (256,)
    assert float(dk.abs().max()) <= 0.05 and float(da.abs().max()) <= 3.0
    assert float(dk.abs().max()) > 0.0          # actually drawing, not zeros
    g2 = torch.Generator().manual_seed(7)
    dk2, da2 = sample_random_deltas(256, g2, 0.05, 3.0)
    assert torch.equal(dk, dk2) and torch.equal(da, da2)
    with pytest.raises(ValueError):
        sample_random_deltas(0, g1, 0.05, 3.0)


def test_random_arm_envelope_from_extreme_base():
    """Out-of-envelope recorded actions + big draws -> clamped inside."""
    aw2, fa2 = _actions(steer=float(steer_of_kappa(torch.tensor(0.3))),
                        accel=6.0)                # both OUTSIDE the envelope
    dk = torch.full((B,), 0.2)
    da = torch.full((B,), 5.0)
    aw, fa = build_cf_actions(aw2, fa2, "random", dk=dk, da=da)
    kap, acc = _kappas(aw, fa)
    assert float(kap.abs().max()) <= KAPPA_MAX + 1e-6
    assert float(acc.abs().max()) <= A_MAX + 1e-6


def test_named_arms_envelope_clamped():
    aw2, fa2 = _actions(steer=float(steer_of_kappa(torch.tensor(0.19))),
                        accel=3.5)
    for arm in ("left", "right", "brake", "throttle"):
        aw, fa = build_cf_actions(aw2, fa2, arm, dkappa=0.05, daccel=2.0)
        kap, acc = _kappas(aw, fa)
        assert float(kap.abs().max()) <= KAPPA_MAX + 1e-6, arm
        assert float(acc.abs().max()) <= A_MAX + 1e-6, arm
    # throttle from 3.5 with +2.0 must saturate at exactly A_MAX
    aw, fa = build_cf_actions(aw2, fa2, "throttle", daccel=2.0)
    assert float(fa[:, :, 1].max()) == pytest.approx(A_MAX)
    # left from kappa 0.19 with +0.05 saturates at exactly KAPPA_MAX
    aw, fa = build_cf_actions(aw2, fa2, "left", dkappa=0.05)
    assert float(kappa_of_steer(fa[0, 0, 0])) == pytest.approx(KAPPA_MAX,
                                                               abs=1e-6)


def test_history_untouched_by_every_arm():
    aw2, fa2 = _actions(steer=0.05, accel=1.0)
    dk = torch.full((B,), 0.03)
    da = torch.full((B,), 1.5)
    for arm in TRAIN_ARMS:
        aw, _fa = build_cf_actions(aw2, fa2, arm, dk=dk, da=da)
        assert torch.equal(aw[:, :-1], aw2[:, :-1]), arm


def test_random_counterfactual_moves_both_channels_per_window():
    aw2, fa2 = _actions(steer=0.0, accel=0.0)
    dk = torch.tensor([0.01, -0.02, 0.0])
    da = torch.tensor([1.0, -2.0, 0.5])
    aw, fa = apply_random_counterfactual(aw2, fa2, dk, da)
    got_k = kappa_of_steer(aw[:, -1, 0])
    assert torch.allclose(got_k, dk, atol=1e-6)
    assert torch.allclose(aw[:, -1, 1], da, atol=1e-6)
    assert torch.allclose(kappa_of_steer(fa[:, 3, 0]), dk, atol=1e-6)
    with pytest.raises(ValueError):
        apply_random_counterfactual(aw2, fa2, dk[:2], da[:2])
    with pytest.raises(ValueError):
        build_cf_actions(aw2, fa2, "random")      # draws required
    with pytest.raises(ValueError):
        build_cf_actions(aw2, fa2, "hold")        # not a training arm


def test_clamp_envelope_noop_inside_envelope():
    aw2, fa2 = _actions(steer=float(steer_of_kappa(torch.tensor(0.1))),
                        accel=2.0)
    aw, fa = clamp_envelope(aw2, fa2)
    assert torch.allclose(aw, aw2, atol=1e-6)
    assert torch.allclose(fa, fa2, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. losses on mocks — shapes, finiteness, grads, both ctrl forms
# ---------------------------------------------------------------------------
S, K = 16, 4


class _MockPredictor(nn.Module):
    """1-step contract of OperativePredictor: forward(states [B,W,S],
    actions [B,W,3]) -> {1: z_hat [B,S]} — a residual linear map so latents
    depend on BOTH the state and the last action (the roll and the losses
    have something to differentiate)."""

    def __init__(self, s=S, a=3):
        super().__init__()
        self.lin_a = nn.Linear(a, s)
        self.lin_s = nn.Linear(s, s)

    def forward(self, win_s, win_a, intent=None):
        return {1: win_s[:, -1] + 0.1 * self.lin_s(win_s[:, -1])
                + 0.1 * self.lin_a(win_a[:, -1])}


def _loss_inputs():
    torch.manual_seed(0)
    from tanitad.models.metric_dynamics import StepDisplacementReadout
    pred = _MockPredictor()
    sr = StepDisplacementReadout(S, hidden=32)
    states = torch.randn(B, W, S)
    aw2, fa2 = _actions(steer=0.02, accel=0.5)
    v0 = torch.full((B,), 8.0)
    gt = torch.randn(B, K, 2) * 0.1
    z_true = torch.randn(B, S)
    dk = torch.tensor([0.01, -0.02, 0.03])
    da = torch.tensor([0.5, -1.0, 1.5])
    return pred, sr, states, aw2, fa2, v0, gt, z_true, dk, da


def test_losses_shapes_finiteness_and_grads():
    pred, sr, states, aw2, fa2, v0, gt, z_true, dk, da = _loss_inputs()
    L = stage_a_losses(pred, sr, states, aw2, fa2, v0, gt, z_true, K,
                       rand_dk=dk, rand_da=da)
    for key in ("loss", "l_ctrl", "l_fact", "l_scene", "l_scene_cf",
                "l_scene_true"):
        v = L[key]
        assert v.ndim == 0 and torch.isfinite(v), key
    assert set(L["l_ctrl_arms"]) == set(TRAIN_ARMS)
    assert all(torch.isfinite(v) for v in L["l_ctrl_arms"].values())
    assert 1 <= L["basis_dims"] <= 8
    assert L["loss"].requires_grad
    L["loss"].backward()
    assert pred.lin_a.weight.grad is not None
    assert torch.isfinite(pred.lin_a.weight.grad).all()
    # frozen-readout contract: grads FLOW THROUGH the readout to the
    # predictor even when its params don't train (requires_grad stays True
    # here only because the mock never froze it — main() freezes it)


def test_losses_absolute_form_and_weight_zero():
    pred, sr, states, aw2, fa2, v0, gt, z_true, dk, da = _loss_inputs()
    L = stage_a_losses(pred, sr, states, aw2, fa2, v0, gt, z_true, K,
                       rand_dk=dk, rand_da=da, ctrl_form="absolute",
                       w_scene=0.0)
    assert torch.isfinite(L["loss"]) and L["ctrl_form"] == "absolute"
    # w_scene=0 removes the scene term from the total but still reports it
    total = (L["l_ctrl"] + L["l_fact"]).detach()
    assert torch.allclose(L["loss"].detach(), total, atol=1e-6)
    with pytest.raises(ValueError):
        stage_a_losses(pred, sr, states, aw2, fa2, v0, gt, z_true, K,
                       rand_dk=dk, rand_da=da, ctrl_form="nope")
    with pytest.raises(ValueError):        # gt horizon mismatch
        stage_a_losses(pred, sr, states, aw2, fa2, v0, gt[:, :K - 1],
                       z_true, K, rand_dk=dk, rand_da=da)


def test_losses_response_zero_when_predictor_ignores_actions():
    """A predictor that IGNORES actions (the §1.12 echo in the limit) has an
    identically-zero counterfactual response, so the response-form L_ctrl
    equals the mean |analytic response| — strictly positive. Pins that the
    loss actually SEES the gain defect."""
    pred, sr, states, aw2, fa2, v0, gt, z_true, dk, da = _loss_inputs()

    class _Deaf(nn.Module):
        def forward(self, win_s, win_a, intent=None):
            return {1: win_s[:, -1]}

    L = stage_a_losses(_Deaf(), sr, states, aw2, fa2, v0, gt, z_true, K,
                       rand_dk=dk, rand_da=da)
    assert float(L["l_ctrl"].detach()) > 0.0
    # and the counterfactual latent deltas are all zero -> scene-cf term 0
    assert float(L["l_scene_cf"]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 3. subspace-complement projection math
# ---------------------------------------------------------------------------
def test_complement_residual_hand_case():
    basis = torch.tensor([[1.0, 0.0, 0.0]])           # e1
    x = torch.tensor([[3.0, 4.0, 5.0]])
    r = complement_residual(x, basis)
    assert torch.allclose(r, torch.tensor([[0.0, 4.0, 5.0]]), atol=1e-6)
    with pytest.raises(ValueError):
        complement_residual(x, torch.eye(4))          # dim mismatch


def test_basis_from_planted_subspace_and_zero_residual():
    torch.manual_seed(1)
    D = 10
    coef = torch.randn(40, 2)
    dirs = torch.zeros(2, D)
    dirs[0, 0] = 1.0
    dirs[1, 1] = 1.0
    deltas = coef @ dirs                              # rank-2 in span(e0,e1)
    q = action_subspace_basis(deltas, n_dims=2)
    assert q.shape == (2, D)
    assert torch.allclose(q @ q.T, torch.eye(2), atol=1e-5)  # orthonormal
    inside = torch.tensor([[2.0, -3.0] + [0.0] * (D - 2)])
    assert float(complement_residual(inside, q).abs().max()) < 1e-5
    e2 = torch.zeros(1, D)
    e2[0, 2] = 1.0
    assert torch.allclose(complement_residual(e2, q), e2, atol=1e-5)


def test_basis_dim_clipping_and_guards():
    d = torch.randn(3, 20)
    assert action_subspace_basis(d, n_dims=8).shape == (3, 20)  # N clips m
    with pytest.raises(ValueError):
        action_subspace_basis(torch.randn(4), 8)      # not 2-D
    with pytest.raises(ValueError):
        action_subspace_basis(torch.full((4, 4), float("nan")), 2)


# ---------------------------------------------------------------------------
# 4. gate JSON branches (incl. no-harm)
# ---------------------------------------------------------------------------
def _ch(sign, gain):
    return {"sign_rate": sign, "gain_median": gain, "n_admissible": 800}


def _probe(lat_sign=0.99, lat_gain=1.0, lon_sign=0.97, lon_gain=0.8,
           p6_dims=3, ade=0.50, p6_computable=True):
    return {"per_channel": {"left": _ch(lat_sign, lat_gain),
                            "right": _ch(lat_sign, lat_gain),
                            "brake": _ch(lon_sign, lon_gain),
                            "throttle": _ch(lon_sign, lon_gain)},
            "p6": ({"computable": True, "dims_for_var_target": p6_dims}
                   if p6_computable else
                   {"computable": False, "reason": "no deltas"}),
            "factual_ade": ade}


def test_gate_all_pass_and_outcomes_verbatim():
    g = stage_a_gate_dict(_probe(), _probe(lat_gain=0.27, lon_sign=0.76))
    assert g["PASS"] is True
    assert all(c["pass"] is True for k, c in g["checks"].items()
               if "not_gated" not in k)
    assert g["outcomes_bound_in_advance"]["PASS"] == OUTCOME_PASS
    assert g["outcomes_bound_in_advance"]["FAIL"] == OUTCOME_FAIL
    assert "joint trunk training (v6)" in OUTCOME_FAIL
    assert "W7 re-run + E1.4 T1 re-run" in OUTCOME_PASS


def test_gate_lateral_gain_fail_the_027_defect():
    g = stage_a_gate_dict(_probe(lat_gain=0.27), _probe())
    assert g["PASS"] is False
    assert g["checks"]["lat_gain_left"]["pass"] is False
    assert g["checks"]["lat_sign_stays_left"]["pass"] is True


def test_gate_longitudinal_sign_fail():
    g = stage_a_gate_dict(_probe(lon_sign=0.79), _probe())
    assert g["PASS"] is False
    assert g["checks"]["lon_sign_brake"]["pass"] is False


def test_gate_lateral_sign_regression_fail():
    g = stage_a_gate_dict(_probe(lat_sign=0.90), _probe(lat_sign=0.995))
    assert g["PASS"] is False
    assert g["checks"]["lat_sign_stays_left"]["pass"] is False
    assert g["checks"]["lat_sign_stays_left"]["pre_value"] == 0.995


def test_gate_p6_dims_fail_and_not_computable():
    g = stage_a_gate_dict(_probe(p6_dims=40), _probe())
    assert g["PASS"] is False and g["checks"]["p6_dims"]["pass"] is False
    g2 = stage_a_gate_dict(_probe(p6_computable=False), _probe())
    assert g2["checks"]["p6_dims"]["pass"] is None
    assert g2["PASS"] is None                        # unknown, not fake-pass


def test_gate_no_harm_branches():
    # +12 % worse than pre -> fail
    g = stage_a_gate_dict(_probe(ade=0.56), _probe(ade=0.50))
    assert g["checks"]["no_harm_factual_ade"]["pass"] is False
    assert g["PASS"] is False
    # exactly at the +10 % cap -> pass
    g2 = stage_a_gate_dict(_probe(ade=0.55), _probe(ade=0.50))
    assert g2["checks"]["no_harm_factual_ade"]["pass"] is True
    assert g2["checks"]["no_harm_factual_ade"]["cap_m"] == \
        pytest.approx(NOHARM_FACTOR * 0.50)
    # missing pre ADE -> None, PASS None
    pre = _probe()
    pre["factual_ade"] = None
    g3 = stage_a_gate_dict(_probe(), pre)
    assert g3["checks"]["no_harm_factual_ade"]["pass"] is None
    assert g3["PASS"] is None


def test_gate_missing_channel_yields_none_and_lon_gain_not_gated():
    post = _probe()
    del post["per_channel"]["right"]
    g = stage_a_gate_dict(post, _probe())
    assert g["checks"]["lat_gain_right"]["pass"] is None
    assert g["PASS"] is None
    # longitudinal gain 0.1 (way off-band) with everything else passing:
    g2 = stage_a_gate_dict(_probe(lon_gain=0.1), _probe())
    assert g2["PASS"] is True
    assert g2["checks"]["lon_gain_brake_reported_not_gated"]["pass"] == \
        "not gated"
    assert set(LAT_CHANNELS) == {"left", "right"}
    assert set(LON_CHANNELS) == {"brake", "throttle"}


# ---------------------------------------------------------------------------
# 5. frozen-proof logic
# ---------------------------------------------------------------------------
class _TinyWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = nn.Linear(4, 4)
        self.encoder = nn.Linear(4, 4)
        self.readout = nn.Linear(4, 2)


def test_selective_md5_include_exclude_sensitivity():
    torch.manual_seed(2)
    w = _TinyWorld()
    inc0 = selective_md5(w, include_prefix="predictor.")
    exc0 = selective_md5(w, exclude_prefix="predictor.")
    with torch.no_grad():
        w.predictor.weight += 1.0
    assert selective_md5(w, include_prefix="predictor.") != inc0
    assert selective_md5(w, exclude_prefix="predictor.") == exc0
    with torch.no_grad():
        w.encoder.weight += 1.0
    assert selective_md5(w, exclude_prefix="predictor.") != exc0


def test_frozen_proof_md5_predictor_invariant():
    torch.manual_seed(3)
    w = _TinyWorld()
    g = nn.Linear(4, 3)
    f0 = frozen_proof_md5(w, g)
    with torch.no_grad():
        w.predictor.weight += 1.0                     # training the predictor
    assert frozen_proof_md5(w, g) == f0               # frozen proof holds
    with torch.no_grad():
        g.weight += 1.0                               # grounding drift
    assert frozen_proof_md5(w, g) != f0               # ... is caught


def test_set_predictor_only_trainable():
    w = _TinyWorld()
    g = nn.Linear(4, 3)
    n_train, n_frozen = set_predictor_only_trainable(w, g)
    assert n_train == sum(p.numel() for p in w.predictor.parameters())
    assert n_frozen == (sum(p.numel() for p in w.parameters()) - n_train
                        + sum(p.numel() for p in g.parameters()))
    for n_, p in w.named_parameters():
        assert p.requires_grad == n_.startswith("predictor."), n_
    assert all(not p.requires_grad for p in g.parameters())
    with pytest.raises(RuntimeError):                 # no predictor.* params
        set_predictor_only_trainable(nn.Linear(2, 2), g)
