"""Standalone tests for the spectral-sizing tool (backlog #0, L2).

The core claim to validate: on latents whose action-conditioned dynamics have a
KNOWN low rank r embedded in a larger state dim S, the estimator must recover the
knee at ~r (not S) and report the spectral tail beyond r as ~0. Plus an end-to-end
run through the real WorldModel latent path.

    pytest "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-14-spectral-sizing-p0/tests" -q
"""

import pytest
import torch

import tanitad.eval.spectral as sp
from tanitad.config import smoke_config
from tanitad.models.fourbrain import WorldModel


def _low_rank_dynamics(n=4000, s=32, r=5, a_dim=2, noise=1e-3, seed=0):
    """Latents with a rank-r linear action-conditioned transition embedded in R^S.

    u_t in R^r evolves u_{t+1} = D u_t + E a_t; z = u @ P with P [r, S]. So the true
    z_t -> z_next operator has rank exactly r; a well-behaved estimator recovers it.
    """
    g = torch.Generator().manual_seed(seed)
    # orthonormal-row embedding: u is recoverable from z without pseudo-inverse
    # noise amplification, so the operator's rank is cleanly r (well-conditioned).
    P = torch.linalg.qr(torch.randn(s, r, generator=g))[0].T   # [r, s], P P^T = I_r
    # D with CONTROLLED, comparable singular values [1.0 .. 0.5] so all r modes sit
    # above the 99% energy line (each carries a meaningful fraction) -> knee == r.
    q1 = torch.linalg.qr(torch.randn(r, r, generator=g))[0]
    q2 = torch.linalg.qr(torch.randn(r, r, generator=g))[0]
    D = q1 @ torch.diag(torch.linspace(1.0, 0.5, r)) @ q2.T
    E = torch.randn(r, a_dim, generator=g)
    u = torch.randn(n, r, generator=g)
    a = torch.randn(n, a_dim, generator=g)
    u_next = u @ D.T + a @ E.T
    z_t = u @ P + noise * torch.randn(n, s, generator=g)
    z_next = u_next @ P + noise * torch.randn(n, s, generator=g)
    return z_t, a, z_next, r


# --------------------------------------------------------------------------- #
# spectral primitives                                                          #
# --------------------------------------------------------------------------- #
def test_effective_rank_extremes():
    assert sp.effective_rank(torch.tensor([1.0, 0, 0, 0])) == pytest.approx(1.0, abs=1e-6)
    flat = torch.ones(8)
    assert sp.effective_rank(flat) == pytest.approx(8.0, rel=1e-6)


def test_energy_knee_and_tail_on_known_rank():
    sv = torch.tensor([10.0, 9.0, 8.0, 1e-4, 1e-4])   # rank ~3
    assert sp.energy_knee(sv, 0.99) == 3
    assert sp.spectral_tail(sv, 3) < 1e-6
    assert sp.spectral_tail(sv, 0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the load-bearing recovery test                                              #
# --------------------------------------------------------------------------- #
def test_recovers_known_low_rank_knee():
    # noise=0 -> z_t is exactly rank r, so the fitted operator is exactly rank r
    # (with obs noise, ridge amplifies finite-sample cross-cov in z_t's near-null
    # directions into a spurious tail; optimal_k stays robust â€” see the separate test).
    z_t, a, z_next, r = _low_rank_dynamics(r=5, s=32, noise=0.0)
    rep = sp.estimate_transition_spectrum(z_t, a, z_next, current_readout_dim=32)
    assert rep.fit_r2 > 0.95                              # linear-operator proxy is appropriate here
    assert rep.energy_knee_k == r                         # knee recovers the true dynamics rank
    assert rep.operator_effective_rank == pytest.approx(r, abs=1.0)
    assert rep.spectral_tail_at[r] < 1e-3                 # ~no energy beyond the true rank
    assert rep.operator_svals[r - 1] > 100 * rep.operator_svals[r]  # sharp gap at r


def test_over_provisioned_flag_when_readout_dwarfs_rank():
    z_t, a, z_next, r = _low_rank_dynamics(r=4, s=48)
    rep = sp.estimate_transition_spectrum(z_t, a, z_next,
                                          current_readout_dim=sp.CURRENT_READOUT_DIM)  # 2048 >> 4
    assert "OVER-PROVISIONED" in rep.recommendation
    assert rep.current_readout_dim == 2048


def test_under_provisioned_flag_when_readout_below_knee():
    z_t, a, z_next, r = _low_rank_dynamics(r=12, s=32)
    rep = sp.estimate_transition_spectrum(z_t, a, z_next, current_readout_dim=3)
    assert "UNDER-PROVISIONED" in rep.recommendation


def test_optimal_k_moves_toward_knee_with_more_samples():
    # more samples -> weaker O(k^2)/N penalty -> k* not smaller than with few samples
    zt_s, a_s, zn_s, r = _low_rank_dynamics(n=300, r=8, s=32, seed=1)
    zt_l, a_l, zn_l, _ = _low_rank_dynamics(n=8000, r=8, s=32, seed=1)
    k_small = sp.estimate_transition_spectrum(zt_s, a_s, zn_s, current_readout_dim=32).optimal_k_theory
    k_large = sp.estimate_transition_spectrum(zt_l, a_l, zn_l, current_readout_dim=32).optimal_k_theory
    assert k_large >= k_small


# --------------------------------------------------------------------------- #
# extraction helpers + end-to-end                                             #
# --------------------------------------------------------------------------- #
def test_pairs_from_states_alignment():
    states = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    actions = torch.zeros(2, 4, 2)
    z_t, a_t, z_next = sp.pairs_from_states(states, actions)
    assert z_t.shape == (6, 3) and z_next.shape == (6, 3) and a_t.shape == (6, 2)
    # z_next is the state one step ahead of z_t
    assert torch.allclose(z_next[0], states[0, 1])
    assert torch.allclose(z_t[0], states[0, 0])


def test_end_to_end_worldmodel_latent_path_is_wellformed():
    cfg = smoke_config()
    world = WorldModel(cfg)
    world.eval()
    n, w, c, s = 12, cfg.predictor.window, cfg.encoder.in_channels, cfg.encoder.image_size
    frames = torch.randn(n, w, c, s, s)
    states = sp.latents_from_world(world, frames)         # [N, W, S]
    z_t, a_t, z_next = sp.pairs_from_states(states, torch.randn(n, w, 2))
    rep = sp.estimate_transition_spectrum(z_t, a_t, z_next,
                                          current_readout_dim=states.shape[-1])
    # Well-formed report; an UNTRAINED encoder yields a degenerate spectrum, so we
    # assert structure only â€” never a sizing claim (P8).
    assert rep.state_dim == states.shape[-1]
    assert len(rep.operator_svals) == rep.state_dim
    assert 1 <= rep.energy_knee_k <= rep.state_dim
    assert isinstance(rep.recommendation, str) and rep.recommendation
