"""refcv7 TOAD search — inversion, the no-worse-than-base guarantee, determinism."""
import torch

from tanitad.refs.refc_sampler import roll_controls
from tanitad.refs.refc_v3 import V3_HORIZONS
from tanitad.refs.refcv7_toad import ToadConfig, fit_slot_controls, toad_search


def rolled(u, v0):
    return roll_controls(u[:, None], v0, V3_HORIZONS, "kappa", 0.1)[:, 0]


def test_fit_inverts_the_programme_integrator():
    v0 = torch.tensor([8.0, 12.0])
    u = torch.zeros(2, 8, 2)
    u[0, :, 0] = 0.5                       # gentle acceleration
    u[1, :, 1] = 0.01                      # constant curvature, R = 100 m
    xy = rolled(u, v0)
    u_fit, rms = fit_slot_controls(xy, v0, V3_HORIZONS)
    assert float(rms.max()) < 0.05         # metres
    assert torch.allclose(rolled(u_fit, v0), xy, atol=0.15)


def _setup():
    v0 = torch.tensor([10.0, 10.0])
    u_base = torch.zeros(2, 8, 2)
    target = rolled(torch.stack([torch.zeros(8, 2), torch.zeros(8, 2)]) + torch.tensor([0.0, 0.004]), v0)

    def reward(xy):                        # higher when closer to a gently curving target
        return -((xy - target[:, None]) ** 2).sum(-1).mean(-1).sqrt()
    return v0, u_base, reward


def test_search_never_returns_worse_than_base_under_its_own_reward():
    v0, u_base, reward = _setup()
    out = toad_search(reward, u_base, v0, V3_HORIZONS, cfg=ToadConfig(), seed=0)
    assert torch.all(out["reward"] >= out["base_reward"] - 1e-6)


def test_search_moves_toward_the_reward():
    v0, u_base, reward = _setup()
    out = toad_search(reward, u_base, v0, V3_HORIZONS,
                      cfg=ToadConfig(iters=10, lam_anchor=0.0), seed=0)
    assert bool(out["took_search"].all())
    assert torch.all(out["reward"] > out["base_reward"])


def test_same_seed_same_plan_different_seed_is_a_fresh_draw():
    v0, u_base, reward = _setup()
    a = toad_search(reward, u_base, v0, V3_HORIZONS, seed=3)["xy"]
    b = toad_search(reward, u_base, v0, V3_HORIZONS, seed=3)["xy"]
    c = toad_search(reward, u_base, v0, V3_HORIZONS, seed=4)["xy"]
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
