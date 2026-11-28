"""The conditioning parameterisation — PI directive 2026-08-26.

⭐ THE CLAIM UNDER TEST: switching channel 0 from the bicycle-model steering proxy
`atan(L·κ)` to the MEASURED YAW RATE `ω = v·κ` is an EXACT conversion in which the
wheelbase cancels, so it needs no re-cache and cannot break parity.

⛔ AND THE GUARD THAT MATTERS MORE: the incumbent must remain bit-identical by
default. This flag changes the predictor's input distribution — every existing
checkpoint and every cross-arm comparison in the programme depends on the default
not moving silently.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from train_v6_staged import (  # noqa: E402
    COND_EGO_STATE, COND_INCUMBENT, _COND_WHEELBASE_M, _lift3,
)


def _mk(b=8, w=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    a2 = torch.stack([torch.rand(b, w, generator=g) * 0.6 - 0.3,      # steer, rad
                      torch.randn(b, w, generator=g)], dim=-1)        # a_long
    v0 = torch.rand(b, generator=g) * 20.0 + 1.0                      # m/s
    return a2, v0


def test_the_default_is_bit_identical_to_the_incumbent():
    """⛔ THE LOAD-BEARING GUARD. If this fails, every banked checkpoint's input
    distribution has moved and no cross-arm number in the programme is comparable."""
    a2, v0 = _mk()
    assert torch.equal(_lift3(a2, v0), _lift3(a2, v0, COND_INCUMBENT))


def test_the_conversion_recovers_the_true_yaw_rate_EXACTLY():
    """⭐ steer = atan(L·κ) ⟹ v·tan(steer)/L = v·κ = ω, and L cancels.

    Built the forward way — from a known curvature — so the test cannot pass by
    reproducing the implementation's own algebra."""
    g = torch.Generator().manual_seed(7)
    kappa = torch.rand(8, 6, generator=g) * 0.1 - 0.05          # 1/m
    v0 = torch.rand(8, generator=g) * 20.0 + 1.0
    steer = torch.atan(_COND_WHEELBASE_M * kappa)               # the forward map
    a2 = torch.stack([steer, torch.zeros_like(steer)], dim=-1)
    got = _lift3(a2, v0, COND_EGO_STATE)[..., 0]
    want = v0[:, None] * kappa                                   # omega = v·κ
    assert torch.allclose(got, want, atol=1e-5), (got[0, :3], want[0, :3])


def test_the_wheelbase_genuinely_cancels():
    """If the wheelbase did NOT cancel, a different L would change the result —
    which would silently make every clip's conditioning depend on a legacy 2.9 m
    constant. Build the same physical situation with two different L and require
    the recovered omega to agree."""
    import train_v6_staged as M
    g = torch.Generator().manual_seed(11)
    kappa = torch.rand(4, 5, generator=g) * 0.08 - 0.04
    v0 = torch.rand(4, generator=g) * 15.0 + 2.0
    out = []
    for L in (2.9, 3.7):
        old = M._COND_WHEELBASE_M
        try:
            M._COND_WHEELBASE_M = L
            a2 = torch.stack([torch.atan(L * kappa), torch.zeros_like(kappa)], -1)
            out.append(M._lift3(a2, v0, COND_EGO_STATE)[..., 0].clone())
        finally:
            M._COND_WHEELBASE_M = old
    assert torch.allclose(out[0], out[1], atol=1e-5)


def test_only_channel_zero_moves():
    """`a_long` and `v` must pass through untouched — the directive changes the
    rotation channel, nothing else."""
    a2, v0 = _mk()
    inc, ego = _lift3(a2, v0, COND_INCUMBENT), _lift3(a2, v0, COND_EGO_STATE)
    assert torch.equal(inc[..., 1], ego[..., 1]), "a_long moved"
    assert torch.equal(inc[..., 2], ego[..., 2]), "v moved"
    assert not torch.equal(inc[..., 0], ego[..., 0]), "channel 0 did NOT move"


def test_the_new_channel_carries_speed_and_the_old_one_does_not():
    """⭐ THE POINT OF THE DIRECTIVE. `steer` is speed-blind: the same steering at
    5 m/s and 20 m/s rotates the image at completely different rates, and the
    incumbent hides that from channel 0."""
    a2, _ = _mk(b=2, w=4, seed=3)
    slow = torch.full((2,), 5.0)
    fast = torch.full((2,), 20.0)
    inc_s, inc_f = _lift3(a2, slow, COND_INCUMBENT), _lift3(a2, fast, COND_INCUMBENT)
    assert torch.equal(inc_s[..., 0], inc_f[..., 0]), "incumbent ch0 should be speed-blind"
    ego_s, ego_f = _lift3(a2, slow, COND_EGO_STATE), _lift3(a2, fast, COND_EGO_STATE)
    assert not torch.allclose(ego_s[..., 0], ego_f[..., 0]), \
        "the measured yaw rate MUST scale with speed"
    # and it scales linearly: 4x the speed => 4x the yaw rate
    assert torch.allclose(ego_f[..., 0], 4.0 * ego_s[..., 0], atol=1e-5)


def test_an_unknown_parameterisation_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown --cond-param"):
        a2, v0 = _mk()
        _lift3(a2, v0, "yaw_only")


def test_zero_steering_gives_zero_yaw_rate_at_any_speed():
    """A sanity anchor with a known value: driving straight cannot rotate you."""
    a2 = torch.zeros(4, 5, 2)
    out = _lift3(a2, torch.tensor([0.0, 5.0, 12.0, 30.0]), COND_EGO_STATE)
    assert torch.allclose(out[..., 0], torch.zeros_like(out[..., 0]), atol=1e-9)
