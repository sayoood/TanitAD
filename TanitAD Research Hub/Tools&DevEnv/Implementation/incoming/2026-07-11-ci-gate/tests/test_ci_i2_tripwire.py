"""Tests for the CI I2 tripwire (Tools&DevEnv backlog #3).

The point of the tripwire is to catch the batch-statistic class of bug. So the
adversarial test is: a deliberately batch-DEPENDENT encoder MUST fail the tripwire.
A batch-independent one MUST pass. Plus one integration check that the real
WorldModel encoder passes (the invariant the gate actually protects in commits).
"""

import torch

from ci_i2_tripwire import check_encode_fn, run_i2_tripwire


def _frames(n=8):
    torch.manual_seed(0)
    return torch.rand(n, 1, 64, 64)


def _linear_encoder(dim=16):
    """Per-frame linear map [B,1,64,64] -> [B,dim]. Batch-INDEPENDENT (good)."""
    torch.manual_seed(1)
    w = torch.randn(64 * 64, dim)
    return lambda x: x.flatten(1) @ w


def _batchstat_encoder(dim=16):
    """Subtracts the BATCH mean before projecting — the BatchNorm bug class:
    identical in training, batch-1 != batch-B at inference (bad)."""
    torch.manual_seed(1)
    w = torch.randn(64 * 64, dim)
    return lambda x: (x - x.mean(dim=0, keepdim=True)).flatten(1) @ w


def test_batch_independent_encoder_passes():
    ok, dev = check_encode_fn(_linear_encoder(), _frames(), tol=1e-4)
    assert ok, f"clean per-frame encoder should pass I2 (dev={dev})"
    assert dev < 1e-4


def test_batch_statistic_encoder_fails():
    # Falsifier: the tripwire must REJECT a batch-dependent encoder.
    ok, dev = check_encode_fn(_batchstat_encoder(), _frames(), tol=1e-4)
    assert not ok, "tripwire failed to catch a batch-statistic layer"
    assert dev > 1e-4


def test_real_worldmodel_encoder_passes():
    # Integration: the shipped encoder is batch-1 consistent (the commit invariant).
    ok, dev = run_i2_tripwire(tol=1e-4, batch=8)
    assert ok, f"real WorldModel encoder violates I2 (dev={dev})"
