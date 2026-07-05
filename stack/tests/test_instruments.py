"""Instrument doctrine tests (D-004). I2 runs on the real encoder — this is the
CI tripwire that prevents the entire BatchNorm class of bugs forever."""

import torch

from tanitad.config import smoke_config
from tanitad.instruments.checks import (i2_batch_consistency, i3_episode_split,
                                        i4_imag_relative)
from tanitad.models.fourbrain import WorldModel


def test_i2_batch_consistency_of_encoder():
    torch.manual_seed(0)
    model = WorldModel(smoke_config()).eval()
    frames = torch.rand(8, 1, 64, 64)
    with torch.no_grad():
        ok, dev = i2_batch_consistency(model.encode, frames, tol=1e-4)
    assert ok, f"encoder violates batch-1 consistency (dev={dev}) — " \
               f"a batch-statistic layer is in the inference path"


def test_i3_split_is_disjoint():
    train, val = i3_episode_split(list(range(20)), val_frac=0.25)
    assert not set(train) & set(val)
    assert len(val) == 5


def test_i4_persistence_ratio():
    z_prev = torch.zeros(4, 8)
    z_true = torch.ones(4, 8)
    perfect = i4_imag_relative(z_true.clone(), z_true, z_prev)
    persistence = i4_imag_relative(z_prev.clone(), z_true, z_prev)
    assert perfect < 0.01
    assert abs(persistence - 1.0) < 1e-5
