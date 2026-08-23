"""Guard: the flagship-speed blind-rollout variant uses metric primitives that
agree bit-for-bit with the validated `blind_rollout` ones — so the 2026-07-18
operative-model curves are read on the SAME instrument as the 2026-07-17 curves.

If someone edits one copy and not the other, this fails loudly.

Run:  <tanitad venv>/python -m pytest tests/ -q   (from the package dir)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import blind_rollout as br            # noqa: E402
import blind_rollout_flagship as brf  # noqa: E402


def _tokens(b=4, n=16, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, d, generator=g)


def test_center_parity():
    x = _tokens()
    assert torch.allclose(br._center(x), brf._center(x))


def test_hidden_cosine_parity():
    true = _tokens(seed=1)
    pred = _tokens(seed=2)
    hidden = torch.ones(true.shape[0], true.shape[1], dtype=torch.bool)
    assert abs(br._hidden_cosine(pred, true, hidden)
               - brf._hidden_cosine(pred, true, hidden)) < 1e-9


def test_rel_l2_parity():
    true = _tokens(seed=3)
    hidden = torch.ones(true.shape[0], true.shape[1], dtype=torch.bool)
    assert abs(br._rel_l2(true + 0.5, true, hidden)
               - brf._rel_l2(true + 0.5, true, hidden)) < 1e-9


def test_inter_sample_cos_parity():
    pred = _tokens(b=6, n=16, d=8, seed=7)
    hidden = torch.zeros(6, 16, dtype=torch.bool)
    hidden[:, 8:] = True
    a = br._inter_sample_cos(pred, hidden)
    b = brf._inter_sample_cos(pred, hidden)
    assert abs(a - b) < 1e-9


def test_flagship_points_at_operative_ckpt():
    # the variant must target the flagship-speed ckpt + physicalai val, not the
    # 07-17 base250cam comma paths (prevents a silent copy-paste regression).
    assert "flagship-speed" in brf.CKPT
    assert "physicalai-val" in brf.VAL
