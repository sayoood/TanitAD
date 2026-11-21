"""PREREG_V8F_LORA Amendment 3 (2026-10-25): every encoder call on the LoRA training path sees EXACTLY the cache builder's batch of 8 frames.
AR-2 run 2's training-path identity probe encoded 9 frames in one call and read rel-L2 2.41e-2 against the cache (75.6 % bitwise equal, the
largest difference one fp8 step), while the builder-shaped identity driver matched it bit for bit. A spy encoder records every call's batch;
the pad rows must carry no output and exactly zero gradient; the constant is pinned to the builder's own loop."""
import re
from pathlib import Path

import pytest
import torch

from tanitad.models import v8f_lora_encoder as L

REPO = Path(__file__).resolve().parents[2]


class Spy(torch.nn.Module):
    """A row-independent stand-in for the encoder: ``pooled(x) = w * mean(x)`` per frame, broadcast to ``[n, 2, 3]``; records every batch size."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64))
        self.calls = []

    def pooled(self, x):
        self.calls.append(int(x.shape[0]))
        f = x.to(torch.float64).mean(dim=(1, 2, 3))
        return (self.w * f)[:, None, None].expand(x.shape[0], 2, 3)


def frames(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, 4, 5), generator=g, dtype=torch.uint8)


def only_builder_shaped(calls):
    return [c for c in calls if c != L.BUILDER_BATCH]


@pytest.mark.parametrize("n,chunk", [(9, 16), (1, 16), (8, 16), (17, 16), (23, 7), (16, 16)])
def test_every_encoder_call_is_builder_shaped_and_the_rows_are_the_real_frames(n, chunk):
    enc, fr = Spy(), frames(n)
    out = L.encode_chunked(enc, fr, chunk, grad=False, device="cpu")
    assert only_builder_shaped(enc.calls) == [], enc.calls
    assert out.shape == (n, 2, 3)
    want = 1.5 * fr.to(torch.float64).mean(dim=(1, 2, 3))
    assert torch.equal(out[:, 0, 0], want)


def test_pad_rows_get_exactly_zero_gradient_through_cached_grad_step():
    enc, fr = Spy(), frames(9, seed=3)
    fmap = torch.tensor([[0, 1, 2], [3, 4, 8]])
    C = torch.arange(1, 1 + 2 * 3 * 2 * 3, dtype=torch.float64).view(2, 3, 2, 3)

    def run_loss(feats):
        return {"terms": {"total": (feats * C).sum()}}

    L.cached_grad_step(enc, fr, fmap, run_loss, chunk=16, device="cpu")
    assert only_builder_shaped(enc.calls) == [], enc.calls
    m = fr.to(torch.float64).mean(dim=(1, 2, 3))
    want = sum(float(C[b, k].sum()) * float(m[int(fmap[b, k])]) for b in range(2) for k in range(3))     # d total / d w, analytic
    assert enc.w.grad is not None and float(enc.w.grad) == pytest.approx(want, rel=0, abs=1e-9)


def test_the_constant_is_the_builders_loop_and_the_identity_drivers():
    ship = (REPO / "stack" / "scripts" / "dinov3_fp8_encode_ship.py").read_text(encoding="utf-8")
    assert re.search(r"for j in range\(0, frames\.shape\[0\], 8\)", ship), "the builder's encode loop moved: re-derive BUILDER_BATCH"
    drv = next(REPO.glob("TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/code/v8f_lora_identity_dev.py"))
    assert re.search(r"^BUILDER_BATCH = 8\b", drv.read_text(encoding="utf-8"), re.M)
    assert L.BUILDER_BATCH == 8


def test_mutant_the_unshaped_call_is_caught(monkeypatch):
    monkeypatch.setattr(L, "pooled_builder_shaped", lambda enc, x: enc.pooled(x))
    enc = Spy()
    L.encode_chunked(enc, frames(9), 16, grad=False, device="cpu")
    assert only_builder_shaped(enc.calls) == [9]
