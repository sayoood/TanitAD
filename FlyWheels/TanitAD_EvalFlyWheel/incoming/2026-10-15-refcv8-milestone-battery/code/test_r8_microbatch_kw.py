"""Pins r8_repoint.patch_microbatch: refcv8's r8_nav / r8_rc must be sliced per micro-part (2026-10-16 probe failure)."""
import sys
from pathlib import Path

import pytest
import torch

BATTERY_CODE = Path("C:/lgs/Lfb47a58376/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/code")
sys.path.insert(0, str(BATTERY_CODE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import microbatch as MB  # noqa: E402

ORIG = tuple(MB._BATCH_KW)


class _Core(torch.nn.Module):
    ego_hist = None


class _Stub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = _Core()

    def forward(self, frames, r8_nav=None, r8_rc=None):
        assert r8_nav.shape[0] == frames.shape[0], (r8_nav.shape, frames.shape)
        assert r8_rc.shape[0] == frames.shape[0], (r8_rc.shape, frames.shape)
        return {"y": frames.sum(1) + r8_nav.sum(1) + r8_rc.sum(1)}


def _run():
    torch.manual_seed(0)
    frames, nav, rc = torch.randn(16, 3), torch.randn(16, 6), torch.randn(16, 4)
    full = _Stub()(frames, r8_nav=nav, r8_rc=rc)["y"]
    mb = MicroBatchForwardFor(_Stub())
    return full, mb(frames, r8_nav=nav, r8_rc=rc)["y"]


def MicroBatchForwardFor(model):
    return MB.MicroBatchForward(model, [3, 3, 3, 3, 4]).install().model


def test_unpatched_micro_split_fails_like_the_probe():
    MB._BATCH_KW = ORIG
    with pytest.raises(AssertionError):
        _run()


def test_patched_split_equals_the_full_batch_exactly():
    import r8_repoint
    MB._BATCH_KW = ORIG
    kw = r8_repoint.patch_microbatch()
    assert "r8_nav" in kw and "r8_rc" in kw
    full, split = _run()
    assert torch.equal(full, split)
    assert r8_repoint.patch_microbatch() == kw            # idempotent
