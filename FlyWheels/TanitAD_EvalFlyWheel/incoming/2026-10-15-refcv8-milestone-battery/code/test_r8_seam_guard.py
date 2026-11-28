"""Pins r8_repoint.patch_seam_guard: eval-only, the clamp's OUTPUT is bit-identical and only the fail-loud raise is gone.
MEASURED 2026-10-16: both step-5,000 seed-0 battery rolls died on `goal_sel-surface graft seam SATURATED` (batch 1-3)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r8_repoint  # noqa: E402

r8_repoint.configure_env()
from tanitad.refs import refc_select as SL  # noqa: E402

ORIG = getattr(SL.apply_seam_clamp, "_r8_orig", SL.apply_seam_clamp)
KW = dict(clamp=1.0, fail=1.5, fail_frac=0.75, surface="goal_sel")


def _saturated(seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, 64, generator=g)
    return base, 3.0 * torch.randn(1, 64, generator=g)          # graft norm ~3x the base: saturated


def test_original_guard_kills_a_batch1_roll_after_patience_windows():
    st = SL.SeamState()
    with pytest.raises(RuntimeError, match="SATURATED"):
        for i in range(5):
            ORIG(*_saturated(i), patience=5, state=st, **KW)


def test_patched_guard_is_output_identical_and_never_raises():
    r8_repoint.patch_seam_guard()
    assert SL.apply_seam_clamp._r8_patched
    st_a, st_b = SL.SeamState(), SL.SeamState()
    before = dict(r8_repoint.SEAM_TELE)
    for i in range(60):
        b, g = _saturated(i)
        out_p, tele_p = SL.apply_seam_clamp(b, g, patience=50, state=st_a, **KW)
        out_o, tele_o = ORIG(b, g, patience=0, state=st_b, **KW)
        assert torch.equal(out_p, out_o)                        # bit-identical
        assert tele_p == tele_o
    t = r8_repoint.SEAM_TELE
    assert t["forwards"] - before["forwards"] == 60
    assert t["saturated_forwards"] - before["saturated_forwards"] == 60
    assert t["fail_loud_suppressed"] - before["fail_loud_suppressed"] == 1   # would have raised once at window 50


def test_below_clamp_is_identity_and_not_counted_as_saturated():
    r8_repoint.patch_seam_guard()
    base = torch.ones(2, 8)
    graft = 0.1 * torch.ones(2, 8)
    before = r8_repoint.SEAM_TELE["saturated_forwards"]
    out, _ = SL.apply_seam_clamp(base, graft, patience=50, state=SL.SeamState(), **KW)
    assert torch.equal(out, base + graft)
    assert r8_repoint.SEAM_TELE["saturated_forwards"] == before
