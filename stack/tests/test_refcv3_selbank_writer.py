"""refcv3_arm.py's nav-compliance sidecar writes ONE ``sel_bank_{cond}`` row per window (MM 4a3d05, 2026-11-19, hold_refcv3_selbank_writer).

MEASURED 2026-11-19 on E2_READ_K2DC/E2_K2_pf/dump_s0 (70 episodes, 2,397 windows): ``sel_bank_nav_true`` held 2,258 rows. ``sel_idx`` indexes
the refined candidate set (149 wide); ``anchor_bank`` is narrower (104..116); the writer's ``bank[si:si + 1]`` is an EMPTY slice for
``si >= len(bank)``, so those windows vanished and taniteval.nav_compliance crashed on every refcv9 E2 dump (STRATEGIC family UNAVAILABLE).

Pinned here: ``_sel_bank_row`` is ALWAYS ``[1, S, 2]`` -- the bank path inside the bank, a NaN row outside it (above AND below) -- and the
writer's call site uses it (the old slice expression is gone from the module). The mutation arm restores the old slice and must go RED.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_TOOLS = _REPO / "taniteval" / "tools"
sys.path.insert(0, str(_REPO / "stack" / "scripts"))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rc = _load("refcv3_arm_selbank_writer_test", _TOOLS / "refcv3_arm.py")
B, S = 7, 8
BANK = torch.arange(B * S * 2, dtype=torch.float32).reshape(B, S, 2)


def _check_rows(fn):
    for si in range(B):                                   # inside the bank: the bank path, exactly
        row = fn(BANK, si)
        assert row.shape == (1, S, 2)
        assert np.array_equal(row[0], BANK[si].numpy())
    for si in (B, B + 1, 148, -1):                        # outside: ONE row, all NaN (never an empty slice)
        row = fn(BANK, si)
        assert row.shape == (1, S, 2), (si, row.shape)
        assert np.isnan(row).all(), si


def test_sel_bank_row_is_one_row_per_window_nan_outside_the_bank():
    _check_rows(rc._sel_bank_row)


def test_the_writer_call_site_uses_the_helper_and_the_old_slice_is_gone():
    src = (_TOOLS / "refcv3_arm.py").read_text(encoding="utf-8")
    # the call site in _navcomp_row (the def line carries the same text, so it is subtracted; MM 11-20: the first form counted 2 == 1)
    assert src.count("_sel_bank_row(bank, si)") - src.count("def _sel_bank_row(bank, si)") == 1
    assert "bank[si:si + 1].cpu().numpy())" not in src                # the old per-window append is gone
    assert src.count("def _sel_bank_row(") == 1


def test_MUTATE_the_old_slice_turns_the_row_check_red():
    def old(bank, si):                                                 # the pre-fix writer expression
        return bank[si:si + 1].cpu().numpy()

    with pytest.raises(AssertionError):
        _check_rows(old)
    _check_rows(rc._sel_bank_row)                                      # restored: GREEN
