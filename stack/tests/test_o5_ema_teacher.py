"""The O5 EMA-teacher target (P0 prerequisite, PI-approved 2026-08-27).

⛔ LOAD-BEARING: `test_the_default_is_bit_identical` (no flag ⇒ nothing
constructed) and `test_resync_after_init_load_copies_the_live_weights` — without
the resync, an `--init-from` arm trains against a FROZEN RANDOM teacher and the
run looks healthy while its target is noise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_STACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

from train_v6_staged import (  # noqa: E402
    STAGE_MAY_INTRODUCE, _resync_ema_o5, build_parser, build_stack_from_args,
)

BASE = ["--out", "UNUSED", "--stage", "S-W", "--frame-h", "64",
        "--frame-w", "160", "--enc-dim", "32", "--enc-depth", "1",
        "--enc-heads", "2", "--readout-grid", "2", "--readout-grid-w", "4",
        "--readout-dim", "16", "--pred-dim", "32", "--pred-depth", "1",
        "--pred-heads", "2", "--window", "4", "--d-tac", "16", "--d-str", "8"]


def _stack(*extra):
    return build_stack_from_args(build_parser().parse_args(BASE + list(extra)))


def test_the_default_is_bit_identical():
    st = _stack()
    assert not hasattr(st, "ema_o5_enc")
    assert not any("ema_o5" in n for n, _ in st.named_parameters())


def test_ema_mode_attaches_frozen_copies_mapped_to_aux():
    st = _stack("--o5-target", "ema")
    assert hasattr(st, "ema_o5_enc") and hasattr(st, "ema_o5_ro")
    for n, p in st.named_parameters():
        if n.startswith(("ema_o5_enc.", "ema_o5_ro.")):
            assert not p.requires_grad
            assert st.group_of(n) == "aux"


def test_the_prefixes_are_S_W_introducible():
    for pref in ("ema_o5_enc.", "ema_o5_ro."):
        assert pref in STAGE_MAY_INTRODUCE["S-W"]


def test_resync_after_init_load_copies_the_live_weights():
    st = _stack("--o5-target", "ema")
    # simulate an init-from load: perturb the LIVE encoder after construction
    with torch.no_grad():
        for p in st.encoder.parameters():
            p.add_(1.0)
    lp = next(st.encoder.parameters())
    tp = next(st.ema_o5_enc.module.parameters())
    assert not torch.allclose(lp, tp)          # teacher is stale (the bug)
    _resync_ema_o5(st)
    tp = next(st.ema_o5_enc.module.parameters())
    assert torch.allclose(lp, tp)              # resync fixed it


def test_update_moves_the_teacher_toward_the_live_weights():
    st = _stack("--o5-target", "ema")
    _resync_ema_o5(st)
    with torch.no_grad():
        for p in st.encoder.parameters():
            p.add_(1.0)
    before = next(st.ema_o5_enc.module.parameters()).clone()
    st.ema_o5_enc.update(st.encoder)
    after = next(st.ema_o5_enc.module.parameters())
    live = next(st.encoder.parameters())
    # moved toward live, but not all the way (decay < 1)
    assert (after - before).abs().max() > 0
    assert (live - after).abs().max() > 0


def test_teacher_forward_is_no_grad():
    st = _stack("--o5-target", "ema")
    x = torch.randn(2, 9, 64, 160)
    out = st.ema_o5_ro.module(st.ema_o5_enc.module(x))
    assert out.grad_fn is None or not out.requires_grad
