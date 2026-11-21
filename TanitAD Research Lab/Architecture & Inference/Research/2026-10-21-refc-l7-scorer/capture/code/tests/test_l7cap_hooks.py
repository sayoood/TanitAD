"""Tests of the capture hooks on the REAL `RefCV3Model` code path (the stack's own tiny refcv8 rig, CPU) and on the SAM3 crop.

Expectations are LITERALS or the test's own independent derivation (a forward pre-hook on the CONSUMER of the tokens), never an expression
over the code under test. Each hook claim ships its deliberate-regression arm.

    PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval' python tests/test_l7cap_hooks.py     (or pytest)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "C:/lgs/Lfb47a58376/stack/tests")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _refcv8_rig as R  # noqa: E402
import l7cap_hooks as H  # noqa: E402
from tanitad.refs import refc_v3 as RV3  # noqa: E402

KEYS = ("traj", "sel_idx", "anchor_traj", "sel_score", "sel_score_v3", "sel_idx_base")


def _rig():
    T = R.trainer()
    cfg, m = R.build(T, True, n_alloc=4, vmax=False)
    return cfg, m, R.batch(cfg)


def _consumer_tap(m):
    """INDEPENDENT derivation of the scene sources: a forward pre-hook on the module that CONSUMES them (the tactical decoder)."""
    seen = {}

    def pre(mod, args, kwargs):
        for k in ("agent_tokens", "agent_pad", "bev_tokens", "bev_pad"):
            if kwargs.get(k) is not None:
                seen[k] = kwargs[k].detach().clone()
        return None
    h = m.tac_decoder_v6.register_forward_pre_hook(pre, with_kwargs=True)
    return seen, h


def test_identity_and_sources():
    cfg, m, bt = _rig()
    base = R.forward(cfg, m, bt)
    base2 = R.forward(cfg, m, bt)
    assert all(torch.equal(base[k], base2[k]) for k in KEYS), "rig is non-deterministic: identity cannot be read"
    assert "r7_bev_tokens" not in base, "refcv8 stores no r7 sources by itself (refcv7_wta is None) -- premise of the patch"
    orig = H.install_scene_capture(RV3, torch, mutation="")
    try:
        seen, h = _consumer_tap(m)
        o = R.forward(cfg, m, bt)
        h.remove()
        # (a) identity: every emitted key bit-identical to the unpatched forward
        for k in KEYS:
            assert torch.equal(base[k], o[k]), f"capture changed emitted key {k}"
        # (b) the sources are in `out` and EQUAL what the consumer actually received
        assert "bev_tokens" in seen and "agent_tokens" in seen, "tap saw no tokens: the rig's decoder is not reached"
        assert torch.equal(o["r7_bev_tokens"], seen["bev_tokens"])
        assert torch.equal(o["r7_agent_tokens"], seen["agent_tokens"])
        for p in ("bev_pad", "agent_pad"):
            if p in seen:
                assert torch.equal(o["r7_" + p], seen[p])
        # (c) detached copies, never the live tensor
        assert not o["r7_bev_tokens"].requires_grad
        # (d) the non-admissible tensors are NOT stored by the hook
        assert "r7_cond" not in o and "r7_v_lim" not in o
    finally:
        H.uninstall_scene_capture(RV3)
    assert "r7_bev_tokens" not in R.forward(cfg, m, bt), "uninstall must restore the original hook"
    assert orig is not None


def test_mutation_swap_sources_is_red():
    """Deliberate regression: a hook that stores the agent tokens as the BEV tokens must FAIL the consumer-equality assertion."""
    cfg, m, bt = _rig()
    H.install_scene_capture(RV3, torch, mutation="swap_sources")
    try:
        seen, h = _consumer_tap(m)
        o = R.forward(cfg, m, bt)
        h.remove()
        same = (o["r7_bev_tokens"].shape == seen["bev_tokens"].shape and torch.equal(o["r7_bev_tokens"], seen["bev_tokens"]))
        assert not same, "the swap mutant was NOT detected by the consumer-equality control"
    finally:
        H.uninstall_scene_capture(RV3)


def test_forbidden_keys_refused():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import fan_roll_r8_l7cap as C
    for k in ("cond", "r7_cond", "v_max_ms", "nav_cmd", "r7_agent_fut", "z_tac"):
        try:
            C.assert_no_forbidden(["anchor_traj", k])
        except SystemExit:
            continue
        raise AssertionError(f"{k} was accepted into an input bank")
    C.assert_no_forbidden(["anchor_traj", "sel_score", "r7_bev_tokens", "r7_agent_pad", "is_alloc", "e9_keep"])   # the admissible set passes


def test_fp16_guard():
    x = np.array([0.5, -3.25, 1e-3], np.float32)
    y, i = H.to_fp16_checked(x)
    assert y.dtype == np.float16 and i["stored"] == "float16"
    big, i2 = H.to_fp16_checked(np.array([7.0e4, 1.0], np.float32))
    assert big.dtype == np.float32 and i2["stored"] == "float32", "an fp16-overflowing array must stay float32"


def test_crop_drivable_literals():
    from tanitad.data import semantic_map_gt_fine as F
    outer = F.EXTENT_REFCV7
    assert tuple(outer.fine_shape) == (1000, 600)            # literal from semantic_map_gt_fine.py:298
    inner = F.EXTENT_V2
    r0, r1, c0, c1 = inner.window_in(outer)
    assert (r0, r1, c0, c1) == (0, 600, 140, 460), (r0, r1, c0, c1)       # (30 - 16) m / 0.1 = 140 columns in
    codes = np.full((1, 1000, 600), 0, np.uint8)            # class 0 everywhere (seen, "no map class")
    codes[0, 0:5, c0:c0 + 5] = 1                            # the whole 5x5 block of coarse cell (0, 0) is drivable
    codes[0, 5:7, c0 + 5:c0 + 10] = 1                       # 2 rows x 5 = 10 cells ...
    codes[0, 7, c0 + 5:c0 + 8] = 1                          # ... + 3 = 13 of 25 cells of coarse cell (1, 1) drivable
    codes[0, 10:15, c0:c0 + 5] = 255                        # coarse cell (2, 0) not seen
    codes[0, 0:5, c0 - 5:c0] = 1                            # OUTSIDE the /2 window (column 139 and left): must not appear
    drv, seen = H.crop_drivable(codes, outer)
    assert drv.shape == (1, 120, 64) and seen.shape == (1, 120, 64)
    assert int(drv[0, 0, 0]) == 255
    assert int(drv[0, 1, 1]) == 133 == int(np.rint(13 * 255 / 25))          # literal rint(13*255/25)
    assert int(drv[0, 0, 1]) == 0
    assert bool(seen[0, 2, 0]) is False and bool(seen[0, 0, 0]) is True
    assert int(drv[0].sum()) == 255 + 133, "mass outside the /2 window leaked into the crop"


def test_crop_mutation_wrong_window_is_red():
    """Deliberate regression: cropping at column 0 instead of the centred 140 moves the block; the literal assertions must FAIL."""
    from tanitad.data import semantic_map_gt_fine as F
    codes = np.zeros((1, 1000, 600), np.uint8)
    codes[0, 0:5, 140:145] = 1
    bad = F.block_fraction_u8(codes[:, 0:600, 0:320])[:, 1]          # the mutant crop (col 0, not col 140)
    assert not (int(bad[0, 0, 0]) == 255), "the wrong-window mutant reads the right value: the crop literal cannot discriminate"


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("PASS", name)
    print("ZZL7CHOOKSPASS", n)
