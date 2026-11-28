"""Literal tests for SPEC AMENDMENT A5 in `g0_refcv7.py` (no GPU, no data).

    PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" python -m pytest -q test_g0_a5.py

Every expectation is a LITERAL. Each guard carries a deliberate-regression arm that must go RED.
"""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _rec():
    return {"model": {"state_dict": {"missing": [], "unexpected": []},
                      "param_breakdown": {"equal": True}, "anchor_file_vs_ckpt_buffers": {},
                      "declared_vs_built": {"mismatches": []}},
            "wrapper_control": {"clause": "PASS"},
            "mutations": {"m1": {"row": {"eval_traj": 9.0}}}}


def _by_seed(vals):
    return {s: {"row": {"eval_traj": v}, "buffers_unchanged": True} for s, v in enumerate(vals)}


def test_t995_literals():
    import g0_refcv7 as G
    assert G.T_995[8] == 3.499
    assert G.T_995[24] == 2.807          # t(0.995, 23), SPEC A5 item 1
    assert G.REGISTERED_SEEDS == (0, 1, 2, 3, 4, 5, 6, 7)
    assert G.A5_MIN_SEEDS == 24


def test_a5_interval_literal():
    """24 seed means alternating 1.00 / 1.02: mean 1.01, sd 0.0102151... -> half-width
    2.807 * 0.0102151 * sqrt(1 + 1/24) + 0.0101 = 0.0393650 (computed once, written as a literal)."""
    import g0_refcv7 as G
    vals = [1.00, 1.02] * 12
    v = G.judge({"eval_traj": 1.049}, _by_seed(vals), _rec(), amend="A5")
    r = v["terms"]["eval_traj"]
    assert r["cls"] == "STOCHASTIC"
    assert abs(r["pi_hi"] - 1.0493650) < 1e-6
    assert r["verdict"] == "OK"
    v2 = G.judge({"eval_traj": 1.0500}, _by_seed(vals), _rec(), amend="A5")
    assert v2["terms"]["eval_traj"]["verdict"] == "OUT"


def test_unregistered_K_refuses_rather_than_defaulting():
    """Regression arm: the old code fell back to t = 3.499 for any K not in the table."""
    import g0_refcv7 as G
    with pytest.raises(ValueError):
        G.judge({"eval_traj": 1.0}, _by_seed([1.0, 1.01, 1.02] * 3 + [1.0, 1.01]), _rec(), amend="A5")


def test_mmap_batches_bit_identical_and_lawonly_roundtrip():
    import torch
    import g0_refcv7 as G
    g = torch.Generator().manual_seed(0)
    eb = {"frames": torch.randint(0, 255, (3, 2, 9, 4, 5), generator=g, dtype=torch.uint8),
          "pose_last": torch.randn(3, 4, generator=g),
          "future_frames": G.LawOnly(torch.randint(0, 255, (3, 9, 4, 5), generator=g,
                                                   dtype=torch.uint8), 4),
          "names": ["a", "b", "c"]}
    with tempfile.TemporaryDirectory() as d:
        mb = G.MmapBatches(Path(d) / "c", [lambda: eb])
        back = mb[0]
        assert torch.equal(back["frames"], eb["frames"])
        assert torch.equal(back["pose_last"], eb["pose_last"])
        assert isinstance(back["future_frames"], G.LawOnly)
        assert back["future_frames"].law_idx == 4
        assert torch.equal(back["future_frames"].u8, eb["future_frames"].u8)
        assert back["names"] == ["a", "b", "c"]
        assert len(list(iter(mb))) == 1
        # deliberate regression: a cache holding OTHER content must not compare equal. (A mapped
        # file cannot be overwritten on Windows -- error 1224 -- so the regression gets its own dir.)
        bad = G.MmapBatches(Path(d) / "bad", [lambda: {"frames": eb["frames"] + 1}])
        assert not torch.equal(bad[0]["frames"], eb["frames"])
        # an EXISTING batch file is reused, never re-made (a crashed G0 re-collates nothing)
        again = G.MmapBatches(Path(d) / "c", [lambda: (_ for _ in ()).throw(AssertionError("re-made"))])
        assert torch.equal(again[0]["frames"], eb["frames"])
        del back
