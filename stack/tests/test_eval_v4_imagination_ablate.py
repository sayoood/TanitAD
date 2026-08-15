"""I4a instrument (`eval_flagship_v4 --imagination-ablate`) — the ablation must
actually break the input, deterministically, and 'none' must be a byte-identical
no-op (an ablated run mislabelled intact would poison every imagination claim).
"""
import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import eval_flagship_v4 as E                                        # noqa: E402


@pytest.fixture()
def imag():
    g = torch.Generator().manual_seed(0)
    return torch.randn(4, 32, 16, generator=g)


def test_none_is_identity_same_object(imag):
    assert E.ablate_imagined(imag, "none") is imag


def test_zero_removes_content_keeps_shape(imag):
    out = E.ablate_imagined(imag, "zero")
    assert out.shape == imag.shape
    assert out.abs().sum() == 0
    assert imag.abs().sum() > 0            # input not mutated in place


def test_shuffle_keeps_marginals_breaks_correspondence(imag):
    out = E.ablate_imagined(imag, "shuffle")
    # marginal statistics preserved: same multiset of per-window blocks
    assert torch.equal(out.sort(dim=0).values, imag.sort(dim=0).values)
    # correspondence broken for EVERY row (roll-by-1 has no fixed points, B>=2)
    assert not torch.equal(out, imag)
    for b in range(imag.shape[0]):
        assert not torch.equal(out[b], imag[b])


def test_shuffle_is_deterministic(imag):
    assert torch.equal(E.ablate_imagined(imag, "shuffle"),
                       E.ablate_imagined(imag, "shuffle"))


def test_unknown_mode_raises(imag):
    with pytest.raises(ValueError):
        E.ablate_imagined(imag, "dropout")


def test_cli_default_is_none_and_threaded():
    # default must stay 'none' — an ablated default would corrupt every quoted run
    sig = inspect.signature(E.collect_planner)
    assert sig.parameters["imagination_ablate"].default == "none"
    src = inspect.getsource(E.main)
    assert "--imagination-ablate" in src
    assert "imagination_ablate=a.imagination_ablate" in src


def test_ablation_applied_after_imagination_inputs():
    # the ablation must operate on the REAL imagined tensor (post-roll), never
    # replace the feed itself — grep the collect_planner source for the order
    src = inspect.getsource(E.collect_planner)
    i_feed = src.index("_imagination_inputs(world, head.cfg, b, st")
    i_abl = src.index("ablate_imagined(goal_kw[\"imagined\"]")
    assert i_feed < i_abl
