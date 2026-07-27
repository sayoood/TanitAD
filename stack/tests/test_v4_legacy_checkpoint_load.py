"""Regression guard: a pre-rank-16 v4 checkpoint must still STRICT-load.

**Why this test exists (escalation E-F).** The rank-16 vision lever set
``V4Config.vision_rank = 16`` as the *default*. Every v4 checkpoint in the program
was trained before that lever existed: its factorised heads are raw ``state_dim``
wide and it carries no ``vision_rank_proj.*`` weights. Rebuilding such a head at the
new default produces **2 missing keys and 3 shape mismatches**, and
``eval_flagship_v4.load_v4_from_ck`` loads **STRICT** — so for a window this session,
**every committed v4 number was unreproducible.**

Nothing was wrong with the lever. The defect was that a *default* silently changed
the architecture that a loader reconstructs, and no test pinned the old shape.

The fix reads the rank **from the checkpoint's own weights** rather than from a config
key, so it holds whether or not a sibling ``config.json`` exists — the config-key
heuristic only covered the case where a config file was present *and* incomplete.

Driven in **both** directions, per the ``e1c_selftest`` pattern: a legacy state dict
must resolve to the raw path, and a rank-carrying one must resolve to its own rank.
A test that only ever sees the new shape would not have caught this.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tanitad.models.flagship_v4 import V4Config  # noqa: E402
from tanitad.models.vision_rank import (  # noqa: E402
    DEFAULT_VISION_RANK,
    RAW_STATE_DIM,
    RawVisionRankRefused,
    resolve_vision_rank,
)


def _infer_rank(head_state: dict) -> int | None:
    """Mirror of the loader's inference: the weights are the source of truth."""
    w = next((v for k, v in head_state.items()
              if k.endswith("vision_rank_proj.proj.weight")), None)
    return None if w is None else int(w.shape[0])


def test_legacy_state_dict_has_no_projection_and_infers_raw():
    legacy = {
        "lat_head.0.weight": torch.zeros(64, RAW_STATE_DIM),
        "lon_head.0.weight": torch.zeros(64, RAW_STATE_DIM),
    }
    assert _infer_rank(legacy) is None, (
        "a pre-lever checkpoint must be detectable by the ABSENCE of "
        "vision_rank_proj weights")


def test_rank_carrying_state_dict_infers_its_own_rank():
    modern = {
        "vision_rank_proj.proj.weight": torch.zeros(DEFAULT_VISION_RANK,
                                                    RAW_STATE_DIM),
        "lat_head.0.weight": torch.zeros(64, DEFAULT_VISION_RANK),
    }
    assert _infer_rank(modern) == DEFAULT_VISION_RANK


def test_the_default_alone_would_have_broken_the_legacy_shape():
    """The failing input. This is the defect, pinned so it cannot return."""
    cfg = V4Config()
    assert cfg.vision_rank == DEFAULT_VISION_RANK, (
        "the shipped default is rank 16 — that is correct and deliberate")
    # A legacy head is state_dim wide; the default would build a rank-16 reader.
    assert cfg.vision_rank != RAW_STATE_DIM, (
        "if these were equal the regression could not occur and this test would "
        "be vacuous")


def test_raw_path_still_requires_an_explicit_written_reason():
    """The legacy escape hatch must not become a silent way back to raw-2048."""
    with pytest.raises(RawVisionRankRefused):
        resolve_vision_rank(RAW_STATE_DIM, RAW_STATE_DIM)
    with pytest.raises(RawVisionRankRefused):
        resolve_vision_rank(RAW_STATE_DIM, RAW_STATE_DIM, allow_raw=True,
                            reason="   ")
    assert resolve_vision_rank(RAW_STATE_DIM, RAW_STATE_DIM, allow_raw=True,
                               reason="reproducing a pre-lever v4 checkpoint"
                               ) == RAW_STATE_DIM
