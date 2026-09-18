"""The two trunk memory levers: they work, they refuse, and OFF changes nothing.

⛔ WHY THEY EXIST. MEASURED 2026-09-18/19 on the real rig: `resnet101.a1_in1k` at
416 x 1024 batch 1 OOMs at **22.34 GB** unpatched and runs at **2.887 GB / 4.2-4.9 s/step**
with `chunk_ckpt=1` + `frozen_bn` — 1.96-2.28 days for a full 40,284-step arm on an 8 GB
card. Without them §10.2's PRIMARY trunk cannot be trained here at all.

⚠️ `--arm hier` calls the trunk on `frames.reshape(b*w, ...)` with window 8, so at batch 1
the backbone sees **24 images, not 3**. That is why this lever is the one that matters, and
why a per-sample intuition misprices it.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from tanitad.models.timm_trunk import build_timm_trunk


def _mk(**kw):
    """A tiny CPU trunk. `resnet18`, no download — the levers are architecture-agnostic
    and this file is about their WIRING, not about a particular backbone's memory."""
    return build_timm_trunk(in_channels=9, image_hw=(64, 128),
                            model_name="resnet18.a1_in1k", pretrained=False, **kw)


# ------------------------------------------------------------------ OFF
def test_the_default_applies_NOTHING():
    """⛔ The parity requirement. A memory flag that perturbs the default path has failed
    regardless of what it enables — every banked arm was trained without these."""
    t = _mk()
    assert t.memory_levers == {"chunk_ckpt": 0, "frozen_bn": False,
                               "bn_pinned": 0, "relu_out_of_place": 0}


def test_SAME_BREATH_CONTROL_bn_really_does_train_by_default():
    """⚠️ Without this, the frozen-BN test below is vacuous: it would pass on a model
    whose BatchNorms were never in training mode to begin with."""
    t = _mk()
    t.train()
    assert t.bn_training_count() > 0


# ------------------------------------------------------------------ ON
def test_frozen_bn_SURVIVES_the_trainers_own_train_call():
    """⛔ THE WHOLE HAZARD. `nn.Module.train()` RECURSES into children and the trainer
    calls `model.train()` every step, so a plain `.eval()` is silently undone on step 1 and
    the run REPORTS a frozen-BN arm while training BN on the batch."""
    t = _mk(frozen_bn=True)
    assert t.memory_levers["bn_pinned"] > 0
    t.train()                      # what the trainer does, every step
    assert t.bn_training_count() == 0


def test_chunking_makes_the_relus_out_of_place():
    """A checkpointed segment runs TWICE; an in-place ReLU then trips autograd's version
    counter on the second pass. The lever's real price includes this."""
    t = _mk(chunk_ckpt=1, frozen_bn=True)
    assert t.memory_levers["relu_out_of_place"] > 0
    assert t.memory_levers["chunk_ckpt"] == 1


# ------------------------------------------------------------------ the refusal
def test_chunking_WITHOUT_frozen_bn_is_REFUSED():
    """⛔ Chunking alone changes BatchNorm — each chunk is a different sub-batch — and
    MEASURED a **-40 %** shift in `ga_trunk` on resnet34. A run that believed it had only
    saved memory would be compared against arms it no longer matches, so this is a refusal
    and not a warning."""
    with pytest.raises(ValueError) as ex:
        _mk(chunk_ckpt=1)
    assert "frozen_bn" in str(ex.value)


# ------------------------------------------------------------------ exactness
def test_chunked_and_unchunked_AGREE_with_bn_frozen():
    """⭐ THE PROPERTY THAT MAKES THE LEVER ADMISSIBLE AT ALL. With BN pinned, chunking is
    pure recomputation: same weights, same dtype, same frames, one extra forward. If this
    drifts, the lever is a different ARM and every number measured under it is
    incomparable — which is exactly what chunking WITHOUT frozen BN does."""
    torch.manual_seed(0)
    a = _mk(frozen_bn=True).eval()
    torch.manual_seed(0)
    b = _mk(chunk_ckpt=1, frozen_bn=True).eval()
    b.load_state_dict(a.state_dict())          # identical weights, not merely same seed
    x = torch.rand(2, 9, 64, 128)
    with torch.no_grad():
        s16a, s32a, pa = a.forward_features(x)
        s16b, s32b, pb = b.forward_features(x)
    for u, v, name in ((s16a, s16b, "s16"), (s32a, s32b, "s32"), (pa, pb, "pooled")):
        assert torch.allclose(u, v, atol=1e-4, rtol=1e-4), (
            f"{name} drifted: max |d| = {float((u - v).abs().max())}")


# ------------------------------------------------------------------ the real bug
def test_the_levers_SURVIVE_a_config_round_trip():
    """⛔ THE DEFECT THIS FILE WAS WRITTEN AFTER, and it cost a live run.

    They were first wired as ad-hoc ATTRIBUTES on `CNNEncoderConfig`. `setattr` succeeds on
    that dataclass, so everything LOOKED right — and the trunk still OOM'd at 22.34 GB,
    because a non-field does not survive the config's own round-trip. A lever that can be
    silently dropped between the CLI and the model is worse than no lever: the run reports
    the flag and trains the other arm.
    """
    from tanitad.refs.refc import CNNEncoderConfig
    names = {f.name for f in dataclasses.fields(CNNEncoderConfig)}
    assert {"trunk_chunk_ckpt", "trunk_frozen_bn"} <= names, (
        "the levers must be real dataclass FIELDS, not attributes set from outside")
    c = dataclasses.replace(CNNEncoderConfig(), trunk_chunk_ckpt=1, trunk_frozen_bn=True)
    assert c.trunk_chunk_ckpt == 1 and c.trunk_frozen_bn is True


def test_the_levers_DO_NOT_CHANGE_THE_STATE_DICT():
    """⛔ CHECKPOINT INTEROPERABILITY, and this one really bit.

    The first implementation wrapped `self.net` in a module, which renamed every backbone
    key (`net.conv1.weight` -> `net.net.conv1.weight`). A checkpoint written by a levered
    run then could not be loaded by an unlevered model, by an eval driver, or by
    `--init-from` — and the failure would surface days later, at LOAD time, on an arm
    already paid for. Chunking the CALL instead of the MODULE keeps the tree identical.
    """
    off = set(_mk().state_dict())
    on = set(_mk(chunk_ckpt=1, frozen_bn=True).state_dict())
    assert off == on, (
        f"state_dict keys moved: only-off={sorted(off - on)[:4]} "
        f"only-on={sorted(on - off)[:4]}")
