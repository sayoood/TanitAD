"""refcv6 §2 — the image trunk: a ``timm`` ImageNet backbone over FRAME HISTORY.

⭐ **Why a pretrained trunk at all.** REF-C is the outlier: every image-trunk
initialisation stated in the end-to-end driving literature is *pretrained*
(TransFuser, UniAD, SparseDrive, Hydra-MDP, DriveTransformer, DriveVLA-W0),
REF-C's 90.5 M trunk is **random**, and an ImageNet-initialised REF-C arm has
**never been run** (`REFCV6_CLARIFICATION.md` §6.2, §6.3 finding 5).

⭐⭐ **PI CORRECTION 2026-09-16 — frame history is REQUIRED**, verbatim:
*"I think it is important to process the image frame history and also ego data
history as inputs for our refcv6"*. So the default is **not** DiffusionDrive's
single frame. The default is:

    K history frames -> the SAME trunk weights, K separate 3-channel passes
                     -> temporal fusion AFTER the trunk, at BOTH strides

⛔ **The K-pass construction is what keeps the ImageNet prior EXACT.** The stem
sees three channels with ImageNet mean/std — precisely the input distribution
its weights were fitted on. The 9-channel stem-inflated variant (one pass, a
3K-channel stem whose weights are repeated and divided by K) is retained as the
**cheaper alternative arm**, not as the default.

⚠️ **K shared passes are cheaper than today's arm for resnet34 and dearer for
resnet101** — MEASURED, not argued; the table below carries both numbers.
refcv5-v2 runs its 90.5 M trunk on all **8 window positions with gradient**
(`refc.py::RefCModel.forward`, the `cfg.hierarchy` branch), which is the bar.

⭐⭐ **The backbone is a NAME, not a hard-code** (PI 2026-09-16).
``TimmTrunkConfig.model_name`` is resolved through ``timm`` and every channel
count is read from ``feature_info``, so resnet34 -> resnet101 -> convnext swaps
need no model-code change. The PI's choice: **primary `resnet101.a1_in1k`**
(the default here), second comparison run **`resnet34.a1_in1k`** (the papers' /
DiffusionDrive's own), selected by ``--trunk-name``. ⚠️ Changing this default
perturbs nothing that exists: ``CNNEncoderConfig.trunk`` is ``"refc"`` by
default, so no pre-refcv6 arm reaches this module at all.

PRIMARY SOURCES, and the code wins where it disagrees with the prose
============================================================================
* backbone id — DiffusionDrive ``diffusiondrivev2_rl_config.py:16-18``:
  ``image_architecture: str = "resnet34"`` with
  ``bkb_path = "ckpts/resnet34.a1_in1k/pytorch_model.bin"`` — the *timm*
  ``a1_in1k`` recipe weight, not torchvision's.
* optimiser — ``diffusiondrivev2_rl_config.py:119-131``: ``AdamW``,
  ``weight_decay = 1e-4``, ``opt_paramwise_cfg`` giving the ``image_encoder``
  an ``lr_mult`` of ``cfg_lr_mult = 0.5``. See :func:`param_groups_dd`.
* ImageNet mean/std — ``E-SEED-2`` measured that skipping the normalisation
  wastes the prior, so it is applied **inside** the trunk and **counted**.

SHAPES AND COST (all MEASURED 2026-09-16 on this box, ``timm`` 1.0.29)
============================================================================
⭐ PI 2026-09-16: the input geometry is **256 x 1024** for all future training,
and the primary backbone is **resnet101** with **resnet34** as the second
comparison run. ``out_indices=(3, 4)`` -> ``reduction [16, 32]``; the CHANNELS
come from ``feature_info`` and are never written down:

=============  ==============  ==============  ===========  ==============
backbone       stride-16       stride-32       backbone P   fwd s/window
=============  ==============  ==============  ===========  ==============
resnet34       256 x 16 x 64   512 x 8 x 32    21,284,672   0.200  (K=3)
resnet101      1024 x 16 x 64  2048 x 8 x 32   42,500,160   0.569  (K=3)
=============  ==============  ==============  ===========  ==============

* **stride-16** is the PERCEPTION map (BEV lift, MAP head, BOX head). An
  oracle on the stride-32 grid tops out at AP 0.3341 against 0.4713 on
  stride-16, which is why perception may not read the stride-32 map.
* **stride-32** is the PLANNER map — **256 tokens** at 1024 px (160 at 640),
  the shape ``AnchoredDiffusionDecoder.feat_proj`` already consumes.

⚠️ **Against today's arm** (refcv5-v2's 90,458,632-parameter trunk, run on all
8 window positions with gradient at 256x640): **1.283 s/window** and
**1,665 MB** of fp32 activations, MEASURED the same way. So resnet34 K=3 at
the NEW geometry is 6.4x faster and needs 1.9x LESS activation memory;
resnet101 K=3 is 2.3x faster but needs 2.0x MORE (3,257 MB/window).

⛔ THE WEIGHTS ARE REAL OR THE BUILD FAILS. ``pretrained=True`` that quietly
falls back to a random init is the worst failure this module can have: the arm
would train, converge, and mean nothing. :func:`_assert_pretrained_loaded`
verifies the built stem against the CHECKPOINT ON DISK (model-agnostic), with a
pinned statistic for the default as a second, cheaper witness.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

__all__ = [
    "IMAGENET_MEAN", "IMAGENET_STD", "IMAGENET_CONV1_ABS_SUM",
    "TimmTrunkConfig", "TimmResNetTrunk", "inflate_stem_",
    "param_groups_dd", "build_timm_trunk", "TemporalFuse",
    "DD_ENCODER_LR_MULT", "DD_WEIGHT_DECAY", "CANDIDATE_BACKBONES",
    "trunk_feature_channels",
]


@functools.lru_cache(maxsize=16)
def trunk_feature_channels(model_name: str,
                           out_indices: tuple[int, ...] = (3, 4)
                           ) -> tuple[int, ...]:
    """``(stride-16 channels, stride-32 channels)`` for a timm backbone.

    ⛔⛔ **THIS IS WHY NOTHING HARD-CODES 512.** resnet34 emits 256 / 512 and
    resnet101 emits **1024 / 2048** (MEASURED 2026-09-16). A config that
    answered 512 for the PI's primary trunk would build the decoder's
    ``feat_proj`` 1536 channels too narrow and the failure would land far from
    here. The answer comes from ``feature_info``, and the model is built
    weightless and cached so asking is cheap.
    """
    import timm
    net = timm.create_model(str(model_name), pretrained=False,
                            features_only=True, out_indices=tuple(out_indices))
    return tuple(int(c) for c in net.feature_info.channels())

# timm's ``resnet34.a1_in1k`` default_cfg mean/std (== the ImageNet constants).
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: ⭐ THE PROOF THAT THE PRIOR IS REAL, for the DEFAULT backbone. MEASURED
#: 2026-09-16 on the downloaded ``timm/resnet34.a1_in1k``
#: ``model.safetensors`` (87,278,522 B, md5 69731b23dc8f6e60d8fcb02478a7a1e4):
#: ``conv1.weight`` (64, 3, 7, 7), mean -0.0028060986660420895,
#: std 0.23252105712890625, ``|w|.sum() = 1279.853759765625``.
#: A He-initialised (64, 3, 7, 7) conv has std sqrt(2 / 147) = 0.1166 — HALF —
#: and ``|w|.sum()`` near 875. The tolerance is wide enough to survive a dtype
#: change and far too narrow for a random init.
IMAGENET_CONV1_ABS_SUM: float = 1279.853759765625
_CONV1_ABS_SUM_RTOL: float = 0.02
#: Per-backbone ``|stem.w|.sum()``, each MEASURED on this box from the real
#: download. ⛔ A backbone with no entry is verified against the CHECKPOINT ON
#: DISK instead (:func:`_reference_stem_weight`); it is never waved through.
_PINNED_STATS: dict[str, float] = {
    "resnet34.a1_in1k": IMAGENET_CONV1_ABS_SUM,     # MEASURED 2026-09-16
    "resnet101.a1_in1k": 1402.778076171875,         # MEASURED 2026-09-16
}

#: The backbones the PI chose between. ⭐ PI 2026-09-16: **primary
#: `resnet101.a1_in1k`**, with `resnet34.a1_in1k` — DiffusionDrive's own — as
#: the second comparison run. Listed so the report and the probe read ONE
#: source. ⚠️ Presence here is not a claim that the weights are cached;
#: :func:`build_timm_trunk` downloads on demand.
CANDIDATE_BACKBONES: tuple[str, ...] = (
    "resnet101.a1_in1k",       # PI 2026-09-16: the PRIMARY
    "resnet34.a1_in1k",        # the paper's / DiffusionDrive's own
    "resnet50.a1_in1k",
    "convnext_tiny.fb_in1k",
    "convnext_small.fb_in1k",
)


@dataclass
class TimmTrunkConfig:
    """The refcv6 trunk.

    ``frames`` (K) and ``mode`` are the two knobs the PI's correction turns on:

    * ``mode="shared"`` (**DEFAULT**) — K separate 3-channel passes through the
      SAME weights, fused after the trunk. The ImageNet prior is exact because
      the stem sees exactly 3 ImageNet-normalised channels.
    * ``mode="inflate"`` — ONE pass with a 3K-channel stem whose ImageNet
      weights are repeated and divided by K (:func:`inflate_stem_`). Cheaper by
      a factor of K in the trunk, and it reproduces the 3-channel response
      **exactly** on a repeated-frame input — which is what makes it a knockout
      of the *input* rather than a different network.
    * ``frames=1`` recovers DiffusionDrive's single frame in either mode.
    """

    #: ⛔ MEMORY LEVERS — both OFF by default, so the default path is unchanged.
    #: `chunk_ckpt = N` recomputes the backbone in leading-batch slices of N
    #: (`torch.utils.checkpoint`, non-reentrant). MEASURED 2026-09-18: resnet101 at
    #: 416x1024 batch 1 goes 22.34 GB (OOM) -> 2.887 GB, 4.2-4.9 s/step on an 8 GB card.
    #: ⚠️ `--arm hier` calls the trunk on `frames.reshape(b*w, ...)` with window 8, so
    #: at batch 1 the backbone sees **24 images, not 3** — which is why this lever is
    #: the one that matters and why a per-sample intuition misprices it.
    chunk_ckpt: int = 0
    #: ⛔ Pin every backbone BatchNorm to its ImageNet running statistics. REQUIRED
    #: with `chunk_ckpt`: chunking alone changes BN and MEASURED a -40 % shift in
    #: `ga_trunk` on resnet34 — a config that fits but is a DIFFERENT ARM. With BN
    #: frozen, chunked and unchunked agree to 7.2e-6.
    frozen_bn: bool = False
    #: ⛔ SPEED LEVERS — both OFF by default, and with both off `_backbone` runs the exact
    #: pre-lever code path. MEASURED 2026-09-23 (torch.profiler, Thor, resnet101 at
    #: 416x1024, batch 8, chunk 8): the step is GPU-BOUND and ~two thirds of GPU time is
    #: MEMORY-BOUND elementwise work (BN 23 %, residual adds 14 %, ReLU 17 %) plus 12 % in
    #: NCHW<->NHWC layout conversions around every convolution.
    #: `bf16`: run the BACKBONE ONLY under bf16 autocast (halves the bytes those ops move;
    #: outputs are cast back to float32, so the fusion, decoders, trajectory integration
    #: and every loss stay in fp32). `channels_last`: keep the backbone in NHWC, removing
    #: the layout conversions cuDNN otherwise inserts. Neither changes a weight or a frame.
    bf16: bool = False
    channels_last: bool = False
    #: fold each FROZEN BatchNorm into its conv (the same function, no separate BN pass);
    #: requires ``frozen_bn``. See :func:`_fold_frozen_bn_`.
    fold_bn: bool = False
    #: ⭐ SPEED, EXACT: compute each DISTINCT frame of overlapping K-stacks ONCE. A D-015 row
    #: stacks raw frames (j, ..., j+K-1), so row i+1 repeats row i's newest K-1 frames: a
    #: window of W rows holds W+K-1 distinct frames, not W*K (refcv6: 10, not 24). With BN
    #: frozen, a frame's features depend on that frame alone, so computing it once and
    #: GATHERING it into every slot is the same function with the same gradients. Verified
    #: per batch on the data, never assumed. Requires ``frozen_bn``; "shared" mode, K >= 2.
    #: See :meth:`TimmResNetTrunk._backbone_dedup`.
    dedup_frames: bool = False
    #: ⭐ SPEED: the backbone through ``torch.compile``. MEASURED 2026-09-23 on Thor (resnet101,
    #: 416x1024, bf16 + NHWC, frozen + folded BN, chunk 8): fwd+bwd 1.53x faster, and CLOSER to
    #: strict fp32 than the eager bf16 path (1.75 % / 4.67 % vs 2.08 % / 5.74 %) -- Inductor keeps
    #: fused intermediates in fp32. The compiled callable is held OUTSIDE the module tree, so
    #: every state_dict key is unchanged. ``compile_backend`` exists for CI (the dev box has no
    #: Triton); a run uses "inductor".
    compile_backbone: bool = False
    compile_backend: str = "inductor"
    #: ⭐ PI 2026-09-16: the PRIMARY. `resnet34.a1_in1k` — the papers' and
    #: DiffusionDrive's own — is the second comparison run, selected by name.
    model_name: str = "resnet101.a1_in1k"
    pretrained: bool = True
    frames: int = 3                    # K — the v2ep stack is 3 at ~10 Hz
    mode: str = "shared"               # "shared" (K passes) | "inflate" (1)
    #: ⚠️ ``concat1x1`` costs ``K*C -> C`` parameters at BOTH strides. On
    #: resnet34 (256/512) that is 0.98 M; on resnet101 (1024/2048) it is
    #: **15.7 M**, a third again on top of a 42.5 M backbone (MEASURED
    #: 2026-09-16). ``attn`` costs ``K*C -> K`` and is ~3 k. Both are
    #: identity-initialised, so they are directly comparable arms.
    fuse: str = "concat1x1"            # "concat1x1" | "attn" | "last"
    #: ⭐ Initialise the fusion so a FRESH multi-frame trunk reproduces the
    #: single-frame trunk exactly (all weight on the NEWEST frame). That makes
    #: history a *removable graft*: any later difference is learned, not an
    #: artefact of adding parameters. Set False for a plain init.
    fuse_identity_init: bool = True
    #: ⭐⭐ PI 2026-09-16: **256 x 1024 cylindrical, same 120 deg field**, for
    #: ALL future training (``f_ref`` 305.577 -> 488.92). That is 1.875 deg per
    #: stride-16 column against today's 3.0, and 3.75 deg per stride-32 column
    #: against 6.0. ⛔ NOTHING in this module hard-codes 40, 20, 64 or 32: the
    #: grids are ``h // 16, w // 16`` and ``h // 32, w // 32``, and the CHANNEL
    #: counts come from timm's ``feature_info``.
    image_hw: tuple[int, int] = (256, 1024)
    imagenet_norm: bool = True
    #: ⛔ C26 -- the RIG-CORRELATED BLACK STRIP. MEASURED 2026-09-23 over all 4,713
    #: clips of the 416x1024 B1 cache: **2,721 (57.73 %)** carry 26-43 fully-black bottom
    #: rows and 1,992 carry ZERO -- bimodal, nothing between. A strip whose presence
    #: identifies the rig is a shortcut (MEASURED on eval-139: an arbitrary label painted
    #: as that strip is 0.899 balanced-accuracy decodable from an 8x16 thumbnail). Zeroing
    #: the bottom N rows of EVERY frame makes the region constant, so it carries no rig
    #: information. 0 = off, bit-identical to every existing arm.
    equalize_bottom_rows: int = 0
    out_indices: tuple[int, ...] = (3, 4)
    #: ⛔ False ONLY for the deliberate-regression arm and for tests that must
    #: build the graph without a download. Every caller that sets it is stamped.
    verify_imagenet_stats: bool = True
    mean: tuple[float, ...] = field(default_factory=lambda: IMAGENET_MEAN)
    std: tuple[float, ...] = field(default_factory=lambda: IMAGENET_STD)

    def __post_init__(self) -> None:
        if int(self.frames) < 1:
            raise ValueError(f"frames (K) must be >= 1, got {self.frames}")
        if str(self.mode) not in ("shared", "inflate"):
            raise ValueError(
                f"mode {self.mode!r} not in ('shared', 'inflate'). 'shared' is "
                f"K passes through one set of weights (the ImageNet prior "
                f"stays exact); 'inflate' is one pass through a 3K-channel "
                f"stem. Anything else has no inflation rule and would silently "
                f"drop the prior.")
        if str(self.fuse) not in ("concat1x1", "attn", "last"):
            raise ValueError(
                f"fuse {self.fuse!r} not in ('concat1x1', 'attn', 'last')")
        h, w = self.image_hw
        if h % 32 or w % 32:
            raise ValueError(
                f"refcv6 trunk: image {h}x{w} — each axis must divide by 32 or "
                f"the stride-32 map is silently mis-sized (the 2026-07-27 "
                f"single-axis check defect, per axis this time).")

    @property
    def in_channels(self) -> int:
        """The channel count of the STACKED input tensor: ``3 * K``."""
        return 3 * int(self.frames)


def inflate_stem_(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Repeat a 3-channel ImageNet stem to ``in_channels`` and divide by the
    repeat count. Returns the new conv (the original is not mutated).

    ⭐ **Why divide.** A conv is a sum over input channels. If the same RGB
    frame is presented ``k`` times and each copy carries the full weight, the
    pre-activation is ``k`` times too large and every BatchNorm statistic the
    ImageNet run measured is wrong. Dividing by ``k`` makes the inflated stem
    an *exact* re-expression:

        sum_{j<k} sum_c (w[:, c] / k) * x[c] == sum_c w[:, c] * x[c]

    so on a repeated-frame input the inflated trunk and the 3-channel trunk
    agree to float tolerance. That is
    ``test_refcv6_trunk.py::test_nine_channel_inflation_reproduces_three``, and
    it is why this is a knockout of the *input*, not a different network.

    ⛔ Carreira & Zisserman's I3D bootstrapping rule, applied across channels
    rather than time. Stated because the "obvious" alternatives — repeat
    without dividing, or zero the extra channels — change the answer on the
    very first forward.
    """
    if conv.in_channels != 3:
        raise ValueError(
            f"inflate_stem_ expects a 3-channel ImageNet stem, got "
            f"{conv.in_channels} — inflating an already-inflated stem would "
            f"divide the prior twice.")
    k, rem = divmod(int(in_channels), 3)
    if rem:
        raise ValueError(
            f"in_channels {in_channels} is not a multiple of 3; there is no "
            f"repeat rule that keeps the RGB ordering.")
    new = nn.Conv2d(int(in_channels), conv.out_channels,
                    kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation,
                    groups=conv.groups, bias=conv.bias is not None,
                    padding_mode=conv.padding_mode)
    with torch.no_grad():
        new.weight.copy_(conv.weight.repeat(1, k, 1, 1) / float(k))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


# ============================================================================
# Temporal fusion — AFTER the trunk, at every stride
# ============================================================================

class TemporalFuse(nn.Module):
    """Fuse ``K`` per-frame feature maps of ``c`` channels into one.

    ``forward(x)`` takes ``[B, K, c, H, W]`` and returns ``[B, c, H, W]``.

    ⭐ **Fusion happens AFTER the trunk, never before it.** Stacking frames on
    the input channel axis forces the pretrained stem to see a distribution it
    was never fitted on; fusing the *features* leaves every ImageNet weight
    reading exactly 3 normalised channels. That is the whole reason the K-pass
    construction is the default.

    ⛔ **``fuse_identity_init`` makes history REMOVABLE.** With it, a freshly
    built K-frame trunk emits *bit-identically* what the single-frame trunk
    emits — all the weight sits on the NEWEST frame (index ``K-1``) and the
    older frames contribute exactly zero. So the first training step starts
    from today's model, and any later difference is LEARNED rather than an
    artefact of having added parameters. Without it, "3 frames beat 1 frame"
    would be confounded with "a randomly-initialised 1x1 conv was inserted".
    """

    def __init__(self, c: int, k: int, kind: str = "concat1x1",
                 identity_init: bool = True):
        super().__init__()
        self.c, self.k, self.kind = int(c), int(k), str(kind)
        self.identity_init = bool(identity_init)
        if self.kind == "last":
            return
        if self.kind == "concat1x1":
            self.proj = nn.Conv2d(self.c * self.k, self.c, 1, bias=True)
            if identity_init:
                with torch.no_grad():
                    self.proj.weight.zero_()
                    self.proj.bias.zero_()
                    # ⭐ identity ON THE NEWEST FRAME: block K-1 of the
                    # concatenation is the identity matrix, every older block
                    # is zero.
                    newest = (self.k - 1) * self.c
                    for i in range(self.c):
                        self.proj.weight[i, newest + i, 0, 0] = 1.0
        elif self.kind == "attn":
            # Per-CELL softmax over the K frames, scored from the concatenated
            # features. Zero weights + a one-hot bias put ALL mass on the
            # newest frame at init, so this is identity-initialised too.
            self.score = nn.Conv2d(self.c * self.k, self.k, 1, bias=True)
            with torch.no_grad():
                self.score.weight.zero_()
                self.score.bias.zero_()
                if identity_init:
                    # 20.0 makes softmax(newest) = 1 - 2e-9 in fp32: the
                    # identity to float tolerance, while still differentiable.
                    self.score.bias[self.k - 1] = 20.0
        else:                                              # pragma: no cover
            raise ValueError(f"unknown fuse kind {self.kind!r}")

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"TemporalFuse expects [B, K, C, H, W], got "
                             f"{tuple(x.shape)}")
        b, k, c, h, w = x.shape
        if k != self.k or c != self.c:
            raise ValueError(
                f"TemporalFuse built for K={self.k}, C={self.c} but got "
                f"K={k}, C={c}")
        if self.kind == "last":
            return x[:, -1]
        flat = x.reshape(b, k * c, h, w)
        if self.kind == "concat1x1":
            return self.proj(flat)
        a = torch.softmax(self.score(flat), dim=1)          # [B, K, H, W]
        return (x * a.unsqueeze(2)).sum(dim=1)


def _relu_out_of_place_(net) -> int:
    """Make every in-place activation out-of-place; returns how many moved.

    ⛔ Required before ANY recomputation: a checkpointed segment runs twice, and an
    in-place ReLU trips autograd's version counter on the second pass. MEASURED: timm's
    own `set_grad_checkpointing(True)` RAISES *"modified by an inplace operation ...
    ReluBackward0 ... version 1"* on resnet34 AND resnet101. The lever's real price
    includes this, and the extra activation memory it costs is inside the numbers above.
    """
    n = 0
    for m in net.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish)) \
                and getattr(m, "inplace", False):
            m.inplace = False
            n += 1
    return n


def _conv_bn_pairs(net) -> list:
    """Every ``(conv, bn)`` pair whose BN directly follows its conv, by NAME, not by guess.

    timm ResNets name them ``convN`` / ``bnN`` (stem and every bottleneck) and put the
    shortcut in a ``Sequential(Conv2d, BatchNorm2d)``. MEASURED on resnet101: 104 convs,
    104 BatchNorms, 104 pairs. A BN whose width does not match its conv is refused.
    """
    pairs = []
    for mod in net.modules():
        ch = dict(mod.named_children())
        for k, c in ch.items():
            if isinstance(c, nn.Conv2d) and k.startswith("conv"):
                b = ch.get("bn" + k[len("conv"):])
                if isinstance(b, nn.BatchNorm2d):
                    pairs.append((c, b))
        if isinstance(mod, nn.Sequential):
            items = list(mod.children())
            for a, b in zip(items, items[1:]):
                if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                    pairs.append((a, b))
    for c, b in pairs:
        if int(b.num_features) != int(c.out_channels):
            raise ValueError(f"conv/bn width mismatch {c.out_channels} vs {b.num_features}")
    return pairs


def _fold_frozen_bn_(net) -> int:
    """Fold every FROZEN BatchNorm into the conv before it; returns how many pairs.

    ⭐ SPEED (MEASURED 2026-09-23 on Thor): with BN pinned to its running statistics each
    BatchNorm is a fixed per-channel affine, yet it cost ~23 % of GPU time as separate
    memory-bound passes forward AND backward. ``BN(conv(x; W)) = conv(x; s*W) + (beta -
    mu*s)`` with ``s = gamma / sqrt(var + eps)`` is the SAME function, so the conv computes
    it directly and the BN becomes the identity. ``gamma``/``beta`` still train (through
    ``s`` and the bias); the running statistics are the frozen constants they already are.

    ⛔ NOTHING IS REPLACED: only the two instances' ``forward`` change, so the module tree
    and every ``state_dict`` key are identical -- a folded run's checkpoint loads into an
    unfolded model and vice versa. ⛔ Frozen BN only: a BN that trains on the batch is not
    a fixed affine, and the trunk refuses the lever without it.
    """
    pairs = _conv_bn_pairs(net)
    for conv, bn in pairs:
        def _fwd(x, _c=conv, _b=bn):
            s = _b.weight / torch.sqrt(_b.running_var + _b.eps)
            w = _c.weight * s.reshape(-1, 1, 1, 1)
            bias = _b.bias - _b.running_mean * s
            if _c.bias is not None:
                bias = bias + _c.bias * s
            return _c._conv_forward(x, w, bias)
        conv.forward = _fwd
        bn.forward = (lambda x: x)   # noqa: E731 -- its affine now lives in the conv
    return len(pairs)


def _freeze_bn_(net) -> int:
    """Pin every BatchNorm to EVAL (ImageNet running stats); returns how many.

    ⛔ IT REFUSES TO BE UN-FROZEN, and that is the whole point. `nn.Module.train()`
    RECURSES into children and the trainer calls `model.train()` every step, so a plain
    `.eval()` would be silently undone on step 1 and the run would REPORT a frozen-BN
    arm while training BN on the batch. Replacing the bound `train` is what makes the
    claim survive contact with the trainer; :meth:`bn_training_count` reads it back from
    the live model rather than trusting this.
    """
    k = 0
    for m in net.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.train = (lambda mode=True, _m=m: _m)   # noqa: E731 — deliberate
            k += 1
    return k


def _chunked_backbone(net, x: Tensor, chunk: int):
    """Run ``net`` over ``x`` in leading-batch slices of ``chunk``, each checkpointed.

    ⭐ THE ONE LEVER THAT DOES NOT CHANGE THE ARM (with BN frozen): same weights, same
    dtype, same frames, same fusion — one extra forward. It is NOT timm's
    `set_grad_checkpointing`, which raises on this backbone (in-place ReLU vs the
    version counter) and, once that is fixed, still does not fit: it checkpoints the
    four stages while the 653 MB stem activation stays resident (MEASURED 15.17 GB).

    ⛔ A FUNCTION, NOT A WRAPPER MODULE, ON PURPOSE. Wrapping `self.net` renamed every
    backbone key (`net.conv1.weight` -> `net.net.conv1.weight`), so a checkpoint from a
    levered run could not be loaded by an unlevered model or by any eval driver — and
    that failure would surface days later, at load time, on an arm already paid for.
    Chunking the CALL leaves the module tree and the state_dict IDENTICAL, so the lever
    can be turned on or off between runs of the same lineage.
    """
    outs = []
    for i in range(0, x.shape[0], chunk):
        outs.append(torch.utils.checkpoint.checkpoint(
            net, x[i:i + chunk], use_reentrant=False))
    n = len(outs[0])
    return [torch.cat([o[j] for o in outs], dim=0) for j in range(n)]

class TimmResNetTrunk(nn.Module):
    """A ``timm`` ImageNet backbone over K history frames.

    ``forward`` honours the REF-C encoder contract — ``(fmap, pooled)`` with
    ``fmap`` the **fused stride-32** planner map — so it is a drop-in for
    :class:`tanitad.refs.refc.ResNetEncoder`. The fused stride-16 perception
    map is reachable without re-running the trunk:

    * :meth:`forward_features` -> ``(s16, s32, pooled)``, the explicit call;
    * :attr:`last_s16`, stashed by the last :meth:`forward`.

    ⚠️ :attr:`last_s16` is a live tensor with its graph attached. A perception
    head that wants it must consume it in the SAME forward.
    """

    def __init__(self, cfg: TimmTrunkConfig | None = None):
        super().__init__()
        self.cfg = cfg or TimmTrunkConfig()
        try:
            import timm
        except ImportError as e:                          # pragma: no cover
            raise ImportError(
                "refcv6 §2 needs `timm` for the ImageNet trunk. Install it "
                "into the run's venv — there is no fallback, because a random "
                "init silently standing in for ImageNet is the one failure "
                "this module exists to prevent.") from e
        self.timm_version = timm.__version__
        net = timm.create_model(self.cfg.model_name,
                                pretrained=bool(self.cfg.pretrained),
                                features_only=True,
                                out_indices=tuple(self.cfg.out_indices))
        if self.cfg.pretrained and self.cfg.verify_imagenet_stats:
            _assert_pretrained_loaded(net, self.cfg.model_name)
        # ⭐ INFLATION HAPPENS AFTER THE VERIFICATION, never before: the
        # statistic is a property of the 3-channel ImageNet stem, and checking
        # it on an inflated stem would check our own arithmetic instead of the
        # download.
        self.k = int(self.cfg.frames)
        if str(self.cfg.mode) == "inflate" and self.k > 1:
            stem_name, stem = _find_stem(net)
            _set_by_path(net, stem_name,
                         inflate_stem_(stem, self.cfg.in_channels))
        self.net = net

        chans = list(net.feature_info.channels())
        reds = list(net.feature_info.reduction())
        # ⛔ READ FROM `feature_info`, never written down: the PI put the trunk
        # SIZE under review, and a hard-coded 256/512 is exactly what would
        # make a resnet50 swap fail somewhere far from here.
        if len(chans) != 2 or reds != [16, 32]:
            raise ValueError(
                f"refcv6 trunk expects exactly the (stride-16, stride-32) "
                f"pair; timm reports channels={chans} reduction={reds} for "
                f"{self.cfg.model_name!r} with out_indices "
                f"{self.cfg.out_indices}.")
        self.s16_dim, self.s32_dim = int(chans[0]), int(chans[1])
        self.feat_dim = self.s32_dim                    # encoder contract
        h, w = self.cfg.image_hw
        self.s16_shape = (h // 16, w // 16)
        self.grid_shape = (h // 32, w // 32)            # encoder contract
        self.last_s16: Tensor | None = None

        # ---- temporal fusion, at BOTH strides --------------------------- #
        # ⛔ Built ONLY on the shared-weight path with K > 1. On `inflate`, or
        # with K = 1, there is a single feature map per stride and a fusion
        # module would be an identity with parameters.
        self.fuse16: nn.Module | None = None
        self.fuse32: nn.Module | None = None
        if str(self.cfg.mode) == "shared" and self.k > 1:
            self.fuse16 = TemporalFuse(self.s16_dim, self.k, self.cfg.fuse,
                                       self.cfg.fuse_identity_init)
            self.fuse32 = TemporalFuse(self.s32_dim, self.k, self.cfg.fuse,
                                       self.cfg.fuse_identity_init)

        # Normalisation constants, TILED over the stack: each RGB triple is its
        # own image and gets its own statistics.
        self.register_buffer(
            "_mean",
            torch.tensor(tuple(self.cfg.mean) * self.k).reshape(1, -1, 1, 1),
            persistent=False)
        self.register_buffer(
            "_std",
            torch.tensor(tuple(self.cfg.std) * self.k).reshape(1, -1, 1, 1),
            persistent=False)
        # ⭐ Counted, not assumed. The test reads it to prove the normalisation
        # happens EXACTLY ONCE per forward — `E-SEED-2`'s trap is "no
        # normalisation"; its twin is "normalised twice by a helpful caller",
        # which halves the contrast just as quietly.
        self.norm_calls: int = 0

        # ---- memory levers, LAST: `feature_info` above must be read from the
        # UNWRAPPED backbone, and the freeze must precede the wrap because it walks
        # `.modules()` on the real network.
        self.memory_levers: dict = {"chunk_ckpt": 0, "frozen_bn": False,
                                    "bn_pinned": 0, "relu_out_of_place": 0,
                                    "bf16": bool(getattr(self.cfg, "bf16", False)),
                                    "channels_last": bool(
                                        getattr(self.cfg, "channels_last", False))}
        if self.memory_levers["channels_last"]:
            # in place: the module tree and the state_dict keys are unchanged
            self.net.to(memory_format=torch.channels_last)
        if bool(getattr(self.cfg, "frozen_bn", False)):
            self.memory_levers["frozen_bn"] = True
            self.memory_levers["bn_pinned"] = _freeze_bn_(self.net)
        if bool(getattr(self.cfg, "fold_bn", False)):
            if not self.memory_levers["frozen_bn"]:
                raise ValueError(
                    "trunk fold_bn without frozen_bn: a BatchNorm that trains on the batch "
                    "is not a fixed affine, so folding it would change the function. Pass "
                    "frozen_bn=True, or do not fold.")
            self.memory_levers["bn_folded"] = _fold_frozen_bn_(self.net)
        if bool(getattr(self.cfg, "dedup_frames", False)):
            if not self.memory_levers["frozen_bn"]:
                raise ValueError(
                    "trunk dedup_frames without frozen_bn: a BatchNorm that trains on the "
                    "batch makes each frame's features depend on WHICH frames share its "
                    "batch, so computing a frame once would not be the same function. Pass "
                    "frozen_bn=True, or do not dedup.")
            if str(self.cfg.mode) != "shared" or self.k < 2:
                raise ValueError(
                    f"trunk dedup_frames needs mode 'shared' with K >= 2 frames per stack "
                    f"(got mode {self.cfg.mode!r}, K {self.k}): only separate per-frame "
                    f"passes have frames to share.")
            self.memory_levers["dedup_frames"] = True
        #: (frame slots, frames computed) of the LAST deduplicated call, and the same pair
        #: ACCUMULATED over every call since the run log last read and reset it -- a training
        #: step calls the trunk more than once (the window, then the future frames).
        self.last_dedup: tuple | None = None
        self.dedup_counts: list = [0, 0]
        _ck = int(getattr(self.cfg, "chunk_ckpt", 0) or 0)
        if _ck > 0:
            # ⚠️ REFUSED, not silently allowed: chunking without frozen BN is a
            # DIFFERENT ARM (MEASURED -40 % on `ga_trunk`), and a run that believed it
            # had only saved memory would be compared against arms it no longer matches.
            if not self.memory_levers["frozen_bn"]:
                raise ValueError(
                    "trunk chunk_ckpt=%d without frozen_bn: chunking changes "
                    "BatchNorm (each chunk is a different sub-batch; MEASURED "
                    "-40%% on resnet34's ga_trunk), so the arm would silently "
                    "stop matching every arm it is compared with. Pass "
                    "frozen_bn=True, or do not chunk." % _ck)
            self.memory_levers["relu_out_of_place"] = _relu_out_of_place_(self.net)
            # ⛔ `self.net` is NOT replaced — see `_chunked_backbone`. The module tree
            # and the state_dict stay identical, so checkpoints remain interchangeable
            # with unlevered runs of the same lineage.
            self.memory_levers["chunk_ckpt"] = _ck
        # ⭐ compile LAST, so it sees the network exactly as the levers above left it. Set with
        # `object.__setattr__` so nn.Module does NOT register the wrapper: the module tree and
        # every state_dict key stay the eager trunk's, and a compiled run's checkpoint loads
        # into an uncompiled model (and back).
        object.__setattr__(self, "_net_fn", None)
        if bool(getattr(self.cfg, "compile_backbone", False)):
            _be = str(getattr(self.cfg, "compile_backend", "inductor") or "inductor")
            # ⛔ THE TRAINER BACKPROPAGATES THROUGH ONE GRAPH MORE THAN ONCE (the conflict
            # detector's per-term gradients use retain_graph=True before the step's backward).
            # AOTAutograd's DONATED BUFFERS forbid that: MEASURED 2026-09-23 on Thor, the first
            # compiled smoke (S11) died at step 1 with "compiled with non-empty donated buffers
            # which requires create_graph=False and retain_graph=False". Its documented switch:
            import torch._functorch.config as _fconfig
            _fconfig.donated_buffer = False
            self.memory_levers["compile_donated_buffer"] = bool(_fconfig.donated_buffer)
            object.__setattr__(self, "_net_fn", torch.compile(self.net, backend=_be))
            self.memory_levers["compile"] = _be

    # -- the normalisation, in one place ---------------------------------- #
    def normalise(self, x: Tensor) -> Tensor:
        """ImageNet mean/std on a ``[B, 3K, H, W]`` float tensor in ``[0, 1]``.

        ⛔ The C26 equalization runs FIRST and in the ``[0, 1]`` domain, so the zeroed
        rows are byte-for-byte the black the strip clips already carry -- then both kinds
        of clip go through the same normalisation. It runs whether or not ImageNet
        normalisation is on, and on EVERY forward (train and eval alike): a mask applied at
        training only would hand the evaluator a distribution the model never saw."""
        _eq = int(getattr(self.cfg, "equalize_bottom_rows", 0) or 0)
        if _eq > 0:
            if _eq >= int(x.shape[-2]):
                raise ValueError(f"equalize_bottom_rows {_eq} >= frame height "
                                 f"{int(x.shape[-2])}")
            x = x.clone()
            x[..., -_eq:, :] = 0.0
            self.equalize_calls = getattr(self, "equalize_calls", 0) + 1
        if not self.cfg.imagenet_norm:
            return x
        if x.shape[1] != self._mean.shape[1]:
            raise ValueError(
                f"trunk normalisation: input has {x.shape[1]} channels but "
                f"the trunk was built for {self._mean.shape[1]} (K="
                f"{self.k} frames x 3)")
        self.norm_calls += 1
        return (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)

    def _backbone(self, x: Tensor):
        """The backbone call, chunked or not. ONE place decides, so the two call
        sites below cannot drift apart.
        """
        ck = int(self.memory_levers.get("chunk_ckpt", 0) or 0)
        bf = bool(self.memory_levers.get("bf16", False))
        cl = bool(self.memory_levers.get("channels_last", False))
        # the compiled backbone when --trunk-compile asked for it; otherwise `self.net` itself
        net = self._net_fn if getattr(self, "_net_fn", None) is not None else self.net
        if not (bf or cl):
            # the exact pre-lever path -- no autocast context is even entered
            if ck <= 0:
                return net(x)
            return _chunked_backbone(net, x, ck)
        if cl:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16,
                            enabled=bf):
            out = net(x) if ck <= 0 else _chunked_backbone(net, x, ck)
        # ⛔ back to float32 and the default layout: everything downstream of the backbone
        # (fusion, decoders, trajectory integration, losses) must see what it always saw.
        return [o.float().contiguous() for o in out]

    def _backbone_dedup(self, per: Tensor):
        """``[N, K, 3, H, W]`` -> the backbone's (s16, s32) for all ``N*K`` frame slots, in the
        ``row * K + position`` order :meth:`forward_features` has always used -- computing each
        DISTINCT frame of adjacent overlapping stacks once and gathering it into its slots.

        ⛔ VERIFIED, NEVER ASSUMED: row ``i+1`` reuses row ``i``'s frames only where the data
        say they ARE the same frame -- exact equality of ``per[i+1, :K-1]`` and ``per[i, 1:]``.
        A window boundary (the next sample, another clip) fails the check and computes all K
        frames, and so does any input without the D-015 structure: the lever is exact on
        every input and merely does nothing where there is nothing to share. The gather's
        backward SUMS each frame's gradient over its slots, which is what the separate
        passes' weight gradients summed to. ``last_dedup`` = (slots, frames computed).
        """
        n, k = int(per.shape[0]), int(per.shape[1])
        same = ((per[1:, :k - 1] == per[:-1, 1:]).flatten(1).all(dim=1).tolist()
                if n > 1 else [])
        slot = [[0] * k for _ in range(n)]
        rows: list = []
        pos: list = []
        for i in range(n):
            first = 0
            if i > 0 and same[i - 1]:
                slot[i][:k - 1] = slot[i - 1][1:]
                first = k - 1
            for j in range(first, k):
                slot[i][j] = len(rows)
                rows.append(i)
                pos.append(j)
        dev = per.device
        uniq = per[torch.tensor(rows, device=dev), torch.tensor(pos, device=dev)]
        f16u, f32u = self._backbone(uniq)
        flat = torch.tensor([u for r in slot for u in r], device=f16u.device)
        self.last_dedup = (n * k, len(rows))
        self.dedup_counts = [self.dedup_counts[0] + n * k, self.dedup_counts[1] + len(rows)]
        return f16u.index_select(0, flat), f32u.index_select(0, flat)

    def forward_features(self, x: Tensor,
                         already_normalised: bool = False
                         ) -> tuple[Tensor, Tensor, Tensor]:
        """``[B, 3K, H, W]`` -> ``(s16, s32, pooled)``, temporally fused.

        ``already_normalised`` is the ONLY way to skip the normalisation, and
        it is explicit on purpose: a caller that normalised upstream must say
        so, rather than the trunk guessing from the value range.

        ⛔ **The frame order is OLDEST -> NEWEST**, matching REF-C's stack
        convention (``D-015``: "latest = [-3:]"). The fusion's identity init
        depends on it, and reversing it would silently make the *oldest* frame
        the one history is measured against.
        """
        if x.ndim != 4:
            raise ValueError(f"trunk expects [B, 3K, H, W], got "
                             f"{tuple(x.shape)}")
        if not already_normalised:
            x = self.normalise(x)
        if str(self.cfg.mode) == "inflate" or self.k == 1:
            s16, s32 = self._backbone(x)
        else:
            # ⭐ K SEPARATE 3-CHANNEL PASSES THROUGH THE SAME WEIGHTS. Folding
            # K into the batch is what makes them one kernel launch while
            # keeping the stem's input exactly 3 ImageNet channels.
            b = x.shape[0]
            if self.memory_levers.get("dedup_frames"):
                f16, f32 = self._backbone_dedup(
                    x.reshape(b, self.k, 3, *x.shape[2:]))
            else:
                per = x.reshape(b, self.k, 3, *x.shape[2:]).reshape(
                    b * self.k, 3, *x.shape[2:])
                f16, f32 = self._backbone(per)
            s16 = self.fuse16(f16.reshape(b, self.k, *f16.shape[1:]))
            s32 = self.fuse32(f32.reshape(b, self.k, *f32.shape[1:]))
        self.last_s16 = s16
        return s16, s32, s32.mean(dim=(2, 3))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """REF-C encoder contract: ``(fmap, pooled)`` at **stride 32**."""
        _s16, s32, pooled = self.forward_features(x)
        return s32, pooled

    # -- reporting --------------------------------------------------------- #
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trunk_param_count(self) -> int:
        """The BACKBONE alone, without the fusion — the number the literature
        quotes (DD's ResNet-34 is 21.8 M)."""
        return sum(p.numel() for p in self.net.parameters())

    def bn_training_count(self) -> int:
        """How many backbone BatchNorms are in TRAINING mode RIGHT NOW.

        ⭐ Read this AFTER the trainer has called `model.train()`, never before: the
        whole hazard is that `.train()` recurses and un-freezes. A frozen-BN arm must
        report 0 here during the real run — a measurement, not a promise.
        """
        return sum(1 for m in self.modules()
                   if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training)

    # -- BatchNorm RECALIBRATION (A7, 2026-09-19) ------------------------- #
    # ⛔⛔ WHY THIS EXISTS. `frozen_bn` pins every backbone BN to EVAL, where it
    # normalises with its STORED running statistics. On an ImageNet-init trunk those
    # are ImageNet's. On a RANDOM-init trunk they are what a fresh BN stores --
    # MEASURED 2026-09-19 on resnet34: 36 layers, every running_mean exactly 0.0,
    # every running_var exactly 1.0 -- so eval-mode BN is `(x-0)/sqrt(1+eps)*1+0`,
    # THE IDENTITY. An ImageNet-vs-random knockout run with the memory levers
    # therefore compared "ImageNet weights + a normalisation" against "random
    # weights + NO normalisation": two variables, one of them invisible.
    # ⇒ Recalibrate BOTH arms' statistics on the SAME data, then keep them frozen,
    # so the conditions differ in their WEIGHTS only.
    def backbone_bns(self) -> list:
        """Every BatchNorm in the BACKBONE, in module order.

        ⭐ Exactly the set :func:`_freeze_bn_` walks (``self.net.modules()``), so
        recalibration and freezing act on the SAME layers by construction.
        :class:`TemporalFuse` carries no normalisation and is correctly outside it.
        """
        return [m for m in self.net.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def bn_stats_snapshot(self) -> dict:
        """A detached float64 CPU copy of every backbone BN's running statistics."""
        bns = self.backbone_bns()
        return {
            "mean": torch.cat([m.running_mean.detach().double().cpu().flatten()
                               for m in bns]),
            "var": torch.cat([m.running_var.detach().double().cpu().flatten()
                              for m in bns]),
            "n_bn": len(bns)}

    @torch.no_grad()
    def recalibrate_bn_(self, batches) -> dict:
        """Re-estimate every backbone BN's running statistics on ``batches``.

        ``batches`` is an iterable of trunk inputs -- ``[N, 3K, H, W]`` floats in
        ``[0, 1]``, exactly what :meth:`forward_features` receives in training.

        ⭐ The canonical ``torch.optim.swa_utils.update_bn`` procedure, applied to the
        backbone only: ``reset_running_stats()``, ``momentum = None`` (the EXACT
        cumulative mean over batches, not an EMA), BN in train mode, forward under
        ``no_grad``. Every module's ``training`` flag, every momentum and the chunk
        lever are restored afterwards, so a frozen trunk comes back exactly as frozen
        -- with new statistics.

        ⛔ CHUNKING IS BYPASSED FOR THE DURATION, AND THAT IS A CORRECTNESS RULE, NOT
        AN OPTIMISATION. With ``chunk_ckpt = 1`` every BN "batch" would be ONE image;
        averaging per-image variances omits the between-image variance, so
        ``running_var`` would be systematically UNDER-estimated. No gradients are kept
        here, so the memory the lever exists to save is not at stake.

        ⚠️ The pinned ``train`` that :func:`_freeze_bn_` installs is bypassed ON
        PURPOSE by writing ``.training`` directly: calling ``.train()`` on a frozen BN
        is a deliberate no-op, and this is the one place it must be overridden.
        """
        bns = self.backbone_bns()
        if not bns:
            raise ValueError(
                "recalibrate_bn_: the backbone has no BatchNorm -- there is nothing "
                "to recalibrate, and a stamp saying otherwise would be false.")
        flags = {m: m.training for m in self.modules()}
        moms = [m.momentum for m in bns]
        ck = self.memory_levers.get("chunk_ckpt", 0)
        n_b = n_i = 0
        try:
            self.eval()                     # nothing else in the trunk may move
            self.memory_levers["chunk_ckpt"] = 0
            for m in bns:
                m.reset_running_stats()
                m.momentum = None
                m.training = True
            for x in batches:
                self.forward_features(x)
                n_b += 1
                n_i += int(x.shape[0])
        finally:
            for m, mom in zip(bns, moms):
                m.momentum = mom
            for m, tr in flags.items():
                m.training = tr
            self.memory_levers["chunk_ckpt"] = ck
        if n_b == 0:
            raise ValueError(
                "recalibrate_bn_ saw ZERO batches: the statistics are now the reset "
                "values (mean 0 / var 1) -- exactly the identity this exists to "
                "remove. Refusing rather than training on it.")
        snap = self.bn_stats_snapshot()
        import hashlib
        h = hashlib.sha256()
        h.update(snap["mean"].numpy().tobytes())
        h.update(snap["var"].numpy().tobytes())
        return {"n_batches": n_b, "n_images": n_i, "n_bn": len(bns),
                "chunk_bypassed": bool(ck), "stats_sha12": h.hexdigest()[:12]}

    @torch.no_grad()
    def measure_true_bn_stats_(self, batches) -> dict:
        """The statistics these weights WOULD have now, leaving the trunk UNCHANGED.

        Recalibrates in place and then restores every BN buffer, so the checkpoint
        and the rest of the run see the frozen statistics exactly as they were.
        """
        bns = self.backbone_bns()
        saved = [(m.running_mean.clone(), m.running_var.clone(),
                  m.num_batches_tracked.clone()) for m in bns]
        try:
            self.recalibrate_bn_(batches)
            return self.bn_stats_snapshot()
        finally:
            for m, (rm, rv, nb) in zip(bns, saved):
                m.running_mean.copy_(rm)
                m.running_var.copy_(rv)
                m.num_batches_tracked.copy_(nb)

    def provenance(self) -> dict:
        """What a ``config.json`` stamp must carry about this trunk."""
        return {
            "trunk": "timm",
            "model_name": self.cfg.model_name,
            "timm_version": self.timm_version,
            "pretrained": bool(self.cfg.pretrained),
            "verified_pretrained": bool(self.cfg.pretrained
                                        and self.cfg.verify_imagenet_stats),
            "frames": self.k,
            "mode": str(self.cfg.mode),
            "fuse": str(self.cfg.fuse) if self.fuse32 is not None else None,
            "fuse_identity_init": bool(self.cfg.fuse_identity_init),
            "in_channels": self.cfg.in_channels,
            "imagenet_norm": bool(self.cfg.imagenet_norm),
            "image_hw": list(self.cfg.image_hw),
            "s16": [self.s16_dim, *self.s16_shape],
            "s32": [self.s32_dim, *self.grid_shape],
            "params": self.param_count(),
            "backbone_params": self.trunk_param_count(),
        }


def bn_staleness(frozen: dict, true: dict) -> dict:
    """How stale FROZEN BatchNorm statistics are against the TRUE ones of the weights.

    ⚠️ Recalibration equalises the START of a frozen-BN run; the statistics then go
    stale as the weights train, and a random trunk moves further than an ImageNet
    one. This is the measured size of that residual, per arm (A7.4):

    * ``bn_staleness_var``  -- median over channels of ``|ln(var_frozen / var_true)|``
    * ``bn_staleness_mean`` -- median over channels of
      ``|mean_frozen - mean_true| / sqrt(var_true)``

    Both are 0.0 exactly when the frozen statistics are the true ones.
    """
    if frozen["var"].shape != true["var"].shape:
        raise ValueError("bn_staleness: the two snapshots cover different channels "
                         "(%s vs %s)" % (tuple(frozen["var"].shape),
                                         tuple(true["var"].shape)))
    eps = 1e-12
    vf = frozen["var"].double().clamp_min(eps)
    vt = true["var"].double().clamp_min(eps)
    lr = (vf / vt).log().abs()
    ms = (frozen["mean"].double() - true["mean"].double()).abs() / vt.sqrt()
    return {"bn_staleness_var": float(lr.median()),
            "bn_staleness_mean": float(ms.median()),
            "n_channels": int(vf.numel())}


def _find_stem(net: nn.Module) -> tuple[str, nn.Conv2d]:
    """-> ``(dotted_name, conv)`` for the FIRST conv the backbone applies.

    ⛔ Found by SEARCH, not by name: ``conv1`` is ResNet's spelling, ConvNeXt's
    is ``stem_0`` / ``stem.0``. Hard-coding one spelling is what would make the
    stem-inflation arm silently skip on a non-ResNet backbone.
    """
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.in_channels == 3:
            return name, mod
    raise AttributeError(
        "refcv6 trunk: no 3-channel Conv2d found on the timm feature model, "
        "so the stem-inflation knockout cannot be applied. Use "
        "mode='shared', which needs no stem surgery.")


def _set_by_path(root: nn.Module, dotted: str, value: nn.Module) -> None:
    parts = dotted.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


def _assert_pretrained_loaded(net: nn.Module, model_name: str) -> None:
    """FAIL LOUD if ``pretrained=True`` did not actually load ImageNet.

    ⛔ ``timm.create_model(..., pretrained=True)`` raises on a *failed*
    download, but a cached-but-wrong file, a ``model_name`` that resolves to an
    untrained variant, or a future ``pretrained`` semantics change would all
    leave a randomly-initialised trunk behind, and the arm would train,
    converge and mean nothing.

    Two witnesses, cheapest first:

    1. **the pinned statistic** for a backbone we have measured (the default).
       A He-initialised stem misses it by a factor of ~1.5 and cannot pass;
    2. **the checkpoint on disk** for anything else: the stem is compared
       against the tensor ``timm`` itself would load. Model-agnostic, and it
       checks the thing that actually matters — that the weights in memory are
       the weights in the file.
    """
    name, stem = _find_stem(net)
    pinned = _PINNED_STATS.get(str(model_name))
    if pinned is not None:
        got = float(stem.weight.detach().abs().sum())
        lo, hi = pinned * (1 - _CONV1_ABS_SUM_RTOL), pinned * (1 + _CONV1_ABS_SUM_RTOL)
        if not (lo <= got <= hi):
            raise RuntimeError(
                f"refcv6 trunk {model_name!r}: {name}.weight |w|.sum() = "
                f"{got:.4f} is outside [{lo:.4f}, {hi:.4f}] — the ImageNet "
                f"weights were NOT loaded (a He init reads ~875). Refusing to "
                f"train a trunk that claims a prior it does not have.")
        return
    # No pinned statistic: compare against the checkpoint timm would load.
    ref = _reference_stem_weight(model_name, stem.weight.shape)
    if ref is None:
        raise RuntimeError(
            f"refcv6 trunk {model_name!r}: no pinned ImageNet statistic and "
            f"the reference checkpoint could not be read, so 'pretrained' "
            f"cannot be VERIFIED. Add a measured entry to `_PINNED_STATS`, or "
            f"pass `verify_imagenet_stats=False` and accept that the run "
            f"record cannot claim a prior.")
    if not torch.allclose(stem.weight.detach().float().cpu(), ref.float(),
                          atol=1e-6):
        raise RuntimeError(
            f"refcv6 trunk {model_name!r}: the built {name}.weight does not "
            f"match the pretrained checkpoint on disk — `pretrained=True` did "
            f"not take effect.")


def _reference_stem_weight(model_name: str, shape) -> Tensor | None:
    """The stem tensor ``timm`` would load for ``model_name``, or ``None``."""
    try:
        import timm
        from timm.models._hub import load_state_dict_from_hf
        cfg = timm.get_pretrained_cfg(model_name)
        hf_id = getattr(cfg, "hf_hub_id", None)
        if not hf_id:
            return None
        sd = load_state_dict_from_hf(hf_id)
        for v in sd.values():
            if hasattr(v, "shape") and tuple(v.shape) == tuple(shape):
                return v.detach().cpu()
    except Exception:                                      # pragma: no cover
        return None
    return None


# ============================================================================
# The optimiser DiffusionDrive uses (`diffusiondrivev2_rl_config.py:119-131`)
# ============================================================================

#: ``cfg_lr_mult`` — the encoder group's multiplier on the head lr.
DD_ENCODER_LR_MULT: float = 0.5
#: ``weight_decay`` — DD's value. NOT PyTorch's AdamW default of 1e-2.
DD_WEIGHT_DECAY: float = 1e-4


def param_groups_dd(model: nn.Module, lr: float,
                    encoder_attr: str = "encoder",
                    encoder_lr_mult: float = DD_ENCODER_LR_MULT,
                    weight_decay: float = DD_WEIGHT_DECAY
                    ) -> list[dict]:
    """DiffusionDrive's two parameter groups. -> ``[encoder_group, head_group]``.

    ``opt_paramwise_cfg`` (``diffusiondrivev2_rl_config.py:125-131``) matches
    parameters by NAME PREFIX (``"image_encoder"``) and applies ``lr_mult``.
    Ours matches on ``encoder_attr`` the same way, because a prefix match is
    what the released code does and a module-identity match would silently
    disagree with it for any shared parameter.

    ⛔ **Every parameter lands in exactly one group, and the function proves
    it.** A partition bug here does not crash: it trains part of the model at
    the wrong rate, or not at all, and the run looks normal.
    """
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    enc, head = [], []
    prefix = f"{encoder_attr}."
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if name.startswith(prefix) else head).append(p)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    if len(enc) + len(head) != total:
        raise AssertionError(
            f"param_groups_dd partitioned {len(enc)} + {len(head)} = "
            f"{len(enc) + len(head)} of {total} trainable tensors")
    if not enc:
        raise ValueError(
            f"param_groups_dd found no parameters under {encoder_attr!r} — "
            f"the encoder group would be empty and the 0.5x rate would apply "
            f"to nothing, which is the DD recipe in name only.")
    return [
        {"params": enc, "lr": lr * float(encoder_lr_mult),
         "weight_decay": float(weight_decay), "name": "encoder"},
        {"params": head, "lr": float(lr),
         "weight_decay": float(weight_decay), "name": "head"},
    ]


def build_timm_trunk(in_channels: int = 9,
                     image_hw: tuple[int, int] = (256, 640),
                     **kw) -> TimmResNetTrunk:
    """Convenience constructor used by the REF-C encoder factory.

    ``in_channels`` is the STACK width the corpus delivers (9 for our 3-frame
    stack); K is derived from it, because a trunk built for a different K than
    the data carries would fail at the first forward instead of at build time.
    """
    k, rem = divmod(int(in_channels), 3)
    if rem or k < 1:
        raise ValueError(
            f"in_channels {in_channels} is not 3 x K — the frame stack must "
            f"be whole RGB frames.")
    kw.setdefault("frames", k)
    return TimmResNetTrunk(TimmTrunkConfig(image_hw=tuple(image_hw), **kw))
