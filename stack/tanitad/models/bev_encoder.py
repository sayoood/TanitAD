"""refcv6 §1/§6: the BEV encoder and the SAM3 MAP head.

``SPEC_REFCV6_V2.md`` §1 puts this between :mod:`tanitad.models.bev_lift` and the
map loss::

    stride-16 map 16x40x256 -> BEV LIFT (parameter-free) -> BEV ENCODER -> BEV feats 120x64
                                                                 |-> MAP head (9-class SAM3 soft CE, seen cells only)

⭐ **The label's own cells, end to end.** :data:`tanitad.data.semantic_map_gt.
CART_SHAPE` is ``(120, 64)`` and ``tanitad.data.bev_raster.GRID_DEFAULT`` is the
same 120 x 64 @ 0.5 m grid; :func:`tanitad.models.bev_lift.build_lift_geometry`
samples the trunk at exactly those cell centres. So **every convolution in this
module is stride 1 with ``padding == (kernel-1)//2 * dilation``** and the module
REFUSES any other build: there is no resampling anywhere between the prediction
and the label, and an interpolation that silently shifted the map by half a cell
(0.25 m) could not hide inside a mIoU.

⛔ **The GT is SAM3 only.** The PI ruled on 2026-09-16: *"We won't use lidar for
bev GT, we will stick to the sam3 maps"* (``SPEC_REFCV6_V2.md`` §0, §6, §7). The
LiDAR BEV GT may still be quoted as an INDEPENDENT evaluation reference; nothing
in this module reads it, and there is no parameter through which it could arrive.

## Why stride 16 and not stride 32

``SPEC_REFCV6_V2.md`` §2: an oracle on the 8 x 20 (stride-32) map tops out at
**AP 0.3341** against **0.4713** at 16 x 40. :class:`BEVEncoder` therefore takes
the lift's output and never sees the planner's stride-32 tokens; the stride is
fixed upstream in the lift geometry and is carried here only to be asserted.

## The loss

:func:`map_soft_ce` is soft cross-entropy against the SAM3 fractions,
``-sum_c p_c log softmax(logits)_c``, averaged over **seen cells only**
(``MapFrames.seen``, i.e. ``not-seen`` fraction < 0.5 in exact integer
arithmetic -- ``semantic_map_gt.ClipMapGT.read``).

⚠️ Three decisions that a "soft cross entropy" spelling hides, all declared:

1. **9 classes, including ``not seen``.** ``SPEC_REFCV6_V2.md`` §1 says *"9-class
   SAM3 soft CE"* and :data:`semantic_map_gt.CHANNELS` has 9 entries with
   ``not seen`` last. On a SEEN cell the not-seen fraction is < 0.5 but usually
   non-zero (a partly-occluded cell), and it is a real, predictable state. It is
   kept as a class rather than renormalised away, so ``p`` is the label as
   measured and nothing is invented.
2. **The target is renormalised to sum 1, and the renormalisation is BOUNDED.**
   ``cart_frac`` is ``uint8/255``, so a cell's fractions sum to 1 only up to
   quantisation. :func:`map_soft_ce` refuses a batch whose per-cell sum leaves
   ``[1 - 9/255, 1 + 9/255]`` (:data:`FRAC_SUM_TOL`) -- a target that does not
   nearly sum to one is not a quantisation residue, it is a different array.
3. **Unseen cells contribute exactly zero**, and ``n_cells`` is returned beside
   the loss. A batch with no seen cell returns ``0.0`` WITH ``n_cells == 0``
   rather than a NaN or a silent skip (the ``slot_set_loss`` reporting rule).

## Metrics

:func:`map_metrics` reports **per class**, never pooled: IoU, the predicted and
label frequency, and the soft accuracy. ``MODEL_REGISTRY``-bound numbers are
read per class because a 9-class mIoU is dominated by ``drivable``, which is
~half of every seen frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data.bev_raster import GRID_DEFAULT
from tanitad.data.semantic_map_gt import CART_SHAPE, CHANNELS, N_CHANNELS

__all__ = [
    "MAP_GRID_HW", "MAP_CLASSES", "N_MAP_CLASSES", "FRAC_SUM_TOL",
    "BEVEncoderConfig", "BEVEncoder", "MapHead", "BEVMapBranch",
    "map_soft_ce", "map_metrics",
]

#: the one spelling of the BEV output shape: the SAM3 label's own cells.
MAP_GRID_HW: tuple[int, int] = CART_SHAPE
MAP_CLASSES: tuple[str, ...] = CHANNELS
N_MAP_CLASSES: int = N_CHANNELS
#: per-cell fraction sums may drift by at most one uint8 step per channel.
FRAC_SUM_TOL: float = N_CHANNELS / 255.0

if MAP_GRID_HW != GRID_DEFAULT.shape:
    raise RuntimeError(
        f"the SAM3 grid {MAP_GRID_HW} and bev_raster.GRID_DEFAULT "
        f"{GRID_DEFAULT.shape} have diverged; the map head predicts on the "
        f"label's own cells, so these two cannot disagree")


# --------------------------------------------------------------------------- #
# the encoder                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BEVEncoderConfig:
    """Widths and dilations of :class:`BEVEncoder`.

    ``dilations`` is the receptive field, and it is the only knob that buys
    context: every layer is stride 1 (see the module docstring), so each 3x3
    conv adds exactly ``dilation`` cells on each side and the half-width is
    ``1`` (the stem) ``+ 2 * sum(dilations)`` (two convs per block). The default
    ``(1, 2, 4, 8)`` reaches +-**31 cells = 15.5 m** each way -- four lane widths
    (3.7 m), the context a cell needs to know it is drivable.
    """

    d_in: int = 128            # BEVLift.d_out
    d_model: int = 96
    dilations: tuple[int, ...] = (1, 2, 4, 8)
    d_out: int = 96
    norm_groups: int = 8

    def __post_init__(self) -> None:
        for name in ("d_in", "d_model", "d_out", "norm_groups"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not self.dilations:
            raise ValueError("dilations must have at least one entry")
        if any(int(d) < 1 for d in self.dilations):
            raise ValueError(f"dilations must be >= 1, got {self.dilations}")
        for w in (self.d_model, self.d_out):
            if int(w) % int(self.norm_groups):
                raise ValueError(f"width {w} must divide by norm_groups "
                                 f"{self.norm_groups}")

    @property
    def receptive_field_cells(self) -> int:
        """Half-width of the receptive field, in BEV cells: the stem's 3x3 (+1)
        plus two 3x3 at each dilation (+2d per block)."""
        return 1 + 2 * sum(int(d) for d in self.dilations)

    @property
    def receptive_field_m(self) -> float:
        return self.receptive_field_cells * float(GRID_DEFAULT.cell_m)


class _DilatedBlock(nn.Module):
    """Residual 3x3 -> 3x3 at one dilation. Stride 1, shape preserved exactly."""

    def __init__(self, d: int, dilation: int, groups: int):
        super().__init__()
        p = dilation                       # (3 - 1) // 2 * dilation
        self.conv1 = nn.Conv2d(d, d, 3, padding=p, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(groups, d)
        self.conv2 = nn.Conv2d(d, d, 3, padding=p, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(groups, d)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        return self.act(x + self.norm2(self.conv2(h)))


class BEVEncoder(nn.Module):
    """``[B, d_in, 120, 64] -> [B, d_out, 120, 64]``, no resampling anywhere.

    ⛔ The output shape is ASSERTED against :data:`MAP_GRID_HW` on every forward,
    not merely produced. A stride-1 stack cannot change the shape, which is
    exactly why the assertion is cheap and why a future edit that introduces a
    pool or an ``interpolate`` fails here instead of at the loss.
    """

    def __init__(self, cfg: BEVEncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or BEVEncoderConfig()
        c = self.cfg
        self.stem = nn.Sequential(
            nn.Conv2d(c.d_in, c.d_model, 3, padding=1, bias=False),
            nn.GroupNorm(c.norm_groups, c.d_model), nn.GELU())
        self.blocks = nn.ModuleList(
            [_DilatedBlock(c.d_model, int(d), c.norm_groups) for d in c.dilations])
        self.out = nn.Sequential(
            nn.Conv2d(c.d_model, c.d_out, 1, bias=False),
            nn.GroupNorm(c.norm_groups, c.d_out))
        self._assert_stride_one()

    def _assert_stride_one(self) -> None:
        for name, m in self.named_modules():
            if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.ConvTranspose2d)):
                raise RuntimeError(f"{name}: {type(m).__name__} resamples the BEV "
                                   f"grid; the map head predicts on the label's "
                                   f"own cells (module docstring)")
            if isinstance(m, nn.Conv2d):
                s = m.stride if isinstance(m.stride, tuple) else (m.stride,) * 2
                k = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,) * 2
                p = m.padding if isinstance(m.padding, tuple) else (m.padding,) * 2
                d = m.dilation if isinstance(m.dilation, tuple) else (m.dilation,) * 2
                if tuple(s) != (1, 1):
                    raise RuntimeError(f"{name}: stride {tuple(s)} != (1, 1)")
                want = tuple((kk - 1) // 2 * dd for kk, dd in zip(k, d))
                if tuple(p) != want:
                    raise RuntimeError(f"{name}: padding {tuple(p)} != {want}, so "
                                       f"the output would not be the label grid")

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, bev: Tensor) -> Tensor:
        if bev.dim() != 4:
            raise ValueError(f"bev must be [B, C, X, Y], got {tuple(bev.shape)}")
        if bev.shape[1] != self.cfg.d_in:
            raise ValueError(f"bev has {bev.shape[1]} channels, the encoder was "
                             f"built for {self.cfg.d_in} (BEVLift.d_out)")
        if tuple(bev.shape[2:]) != MAP_GRID_HW:
            raise ValueError(f"bev is {tuple(bev.shape[2:])}, the SAM3 label grid "
                             f"is {MAP_GRID_HW}: the lift geometry was built for a "
                             f"different grid")
        x = self.stem(bev)
        for blk in self.blocks:
            x = blk(x)
        x = self.out(x)
        if tuple(x.shape[2:]) != MAP_GRID_HW:      # unreachable by construction
            raise RuntimeError(f"BEVEncoder changed the grid to "
                               f"{tuple(x.shape[2:])} != {MAP_GRID_HW}")
        return x


class MapHead(nn.Module):
    """``[B, d, 120, 64] -> [B, 9, 120, 64]`` logits, one 1x1 conv.

    ⭐ Deliberately a SINGLE 1x1: the context lives in :class:`BEVEncoder`, and a
    deep head here would make the reported map quality a statement about the
    head rather than about what the lifted trunk features carry -- the
    ``BEVOccupancyHead`` ``PARAM_BAND`` precedent
    (``scripts/train_p8_occupancy.py:162-190``).
    """

    def __init__(self, d_in: int = 96, n_classes: int = N_MAP_CLASSES):
        super().__init__()
        self.n_classes = int(n_classes)
        self.proj = nn.Conv2d(int(d_in), self.n_classes, 1)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, feats: Tensor) -> Tensor:
        return self.proj(feats)


class BEVMapBranch(nn.Module):
    """:class:`BEVEncoder` + :class:`MapHead`, returning BOTH outputs.

    The BEV FEATURES (not the logits) are what the 3-D box head and the planner
    couplings consume, so this returns them explicitly rather than making a
    caller reach inside; a caller that only wanted logits would otherwise run the
    encoder twice.
    """

    def __init__(self, cfg: BEVEncoderConfig | None = None,
                 n_classes: int = N_MAP_CLASSES):
        super().__init__()
        self.encoder = BEVEncoder(cfg)
        self.head = MapHead(self.encoder.cfg.d_out, n_classes)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, bev: Tensor) -> dict:
        feats = self.encoder(bev)
        return {"bev_feats": feats, "map_logits": self.head(feats)}


# --------------------------------------------------------------------------- #
# the loss                                                                     #
# --------------------------------------------------------------------------- #
def map_soft_ce(logits: Tensor, frac: Tensor, seen: Tensor, *,
                check_sum: bool = True, tol: float = FRAC_SUM_TOL) -> dict:
    """Soft cross-entropy against the SAM3 fractions, on SEEN cells only.

    ``logits`` ``[B, 9, X, Y]`` · ``frac`` ``[B, 9, X, Y]`` in [0, 1]
    (``MapFrames.cart``) · ``seen`` ``[B, X, Y]`` bool (``MapFrames.seen``).

    Returns ``{"loss", "n_cells", "per_class", "n_per_class"}``. ``loss`` is the
    mean over seen cells of ``-sum_c p_c log softmax(logits)_c`` with ``p``
    renormalised to sum 1; ``per_class`` is the same quantity attributed to each
    class ``c`` (the ``p_c``-weighted term), reported separately for the reason
    :func:`map_metrics` reports per class.

    ⛔ Refuses (``ValueError``) a target whose per-cell fractions do not sum to 1
    within ``tol`` on a SEEN cell -- see the module docstring, decision 2. Pass
    ``check_sum=False`` only in a shape test with synthetic targets.
    """
    if logits.dim() != 4 or frac.dim() != 4 or seen.dim() != 3:
        raise ValueError(f"shapes must be logits [B,C,X,Y], frac [B,C,X,Y], seen "
                         f"[B,X,Y]; got {tuple(logits.shape)}, {tuple(frac.shape)}, "
                         f"{tuple(seen.shape)}")
    if logits.shape != frac.shape:
        raise ValueError(f"logits {tuple(logits.shape)} != frac {tuple(frac.shape)}")
    if tuple(seen.shape) != (logits.shape[0],) + tuple(logits.shape[2:]):
        raise ValueError(f"seen {tuple(seen.shape)} does not match logits "
                         f"{tuple(logits.shape)}")
    if seen.dtype != torch.bool:
        raise ValueError(f"seen must be bool, got {seen.dtype} -- a float mask "
                         f"would silently weight cells instead of selecting them")
    p = frac.to(logits.dtype)
    if bool((p < -1e-6).any()) or bool((p > 1.0 + 1e-6).any()):
        raise ValueError("SAM3 fractions outside [0, 1]: the target is not a "
                         "uint8/255 fraction array")
    s = p.sum(dim=1)                                        # [B, X, Y]
    m = seen
    if check_sum and bool(m.any()):
        bad = ((s - 1.0).abs() > float(tol)) & m
        if bool(bad.any()):
            worst = float((s - 1.0).abs()[m].max())
            raise ValueError(
                f"{int(bad.sum())} of {int(m.sum())} seen cells have class "
                f"fractions summing to 1 +- {worst:.4f}, outside the uint8 "
                f"quantisation bound {tol:.4f}: this is not a rounding residue")
    p = p / s.clamp_min(1e-6).unsqueeze(1)
    logp = F.log_softmax(logits, dim=1)
    per_cell_c = -(p * logp)                                # [B, C, X, Y]
    mf = m.unsqueeze(1).to(logits.dtype)
    n = int(m.sum())
    denom = float(max(n, 1))
    per_class = (per_cell_c * mf).sum(dim=(0, 2, 3)) / denom
    loss = per_class.sum() if n else logits.sum() * 0.0
    # cells whose label mass is in class c, for the per-class denominators
    n_per_class = (p * mf).sum(dim=(0, 2, 3)).detach()
    return {"loss": loss, "n_cells": n,
            "per_class": per_class.detach(), "n_per_class": n_per_class}


@torch.no_grad()
def map_metrics(logits: Tensor, frac: Tensor, seen: Tensor) -> dict:
    """Per-class IoU / frequencies on SEEN cells, never pooled.

    The label class of a cell is its ARGMAX fraction (the hard label the soft
    target implies); ``iou`` is then the usual intersection-over-union of the
    predicted argmax against it. ``freq_label`` / ``freq_pred`` are reported
    beside every IoU because a class at 0.4 % of cells has an IoU that is noise,
    and a pooled mIoU would hide which classes those are.
    """
    if not bool(seen.any()):
        z = torch.zeros(logits.shape[1], dtype=torch.float64)
        return {"n_cells": 0, "iou": z, "freq_label": z.clone(),
                "freq_pred": z.clone(), "acc": float("nan"),
                "classes": MAP_CLASSES[:logits.shape[1]]}
    C = logits.shape[1]
    m = seen
    pred = logits.argmax(dim=1)[m]                          # [n]
    lab = frac.argmax(dim=1)[m]
    n = int(pred.numel())
    iou, fl, fp = [], [], []
    for c in range(C):
        pc, lc = pred == c, lab == c
        inter = int((pc & lc).sum())
        union = int((pc | lc).sum())
        iou.append(inter / union if union else float("nan"))
        fl.append(int(lc.sum()) / n)
        fp.append(int(pc.sum()) / n)
    return {"n_cells": n,
            "iou": torch.tensor(iou, dtype=torch.float64),
            "freq_label": torch.tensor(fl, dtype=torch.float64),
            "freq_pred": torch.tensor(fp, dtype=torch.float64),
            "acc": float((pred == lab).to(torch.float64).mean()),
            "classes": MAP_CLASSES[:C]}


if __name__ == "__main__":
    br = BEVMapBranch()
    c = br.encoder.cfg
    print(f"BEVEncoder(d_in={c.d_in}, d_model={c.d_model}, dilations="
          f"{c.dilations}) receptive field +-{c.receptive_field_cells} cells "
          f"(+-{c.receptive_field_m:.1f} m)")
    print(f"  encoder {br.encoder.n_params:,} + head {br.head.n_params:,} "
          f"= {br.n_params:,} parameters")
