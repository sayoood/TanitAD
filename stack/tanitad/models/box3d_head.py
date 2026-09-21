"""refcv6 §6: the 3-D BOX head -- the existing agent seam, given z and h.

``SPEC_REFCV6_V2.md`` §6, verbatim:

    | **BOX (3-D)** | stride-16 tokens (+ BEV features) | ``obstacle.offline``
    | **cuboids** (87,481 measured, ground-standing bottom faces) | Hungarian
    | matching; (x, y, z, l, w, h, yaw) + class. Today's seam emits
    | (cx, cy, l, w, yaw): **add z and h**

⛔ **This is the SAME seam, widened -- not a second detector.** The brief's rule
("extend the existing ``refc_agents.py`` seam rather than inventing a second one;
keep its decode contract intact") is enforced mechanically, not by convention:

* :data:`SLOT3D_FIELDS` is :data:`agent_slots.SLOT_FIELDS` with ``cz`` and ``h``
  **appended**, so every existing slice keeps its offset and a 2-D checkpoint's
  head weights occupy the same columns they always did;
* :meth:`Box3DSlotDecoder.decode` returns **every key**
  :meth:`agent_slots.AgentSlotDecoder.decode` returns, with the same shapes and
  the same meanings -- ``box`` is still ``[B, N, 4] = (cx, cy, l, w)`` in metres
  -- and ADDS ``box3d``, ``cz`` and ``h``. ``tests/test_refcv6_perception.py``
  asserts the 2-D keys are bit-identical to the base decoder's on the shared
  columns;
* the **matcher is untouched**. :func:`agent_slots.match_slots` costs
  ``presence + cls + centre + size`` in the BEV plane and is reused unchanged, so
  turning 3-D on does not silently change WHICH slot matches WHICH agent. z and
  h enter the LOSS only. (Adding them to the cost is a registered knockout, not
  the default -- see :data:`MATCH_INCLUDES_Z`.)

## Where z and h come from, and what happens when they do not

The FIRST eval agent join (``2026-09-06-b1-agent-join``) carried
``cx, cy, yaw, l, w, occ, track_id, cls`` and **no z, no h** -- MEASURED on
``b1eval_agents.jsonl.xz`` 2026-09-16. The cuboids in
``labels/obstacle.offline/*.parquet`` do carry ``center_z`` and ``size_z``;
:mod:`tanitad.data.agent_cuboid_gt` is the reader that recovers them.

⭐ SUPERSEDED 2026-09-17 (``b1-agent-join-3d-20260917``, landed 24065b6): the
3-D join DOES carry them, under the join's own spellings ``cz`` and ``h`` --
**not** the parquet's ``center_z``/``size_z``, which is a trap a probe of mine
walked into. MEASURED on ``b1eval_agents_3d.jsonl.xz``: 139 clips, 26,394 lines,
**905,512 / 905,512 agent-frames carry z+h (100.00 %)**, cuboid bottoms a median
**-0.103 m** off the ego ground plane. The last hop -- join to ``zh_targets`` to
``box3d_set_loss`` with ``n["z"] > 0`` -- is verified END TO END by
``tests/test_refcv6_perception_realdata.py::test_the_3d_join_reaches_the_height_targets``
(green on the artifact, RED under a mutation that makes the join answer
all-False, skipped when ``$TANITAD_AGENT_JOIN3D`` is unset).

⛔ Until a join carries them, ``tgt["zh_mask"]`` is **all False** and the z/h
terms are computed over ZERO items and SAY SO (``n_z == 0`` in the returned
counts) -- the ``rates_mask`` rule from ``slot_set_loss``: *"a missing rate is
MASKED, never zero-filled"*. A ground-standing prior (``z = h/2``) is available
as :func:`ground_standing_z` for the ORACLE rung, and is flagged inadmissible as
a capability claim wherever it is used, because it is a label-derived constant,
not a prediction.

## The self-test that reads a KNOWN value

:func:`box3d_ap` is average precision over a matched detection set at a 3-D
centre-distance threshold. The brief requires it to read a known value:
``tests/test_refcv6_perception.py`` pins **perfect predictions -> AP exactly 1.0**
and **random predictions -> AP within sampling error of the base rate**, where
the base rate is computed in closed form from the threshold and the sampling box
(:func:`random_ap_base_rate`), NOT read off the same run it is compared with.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data.bev_raster import GRID_DEFAULT
from tanitad.models.agent_slots import (
    AGENT_CLASSES, N_QUERIES_DEFAULT, SLOT_FIELDS, SLOT_SLICES, SLOT_WIDTH,
    AgentSlotDecoder, SlotDecodeRanges, match_slots, slot_set_loss,
)

__all__ = [
    "SLOT3D_FIELDS", "SLOT3D_SLICES", "SLOT3D_WIDTH", "N_EXTRA_3D",
    "MATCH_INCLUDES_Z", "Z_RANGE_M", "H_RANGE_M", "BOX3D_LOSS_W",
    "Box3DSlotDecoder", "Box3DMemory", "box3d_set_loss", "box3d_ap",
    "random_ap_base_rate", "ground_standing_z", "zh_targets",
]

#: ⛔ APPENDED, never interleaved: every :data:`agent_slots.SLOT_SLICES` offset
#: survives, so the 2-D decode is the same arithmetic on the same columns.
N_EXTRA_3D: int = 2
SLOT3D_FIELDS: tuple[tuple[str, int], ...] = SLOT_FIELDS + (("cz", 1), ("h", 1))
SLOT3D_WIDTH: int = SLOT_WIDTH + N_EXTRA_3D


def _slices3d() -> dict[str, slice]:
    out, off = {}, 0
    for name, n in SLOT3D_FIELDS:
        out[name] = slice(off, off + n)
        off += n
    return out


SLOT3D_SLICES: dict[str, slice] = _slices3d()
for _k, _v in SLOT_SLICES.items():          # the contract, asserted at import
    if SLOT3D_SLICES[_k] != _v:
        raise RuntimeError(
            f"3-D widening moved the 2-D field {_k!r} from {_v} to "
            f"{SLOT3D_SLICES[_k]}: a 2-D checkpoint's head columns would no "
            f"longer mean what they meant")
del _k, _v

#: ⛔ OFF by default. The matcher stays :func:`agent_slots.match_slots` so that
#: switching 3-D on cannot silently change the assignment (module docstring).
MATCH_INCLUDES_Z: bool = False

#: Decode ranges for the two new channels, in metres.
#: MEASURED 2026-09-16 on 51,844 cuboids of 12 ``obstacle_offline_b1eval``
#: clips: ``center_z`` p1/p50/p99 = -1.637 / 0.931 / 3.668 m, ``size_z`` p1/p50/
#: p99 = 1.387 / 1.594 / 3.990 m. ⚠️ These are DECODE CONSTANTS, not clamps --
#: the same rule :class:`agent_slots.SlotDecodeRanges` carries: a head that
#: CANNOT express a 4.5 m truck would report a systematic error as a success.
Z_RANGE_M: float = 4.0
H_RANGE_M: float = 2.0

#: Loss weights for the two new terms, in METRES like ``centre``/``size``.
BOX3D_LOSS_W: dict[str, float] = {"z": 1.0, "h": 1.0}


# --------------------------------------------------------------------------- #
# memory: stride-16 tokens (+ BEV features)                                    #
# --------------------------------------------------------------------------- #
class Box3DMemory(nn.Module):
    """Build the decoder memory from the **stride-16** trunk map and the BEV
    features, as ``SPEC_REFCV6_V2.md`` §6 specifies ("stride-16 tokens (+ BEV
    features)").

    ``image`` ``[B, C_img, *image_hw]`` -> ``image_hw`` tokens, and ``bev``
    ``[B, C_bev, 120, 64]`` -> ``bev_tokens_hw`` tokens by **average pooling**
    (the map head's own prediction happens at full resolution upstream; this is
    the box head's read of it, and a 7,680-token memory would make the decoder
    the experiment).

    ⛔ **``image_hw`` and ``d_image`` are REQUIRED, and neither has a default
    (PI 2026-09-16).** The stride-16 map is 16 x 40 on the 256 x 640 cache and
    **16 x 64** on the 256 x 1024 one; the channel count is **256** for
    ``resnet34`` and **1024** for ``resnet101``. Build both from
    :class:`tanitad.models.trunk_shapes.TrunkSpec`::

        spec = TrunkSpec.from_timm("resnet101", FRAME_256x1024)
        mem = Box3DMemory(d_image=spec.perception.channels, d_bev=96,
                          d_model=256, image_hw=spec.perception.hw)

    A default here is how a head silently pins one geometry: the positional
    table is per token, so a wrong ``image_hw`` is a geometry mismatch, not a
    resize -- which is why :meth:`forward` refuses rather than interpolates.

    ⚠️ Pooling here is legitimate and pooling in :class:`bev_encoder.BEVEncoder`
    is not, because nothing downstream of THIS pool is compared to a per-cell
    label. The two facts are kept apart on purpose.

    ⛔ ONE argument family, both vision. There is no parameter through which a
    label could arrive -- the ``AgentSlotDecoder.forward`` audit, extended.
    """

    def __init__(self, d_image: int, d_bev: int, d_model: int, *,
                 image_hw: tuple[int, int],
                 bev_tokens_hw: tuple[int, int] = (30, 16),
                 use_bev: bool = True,
                 perception_stride: int = 16):
        super().__init__()
        if image_hw is None or len(tuple(image_hw)) != 2:
            raise ValueError(
                "image_hw is required: refcv6 runs 16x40 (256x640 cache) and "
                "16x64 (256x1024, PI 2026-09-16). Take it from "
                "TrunkSpec.perception.hw, never a literal.")
        self.perception_stride = int(perception_stride)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.bev_tokens_hw = (int(bev_tokens_hw[0]), int(bev_tokens_hw[1]))
        self.use_bev = bool(use_bev)
        self.d_model = int(d_model)
        self.img_proj = nn.Linear(int(d_image), self.d_model)
        self.bev_proj = nn.Linear(int(d_bev), self.d_model) if self.use_bev else None
        n_img = self.image_hw[0] * self.image_hw[1]
        n_bev = self.bev_tokens_hw[0] * self.bev_tokens_hw[1] if self.use_bev else 0
        self.n_tokens = n_img + n_bev
        self.src_embed = nn.Parameter(torch.zeros(1, 2, self.d_model))
        nn.init.trunc_normal_(self.src_embed, std=0.02)

    def forward(self, image: Tensor, bev: Tensor | None = None) -> Tensor:
        if image.shape[1] != self.img_proj.in_features:
            raise ValueError(
                f"image feature map has {image.shape[1]} channels, the box head "
                f"was built for {self.img_proj.in_features}. ⛔ The channel "
                f"count is a PARAMETER (PI 2026-09-16): resnet34 gives 256 at "
                f"stride 16, resnet101 gives 1024. Read it from "
                f"TrunkSpec.perception.channels, never a literal.")
        if tuple(image.shape[2:]) != self.image_hw:
            got_h, got_w = int(image.shape[2]), int(image.shape[3])
            halved = (got_h * 2, got_w * 2) == self.image_hw
            raise ValueError(
                f"image feature map is {got_h}x{got_w}, the box head was built "
                f"for {self.image_hw[0]}x{self.image_hw[1]}"
                + (f". ⛔ That is HALF the height and width: this looks like the "
                   f"STRIDE-{2 * self.perception_stride} map. refcv6 perception "
                   f"reads stride {self.perception_stride} -- an oracle on the "
                   f"stride-32 map caps at AP 0.3341 vs 0.4713 "
                   f"(SPEC_REFCV6_V2.md §2)" if halved else
                   f". The positional table is per token, so this is a geometry "
                   f"mismatch, not a resize. 16x40 is the 256x640 cache and "
                   f"16x64 the 256x1024 one (PI 2026-09-16) -- rebuild the head "
                   f"from TrunkSpec.perception.hw."))
        tok = self.img_proj(image.flatten(2).transpose(1, 2))
        tok = tok + self.src_embed[:, 0:1]
        if not self.use_bev:
            return tok
        if bev is None:
            raise ValueError("this Box3DMemory was built with use_bev=True but "
                             "no BEV features were passed")
        bt = F.adaptive_avg_pool2d(bev, self.bev_tokens_hw)
        bt = self.bev_proj(bt.flatten(2).transpose(1, 2)) + self.src_embed[:, 1:2]
        return torch.cat([tok, bt], dim=1)


# --------------------------------------------------------------------------- #
# the decoder                                                                  #
# --------------------------------------------------------------------------- #
class Box3DSlotDecoder(AgentSlotDecoder):
    """:class:`agent_slots.AgentSlotDecoder` widened to 3-D cuboids.

    Everything except the head width and :meth:`decode` is inherited, so the
    query table, the transformer, the presence prior and the parameter band are
    literally the same code -- the duplication that made two geometries drift
    apart in the ``advect`` precedent cannot happen here.
    """

    def __init__(self, d_memory: int, n_memory: int, *,
                 n_queries: int = N_QUERIES_DEFAULT, d_model: int = 256,
                 depth: int = 3, n_heads: int = 8,
                 ranges: SlotDecodeRanges | None = None,
                 z_range_m: float = Z_RANGE_M, h_range_m: float = H_RANGE_M,
                 enforce_band: bool = True):
        super().__init__(d_memory, n_memory, n_queries=n_queries,
                         d_model=d_model, depth=depth, n_heads=n_heads,
                         ranges=ranges, enforce_band=False)
        self.z_range_m, self.h_range_m = float(z_range_m), float(h_range_m)
        old = self.head
        self.head = nn.Linear(self.d_model, SLOT3D_WIDTH)
        with torch.no_grad():                 # keep the 2-D init exactly
            self.head.weight[:SLOT_WIDTH] = old.weight
            self.head.bias[:SLOT_WIDTH] = old.bias
            # h is softplus'd: bias it at the measured median height, 1.594 m
            self.head.bias[SLOT3D_SLICES["h"]] = math.log(
                math.expm1(1.594 / self.h_range_m))
            self.head.bias[SLOT3D_SLICES["cz"]] = 0.0
        if enforce_band:
            from tanitad.models.agent_slots import PARAM_BAND
            n = self.n_params
            if not (PARAM_BAND[0] <= n <= PARAM_BAND[1]):
                raise ValueError(
                    f"Box3DSlotDecoder has {n:,} params -- outside the §6 "
                    f"pre-registered band {PARAM_BAND} (the AgentSlotDecoder "
                    f"rule, unchanged: a bigger head stops measuring what the "
                    f"LATENT carries). Pass enforce_band=False for shape tests.")

    def decode(self, raw: Tensor) -> dict:
        """2-D contract + ``box3d``/``cz``/``h``.

        ``box3d`` ``[B, N, 7]`` = ``(x, y, z, l, w, h, yaw)`` in metres/radians,
        the order ``SPEC_REFCV6_V2.md`` §6 names. ``h`` is ``softplus``-decoded
        (a height is positive, exactly as ``l``/``w`` are); ``z`` is linear,
        because a cuboid centre below the road plane is a real state on a
        descending grade.
        """
        out = super().decode(raw[..., :SLOT_WIDTH])
        s = SLOT3D_SLICES
        cz = raw[..., s["cz"]].squeeze(-1) * self.z_range_m
        h = F.softplus(raw[..., s["h"]].squeeze(-1)) * self.h_range_m
        out["raw"] = raw                       # the FULL row, not the 2-D slice
        out["cz"], out["h"] = cz, h
        b = out["box"]
        out["box3d"] = torch.stack(
            [b[..., 0], b[..., 1], cz, b[..., 2], b[..., 3], h, out["yaw"]],
            dim=-1)
        return out


# --------------------------------------------------------------------------- #
# targets and the loss                                                         #
# --------------------------------------------------------------------------- #
def ground_standing_z(h: Tensor | np.ndarray, *, bottom_m: float = 0.0):
    """``z = bottom_m + h/2`` -- the ground-standing prior.

    ⚠️ **Inadmissible as a capability claim.** MEASURED 2026-09-16 on 51,844
    ``obstacle_offline_b1eval`` cuboids, the bottom-face median is 0.039 m
    (automobile), 0.118 m (person), 0.051 m (heavy_truck) -- ground-standing to
    ~0.1 m. That makes this a good ORACLE rung and a good sanity bound; it is
    label-derived arithmetic, not a prediction, and any arm that uses it is
    marked so wherever it is reported.
    """
    return float(bottom_m) + h / 2.0


def zh_targets(tgt: dict, cz=None, h=None, *, mask=None) -> dict:
    """Attach ``cz``/``h``/``zh_mask`` to a 2-D target dict IN A COPY.

    ``tgt`` is what :func:`agent_slots.targets_from_join` returns. With ``cz``
    and ``h`` omitted the mask is all-False and the z/h terms are then computed
    over zero items and say so -- never zero-filled (module docstring).
    """
    out = dict(tgt)
    box = tgt["box"]
    shape = box.shape[:2]
    dev, dt = box.device, box.dtype
    if cz is None or h is None:
        out["cz"] = torch.zeros(shape, device=dev, dtype=dt)
        out["h"] = torch.zeros(shape, device=dev, dtype=dt)
        out["zh_mask"] = torch.zeros(shape, device=dev, dtype=torch.bool)
        return out
    out["cz"] = torch.as_tensor(cz, device=dev, dtype=dt).reshape(shape)
    out["h"] = torch.as_tensor(h, device=dev, dtype=dt).reshape(shape)
    m = (torch.ones(shape, device=dev, dtype=torch.bool) if mask is None
         else torch.as_tensor(mask, device=dev, dtype=torch.bool).reshape(shape))
    out["zh_mask"] = m & tgt["valid"]
    return out


def box3d_set_loss(pred: dict, tgt: dict, *, match: dict | None = None,
                   weights: dict | None = None,
                   cls_class_weight=None) -> dict:
    """:func:`agent_slots.slot_set_loss` + the ``z`` and ``h`` terms.

    ⭐ The 2-D part is CALLED, not re-implemented, so every term, mask and
    reporting rule it carries is the same one. The additions are two metre-unit
    L1 terms under ``tgt["zh_mask"]``, reported with their own counts.

    ⛔ With no 3-D labels (``zh_mask`` all False, the state of today's eval join)
    ``loss_z`` and ``loss_h`` are ``0.0`` with ``n["z"] == n["h"] == 0``, and
    ``total`` is then EXACTLY the 2-D total -- asserted by
    ``tests/test_refcv6_perception.py::test_no_zh_labels_is_the_2d_loss_exactly``.
    """
    w = {**BOX3D_LOSS_W, **(weights or {})}
    m = match or match_slots(pred, tgt)
    out = slot_set_loss(pred, tgt, match=m,
                        weights={k: v for k, v in (weights or {}).items()
                                 if k not in BOX3D_LOSS_W},
                        cls_class_weight=cls_class_weight)
    if "cz" not in pred or "h" not in pred:
        raise ValueError("box3d_set_loss needs a 3-D prediction (keys 'cz', "
                         "'h'); this looks like a 2-D AgentSlotDecoder output")
    zm = tgt.get("zh_mask")
    if zm is None:
        raise ValueError("tgt has no 'zh_mask' -- run zh_targets(tgt, ...) "
                         "first; an absent 3-D label is a MASK, not a zero")
    dev = pred["cz"].device
    acc_z = pred["cz"].new_zeros(())
    acc_h = pred["h"].new_zeros(())
    n_z = 0
    for b in range(int(pred["cz"].shape[0])):
        r, c = m["rows"][b].to(dev), m["cols"][b].to(dev)
        if r.numel() == 0:
            continue
        keep = zm[b][c]
        if not bool(keep.any()):
            continue
        acc_z = acc_z + (pred["cz"][b][r][keep] - tgt["cz"][b][c][keep]).abs().sum()
        acc_h = acc_h + (pred["h"][b][r][keep] - tgt["h"][b][c][keep]).abs().sum()
        n_z += int(keep.sum())
    out["loss_z"] = acc_z / max(n_z, 1)
    out["loss_h"] = acc_h / max(n_z, 1)
    out["total"] = out["total"] + w["z"] * out["loss_z"] + w["h"] * out["loss_h"]
    out["n"] = {**out["n"], "z": n_z, "h": n_z}
    out["_weights"] = {**out["_weights"], **w}
    return out


# --------------------------------------------------------------------------- #
# AP -- the self-test that must read a KNOWN value                             #
# --------------------------------------------------------------------------- #
def box3d_ap(pred: dict, tgt: dict, *, dist_thresh_m: float = 2.0,
             use_z: bool = True, score: str = "presence") -> dict:
    """Average precision of the 3-D boxes at a CENTRE-DISTANCE threshold.

    Greedy score-ordered matching (the nuScenes centre-distance convention, not
    3-D IoU: IoU at these ranges is dominated by size error and would hide a
    metre of range error behind a large box). One target per detection, one
    detection per target, per batch element.

    ``AP`` is the area under the precision-recall curve by the **all-point
    interpolation** ``sum_k (r_k - r_{k-1}) * p_k`` -- so a run in which every
    target is matched before any false positive reads EXACTLY 1.0, which is the
    known value :func:`box3d_ap`'s self-test reads.

    ``use_z``: measure the distance in 3-D (x, y, z) or in the BEV plane.
    ``score``: ``"presence"`` (sigmoid of the presence logit) or ``"cls"``
    (1 - softmax probability of a no-object column; unused today, kept so the
    caller never invents a second scoring rule at the call site).
    """
    per_elem, n_gt_elem = box3d_match_rows(pred, tgt, dist_thresh_m=dist_thresh_m,
                                           use_z=use_z, score=score)
    rows = [r for elem in per_elem for r in elem]
    return ap_from_rows(rows, sum(n_gt_elem))


def box3d_match_rows(pred: dict, tgt: dict, *, dist_thresh_m: float = 2.0,
                     use_z: bool = True, score: str = "presence"):
    """:func:`box3d_ap`'s greedy matching, per batch element, exposed.

    Returns ``(per_elem, n_gt_elem)``: ``per_elem[b]`` is the list of
    ``(conf, hit, pred_slot, gt_slot)`` rows in the matching order, with ``gt_slot``
    the matched target's index in ``tgt``'s slot axis (``-1`` for a miss), and
    ``n_gt_elem[b]`` is that element's number of valid targets.

    ⭐ Split out (2026-09-19, PREREG_S1 S1A.4) so a caller can (a) take the AP's
    OWN matched pairs, e.g. for a velocity error, rather than re-deriving a second
    matcher, and (b) pool rows over a RESAMPLED set of elements for an
    episode-cluster bootstrap. :func:`box3d_ap` is recomposed from this and
    :func:`ap_from_rows` and returns exactly what it returned before (pinned by
    ``tests/test_box3d_match_rows.py``).
    """
    if score not in ("presence", "cls"):
        raise ValueError(f"score must be 'presence' or 'cls', got {score!r}")
    key = "box3d" if use_z else "box"
    if key not in pred:
        raise ValueError(f"pred has no {key!r}")
    B = int(pred[key].shape[0])
    per_elem, n_gt_elem = [], []
    with torch.no_grad():
        conf_all = (pred["presence_logit"].sigmoid() if score == "presence"
                    else pred["cls_logits"].softmax(-1).max(-1).values)
        for b in range(B):
            rows = []
            valid = tgt["valid"][b].nonzero(as_tuple=False).flatten()
            n_gt_elem.append(int(valid.numel()))
            if use_z:
                pc = pred["box3d"][b][:, :3]
                tc = torch.stack([tgt["box"][b][valid][:, 0],
                                  tgt["box"][b][valid][:, 1],
                                  tgt["cz"][b][valid]], dim=-1)
            else:
                pc = pred["box"][b][:, :2]
                tc = tgt["box"][b][valid][:, :2]
            conf = conf_all[b]
            order = torch.argsort(conf, descending=True)
            taken = torch.zeros(valid.numel(), dtype=torch.bool)
            for i in order.tolist():
                if valid.numel() == 0:
                    rows.append((float(conf[i]), 0, i, -1))
                    continue
                d = (tc - pc[i][None, :]).norm(dim=-1)
                d = torch.where(taken.to(d.device), torch.full_like(d, float("inf")), d)
                j = int(torch.argmin(d))
                hit = float(d[j]) <= float(dist_thresh_m)
                if hit:
                    taken[j] = True
                rows.append((float(conf[i]), 1 if hit else 0, i,
                             int(valid[j]) if hit else -1))
            per_elem.append(rows)
    return per_elem, n_gt_elem


def ap_from_rows(rows, n_gt: int) -> dict:
    """All-point-interpolated AP from pooled ``(conf, hit, ...)`` rows -- the
    arithmetic :func:`box3d_ap` has always used, unchanged. The sort is stable,
    so pooled rows in element order give exactly the historical result."""
    if n_gt == 0:
        return {"ap": float("nan"), "n_gt": 0, "n_pred": len(rows),
                "precision": [], "recall": []}
    rows = sorted(rows, key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows], dtype=np.float64)
    fp = np.cumsum([1 - r[1] for r in rows], dtype=np.float64)
    rec = tp / float(n_gt)
    prec = tp / np.maximum(tp + fp, 1e-12)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return {"ap": ap, "n_gt": n_gt, "n_pred": len(rows),
            "precision": prec.tolist(), "recall": rec.tolist()}


def random_ap_base_rate(n_gt: int, n_pred: int, dist_thresh_m: float,
                        extent_m: tuple[float, float, float]) -> float:
    """Closed-form expected precision of UNIFORM random boxes -- the known value
    the random half of the self-test is compared against.

    With ``n_gt`` targets uniform in a box of side lengths ``extent_m`` and a
    detection placed uniformly in the same box, the chance that SOME target
    falls inside the detection's threshold ball is
    ``1 - (1 - v_ball / v_box)^n_gt`` with ``v_ball`` the volume of the
    ``dist_thresh_m`` ball (or disc, when ``extent_m`` has a zero side). AP of a
    score-ordered list whose hits are independent of the score converges to that
    hit rate, because precision is then flat in recall.

    ⚠️ A BOUND, not a sample statistic: the ball may leave the box near an edge,
    so this OVERSTATES the rate slightly and the test asserts the measured AP is
    below it and above a floor -- never that it equals a number read off the
    same run it is checking.
    """
    ex = [float(e) for e in extent_m]
    r = float(dist_thresh_m)
    dims = [e for e in ex if e > 0]
    v_box = 1.0
    for e in dims:
        v_box *= e
    if len(dims) == 3:
        v_ball = 4.0 / 3.0 * math.pi * r ** 3
    elif len(dims) == 2:
        v_ball = math.pi * r ** 2
    else:
        v_ball = 2.0 * r
    p_one = min(v_ball / v_box, 1.0)
    return float(1.0 - (1.0 - p_one) ** max(int(n_gt), 0))


if __name__ == "__main__":
    from tanitad.models.trunk_shapes import (
        FRAME_256x640, FRAME_256x1024, TrunkSpec,
    )
    for frame, tag in ((FRAME_256x640, "256x640"), (FRAME_256x1024, "256x1024")):
        for trunk in ("resnet34", "resnet101"):
            spec = TrunkSpec.from_timm(trunk, frame)
            p = spec.perception
            mem = Box3DMemory(d_image=p.channels, d_bev=96, d_model=256,
                              image_hw=p.hw)
            dec = Box3DSlotDecoder(d_memory=256, n_memory=mem.n_tokens,
                                   enforce_band=False)
            print(f"{tag:9s} {trunk:10s} s{p.stride} {p.channels:5d}ch "
                  f"{p.hw[0]}x{p.hw[1]} -> memory {mem.n_tokens} tokens "
                  f"({sum(q.numel() for q in mem.parameters()):,} p) + "
                  f"decoder {dec.n_params:,} p")
    print(f"slot width {SLOT_WIDTH} -> {SLOT3D_WIDTH}; "
          f"AGENT_CLASSES ({len(AGENT_CLASSES)}); "
          f"BEV grid {GRID_DEFAULT.shape} @ {GRID_DEFAULT.cell_m} m (UNMOVED)")
