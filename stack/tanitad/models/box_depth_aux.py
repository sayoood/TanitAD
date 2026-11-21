"""refcv9 C3 -- arm B1: the TRAINING-ONLY box-derived depth auxiliary head on the trunk's s8 and s16 maps (REFCV9_DESIGN 2.3 "Box-derived
foreground depth (B1)", 4.3 row B1, 6b D4).  The flag contract with the trainer lives in ``tanitad/train/box_depth_aux_flag.py``.

WHAT IT IS (Sparse4Dv2's form: an auxiliary on the image features, removed at inference).  Two small conv branches read the trunk's
stride-8 map (the newest frame; the s8 tap that ``--map-hires on`` enables) and its stride-16 map (temporally fused) and emit LID
logits over ``N_BINS`` = 80 bins on 0-100 m at their own grids (52 x 128 and 26 x 64 on the 416 x 1024 frame).  The target is the
E0g builder's (``tanitad/data/box_depth_targets.py``, a pinned byte copy): every ``obstacle.offline`` cuboid of the batch's own
join3d labels is ray-cast through the CYLINDRICAL model into per-pixel nearest-surface range, LID-binned, TWO-HOT soft, aggregated
to cells (FG iff >= 50 % foreground pixels, target = the mean of the cell's pixel targets).

PINNED LOSS FORM (do not change here): per stride, ``alpha_fg * mean_{FG cells} soft-CE``; the two strides averaged.
* ``alpha_fg`` = 3.25 is therefore a CONSTANT LOSS SCALE; ``alpha_bg`` = 0.25 is INERT with box-only depth (D4 dropped LiDAR): a
  background cell carries NO depth label, its target is all zeros, so its CE is exactly 0 whatever its weight.  No background label
  is invented (the road-plane ray is a refcv10 question).  The loss therefore touches OBJECT cells only, by construction; the
  tests pin it (logits perturbed on non-FG cells leave the loss and every gradient unchanged).
* Depth definition: the card's default ``range`` (Euclidean along the pixel ray; what a LiDAR measures).
* Near-field dominance (the E0g card's decision 3) is DISCLOSED, not corrected: a per-cell CE is a near-field loss (MEASURED share
  per LID bin: ``raw/B1_BIN_SHARE.json``).

C3B: the targets are built OFF the training step -- ``train/box_depth_aux_flag.item_targets`` calls ``boxes_from_batch`` + ``build_targets`` on ONE
dataset item inside the DataLoader worker (``V3Dataset.__getitem__``), and the step only moves the collated tensors (the batch build below is the
reference the KA compares against, byte for byte).

``targets_from_batch`` reads the cuboids ALREADY in the batch (``agent_box`` (cx, cy, l, w), ``agent_yaw``, ``agent_cz``, ``agent_h``,
masked by ``agent_valid & agent_zh_mask`` on labelled rows): no new data.  The camera is the clip's own ``RigCamera`` (the bank
``model._rig_camera`` resolves per row).  Pixels the trunk never sees (the bottom ``equalize_bottom_rows`` rows are zeroed on every
forward) are passed as the builder's ``observed=False``: a cell touching them is IGNORE, never a target.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tanitad.data import box_depth_targets as BD

N_BINS = BD.N_BINS                 # 80 (free in the design; the builder's default)
ALPHA_FG = BD.ALPHA_FG             # 3.25 -- a constant loss scale under box-only depth
ALPHA_BG = BD.ALPHA_BG             # 0.25 -- INERT with box-only depth (a background cell has no label)
STRIDES = BD.STRIDES               # (8, 16)
DEPTH_KIND = "range"               # the card's default (decision 1)
MID = 128                          # branch width (UNPINNED by the design; mine)
KEYS_NEEDED = ("agent_box", "agent_yaw", "agent_valid", "agent_cz", "agent_h", "agent_zh_mask", "agent_ep")


class _Branch(nn.Sequential):
    """1x1 reduce -> 3x3 -> 1x1 to the LID logits (no normalisation layer: no running statistics, no batch coupling)."""

    def __init__(self, c_in: int, mid: int, n_bins: int):
        super().__init__(nn.Conv2d(c_in, mid, 1), nn.ReLU(inplace=False), nn.Conv2d(mid, mid, 3, padding=1),
                         nn.ReLU(inplace=False), nn.Conv2d(mid, n_bins, 1))


class BoxDepthAuxHead(nn.Module):
    """LID depth logits at s8 and s16.  Called ONLY by the trainer's loss -- never by ``RefCV3Model.forward`` (training-only)."""

    def __init__(self, c8: int, c16: int, *, mid: int = MID, n_bins: int = N_BINS):
        super().__init__()
        self.c8, self.c16, self.mid, self.n_bins = int(c8), int(c16), int(mid), int(n_bins)
        self.s8 = _Branch(self.c8, self.mid, self.n_bins)
        self.s16 = _Branch(self.c16, self.mid, self.n_bins)

    def forward(self, fmap_s8: Tensor, fmap_s16: Tensor) -> dict:
        return {8: self.s8(fmap_s8), 16: self.s16(fmap_s16)}


# --------------------------------------------------------------------------------------------------------------- #
# targets from the batch                                                                                          #
# --------------------------------------------------------------------------------------------------------------- #
def boxes_from_batch(batch: dict):
    """-> ([B] list of [M_b, 7] float64 (cx, cy, cz, l, w, h, yaw) in the rig frame, stats).  A row is used iff ``agent_label`` is True
    (NO_LABEL is not 'clear'); a cuboid iff ``agent_valid & agent_zh_mask`` (no z/h label -> it cannot be ray-cast: COUNTED)."""
    miss = [k for k in KEYS_NEEDED if k not in batch]
    if miss:
        raise SystemExit("[box-depth-aux] the batch carries no %s: the targets read the box3d cuboids already in the batch "
                         "(--agent-join + --join3d); refuse, do not skip" % miss)
    box = batch["agent_box"].detach().cpu().double().numpy()
    yaw = batch["agent_yaw"].detach().cpu().double().numpy()
    cz = batch["agent_cz"].detach().cpu().double().numpy()
    hh = batch["agent_h"].detach().cpu().double().numpy()
    valid = batch["agent_valid"].detach().cpu().numpy().astype(bool)
    zh = batch["agent_zh_mask"].detach().cpu().numpy().astype(bool)
    B = box.shape[0]
    lab = (np.ones(B, bool) if batch.get("agent_label") is None
           else batch["agent_label"].detach().cpu().numpy().astype(bool).reshape(-1))
    out, n_box, n_nozh = [], 0, 0
    for b in range(B):
        if not lab[b]:
            out.append(np.zeros((0, 7)))
            continue
        k = valid[b] & zh[b]
        n_nozh += int((valid[b] & ~zh[b]).sum())
        bx = np.stack([box[b, :, 0], box[b, :, 1], cz[b], box[b, :, 2], box[b, :, 3], hh[b], yaw[b]], axis=-1)[k]
        out.append(bx)
        n_box += int(k.sum())
    return out, {"n_rows": int(B), "n_rows_labelled": int(lab.sum()), "n_boxes": n_box, "n_boxes_no_zh": n_nozh}


def camera_of(rc, *, height: int | None = None, width: int | None = None) -> BD.Camera:
    """A ``RigCamera`` -> the builder's ``Camera`` (same R, t, canonical frame: H, W, f_ref, projection)."""
    fr = rc.frame
    return BD.Camera(rc.R_cam_to_rig.detach().cpu().double().numpy(), rc.t_cam_in_rig.detach().cpu().double().numpy(),
                     height=int(height or fr.height), width=int(width or fr.width), f_ref=float(fr.f_ref),
                     projection=str(fr.projection))


def observed_mask(height: int, width: int, equalize_bottom_rows: int = 0):
    """``None`` (all pixels observed) or the mask with the trunk's zeroed bottom rows False."""
    if int(equalize_bottom_rows) <= 0:
        return None
    m = np.ones((int(height), int(width)), bool)
    m[-int(equalize_bottom_rows):, :] = False
    return m


def build_targets(boxes_list, cams, *, equalize_bottom_rows: int = 0, strides=STRIDES, threads: int = 1, depth_kind: str = DEPTH_KIND):
    """Per-row E0g targets -> ``{stride: (target [B, D, Hs, Ws] float32 np, fg [B, Hs, Ws] bool np, weight [B, Hs, Ws] float32 np)}``.
    A row with no cuboid is all-zero / all-False (and its cells are BG: weight alpha_bg, target 0 -> an inert loss)."""
    B = len(boxes_list)
    if len(cams) != B:
        raise SystemExit("[box-depth-aux] %d cameras for %d rows" % (len(cams), B))
    if any(c is None for c in cams):
        raise SystemExit("[box-depth-aux] a row has no camera (model._rig_camera is None / the bank does not cover the clip): the "
                         "ray-cast has no ray -- refuse, do not skip")

    def one(i):
        cam = cams[i]
        obs = observed_mask(cam.height, cam.width, equalize_bottom_rows)
        return BD.build_depth_targets(boxes_list[i], cam, strides=strides, depth_kind=depth_kind, observed=obs)

    if int(threads) > 1 and B > 1:
        with ThreadPoolExecutor(max_workers=int(threads)) as ex:
            dts = list(ex.map(one, range(B)))
    else:
        dts = [one(i) for i in range(B)]
    return {s: (np.stack([d.grids[s].target for d in dts]), np.stack([d.grids[s].fg for d in dts]),
                np.stack([d.grids[s].weight for d in dts])) for s in strides}


# --------------------------------------------------------------------------------------------------------------- #
# the loss                                                                                                        #
# --------------------------------------------------------------------------------------------------------------- #
def depth_loss(logits: dict, targets: dict):
    """``logits`` {stride: [B, D, Hs, Ws]}, ``targets`` {stride: (target, fg, weight) tensors} -> (loss, metrics).

    Per stride ``sum(weight * CE) / n_fg`` with CE the soft cross-entropy against the cell's soft LID target and ``weight`` the
    builder's map (alpha_fg on FG, alpha_bg on BG, 0 on IGNORE).  A BG cell's target is all zeros, so its CE is exactly 0: the
    loss is carried by FG (object) cells only.  The strides are averaged; a stride with no FG cell contributes an exact 0 that keeps
    the graph (``logits.sum() * 0``)."""
    terms, met = [], {}
    for s, lg in logits.items():
        tgt, fg, wt = targets[s]
        ce = -(tgt * F.log_softmax(lg.float(), dim=1)).sum(dim=1)          # [B, Hs, Ws]
        n = fg.sum()
        met["n_fg_cells_s%d" % s] = float(n)
        if int(n) == 0:
            terms.append(lg.sum() * 0.0)
            met["ce_s%d" % s] = 0.0
            continue
        t = (wt * ce).sum() / n.to(ce.dtype)
        terms.append(t)
        met["ce_s%d" % s] = float(((ce * fg).sum() / n).detach())
    loss = sum(terms) / float(len(terms))
    return loss, met


def to_tensors(built: dict, device) -> dict:
    return {s: (torch.from_numpy(t).to(device), torch.from_numpy(f).to(device), torch.from_numpy(w).to(device))
            for s, (t, f, w) in built.items()}
