#!/usr/bin/env python3
"""render_refcv6_map_video.py -- refcv6's SAM3 semantic-map head at ONE checkpoint, frame by frame.

PI (Sayed), 2026-09-26, verbatim: *"Can we visualize the sematic map output of the last checkpoint and
render a video to examine the results"*.

⛔ READ ``RENDER_REFCV6_MAP_VIDEO.md`` (beside this file) FIRST: it fixes the clip rule BEFORE any render,
cites the IoU rule by file:line and states what this reel is NOT (it is a PERCEPTION DIAGNOSTIC in OPEN
LOOP -- never driving performance, never a four-family result).

WHAT IS ON EVERY FRAME (the viz standard: camera + metric BEV + text overlay TOGETHER)
-----------------------------------------------------------------------------------
1. FRONT CAMERA -- ``item["frames"][-1][-3:]``, the exact uint8 bytes the trunk is handed, with the GT
   future path (green) and refcv6's SELECTED trajectory ``out["traj"]`` (orange) projected through
   ``render_refcv3_video.CylProjector`` and the clip's MEASURED extrinsics (imported, never copied).
2. BEV -- PREDICTED MAP: argmax of ``softmax(out["perception"]["map_logits"])``; cells outside
   ``map_seen & map_valid`` hatched (two hatches, from the two masks).
3. BEV -- SAM3 GT MAP: argmax of ``batch["map_frac"]`` with the SAME hatches from the SAME masks.
4. DRIVABLE AGREEMENT: TP / FP / FN / TN under the TRAINER'S rule, the frame IoU and the clip mean.
5. HUD: run, step, hybrid note, the diagnostic caveat, clip sha12, t (label clock), measured v0, nav
   token (a GIVEN INPUT), the 9-class palette and a per-frame IoU sparkline.

THE FORWARD IS THE TRAINER'S, INTERRUPTED -- NOT RE-IMPLEMENTED
---------------------------------------------------------------
``refc_v3_train.compute_losses_v3`` is called on a batch of one window with a forward hook on the model
that captures ``out`` and raises. Every line of input plumbing before ``model(...)`` therefore runs as
written (``set_ego_window``, the per-clip lift geometry via ``_lift_bank.for_episodes(map_ep)``,
``v_max_ms`` / ``v_max_valid``, nav, ``ego_state``) and nothing after it (the losses, which would need
the LOSS-only agent / 3-D targets and the LAW frames) runs at all.

THE IoU RULE (``refc_v3_train.py:4506-4521`` @ 82c2331) -- ``drivable_iou_rule`` below, same arithmetic,
same tensors, same device, softmax in the logits' native dtype; the scored-cell count is cross-checked
per window against the programme's own ``refcv6_perception_branch.map_loss_row`` ``n_map_cells``.

Clip ids never leave this process: every artifact carries ``sha12`` only, and every JSON written is
scrubbed and REFUSED if a raw UUID survives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

# ============================================================================================= #
# identity of what is rendered                                                                   #
# ============================================================================================= #
RUN_LABEL = "refcv6-r101-s0"
HYBRID_NOTE = "hybrid: F3 + true label clock from step 34,500"
DIAG_NOTE = "perception diagnostic, open loop \u2014 not driving performance"
DEFAULTS = {
    "repo": "C:/Users/Admin/ev6_82c2331",
    "kit": "D:/refcv6_eval_kit",
    "battery_code": "C:/Users/Admin/ev6_battery/code",
    "ckpt": "D:/refcv6_eval_kit/ckpt/ckpt_35000.pt",
    "config": "C:/Users/Admin/ev6_battery/raw/thor_reads/config_resume34500_20260926.json",
    "metrics": "C:/Users/Admin/ev6_battery/raw/thor_reads/metrics_20260926T1500.jsonl",
    "out_dir": ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                "2026-09-26-refcv6-map-video/raw"),
    "work_dir": "C:/Users/Admin/qland/work/mapvid",
}
EXPECT = {"step": 35000,
          "ckpt_md5": "68a4ef3bbb0a3ec707b04f0fb2b6b86f",      # ckpt/MD5SUMS (3-way equal on Thor)
          "config_md5": "a3193a4685ce0d07a6ae6b89fdc994a8"}    # chain_final_v2.sh RESUME_CFG_MD5

#: SAM3 channel order -- a LITERAL, asserted against tanitad.data.semantic_map_gt.CHANNELS at run time
CHANNELS = ("seen, no map class", "drivable", "lane / road line", "crosswalk", "arrow / text",
            "non-drivable edge", "hatched area", "sidewalk / verge", "not seen")
DRIVABLE = 1
NOT_SEEN = 8
#: CART_SPEC (semantic_map_gt.py:80): x 0..60 m ahead, y +-16 m, 0.5 m cells -> [120, 64];
#: row 0 = x in [0, 0.5) m, col 0 = y in [-16, -15.5) m = the RIGHT side, +y is LEFT.
GRID_X, GRID_Y, CELL_M, X_MAX_M, Y_HALF_M = 120, 64, 0.5, 60.0, 16.0
#: the brief's order: one clip per v7 nav token
NAV_ORDER = ("left", "right", "follow")
#: known-value control: logit magnitude of the one-hot (softmax(30*e_c)[c] = 1 - 8e-13)
CTL_K = 30.0

# ============================================================================================= #
# colour semantics -- one meaning per colour                                                     #
# ============================================================================================= #
C_GT = (110, 231, 138)            # GROUND TRUTH path, everywhere
C_SEL = (255, 158, 61)            # refcv6's OWN selection out["traj"], everywhere
PALETTE = ((84, 92, 108),         # 0 seen, no map class
           (52, 104, 186),        # 1 drivable
           (236, 236, 236),       # 2 lane / road line
           (226, 88, 200),        # 3 crosswalk
           (156, 120, 236),       # 4 arrow / text
           (214, 64, 64),         # 5 non-drivable edge
           (64, 196, 206),        # 6 hatched area
           (150, 116, 84),        # 7 sidewalk / verge
           (26, 28, 34))          # 8 not seen
C_TP, C_FP, C_FN, C_TN = (70, 150, 255), (240, 70, 110), (250, 232, 100), (46, 50, 60)
HATCH_BG, HATCH_A, HATCH_B = (14, 16, 21), (72, 78, 92), (150, 140, 96)
C_BG, C_PANEL, C_BAN = (9, 12, 17), (16, 21, 29), (15, 20, 28)
C_FG, C_DIM, C_DIM2, C_WARN = (233, 238, 245), (140, 152, 168), (104, 116, 132), (245, 180, 90)
C_GIVEN = (240, 190, 90)


#: ``--boxes``: the 10 agent classes, a LITERAL asserted against
#: ``tanitad.models.agent_slots.AGENT_CLASSES`` (== ``bev_raster.ALL_CLASSES``) at run time
AGENT_CLASSES = ("automobile", "heavy_truck", "bus", "other_vehicle", "trailer", "person",
                 "rider", "stroller", "animal", "protruding_object")
AGENT_PALETTE = ((80, 170, 255), (255, 90, 90), (250, 235, 70), (200, 120, 255),
                 (255, 120, 200), (255, 255, 255), (120, 255, 230), (255, 200, 150),
                 (170, 130, 90), (160, 160, 175))
#: ``--boxes`` clip rule (RENDER_REFCV6_MAP_VIDEO.md §8, fixed before rendering)
BOX_SEL_RANGE_M = 30.0
TURN_DEG = 30.0
#: the camera's horizontal half-field (refc_agents.FOV_HALF_ANGLE_RAD = 60 deg, cylindrical)
FOV_HALF_DEG = 60.0


def _cdist(a, b) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def assert_palette_disjoint(min_dist: float = 60.0) -> dict:
    """⛔ No class / agreement / agent colour may read as the GT green or the model orange."""
    rep = {}
    for name, c in ([(f"class{i}", c) for i, c in enumerate(PALETTE)]
                    + [("TP", C_TP), ("FP", C_FP), ("FN", C_FN), ("TN", C_TN)]
                    + [(f"agent_{n}", c) for n, c in zip(AGENT_CLASSES, AGENT_PALETTE)]):
        d = min(_cdist(c, C_GT), _cdist(c, C_SEL))
        rep[name] = round(d, 1)
        if d < min_dist:
            raise SystemExit(f"[mapvid] colour {name} {c} is within {d:.1f} of the GT green / model "
                             f"orange (floor {min_dist}): a viewer could read it as a path")
    return rep


# ============================================================================================= #
# layout (1920 x 1200; both even for yuv420p)                                                    #
# ============================================================================================= #
W_TOT, H_TOT, PAD = 1920, 1200, 16
H_BAN = 76
Y_CAM, CAM_W, CAM_H = 84, 1024, 416
X_HUD = PAD + CAM_W + PAD
W_HUD = W_TOT - X_HUD - PAD
S = 5                                                   # px per 0.5 m cell
MAP_W, MAP_H = GRID_Y * S, GRID_X * S                   # 320 x 600
Y_BEV = Y_CAM + CAM_H + 12
BEV_TITLE_H, BEV_LM = 44, 46
PANEL_W = BEV_LM + MAP_W
X_PANELS = (PAD, PAD + PANEL_W + 14, PAD + 2 * (PANEL_W + 14))
Y_MAP = Y_BEV + BEV_TITLE_H
X_INFO = X_PANELS[2] + PANEL_W + 16
W_INFO = W_TOT - X_INFO - PAD


# ============================================================================================= #
# THE RULE + THE CONTROLS (pure; importable by the unit test without the stack)                  #
# ============================================================================================= #
def drivable_iou_rule(map_logits, map_frac, map_seen, map_valid=None, *, drivable_ch: int = DRIVABLE):
    """``refc_v3_train.py:4506-4521`` (82c2331), the same arithmetic on the same tensors.

    ``map_logits`` / ``map_frac`` ``[B, C, X, Y]``, ``map_seen`` / ``map_valid`` ``[B, X, Y]`` bool.
    ``map_valid`` is ANDed in exactly when the trainer ANDs it (``model._map_lift_valid_mask`` and a
    ``map_valid`` output) -- the caller decides, as the trainer's caller does. Threshold ``>= 0.5`` on
    BOTH sides; softmax in the logits' OWN dtype (the trainer never casts). The IoU is ``0.0`` when
    the union is empty (the trainer's literal convention) and the returned ``union`` says when."""
    sn = map_seen
    if map_valid is not None:
        sn = sn & map_valid
    prob = map_logits.softmax(dim=1)[:, drivable_ch]
    gt = (map_frac[:, drivable_ch] >= 0.5) & sn
    pr = (prob >= 0.5) & sn
    inter = float((gt & pr).sum())
    union = float((gt | pr).sum())
    n_sn = float(sn.sum())
    return {
        "iou": inter / union if union else 0.0,
        "inter": inter, "union": union, "n_scored": n_sn,
        "pred_drivable_frac": float(pr.sum()) / n_sn if n_sn else 0.0,
        "gt_drivable_frac": float(gt.sum()) / n_sn if n_sn else 0.0,
        "pred_drivable_prob_mean": float((prob * sn).sum()) / n_sn if n_sn else 0.0,
        "_scored": sn, "_gt": gt, "_pr": pr,
    }


def gt_as_logits(map_frac, *, drivable_ch: int = DRIVABLE, k: float = CTL_K, consistent: bool = True):
    """The KNOWN-VALUE control's input: the GT map as "logits".

    ``consistent=False`` -- the literal one-hot ``k * e_{argmax map_frac}``.
    ``consistent=True``  -- the same, except on MIXED cells (argmax is drivable but its fraction is
    < 0.5), where every logit is ``k - 1`` and the drivable one ``k``: the argmax is kept and
    ``p(drivable) = e / (e + 8) = 0.2536 < 0.5``, so the argmax panel AND the >= 0.5 rule both
    reproduce the GT. ``impossible`` counts cells no logits can serve both rules on (argmax is not
    drivable yet the drivable fraction is >= 0.5); it must read 0 for fractions that sum to 1."""
    import torch
    c = map_frac.argmax(dim=1)                                        # [B, X, Y]
    n_ch = int(map_frac.shape[1])
    z = torch.nn.functional.one_hot(c, n_ch).permute(0, 3, 1, 2).to(torch.float32) * float(k)
    drv_arg = c == drivable_ch
    drv_thr = map_frac[:, drivable_ch] >= 0.5
    mixed = drv_arg & ~drv_thr
    impossible = ~drv_arg & drv_thr
    if consistent:
        base = torch.full_like(z, float(k) - 1.0)
        z = torch.where(mixed[:, None], base, z)
        z[:, drivable_ch] = torch.where(mixed, torch.full_like(z[:, drivable_ch], float(k)),
                                        z[:, drivable_ch])
    return z, {"n_mixed": int(mixed.sum()), "n_impossible": int(impossible.sum()),
               "_mixed": mixed}


def argmax_classes(logits):
    """``[B, C, X, Y]`` -> ``[B, X, Y]``: argmax over the 9-class softmax (computed in fp32 so a
    low-precision softmax cannot manufacture ties the logits do not have)."""
    return logits.float().softmax(dim=1).argmax(dim=1)


def grid_to_image(a: np.ndarray) -> np.ndarray:
    """GRID order ``[X=120, Y=64, ...]`` (row 0 = nearest, col 0 = RIGHT) -> IMAGE order upscaled by
    ``S``: row 0 at the BOTTOM, col 0 at the RIGHT edge, ego at the bottom centre."""
    a = np.asarray(a)[::-1, ::-1]
    return np.repeat(np.repeat(a, S, axis=0), S, axis=1)


def m2px(x_m: float, y_m: float) -> tuple:
    """Metric ego-frame point (x forward, y LEFT) -> pixel in the ``MAP_W x MAP_H`` panel image."""
    return ((Y_HALF_M - float(y_m)) / CELL_M * S, (X_MAX_M - float(x_m)) / CELL_M * S)


_HATCH = None


def hatches():
    """(A) diagonal lines = SAM3 never saw the cell; (B) dots = SAM3 saw it but the lift's camera does
    not reach it at this instant. Both are UNSCORED; neither carries a class colour."""
    global _HATCH
    if _HATCH is None:
        yy, xx = np.mgrid[0:MAP_H, 0:MAP_W]
        a = np.empty((MAP_H, MAP_W, 3), np.uint8)
        a[:] = HATCH_BG
        a[((xx + yy) % 8) < 2] = HATCH_A
        b = np.empty_like(a)
        b[:] = HATCH_BG
        b[((xx % 6) < 2) & ((yy % 6) < 2)] = HATCH_B
        _HATCH = (a, b)
    return _HATCH


def compose_cells(rgb_cells: np.ndarray, seen: np.ndarray, valid) -> np.ndarray:
    """``[X, Y, 3]`` colours + ``[X, Y]`` masks -> ``[MAP_H, MAP_W, 3]`` with the two hatches."""
    img = grid_to_image(rgb_cells).copy()
    ha, hb = hatches()
    m_a = grid_to_image(~seen)
    img[m_a] = ha[m_a]
    if valid is not None:
        m_b = grid_to_image(seen & ~valid)
        img[m_b] = hb[m_b]
    return img


def class_map_array(cls_map: np.ndarray, seen: np.ndarray, valid) -> np.ndarray:
    """ONE function for BOTH class panels -- so the known-value control compares like with like."""
    pal = np.asarray(PALETTE, dtype=np.uint8)
    return compose_cells(pal[np.asarray(cls_map, dtype=np.int64)], seen, valid)


def agreement_array(gt: np.ndarray, pr: np.ndarray, scored: np.ndarray, seen: np.ndarray,
                    valid) -> np.ndarray:
    rgb = np.empty(gt.shape + (3,), np.uint8)
    rgb[:] = C_TN
    rgb[gt & pr] = C_TP
    rgb[pr & ~gt] = C_FP
    rgb[gt & ~pr] = C_FN
    return compose_cells(rgb, seen, valid)


# ============================================================================================= #
# --boxes: pure geometry + the known-value control's input (importable without the stack)        #
# ============================================================================================= #
#: the 12 edges of a cuboid whose corners are ordered bottom 0-3, top 4-7 (same footprint order)
CUBOID_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7))


def box_corners_xy(cx: float, cy: float, l: float, w: float, yaw: float) -> np.ndarray:
    """``[4, 2]`` footprint corners in the rig frame (+x fwd, +y LEFT), ordered front-left,
    front-right, rear-right, rear-left. ``yaw`` is the heading about +z from +x (the slot decode's
    ``atan2(sin, cos)``, ``agent_slots.py:674``); ``l`` runs along the heading."""
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    hl, hw = 0.5 * float(l), 0.5 * float(w)
    loc = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float64)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return loc @ rot.T + np.array([float(cx), float(cy)], dtype=np.float64)


def cuboid_corners(cx, cy, cz, l, w, h, yaw) -> np.ndarray:
    """``[8, 3]``: the footprint at ``cz - h/2`` (corners 0-3) and at ``cz + h/2`` (4-7). ``cz`` is
    the cuboid CENTRE height (``box3d_head.py:268``: ``box3d = (x, y, z, l, w, h, yaw)``)."""
    fp = box_corners_xy(cx, cy, l, w, yaw)
    z0, z1 = float(cz) - 0.5 * float(h), float(cz) + 0.5 * float(h)
    return np.concatenate([np.c_[fp, np.full(4, z0)], np.c_[fp, np.full(4, z1)]], axis=0)


def in_bev_range(cx, cy) -> np.ndarray:
    """The BEV panel == the slot DECODE BOX (``SlotDecodeRanges``: x 0..60 m, |y| <= 16 m)."""
    cx, cy = np.asarray(cx, np.float64), np.asarray(cy, np.float64)
    return (cx >= 0.0) & (cx <= X_MAX_M) & (np.abs(cy) <= Y_HALF_M)


def in_camera_field(cx, cy, half_deg: float = FOV_HALF_DEG) -> np.ndarray:
    """``refc_agents.filter_targets_to_visible``'s azimuth predicate, ``|atan2(cy, cx)| <= 60 deg``
    and ``cx >= 0`` -- used by the CLIP RULE's count only; the render calls the trainer's own
    ``visible_target_filter`` for every target set it scores."""
    cx, cy = np.asarray(cx, np.float64), np.asarray(cy, np.float64)
    return (np.arctan2(np.abs(cy), cx) <= math.radians(float(half_deg))) & (cx >= 0.0)


def heading_change_deg(yaw: np.ndarray, now_row: int, horizon: int = 60) -> float:
    """Largest ``|wrap(yaw[now + k] - yaw[now])|`` in degrees over the VALID part of the 6 s future
    (``k = 1..horizon``, rows past the clip end are not futures: ``future_valid_ext`` rule,
    ``refc_v3_train.py:3066-3068``). 0.0 when no future row exists."""
    yaw = np.asarray(yaw, np.float64).reshape(-1)
    n = int(now_row)
    k = np.arange(n + 1, min(n + 1 + int(horizon), yaw.shape[0]))
    if k.size == 0:
        return 0.0
    d = (yaw[k] - yaw[n] + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.degrees(np.abs(d)).max())


def gt_as_slots(box, yaw, cls, valid, n_slots: int, k: float = 20.0) -> dict:
    """The boxes known-value control's input: the GT set written INTO the predicted-slot format.

    ``box`` ``[A, 4]`` (cx, cy, l, w) · ``yaw`` ``[A]`` · ``cls`` ``[A]`` (long, -1 = unknown) ·
    ``valid`` ``[A]``. Slot ``i`` carries valid target ``i`` with ``presence_logit = +k`` and a
    one-hot ``k`` class logit; every other slot is EMPTY (``presence_logit = -k``) and parked at
    (1000, 1000) m, outside every range. Returns ``[1, N, ...]`` tensors in the decode's key names
    (``agent_slots.py:668-684``) and ``src`` = the target index each slot copies (-1 = empty)."""
    import torch
    box = torch.as_tensor(box, dtype=torch.float32)
    yaw = torch.as_tensor(yaw, dtype=torch.float32)
    cls = torch.as_tensor(cls, dtype=torch.long)
    idx = torch.as_tensor(valid, dtype=torch.bool).nonzero(as_tuple=False).flatten()
    n = max(int(n_slots), int(idx.numel()))
    out_box = torch.tensor([[1000.0, 1000.0, 1.0, 1.0]]).repeat(n, 1)
    out_yaw = torch.zeros(n)
    pres = torch.full((n,), -float(k))
    logits = torch.zeros(n, len(AGENT_CLASSES))
    src = torch.full((n,), -1, dtype=torch.long)
    for j, i in enumerate(idx.tolist()):
        out_box[j] = box[i]
        out_yaw[j] = yaw[i]
        pres[j] = float(k)
        if int(cls[i]) >= 0:
            logits[j, int(cls[i])] = float(k)
        src[j] = i
    return {"box": out_box[None], "yaw": out_yaw[None], "presence_logit": pres[None],
            "cls_logits": logits[None],
            "yaw_vec": torch.stack([torch.sin(out_yaw), torch.cos(out_yaw)], -1)[None],
            "src": src}


def detection_readout(slots: dict, tgt: dict, *, gate: float, fns: dict,
                      use_z: bool = False) -> dict:
    """ONE window (batch 1) of one detector head, scored by the TRAINER'S OWN rule.

    * target set: ``refc_agents.visible_target_filter`` (``refc_agents.py:410-433``: the 120 deg
      field ∩ the slot decode box), applied BEFORE matching exactly as ``box3d_set_loss``
      (``box3d_head.py:376-411``) and ``agent_losses`` (``refc_agents.py:849-852``) apply it;
    * matching: ``agent_slots.match_slots`` (``agent_slots.py:788-823``, Hungarian on
      ``presence + cls + centre + 0.5 size``, ``agent_slots.py:216-221``). It assigns a slot to
      EVERY target, so ``n_matched == n_targets`` is an IDENTITY (``box_quality.py``'s control),
      and the confidence of the assigned slot is reported beside it;
    * "detected": ``sigmoid(presence_logit) >= gate``, the model's own configured
      ``AgentSeamConfig.presence_gate`` (``refc_agents.py:197-199``), counted over the BEV range;
    * greedy: ``box3d_head.box3d_match_rows`` at 2 m (``box3d_head.py:482-536``), the programme's
      AP matcher, for the precision / recall of the CONFIDENT set.
    """
    import torch
    tv = fns["visible_target_filter"](tgt)
    m = fns["match_slots"](slots, tv)
    p = slots["presence_logit"][0].detach().float().sigmoid().cpu().numpy()
    box = slots["box"][0].detach().float().cpu().numpy()
    yaw = slots["yaw"][0].detach().float().cpu().numpy()
    cls = slots["cls_logits"][0].detach().float().argmax(-1).cpu().numpy()
    cz = slots["cz"][0].detach().float().cpu().numpy() if "cz" in slots else None
    hh = slots["h"][0].detach().float().cpu().numpy() if "h" in slots else None
    gbox = tgt["box"][0].detach().float().cpu().numpy()
    gval = tgt["valid"][0].detach().cpu().numpy().astype(bool)
    targ = tv["valid"][0].detach().cpu().numpy().astype(bool)
    conf = p >= float(gate)
    inr = in_bev_range(box[:, 0], box[:, 1])
    rows = m["rows"][0].cpu().numpy().astype(np.int64)
    cols = m["cols"][0].cpu().numpy().astype(np.int64)
    d = box[rows, :2] - gbox[cols, :2] if rows.size else np.zeros((0, 2))
    l2 = np.sqrt((d ** 2).sum(-1)) if rows.size else np.zeros(0)
    l1 = np.abs(d).sum(-1) if rows.size else np.zeros(0)
    greedy = {}
    for tag, z in (("bev", False), ("3d", True)):
        if z and not use_z:
            continue
        per_elem, n_gt = fns["box3d_match_rows"](slots, tv, dist_thresh_m=2.0, use_z=z,
                                                 score="presence")
        rws = per_elem[0]
        tp = sum(r[1] for r in rws if r[0] >= float(gate))
        n_conf = sum(1 for r in rws if r[0] >= float(gate))
        greedy[tag] = {"rows": [(float(r[0]), int(r[1])) for r in rws], "n_gt": int(n_gt[0]),
                       "tp_at_gate": int(tp), "n_conf": int(n_conf),
                       "hit_pairs": [(int(r[2]), int(r[3])) for r in rws if r[1] and r[0] >= gate]}
    return {
        "gate": float(gate), "p": p, "box": box, "yaw": yaw, "cls": cls, "cz": cz, "h": hh,
        "conf": conf, "in_range": inr, "n_pred": int((conf & inr).sum()),
        "n_conf_all": int(conf.sum()),
        "gt_box": gbox, "gt_valid": gval, "gt_target": targ,
        "gt_yaw": tgt["yaw"][0].detach().float().cpu().numpy(),
        "gt_cls": tgt["cls"][0].detach().cpu().numpy(),
        "gt_cz": (tgt["cz"][0].detach().float().cpu().numpy() if "cz" in tgt else None),
        "gt_h": (tgt["h"][0].detach().float().cpu().numpy() if "h" in tgt else None),
        "gt_zh": (tgt["zh_mask"][0].detach().cpu().numpy().astype(bool) if "zh_mask" in tgt
                  else None),
        "n_gt_in_range": int((gval & in_bev_range(gbox[:, 0], gbox[:, 1])).sum()),
        "n_targets": int(targ.sum()), "n_dropped": int(sum(m["n_dropped"])),
        "rows": rows, "cols": cols, "n_matched": int(rows.size),
        "n_matched_conf": int(conf[rows].sum()) if rows.size else 0,
        "centre_l2_mean": float(l2.mean()) if rows.size else None,
        "centre_l1_mean": float(l1.mean()) if rows.size else None,
        "centre_l2": l2, "greedy": greedy,
    }


def draw_dashed_polygon(d, pts, fill, width=1, dash=5.0, gap=4.0):
    pts = list(pts) + [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 0:
            continue
        ux, uy = (x1 - x0) / seg, (y1 - y0) / seg
        s = 0.0
        while s < seg:
            e = min(s + dash, seg)
            d.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill=fill, width=width)
            s = e + gap


def draw_detection_panel(base_arr: np.ndarray, det3d: dict, det_ag: dict | None):
    """The 4th BEV panel: the predicted map DIMMED as context, then
    GT (green: solid = a trainer TARGET, dashed = in range but outside the camera's 120 deg field),
    the 3-D box head's confident slots (SOLID, class colour, light fill), the agent head's confident
    slots (DASHED, class colour), and a thin white link for every Hungarian pair whose slot is
    confident. Everything is metric (``m2px``) and clipped at the panel edge."""
    from PIL import Image, ImageDraw
    base = (base_arr.astype(np.float32) * 0.38).astype(np.uint8)
    im = draw_map_overlays(base, None, None, None, paths=False)
    d = ImageDraw.Draw(im, "RGBA")
    gb, gv, gt_t = det3d["gt_box"], det3d["gt_valid"], det3d["gt_target"]
    for i in np.nonzero(gv & in_bev_range(gb[:, 0], gb[:, 1]))[0]:
        yaw_i = float(det3d["gt_yaw"][i]) if "gt_yaw" in det3d else 0.0
        pts = [m2px(x, y) for x, y in box_corners_xy(gb[i, 0], gb[i, 1], gb[i, 2], gb[i, 3], yaw_i)]
        if gt_t[i]:
            d.polygon(pts, outline=C_GT + (255,), width=2)
        else:
            draw_dashed_polygon(d, pts, C_GT + (150,), width=1, dash=3.0, gap=3.0)
    for det, solid in ((det3d, True), (det_ag, False)):
        if det is None:
            continue
        for j in np.nonzero(det["conf"] & det["in_range"])[0]:
            col = AGENT_PALETTE[int(det["cls"][j]) % len(AGENT_PALETTE)]
            pts = [m2px(x, y) for x, y in box_corners_xy(det["box"][j, 0], det["box"][j, 1],
                                                           det["box"][j, 2], det["box"][j, 3],
                                                           det["yaw"][j])]
            # fill opacity carries the slot's own confidence above the gate (a DISPLAY of p, not a
            # second threshold): every drawn slot passed sigmoid(presence) >= gate
            span_ = max(1.0 - float(det["gate"]), 1e-6)
            a_ = int(25 + 175 * min(max((float(det["p"][j]) - float(det["gate"])) / span_, 0.0), 1.0))
            if solid:
                d.polygon(pts, fill=col + (a_,), outline=col + (255,), width=2)
            else:
                draw_dashed_polygon(d, pts, col + (255,), width=1)
    for r, c in zip(det3d["rows"], det3d["cols"]):
        if det3d["conf"][r]:
            d.line([m2px(det3d["box"][r, 0], det3d["box"][r, 1]), m2px(gb[c, 0], gb[c, 1])],
                   fill=(255, 255, 255, 170), width=1)
    ex, ey = m2px(0.0, 0.0)
    d.polygon([(ex - 6, ey), (ex + 6, ey), (ex, ey - 12)], fill=(240, 244, 250, 255))
    return im


def draw_map_info(rv, cv, d, F, xi, wi, n_tp, n_fp, n_fn, iou, sel_idx, seed, gt_trunc_s,
                  cls_pred, cls_gt, sc_np, prow, els):
    """The map video's info column, unchanged (moved here from the frame loop)."""
    yi = Y_BEV + 2
    d.text((xi, yi), "DRIVABLE AGREEMENT — legend", fill=C_FG, font=F["subb"])
    swatch(cv, d, xi, yi + 24, C_TP, F, "TP — model AND SAM3 drivable")
    swatch(cv, d, xi, yi + 44, C_FP, F, "FP — model drivable, SAM3 not")
    swatch(cv, d, xi, yi + 64, C_FN, F, "FN — SAM3 drivable, model missed it")
    swatch(cv, d, xi, yi + 84, C_TN, F, "TN — scored, neither drivable")
    swatch(cv, d, xi, yi + 104, None, F, "unscored: never seen", pattern="A")
    swatch(cv, d, xi + 250, yi + 104, None, F, "unscored: lift cannot reach it now", pattern="B")
    d.text((xi, yi + 130), f"this frame: TP {n_tp:,} · FP {n_fp:,} · FN {n_fn:,}  "
           f"→ IoU {n_tp}/{n_tp + n_fp + n_fn} = {iou:.3f}", fill=C_FG, font=F["sub"])
    d.text((xi, yi + 160), "PATHS (camera + both class panels)", fill=C_FG, font=F["subb"])
    d.line([(xi, yi + 190), (xi + 36, yi + 190)], fill=C_GT, width=4)
    d.text((xi + 44, yi + 181), "GT future path (ego-future poses; dots = the 8 horizon slots)",
           fill=C_DIM, font=F["tiny"])
    d.line([(xi, yi + 210), (xi + 36, yi + 210)], fill=C_SEL, width=4)
    d.text((xi + 44, yi + 201), f"refcv6's SELECTED path out['traj'] (anchor #{sel_idx}; "
           f"circles = its 8 emitted slots; DDIM seed {seed})", fill=C_DIM, font=F["tiny"])
    if gt_trunc_s is not None:
        d.text((xi, yi + 222), f"(!) GT drawn to {gt_trunc_s:g} s only — the clip ends "
               "before the 6 s horizon", fill=C_WARN, font=F["tiny"])
    d.text((xi, yi + 246), "PER-CLASS (programme's map_metrics, argmax, scored cells)",
           fill=C_FG, font=F["subb"])
    d.text((xi, yi + 266), "class", fill=C_DIM2, font=F["micro"])
    d.text((xi + 190, yi + 266), "model %", fill=C_DIM2, font=F["micro"])
    d.text((xi + 270, yi + 266), "SAM3 %", fill=C_DIM2, font=F["micro"])
    d.text((xi + 350, yi + 266), "IoU", fill=C_DIM2, font=F["micro"])
    n_sc = max(int(sc_np.sum()), 1)
    for i_c, name in enumerate(CHANNELS):
        yy = yi + 282 + i_c * 17
        d.rectangle([xi, yy + 3, xi + 10, yy + 13], fill=PALETTE[i_c])
        d.text((xi + 16, yy), name, fill=C_DIM, font=F["tiny"])
        d.text((xi + 190, yy), f"{100 * float((cls_pred[sc_np] == i_c).sum()) / n_sc:5.1f}",
               fill=C_FG, font=F["tiny"])
        d.text((xi + 270, yy), f"{100 * float((cls_gt[sc_np] == i_c).sum()) / n_sc:5.1f}",
               fill=C_FG, font=F["tiny"])
        v = prow.get(f"map_iou_{i_c}")
        d.text((xi + 350, yy), "—" if v is None else f"{float(v):.3f}", fill=C_FG,
               font=F["tiny"])
    yv = yi + 282 + 9 * 17 + 10
    d.text((xi, yv), "VIZ-STANDARD SLOTS (check_frame)", fill=C_FG, font=F["subb"])
    n_sl = 0
    for e in els:
        if e.element in ("camera", "bev"):
            continue
        d.text((xi, yv + 20 + n_sl * 15), rv.fit(d, e.hud_text(), F["micro"], wi - 4),
               fill=C_DIM, font=F["micro"])
        n_sl += 1
    yn = H_TOT - 70
    for li, ln in enumerate(rv.wrap(
            d, "Class panels = ARGMAX; the agreement panel and every IoU = the trainer's >= 0.5 "
               "THRESHOLD (refc_v3_train.py:4506-4521). A window's IoU is not an eval result: "
               "one clip is one episode, its windows are not independent, and no interval is "
               "quoted.", F["micro"], wi)):
        d.text((xi, yn + li * 14), ln, fill=C_DIM2, font=F["micro"])


def draw_boxes_info(rv, cv, d, F, xi, wi, n_tp, n_fp, n_fn, iou, sel_idx, seed, gt_trunc_s,
                    cls_pred, cls_gt, sc_np, prow, n_mixed, els):
    """``--boxes``: the same legends in a 366 px column, plus the agent classes and box styles."""
    yi = Y_BEV + 2
    d.text((xi, yi), "DRIVABLE AGREEMENT", fill=C_FG, font=F["subb"])
    c2 = xi + 180
    swatch(cv, d, xi, yi + 22, C_TP, F, "TP both", w=18, h=12)
    swatch(cv, d, c2, yi + 22, C_FP, F, "FP model only", w=18, h=12)
    swatch(cv, d, xi, yi + 40, C_FN, F, "FN SAM3 only", w=18, h=12)
    swatch(cv, d, c2, yi + 40, C_TN, F, "TN neither", w=18, h=12)
    swatch(cv, d, xi, yi + 58, None, F, "never seen", w=18, h=12, pattern="A")
    swatch(cv, d, c2, yi + 58, None, F, "lift can't reach now", w=18, h=12, pattern="B")
    d.text((xi, yi + 78), f"frame: TP {n_tp:,} FP {n_fp:,} FN {n_fn:,} → IoU {iou:.3f}",
           fill=C_FG, font=F["tiny"])
    d.text((xi, yi + 102), "DETECTED AGENTS — class colours", fill=C_FG, font=F["subb"])
    for i_c, name in enumerate(AGENT_CLASSES):
        col_, row_ = i_c % 2, i_c // 2
        x_ = xi + col_ * 183
        y_ = yi + 124 + row_ * 16
        d.rectangle([x_, y_ + 2, x_ + 14, y_ + 12], outline=AGENT_PALETTE[i_c], width=2)
        d.text((x_ + 20, y_), name, fill=C_DIM, font=F["tiny"])
    styles = ["solid box = 3-D box head (perception branch)",
              "dashed box = agent head (the planner's agent tokens)",
              "green box = GT (agent join + join3d), a trainer target",
              "green dashed = GT in range, outside the 120° field",
              "white link = Hungarian pair (trainer rule), slot confident",
              "camera: cuboids = GT targets (green) + 3-D head (class)"]
    for li, s_ in enumerate(styles):
        d.text((xi, yi + 210 + li * 14), s_, fill=C_DIM, font=F["micro"])
    yp = yi + 300
    d.text((xi, yp), "PATHS", fill=C_FG, font=F["subb"])
    d.line([(xi, yp + 28), (xi + 30, yp + 28)], fill=C_GT, width=4)
    d.text((xi + 38, yp + 20), "GT future path (dots = 8 horizon slots)", fill=C_DIM,
           font=F["tiny"])
    d.line([(xi, yp + 46), (xi + 30, yp + 46)], fill=C_SEL, width=4)
    d.text((xi + 38, yp + 38), f"selected out['traj'] (anchor #{sel_idx}, DDIM seed {seed})",
           fill=C_DIM, font=F["tiny"])
    if gt_trunc_s is not None:
        d.text((xi, yp + 56), f"(!) GT only to {gt_trunc_s:g} s — the clip ends first",
               fill=C_WARN, font=F["tiny"])
    yc = yp + 78
    d.text((xi, yc), "PER-CLASS MAP (argmax, scored cells)", fill=C_FG, font=F["subb"])
    d.text((xi, yc + 18), "class", fill=C_DIM2, font=F["micro"])
    d.text((xi + 170, yc + 18), "model %", fill=C_DIM2, font=F["micro"])
    d.text((xi + 230, yc + 18), "SAM3 %", fill=C_DIM2, font=F["micro"])
    d.text((xi + 295, yc + 18), "IoU", fill=C_DIM2, font=F["micro"])
    n_sc = max(int(sc_np.sum()), 1)
    for i_c, name in enumerate(CHANNELS):
        yy = yc + 32 + i_c * 15
        d.rectangle([xi, yy + 3, xi + 10, yy + 12], fill=PALETTE[i_c])
        d.text((xi + 16, yy), name, fill=C_DIM, font=F["tiny"])
        d.text((xi + 170, yy), f"{100 * float((cls_pred[sc_np] == i_c).sum()) / n_sc:5.1f}",
               fill=C_FG, font=F["tiny"])
        d.text((xi + 230, yy), f"{100 * float((cls_gt[sc_np] == i_c).sum()) / n_sc:5.1f}",
               fill=C_FG, font=F["tiny"])
        v = prow.get(f"map_iou_{i_c}")
        d.text((xi + 295, yy), "—" if v is None else f"{float(v):.3f}", fill=C_FG,
               font=F["tiny"])
    ya = yc + 32 + 9 * 15 + 4
    d.text((xi, ya), rv.fit(d, f"argmax acc {100 * float(prow.get('map_acc', float('nan'))):.1f} % "
                               f"· soft-CE {float(prow['loss']):.3f} · mixed cells {n_mixed}",
                            F["micro"], wi - 4), fill=C_DIM, font=F["micro"])
    yv = ya + 22
    d.text((xi, yv), "VIZ-STANDARD SLOTS (check_frame)", fill=C_FG, font=F["subb"])
    n_sl = 0
    for e in els:
        if e.element in ("camera", "bev"):
            continue
        d.text((xi, yv + 20 + n_sl * 14), rv.fit(d, e.hud_text(), F["micro"], wi - 4),
               fill=C_DIM, font=F["micro"])
        n_sl += 1


#: Samples per cuboid edge on the camera overlay. A straight 3-D edge is a CURVE on the cylinder
#: (``v = f*y/hypot(x, z)`` varies along a horizontal line), so each edge is projected as a sampled 3-D
#: segment, never as the chord between its two projected corners.
CUBOID_EDGE_SAMPLES: int = 32


def cuboid_edge_runs(cam, corners: np.ndarray, n_samp: int = CUBOID_EDGE_SAMPLES,
                     refine: int = 24) -> list:
    """Every edge of an ``[8, 3]`` rig-frame cuboid, projected with the model's OWN per-clip ``RigCamera``
    as a SAMPLED 3-D segment. Returns ``[(edge_index, [(col, row), ...]), ...]``: one entry per maximal
    IN-FRAME run of an edge, whose end points are bisected onto the frame border when the edge leaves it.

    ⛔ WHY (box-head audit 2026-09-27, the PI's "one surface" frames). The previous ``draw_cuboid_cam``
    drew an edge only when BOTH its corners landed in the frame. A cuboid that crosses the +-60 deg border
    therefore lost every border-crossing edge and read as one flat face: bus #39 in 9f8bedcfb9de, window
    104, whose rear corners lie beyond 60 deg. It also drew each kept edge as the straight chord between
    its projected corners, although the cylinder bends a horizontal edge. The projection itself was right
    (verified independently: raw-mp4 re-render, LiDAR through its own extrinsic, the SAM3 camera model);
    only the DRAWING dropped and straightened edges."""
    import torch
    cor = np.asarray(corners, dtype=np.float64)
    tt = np.linspace(0.0, 1.0, int(n_samp) + 1)

    def _proj(p):
        c, r, ok = cam.project(torch.as_tensor(np.atleast_2d(p), dtype=torch.float64))
        return c.numpy(), r.numpy(), ok.numpy().astype(bool)

    out = []
    for e, (a, b) in enumerate(CUBOID_EDGES):
        pa, pb = cor[a], cor[b]

        def _at(s, pa=pa, pb=pb):
            return pa + (pb - pa) * float(s)

        def _border(s_in, s_out):
            """Bisect between an in-frame parameter and an out-of-frame one; returns the in-frame end."""
            for _ in range(int(refine)):
                m = 0.5 * (s_in + s_out)
                if bool(_proj(_at(m))[2][0]):
                    s_in = m
                else:
                    s_out = m
            c, r, _ok = _proj(_at(s_in))
            return (float(c[0]), float(r[0]))

        col, row, ok = _proj(pa[None, :] + (pb - pa)[None, :] * tt[:, None])
        run = []
        for k in range(len(tt)):
            if ok[k]:
                if not run and k > 0:
                    run.append(_border(tt[k], tt[k - 1]))
                run.append((float(col[k]), float(row[k])))
            elif run:
                run.append(_border(tt[k - 1], tt[k]))
                if len(run) >= 2:
                    out.append((e, run))
                run = []
        if len(run) >= 2:
            out.append((e, run))
    return out


def draw_cuboid_cam(cd, cam, corners: np.ndarray, colour, width: int = 2) -> int:
    """Project an ``[8, 3]`` rig-frame cuboid with the model's OWN per-clip ``RigCamera``
    (``rig_projection.RigCamera.project``: the cylindrical formula, ``rig_projection.py:272-314``)
    and draw every IN-FRAME part of every edge as a sampled curve, clipped at the frame border
    (:func:`cuboid_edge_runs`). Returns the number of edges with at least one drawn part."""
    runs = cuboid_edge_runs(cam, corners)
    for _e, pts in runs:
        cd.line(pts, fill=colour, width=width, joint="curve" if width > 1 else None)
    return len({e for e, _pts in runs})


# ============================================================================================= #
# small utilities                                                                                #
# ============================================================================================= #
_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def sha12(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:12]


def scrub(obj):
    if isinstance(obj, str):
        return _UUID.sub(lambda m: "sha12:" + sha12(m.group(0)), obj)
    if isinstance(obj, dict):
        return {scrub(k): scrub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [scrub(v) for v in obj]
    return obj


def write_json(path, obj) -> str:
    txt = json.dumps(scrub(obj), indent=1, ensure_ascii=True, default=str)
    if _UUID.search(txt):
        raise SystemExit(f"[mapvid] refusing to write {path}: a raw clip UUID survived the scrub")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(txt, encoding="utf-8")
    return str(path)


def md5_file(p, chunk=1 << 22) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(chunk), b""):
            h.update(c)
    return h.hexdigest()


def _p(*a):
    print(*a, flush=True)


class RamWatch:
    """A thread that ENDS THE PROCESS if the host's available RAM falls under the floor (dev-box rule:
    never put CPU load on the box below 8 GB free). It records the minimum it saw and the process's
    peak working set, and writes the done-marker first when one is armed."""

    def __init__(self, floor_gb: float, on_kill=None, kill: bool = True):
        import psutil
        self.ps = psutil
        self.floor = float(floor_gb)
        self.min_avail = 1e9
        self.peak_rss = 0.0
        self.on_kill = on_kill
        #: False on the GPU render: the main loop PAUSES between windows instead (its CPU share per
        #: window is a decode + a draw), so another process's RAM use stalls this job, never kills it
        self.kill = bool(kill)
        self.n_below = 0
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def avail_gb(self) -> float:
        return self.ps.virtual_memory().available / 2 ** 30

    def _run(self):
        pr = self.ps.Process()
        while not self.stop.is_set():
            av = self.avail_gb()
            self.min_avail = min(self.min_avail, av)
            try:
                mi = pr.memory_info()
                self.peak_rss = max(self.peak_rss, getattr(mi, "peak_wset", mi.rss) / 2 ** 30)
            except Exception:                                   # noqa: BLE001
                pass
            if av < self.floor and not self.kill:
                self.n_below += 1
            elif av < self.floor:
                _p(f"[mapvid:RAM] available {av:.2f} GB < floor {self.floor:.1f} GB -- ENDING the "
                   f"process (dev-box rule)")
                if self.on_kill:
                    try:
                        self.on_kill(av)
                    except Exception:                           # noqa: BLE001
                        pass
                os._exit(4)
            time.sleep(0.5)

    def report(self) -> dict:
        return {"floor_gb": self.floor, "min_available_gb": round(self.min_avail, 2),
                "process_peak_working_set_gb": round(self.peak_rss, 2), "kill_mode": self.kill,
                "n_samples_below_floor": self.n_below}


# ============================================================================================= #
# drawing                                                                                        #
# ============================================================================================= #
def draw_map_overlays(img_arr: np.ndarray, gt_dense, gt_slots, sel, *, paths: bool = True):
    """Range lines every 10 m, lateral lines at y = -10 / 0 / +10 m, the ego, and (``paths``) the GT
    path (green, dense) + the selected path (orange, its emitted slots circled). Paths leaving the
    60 m x +-16 m grid are CLIPPED at the panel edge, never rescaled."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img_arr)
    d = ImageDraw.Draw(im, "RGBA")
    for r in range(10, int(X_MAX_M), 10):
        _, py = m2px(r, 0.0)
        d.line([(0, py), (MAP_W, py)], fill=(255, 255, 255, 50), width=1)
    for yl in (-10.0, 0.0, 10.0):
        px, _ = m2px(0.0, yl)
        d.line([(px, 0), (px, MAP_H)], fill=(255, 255, 255, 60 if yl == 0.0 else 34), width=1)
    if paths:
        if gt_dense is not None and len(gt_dense) >= 1:
            pts = [m2px(0.0, 0.0)] + [m2px(x, y) for x, y in gt_dense]
            d.line(pts, fill=C_GT + (255,), width=4, joint="curve")
        for x, y in (gt_slots if gt_slots is not None else []):
            px, py = m2px(x, y)
            d.ellipse([px - 3, py - 3, px + 3, py + 3], fill=C_GT + (255,))
        if sel is not None:
            pts = [m2px(0.0, 0.0)] + [m2px(x, y) for x, y in sel]
            d.line(pts, fill=C_SEL + (255,), width=3, joint="curve")
            for x, y in sel:
                px, py = m2px(x, y)
                d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=C_SEL + (255,), width=2,
                          fill=(11, 15, 21, 255))
    ex, ey = m2px(0.0, 0.0)
    d.polygon([(ex - 6, ey), (ex + 6, ey), (ex, ey - 12)], fill=(240, 244, 250, 255))
    return im


def draw_bev_block(canvas, x0: int, title: str, sub: str, map_im, F, sub_col=None):
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    d.text((x0, Y_BEV + 2), title, fill=C_FG, font=F["subb"])
    d.text((x0, Y_BEV + 22), sub, fill=sub_col or C_DIM, font=F["tiny"])
    canvas.paste(map_im, (x0 + BEV_LM, Y_MAP))
    d.rectangle([x0 + BEV_LM - 1, Y_MAP - 1, x0 + BEV_LM + MAP_W, Y_MAP + MAP_H], outline=(60, 70, 84))
    for r in range(10, int(X_MAX_M), 10):
        _, py = m2px(r, 0.0)
        d.text((x0 + 2, Y_MAP + py - 7), f"{r} m", fill=C_DIM2, font=F["tiny"])
    d.text((x0 + 2, Y_MAP + 1), f"{int(X_MAX_M)} m", fill=C_DIM2, font=F["tiny"])     # the top edge
    for yl, lab in ((10.0, "10 m L"), (0.0, "0"), (-10.0, "10 m R")):
        px, _ = m2px(0.0, yl)
        tw = d.textlength(lab, font=F["tiny"])
        d.text((x0 + BEV_LM + px - tw / 2, Y_MAP + MAP_H + 4), lab, fill=C_DIM2, font=F["tiny"])


def draw_spark(d, x0, y0, w, h, ious, n_total, ref, F):
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(12, 15, 20), outline=(40, 48, 60))

    def ypx(v):
        return y0 + h - 6 - max(0.0, min(1.0, v)) * (h - 12)

    def xpx(i):
        return x0 + 8 + (i / max(n_total - 1, 1)) * (w - 16)
    for v in (0.0, 0.5, 1.0):
        d.line([(x0 + 1, ypx(v)), (x0 + w - 1, ypx(v))], fill=(34, 42, 53))
        d.text((x0 + w + 4, ypx(v) - 7), f"{v:.1f}", fill=C_DIM2, font=F["micro"])
    if ref is not None:
        yy = ypx(ref)
        for xx in range(int(x0 + 2), int(x0 + w - 2), 10):
            d.line([(xx, yy), (xx + 5, yy)], fill=C_DIM, width=1)
    pts = [(xpx(i), ypx(v)) for i, v in enumerate(ious)]
    if len(pts) >= 2:
        d.line(pts, fill=C_TP, width=2)
    if pts:
        px, py = pts[-1]
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=C_FG)
        mu = float(np.mean(ious))
        d.line([(x0 + 2, ypx(mu)), (px, ypx(mu))], fill=(200, 210, 225), width=1)


def swatch(cv, d, x, y, col, F, text, w=24, h=14, pattern=None):
    """A legend chip. A hatch chip is CUT FROM THE SAME ARRAY the panels use, so the legend cannot
    drift from the pixels it explains."""
    from PIL import Image
    if pattern is None:
        d.rectangle([x, y, x + w, y + h], fill=col, outline=(90, 100, 115))
    else:
        arr = hatches()[0 if pattern == "A" else 1][:h, :w]
        cv.paste(Image.fromarray(np.ascontiguousarray(arr)), (x, y))
        d.rectangle([x, y, x + w, y + h], outline=(90, 100, 115))
    d.text((x + w + 6, y - 1), text, fill=C_DIM, font=F["tiny"])


class MapDumper:
    """``--dump-map``: the map head's per-cell output for the CLASS ANALYSIS, one row per window.

    Per window: ``probs`` = softmax(map_logits) as float16 ``[9, 120, 64]``; ``frac`` = the GT
    fractions as the label file's own uint8 (x255, recovered exactly by rounding the loader's
    float32 ``u8/255``); ``seen`` / ``valid`` bool ``[120, 64]`` (``valid`` = the lift's
    ``map_valid``, all-True when the run does not AND it); plus a ``meta`` row (sha12 only).
    Cell (i, j) sits at x = (i + 0.5) * 0.5 m ahead, y = -16 + (j + 0.5) * 0.5 m (col 0 = RIGHT).
    Chunks are written as ``.npz`` to LOCAL disk; each (clip, window) is dumped ONCE."""

    def __init__(self, out_dir, chunk: int = 128):
        self.dir = Path(out_dir)
        if self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True)
        self.chunk = int(chunk)
        self.buf = {"probs": [], "frac": [], "seen": [], "valid": []}
        self.meta, self.files, self.keys, self.n = [], [], set(), 0

    def has(self, s12: str, t: int) -> bool:
        return (str(s12), int(t)) in self.keys

    def add(self, lg, frac, seen, valid, meta: dict) -> bool:
        import torch
        key = (str(meta["clip_sha12"]), int(meta["t_start_row"]))
        if key in self.keys:
            return False
        self.keys.add(key)
        self.buf["probs"].append(lg.float().softmax(dim=1)[0].to(torch.float16).cpu().numpy())
        self.buf["frac"].append(torch.round(frac[0].float() * 255.0).clamp(0, 255)
                                .to(torch.uint8).cpu().numpy())
        self.buf["seen"].append(seen[0].cpu().numpy().astype(bool))
        self.buf["valid"].append(np.ones_like(self.buf["seen"][-1]) if valid is None
                                 else valid[0].cpu().numpy().astype(bool))
        self.meta.append({**meta, "row": self.n, "chunk": len(self.files)})
        self.n += 1
        if len(self.buf["probs"]) >= self.chunk:
            self.flush()
        return True

    def flush(self) -> None:
        if not self.buf["probs"]:
            return
        path = self.dir / f"chunk_{len(self.files):03d}.npz"
        np.savez(path, **{k: np.stack(v) for k, v in self.buf.items()})
        self.files.append(str(path))
        self.buf = {k: [] for k in self.buf}

    def close(self) -> dict:
        self.flush()
        with open(self.dir / "meta.jsonl", "w", encoding="utf-8") as fh:
            for m in self.meta:
                fh.write(json.dumps(scrub(m), ensure_ascii=True) + "\n")
        return {"dir": str(self.dir), "n_windows": self.n, "chunks": self.files,
                "layout": "probs f16 [N,9,120,64] softmax(map_logits); frac u8 [N,9,120,64] (x255); "
                          "seen/valid bool [N,120,64]; row i -> x=(i+0.5)*0.5 m, col j -> "
                          "y=-16+(j+0.5)*0.5 m (col 0 RIGHT)"}


# ============================================================================================= #
# the model forward -- the trainer's own, interrupted after model(...)                            #
# ============================================================================================= #
class _ForwardCaptured(Exception):
    pass


def forward_like_trainer(tr, model, batch, device, *, mode, ablate_frames, seed):
    import torch
    box = {}

    def hook(_m, _inp, out):
        box["out"] = out
        raise _ForwardCaptured()
    h = model.register_forward_hook(hook)
    torch.manual_seed(int(seed))                       # seeds the CPU AND every CUDA generator
    try:
        with torch.no_grad():
            tr.compute_losses_v3(model, batch, device, mode=mode, ablate_frames=ablate_frames)
        raise SystemExit("[mapvid] compute_losses_v3 returned WITHOUT the forward hook firing -- "
                         "the model call was not intercepted; refusing to use a loss-path output")
    except _ForwardCaptured:
        pass
    finally:
        h.remove()
    return box["out"]


# ============================================================================================= #
# main                                                                                           #
# ============================================================================================= #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("smoke", "render"), required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                    help="render mode only; --mode smoke is ALWAYS cpu (CUDA hidden)")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--expect-step", type=int, default=EXPECT["step"])
    ap.add_argument("--expect-ckpt-md5", default=EXPECT["ckpt_md5"])
    ap.add_argument("--expect-config-md5", default=EXPECT["config_md5"])
    ap.add_argument("--seed", type=int, default=0, help="inference seed, re-seeded before EVERY window")
    ap.add_argument("--smoke-windows", type=int, default=2)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--title-s", type=float, default=2.5)
    ap.add_argument("--ram-floor-gb", type=float, default=8.0,
                    help="CPU forwards: the watchdog ENDS the process below this; GPU: start gate")
    ap.add_argument("--ram-pause-gb", type=float, default=2.5,
                    help="GPU mode: pause between windows while available RAM is below this")
    ap.add_argument("--cpu-start-ram-gb", type=float, default=10.0,
                    help="refuse to START a CPU forward below this much available RAM")
    ap.add_argument("--cpu-render-ram-gb", type=float, default=12.0,
                    help="the CPU fallback of --mode render needs at least this much free RAM")
    ap.add_argument("--gate-wait-min", type=float, default=90.0)
    ap.add_argument("--done-marker", default="C:/Users/Admin/qland/work/mapvid/RENDER_DONE")
    ap.add_argument("--tag", default=None, help="output stem (default refcv6_map_step<step>[_smoke])")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--max-windows", type=int, default=0, help="debug: cap windows per clip (0 = all)")
    ap.add_argument("--boxes", action="store_true",
                    help="add the DETECTED AGENTS (3-D box head + agent head) and the GT agent "
                         "boxes: a 4th BEV panel, cuboids in the camera, a per-frame detection "
                         "line; the agent join + join3d are attached; the clip rule is §8's")
    ap.add_argument("--dump-map", action="store_true",
                    help="after the render, in the SAME process: dump the map head's per-cell "
                         "softmax + the GT fractions + masks over the in-run eval windows, the "
                         "--map-clips and every rendered window (for the class analysis)")
    ap.add_argument("--dump-min-windows", type=int, default=1000)
    ap.add_argument("--dump-dir", default=None, help="default <work-dir>/mapdump_<tag>")
    ap.add_argument("--map-clips", default="693335b810c3,0191487845ef,00213e99adca",
                    help="sha12 of the map video's clips (their windows join the dump)")
    ap.add_argument("--smoke-dump-windows", type=int, default=2)
    ap.add_argument("--gpu-smoke", action="store_true",
                    help="--mode render on the GPU with SMOKE-sized work (1 clip, --smoke-windows, "
                         "--smoke-dump-windows), outputs in <out-dir>/smoke_gpu, NO done-marker -- "
                         "for a box whose host RAM cannot hold a CPU forward above the floor")
    return ap.parse_args(argv)


def main(argv=None):  # noqa: C901 -- one linear pipeline, sectioned
    a = parse_args(argv)
    t_all = time.time()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                                           # noqa: BLE001
        pass
    # ---- environment, BEFORE torch touches a device --------------------------------------- #
    os.environ["REFCV6_REPO"] = a.repo
    os.environ["REFCV6_KIT"] = a.kit
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    device = "cpu" if a.mode == "smoke" else a.device
    if a.mode == "smoke":
        # ⚠️ an EMPTY value DELETES the variable on Windows (MEASURED 2026-09-26, _common.py); -1 hides all
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    small = a.mode == "smoke" or bool(a.gpu_smoke)
    tag = a.tag or f"refcv6_map_step{a.expect_step}" + ("_smoke" if small else "")
    out_dir = Path(a.out_dir) / ("smoke" if a.mode == "smoke" else ("smoke_gpu" if a.gpu_smoke else ""))
    work = Path(a.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rec: dict = {"tool": "taniteval/tools/render_refcv6_map_video.py", "mode": a.mode,
                 "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv[1:],
                 "run_label": RUN_LABEL, "hybrid_note": HYBRID_NOTE, "caveat": DIAG_NOTE,
                 "departures": []}
    marker = {"path": a.done_marker if (a.mode == "render" and not a.gpu_smoke) else None,
              "written": False}

    def write_marker(status: str, **kw):
        if not marker["path"] or marker["written"]:
            return
        Path(marker["path"]).parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(scrub({"status": status, "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "tool": "render_refcv6_map_video.py", **kw}), ensure_ascii=True)
        Path(marker["path"]).write_text(line + "\n", encoding="utf-8")
        marker["written"] = True
        _p(f"[mapvid] wrote done-marker {marker['path']}: {line}")

    try:
        return _run(a, device, tag, out_dir, work, rec, write_marker, t_all)
    except SystemExit as e:
        write_marker("failed", reason=str(e)[:400])
        raise
    except BaseException as e:                                  # noqa: BLE001
        write_marker("failed", reason=f"{type(e).__name__}: {str(e)[:400]}")
        raise


def _run(a, device, tag, out_dir, work, rec, write_marker, t_all):  # noqa: C901
    small = a.mode == "smoke" or bool(a.gpu_smoke)
    rec["gpu_smoke"] = bool(a.gpu_smoke)
    import psutil
    rec["palette_distance_to_paths"] = assert_palette_disjoint()
    # ---- the GPU gate (render on cuda only) --------------------------------------------------- #
    if a.mode == "render" and device == "cuda":
        sys.path.insert(0, a.battery_code)
        import gpu_gate as gg                                      # the battery's own gate, reused
        t0g, rows = time.time(), []
        while True:
            g = gg.gate()
            rows.append(g)
            _p(f"[mapvid:gate] {json.dumps(g)}")
            if g["ok"]:
                break
            if time.time() - t0g > a.gate_wait_min * 60:
                break
            time.sleep(60)
        rec["gpu_gate"] = {"final": rows[-1], "n_checks": len(rows),
                           "waited_s": round(time.time() - t0g, 1)}
        if not rows[-1]["ok"]:
            av = psutil.virtual_memory().available / 2 ** 30
            if av >= a.cpu_render_ram_gb:
                device = "cpu"
                rec["departures"].append(
                    f"GPU gate did not pass within {a.gate_wait_min:.0f} min "
                    f"({rows[-1]}); rendered on CPU with {av:.1f} GB available (>= "
                    f"{a.cpu_render_ram_gb} GB, the brief's CPU-fallback floor)")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            else:
                write_marker("blocked", reason="GPU gate did not pass and host RAM is below the CPU "
                                               "fallback floor", gate=rows[-1], available_gb=round(av, 2))
                raise SystemExit(f"[mapvid] BLOCKED: GPU gate not passed in {a.gate_wait_min:.0f} min "
                                 f"and only {av:.1f} GB RAM available (< {a.cpu_render_ram_gb})")
    rec["device"] = device
    # ---- RAM ----------------------------------------------------------------------------------- #
    av0 = psutil.virtual_memory().available / 2 ** 30
    if device == "cpu" and av0 < a.cpu_start_ram_gb:
        raise SystemExit(f"[mapvid:RAM] REFUSED: {av0:.2f} GB available < {a.cpu_start_ram_gb} GB "
                         f"needed to START a CPU forward (floor {a.ram_floor_gb} + headroom)")
    watch = RamWatch(a.ram_floor_gb, on_kill=lambda av: write_marker(
        "failed", reason=f"RAM watchdog: available {av:.2f} GB < {a.ram_floor_gb} GB"),
        kill=(device == "cpu"))
    rec["ram_at_start_gb"] = round(av0, 2)
    _p(f"[mapvid] mode={a.mode} device={device} RAM available {av0:.2f} GB, floor {a.ram_floor_gb}")

    # ---- the code tree, ASSERTED before any model code ------------------------------------------ #
    sys.path.insert(0, a.battery_code)
    import refcv6_loader as L                                      # the battery's loader, reused
    L.bootstrap()
    import tanitad
    want = os.path.normcase(os.path.abspath(os.path.join(a.repo, "stack")))
    got = os.path.normcase(os.path.abspath(tanitad.__file__))
    if not got.startswith(want):
        raise SystemExit(f"[mapvid] tanitad imported from {got}, not from {want}")
    rec["tanitad_file"] = tanitad.__file__
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if device == "cpu" and torch.cuda.is_available() and a.mode == "smoke":
        raise SystemExit("[mapvid] CUDA is visible in the CPU smoke -- refusing (no GPU in the smoke)")
    tr = L.trainer()
    from tanitad.data import semantic_map_gt as sem
    from tanitad.data import perception_targets as ptg
    from tanitad.data import v2_dataset as v2d
    from tanitad.refs import refb
    from tanitad.viz_standard import VizElement, check_frame
    import refb_labels
    if tuple(sem.CHANNELS) != CHANNELS or sem.CHANNELS.index("drivable") != DRIVABLE \
            or sem.NOT_SEEN_CHANNEL != NOT_SEEN:
        raise SystemExit(f"[mapvid] semantic_map_gt.CHANNELS {sem.CHANNELS} != the literal {CHANNELS}")
    if sem.CART_SPEC != {"x_max_m": X_MAX_M, "y_half_m": Y_HALF_M, "cell_m": CELL_M,
                         "shape": [GRID_X, GRID_Y]}:
        raise SystemExit(f"[mapvid] CART_SPEC {sem.CART_SPEC} != this renderer's grid")
    # the refcv3 renderer's projector / extrinsics loader / drawing helpers -- IMPORTED, not copied
    import importlib.util

    def _by_path(name, p):
        spec = importlib.util.spec_from_file_location(name, str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    tools = Path(a.repo) / "taniteval" / "tools"
    rcv3 = _by_path("render_refcv3_video_for_mapvid", tools / "render_refcv3_video.py")
    paix = _by_path("pai_extrinsics_table_for_mapvid", tools / "pai_extrinsics_table.py")
    rec["reused"] = {"render_refcv3_video": str(tools / "render_refcv3_video.py"),
                     "render_refcv3_video_md5": md5_file(tools / "render_refcv3_video.py"),
                     "pai_extrinsics_table": str(tools / "pai_extrinsics_table.py"),
                     "refcv6_loader": str(Path(a.battery_code) / "refcv6_loader.py"),
                     "refcv6_loader_md5": md5_file(Path(a.battery_code) / "refcv6_loader.py"),
                     "trainer_md5": md5_file(Path(a.repo) / "stack/scripts/refc_v3_train.py")}
    from PIL import Image, ImageDraw
    F = {"ban": rcv3.font(24, True), "hud": rcv3.font(17), "sub": rcv3.font(15),
         "subb": rcv3.font(15, True), "tiny": rcv3.font(13), "micro": rcv3.font(11),
         "big": rcv3.font(34, True), "med": rcv3.font(20, True), "title": rcv3.font(40, True)}

    # ---- the checkpoint + config, VERIFIED BY CONTENT ------------------------------------------ #
    t0 = time.time()
    ck_md5 = md5_file(a.ckpt)
    cfg_md5 = md5_file(a.config)
    rec["ckpt"] = {"path": a.ckpt, "md5": ck_md5, "bytes": os.path.getsize(a.ckpt),
                   "md5_s": round(time.time() - t0, 1)}
    rec["config"] = {"path": a.config, "md5": cfg_md5}
    if ck_md5 != a.expect_ckpt_md5:
        raise SystemExit(f"[mapvid] ckpt md5 {ck_md5} != expected {a.expect_ckpt_md5}")
    if cfg_md5 != a.expect_config_md5:
        raise SystemExit(f"[mapvid] config md5 {cfg_md5} != expected {a.expect_config_md5}")
    config = L.load_config(a.config)
    # the run's OWN in-run eval row at this step (control b's reference)
    rows = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in rows if r.get("step") == a.expect_step and "eval_loss" in r]
    if len(ev) != 1:
        raise SystemExit(f"[mapvid] {len(ev)} in-run eval rows at step {a.expect_step} in {a.metrics}")
    inrun = {k: v for k, v in ev[0].items() if "map" in k or k in ("step", "eval_batches", "eval_windows")}
    rec["inrun_eval_row"] = {"path": a.metrics, "md5": md5_file(a.metrics), "row": inrun,
                             "aggregation": "mean over eval_batches of a per-batch POOLED IoU "
                                            "(refc_v3_train.py:8377-8401); a DIFFERENT window set"}
    ref_iou = float(inrun["eval_map_iou_drivable"])
    _p(f"[mapvid] ckpt md5 {ck_md5} (== expected), config md5 {cfg_md5} (== expected); in-run "
       f"eval_map_iou_drivable @ {a.expect_step} = {ref_iou}")

    # ---- the MODEL: the battery's loader (strict), ckpt read through mmap ------------------------ #
    _tl = torch.load

    def _tl_mmap(f, *aa, **kk):
        if str(f) == str(a.ckpt):
            kk.setdefault("mmap", True)
        return _tl(f, *aa, **kk)
    torch.load = _tl_mmap
    try:
        model, cfg, targs, mrec = L.build_model(config, a.ckpt, device=device)
    finally:
        torch.load = _tl
    sd = mrec["state_dict"]
    rec["model"] = {k: mrec[k] for k in ("argv_remap", "departures", "state_dict", "param_breakdown",
                                         "anchor_file_vs_ckpt_buffers", "trunk_memory_levers_built",
                                         "mode", "sampler", "decoder_steps", "build_s", "perception",
                                         "rig_camera") if k in mrec}
    rec["model"]["state_dict"] = {k: v for k, v in sd.items() if k != "ckpt_keys"}
    rec["model"]["state_dict"]["ckpt_keys"] = sd.get("ckpt_keys")
    if int(sd.get("step") or -1) != int(a.expect_step):
        raise SystemExit(f"[mapvid] STEP MISMATCH: ck['step'] = {sd.get('step')!r}, expected "
                         f"{a.expect_step}")
    if sd["missing"] or sd["unexpected"]:
        raise SystemExit(f"[mapvid] strict load not clean: {sd['missing'][:5]} {sd['unexpected'][:5]}")
    if not mrec["param_breakdown"]["equal"]:
        raise SystemExit("[mapvid] param_breakdown differs from config.json")
    bad_anc = [k for k, v in mrec["anchor_file_vs_ckpt_buffers"].items() if v["max_abs_diff"] != 0.0]
    if bad_anc:
        raise SystemExit(f"[mapvid] anchor file != checkpoint anchor buffers: {bad_anc}")
    if getattr(model, "_perception", None) is None or getattr(model, "_lift_bank", None) is None:
        raise SystemExit("[mapvid] this build has no perception branch / lift bank -- no map head")
    ag = getattr(cfg.core, "agents", None)
    if ag is not None and bool(getattr(ag, "oracle", False)):
        raise SystemExit("[mapvid] --agents oracle: the forward would need agent_gt from the join; "
                         "this renderer skips the join and refuses rather than feeding nothing")
    use_valid_mask = bool(getattr(model, "_map_lift_valid_mask", True))
    rec["map_lift_valid_mask"] = use_valid_mask
    mode = getattr(targs, "mode", "diffusion")
    ablate = bool(getattr(targs, "ablate_frames", False))
    W = int(cfg.core.window)
    horizons = tuple(int(h) for h in cfg.core.trajectory.horizons)
    eq_rows = int(getattr(targs, "equalize_bottom_rows", 0) or 0)
    # ⛔ D-REFCV6-EQUALIZE-DROPPED, READ OFF THE BUILT MODULES (never from the argv). The trainer's
    # pin sets `cfg.core.encoder.trunk_equalize_bottom_rows` (refc_v3_train.py:381) as an
    # UNDECLARED attribute, and the --image-hw rebuild `dataclasses.replace(enc, ...)` at :459
    # carries declared fields only -- so the TRUNK can zero nothing while the LIFT bank, built from
    # the argv (:6922-6925), still treats the bottom rows as unobserved. The loader replays the same
    # pin, so this build reproduces the run either way; the CAPTION must say which it is.
    _trunks = [m_ for m_ in model.modules() if type(m_).__name__ == "TimmResNetTrunk"]
    trunk_eq = (int(getattr(getattr(_trunks[0], "cfg", None), "equalize_bottom_rows", 0) or 0)
                if _trunks else None)
    lift_eq = int(getattr(model._lift_bank, "equalize_bottom_rows", 0) or 0)
    rec["equalize_bottom_rows"] = {"argv": eq_rows, "trunk_built": trunk_eq, "lift_bank": lift_eq,
                                   "n_trunks": len(_trunks)}
    _p(f"[mapvid] equalize_bottom_rows: argv {eq_rows}, trunk as BUILT {trunk_eq}, lift bank "
       f"{lift_eq}")
    det_fns, gate = None, None
    if a.boxes:
        from tanitad.models import agent_slots as _as
        from tanitad.models import box3d_head as _b3
        from tanitad.refs import refc_agents as _ra
        if tuple(_as.AGENT_CLASSES) != AGENT_CLASSES:
            raise SystemExit(f"[mapvid] agent_slots.AGENT_CLASSES {_as.AGENT_CLASSES} != the literal "
                             f"{AGENT_CLASSES}")
        _agc = ((config.get("seams") or {}).get("agents") or {})
        if "presence_gate" not in _agc:
            raise SystemExit("[mapvid] config.json seams.agents carries no presence_gate -- the "
                             "model's own detection threshold is unknown; refusing to invent one")
        gate = float(_agc["presence_gate"])
        _built_gate = getattr(getattr(cfg.core, "agents", None), "presence_gate", None)
        if _built_gate is not None and float(_built_gate) != gate:
            raise SystemExit(f"[mapvid] presence_gate: config.json {gate} != built {_built_gate}")
        if abs(float(_ra.FOV_HALF_ANGLE_RAD) - math.radians(FOV_HALF_DEG)) > 1e-12:
            raise SystemExit("[mapvid] refc_agents.FOV_HALF_ANGLE_RAD is not 60 deg")
        det_fns = {"match_slots": _as.match_slots,
                   "visible_target_filter": _ra.visible_target_filter,
                   "box3d_match_rows": _b3.box3d_match_rows, "ap_from_rows": _b3.ap_from_rows}
        rec["detections"] = {
            "gate": gate, "gate_source": "config.json seams.agents.presence_gate == "
                                         "AgentSeamConfig.presence_gate (refc_agents.py:197-199), "
                                         "a threshold on sigmoid(presence_logit)",
            "heads": {"box3d": "out['perception']['box_slots'] (Box3DSlotDecoder, "
                               "box3d_head.py:227-284)",
                      "agent": "out['agent_slots'] (core AgentSlotDecoder, the planner's agent "
                               "tokens; agent_slots.py:546-684)"},
            "target_set": "refc_agents.visible_target_filter (refc_agents.py:410-433)",
            "matcher": "agent_slots.match_slots (agent_slots.py:788-823)",
            "greedy": "box3d_head.box3d_match_rows at 2.0 m (box3d_head.py:482-536)"}
    _p(f"[mapvid] model built ({mrec['build_s']} s) step {sd['step']} strict 0/0, window {W}, "
       f"horizons {horizons}, mode {mode}, sampler {mrec.get('sampler')}, steps "
       f"{mrec.get('decoder_steps')}, valid-mask {use_valid_mask}, RAM avail "
       f"{watch.avail_gb():.2f} GB")

    # ---- CLIP SELECTION (the fixed rule, RENDER_REFCV6_MAP_VIDEO.md §2) -------------------------- #
    class MapWindows(tr.V3Dataset):
        """The trainer's V3Dataset, u8 frames, WITHOUT future-frame decode: only the LAW loss reads
        ``future_frames`` and it never runs here (refcv6_roll.py's rule, verbatim contract)."""
        u8_frames = True

        def _window_u8(self, i: int) -> dict:
            e_i, t = self.index[i]
            ep = self.episodes[e_i]
            w = self.window
            return {"frames": ep.frames[t:t + w], "actions": ep.actions[t:t + w],
                    "future_frames": torch.zeros(0, dtype=torch.uint8),
                    "future_actions": ep.actions[t + w:t + w + self.max_horizon],
                    "future_poses": ep.poses[t + w:t + w + self.max_horizon],
                    "pose_last": ep.poses[t + w - 1], "episode_id": ep.episode_id}

    def clip_of(ep):
        fp = ep.frames
        return Path(fp._cache.files[fp._clip]).name.split(".")[0]

    t0 = time.time()
    if a.boxes:
        # §8's rule counts GT agent boxes, so the selection dataset carries the agent join (the 3-D
        # join is not needed to COUNT and is left off here); the SAM3 map target rides along for
        # the dump pass, which reuses this full 139-clip dataset.
        targs_sel = argparse.Namespace(**vars(targs))
        targs_sel.join3d = None
        s_ds, s_eps, s_rec = L.build_eval_dataset(model, cfg, targs_sel, config,
                                                  with_perception_targets=True,
                                                  dataset_cls=MapWindows)
    else:
        s_ds, s_eps, s_rec = L.build_eval_dataset(model, cfg, targs, config,
                                                  with_perception_targets=False,
                                                  dataset_cls=MapWindows)
    man = v2d.load_or_build_manifest(targs.eval_cache, verbose=False)
    n_stacks = sorted(set(int(x) for x in man["n_stack"]))
    if len(n_stacks) != 1:
        raise SystemExit(f"[mapvid] mixed n_stack {n_stacks}")
    n_stack = n_stacks[0]
    store = ptg.MapGTStore(Path(targs.map_gt_root), max_open=4)
    by_ep: dict = {}
    for wi, (e_i, t) in enumerate(s_ds.index):
        by_ep.setdefault(e_i, []).append(int(t))
    nav_names = list(refb.NAV_COMMANDS)
    cands = []
    for e_i, ep in enumerate(s_eps):
        cid = clip_of(ep)
        ts = sorted(by_ep.get(e_i, []))
        nav_idx = (s_ds._nav_by_sid or {}).get(int(ep.episode_id))
        if not ts:
            continue
        cov = ptg.require_map_coverage([(cid, t + W - 1) for t in ts], store, n_stack=n_stack,
                                       raise_on_low=False)
        cand = {"sha12": sha12(cid), "_cid": cid, "episode_index": e_i,
                "nav": None if nav_idx is None else nav_names[int(nav_idx)],
                "n_windows": len(ts), "n_ok": int(cov["n_ok"]),
                "n_no_file": int(cov["n_no_file"]),
                "n_out_of_range": int(cov["n_frame_out_of_range"]),
                "n_inconclusive": int(cov["n_inconclusive"])}
        if a.boxes:
            # the trainer's own GT block at each window's NOW (V3Dataset._agent_item,
            # refc_v3_train.py:2764-2864) and the GT path's heading from the episode poses
            yaw_np = ep.poses[:, 2].double().numpy()
            n_lab = n30 = n30_all = n_turn_w = 0
            turn_max = 0.0
            for t in ts:
                f_now = t + W - 1
                hc = heading_change_deg(yaw_np, f_now)
                turn_max = max(turn_max, hc)
                n_turn_w += int(hc > TURN_DEG)
                it = s_ds._agent_item(ep, f_now)
                if not bool(it["agent_label"]):
                    continue
                n_lab += 1
                b = it["agent_box"].double().numpy()
                v = it["agent_valid"].numpy().astype(bool)
                near = v & (np.hypot(b[:, 0], b[:, 1]) <= BOX_SEL_RANGE_M)
                n30_all += int(near.sum())
                n30 += int((near & in_camera_field(b[:, 0], b[:, 1])).sum())
            cand.update({"n_windows_agent_labelled": n_lab, "n_boxes_30m_in_field": n30,
                         "n_boxes_30m_any_direction": n30_all,
                         "turn_max_deg": round(turn_max, 2), "n_windows_turn_gt_30deg": n_turn_w})
        cands.append(cand)
    store.close()
    full = [c for c in cands if c["n_ok"] == c["n_windows"] and c["n_windows"] > 0]
    chosen, swapped = [], None
    if a.boxes:
        pool = sorted((c for c in full if c["n_windows_agent_labelled"] > 0),
                      key=lambda c: (-c["n_boxes_30m_in_field"], c["sha12"]))
        chosen = pool[:3]
        if not any(c["turn_max_deg"] > TURN_DEG for c in chosen):
            turners = [c for c in pool[3:] if c["turn_max_deg"] > TURN_DEG]
            if not turners:
                raise SystemExit("[mapvid] no fully covered, labelled eval clip turns > 30 deg")
            swapped = {"out": chosen[2]["sha12"], "in": turners[0]["sha12"],
                       "in_rank": pool.index(turners[0]) + 1}
            chosen = chosen[:2] + [turners[0]]
        rule_txt = ("candidates = eval clips with full SAM3 coverage and >= 1 agent-labelled window; "
                    "ranked by the number of GT agent boxes (agent join, trainer's _agent_item at "
                    "each window's NOW) within 30 m of the ego AND inside the camera's 120 deg field "
                    "(the trainer's visible predicate), summed over the clip's eval windows, ties by "
                    "sha12; take the top 3; if none of them has a window whose GT path heading "
                    "changes by > 30 deg within its valid 6 s, the 3rd is replaced by the highest-"
                    "ranked clip that does")
    else:
        for nav in NAV_ORDER:
            pool = sorted((c for c in full if c["nav"] == nav), key=lambda c: c["sha12"])
            if not pool:
                raise SystemExit(f"[mapvid] no fully covered eval clip with nav {nav!r}")
            chosen.append(pool[0])
        rule_txt = ("candidates = eval clips with full SAM3 coverage (require_map_coverage n_ok == "
                    "n_windows over the clip's own eval windows); one clip per v7 nav token in the "
                    f"order {list(NAV_ORDER)}; within a token the smallest sha12 (lowercase hex)")
    sel_rec = {
        "rule": rule_txt, "mode": "boxes" if a.boxes else "map",
        "n_eval_clips": len(s_eps), "n_clips_with_windows": len(cands),
        "n_full_coverage": len(full),
        "n_full_coverage_by_nav": {n: sum(1 for c in full if c["nav"] == n) for n in NAV_ORDER},
        "n_by_nav_all": {n: sum(1 for c in cands if c["nav"] == n) for n in NAV_ORDER},
        "not_full": [{k: v for k, v in c.items() if not k.startswith("_")}
                     for c in cands if c not in full],
        "chosen": [{k: v for k, v in c.items() if not k.startswith("_")} for c in chosen],
        "turn_swap": swapped,
        "n_stack": n_stack, "window": W, "selection_s": round(time.time() - t0, 1),
    }
    if a.boxes:
        sel_rec["ranking_top10"] = [
            {k: c[k] for k in ("sha12", "nav", "n_boxes_30m_in_field", "n_boxes_30m_any_direction",
                               "n_windows_agent_labelled", "turn_max_deg",
                               "n_windows_turn_gt_30deg")} for c in pool[:10]]
    sel_path = Path(a.out_dir) / "clip_selection.json"
    if sel_path.exists():
        prev = json.loads(sel_path.read_text(encoding="utf-8"))
        if [c["sha12"] for c in prev.get("chosen", [])] != [c["sha12"] for c in chosen]:
            raise SystemExit(f"[mapvid] the clip rule chose {[c['sha12'] for c in chosen]} but "
                             f"{sel_path} records {[c['sha12'] for c in prev.get('chosen', [])]}")
        sel_rec["agrees_with_previous_run"] = True
    write_json(sel_path, sel_rec)
    _p(f"[mapvid] selection: {len(full)}/{len(cands)} clips fully covered; chosen "
       + ", ".join(f"{c['nav']}={c['sha12']} ({c['n_windows']} w"
                   + (f", {c['n_boxes_30m_in_field']} boxes<=30m, turn {c['turn_max_deg']} deg"
                      if a.boxes else "") + ")" for c in chosen)
       + (f"; turn swap {swapped}" if swapped else "") + f"  [{sel_rec['selection_s']} s]")
    full_ds = (s_ds, s_eps) if (a.dump_map and a.boxes) else None
    del s_ds, s_eps
    # ---- the RENDER dataset: ONLY the chosen clips, WITH the SAM3 map target -------------------- #
    keep_cids = {c["_cid"] for c in (chosen[:1] if small else chosen)}
    _orig_bvp = v2d.build_v2_providers
    _orig_ctab = tr._clip_table_for_caches

    def _only_chosen(*aa, **kk):
        eps_all = _orig_bvp(*aa, **kk)
        keep = [e for e in eps_all if clip_of(e) in keep_cids]
        if len(keep) != len(keep_cids):
            raise SystemExit(f"[mapvid] provider filter kept {len(keep)} of {len(keep_cids)} clips")
        return keep

    def _ctab_chosen(*aa, **kk):
        # the 3-D join is opened for `clips=set(table.values())`; restricted to the rendered
        # clips it reads (and holds) 3 clips' cuboids instead of 139 -- the same file, same rows
        tab, ns = _orig_ctab(*aa, **kk)
        return {k: v for k, v in tab.items() if v in keep_cids}, ns
    targs_map = argparse.Namespace(**vars(targs))
    if not a.boxes:
        targs_map.agent_join = None                   # LOSS-only target (departure 2)
        targs_map.join3d = None                       # LOSS-only target (departure 2)
    v2d.build_v2_providers = _only_chosen
    if a.boxes:
        tr._clip_table_for_caches = _ctab_chosen
    try:
        e_ds, e_eps, drec = L.build_eval_dataset(model, cfg, targs_map, config,
                                                 with_perception_targets=True, dataset_cls=MapWindows)
    finally:
        v2d.build_v2_providers = _orig_bvp
        tr._clip_table_for_caches = _orig_ctab
    rec["dataset"] = drec
    if a.boxes and (drec.get("agent_join") is None or drec.get("join3d") is None):
        raise SystemExit("[mapvid] --boxes but the agent join / 3-D join was not attached")
    rec["departures"] += [
        "providers filtered to the rendered clips before build_eval_dataset (Q3_ONLY_EPISODE_SHA12 "
        "pattern) -- RAM",
        ("agent join + 3-D join ATTACHED (GT boxes); the 3-D join opened for the rendered clips only "
         "(_clip_table_for_caches restricted to them); the forward reads neither "
         "(--agents head, agents.oracle asserted False)") if a.boxes else
        ("agent join and 3-D join NOT attached: LOSS-only targets; the forward reads neither "
         "(--agents head, agents.oracle asserted False)"),
        "future_frames not decoded (only the LAW loss reads them; it never runs)",
        "forward = refc_v3_train.compute_losses_v3 interrupted by a forward hook right after "
        "model(...) returns",
        f"DDIM draw seeded per window: torch.manual_seed({a.seed}) before every forward",
        "torch.load(ckpt, mmap=True) -- same bytes, paged on touch",
    ] + list(mrec.get("departures", []))
    _p(f"[mapvid] render dataset: {drec['n_episodes']} episodes -> {drec['n_windows']} windows; "
       f"map_gt {json.dumps(scrub(drec.get('map_gt', {}).get('states')))}; RAM avail "
       f"{watch.avail_gb():.2f} GB")
    if drec.get("map_gt") is None:
        raise SystemExit("[mapvid] the SAM3 map target was not attached (no --map-gt-root?)")
    tok_of_legacy = {v: k for k, v in tr.NAV_TOKEN_TO_LEGACY.items()}
    extr_tab = rcv3.load_extrinsics(getattr(targs, "agent_rig_extrinsics", None),
                                    [c["_cid"] for c in chosen])
    rec["extrinsics"] = {"path": getattr(targs, "agent_rig_extrinsics", None),
                         "md5": md5_file(getattr(targs, "agent_rig_extrinsics")),
                         "n_clips_found": len(extr_tab)}
    pframe = tr._perc.frame_for_model(model)

    # ---- per-clip state ------------------------------------------------------------------------ #
    ep_index_of = {clip_of(ep): i for i, ep in enumerate(e_eps)}
    wins_of = {}
    for wi, (e_i, t) in enumerate(e_ds.index):
        wins_of.setdefault(e_i, []).append((int(t), wi))
    frames_dir = work / f"frames_{tag}"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    pf_path = out_dir / ("per_frame_smoke.jsonl" if small else "per_frame.jsonl")
    pf = open(pf_path, "w", encoding="utf-8")
    clips_out, frame_files, viz_rows, iou_by = [], [], None, {}
    dev_t = torch.device(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    run_list = chosen[:1] if small else chosen
    ctl_panel_clip = run_list[0]["sha12"]            # the pixel-equality control on this clip's frames
    dumper = (MapDumper(a.dump_dir or (work / f"mapdump_{tag}")) if a.dump_map else None)
    # layout: 3 BEV panels (map video) or 4 (--boxes: + DETECTED AGENTS); identical to the map
    # video's constants when --boxes is off
    XP = tuple(PAD + i_ * (PANEL_W + 14) for i_ in range(4 if a.boxes else 3))
    XI = XP[-1] + PANEL_W + 16
    WI = W_TOT - XI - PAD
    if not a.boxes and (XP != X_PANELS or XI != X_INFO or WI != W_INFO):
        raise SystemExit("[mapvid] the map-mode layout drifted from the banked video's")
    for ci, c in enumerate(run_list):
        cid, s12, nav = c["_cid"], c["sha12"], c["nav"]
        e_i = ep_index_of[cid]
        ep = e_eps[e_i]
        wl = sorted(wins_of[e_i])
        if small:
            wl = wl[:a.smoke_windows]
        elif a.max_windows:
            wl = wl[:a.max_windows]
        n_w_clip = len(sorted(wins_of[e_i]))
        # ---- this clip's camera model -------------------------------------------------------- #
        pay_path = Path(ep.frames._cache.cache_dir) / ep.frames._cache.files[ep.frames._clip]
        pay = torch.load(str(pay_path), map_location="cpu", weights_only=False, mmap=True)
        fr_dict = dict(pay.get("frame") or {})
        codec = pay.get("codec")
        del pay
        if (int(fr_dict.get("height", -1)), int(fr_dict.get("width", -1))) != (int(pframe.height),
                                                                               int(pframe.width)) \
                or str(fr_dict.get("projection")) != str(pframe.projection) \
                or abs(float(fr_dict.get("f_ref", -1)) - float(pframe.f_ref)) > 1e-6:
            raise SystemExit(f"[mapvid] payload frame {fr_dict} != the model's frame {pframe}")
        extr = extr_tab.get(cid)
        proj = rcv3.CylProjector(fr_dict, extr)
        R_chk = None
        if extr is not None:
            R_pai = np.asarray(paix.quat_to_R(float(extr["qx"]), float(extr["qy"]), float(extr["qz"]),
                                              float(extr["qw"])), dtype=np.float64)
            R_chk = float(np.abs(R_pai - proj.R).max())
            if R_chk > 1e-12:
                raise SystemExit(f"[mapvid] CylProjector R and pai_extrinsics_table R differ by {R_chk}")
        cam_label = proj.label()
        _p(f"[clip {ci + 1}/{len(run_list)}] {s12} nav={nav} windows {len(wl)}/{n_w_clip}; "
           f"{cam_label}")
        rig_cam, proj_xchk = None, None
        det_acc = {h: {"ap_rows_bev": [], "ap_rows_3d": [], "n_gt": 0, "l2": [], "l1": [],
                       "n_matched": 0, "n_matched_conf": 0, "tp_bev": 0, "n_conf": 0, "n_pred": 0,
                       "n_gt_in_range": 0, "n_windows": 0, "n_windows_no_label": 0}
                   for h in ("box3d", "agent")}
        box_ctl = {"n_windows": 0, "n_targets": 0, "n_self_matched": 0, "max_centre_err_m": 0.0,
                   "greedy_tp": 0, "max_corner_diff_m": 0.0, "trainer_centre_term_max_abs_diff": 0.0,
                   "n_trainer_term_checked": 0}
        if a.boxes:
            # ⭐ the model's OWN per-clip camera (tr._build_rig_camera -> RigCameraBank, keyed by the
            # stable episode id), which is what the monocular terms would project with
            _bank = getattr(model, "_rig_camera", None)
            rig_cam = _bank.get(int(ep.episode_id)) if _bank is not None else None
            if rig_cam is None:
                raise SystemExit(f"[mapvid] no RigCamera for clip {s12} in model._rig_camera")
            rf = rig_cam.frame
            if (int(rf.height), int(rf.width), str(rf.projection)) != (
                    int(fr_dict["height"]), int(fr_dict["width"]), str(fr_dict["projection"])) or \
                    abs(float(rf.f_ref) - float(fr_dict["f_ref"])) > 1e-6:
                raise SystemExit(f"[mapvid] RigCamera frame {rf} != payload frame {fr_dict}")
            # the two projectors must agree on ground points (same formula, same extrinsics table)
            gp = np.array([[x, y] for x in (4.0, 8.0, 15.0, 30.0, 55.0) for y in (-6.0, 0.0, 6.0)])
            cpx = proj(gp, up=1)
            _c, _r, _ok = rig_cam.project(torch.as_tensor(np.c_[gp, np.zeros(len(gp))],
                                                          dtype=torch.float64))
            dd = [math.hypot(q[0] - float(_c[i]), q[1] - float(_r[i]))
                  for i, q in enumerate(cpx) if q is not None and bool(_ok[i])]
            proj_xchk = {"n_points": len(dd), "max_px": float(max(dd)) if dd else None}
            if not dd or max(dd) > 0.01:
                raise SystemExit(f"[mapvid] CylProjector vs RigCamera disagree: {proj_xchk}")
            _p(f"  [projector cross-check] CylProjector vs model RigCamera on {len(dd)} ground "
               f"points: max {max(dd):.2e} px")
        ious, rows_clip, t_clip0 = [], [], time.time()
        sum_inter = sum_union = 0.0
        ctl = {"n": 0, "consistent_iou_eq_1": 0, "consistent_union0": 0, "consistent_min_iou": 1.0,
               "onehot_iou_min": 1.0, "onehot_iou_sum": 0.0, "n_mixed_total": 0, "n_impossible_total": 0,
               "panel_equal": 0, "panel_checked": 0, "consistent_panel_equal": 0,
               "mut_flip_iou_max": 0.0, "mut_flip_red": 0, "mut_chan_red": 0,
               "cellcount_agree": 0}
        draw_indep = None
        for k_i, (t, wi) in enumerate(wl):
            t_pause = time.time()
            # ⚠️ GPU mode: the 8 GB floor is the START gate (the battery's gpu_gate rule); this
            # process's OWN ~3 GB then sits inside the machine's total. Pausing on the floor would
            # pause on our own footprint forever, so the per-window pause is an EMERGENCY threshold
            # (--ram-pause-gb): only a box genuinely running out of memory stalls the render.
            while watch.avail_gb() < a.ram_pause_gb and device == "cuda":
                if time.time() - t_pause > 3600:
                    raise SystemExit(f"[mapvid:RAM] host RAM stayed under {a.ram_pause_gb} GB for "
                                     f"60 min -- stopping (another process holds the box)")
                _p(f"[mapvid:RAM] available {watch.avail_gb():.2f} GB < {a.ram_pause_gb} GB -- "
                   f"pausing 10 s")
                time.sleep(10)
            if time.time() - t_pause > 1:
                rec.setdefault("ram_pauses_s", []).append(round(time.time() - t_pause, 1))
            t_w = time.time()
            item = e_ds[wi]
            batch = torch.utils.data.default_collate([item])
            out = forward_like_trainer(tr, model, batch, device, mode=mode, ablate_frames=ablate,
                                       seed=a.seed)
            pout = out.get("perception")
            if pout is None or "map_logits" not in pout:
                raise SystemExit("[mapvid] out['perception']['map_logits'] missing -- no map head output")
            lg = pout["map_logits"].detach()
            mv = pout.get("map_valid")
            mv = mv.detach() if mv is not None else None
            sel = out["traj"][0].detach().float().cpu().numpy()
            sel_idx = int(out["sel_idx"][0]) if "sel_idx" in out else None
            has_route = "route_logits" in out
            box_slots = pout.get("box_slots") if a.boxes else None
            agent_slots_out = out.get("agent_slots") if a.boxes else None
            if a.boxes and (box_slots is None or agent_slots_out is None):
                raise SystemExit(f"[mapvid] --boxes but the forward emitted box_slots="
                                 f"{box_slots is not None}, agent_slots={agent_slots_out is not None}")
            del out
            frac = batch["map_frac"].to(dev_t)
            seen = batch["map_seen"].to(dev_t)
            label = bool(batch["map_label"][0])
            vmask = mv if (use_valid_mask and mv is not None) else None
            r = drivable_iou_rule(lg, frac, seen, vmask)
            prow = tr._perc.map_loss_row(lg, frac, seen, lift_valid=vmask, with_metrics=True)
            cell_ok = float(prow["n_map_cells"]) == r["n_scored"]
            if not cell_ok:
                raise SystemExit(f"[mapvid] scored cells {r['n_scored']} != map_loss_row n_map_cells "
                                 f"{prow['n_map_cells']} -- two cell sets")
            ctl["cellcount_agree"] += 1
            # ---- the draw-independence check (first window of each clip) ------------------------ #
            if k_i == 0:
                out2 = forward_like_trainer(tr, model, batch, device, mode=mode, ablate_frames=ablate,
                                            seed=a.seed + 1)
                lg2 = out2["perception"]["map_logits"].detach()
                sel2 = out2["traj"][0].detach().float().cpu().numpy()
                mv2 = out2["perception"].get("map_valid")
                del out2
                r2 = drivable_iou_rule(lg2, frac, seen, mv2 if (use_valid_mask and mv2 is not None)
                                       else None)
                draw_indep = {"seeds": [a.seed, a.seed + 1],
                              "map_logits_max_abs_diff": float((lg2.float() - lg.float()).abs().max()),
                              "drivable_mask_identical": bool(torch.equal(r2["_pr"], r["_pr"])),
                              "argmax_identical": bool(torch.equal(argmax_classes(lg2),
                                                                   argmax_classes(lg))),
                              "iou_seed_a": r["iou"], "iou_seed_b": r2["iou"],
                              "traj_max_abs_diff_m": float(np.abs(sel2 - sel).max())}
                _p(f"  [draw-independence] {json.dumps(draw_indep)}")
            # ---- controls (a): GT as logits through the SAME path ------------------------------- #
            z_c, info_c = gt_as_logits(frac, consistent=True)
            z_o, info_o = gt_as_logits(frac, consistent=False)
            rc = drivable_iou_rule(z_c, frac, seen, vmask)
            ro = drivable_iou_rule(z_o, frac, seen, vmask)
            # mutations that MUST go red: the lateral mirror (orientation defect) and the prediction's
            # channels read one off (the GT side is untouched -- a prediction-side defect only)
            rflip = drivable_iou_rule(z_c.flip(-1), frac, seen, vmask)
            rchan = drivable_iou_rule(z_c.roll(shifts=1, dims=1), frac, seen, vmask)
            ctl["n"] += 1
            if rc["union"] == 0:
                ctl["consistent_union0"] += 1
            else:
                ctl["consistent_iou_eq_1"] += int(rc["iou"] == 1.0)
                ctl["consistent_min_iou"] = min(ctl["consistent_min_iou"], rc["iou"])
            ctl["onehot_iou_min"] = min(ctl["onehot_iou_min"], ro["iou"])
            ctl["onehot_iou_sum"] += ro["iou"]
            ctl["n_mixed_total"] += info_o["n_mixed"]
            ctl["n_impossible_total"] += info_c["n_impossible"]
            ctl["mut_flip_iou_max"] = max(ctl["mut_flip_iou_max"], rflip["iou"])
            ctl["mut_flip_red"] += int(rflip["iou"] != 1.0)
            ctl["mut_chan_red"] += int(rchan["iou"] != 1.0)
            # ---- --boxes: both detector heads, scored by the trainer's own rule ------------------ #
            det3d = det_ag = None
            ag_label = None
            if a.boxes:
                ag_label = bool(batch["agent_label"][0])
                tgt = {"box": batch["agent_box"].to(dev_t), "yaw": batch["agent_yaw"].to(dev_t),
                       "cls": batch["agent_cls"].to(dev_t), "valid": batch["agent_valid"].to(dev_t),
                       "occ": batch["agent_occ"].to(dev_t), "rates": batch["agent_rates"].to(dev_t),
                       "rates_mask": batch["agent_rates_mask"].to(dev_t),
                       "cz": batch["agent_cz"].to(dev_t), "h": batch["agent_h"].to(dev_t),
                       "zh_mask": batch["agent_zh_mask"].to(dev_t)}
                det3d = detection_readout(box_slots, tgt, gate=gate, fns=det_fns, use_z=True)
                det_ag = detection_readout(agent_slots_out, tgt, gate=gate, fns=det_fns, use_z=False)
                # ⭐ IDENTITY CHECK of the pairing: the trainer's own box3d centre term on this window
                # (box3d_loss_row -> box3d_set_loss, box3d_head.py:327-451) must equal the L1 mean
                # over OUR matched pairs -- same filter, same matcher, two call sites.
                if ag_label and det3d["n_matched"]:
                    brow = tr._perc.box3d_loss_row(box_slots, tgt, visible_filter=True)
                    dterm = abs(float(brow["box3d_centre"]) - float(det3d["centre_l1_mean"]))
                    box_ctl["trainer_centre_term_max_abs_diff"] = max(
                        box_ctl["trainer_centre_term_max_abs_diff"], dterm)
                    box_ctl["n_trainer_term_checked"] += 1
                    if dterm > 1e-4 or int(brow["box3d_n_target"]) != det3d["n_targets"]:
                        raise SystemExit(f"[mapvid] pairing != the trainer's: centre term "
                                         f"{float(brow['box3d_centre'])} vs {det3d['centre_l1_mean']}, "
                                         f"n_target {brow['box3d_n_target']} vs {det3d['n_targets']}")
                # ⭐ THE KNOWN-VALUE CONTROL: the GT targets written INTO the slot format and sent
                # through the SAME readout + drawing path must each match ITSELF at exactly 0.0 m
                fake = gt_as_slots(tgt["box"][0].cpu(), tgt["yaw"][0].cpu(), tgt["cls"][0].cpu(),
                                   det3d["gt_target"], n_slots=int(box_slots["box"].shape[1]))
                src = fake.pop("src").numpy()
                dctl = detection_readout({k: v.to(dev_t) for k, v in fake.items()}, tgt, gate=gate,
                                         fns=det_fns, use_z=False)
                box_ctl["n_windows"] += 1
                box_ctl["n_targets"] += dctl["n_targets"]
                box_ctl["n_self_matched"] += int(sum(int(src[r_]) == int(c_)
                                                     for r_, c_ in zip(dctl["rows"], dctl["cols"])))
                if dctl["n_matched"]:
                    box_ctl["max_centre_err_m"] = max(box_ctl["max_centre_err_m"],
                                                      float(dctl["centre_l2"].max()))
                box_ctl["greedy_tp"] += dctl["greedy"]["bev"]["tp_at_gate"]
                for r_, c_ in zip(dctl["rows"], dctl["cols"]):
                    cp_ = box_corners_xy(*dctl["box"][r_], dctl["yaw"][r_])
                    cg_ = box_corners_xy(*det3d["gt_box"][c_], det3d["gt_yaw"][c_])
                    box_ctl["max_corner_diff_m"] = max(box_ctl["max_corner_diff_m"],
                                                       float(np.abs(cp_ - cg_).max()))
                for hname, det in (("box3d", det3d), ("agent", det_ag)):
                    acc = det_acc[hname]
                    acc["n_windows"] += 1
                    if not ag_label:
                        acc["n_windows_no_label"] += 1
                        continue
                    acc["ap_rows_bev"] += det["greedy"]["bev"]["rows"]
                    if "3d" in det["greedy"]:
                        acc["ap_rows_3d"] += det["greedy"]["3d"]["rows"]
                    acc["n_gt"] += det["n_targets"]
                    acc["l2"] += det["centre_l2"].tolist()
                    acc["l1"] += (np.abs(det["box"][det["rows"], :2]
                                         - det["gt_box"][det["cols"], :2]).sum(-1).tolist()
                                  if det["n_matched"] else [])
                    acc["n_matched"] += det["n_matched"]
                    acc["n_matched_conf"] += det["n_matched_conf"]
                    acc["tp_bev"] += det["greedy"]["bev"]["tp_at_gate"]
                    acc["n_conf"] += det["greedy"]["bev"]["n_conf"]
                    acc["n_pred"] += det["n_pred"]
                    acc["n_gt_in_range"] += det["n_gt_in_range"]
            if dumper is not None:
                dumper.add(lg, frac, seen, vmask, {
                    "set": "boxes_clip", "clip_sha12": s12, "t_start_row": int(t),
                    "now_row": int(t + W - 1),
                    "map_raw_frame": (int(item["map_raw_frame"]) if "map_raw_frame" in item
                                      else None),
                    "turn_deg": heading_change_deg(ep.poses[:, 2].double().numpy(), int(t + W - 1)),
                    "v0_ms": float(item["pose_last"][3])})
            # ---- numpy views for drawing --------------------------------------------------------- #
            seen_np = seen[0].cpu().numpy().astype(bool)
            valid_np = vmask[0].cpu().numpy().astype(bool) if vmask is not None else None
            cls_pred = argmax_classes(lg)[0].cpu().numpy()
            cls_gt = frac[0].argmax(dim=0).cpu().numpy()
            gt_np, pr_np = r["_gt"][0].cpu().numpy(), r["_pr"][0].cpu().numpy()
            sc_np = r["_scored"][0].cpu().numpy()
            mixed_np = info_o["_mixed"][0].cpu().numpy() & sc_np
            n_mixed_scored = int(mixed_np.sum())
            # ---- GT paths: the trainer's own label function over every 0.1 s step ---------------- #
            pl = item["pose_last"].float()
            fut = item["future_poses_ext"].float()
            fv = item["future_valid_ext"].bool()
            dense = refb_labels.waypoint_targets(pl[None], fut[None], tuple(range(1, 61)))[0].numpy()
            gt_dense = dense[fv.numpy()]
            slot_valid = [bool(fv[h - 1]) for h in horizons]
            gt_slots = refb_labels.waypoint_targets(pl[None], fut[None], horizons)[0].numpy()
            gt_slots_v = gt_slots[np.asarray(slot_valid)]
            gt_trunc_s = None if all(slot_valid) else (
                round(horizons[max(j for j, v in enumerate(slot_valid) if v)] * 0.1, 1)
                if any(slot_valid) else 0.0)
            v0 = float(pl[3])
            nav_i = int(item["nav_cmd"])
            nav_ok = bool(item.get("nav_valid", torch.tensor(False)))
            nav_name = nav_names[nav_i]
            if nav_name != nav:
                raise SystemExit(f"[mapvid] item nav {nav_name} != the selection's nav {nav}")
            nav_tok = tok_of_legacy.get(nav_name, "?")
            t_lab = float(e_ds._now_s(ep, t)) if hasattr(e_ds, "_now_s") else None
            raw_fr = int(item["map_raw_frame"]) if "map_raw_frame" in item else None
            ious.append(r["iou"])
            iou_by[(ci, k_i)] = (r["iou"], t_lab)
            sum_inter += r["inter"]
            sum_union += r["union"]
            clip_mean = float(np.mean(ious))
            clip_pooled = sum_inter / sum_union if sum_union else 0.0
            # ---- the viz standard: declare, or do not render -------------------------------------- #
            els = [
                (VizElement.present("camera", "GT path + selected out['traj'] projected",
                                    source="render_refcv3_video.CylProjector(payload['frame'], per-clip "
                                           "extrinsics) on item['frames'][-1][-3:]", kind="derived")
                 if proj.enabled else
                 VizElement.unavailable("camera", "no per-clip extrinsics for this clip -- the BEV "
                                                  "panels carry the comparison")),
                VizElement.present("bev", "predicted map | SAM3 GT map | drivable agreement "
                                          + ("| detected agents vs GT boxes " if a.boxes else "")
                                          + "(+ GT path, selected path)",
                                   source="softmax(out['perception']['map_logits']).argmax / "
                                          "batch['map_frac'].argmax / trainer IoU rule"
                                          + (" / out['perception']['box_slots'] + out['agent_slots'] "
                                             "at sigmoid(presence) >= presence_gate vs the agent join"
                                             if a.boxes else ""),
                                   kind="model_output", conditioned_on=("nav_cmd",)),
                VizElement.unavailable("tactical", "not drawn: this reel examines the map head only; "
                                                   "refcv6's tactical decoder is scored by the battery"),
                (VizElement.unavailable("strategic", "not drawn: this reel examines the map head only")
                 if has_route else
                 VizElement.unavailable("strategic", "this build emits no route head "
                                                     "(--no-strategic)")),
                VizElement.unavailable("ade", "not computed here: a perception diagnostic; trajectory "
                                              "metrics are the battery's four-family T1 eval"),
                VizElement.present("strategic_input", nav_name,
                                   source="V3Dataset.enable_nav_from_v7 (v8 eval labels, oracle nav, "
                                          "provenance ego-future)", kind="given_input"),
            ]
            els = check_frame(els, where=f"refcv6 map @ {s12}/t{t}")
            if viz_rows is None:
                viz_rows = [e.to_dict() for e in els]
            # ---- panels ----------------------------------------------------------------------- #
            pred_arr = class_map_array(cls_pred, seen_np, valid_np)
            gt_arr = class_map_array(cls_gt, seen_np, valid_np)
            agr_arr = agreement_array(gt_np, pr_np, sc_np, seen_np, valid_np)
            pred_im = draw_map_overlays(pred_arr, gt_dense, gt_slots_v, sel)
            gt_im = draw_map_overlays(gt_arr, gt_dense, gt_slots_v, sel)
            agr_im = draw_map_overlays(agr_arr, None, None, None, paths=False)
            if s12 == ctl_panel_clip:
                # ⭐ control (a), PIXELS: the GT-as-logits through the PREDICTION path must redraw the
                # GT panel exactly -- overlays included (both see the identical overlay inputs).
                c_arr = class_map_array(argmax_classes(z_o)[0].cpu().numpy(), seen_np, valid_np)
                c_im = draw_map_overlays(c_arr, gt_dense, gt_slots_v, sel)
                c2_arr = class_map_array(argmax_classes(z_c)[0].cpu().numpy(), seen_np, valid_np)
                c2_im = draw_map_overlays(c2_arr, gt_dense, gt_slots_v, sel)
                ctl["panel_checked"] += 1
                ctl["panel_equal"] += int(np.array_equal(np.asarray(c_im), np.asarray(gt_im)))
                ctl["consistent_panel_equal"] += int(np.array_equal(np.asarray(c2_im), np.asarray(gt_im)))
            # ---- the canvas --------------------------------------------------------------------- #
            cv = Image.new("RGB", (W_TOT, H_TOT), C_BG)
            d = ImageDraw.Draw(cv)
            d.rectangle([0, 0, W_TOT, H_BAN], fill=C_BAN)
            d.text((PAD, 6), f"{RUN_LABEL} \u00b7 step {a.expect_step:,} \u00b7 {HYBRID_NOTE}",
                   fill=C_FG, font=F["ban"])
            d.text((PAD, 40), DIAG_NOTE + "  \u00b7  SAM3 semantic-map head: map_logits 9\u00d7120\u00d764 "
                   "@ 0.5 m (x 0\u202660 m ahead, y \u00b116 m)", fill=C_WARN, font=F["hud"])
            rt = f"clip {ci + 1}/{len(run_list)} \u00b7 sha12 {s12} \u00b7 window {k_i + 1}/{len(wl)}"
            d.text((W_TOT - PAD - d.textlength(rt, font=F["med"]), 8), rt, fill=C_FG, font=F["med"])
            # camera
            rgb = item["frames"][-1, -3:].permute(1, 2, 0).contiguous().numpy()
            cam = Image.fromarray(rgb)
            if cam.size != (CAM_W, CAM_H):
                raise SystemExit(f"[mapvid] camera frame {cam.size} != layout {(CAM_W, CAM_H)}")
            cd = ImageDraw.Draw(cam, "RGBA")
            if proj.enabled:
                hz = proj([[1e5, 0.0]], up=1)
                if hz and hz[0] is not None:
                    yh = hz[0][1]
                    cd.line([(0, yh), (CAM_W, yh)], fill=(150, 132, 62, 200), width=1)
                    cd.text((8, yh - 15), "horizon predicted from the clip's own extrinsics",
                            fill=(190, 170, 90), font=F["micro"])
                rcv3.polyline(cd, proj(np.concatenate([np.zeros((1, 2)), gt_dense]) if len(gt_dense)
                                       else np.zeros((0, 2)), up=1), C_GT, 6)
                rcv3.polyline(cd, proj(rcv3.densify(sel), up=1), C_SEL, 4)
                for q in proj(sel, up=1):
                    if q is not None:
                        cd.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], outline=C_SEL, width=3)
                for q in proj(gt_slots_v, up=1):
                    if q is not None:
                        cd.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=C_GT)
            else:
                cd.rectangle([0, CAM_H // 2 - 20, CAM_W, CAM_H // 2 + 20], fill=(12, 16, 22))
                cd.text((14, CAM_H // 2 - 10), "CAMERA OVERLAY DISABLED -- no per-clip extrinsics",
                        fill=C_WARN, font=F["hud"])
            if a.boxes and rig_cam is not None:
                # GT cuboids (the trainer's TARGETS: in-field, decode box) in the GT green; the 3-D
                # box head's CONFIDENT slots in their class colour. A GT row without a 3-D label is
                # drawn as its ground footprint (z = 0, h = 0), never with an invented height.
                for i_g in np.nonzero(det3d["gt_target"])[0]:
                    gb_ = det3d["gt_box"][i_g]
                    has_z = det3d["gt_zh"] is not None and bool(det3d["gt_zh"][i_g])
                    draw_cuboid_cam(cd, rig_cam, cuboid_corners(
                        gb_[0], gb_[1], det3d["gt_cz"][i_g] if has_z else 0.0, gb_[2], gb_[3],
                        det3d["gt_h"][i_g] if has_z else 0.0, det3d["gt_yaw"][i_g]),
                        C_GT + (255,), width=2)
                for j_p in np.nonzero(det3d["conf"] & det3d["in_range"])[0]:
                    pb_ = det3d["box"][j_p]
                    draw_cuboid_cam(cd, rig_cam, cuboid_corners(
                        pb_[0], pb_[1], det3d["cz"][j_p], pb_[2], pb_[3], det3d["h"][j_p],
                        det3d["yaw"][j_p]),
                        AGENT_PALETTE[int(det3d["cls"][j_p]) % len(AGENT_PALETTE)] + (255,), width=2)
            if max(lift_eq, trunk_eq or 0) > 0:
                n_eq = max(lift_eq, trunk_eq or 0)
                ye = CAM_H - n_eq
                for xx in range(0, CAM_W, 14):
                    cd.line([(xx, ye), (xx + 7, ye)], fill=(245, 180, 90, 210), width=1)
                if trunk_eq:
                    eq_txt = f"\u2193 bottom {trunk_eq} rows are ZEROED inside the trunk (as built)"
                else:
                    eq_txt = (f"\u2193 bottom {n_eq} rows: NOT zeroed by the trunk (--equalize-bottom-rows "
                              f"{eq_rows} dropped by the config rebuild, refc_v3_train.py:381/:459); "
                              f"the BEV lift treats them as unobserved")
                cd.text((CAM_W - 8 - cd.textlength(eq_txt, font=F["micro"]), ye + 3), eq_txt,
                        fill=(245, 180, 90), font=F["micro"])
            cv.paste(cam, (PAD, Y_CAM))
            d.rectangle([PAD, Y_CAM, PAD + CAM_W, Y_CAM + 20], fill=(9, 12, 17))
            d.text((PAD + 6, Y_CAM + 2),
                   rcv3.fit(d, f"FRONT CAMERA \u2014 last input frame (the trunk's bytes) \u00b7 {cam_label}",
                            F["tiny"], CAM_W - 12),
                   fill=C_DIM, font=F["tiny"])
            # HUD panel
            d.rectangle([X_HUD, Y_CAM, X_HUD + W_HUD, Y_CAM + CAM_H], fill=C_PANEL)
            xh, yh0 = X_HUD + 14, Y_CAM + 10
            tl = (f"t = {t_lab:.2f} s (label clock)" if t_lab is not None else "t = n/a")
            d.text((xh, yh0), f"{tl}  \u00b7  NOW = window row {t + W - 1}"
                   + (f", raw frame {raw_fr}" if raw_fr is not None else ""), fill=C_FG, font=F["hud"])
            d.text((xh, yh0 + 24), f"v0 = {v0:5.2f} m/s  (MEASURED at t0, the only ego scalar fed)",
                   fill=C_FG, font=F["hud"])
            nav_txt = f"nav = {nav_name.upper()}  ({nav_tok})"
            d.text((xh, yh0 + 48), nav_txt, fill=C_GIVEN, font=F["hud"])
            d.text((xh + d.textlength(nav_txt, font=F["hud"]) + 16, yh0 + 51),
                   "GIVEN INPUT, not a prediction \u2014 v7 oracle nav (provenance ego-future)"
                   + ("" if nav_ok else " \u00b7 nav_valid False"), fill=C_DIM, font=F["tiny"])
            yk = yh0 + 80
            if label:
                d.text((xh, yk), "drivable IoU", fill=C_DIM, font=F["sub"])
                d.text((xh, yk + 16), f"{r['iou']:.3f}", fill=C_FG, font=F["big"])
                d.text((xh + 150, yk), "clip mean so far", fill=C_DIM, font=F["sub"])
                d.text((xh + 150, yk + 16), f"{clip_mean:.3f}", fill=C_FG, font=F["big"])
                d.text((xh + 300, yk), f"clip pooled \u03a3I/\u03a3U (n={len(ious)})",
                       fill=C_DIM, font=F["sub"])
                d.text((xh + 300, yk + 16), f"{clip_pooled:.3f}", fill=C_FG, font=F["big"])
                d.text((xh + 560, yk), f"in-run eval @{a.expect_step // 1000}k", fill=C_DIM,
                       font=F["sub"])
                d.text((xh + 560, yk + 16), f"{ref_iou:.3f}", fill=C_DIM, font=F["big"])
                d.text((xh + 560, yk + 58), "128 OTHER windows \u2014 sanity range only", fill=C_DIM2,
                       font=F["micro"])
            else:
                d.text((xh, yk + 10), "NO SAM3 label for this window (map_label False) \u2014 IoU not "
                       "defined here", fill=C_WARN, font=F["hud"])
            d.text((xh, yk + 76),
                   f"scored cells {int(r['n_scored']):,} of {GRID_X * GRID_Y:,} (seen & lift-valid)"
                   f"  \u00b7  model drivable {100 * r['pred_drivable_frac']:.1f} %  \u00b7  SAM3 "
                   f"drivable {100 * r['gt_drivable_frac']:.1f} %  \u00b7  mean p(drivable) "
                   f"{r['pred_drivable_prob_mean']:.3f}", fill=C_FG, font=F["tiny"])
            if a.boxes:
                d.text((xh, yk + 94), f"DETECTED AGENTS  (p = sigmoid(presence) \u2265 {gate:.2f}, the "
                       "model's own presence_gate; counts inside the BEV range)", fill=C_FG,
                       font=F["subb"])
                for li_, (hn_, det_) in enumerate((("3-D box head", det3d), ("agent head", det_ag))):
                    if not ag_label:
                        txt_ = f"{hn_}: pred {det_['n_pred']}  \u00b7  GT: NO agent label on this frame"
                    else:
                        gg_ = det_["greedy"]["bev"]
                        rc_ = gg_["tp_at_gate"] / det_["n_targets"] if det_["n_targets"] else None
                        pc_ = gg_["tp_at_gate"] / gg_["n_conf"] if gg_["n_conf"] else None
                        txt_ = (f"{hn_}: pred {det_['n_pred']} \u00b7 GT {det_['n_gt_in_range']} "
                                f"({det_['n_targets']} targets) \u00b7 Hungarian {det_['n_matched']}/"
                                f"{det_['n_targets']}, centre err "
                                + ("\u2014" if det_["centre_l2_mean"] is None else
                                   f"{det_['centre_l2_mean']:.2f} m")
                                + f" \u00b7 greedy 2 m TP {gg_['tp_at_gate']}"
                                + ("" if rc_ is None else f" R {rc_:.2f}")
                                + ("" if pc_ is None else f" P {pc_:.2f}"))
                    d.text((xh, yk + 116 + li_ * 18), rcv3.fit(d, txt_, F["tiny"], W_HUD - 28),
                           fill=C_FG if li_ == 0 else C_DIM, font=F["tiny"])
                ys = yk + 156
                d.text((xh, ys), f"per-frame drivable IoU over this clip (dashed = in-run eval "
                       f"@{a.expect_step // 1000}k, other windows; thin line = running mean)",
                       fill=C_DIM, font=F["micro"])
                draw_spark(d, xh, ys + 14, W_HUD - 70, 40, ious, len(wl), ref_iou, F)
                yl = ys + 62
            else:
                d.text((xh, yk + 94),
                       f"argmax accuracy on scored cells "
                       f"{100 * float(prow.get('map_acc', float('nan'))):.1f} %"
                       f"  \u00b7  soft-CE map loss {float(prow['loss']):.4f}  \u00b7  mixed cells "
                       f"(argmax drivable, fraction < 0.5) {n_mixed_scored}", fill=C_DIM,
                       font=F["tiny"])
                ys = yk + 118
                d.text((xh, ys), f"per-frame drivable IoU over this clip (dashed = in-run eval "
                       f"@{a.expect_step // 1000}k, other windows; thin line = running mean)",
                       fill=C_DIM, font=F["micro"])
                draw_spark(d, xh, ys + 16, W_HUD - 70, 64, ious, len(wl), ref_iou, F)
                yl = ys + 92
            d.text((xh, yl), "CLASS PALETTE (both class panels)", fill=C_FG, font=F["subb"])
            for i_c, name in enumerate(CHANNELS):
                col, row = i_c % 3, i_c // 3
                swatch(cv, d, xh + col * 272, yl + 22 + row * 18, PALETTE[i_c], F, f"{i_c} {name}",
                       h=12, w=18)
            swatch(cv, d, xh, yl + 80, None, F, "unscored: SAM3 never saw it (~map_seen)", pattern="A")
            swatch(cv, d, xh + 410, yl + 80, None, F, "unscored: SAM3 saw it, the lift cannot reach it "
                   "now (~map_valid)", pattern="B")
            # BEV panels
            draw_bev_block(cv, XP[0], "BEV \u2014 PREDICTED MAP (refcv6)",
                           "argmax softmax(map_logits) \u00b7 hatched = unscored", pred_im, F)
            draw_bev_block(cv, XP[1], "BEV \u2014 SAM3 GT MAP",
                           "argmax map_frac (clip-lifetime label) \u00b7 same hatches", gt_im, F)
            draw_bev_block(cv, XP[2], "DRIVABLE AGREEMENT",
                           "trainer rule: \u2265 0.5 on both sides, seen & lift-valid", agr_im, F)
            n_tp = int((gt_np & pr_np).sum())
            n_fp = int((pr_np & ~gt_np).sum())
            n_fn = int((gt_np & ~pr_np).sum())
            if a.boxes:
                det_im = draw_detection_panel(pred_arr, det3d, det_ag)
                draw_bev_block(cv, XP[3], f"DETECTED AGENTS (p \u2265 {gate:.2f})",
                               "3-D box head solid \u00b7 agent head dashed \u00b7 GT green", det_im, F)
                draw_boxes_info(rcv3, cv, d, F, XI, WI, n_tp, n_fp, n_fn, r["iou"], sel_idx, a.seed,
                                gt_trunc_s, cls_pred, cls_gt, sc_np, prow, n_mixed_scored, els)
            else:
                draw_map_info(rcv3, cv, d, F, XI, WI, n_tp, n_fp, n_fn, r["iou"], sel_idx, a.seed,
                              gt_trunc_s, cls_pred, cls_gt, sc_np, prow, els)
            fpath = frames_dir / f"c{ci}_w{k_i:04d}.png"
            cv.save(fpath, compress_level=1)
            frame_files.append(str(fpath))
            # ---- the per-frame row -------------------------------------------------------------- #
            row = {"clip_rank": ci + 1, "clip_sha12": s12, "nav": nav, "nav_token": nav_tok,
                   "win_rank": k_i + 1, "n_windows_clip": len(wl), "t_start_row": int(t),
                   "now_row": int(t + W - 1), "map_raw_frame": raw_fr,
                   "t_label_s": None if t_lab is None else round(t_lab, 4), "v0_ms": round(v0, 4),
                   "map_label": label, "iou": r["iou"], "inter": r["inter"], "union": r["union"],
                   "n_scored": r["n_scored"], "n_seen": int(seen_np.sum()),
                   "n_scored_prog_map_loss_row": float(prow["n_map_cells"]),
                   "pred_drivable_frac": round(r["pred_drivable_frac"], 6),
                   "gt_drivable_frac": round(r["gt_drivable_frac"], 6),
                   "pred_drivable_prob_mean": round(r["pred_drivable_prob_mean"], 6),
                   "map_soft_ce": round(float(prow["loss"]), 6),
                   "map_acc_argmax": round(float(prow.get("map_acc", float("nan"))), 6),
                   "map_iou_argmax_per_class": {CHANNELS[i]: (None if prow.get(f"map_iou_{i}") is None
                                                              else round(float(prow[f"map_iou_{i}"]), 6))
                                                for i in range(len(CHANNELS))},
                   "tp": n_tp, "fp": n_fp, "fn": n_fn, "n_mixed_scored": n_mixed_scored,
                   "clip_mean_so_far": round(clip_mean, 6), "clip_pooled_so_far": round(clip_pooled, 6),
                   "ctl_consistent_iou": rc["iou"], "ctl_consistent_union": rc["union"],
                   "ctl_onehot_iou": ro["iou"], "ctl_n_mixed_all_cells": info_o["n_mixed"],
                   "ctl_n_impossible": info_c["n_impossible"],
                   "mut_flip_iou": rflip["iou"], "mut_channel_iou": rchan["iou"],
                   "sel_idx": sel_idx, "sel_traj_m": np.round(sel, 3).tolist(),
                   "gt_slots_m": np.round(gt_slots, 3).tolist(), "gt_slot_valid": slot_valid,
                   "map_logits_dtype": str(lg.dtype), "seed": a.seed,
                   "wall_s": round(time.time() - t_w, 2)}
            if a.boxes:
                row["agent_label"] = ag_label
                for hn_, det_ in (("box3d", det3d), ("agent", det_ag)):
                    gb_ = det_["greedy"]["bev"]
                    row[hn_] = {
                        "n_pred_conf_in_range": det_["n_pred"], "n_conf_all": det_["n_conf_all"],
                        "n_gt_in_range": det_["n_gt_in_range"], "n_targets": det_["n_targets"],
                        "n_matched_hungarian": det_["n_matched"],
                        "n_matched_hungarian_conf": det_["n_matched_conf"],
                        "centre_err_l2_mean_m": (None if det_["centre_l2_mean"] is None
                                                 else round(det_["centre_l2_mean"], 4)),
                        "centre_err_l1_mean_m": (None if det_["centre_l1_mean"] is None
                                                 else round(det_["centre_l1_mean"], 4)),
                        "greedy2m_bev_tp_at_gate": gb_["tp_at_gate"],
                        "greedy2m_bev_n_conf": gb_["n_conf"]}
                    if "3d" in det_["greedy"]:
                        row[hn_]["greedy2m_3d_tp_at_gate"] = det_["greedy"]["3d"]["tp_at_gate"]
                row["box_ctl_self_match"] = {"n_targets": dctl["n_targets"],
                                             "n_matched": dctl["n_matched"],
                                             "max_centre_err_m": (float(dctl["centre_l2"].max())
                                                                  if dctl["n_matched"] else 0.0)}
            pf.write(json.dumps(scrub(row), ensure_ascii=True) + "\n")
            pf.flush()
            rows_clip.append(row)
            _p(f"  w{k_i + 1:03d}/{len(wl)} t={t} iou {r['iou']:.4f} mean {clip_mean:.4f} pooled "
               f"{clip_pooled:.4f} | ctl {rc['iou']:.4f}/{ro['iou']:.4f} flip {rflip['iou']:.3f} | "
               f"{time.time() - t_w:.1f} s | RAM {watch.avail_gb():.1f} GB")
        # ---- the clip's summary + its title card -------------------------------------------------- #
        iv = [rw["iou"] for rw in rows_clip if rw["map_label"]]
        cstat = {"clip_rank": ci + 1, "clip_sha12": s12, "nav": nav, "nav_token": tok_of_legacy.get(nav),
                 "n_windows_rendered": len(rows_clip), "n_windows_clip": n_w_clip,
                 "n_labelled": len(iv),
                 "iou_mean": float(np.mean(iv)) if iv else None,
                 "iou_pooled": (sum(rw["inter"] for rw in rows_clip) /
                                max(sum(rw["union"] for rw in rows_clip), 1e-12)) if iv else None,
                 "iou_min": float(np.min(iv)) if iv else None,
                 "iou_median": float(np.median(iv)) if iv else None,
                 "iou_max": float(np.max(iv)) if iv else None,
                 "n_union_zero": sum(1 for rw in rows_clip if rw["union"] == 0),
                 "t_label_s_range": [rows_clip[0]["t_label_s"], rows_clip[-1]["t_label_s"]],
                 "v0_ms_range": [min(rw["v0_ms"] for rw in rows_clip), max(rw["v0_ms"] for rw in rows_clip)],
                 "pred_drivable_frac_mean": float(np.mean([rw["pred_drivable_frac"] for rw in rows_clip])),
                 "gt_drivable_frac_mean": float(np.mean([rw["gt_drivable_frac"] for rw in rows_clip])),
                 "map_acc_argmax_mean": float(np.mean([rw["map_acc_argmax"] for rw in rows_clip])),
                 "map_soft_ce_mean": float(np.mean([rw["map_soft_ce"] for rw in rows_clip])),
                 "camera": {"frame": fr_dict, "codec": codec, "extrinsics_found": proj.enabled,
                            "label": cam_label, "R_vs_pai_extrinsics_table_max_abs": R_chk},
                 "controls": {**ctl, "onehot_iou_mean": ctl["onehot_iou_sum"] / max(ctl["n"], 1),
                              "pixel_check_on_this_clip": s12 == ctl_panel_clip},
                 "draw_independence": draw_indep,
                 "wall_s": round(time.time() - t_clip0, 1)}
        if a.boxes:
            cstat["selection"] = {k: c[k] for k in ("n_boxes_30m_in_field", "n_boxes_30m_any_direction",
                                                    "n_windows_agent_labelled", "turn_max_deg",
                                                    "n_windows_turn_gt_30deg")}
            cstat["projector_cross_check"] = proj_xchk
            det_sum = {}
            for hn_, acc in det_acc.items():
                ap_b = det_fns["ap_from_rows"](acc["ap_rows_bev"], acc["n_gt"])
                ap_3 = (det_fns["ap_from_rows"](acc["ap_rows_3d"], acc["n_gt"])
                        if acc["ap_rows_3d"] else None)
                det_sum[hn_] = {
                    "n_windows": acc["n_windows"], "n_windows_no_agent_label": acc["n_windows_no_label"],
                    "n_targets": acc["n_gt"], "n_gt_in_range": acc["n_gt_in_range"],
                    "n_pred_conf_in_range": acc["n_pred"],
                    "n_matched_hungarian": acc["n_matched"],
                    "n_matched_hungarian_conf": acc["n_matched_conf"],
                    "centre_err_l2_mean_m": float(np.mean(acc["l2"])) if acc["l2"] else None,
                    "centre_err_l2_median_m": float(np.median(acc["l2"])) if acc["l2"] else None,
                    "centre_err_l1_mean_m": float(np.mean(acc["l1"])) if acc["l1"] else None,
                    "greedy2m_bev_tp_at_gate": acc["tp_bev"], "n_conf_all": acc["n_conf"],
                    "recall_at_gate_bev": acc["tp_bev"] / acc["n_gt"] if acc["n_gt"] else None,
                    "precision_at_gate_bev": acc["tp_bev"] / acc["n_conf"] if acc["n_conf"] else None,
                    "ap_2m_bev": ap_b["ap"], "ap_2m_3d": None if ap_3 is None else ap_3["ap"]}
            cstat["detections"] = det_sum
            cstat["box_control"] = {**box_ctl,
                                    "pass": bool(box_ctl["n_self_matched"] == box_ctl["n_targets"]
                                                 and box_ctl["max_centre_err_m"] == 0.0
                                                 and box_ctl["greedy_tp"] == box_ctl["n_targets"]
                                                 and box_ctl["max_corner_diff_m"] == 0.0)}
        clips_out.append(cstat)
        _p(f"[clip {ci + 1}] {s12} {nav}: mean IoU {cstat['iou_mean']} pooled {cstat['iou_pooled']} "
           f"(n={cstat['n_labelled']}); controls {json.dumps(cstat['controls'])}")
        tc = title_card(F, ci, len(run_list), cstat, sel_rec, ref_iou, a.expect_step, a.seed)
        tpath = frames_dir / f"c{ci}_title.png"
        tc.save(tpath, compress_level=1)
    pf.close()
    # ---- --dump-map: the class-analysis dump, SAME process / SAME model load ------------------- #
    if dumper is not None:
        t_d = time.time()
        if full_ds is None:
            raise SystemExit("[mapvid] --dump-map needs --boxes (its selection builds the full "
                             "139-clip dataset the in-run windows live in)")
        f_ds, f_eps = full_ds
        n_inrun = int(targs.eval_batches) * int(targs.batch)
        perm_all = torch.randperm(len(f_ds), generator=torch.Generator().manual_seed(12345)).tolist()
        perm_inrun = L.inrun_eval_perm(f_ds, int(targs.eval_batches), int(targs.batch))
        if perm_all[:n_inrun] != list(perm_inrun):
            raise SystemExit("[mapvid] the continued permutation does not start with the in-run perm")
        s12_of = [sha12(clip_of(ep_)) for ep_ in f_eps]
        no_gt = {c_["sha12"] for c_ in cands if c_["n_ok"] == 0}
        map_set = {s_.strip() for s_ in a.map_clips.split(",") if s_.strip()}
        tags: dict = {}                                        # key (sha12, t) -> set of tags
        for k_ in dumper.keys:
            tags.setdefault(k_, set()).add("boxes_clip")
        want: dict = {}                                        # f_ds window index -> key
        n_skip_nogt = 0

        def _want(wi_, tag_):
            nonlocal n_skip_nogt
            e_, t_ = f_ds.index[wi_]
            key_ = (s12_of[e_], int(t_))
            if key_[0] in no_gt:
                n_skip_nogt += 1
                return
            tags.setdefault(key_, set()).add(tag_)
            want.setdefault(wi_, key_)
        if small:
            for wi_ in perm_inrun[:a.smoke_dump_windows]:
                _want(wi_, "inrun_eval")
        else:
            for wi_ in perm_inrun:
                _want(wi_, "inrun_eval")
            for wi_, (e_, t_) in enumerate(f_ds.index):
                if s12_of[e_] in map_set:
                    _want(wi_, "map_clip")
            k_ = n_inrun
            while len(tags) < a.dump_min_windows and k_ < len(perm_all):
                _want(perm_all[k_], "inrun_perm_ext")
                k_ += 1
        todo = sorted((wi_ for wi_, key_ in want.items() if not dumper.has(*key_)),
                      key=lambda wi_: tuple(f_ds.index[wi_]))
        _p(f"[mapvid:dump] {len(tags)} unique windows wanted ({len(dumper.keys)} already dumped by the "
           f"render); {len(todo)} forwards to run; {n_skip_nogt} window picks skipped (clip has no "
           f"SAM3 GT file)")
        for n_, wi_ in enumerate(todo):
            e_, t_ = f_ds.index[wi_]
            item_ = f_ds[wi_]
            b_ = torch.utils.data.default_collate([item_])
            o_ = forward_like_trainer(tr, model, b_, device, mode=mode, ablate_frames=ablate,
                                      seed=a.seed)
            lg_ = o_["perception"]["map_logits"].detach()
            mv_ = o_["perception"].get("map_valid")
            del o_
            vm_ = mv_.detach() if (use_valid_mask and mv_ is not None) else None
            key_ = want[wi_]
            dumper.add(lg_, b_["map_frac"].to(dev_t), b_["map_seen"].to(dev_t), vm_, {
                "set": sorted(tags[key_]), "clip_sha12": key_[0], "t_start_row": int(t_),
                "now_row": int(t_ + W - 1),
                "map_raw_frame": (int(item_["map_raw_frame"]) if "map_raw_frame" in item_
                                  else None),
                "turn_deg": heading_change_deg(f_eps[e_].poses[:, 2].double().numpy(),
                                               int(t_ + W - 1)),
                "v0_ms": float(item_["pose_last"][3]), "map_label": bool(b_["map_label"][0])})
            if (n_ + 1) % 50 == 0:
                _p(f"[mapvid:dump] {n_ + 1}/{len(todo)} ({time.time() - t_d:.0f} s, RAM "
                   f"{watch.avail_gb():.1f} GB)")
        for m_ in dumper.meta:                             # membership of EVERY dumped window
            m_["set"] = sorted(tags[(m_["clip_sha12"], int(m_["t_start_row"]))])
        # ⭐ THE LIFT'S OWN SAMPLING GEOMETRY for the resolution question: the bank's cached
        # grid [Z, X, Y, 2] is normalised for grid_sample(align_corners=False) on the stride-16
        # feature map (bev_lift.py:47, :112-120), so the feature pixel each BEV cell reads is
        # recoverable exactly. Saved for the rendered clips and the map video's clips.
        _geo_clips = sorted({c_["sha12"] for c_ in run_list} | map_set)
        _fh, _fw = int(pframe.height) // 16, int(pframe.width) // 16
        for e_, ep_ in enumerate(f_eps):
            if s12_of[e_] not in _geo_clips:
                continue
            g_, v_ = model._lift_bank.geometry(int(ep_.episode_id))
            np.savez(Path(dumper.dir) / f"lift_geometry_{s12_of[e_]}.npz",
                     grid=g_.detach().float().cpu().numpy(), valid=v_.detach().cpu().numpy(),
                     heights_m=np.asarray(model._lift_bank.heights_m, np.float32),
                     feat_hw=np.asarray([_fh, _fw], np.int64))
        rec["map_dump"] = {**dumper.close(), "n_skipped_no_gt": n_skip_nogt,
                           "n_by_set": {s_: sum(1 for v_ in tags.values() if s_ in v_)
                                        for s_ in ("inrun_eval", "boxes_clip", "map_clip",
                                                   "inrun_perm_ext")},
                           "inrun_perm": {"seed": 12345, "n": n_inrun,
                                          "source": "refcv6_loader.inrun_eval_perm (train():7661-7663)"},
                           "wall_s": round(time.time() - t_d, 1)}
        _p(f"[mapvid:dump] done: {rec['map_dump']['n_windows']} windows -> {rec['map_dump']['dir']} "
           f"({rec['map_dump']['wall_s']} s); by set {rec['map_dump']['n_by_set']}")
    rec["viz_standard_elements"] = viz_rows
    rec["ram"] = watch.report()
    if device == "cuda":
        rec["cuda_max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    # ---- the sequence: title card, then every window, per clip ---------------------------------- #
    seq = work / f"seq_{tag}"
    if seq.exists():
        shutil.rmtree(seq)
    seq.mkdir(parents=True)
    n_title = max(1, int(round(a.title_s * a.fps)))
    order = []
    for ci in range(len(run_list)):
        order += [frames_dir / f"c{ci}_title.png"] * n_title
        order += sorted(frames_dir.glob(f"c{ci}_w*.png"))
    for n, src in enumerate(order):
        dst = seq / f"f{n:06d}.png"
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = out_dir / f"{tag}.mp4"

    def encode(dst, crf, scale=None):
        cmd = [ffmpeg, "-y", "-r", str(a.fps), "-i", str(seq / "f%06d.png"), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"]
        if scale:
            cmd[-4:-4] = ["-vf", f"scale={scale}:-2:flags=lanczos"]
        res = subprocess.run(cmd + [str(dst)], capture_output=True, text=True)
        if res.returncode != 0 or not Path(dst).exists() or Path(dst).stat().st_size == 0:
            raise SystemExit(f"[mapvid] ffmpeg failed ({res.returncode}): {res.stderr[-800:]}")
        return {"path": str(dst), "bytes": Path(dst).stat().st_size, "crf": crf, "scale": scale,
                "cmd": " ".join(cmd[1:])}
    rec["video"] = {"frames": len(order), "fps": a.fps, "title_frames_per_clip": n_title,
                    "resolution": f"{W_TOT}x{H_TOT}", "duration_s": round(len(order) / a.fps, 1),
                    "ffmpeg": ffmpeg, "full": encode(mp4, 18)}
    if rec["video"]["full"]["bytes"] >= 30 * 2 ** 20:
        rec["video"]["small"] = encode(out_dir / f"{tag}_small.mp4", 26, scale=1600)
    # ---- decode-back verification (verify_mp4.py, reused) ----------------------------------------- #
    vpy = tools / "verify_mp4.py"
    vfiles = [rec["video"]["full"]["path"]] + ([rec["video"]["small"]["path"]]
                                               if "small" in rec["video"] else [])
    vr = subprocess.run([sys.executable, str(vpy)] + vfiles, capture_output=True, text=True)
    rec["video"]["verify_mp4"] = {"exit": vr.returncode, "stdout_tail": vr.stdout[-2500:],
                                  "stderr_tail": vr.stderr[-800:]}
    _p(vr.stdout[-2500:])
    # ---- contact sheet (render: 4 keyframes per clip at quantiles; smoke: every frame) ------------ #
    sheet = out_dir / ("contact_sheet_smoke.png" if small else "contact_sheet.png")
    rec["contact_sheet"] = make_contact_sheet(sheet, frames_dir, clips_out, F,
                                              "smoke" if small else a.mode, iou_by,
                                              step=a.expect_step)
    # ---- per-clip + record ------------------------------------------------------------------------ #
    per_clip = {
        "what": ("refcv6 SAM3 map head, drivable IoU under the TRAINER'S rule "
                 "(refc_v3_train.py:4506-4521 @ 82c2331) on EVERY eval window of each rendered clip"),
        "caveat": DIAG_NOTE + "; one clip = one episode: windows are correlated, no interval quoted",
        "run": RUN_LABEL, "step": a.expect_step, "hybrid_note": HYBRID_NOTE,
        "ckpt_md5": rec["ckpt"]["md5"], "config": rec["config"],
        "inrun_eval_map_iou_drivable": {"value": ref_iou, "step": a.expect_step,
                                        "windows": inrun.get("eval_windows"),
                                        "batches": inrun.get("eval_batches"),
                                        "note": "a DIFFERENT window set -- a sanity range, not an identity"},
        "estimators": {"iou_mean": "mean over windows of the per-window IoU",
                       "iou_pooled": "sum of intersections / sum of unions over the clip's windows",
                       "interval": "none -- windows of one episode are not independent"},
        "clips": clips_out, "selection": sel_rec["chosen"], "device": device, "seed": a.seed,
        "equalize_bottom_rows": rec["equalize_bottom_rows"],
    }
    if a.boxes:
        per_clip["detections_rule"] = rec["detections"]
        _ib = {k: v for k, v in ev[0].items() if k.startswith(("eval_box3d", "eval_agent"))}
        per_clip["inrun_eval_detection_terms"] = {
            "row": _ib, "note": "per-batch means over 8 x 16 OTHER windows; *_centre is the L1 "
                                "centre term |dx|+|dy| over Hungarian pairs (agent_slots.py:885) -- a "
                                "sanity range for centre_err_l1_mean_m, not an identity"}
    write_json(out_dir / ("per_clip_smoke.json" if small else "per_clip.json"), per_clip)
    rec["per_frame"] = str(pf_path)
    rec["wall_s"] = round(time.time() - t_all, 1)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(out_dir / ("render_record_smoke.json" if small else "render_record.json"), rec)
    watch.stop.set()
    if not a.keep_frames:
        shutil.rmtree(seq, ignore_errors=True)
    write_marker("ok", video=rec["video"]["full"]["path"], bytes=rec["video"]["full"]["bytes"],
                 frames=rec["video"]["frames"],
                 clips=[{"sha12": c["clip_sha12"], "nav": c["nav"], "iou_mean": c["iou_mean"],
                         "n": c["n_labelled"],
                         **({"box3d_ap_2m_bev": c["detections"]["box3d"]["ap_2m_bev"],
                             "box3d_n_targets": c["detections"]["box3d"]["n_targets"]}
                            if c.get("detections") else {})} for c in clips_out],
                 map_dump=(None if "map_dump" not in rec else
                           {"n_windows": rec["map_dump"]["n_windows"], "dir": rec["map_dump"]["dir"]}),
                 wall_s=rec["wall_s"])
    _p(f"[mapvid] DONE {rec['video']['full']['path']} ({rec['video']['full']['bytes']:,} B, "
       f"{rec['video']['frames']} frames) in {rec['wall_s']} s; RAM {json.dumps(rec['ram'])}")
    return 0


def title_card(F, ci, n_clips, cs, sel_rec, ref_iou, step, seed):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W_TOT, H_TOT), C_BG)
    d = ImageDraw.Draw(im)
    x = 120
    d.text((x, 110), f"Clip {ci + 1} of {n_clips}  \u2014  nav {cs['nav'].upper()}  ({cs['nav_token']}, a "
           "GIVEN INPUT)", fill=C_FG, font=F["title"])
    tr_ = cs["t_label_s_range"]
    t_txt = ("n/a" if None in tr_ else f"{tr_[0]:.1f} \u2026 {tr_[1]:.1f} s (label clock)")
    d.text((x, 172), f"sha12 {cs['clip_sha12']}  \u00b7  {cs['n_windows_rendered']} eval windows, every "
           f"one, in time order  \u00b7  t = {t_txt}  \u00b7  "
           f"v0 {cs['v0_ms_range'][0]:.1f} \u2026 {cs['v0_ms_range'][1]:.1f} m/s", fill=C_DIM, font=F["med"])
    d.text((x, 250), f"{RUN_LABEL} \u00b7 step {step:,} \u00b7 {HYBRID_NOTE}", fill=C_FG, font=F["ban"])
    d.text((x, 290), DIAG_NOTE, fill=C_WARN, font=F["ban"])
    y = 370
    if cs["iou_mean"] is not None:
        d.text((x, y), "drivable IoU over this clip's windows (trainer rule, \u2265 0.5 both sides, "
               "seen & lift-valid cells):", fill=C_DIM, font=F["hud"])
        d.text((x, y + 30), f"mean {cs['iou_mean']:.3f}   \u00b7   pooled {cs['iou_pooled']:.3f}   "
               f"\u00b7   n = {cs['n_labelled']} windows   \u00b7   min {cs['iou_min']:.3f} / median "
               f"{cs['iou_median']:.3f} / max {cs['iou_max']:.3f}", fill=C_FG, font=F["med"])
        d.text((x, y + 66), f"in-run eval_map_iou_drivable @ step {step:,} = {ref_iou:.3f} over 128 OTHER "
               "windows \u2014 a sanity range, not an identity. One clip = one episode: no interval is "
               "quoted.", fill=C_DIM, font=F["hud"])
    ct = cs["controls"]
    d.text((x, y + 120), "controls on this clip:", fill=C_DIM, font=F["hud"])
    lines = [
        f"known-value (GT as threshold-consistent logits): IoU == 1.0 on {ct['consistent_iou_eq_1']}/"
        f"{ct['n'] - ct['consistent_union0']} windows with a non-empty union",
        f"literal one-hot(argmax): mean IoU {ct['onehot_iou_mean']:.4f} (mixed cells argmax-drivable "
        f"but < 0.5: {ct['n_mixed_total']:,} over the clip)",
        f"mutations must go RED: lateral mirror {ct['mut_flip_red']}/{ct['n']}, drivable channel off by "
        f"one {ct['mut_chan_red']}/{ct['n']}",
    ]
    if ct.get("panel_checked"):
        lines.append(f"pixel check: GT-as-logits panel == GT panel on {ct['consistent_panel_equal']}/"
                     f"{ct['panel_checked']} frames (literal one-hot: {ct['panel_equal']}/"
                     f"{ct['panel_checked']})")
    di = cs.get("draw_independence") or {}
    if di:
        lines.append(f"DDIM draw independence (seed {di['seeds'][0]} vs {di['seeds'][1]}, first window): "
                     f"map_logits max |diff| {di['map_logits_max_abs_diff']:.2e}, drivable mask identical "
                     f"{di['drivable_mask_identical']}, selected path moved {di['traj_max_abs_diff_m']:.2f} m")
    bc = cs.get("box_control")
    if bc:
        lines.append(f"boxes known-value: GT targets fed through the predicted-box path matched THEMSELVES "
                     f"{bc['n_self_matched']}/{bc['n_targets']}, max centre error {bc['max_centre_err_m']:.1e} m, "
                     f"max corner diff {bc['max_corner_diff_m']:.1e} m → "
                     f"{'PASS' if bc['pass'] else 'FAIL'}")
    for hn, det in (cs.get("detections") or {}).items():
        def _f(v, fmt):
            return "—" if v is None else format(v, fmt)
        lines.append(f"{'3-D box head' if hn == 'box3d' else 'agent head'}: {det['n_targets']} GT targets "
                     f"over {det['n_windows'] - det['n_windows_no_agent_label']} labelled windows · "
                     f"Hungarian centre err {_f(det['centre_err_l2_mean_m'], '.2f')} m (mean) · "
                     f"greedy 2 m at p ≥ gate: recall {_f(det['recall_at_gate_bev'], '.2f')}, "
                     f"precision {_f(det['precision_at_gate_bev'], '.2f')} · AP@2 m (BEV) "
                     f"{_f(det['ap_2m_bev'], '.3f')}")
    for li, ln in enumerate(lines):
        d.text((x + 20, y + 150 + li * 28), ln, fill=C_FG, font=F["hud"])
    y2 = y + 150 + len(lines) * 28 + 30
    d.text((x, y2), "how to read a frame:", fill=C_DIM, font=F["hud"])
    guide = [
        "top-left: the front camera (last input frame) \u00b7 green = GT future path, orange = refcv6's own "
        "selected path out['traj']",
        "bottom: PREDICTED map (argmax) \u00b7 SAM3 GT map (argmax) \u00b7 DRIVABLE AGREEMENT (TP blue, FP "
        "pink, FN yellow) \u2014 ego at the bottom centre, lines every 10 m",
        "hatched cells are NOT scored: diagonal = SAM3 never saw the cell, dots = seen, but the lift's "
        "camera cannot reach it at this instant",
        f"selection rule: {sel_rec['rule']}",
        f"orange path = one DDIM draw (torch.manual_seed({seed}) before every window); the map head does "
        "not depend on the draw (checked above)",
    ]
    for li, ln in enumerate(guide):
        for lj, sub in enumerate(_wrap_text(d, ln, F["hud"], W_TOT - 2 * x - 20)):
            d.text((x + 20, y2 + 30 + li * 52 + lj * 22), sub, fill=C_DIM, font=F["hud"])
    return im


def _wrap_text(d, text, f, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = f"{cur} {w}".strip()
        if d.textlength(t, font=f) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def make_contact_sheet(path, frames_dir, clips_out, F, mode, iou_by=None, *, step: int):
    """``step`` is REQUIRED: the header once hard-coded "step 35,000" and stamped the step-38,000
    boxes sheet with the wrong checkpoint while every frame inside it read 38,000 (2026-09-26)."""
    from PIL import Image, ImageDraw
    iou_by = iou_by or {}
    picks = []
    for ci, cs in enumerate(clips_out):
        files = sorted(frames_dir.glob(f"c{ci}_w*.png"))
        if not files:
            continue
        if mode == "smoke":
            idx = list(range(len(files)))
        else:
            n = len(files)
            idx = sorted(set(min(n - 1, int(round(q * (n - 1)))) for q in (0.1, 0.37, 0.63, 0.9)))
        picks += [(ci, cs, i, files[i]) for i in idx]
    cols = 4
    tw, th = W_TOT // 4, H_TOT // 4
    cap = 22
    rows = max(1, math.ceil(len(picks) / cols))
    sheet = Image.new("RGB", (cols * tw, 56 + rows * (th + cap)), C_BG)
    d = ImageDraw.Draw(sheet)
    d.text((12, 8), f"{RUN_LABEL} \u00b7 step {int(step):,} \u00b7 {HYBRID_NOTE} \u2014 {DIAG_NOTE}",
           fill=C_FG, font=F["hud"])
    d.text((12, 32), "keyframes at quantiles 0.1 / 0.37 / 0.63 / 0.9 of each clip's windows; caption = "
           "clip sha12 \u00b7 nav \u00b7 window \u00b7 frame IoU (trainer rule)", fill=C_DIM, font=F["tiny"])
    picked = []
    for k, (ci, cs, i, fp) in enumerate(picks):
        im = Image.open(fp).convert("RGB").resize((tw, th), Image.LANCZOS)
        r_, c_ = divmod(k, cols)
        x0, y0 = c_ * tw, 56 + r_ * (th + cap)
        sheet.paste(im, (x0, y0 + cap))
        iou, t_lab = iou_by.get((ci, i), (None, None))
        cap_txt = (f"{cs['clip_sha12']} \u00b7 {cs['nav']} \u00b7 window {i + 1}/{cs['n_windows_rendered']}"
                   + ("" if t_lab is None else f" \u00b7 t {t_lab:.1f} s")
                   + ("" if iou is None else f" \u00b7 IoU {iou:.3f}"))
        d.text((x0 + 6, y0 + 3), cap_txt, fill=C_FG, font=F["tiny"])
        picked.append({"clip_sha12": cs["clip_sha12"], "nav": cs["nav"], "window": i + 1, "iou": iou,
                       "t_label_s": t_lab})
    sheet.save(path)
    return {"path": str(path), "n_keyframes": len(picks), "size": list(sheet.size), "keyframes": picked}


if __name__ == "__main__":
    sys.exit(main())
