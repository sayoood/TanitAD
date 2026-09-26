"""nuScenes open-loop planning -> TanitEval, PROTOCOL-VERBATIM (EvalFlyWheel W6 / E5, 2026-09-19).

⛔ NOT A TANITAD CRITERION — AND NOTHING HERE CAN MAKE IT ONE
------------------------------------------------------------
nuScenes open-loop planning is inadmissible as a TanitAD criterion (register ``H-EVAL-6``
SUPPORTED) and SKIP claim-bearing in the PI-approved portfolio (``D-BENCH-PORT``, 2026-08-29).
Every artifact written here carries ``claim_bearing: false``; the value is a module constant,
there is no parameter that sets it, and :func:`build_artifact` re-asserts it on the way out.
Why (``products/P7-TanitEval/benchmarks/NUSCENES_PROTOCOL.md``): there is no official protocol;
the averaging convention alone moves ONE checkpoint 0.72 -> 1.22 m and flips the UniAD/VAD
ranking; the GT human trajectory scores 0.36-0.96 % collision; and the "high-level command" the
field feeds its planners is the GT future thresholded at +-2 m (our route-echo defect).

WHAT IS VERBATIM (PUBLISHED-CODE, pinned — extracts + blob-sha checks in the W6 package
``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-nuscenes-planning-harness/raw/reference_extracts``)
---------------------------------------------------------------------------------------------
* the three collision/L2 KERNELS — ST-P3 ``stp3/metrics.py:263-396 @69aabef``, UniAD
  ``planning_metrics.py:15-149 @609ee08`` (x flip in ``update``, masked L2), VAD
  ``metric_stp3.py:166-308 @1688c4b`` (row-flipped grid; its point check reads the wrong cell —
  reproduced, see :func:`kernel_vad`). AD-MLP re-uses ST-P3's (``deps/stp3 @4b93ba0``).
* the two REDUCTIONS — UniAD's own switch ``nuscenes_e2e_dataset.py:1041-1044 @609ee08``:
  ``uniad`` = value AT t, ``stp3`` = mean of the per-timestep values UP TO t
  (== ST-P3 ``evaluate.py:71-73,135-137,166``). UniAD's first public commit
  (``4e91222 :1024-1036``) printed at-t only; the ``stp3`` option arrived 2024-03-19 (``33cd8ec``).
* the SAMPLE SETS — UniAD: every sample, missing future steps masked to 0 error (denominator
  still counts the sample); VAD: ``fut_valid_flag`` (6 future samples); ST-P3: 3 past + 6 future
  samples in one scene. On full val these are 6,019 / 5,119 / 4,819 (the last two are the counts
  BEV-Planner and AD-MLP publish) — :func:`expected_sample_counts`.
* the GT — the LIDAR_TOP sensor ORIGIN at the next 6 keyframes in the t0 LiDAR frame (all three).
* the OCCUPANCY builders for UniAD (vehicles only, invisible included, invalid frames = 255) and
  VAD (agents at t0: >=1 lidar point, |x|<15, |y|<30, 10 classes; futures from the annotation
  chain; category-index sets {2..8} / {14..23}) — rasterised with a PORT of ``cv2.fillPoly``
  (OpenCV 4.5.4 ``drawing.cpp``), because the venv has no OpenCV. ⚠️ Parity with a real cv2 is
  UNVERIFIED on this box (the parity test skips without cv2 and is reported NOT RUN).

WHAT IS NOT (named, not hidden)
-------------------------------
* ST-P3's occupancy builder (per-frame labels warped into t0 with ``grid_sample``) — NOT
  implemented; the ST-P3 pipeline therefore scores L2 only, collision is ``UNAVAILABLE``.
* Box-library yaw conventions inside mmdet3d are not replicated: the builders rasterise the
  PHYSICAL footprint (centre, size, heading). A reference whose box library mis-orients a box
  would differ; UNVERIFIED until a parity run against the reference label pipelines.
* Accumulation is float64 (the references accumulate float32); differences sit far below the
  published 2-decimal precision.

INPUT ADAPTER (our models)
--------------------------
CAM_FRONT is a rectified pinhole (``docs/schema_nuscenes.md``: "All camera images come
undistorted and rectified"). It is RAY-RESAMPLED into the cylindrical training frame by
``tanitad.data.calib.pinhole_rectify(..., frame=...)`` — never a resize — and the observed mask is
STAMPED: nominal intrinsics into ``PHYSICALAI_WIDE120_256x640`` observe 70,737 / 163,840 px
(0.4317, columns 145..488), because CAM_FRONT spans ~64.6 deg of the frame's 120 deg.
Model plans (our ego frame, x forward / y left) are converted to the LiDAR-origin trajectory with
the lever arm, since that is what every reference scores against.

Usage::

    python -m adapters.nuscenes_planning selftest
    python -m adapters.nuscenes_planning geometry
    python -m adapters.nuscenes_planning run --nuscenes-root D:/Data/nuscenes --split val \
        --protocol nuScenes_OL_L2_uniad --arms GT,STOP,CV --out <run_dir>
"""
from __future__ import annotations

import csv
import datetime as _dt
import enum
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# 0. Constants and provenance                                                   #
# --------------------------------------------------------------------------- #
SCHEMA = "taniteval.nuscenes_planning/1"

#: ⛔ The only value this module will ever write. Not a parameter anywhere.
CLAIM_BEARING = False
CLAIM_BEARING_REASON = (
    "H-EVAL-6 (SUPPORTED): nuScenes open-loop planning is inadmissible as a TanitAD criterion - "
    "no official protocol; the L2 averaging convention alone moves one VAD checkpoint 0.72 -> "
    "1.22 m (+70 %) and flips the UniAD/VAD ranking; the GT human trajectory scores 0.36-0.96 % "
    "collision; the field's 'high-level command' is the GT future thresholded at +-2 m (our "
    "route-echo defect). D-BENCH-PORT (PI, 2026-08-29): SKIP claim-bearing. Usable ONLY as a "
    "cited external-comparability row.")
REGISTER_ROWS = ("H-EVAL-6", "D-BENCH-PORT")

PINS = {
    "stp3": {"repo": "OpenDriveLab/ST-P3", "commit": "69aabefd2610951d9e34238142776ed2228673be",
             "kernel": "stp3/metrics.py:263-396", "reduction": "evaluate.py:71-73,135-137,162-166",
             "samples": "stp3/datas/NuscenesData.py:96-148", "gt": "stp3/datas/NuscenesData.py:505-532"},
    "uniad": {"repo": "OpenDriveLab/UniAD", "commit": "609ee083ea51c3521c323f1279dfc4cee0e60467",
              "kernel": "projects/mmdet3d_plugin/uniad/dense_heads/planning_head_plugin/planning_metrics.py:15-149",
              "reduction": "projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py:1030-1050",
              "first_public_commit": "4e91222da855fa83a8caa55f512536e82f9f9818 (nuscenes_e2e_dataset.py:1024-1036, at-t only)",
              "stp3_option_added": "33cd8ecf7ad5ea4b7d779a2176841d3c1c4d4174 (2024-03-19)",
              "samples_gt": "projects/mmdet3d_plugin/datasets/data_utils/trajectory_api.py:226-282",
              "occupancy": "projects/mmdet3d_plugin/datasets/pipelines/occflow_label.py:11-160; base_e2e.py:560-561"},
    "vad": {"repo": "hustvl/VAD", "commit": "1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8",
            "kernel": "projects/mmdet3d_plugin/VAD/planner/metric_stp3.py:75-308",
            "per_sample": "projects/mmdet3d_plugin/VAD/VAD.py:593-641",
            "reduction": "projects/mmdet3d_plugin/datasets/nuscenes_vad_dataset.py:1795-1813",
            "samples_gt": "tools/data_converter/vad_nuscenes_converter.py:248-254,349-459",
            "agent_filters": "nuscenes_vad_dataset.py:1226-1229; transform_3d.py:11-56; VAD_base_e2e.py:11,330-340"},
    "admlp": {"repo": "E2E-AD/AD-MLP", "commit": "4b93ba085ee47474152f282177865796ea577fc0",
              "evaluator": "deps/stp3/evaluate_for_mlp.py:53-120 (ST-P3 reduction on filter_token.pkl)"},
    "devkit": {"repo": "nutonomy/nuscenes-devkit", "commit": "b40adc467b919192899405d9b77871afee8efa07",
               "splits": "python-sdk/nuscenes/utils/splits.py:102-150"},
    "opencv_fillpoly": {"repo": "opencv/opencv", "tag": "4.5.4",
                        "source": "modules/imgproc/src/drawing.cpp:80-145,159-297,1258-1471,1984-2014; "
                                  "include/opencv2/imgproc.hpp:4832-4912"},
    "mmdet3d_names": {"repo": "open-mmlab/mmdetection3d", "tag": "v0.17.1",
                      "source": "mmdet3d/datasets/nuscenes_dataset.py:54-69,110-112"},
}

#: The six future keyframes every reference scores (2 Hz, 3 s).
HORIZONS_S = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
N_FUTURE = 6
COLUMNS = ("0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s")
HEADLINE_HORIZONS = ("1s", "2s", "3s", "avg_1_2_3s")

#: Identical in ST-P3 (config.py:70-79), UniAD (planning_metrics.py:22-33), VAD (metric_stp3.py:14-33).
EGO_W, EGO_H = 1.85, 4.084
BOUND = (-50.0, 50.0, 0.5)
DX = np.float32(0.5)                 # gen_dx_bx: row[2]
BX = np.float32(-50.0 + 0.5 / 2.0)   # gen_dx_bx: row[0] + row[2]/2 = -49.75
BEV = 200                            # (50 - -50) / 0.5

#: mmdet3d v0.17.1 NuScenesDataset.NameMapping / CLASSES (nuscenes_dataset.py:54-69,110-112).
NAME_MAPPING = {
    "movable_object.barrier": "barrier", "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus", "vehicle.bus.rigid": "bus", "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle", "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian", "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "movable_object.trafficcone": "traffic_cone", "vehicle.trailer": "trailer",
    "vehicle.truck": "truck"}
DET_CLASSES = ("car", "truck", "trailer", "bus", "construction_vehicle", "bicycle",
               "motorcycle", "pedestrian", "traffic_cone", "barrier")
#: UniAD occflow_label.py:30-31 (only_vehicle=True, base_e2e.py:560-561).
UNIAD_VEHICLE_CLASSES = ("car", "bus", "construction_vehicle", "bicycle", "motorcycle",
                         "truck", "trailer")
#: VAD metric_stp3.py:35-38 — raw ``nusc.category`` INDICES (not names). ⚠️ v1.0 has 23
#: categories (0..22), so index 23 can never match; which names 2..8 / 14..22 denote depends on
#: category.json's ORDER — :func:`vad_category_index_audit` prints it on first contact.
VAD_HUMAN_INDEX = frozenset(range(2, 9))
VAD_VEHICLE_INDEX = frozenset(range(14, 24))
#: VAD's INTENDED classes, selected by NAME so the answer cannot depend on category.json's ORDER.
#: Pre-registered 2026-09-26 (…/2026-09-26-suite-runnability-audit/raw/nuscenes/PREREG_VAD_NAME_BASED.md,
#: sha256 3ca3f905…). Under the 32-entry lidarseg ordering VAD's literal index sets are exactly these
#: two name families (W6's F10 — INHERITED), so on the metadata VAD itself used this reproduces it
#: verbatim; on the 23-entry base metadata the index sets pick barriers and cones instead.
VAD_PEDESTRIAN_PREFIX = "human.pedestrian."
VAD_VEHICLE_PREFIX = "vehicle."


def vad_target_by_name(cat: str) -> int:
    """VAD occupancy target by category NAME: 1 = vehicle map, 2 = pedestrian map, 0 = not an agent.

    Replaces ``1 if idx in VAD_VEHICLE_INDEX else (2 if idx in VAD_HUMAN_INDEX else 0)``, whose answer
    depended on the ORDER of category.json. Applied AFTER the unchanged ``NAME_MAPPING ∈ DET_CLASSES``
    filter, so the effective set is 12 categories: 4 pedestrians (adult, child, construction_worker,
    police_officer) and 8 vehicles (car, truck, bus.bendy, bus.rigid, trailer, construction,
    motorcycle, bicycle). ⛔ Barriers and traffic cones are detection classes but in NEITHER map.
    """
    if cat.startswith(VAD_VEHICLE_PREFIX):
        return 1
    if cat.startswith(VAD_PEDESTRIAN_PREFIX):
        return 2
    return 0
#: VAD_base_e2e.py:11 point_cloud_range -> CustomObjectRangeFilter bev range (x_min, y_min, x_max, y_max).
VAD_BEV_RANGE = (-15.0, -30.0, 15.0, 30.0)

#: devkit can_bus_api.py:51-53 + ST-P3 NuscenesData.py:103 (scene-0419 lacks vehicle monitor).
STP3_SCENE_BLACKLIST = frozenset([419, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172,
                                  173, 174, 175, 176, 309, 310, 311, 312, 313, 314])
STP3_RECEPTIVE_FIELD = 3        # stp3/configs/nuscenes/Planning.yml TIME_RECEPTIVE_FIELD
STP3_N_FUTURE = 6               # stp3/configs/nuscenes/Planning.yml N_FUTURE_FRAMES


class RefusedInput(ValueError):
    """An input the protocol (or our binding rules) does not admit — never silently repaired."""


# --------------------------------------------------------------------------- #
# 1. Tags — the only way a number can exist is with its convention attached     #
# --------------------------------------------------------------------------- #
class Reduction(enum.Enum):
    """How the six per-timestep values become the 1 s / 2 s / 3 s columns."""
    AT_T = "uniad-noavg"            # value AT t        (UniAD nuscenes_e2e_dataset.py:1043-1044)
    AVG_UP_TO_T = "stp3-temavg"     # mean UP TO t      (ST-P3 evaluate.py:166; UniAD :1041-1042)


class Pipeline(enum.Enum):
    """Whose kernel, sample set and occupancy produced the per-timestep values."""
    UNIAD = "uniad@609ee08"
    VAD = "vad@1688c4b"
    STP3 = "stp3@69aabef"


#: W1's CLOSED protocol set for ``nuscenes_ol`` (bench_run.schema.json) -> the implementation.
#: ``nuScenes_OL_L2_stp3`` binds the VAD pipeline: VAD is the source of every widely-cited
#: averaged-convention row (VAD, SparseDrive L2, DiffusionDrive, Senna) and of PARA-Drive's
#: "VAD evaluation methodology"; the ST-P3 pipeline is available as a declared variant.
PROTOCOLS = {
    "nuScenes_OL_L2_uniad": (Reduction.AT_T, Pipeline.UNIAD),
    "nuScenes_OL_L2_stp3": (Reduction.AVG_UP_TO_T, Pipeline.VAD),
}


@dataclass(frozen=True)
class ProtocolTag:
    """(reduction, pipeline) — attached to EVERY number. Build with :func:`protocol_tag` or
    :func:`variant_tag`; a bare string is refused wherever a tag is required."""
    reduction: Reduction
    pipeline: Pipeline

    def __post_init__(self):
        if not isinstance(self.reduction, Reduction):
            raise TypeError(f"reduction must be a Reduction member, got {self.reduction!r}")
        if not isinstance(self.pipeline, Pipeline):
            raise TypeError(f"pipeline must be a Pipeline member, got {self.pipeline!r}")

    @property
    def protocol(self) -> str | None:
        """The W1 closed-set name when this is a PUBLISHED combination, else None."""
        for name, (red, pipe) in PROTOCOLS.items():
            if red is self.reduction and pipe is self.pipeline:
                return name
        return None

    @property
    def published_family(self) -> str:
        """``NUSCENES_PROTOCOL.md`` §7.1 tag family, or ``variant:`` for an attribution-only mix."""
        if self.reduction is Reduction.AT_T and self.pipeline is Pipeline.UNIAD:
            return "uniad-noavg"
        if self.reduction is Reduction.AVG_UP_TO_T and self.pipeline in (Pipeline.VAD, Pipeline.STP3):
            return "stp3-temavg"
        return f"variant:{self.reduction.value}+{self.pipeline.value}"

    def __str__(self) -> str:
        return f"nuscenes-planning/{self.reduction.value}+{self.pipeline.value}"

    def to_dict(self) -> dict:
        return {"tag": str(self), "reduction": self.reduction.value,
                "pipeline": self.pipeline.value, "protocol": self.protocol,
                "published_family": self.published_family}


def protocol_tag(name: str) -> ProtocolTag:
    """The tag of a W1 closed-set protocol name. Anything else is refused."""
    if name not in PROTOCOLS:
        raise RefusedInput(f"unknown nuScenes protocol {name!r}; closed set: {sorted(PROTOCOLS)}")
    red, pipe = PROTOCOLS[name]
    return ProtocolTag(red, pipe)


def variant_tag(reduction: Reduction, pipeline: Pipeline) -> ProtocolTag:
    """An attribution variant (e.g. PARA-Drive's 'remove averaging' step = AT_T on VAD's
    pipeline). Tagged like any other number; never a leaderboard protocol (``.protocol`` None)."""
    return ProtocolTag(reduction, pipeline)


# --------------------------------------------------------------------------- #
# 2. Kernels — verbatim ports (numpy float32 == torch float32 for these ops)     #
# --------------------------------------------------------------------------- #
def _point_in_polygon(xp: np.ndarray, yp: np.ndarray, x: float, y: float) -> bool:
    """Crossing-number test (skimage ``_shared/geometry.pyx::point_in_polygon``)."""
    c = False
    j = len(xp) - 1
    for i in range(len(xp)):
        if (((yp[i] <= y) and (y < yp[j])) or ((yp[j] <= y) and (y < yp[i]))) and \
                (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]):
            c = not c
        j = i
    return c


def _polygon_pixels(r: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``skimage.draw.polygon(r, c)`` with ``shape=None``: integer pixels inside the polygon."""
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    minr, maxr = int(max(0, r.min())), int(math.ceil(r.max()))
    minc, maxc = int(max(0, c.min())), int(math.ceil(c.max()))
    rr, cc = [], []
    for ri in range(minr, maxr + 1):
        for ci in range(minc, maxc + 1):
            if _point_in_polygon(c, r, ci, ri):
                rr.append(ri)
                cc.append(ci)
    return np.asarray(rr, dtype=np.int64), np.asarray(cc, dtype=np.int64)


def ego_box_rc() -> np.ndarray:
    """The rasterised ego footprint ``rc`` [32, 2] shared by all three kernels
    (UniAD planning_metrics.py:49-58, ST-P3 metrics.py:298-307, VAD metric_stp3.py:177-186).

    No pixel centre lies on the box edges (they sit at 96.416/104.584 and 97.65/101.35), so every
    point-in-polygon edge convention yields the same 32 pixels — the count the reference's own
    comment states (``# (n_future, 32, 2)``, planning_metrics.py:64)."""
    pts = np.array([[-EGO_H / 2. + 0.5, EGO_W / 2.], [EGO_H / 2. + 0.5, EGO_W / 2.],
                    [EGO_H / 2. + 0.5, -EGO_W / 2.], [-EGO_H / 2. + 0.5, -EGO_W / 2.]])
    bx = np.array([BX, BX], dtype=np.float32)
    dx = np.array([DX, DX], dtype=np.float32)
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    rr, cc = _polygon_pixels(pts[:, 1], pts[:, 0])
    return np.concatenate([rr[:, None], cc[:, None]], axis=-1)


_RC = ego_box_rc()


def _single_coll_stp3(traj: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """ST-P3 / UniAD ``evaluate_single_coll`` (identical bodies): box raster per timestep."""
    n = traj.shape[0]
    t = traj.astype(np.float32).reshape(n, 1, 2).copy()
    t[:, :, [0, 1]] = t[:, :, [1, 0]]
    t = t / DX
    t = t + _RC                                   # float32 + int64 -> float64 (numpy, as in ref)
    r = np.clip(t[:, :, 0].astype(np.int32), 0, BEV - 1)
    c = np.clip(t[:, :, 1].astype(np.int32), 0, BEV - 1)
    out = np.zeros(n, dtype=bool)
    for k in range(n):
        out[k] = bool(np.any(seg[k, r[k], c[k]]))
    return out


def _coll_stp3_family(trajs: np.ndarray, gt: np.ndarray, seg: np.ndarray):
    """ST-P3/UniAD ``evaluate_coll`` after its ``* [-1, 1]`` flip. Returns per-sample per-step
    (obj_col, obj_box_col, gt_box_col) as int8 arrays [N, T]."""
    B, T, _ = trajs.shape
    trajs = (trajs.astype(np.float32) * np.array([-1, 1], dtype=np.float32))
    gt = (gt.astype(np.float32) * np.array([-1, 1], dtype=np.float32))
    obj = np.zeros((B, T), np.int8)
    box = np.zeros((B, T), np.int8)
    gtc = np.zeros((B, T), np.int8)
    ti = np.arange(T)
    for i in range(B):
        g = _single_coll_stp3(gt[i], seg[i])
        xx, yy = trajs[i, :, 0], trajs[i, :, 1]
        yi = ((yy - BX) / DX).astype(np.int64)            # torch .long(): trunc toward 0
        xi = ((xx - BX) / DX).astype(np.int64)
        m1 = (yi >= 0) & (yi < BEV) & (xi >= 0) & (xi < BEV) & ~g
        obj[i, ti[m1]] = (seg[i, ti[m1], yi[m1], xi[m1]] != 0).astype(np.int8)
        m2 = ~g
        b = _single_coll_stp3(trajs[i], seg[i])
        box[i, ti[m2]] = b[ti[m2]].astype(np.int8)
        gtc[i] = g.astype(np.int8)
    return obj, box, gtc


@dataclass(frozen=True)
class KernelOutput:
    """Per-sample, per-timestep values from ONE pipeline's kernel, before any reduction.

    ``scored`` is the pipeline's sample filter (UniAD: all; VAD: fut_valid; ST-P3: sequence);
    ``valid`` [N, T] is the GT-timestep mask (UniAD scores masked steps as 0 but COUNTS the
    sample). ``obj_col`` / ``obj_box_col`` already carry the GT-exclusion rule;
    ``gt_box_col`` is the RAW GT-trajectory collision (the instrument's false-positive floor)."""
    pipeline: Pipeline
    l2: np.ndarray
    obj_col: np.ndarray
    obj_box_col: np.ndarray
    gt_box_col: np.ndarray
    valid: np.ndarray
    scored: np.ndarray
    collision_available: bool = True
    collision_unavailable_reason: str = ""

    def __post_init__(self):
        if not isinstance(self.pipeline, Pipeline):
            raise TypeError("KernelOutput.pipeline must be a Pipeline member")
        n = self.l2.shape[0]
        for a in (self.obj_col, self.obj_box_col, self.gt_box_col, self.valid):
            if a.shape != (n, N_FUTURE):
                raise ValueError(f"kernel arrays must be [N, {N_FUTURE}], got {a.shape}")
        if self.scored.shape != (n,):
            raise ValueError("scored must be [N]")


def _check_inputs(pred, gt, occ=None):
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.ndim != 3 or pred.shape[1:] != (N_FUTURE, 2) or gt.shape != pred.shape:
        raise RefusedInput(f"pred/gt must be [N, {N_FUTURE}, 2] in the LiDAR frame; got "
                           f"{pred.shape} / {gt.shape}")
    if occ is not None:
        occ = np.asarray(occ)
        if occ.shape != (pred.shape[0], N_FUTURE, BEV, BEV):
            raise RefusedInput(f"occupancy must be [N, {N_FUTURE}, {BEV}, {BEV}], got {occ.shape}")
    return pred, gt, occ


def kernel_uniad(pred, gt, gt_mask, occ) -> KernelOutput:
    """UniAD ``PlanningMetric.update`` (planning_metrics.py:120-142 @609ee08).

    ``update`` negates x of pred AND gt, L2 is masked ``sqrt(((d)**2 * mask).sum(-1))``, then
    ``evaluate_coll`` negates x again — so the collision grid is indexed rows=forward,
    cols=RIGHT (UniAD's LiDAR-frame occupancy, occflow_label.py:147-157)."""
    pred, gt, occ = _check_inputs(pred, gt, occ)
    m = np.asarray(gt_mask, dtype=bool)
    if m.shape != pred.shape[:2]:
        raise RefusedInput(f"gt_mask must be [N, {N_FUTURE}], got {m.shape}")
    p = pred.copy()
    g = gt.copy()
    p[..., 0] = -p[..., 0]
    g[..., 0] = -g[..., 0]
    mask2 = np.repeat(m[..., None], 2, axis=-1).astype(np.float32)
    l2 = np.sqrt((((p - g) ** 2) * mask2).sum(axis=-1)).astype(np.float64)
    obj, box, gtc = _coll_stp3_family(p, g, occ)
    return KernelOutput(Pipeline.UNIAD, l2, obj, box, gtc, m.copy(),
                        np.ones(pred.shape[0], dtype=bool))


def kernel_stp3(pred, gt, occ=None, scored=None) -> KernelOutput:
    """ST-P3 ``PlanningMetric.update`` (metrics.py:376-389 @69aabef); AD-MLP's deps copy is the
    same with the UniAD x-flip commented out (deps/stp3/stp3/planning_metrics.py:133-134).

    Collision grid rows=forward, cols=LEFT (ST-P3's yaw-aligned ego-frame labels,
    NuscenesData.py:336-345). ``occ=None`` -> collision UNAVAILABLE (the ST-P3 occupancy builder
    — per-frame labels warped into t0 — is not implemented here; named gap)."""
    pred, gt, occ = _check_inputs(pred, gt, occ)
    n = pred.shape[0]
    l2 = np.sqrt(((pred - gt) ** 2).sum(axis=-1)).astype(np.float64)
    sc = np.ones(n, dtype=bool) if scored is None else np.asarray(scored, dtype=bool)
    if occ is None:
        z = np.zeros((n, N_FUTURE), np.int8)
        return KernelOutput(Pipeline.STP3, l2, z, z.copy(), z.copy(),
                            np.ones((n, N_FUTURE), bool), sc, collision_available=False,
                            collision_unavailable_reason=(
                                "ST-P3 occupancy builder (per-frame BEV labels warped into t0 by "
                                "cumulative_warp_features_reverse) is not implemented in this "
                                "harness; supply ST-P3's own occupancy to score collision"))
    obj, box, gtc = _coll_stp3_family(pred, gt, occ)
    return KernelOutput(Pipeline.STP3, l2, obj, box, gtc, np.ones((n, N_FUTURE), bool), sc)


def _single_coll_vad(traj: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """VAD ``evaluate_single_coll`` (metric_stp3.py:166-240): the row index is FLIPPED
    (``r = 200 - (...)``) to match its ``lidar2cv_rot`` occupancy (rows = -forward)."""
    n = traj.shape[0]
    t = traj.astype(np.float32).reshape(n, 1, 2).copy()
    t[:, :, [0, 1]] = t[:, :, [1, 0]]
    t = t / DX
    t = t + _RC
    r = np.clip((np.int64(BEV) - t[:, :, 0]).astype(np.int32), 0, BEV - 1)
    c = np.clip(t[:, :, 1].astype(np.int32), 0, BEV - 1)
    out = np.zeros(n, dtype=bool)
    for k in range(n):
        out[k] = bool(np.any(seg[k, r[k], c[k]]))
    return out


def kernel_vad(pred, gt, occ, fut_valid, unavailable_reason: str = "") -> KernelOutput:
    """VAD ``compute_planner_metric_stp3`` + ``evaluate_coll`` (VAD.py:593-641,
    metric_stp3.py:242-308 @1688c4b). No x flip; L2 is the per-waypoint Euclidean distance.

    ⚠️ VERBATIM DEFECT: the point check (``obj_col``) computes
    ``xi = ((-bx[0]/2 - yy)/dx)``, ``yi = ((-bx[1]/2 + xx)/dx)`` = ``49.75 - 2y``, ``49.75 + 2x``
    (metric_stp3.py:272-273) — ~50 cells from the ego (the grid centre is 100). A waypoint at
    (0, 10) reads cell (29, 49). Reproduced, because verbatim is the contract; VAD's published
    collision column is ``obj_box_col`` (the box check), which is not affected."""
    pred, gt, occ = _check_inputs(pred, gt, occ)
    n = pred.shape[0]
    fv = np.asarray(fut_valid, dtype=bool)
    l2 = np.sqrt(((pred - gt) ** 2).sum(axis=-1)).astype(np.float64)
    if occ is None:
        # ⛔ REFUSED, never guessed: L2 stays valid, collision is not computed. The caller passes
        # ``occ=None`` when VAD's literal category indices do not select VAD's intended classes on
        # this category.json (see :func:`vad_category_order_ok`) — the grid would otherwise hold
        # the wrong objects and yield a plausible, wrong collision rate.
        z = np.zeros((n, N_FUTURE), np.int8)
        return KernelOutput(Pipeline.VAD, l2, z, z.copy(), z.copy(), np.ones((n, N_FUTURE), bool), fv,
                            collision_available=False,
                            collision_unavailable_reason=unavailable_reason or (
                                "VAD occupancy not supplied; collision not computed"))
    obj = np.zeros((n, N_FUTURE), np.int8)
    box = np.zeros((n, N_FUTURE), np.int8)
    gtc = np.zeros((n, N_FUTURE), np.int8)
    ti = np.arange(N_FUTURE)
    half = np.float32(-BX / np.float32(2))                   # -bx/2 = 24.875
    for i in range(n):
        g = _single_coll_vad(gt[i], occ[i])
        xx, yy = pred[i, :, 0], pred[i, :, 1]
        xi = ((half - yy) / DX).astype(np.int64)
        yi = ((half + xx) / DX).astype(np.int64)
        m1 = (xi >= 0) & (xi < BEV) & (yi >= 0) & (yi < BEV) & ~g
        obj[i, ti[m1]] = (occ[i, ti[m1], xi[m1], yi[m1]] != 0).astype(np.int8)
        m2 = ~g
        b = _single_coll_vad(pred[i], occ[i])
        box[i, ti[m2]] = b[ti[m2]].astype(np.int8)
        gtc[i] = g.astype(np.int8)
    return KernelOutput(Pipeline.VAD, l2, obj, box, gtc, np.ones((n, N_FUTURE), bool), fv)


# --------------------------------------------------------------------------- #
# 3. Reductions — the ONLY place the two conventions differ                    #
# --------------------------------------------------------------------------- #
def _reduce_at_t(v: np.ndarray) -> dict:
    """UniAD ``planning_evaluation_strategy == "uniad"``: ``value[i]`` (…dataset.py:1043-1044)."""
    cols = {COLUMNS[i]: float(v[i]) for i in range(N_FUTURE)}
    h = {"1s": float(v[1]), "2s": float(v[3]), "3s": float(v[5])}
    h["avg_1_2_3s"] = (h["1s"] + h["2s"] + h["3s"]) / 3.0
    return {"columns": cols, "headline": h}


def _reduce_avg_up_to_t(v: np.ndarray) -> dict:
    """ST-P3 ``value.mean()`` over the first 2/4/6 steps (evaluate.py:166) == UniAD
    ``strategy == "stp3"``: ``value[:i+1].mean()`` (…dataset.py:1041-1042)."""
    cols = {COLUMNS[i]: float(np.mean(v[:i + 1])) for i in range(N_FUTURE)}
    h = {"1s": float(np.mean(v[:2])), "2s": float(np.mean(v[:4])), "3s": float(np.mean(v[:6]))}
    h["avg_1_2_3s"] = (h["1s"] + h["2s"] + h["3s"]) / 3.0
    return {"columns": cols, "headline": h}


#: The dispatch the source-level mutation arm in the tests swaps. Keep it one literal.
_REDUCERS = {Reduction.AT_T: _reduce_at_t, Reduction.AVG_UP_TO_T: _reduce_avg_up_to_t}


def reduce_per_timestep(v: Sequence[float], reduction: Reduction) -> dict:
    """Six per-timestep values -> columns + headline, under ONE convention (returned tagged
    only through :func:`score`; this raw helper exists for the published-row cross-check)."""
    if not isinstance(reduction, Reduction):
        raise TypeError("reduction must be a Reduction member")
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (N_FUTURE,):
        raise RefusedInput(f"need {N_FUTURE} per-timestep values, got shape {arr.shape}")
    return _REDUCERS[reduction](arr)


# --------------------------------------------------------------------------- #
# 4. Tagged results — impossible to hold an untagged or mixed number            #
# --------------------------------------------------------------------------- #
METRICS = {
    "L2_m": "L2 distance [m] between the planned and the logged LiDAR-origin waypoint",
    "collision_box_pct": "ego-box collision rate [%] (obj_box_col) — the published 'Col.' column",
    "collision_point_pct": "ego-centre-point collision rate [%] (obj_col) — NOT the published column",
    "gt_collision_box_pct": "RAW collision rate [%] of the GT human trajectory itself — the "
                            "instrument's false-positive floor (PARA-Drive fn. 4). Denominator: ALL "
                            "scored samples (masked steps contribute 0), as the reference compute() — "
                            "NOT valid steps only (see gt_collision_box_pct_valid_steps_only)",
}


@dataclass(frozen=True)
class TaggedValue:
    """One number, never without its protocol tag, metric, horizon and n."""
    tag: ProtocolTag
    metric: str
    horizon: str
    value: float
    n_samples: int

    def __post_init__(self):
        if not isinstance(self.tag, ProtocolTag):
            raise TypeError(f"TaggedValue needs a ProtocolTag, got {type(self.tag).__name__}")
        if self.metric not in METRICS:
            raise RefusedInput(f"unknown metric {self.metric!r}")
        if self.horizon not in COLUMNS + HEADLINE_HORIZONS:
            raise RefusedInput(f"unknown horizon {self.horizon!r}")
        if not isinstance(self.value, float) or not math.isfinite(self.value):
            raise RefusedInput(f"value must be a finite float, got {self.value!r}")

    def to_dict(self) -> dict:
        return {"protocol_tag": str(self.tag), "protocol": self.tag.protocol,
                "metric": self.metric, "horizon": self.horizon,
                "value": self.value, "n_samples": self.n_samples}


@dataclass(frozen=True)
class ConventionResult:
    """Every value of one pipeline under ONE reduction. Construction refuses a second tag."""
    tag: ProtocolTag
    rows: tuple
    n_samples: int
    collision_available: bool
    collision_unavailable_reason: str = ""

    def __post_init__(self):
        if not isinstance(self.tag, ProtocolTag):
            raise TypeError("ConventionResult needs a ProtocolTag")
        for r in self.rows:
            if not isinstance(r, TaggedValue):
                raise TypeError("rows must be TaggedValue")
            if r.tag != self.tag:
                raise RefusedInput(f"MIXED CONVENTIONS refused: row tagged {r.tag} inside a "
                                   f"result tagged {self.tag}")

    def get(self, metric: str, horizon: str) -> TaggedValue:
        for r in self.rows:
            if r.metric == metric and r.horizon == horizon:
                return r
        raise KeyError(f"{metric}@{horizon} not in {self.tag}")

    def to_dict(self) -> dict:
        block = {"protocol_tag": self.tag.to_dict(), "n_samples": self.n_samples,
                 "collision": ({"status": "OK"} if self.collision_available else
                               {"status": "UNAVAILABLE", "reason": self.collision_unavailable_reason,
                                "n": self.n_samples}),
                 "rows": [r.to_dict() for r in self.rows]}
        for r in block["rows"]:
            if r["protocol_tag"] != str(self.tag):              # belt and braces
                raise RefusedInput("row/block tag mismatch while serialising")
        return block


def per_timestep_means(k: KernelOutput) -> dict:
    """Six per-timestep means over the pipeline's scored samples (the ``compute()`` of each ref).

    UniAD: sum over ALL samples / total (masked steps contribute 0). VAD / ST-P3: mean over the
    scored samples only. Collision rates in percent, exactly as the papers print them."""
    sc = k.scored
    n = int(sc.sum())
    if n == 0:
        raise RefusedInput(f"{k.pipeline.value}: no scored samples")
    out = {"n": n,
           "L2_m": k.l2[sc].sum(axis=0) / n,
           "collision_box_pct": 100.0 * k.obj_box_col[sc].sum(axis=0) / n,
           "collision_point_pct": 100.0 * k.obj_col[sc].sum(axis=0) / n}
    v = k.valid[sc]
    num = (k.gt_box_col[sc] * v).sum(axis=0)           # invalid (255-filled) steps contribute 0
    # ⭐ DEFAULT = the reference compute() denominator: ALL scored samples, masked steps contributing 0 —
    # the SAME convention as the model-arm columns above. Pre-registered and CONFIRMED 2026-09-26
    # (…/raw/nuscenes/PREREG_UNIAD_GT_FLOOR_GAP.md, sha256 08677e75…): with it the UniAD GT floor reads
    # 0.3655 / 0.3821 / 0.3655 % at 1/2/3 s against PARA-Drive Tab. 8's 0.35 / 0.38 / 0.35 % (+4.4/+0.6/+4.4 %),
    # reproducing its 1 s = 3 s < 2 s shape. The valid-steps-only denominator used before read
    # 0.3847 / 0.4244 / 0.4298 — +10/+12/+23 %, growing with horizon exactly as the invalid fraction does.
    # For VAD / ST-P3 every scored sample has a full future, so the two denominators coincide there.
    out["gt_collision_box_pct"] = 100.0 * num / n
    # the previous convention, KEPT and LABELLED so every earlier banked number stays explainable
    out["gt_collision_box_pct_valid_steps_only"] = 100.0 * num / np.maximum(v.sum(axis=0), 1)
    out["gt_valid_n"] = v.sum(axis=0)
    return out


def score(k: KernelOutput, tag: ProtocolTag) -> ConventionResult:
    """The only producer of numbers. ``tag`` has no default and must be a :class:`ProtocolTag`
    whose pipeline is the kernel's own — a UniAD kernel cannot be labelled VAD."""
    if not isinstance(tag, ProtocolTag):
        raise TypeError(f"score() needs a ProtocolTag (got {type(tag).__name__}); build one with "
                        f"protocol_tag(name) or variant_tag(reduction, pipeline)")
    if not isinstance(k, KernelOutput):
        raise TypeError("score() needs a KernelOutput")
    if tag.pipeline is not k.pipeline:
        raise RefusedInput(f"tag pipeline {tag.pipeline.value} != kernel pipeline {k.pipeline.value}")
    m = per_timestep_means(k)
    rows = []
    metrics = ["L2_m"] + (["collision_box_pct", "collision_point_pct", "gt_collision_box_pct"]
                          if k.collision_available else [])
    for metric in metrics:
        red = _REDUCERS[tag.reduction](m[metric])
        for col, val in red["columns"].items():
            rows.append(TaggedValue(tag, metric, col, float(val), m["n"]))
        for h, val in red["headline"].items():
            rows.append(TaggedValue(tag, metric, h, float(val), m["n"]))
    return ConventionResult(tag, tuple(rows), m["n"], k.collision_available,
                            k.collision_unavailable_reason)


# --------------------------------------------------------------------------- #
# 5. Artifact — claim_bearing:false by construction                             #
# --------------------------------------------------------------------------- #
COMMAND_SOURCES = ("none", "predicted_goal")
#: The field's standard input — refused (NUSCENES_PROTOCOL.md §6.3, registry nusc.plan.command_source).
REFUSED_COMMAND_SOURCES = ("gt_future_derived", "gt_ego_fut_cmd", "ego_fut_cmd", "gt")


def build_artifact(result: ConventionResult, *, arm: str, arm_kind: str,
                   declared_inputs: Sequence[str], command_source: str,
                   split: Mapping, gt_control: Mapping | None = None,
                   nonstraight: Mapping | None = None, input_geometry: Mapping | None = None,
                   families: Mapping | None = None, provenance: Mapping | None = None) -> dict:
    """The TanitEval artifact for one arm under one convention.

    Registry keys (CRITERIA_REGISTRY.json benchmarks.nuscenes.tasks.planning_openloop):
    ``benchmark.nuscenes.planning.{protocol_tag, gt_control, command_source, nonstraight}``.
    ⛔ No ``four_families`` / ``headline.ade_*`` key is emitted: those are the checker's in-scope
    markers, and this artifact must never be scored as a TanitAD driving eval."""
    if not isinstance(result, ConventionResult):
        raise TypeError("build_artifact needs a ConventionResult")
    if command_source in REFUSED_COMMAND_SOURCES:
        raise RefusedInput(
            f"command_source {command_source!r} is the GT future thresholded at +-2 m handed back "
            f"as an input (VAD vad_nuscenes_converter.py:452-457; UniAD trajectory_api.py:272-280) "
            f"— inadmissible under the 2026-08-03 goal-input rule. Use 'none' or 'predicted_goal'.")
    if command_source not in COMMAND_SOURCES:
        raise RefusedInput(f"command_source must be one of {COMMAND_SOURCES}, got {command_source!r}")
    if arm_kind not in ("model", "floor", "reference"):
        raise RefusedInput(f"arm_kind must be model|floor|reference, got {arm_kind!r}")
    if arm_kind == "model" and not input_geometry:
        raise RefusedInput("a model arm needs its input-geometry stamp (frame, observed_frac)")
    tag = result.tag
    art = {
        "schema": SCHEMA,
        "claim_bearing": CLAIM_BEARING,
        "claim_bearing_reason": CLAIM_BEARING_REASON,
        "register_rows": list(REGISTER_ROWS),
        "tier": "OPEN-LOOP-L2",
        "loop": {"type": "OPEN", "note": "single-shot plan from t0 inputs scored against the "
                                          "logged future; nothing is ever driven (EVAL_DOCTRINE: "
                                          "only T2 is closed loop)"},
        "arm": {"name": arm, "kind": arm_kind, "declared_inputs": list(declared_inputs)},
        # the registry's canonical tag key path (protocol_tags.key_paths) so the criteria checker
        # reads a REGISTERED tag rather than a free-text string
        "protocol": {"nuscenes_protocol": tag.protocol, "reduction": tag.reduction.value,
                     "pipeline": tag.pipeline.value, "command_source": command_source,
                     "inference_inputs": list(declared_inputs),
                     "tier_note": "single-shot plan from t0 inputs, scored against the logged "
                                  "future; nothing is driven (EVAL_DOCTRINE: only T2 is closed loop)"},
        "benchmark": {"nuscenes": {"planning": {
            "protocol_tag": str(tag),
            "protocol": tag.protocol,
            "published_family": tag.published_family,
            "reference_implementation": PINS[{Pipeline.UNIAD: "uniad", Pipeline.VAD: "vad",
                                              Pipeline.STP3: "stp3"}[tag.pipeline]],
            "command_source": command_source,
            "gt_control": dict(gt_control) if gt_control else {
                "status": "UNAVAILABLE", "reason": "not computed for this artifact",
                "n": result.n_samples},
            "nonstraight": dict(nonstraight) if nonstraight else {
                "status": "UNAVAILABLE", "reason": "subset not computed", "n": 0},
            "results": result.to_dict(),
        }}},
        "split": dict(split),
        "input_geometry": dict(input_geometry) if input_geometry else None,
        "four_family_supplements": dict(families) if families else {
            f: {"status": "UNAVAILABLE", "reason": "not computed", "n": result.n_samples}
            for f in ("longitudinal", "lateral", "tactical", "strategic")},
        "estimator": {"interval": {"status": "UNAVAILABLE",
                                   "reason": "no cluster unit pre-registered for nuScenes (scene vs "
                                             "log); the published rows carry no interval either",
                                   "n": result.n_samples},
                      "deprecated_and_refused": "overlapping_holdout_se is NOT used"},
        "provenance": dict(provenance or {}),
    }
    if art["claim_bearing"] is not False:                        # cannot happen; asserted anyway
        raise AssertionError("claim_bearing must be False")
    return art


# --------------------------------------------------------------------------- #
# 6. cv2.fillPoly port — OpenCV 4.5.4, LINE_8, shift 0 (drawing.cpp)            #
# --------------------------------------------------------------------------- #
_XY_SHIFT = 16
_XY_ONE = 1 << _XY_SHIFT


def _c_div(a: int, b: int) -> int:
    """C/C++ integer division (truncation toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _cv_clip_line(w: int, h: int, x1: int, y1: int, x2: int, y2: int):
    """``clipLine(Size2l, Point2l&, Point2l&)`` (drawing.cpp:92-145)."""
    right, bottom = w - 1, h - 1
    if w <= 0 or h <= 0:
        return False, x1, y1, x2, y2
    c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8
    c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8
    if (c1 & c2) == 0 and (c1 | c2) != 0:
        if c1 & 12:
            a = 0 if c1 < 8 else bottom
            x1 += int((a - y1) * (x2 - x1) / (y2 - y1))      # (int64)(double ...): trunc
            y1 = a
            c1 = (x1 < 0) + (x1 > right) * 2
        if c2 & 12:
            a = 0 if c2 < 8 else bottom
            x2 += int((a - y2) * (x2 - x1) / (y2 - y1))
            y2 = a
            c2 = (x2 < 0) + (x2 > right) * 2
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1:
                a = 0 if c1 == 1 else right
                y1 += int((a - x1) * (y2 - y1) / (x2 - x1))
                x1 = a
                c1 = 0
            if c2:
                a = 0 if c2 == 1 else right
                y2 += int((a - x2) * (y2 - y1) / (x2 - x1))
                x2 = a
                c2 = 0
    return (c1 | c2) == 0, x1, y1, x2, y2


def _cv_line8(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, value) -> None:
    """``Line(img, pt1, pt2, color, 8)`` via ``LineIterator(img, …, 8, leftToRight=true)``
    (drawing.cpp:159-297; imgproc.hpp:4832-4912)."""
    h, w = img.shape[:2]
    if not (0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h):
        ok, x1, y1, x2, y2 = _cv_clip_line(w, h, x1, y1, x2, y2)
        if not ok:
            return
    delta_x, delta_y = 1, 1
    dx, dy = x2 - x1, y2 - y1
    if dx < 0:                                    # leftToRight
        dx, dy = -dx, -dy
        x1, y1 = x2, y2
    if dy < 0:
        dy = -dy
        delta_y = -1
    vert = dy > dx
    if vert:
        dx, dy = dy, dx
        delta_x, delta_y = delta_y, delta_x
    err = dx - (dy + dy)
    plus_delta, minus_delta = dx + dx, -(dy + dy)
    minus_shift, plus_shift, minus_step, plus_step = delta_x, 0, 0, delta_y
    if vert:
        plus_step, plus_shift = plus_shift, plus_step
        minus_step, minus_shift = minus_shift, minus_step
    px, py = x1, y1
    for _ in range(dx + 1):
        img[py, px] = value
        neg = err < 0
        err += minus_delta + (plus_delta if neg else 0)
        px += minus_shift + (plus_shift if neg else 0)       # cols
        py += minus_step + (plus_step if neg else 0)         # rows


def cv_fill_poly(img: np.ndarray, pts: Sequence[Sequence[int]], value=1) -> np.ndarray:
    """``cv2.fillPoly(img, [pts], value)`` for ONE contour of INTEGER (x, y) vertices,
    lineType=LINE_8, shift=0 — a port of drawing.cpp ``fillPoly -> CollectPolyEdges ->
    FillEdgeCollection`` (OpenCV 4.5.4). In place; returns ``img``."""
    h, w = img.shape[:2]
    v = [(int(x), int(y)) for x, y in pts]
    if not v:
        return img
    edges = []
    x0, y0 = v[-1]
    p0x, p0y = x0 << _XY_SHIFT, y0
    for x1_, y1_ in v:
        p1x, p1y = x1_ << _XY_SHIFT, y1_
        t0x = (p0x + (_XY_ONE >> 1)) >> _XY_SHIFT
        t1x = (p1x + (_XY_ONE >> 1)) >> _XY_SHIFT
        _cv_line8(img, t0x, p0y, t1x, p1y, value)
        if p0y != p1y:
            if p0y < p1y:
                e = _PolyEdge(y0=p0y, y1=p1y, x=p0x)
            else:
                e = _PolyEdge(y0=p1y, y1=p0y, x=p1x)
            e.dx = _c_div(p1x - p0x, p1y - p0y)
            edges.append(e)
        p0x, p0y = p1x, p1y
    _cv_fill_edge_collection(img, edges, value)
    return img


class _PolyEdge:
    """``struct PolyEdge`` (drawing.cpp:50-58): y0, y1, x and dx in 16.16 fixed point, next."""
    __slots__ = ("y0", "y1", "x", "dx", "next")

    def __init__(self, y0=0, y1=0, x=0, dx=0):
        self.y0, self.y1, self.x, self.dx, self.next = y0, y1, x, dx, None


def _cv_fill_edge_collection(img: np.ndarray, edges: list, value) -> None:
    """``FillEdgeCollection`` (drawing.cpp:1327-1471), ported LITERALLY — the singly-linked
    active list, the interleaved remove/insert/draw walk, the ``[ceil, floor]`` fixed-point spans
    and the shrinking bubble sort — so equal-x and multi-edge cases order exactly as in C."""
    h, w = img.shape[:2]
    total = len(edges)
    if total < 2:
        return
    y_max, y_min = -2 ** 31, 2 ** 31 - 1
    x_max, x_min = -(2 ** 63), 2 ** 63 - 1
    for e1 in edges:
        x1 = e1.x + (e1.y1 - e1.y0) * e1.dx
        y_min, y_max = min(y_min, e1.y0), max(y_max, e1.y1)
        x_min, x_max = min(x_min, e1.x, x1), max(x_max, e1.x, x1)
    if y_max < 0 or y_min >= h or x_max < 0 or x_min >= (w << _XY_SHIFT):
        return
    edges = sorted(edges, key=lambda e: (e.y0, e.x, e.dx))            # CmpEdges
    tmp = _PolyEdge(y0=2 ** 31 - 1)
    edges.append(tmp)                                                 # the y0 = INT_MAX sentinel
    i = 0
    tmp.next = None
    e = edges[i]
    y_max = min(y_max, h)
    y = e.y0
    while y < y_max:
        sort_flag = 0
        draw = 0
        clipline = y < 0
        prelast = tmp
        last = tmp.next
        while last is not None or e.y0 == y:
            if last is not None and last.y1 == y:
                prelast.next = last.next                              # exclude a finished edge
                last = last.next
                continue
            keep_prelast = prelast
            if last is not None and (e.y0 > y or last.x < e.x):
                prelast = last                                        # advance
                last = last.next
            elif i < total:
                prelast.next = e                                      # insert e before last
                e.next = last
                prelast = e
                i += 1
                e = edges[i]
            else:
                break
            if draw:
                if not clipline:
                    if keep_prelast.x > prelast.x:
                        x1 = (prelast.x + _XY_ONE - 1) >> _XY_SHIFT
                        x2 = keep_prelast.x >> _XY_SHIFT
                    else:
                        x1 = (keep_prelast.x + _XY_ONE - 1) >> _XY_SHIFT
                        x2 = prelast.x >> _XY_SHIFT
                    if x1 < w and x2 >= 0:
                        x1 = max(x1, 0)
                        x2 = min(x2, w - 1)
                        if x2 >= x1:
                            img[y, x1:x2 + 1] = value
                keep_prelast.x += keep_prelast.dx
                prelast.x += prelast.dx
            draw ^= 1
        # sort edges (using bubble sort) — drawing.cpp:1440-1469
        keep_prelast = None
        while True:
            prelast = tmp
            last = tmp.next
            while last is not keep_prelast and last is not None and last.next is not None:
                te = last.next
                if last.x > te.x:
                    prelast.next = te
                    last.next = te.next
                    te.next = last
                    prelast = te
                    sort_flag = 1
                else:
                    prelast = last
                    last = te
            keep_prelast = prelast
            if not (sort_flag and keep_prelast is not tmp.next and keep_prelast is not tmp):
                break
        y += 1


# --------------------------------------------------------------------------- #
# 7. nuScenes metadata -> harness inputs (metadata only; no image bytes)        #
# --------------------------------------------------------------------------- #
def _quat_to_rotmat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _tf(rot_q, trans) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_to_rotmat(rot_q)
    T[:3, 3] = np.asarray(trans, dtype=np.float64)
    return T


class Meta:
    """Token index over the nuScenes JSON tables (the 13 of ``tanitad.data.nuscenes.TABLES``).

    Built from ``tanitad.data.nuscenes.load_tables`` (which REFUSES with the human acquisition
    steps when the metadata is absent) or from a dict of tables (synthetic fixtures)."""

    def __init__(self, tables: Mapping[str, list]):
        self.t = {k: list(v) for k, v in tables.items()}
        self.by = {k: {r["token"]: r for r in v if "token" in r} for k, v in self.t.items()}
        self.channel_of_cs = {}
        for cs in self.t.get("calibrated_sensor", []):
            s = self.by.get("sensor", {}).get(cs.get("sensor_token"))
            if s is not None:
                self.channel_of_cs[cs["token"]] = s["channel"]
        self.scene_by_name = {s["name"]: s for s in self.t.get("scene", [])}
        self.samples_of_scene = {}
        for sc in self.t.get("scene", []):
            chain, tok, seen = [], sc.get("first_sample_token", ""), set()
            while tok and tok in self.by["sample"] and tok not in seen:
                seen.add(tok)
                chain.append(self.by["sample"][tok])
                tok = self.by["sample"][tok].get("next", "")
            self.samples_of_scene[sc["token"]] = chain
        self.key_sd = {}                    # (sample_token, channel) -> sample_data (keyframe)
        for sd in self.t.get("sample_data", []):
            if sd.get("is_key_frame"):
                ch = self.channel_of_cs.get(sd.get("calibrated_sensor_token"))
                if ch is not None:
                    self.key_sd[(sd["sample_token"], ch)] = sd
        self.anns_of_sample = {}
        for a in self.t.get("sample_annotation", []):
            self.anns_of_sample.setdefault(a["sample_token"], []).append(a)
        self.cat_of_instance = {i["token"]: self.by["category"][i["category_token"]]["name"]
                                for i in self.t.get("instance", [])
                                if i.get("category_token") in self.by.get("category", {})}
        self.cat_index = {c["name"]: k for k, c in enumerate(self.t.get("category", []))}

    @classmethod
    def load(cls, root: str | os.PathLike, version: str = "v1.0-trainval") -> "Meta":
        _ensure_stack_on_path()
        from tanitad.data.nuscenes import load_tables
        return cls(load_tables(root, version))

    # -- poses ---------------------------------------------------------------- #
    def sensor_to_global(self, sd: dict) -> np.ndarray:
        ep = self.by["ego_pose"][sd["ego_pose_token"]]
        cs = self.by["calibrated_sensor"][sd["calibrated_sensor_token"]]
        return _tf(ep["rotation"], ep["translation"]) @ _tf(cs["rotation"], cs["translation"])

    def ego_to_global(self, sd: dict) -> np.ndarray:
        ep = self.by["ego_pose"][sd["ego_pose_token"]]
        return _tf(ep["rotation"], ep["translation"])

    def lidar_sd(self, sample: dict) -> dict:
        sd = self.key_sd.get((sample["token"], "LIDAR_TOP"))
        if sd is None:
            raise RefusedInput(f"sample {sample['token']} has no LIDAR_TOP keyframe")
        return sd

    def future_samples(self, sample: dict, n: int = N_FUTURE) -> list:
        out, s = [], sample
        for _ in range(n):
            nxt = s.get("next", "")
            if not nxt:
                break
            s = self.by["sample"][nxt]
            out.append(s)
        return out

    def category_of(self, ann: dict) -> str:
        return ann.get("category_name") or self.cat_of_instance.get(ann.get("instance_token"), "")


def gt_trajectory(meta: Meta, sample: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LIDAR_TOP origin at the next 6 keyframes in the t0 LiDAR frame (x right, y forward).

    Same quantity in all three references: ST-P3 ``get_gt_trajectory`` (sensor-from-global at
    t0 · global-from-sensor at t, NuscenesData.py:505-523), UniAD ``get_sdc_planning_label``
    (a pseudo box at the LiDAR origin, trajectory_api.py:226-270), VAD converter (:431-450).
    Returns ``(xy [6,2], valid [6] bool, yaw [6] rad in the t0 LiDAR frame)``; invalid steps
    are zeros (UniAD ``planning_all`` zeros)."""
    T0_inv = np.linalg.inv(meta.sensor_to_global(meta.lidar_sd(sample)))
    xy = np.zeros((N_FUTURE, 2))
    yaw = np.zeros(N_FUTURE)
    valid = np.zeros(N_FUTURE, dtype=bool)
    for j, s in enumerate(meta.future_samples(sample)):
        T = T0_inv @ meta.sensor_to_global(meta.lidar_sd(s))
        xy[j] = T[:2, 3]
        yaw[j] = math.atan2(T[1, 0], T[0, 0])
        valid[j] = True
    return xy, valid, yaw


def gt_command(xy: np.ndarray, valid: np.ndarray) -> str:
    """The field's GT-derived command (UniAD trajectory_api.py:272-280; VAD converter
    :452-457; ST-P3 NuscenesData.py:525-530): final lateral offset >= 2 m -> RIGHT,
    <= -2 m -> LEFT. ⛔ A LABEL for stratification only — never an input (refused by API)."""
    v = np.asarray(valid, dtype=bool)
    if not v.any():
        return "FORWARD"
    x = float(np.asarray(xy)[v][-1][0])
    return "RIGHT" if x >= 2 else ("LEFT" if x <= -2 else "FORWARD")


def expected_sample_counts(n_samples: int = 6019, n_scenes: int = 150,
                           scene_names: Iterable[str] | None = None) -> dict:
    """The three sample sets on a split, assuming every scene has >= 8 samples.

    Full val: 6,019 / 5,119 / 4,819 — BEV-Planner (2312.03031 App.: "the number of final valid
    samples is 5119") and AD-MLP (2305.10430 §3.3: "all 4819 ones") publish the last two.

    ⛔ This is ARITHMETIC — it reads no data and IGNORES ST-P3's scene blacklist. It agrees with the real
    :func:`sample_sets` on val only because no val scene is blacklisted. MEASURED 2026-09-26 on train
    (700 scenes, 28,130 samples): **16** train scenes are blacklisted and the formula's ST-P3 count is
    **22,530 against a measured 22,020** (+510). Pass ``scene_names`` and it REFUSES wherever that
    assumption fails, so the formula can never silently stand in for the measurement."""
    if scene_names is not None:
        blk = sorted(n for n in scene_names if n[-4:].isdigit() and int(n[-4:]) in STP3_SCENE_BLACKLIST)
        if blk:
            raise RefusedInput(f"expected_sample_counts is a blacklist-blind formula, but {len(blk)} of these "
                               f"scenes are on ST-P3's blacklist ({blk[:6]}{'…' if len(blk) > 6 else ''}); "
                               f"its ST-P3 count would be wrong — measure with sample_sets() instead")
    return {"uniad": n_samples, "vad": n_samples - N_FUTURE * n_scenes,
            "stp3": n_samples - (STP3_RECEPTIVE_FIELD - 1 + STP3_N_FUTURE) * n_scenes}


def sample_sets(meta: Meta, scene_names: Iterable[str]) -> list[dict]:
    """Every sample of the listed scenes in chain order, flagged per pipeline.

    UniAD scores all (masking); VAD ``fut_valid`` (converter :248-254); ST-P3 needs
    ``TIME_RECEPTIVE_FIELD-1`` = 2 previous and 6 next samples in the SAME scene
    (NuscenesData.py:124-148) and a non-blacklisted scene (:103-109)."""
    rows = []
    for name in scene_names:
        sc = meta.scene_by_name.get(name)
        if sc is None:
            raise RefusedInput(f"scene {name!r} not in the metadata")
        chain = meta.samples_of_scene[sc["token"]]
        blk = int(name[-4:]) in STP3_SCENE_BLACKLIST if name[-4:].isdigit() else False
        n = len(chain)
        for i, s in enumerate(chain):
            rows.append({"sample_token": s["token"], "scene_name": name, "index_in_scene": i,
                         "uniad": True,
                         "vad": (n - 1 - i) >= N_FUTURE,
                         "stp3": (not blk) and i >= STP3_RECEPTIVE_FIELD - 1
                                 and (n - 1 - i) >= STP3_N_FUTURE})
    return rows


def _box_corners_xy(center_xy, length, width, yaw) -> np.ndarray:
    """Footprint corners (cyclic) of a box: +-L/2 along the heading, +-W/2 across it."""
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[length / 2, width / 2], [-length / 2, width / 2],
                    [-length / 2, -width / 2], [length / 2, -width / 2]])
    R = np.array([[c, -s], [s, c]])
    return loc @ R.T + np.asarray(center_xy, dtype=np.float64)


def _ann_in_frame(ann: dict, T_frame_from_global: np.ndarray):
    """(center xy, yaw) of a global-frame annotation expressed in a sensor frame."""
    Rg = _quat_to_rotmat(ann["rotation"])
    p = T_frame_from_global @ np.array([*ann["translation"], 1.0])
    R = T_frame_from_global[:3, :3] @ Rg
    return p[:2], math.atan2(R[1, 0], R[0, 0])


def occupancy_uniad(meta: Meta, sample: dict) -> np.ndarray:
    """UniAD planning occupancy [6, 200, 200] uint8 (occflow_label.py:100-160, base_e2e.py:560-561).

    Every annotation of each future keyframe (``occ_filter_by_valid_flag`` asserted False,
    nuscenes_e2e_dataset.py:686 — so 0-point and invisible boxes COUNT), mapped name in
    UniAD's 7 vehicle classes (pedestrians EXCLUDED), reframed to the t0 LiDAR frame, corners
    ``round((c + 50) / 0.5)`` -> ``fillPoly`` at (col = x_right, row = y_forward). A missing future
    frame is filled with ``ignore_index`` 255 (occflow_label.py:111-116)."""
    occ = np.zeros((N_FUTURE, BEV, BEV), np.uint8)
    T0_inv = np.linalg.inv(meta.sensor_to_global(meta.lidar_sd(sample)))
    fut = meta.future_samples(sample)
    for j in range(N_FUTURE):
        if j >= len(fut):
            occ[j] = 255
            continue
        for a in meta.anns_of_sample.get(fut[j]["token"], []):
            name = NAME_MAPPING.get(meta.category_of(a))
            if name not in UNIAD_VEHICLE_CLASSES:
                continue
            (cx, cy), yaw = _ann_in_frame(a, T0_inv)
            w, l = float(a["size"][0]), float(a["size"][1])
            corners = _box_corners_xy((cx, cy), l, w, yaw)
            px = np.round((corners - BX + DX / 2.0) / DX).astype(np.int32)
            cv_fill_poly(occ[j], [(int(p[0]), int(p[1])) for p in px], 1)
    return occ


def occupancy_vad(meta: Meta, sample: dict) -> np.ndarray:
    """VAD planning occupancy [6, 200, 200] uint8 (metric_stp3.py:86-163 + the data path).

    Agents are the annotations at t0 with ``num_lidar_pts > 0`` (nuscenes_vad_dataset.py:1229),
    centre strictly inside ``(-15,-30,15,30)`` (CustomObjectRangeFilter, VAD_base_e2e.py:11) and a
    mapped detection class (CustomObjectNameFilter). Their futures follow the ANNOTATION ``next``
    chain (converter :367-391), expressed in the t0 LiDAR frame, with the t0 box size. Category is
    the raw ``nusc.category`` index: {14..23} -> vehicle map, {2..8} -> pedestrian map; the
    planning occupancy is their OR (VAD.py:618-620). Raster: ``lidar2cv_rot`` -> (col = 2x+100,
    row = 100-2y), ``np.round`` -> ``fillPoly``."""
    occ = np.zeros((N_FUTURE, BEV, BEV), np.uint8)
    T0_inv = np.linalg.inv(meta.sensor_to_global(meta.lidar_sd(sample)))
    start = BX
    for a in meta.anns_of_sample.get(sample["token"], []):
        if int(a.get("num_lidar_pts", 0)) <= 0:
            continue
        cat = meta.category_of(a)
        if NAME_MAPPING.get(cat) not in DET_CLASSES:
            continue
        (x0, y0), _ = _ann_in_frame(a, T0_inv)
        x_min, y_min, x_max, y_max = VAD_BEV_RANGE
        if not (x0 > x_min and y0 > y_min and x0 < x_max and y0 < y_max):
            continue
        # by NAME, not by category.json INDEX -- see vad_target_by_name (pre-registered 2026-09-26)
        target = vad_target_by_name(cat)
        if target == 0:
            continue
        w, l = float(a["size"][0]), float(a["size"][1])
        cur = a
        for j in range(N_FUTURE):
            nxt = cur.get("next", "")
            if not nxt:
                break
            cur = meta.by["sample_annotation"][nxt]
            (xa, ya), yaw_a = _ann_in_frame(cur, T0_inv)
            corners = _box_corners_xy((xa, ya), l, w, yaw_a)            # (4, 2) lidar
            cv_pts = np.stack([corners[:, 0], -corners[:, 1]], axis=1)  # lidar2cv_rot
            px = np.round((cv_pts - start + DX / 2.0) / DX).astype(np.int32)
            cv_fill_poly(occ[j], [(int(p[0]), int(p[1])) for p in px], 1)
    return occ


def vad_category_index_audit(meta: Meta) -> dict:
    """First-contact check: which category NAMES do VAD's literal index sets select here?"""
    names = [c["name"] for c in meta.t.get("category", [])]
    return {"n_categories": len(names),
            "human_index_2_8": {i: names[i] for i in sorted(VAD_HUMAN_INDEX) if i < len(names)},
            "vehicle_index_14_23": {i: names[i] for i in sorted(VAD_VEHICLE_INDEX) if i < len(names)},
            "indices_out_of_range": sorted(i for i in VAD_HUMAN_INDEX | VAD_VEHICLE_INDEX
                                           if i >= len(names)),
            "human_names_not_selected": [n for i, n in enumerate(names)
                                         if n.startswith("human.") and i not in VAD_HUMAN_INDEX],
            "vehicle_names_not_selected": [n for i, n in enumerate(names)
                                           if n.startswith("vehicle.") and i not in VAD_VEHICLE_INDEX]}


def vad_category_order_ok(audit: dict) -> tuple[bool, str]:
    """Do VAD's LITERAL index sets select VAD's INTENDED classes on this ``category.json``?

    ⛔ WHY (MEASURED 2026-09-26, first contact with real metadata). VAD selects colliding agents by
    raw ``category.json`` INDEX — ``{2..8}`` pedestrian, ``{14..23}`` vehicle — which is right only for
    the 32-entry lidarseg ordering. The base ``v1.0-trainval_meta`` ships **23** entries, and there the
    same indices pick ``animal`` and ``vehicle.car`` as "human", barriers / traffic cones / debris /
    bicycle racks as "vehicle", and MISS ``human.pedestrian.adult``/``child`` and every car, truck,
    bus, motorcycle and bicycle; index 23 does not exist. The VAD-protocol GT-collision floor then read
    **0.359 %** against PARA-Drive Table 8's published **0.96 %** — 2.7x too low, in exactly the
    direction dropping most road users predicts.

    ⭐ This function existed only as ``vad_category_index_audit``, which the benchmark NEVER CALLED —
    the eighth "built, tested, unreachable from its caller" instance. From 55aa747 it REFUSED VAD
    collision on a False; since the pre-registered name-based fix PASSED its external gate
    (GT-collision floor 1.035 / 0.987 / 0.938 % vs PARA-Drive Tab. 8's 1.02 / 0.96 / 0.91 %, all within
    3 %) selection is BY NAME (:func:`vad_target_by_name`) and this is an AUDIT, recorded on every
    VAD run: it says whether VAD's own index-based code would have been right on this metadata.

    True iff: no index is out of range, the pedestrian set selects ONLY ``human.*`` names and ALL of
    them, and the vehicle set selects ONLY ``vehicle.*`` names and ALL of them.
    """
    bad = []
    if audit["indices_out_of_range"]:
        bad.append(f"indices {audit['indices_out_of_range']} out of range for "
                   f"{audit['n_categories']} categories")
    wrong_h = {i: n for i, n in audit["human_index_2_8"].items() if not n.startswith("human.")}
    wrong_v = {i: n for i, n in audit["vehicle_index_14_23"].items() if not n.startswith("vehicle.")}
    if wrong_h:
        bad.append(f"pedestrian indices select non-human {wrong_h}")
    if wrong_v:
        bad.append(f"vehicle indices select non-vehicle {wrong_v}")
    if audit["human_names_not_selected"]:
        bad.append(f"pedestrians MISSED {audit['human_names_not_selected']}")
    if audit["vehicle_names_not_selected"]:
        bad.append(f"vehicles MISSED {audit['vehicle_names_not_selected']}")
    if not bad:
        return True, ""
    return False, ("VAD's INDEX-based selection would be WRONG here: its literal category indices ({2..8} pedestrian, "
                   "{14..23} vehicle) assume the 32-entry lidarseg category.json ordering, but this "
                   f"one has {audit['n_categories']} entries — " + "; ".join(bad)
                   + ". A grid built from them would hold the wrong objects; the harness selects by "
                   "NAME instead (vad_target_by_name), so this is recorded, not refused.")


# --------------------------------------------------------------------------- #
# 8. Input adapter — nuScenes CAM_FRONT -> our cylindrical training frame        #
# --------------------------------------------------------------------------- #
def _ensure_stack_on_path() -> None:
    here = Path(__file__).resolve()
    stack = here.parents[2] / "stack"
    if stack.is_dir() and str(stack) not in sys.path:
        sys.path.insert(0, str(stack))


def model_frame(name: str = "wide"):
    """The training frames: ``wide`` = PHYSICALAI_WIDE120_256x640 (the frame E2 fed refcv4b),
    ``rig_clean`` = PHYSICALAI_RIG_CLEAN_176x624 (a pure slice [40:216, 8:632] of it)."""
    _ensure_stack_on_path()
    from tanitad.data import calib
    frames = {"wide": calib.PHYSICALAI_WIDE120_256x640,
              "rig_clean": calib.PHYSICALAI_RIG_CLEAN_176x624}
    if name not in frames:
        raise RefusedInput(f"unknown model frame {name!r}; choose from {sorted(frames)}")
    return frames[name]


def camera_intrinsics(meta: Meta, sd: dict):
    """PER-SAMPLE CAM_FRONT intrinsics (never the nominal constant); dist = 0 because nuScenes
    images are undistorted and rectified (docs/schema_nuscenes.md: calibrated_sensor)."""
    _ensure_stack_on_path()
    from tanitad.data.calib import PinholeIntrinsics
    k = meta.by["calibrated_sensor"][sd["calibrated_sensor_token"]].get("camera_intrinsic") or []
    if not k:
        raise RefusedInput(f"sample_data {sd.get('token')} has no camera_intrinsic")
    return PinholeIntrinsics(fx=float(k[0][0]), fy=float(k[1][1]), cx=float(k[0][2]),
                             cy=float(k[1][2]), width=int(sd.get("width", 1600)),
                             height=int(sd.get("height", 900)))


def geometry_stamp(intr, frame) -> dict:
    """The observed-mask stamp every model arm must carry: how much of OUR frame the nuScenes
    camera actually fills (a ~64.6 deg pinhole inside a 120 deg cylinder)."""
    _ensure_stack_on_path()
    from tanitad.data.calib import pinhole_rectify_grid
    _, mask = pinhole_rectify_grid(intr, int(intr.height), int(intr.width), frame=frame)
    m = mask.numpy().astype(bool)
    cols = np.flatnonzero(m.any(axis=0))
    rows_c = np.flatnonzero(m[:, (frame.width - 1) // 2])
    return {"frame": frame.tag(), "frame_hw": [frame.height, frame.width],
            "frame_f_ref": float(frame.f_ref), "projection": frame.projection,
            "frame_hfov_deg": round(frame.hfov_deg, 4),
            "camera_hfov_deg": round(intr.hfov_deg, 4), "camera_vfov_deg": round(intr.vfov_deg, 4),
            "camera_intrinsics": {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                                  "width": intr.width, "height": intr.height},
            "observed_px": int(m.sum()), "total_px": int(m.size),
            "observed_frac": float(m.mean()),
            "observed_cols": [int(cols[0]), int(cols[-1])] if cols.size else None,
            "observed_rows_centre_col": [int(rows_c[0]), int(rows_c[-1])] if rows_c.size else None,
            "method": "tanitad.data.calib.pinhole_rectify (ray resample, zeros outside, never a resize)"}


def rectify_cam_front(images_u8, intr, frame):
    """[T,3,H,W] uint8 CAM_FRONT -> [T,3,frame.h,frame.w] uint8 + the stamp."""
    _ensure_stack_on_path()
    import torch
    from tanitad.data.calib import pinhole_rectify
    vid = images_u8 if isinstance(images_u8, torch.Tensor) else torch.from_numpy(np.asarray(images_u8))
    out = pinhole_rectify(vid, intr, frame=frame)
    stamp = geometry_stamp(intr, frame)
    stamp["observed_frac_of_this_call"] = float(pinhole_rectify.last_observed_frac)
    return out, stamp


HISTORY_CONSTRUCTIONS = ("EXACT", "NT", "ST")


def history_slots(t0_us: int, frames: Sequence[tuple[int, str]], n_raw: int,
                  construction: str, dt_s: float = 0.1, tol_s: float = 0.05) -> dict:
    """Which CAM_FRONT image fills each 0.1 s slot of the model's raw history (oldest first).

    ``frames`` = [(timestamp_us, filename)] of CAM_FRONT sample_data at or before t0 (keyframes
    AND sweeps if extracted). EXACT: nearest frame within ``tol_s`` or REFUSED — needs the 12 Hz
    sweeps; NT: nearest in time whatever the gap (E2's declared sensitivity arm); ST: every slot
    <- the t0 frame (E2's static history). The data tier each needs is the PI's download choice."""
    if construction not in HISTORY_CONSTRUCTIONS:
        raise RefusedInput(f"construction must be one of {HISTORY_CONSTRUCTIONS}")
    fr = sorted((int(t), f) for t, f in frames if int(t) <= t0_us)
    if not fr or fr[-1][0] != t0_us:
        raise RefusedInput("the t0 frame itself is missing from `frames`")
    slots = [t0_us - int(round((n_raw - 1 - k) * dt_s * 1e6)) for k in range(n_raw)]
    chosen, errs = [], []
    for s in slots:
        if construction == "ST":
            chosen.append(fr[-1][1])
            errs.append((t0_us - s) / 1e6)
            continue
        best = min(fr, key=lambda tf: (abs(tf[0] - s), -tf[0]))
        e = abs(best[0] - s) / 1e6
        if construction == "EXACT" and e > tol_s:
            raise RefusedInput(f"EXACT history needs a CAM_FRONT frame within {tol_s}s of slot "
                               f"t0-{(t0_us - s) / 1e6:.1f}s; nearest is {e:.3f}s away — the "
                               f"camera SWEEPS (v1.0-trainval*_blobs_camera.tgz) are required")
        chosen.append(best[1])
        errs.append(e)
    return {"construction": construction, "slots_s": [(s - t0_us) / 1e6 for s in slots],
            "files": chosen, "abs_time_error_s": errs, "max_abs_time_error_s": max(errs)}


#: E2's model output knots (tanitad_navsim_bridge.py:40): horizons (5..60) @ 10 Hz.
KNOT_T_S = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
HEADING_HOLD_SPEED_MS = 1.0          # E2 SPEC §9 amendment: hold the heading below 1 m/s


def plan_to_lidar(knots_xy_ego, l2e_translation, l2e_rotation, knot_t=KNOT_T_S,
                  out_t=HORIZONS_S) -> tuple[np.ndarray, np.ndarray]:
    """Our plan (ego frame at t0: x forward, y left; knots at ``knot_t``) -> the LiDAR-origin
    trajectory the references score, in the t0 LiDAR frame.

    Method: a C2 'not-a-knot' cubic spline in time through the origin and every knot (E2's
    declared method), evaluated at ``out_t``; heading = tangent angle, held below
    ``HEADING_HOLD_SPEED_MS``; LiDAR origin = ego point + R(psi) * lever arm; then into the
    t0 LiDAR axes. ⚠️ ASSUMPTION (UNVERIFIED): the model's ego point is the nuScenes ego-frame
    origin (rear axle); physicalai.py does not state PhysicalAI's egomotion origin."""
    from scipy.interpolate import CubicSpline
    k = np.asarray(knots_xy_ego, dtype=np.float64)
    tt = np.concatenate([[0.0], np.asarray(knot_t, dtype=np.float64)])
    if k.shape != (len(knot_t), 2):
        raise RefusedInput(f"knots must be [{len(knot_t)}, 2], got {k.shape}")
    pts = np.vstack([[0.0, 0.0], k])
    sx = CubicSpline(tt, pts[:, 0], bc_type="not-a-knot")
    sy = CubicSpline(tt, pts[:, 1], bc_type="not-a-knot")
    t = np.asarray(out_t, dtype=np.float64)
    pe = np.stack([sx(t), sy(t)], axis=1)
    vx, vy = sx(t, 1), sy(t, 1)
    psi = np.zeros(len(t))
    last = 0.0
    for i in range(len(t)):
        if math.hypot(vx[i], vy[i]) >= HEADING_HOLD_SPEED_MS:
            last = math.atan2(vy[i], vx[i])
        psi[i] = last
    return ego_path_to_lidar(pe, psi, l2e_translation, l2e_rotation), psi


def ego_path_to_lidar(pe_xy, psi, l2e_translation, l2e_rotation) -> np.ndarray:
    """Ego-frame points + headings at t0 -> LiDAR-origin positions in the t0 LiDAR frame."""
    lt = np.asarray(l2e_translation, dtype=np.float64)
    R = _quat_to_rotmat(l2e_rotation)
    out = np.zeros((len(pe_xy), 2))
    for i, (p, h) in enumerate(zip(np.asarray(pe_xy, dtype=np.float64), psi)):
        c, s = math.cos(h), math.sin(h)
        arm = np.array([c * lt[0] - s * lt[1], s * lt[0] + c * lt[1]])
        d = np.array([p[0] + arm[0] - lt[0], p[1] + arm[1] - lt[1], 0.0])
        out[i] = (R.T @ d)[:2]
    return out


# --------------------------------------------------------------------------- #
# 9. Arms: the GT control (reference) and the trivial floors                    #
# --------------------------------------------------------------------------- #
def ego_speed_backward(meta: Meta, lidar_sd: dict) -> tuple[float, str]:
    """v0 at t0 from the ego poses of the previous LIDAR_TOP sample_data (~20 Hz sweeps, all
    in the metadata) — causal: no pose after t0 is read. Returns (v0 m/s, source)."""
    prev = lidar_sd.get("prev", "")
    if not prev:
        return 0.0, "no previous LIDAR_TOP pose (scene start) -> v0 := 0"
    p = meta.by["sample_data"].get(prev)
    if p is None:
        return 0.0, "previous LIDAR_TOP sample_data not in the metadata -> v0 := 0"
    e0 = meta.by["ego_pose"][lidar_sd["ego_pose_token"]]
    e1 = meta.by["ego_pose"][p["ego_pose_token"]]
    dt = (int(lidar_sd["timestamp"]) - int(p["timestamp"])) * 1e-6
    if dt <= 0:
        return 0.0, "non-positive dt -> v0 := 0"
    d = np.asarray(e0["translation"][:2]) - np.asarray(e1["translation"][:2])
    return float(np.hypot(*d) / dt), "ego_pose backward difference (LIDAR_TOP sweep)"


def arm_trajectory(meta: Meta, sample: dict, arm: str, gt_xy: np.ndarray) -> tuple[np.ndarray, dict]:
    """GT (reference: the logged human trajectory), STOP (floor), CV (floor: straight at v0)."""
    if arm == "GT":
        return gt_xy.copy(), {}
    if arm == "STOP":
        return np.zeros((N_FUTURE, 2)), {}
    if arm == "CV":
        sd = meta.lidar_sd(sample)
        v0, src = ego_speed_backward(meta, sd)
        cs = meta.by["calibrated_sensor"][sd["calibrated_sensor_token"]]
        fwd = (_quat_to_rotmat(cs["rotation"]).T @ np.array([1.0, 0.0, 0.0]))[:2]
        fwd = fwd / (np.linalg.norm(fwd) or 1.0)
        t = np.asarray(HORIZONS_S)[:, None]
        return v0 * t * fwd[None, :], {"v0_mps": v0, "v0_source": src}
    raise RefusedInput(f"unknown built-in arm {arm!r}")


ARM_SPECS = {
    "GT": {"kind": "reference", "declared_inputs": ["logged_future_lidar_origin (REFERENCE - the "
                                                    "instrument's own floor, not a plan)"]},
    "STOP": {"kind": "floor", "declared_inputs": []},
    "CV": {"kind": "floor", "declared_inputs": ["ego_pose(t<=t0) -> v0, heading (causal)"]},
}


# --------------------------------------------------------------------------- #
# 10. Four-family SUPPLEMENTS (TanitEval-side; NOT part of the published protocol)#
# --------------------------------------------------------------------------- #
def family_supplements(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, scored: np.ndarray,
                       commands_gt: Sequence[str]) -> dict:
    """Per-family status + n + reason, for W1's summary schema. Computed on the SAME scored
    samples and at-t horizons (1/2/3 s = steps 1/3/5), never pooled into one score."""
    sc = np.asarray(scored, dtype=bool)
    ok = sc & np.asarray(valid, dtype=bool).all(axis=1)
    n = int(ok.sum())
    base = {"not_part_of_published_protocol": True}
    if n == 0:
        reason = "no scored sample has a complete 6-step GT"
        return {f: {"status": "UNAVAILABLE", "reason": reason, "n": 0, **base}
                for f in ("longitudinal", "lateral", "tactical", "strategic")}
    P, G = np.asarray(pred)[ok], np.asarray(gt)[ok]
    Gz = np.concatenate([np.zeros((n, 1, 2)), G], axis=1)
    Pz = np.concatenate([np.zeros((n, 1, 2)), P], axis=1)
    tang = np.zeros_like(G)
    for k in range(N_FUTURE):
        a, b = Gz[:, k], Gz[:, min(k + 2, N_FUTURE)]
        tang[:, k] = b - a
    norm = np.linalg.norm(tang, axis=-1, keepdims=True)
    fallback = np.array([0.0, 1.0])                     # LiDAR forward when the GT is ~static
    tang = np.where(norm > 1e-6, tang / np.maximum(norm, 1e-12), fallback)
    nrm = np.stack([tang[..., 1], -tang[..., 0]], axis=-1)
    d = P - G
    along = (d * tang).sum(-1)
    cross = (d * nrm).sum(-1)
    sp_p = np.linalg.norm(np.diff(Pz, axis=1), axis=-1) / 0.5
    sp_g = np.linalg.norm(np.diff(Gz, axis=1), axis=-1) / 0.5
    hp = np.arctan2(np.diff(Pz, axis=1)[..., 0], np.diff(Pz, axis=1)[..., 1])
    hg = np.arctan2(np.diff(Gz, axis=1)[..., 0], np.diff(Gz, axis=1)[..., 1])
    dh = np.degrees(np.abs((hp - hg + np.pi) % (2 * np.pi) - np.pi))
    moving = sp_g > 1.0
    idx = {"1s": 1, "2s": 3, "3s": 5}
    lon = {"along_track_abs_m": {h: float(np.abs(along[:, i]).mean()) for h, i in idx.items()},
           "along_track_signed_m": {h: float(along[:, i].mean()) for h, i in idx.items()},
           "speed_abs_err_mps": {h: float(np.abs(sp_p[:, i] - sp_g[:, i]).mean()) for h, i in idx.items()}}
    lat = {"cross_track_abs_m": {h: float(np.abs(cross[:, i]).mean()) for h, i in idx.items()},
           "heading_abs_err_deg_moving": {h: (float(dh[:, i][moving[:, i]].mean())
                                              if moving[:, i].any() else None) for h, i in idx.items()},
           "n_moving": {h: int(moving[:, i].sum()) for h, i in idx.items()}}
    cmd_g = [c for c, o in zip(commands_gt, ok) if o]
    cmd_p = [gt_command(p, np.ones(N_FUTURE, bool)) for p in P]
    classes = ("LEFT", "FORWARD", "RIGHT")
    conf = [[sum(1 for a, b in zip(cmd_g, cmd_p) if a == r and b == c) for c in classes] for r in classes]
    agree = sum(1 for a, b in zip(cmd_g, cmd_p) if a == b) / n
    return {
        "longitudinal": {"status": "PARTIAL", "n": n, "metrics": lon, **base,
                         "reason": "along-track + speed error only; distance-keeping (headway/TTC "
                                   "to the lead agent) is not computed by this harness"},
        "lateral": {"status": "PARTIAL", "n": n, "metrics": lat, **base,
                    "reason": "cross-track + heading error only; curvature and yaw-rate error at "
                              "0.5 s waypoint spacing are not computed"},
        "tactical": {"status": "PARTIAL", "n": n, **base,
                     "metrics": {"manoeuvre_agreement": agree, "classes": list(classes),
                                 "confusion_gt_rows_pred_cols": conf},
                     "reason": "manoeuvre decision = the +-2 m final-offset class of the plan vs "
                               "the GT class (a LABEL, never an input); no goal-setting target"},
        "strategic": {"status": "UNAVAILABLE", "n": n, **base,
                      "reason": "the nuScenes open-loop protocol has no route/goal target; a route "
                                "could come from the map expansion + CAN 'route' - not implemented"},
    }


# --------------------------------------------------------------------------- #
# 11. The splits (devkit splits.py @b40adc4) and the run directory              #
# --------------------------------------------------------------------------- #
#: nutonomy/nuscenes-devkit@b40adc4 python-sdk/nuscenes/utils/splits.py:104-123 (val),
#: :146-150 (mini). The test re-parses the byte-identical vendored copy
#: (W6 package raw/devkit_splits_b40adc4.py.txt, git blob 6988b5c8) and requires equality.
DEVKIT_VAL_SCENES = (
    'scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
    'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
    'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
    'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
    'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
    'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
    'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
    'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
    'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
    'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
    'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
    'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
    'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
    'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
    'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
    'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
    'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
    'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
    'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073',
)
DEVKIT_MINI_VAL_SCENES = ('scene-0103', 'scene-0916')
DEVKIT_MINI_TRAIN_SCENES = (
    'scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100',
)
_SPLITS = {"val": DEVKIT_VAL_SCENES, "mini_val": DEVKIT_MINI_VAL_SCENES,
           "mini_train": DEVKIT_MINI_TRAIN_SCENES}


def devkit_split(name: str) -> list[str]:
    """Scene names of a devkit split (val / mini_val / mini_train — the planning splits)."""
    if name not in _SPLITS:
        raise RefusedInput(f"unknown split {name!r}; planning splits: {sorted(_SPLITS)}")
    return list(_SPLITS[name])


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30)
        h = out.stdout.strip()
        return h if len(h) == 40 else "UNAVAILABLE"
    except Exception:                                   # noqa: BLE001
        return "UNAVAILABLE"


def evaluate_arms(meta: Meta, scene_names: Sequence[str], protocol: str,
                  arms: Sequence[str], model_arms: Mapping[str, dict] | None = None,
                  limit: int | None = None) -> dict:
    """Score built-in arms (GT/STOP/CV) and optional model arms under ONE protocol.

    ``model_arms[name] = {"plan_fn": fn(meta, sample) -> xy[6,2] LiDAR frame, "declared_inputs":
    [...], "input_geometry": {...stamp...}, "command_source": "none"|"predicted_goal"}``."""
    tag = protocol_tag(protocol)
    rows = sample_sets(meta, scene_names)
    if limit:
        rows = rows[:limit]
    key = {Pipeline.UNIAD: "uniad", Pipeline.VAD: "vad", Pipeline.STP3: "stp3"}[tag.pipeline]
    n = len(rows)
    gt = np.zeros((n, N_FUTURE, 2))
    valid = np.zeros((n, N_FUTURE), bool)
    occ = np.zeros((n, N_FUTURE, BEV, BEV), np.uint8) if tag.pipeline is not Pipeline.STP3 else None
    # ⛔ The VAD pipeline selects colliding agents by raw category INDEX. Decide BEFORE the grid is
    # built: if those indices do not pick VAD's intended classes here, refuse collision (L2 stands)
    # rather than compute a plausible collision rate over the wrong objects.
    vad_audit, coll_refusal = None, ""
    if tag.pipeline is Pipeline.VAD:
        raw_audit = vad_category_index_audit(meta)
        order_ok, order_reason = vad_category_order_ok(raw_audit)
        # ⭐ Selection is BY NAME (vad_target_by_name), so category.json's ORDER no longer decides which
        # objects are in the grid. The order audit is still RECORDED -- it says whether VAD's own
        # index-based code would have been right on this metadata -- but it no longer refuses.
        names = [c["name"] for c in meta.t.get("category", [])]
        sel = {n: vad_target_by_name(n) for n in names if NAME_MAPPING.get(n) in DET_CLASSES}
        peds = sorted(n for n, t in sel.items() if t == 2)
        vehs = sorted(n for n, t in sel.items() if t == 1)
        if not peds or not vehs:                   # a degenerate category.json: refuse, never guess
            coll_refusal = (f"VAD collision REFUSED: the name rule selects {len(peds)} pedestrian and "
                            f"{len(vehs)} vehicle categories on this category.json "
                            f"({len(names)} entries) -- a grid missing a whole map is not VAD's metric")
        vad_audit = {"selection": "by_name (vad_target_by_name; pre-registered 2026-09-26)",
                     "selected_pedestrians": peds, "selected_vehicles": vehs,
                     "refusal_reason": coll_refusal or None,
                     "index_order_ok": order_ok,
                     "index_order_note": (order_reason or "VAD's literal index sets would have selected the "
                                                          "intended classes on this category.json"),
                     "n_categories": raw_audit["n_categories"],
                     "indices_out_of_range": raw_audit["indices_out_of_range"]}
        if coll_refusal:
            occ = None
    scored = np.array([r[key] for r in rows], bool)
    cmds = []
    samples = [meta.by["sample"][r["sample_token"]] for r in rows]
    for i, s in enumerate(samples):
        gt[i], valid[i], _ = gt_trajectory(meta, s)
        cmds.append(gt_command(gt[i], valid[i]))
        if occ is not None:
            occ[i] = occupancy_uniad(meta, s) if tag.pipeline is Pipeline.UNIAD else occupancy_vad(meta, s)
    out = {"tag": tag, "rows": rows, "gt": gt, "valid": valid, "scored": scored,
           "commands_gt": cmds, "arms": {}, "vad_category_audit": vad_audit}
    all_arms = list(arms) + list((model_arms or {}).keys())
    for arm in all_arms:
        extra = {}
        if arm in (model_arms or {}):
            spec = model_arms[arm]
            pred = np.stack([np.asarray(spec["plan_fn"](meta, s), dtype=np.float64) for s in samples])
            kind, declared = "model", list(spec.get("declared_inputs", []))
            geom = spec.get("input_geometry") or (spec["geometry_fn"]() if "geometry_fn" in spec else None)
            cmd_src = spec.get("command_source", "none")
        else:
            if arm not in ARM_SPECS:
                raise RefusedInput(f"unknown arm {arm!r}; built-ins {sorted(ARM_SPECS)}")
            preds = []
            for i, s in enumerate(samples):
                p, info = arm_trajectory(meta, s, arm, gt[i])
                preds.append(p)
                if info:
                    extra.setdefault("per_sample", []).append(info)
            pred = np.stack(preds)
            kind, declared = ARM_SPECS[arm]["kind"], ARM_SPECS[arm]["declared_inputs"]
            geom, cmd_src = None, "none"
        if tag.pipeline is Pipeline.UNIAD:
            k = kernel_uniad(pred, gt, valid, occ)
        elif tag.pipeline is Pipeline.VAD:
            k = kernel_vad(pred, gt, occ, scored, unavailable_reason=coll_refusal)
        else:
            k = kernel_stp3(pred, gt, None, scored)
        res = score(k, tag)
        out["arms"][arm] = {"kind": kind, "declared_inputs": declared, "pred": pred, "kernel": k,
                            "result": res, "input_geometry": geom, "command_source": cmd_src,
                            "families": family_supplements(pred, gt, valid, scored, cmds),
                            "extra": extra}
    return out


def _nonstraight_block(ev: dict, arm: str) -> dict:
    """PARA-Drive's targeted subset rule (command != keep forward; 686 frames on val) applied to
    the scored samples, reported with its n beside full val (registry nusc.plan.nonstraight)."""
    sc = ev["scored"]
    ns = np.array([c != "FORWARD" for c in ev["commands_gt"]]) & sc
    n = int(ns.sum())
    if n == 0:
        return {"status": "UNAVAILABLE", "reason": "no scored non-straight sample", "n": 0}
    k = ev["arms"][arm]["kernel"]
    sub = KernelOutput(k.pipeline, k.l2, k.obj_col, k.obj_box_col, k.gt_box_col, k.valid, ns,
                       k.collision_available, k.collision_unavailable_reason)
    res = score(sub, ev["tag"])
    return {"status": "OK", "n": n, "rule": "GT-derived command != FORWARD (+-2 m final offset)",
            "paradrive_val_count_for_reference": 686,
            "L2_m_avg_1_2_3s": res.get("L2_m", "avg_1_2_3s").value,
            "collision_box_pct_avg_1_2_3s": (res.get("collision_box_pct", "avg_1_2_3s").value
                                             if res.collision_available else None)}


def _gt_control_block(ev: dict) -> dict:
    k = ev["arms"][next(iter(ev["arms"]))]["kernel"]
    if not k.collision_available:
        return {"status": "UNAVAILABLE", "reason": k.collision_unavailable_reason,
                "n": int(ev["scored"].sum())}
    m = per_timestep_means(k)
    red = _REDUCERS[ev["tag"].reduction](m["gt_collision_box_pct"])
    red_valid = _REDUCERS[ev["tag"].reduction](m["gt_collision_box_pct_valid_steps_only"])
    return {"status": "OK", "n": int(m["n"]), "protocol_tag": str(ev["tag"]),
            "gt_collision_box_pct": red["headline"], "gt_collision_box_pct_columns": red["columns"],
            "gt_collision_denominator": "all scored samples, masked steps contribute 0 (reference compute())",
            "gt_collision_box_pct_valid_steps_only": red_valid["headline"],
            "note": "RAW box collision of the GT human trajectory under this protocol's own grid "
                    "(PARA-Drive fn. 4). The references EXCLUDE GT-colliding steps from every "
                    "arm's numerator, so in-protocol the GT arm scores 0 by construction; this "
                    "raw rate is the false-positive floor. PARA-Drive Table 8: UniAD protocol "
                    "0.36 %, VAD protocol 0.96 %. Quote no arm's collision within ~2x of it."}


SCORE_CSV_HEADER = ["sample_token", "scene_name", "scored", "step", "t_s", "pred_x", "pred_y",
                    "gt_x", "gt_y", "gt_valid", "l2_m", "obj_col", "obj_box_col", "gt_box_col"]


def _pipe_key(p: Pipeline) -> str:
    return {Pipeline.UNIAD: "uniad", Pipeline.VAD: "vad", Pipeline.STP3: "stp3"}[p]


def build_run_outputs(ev: dict, *, split_name: str, n_scenes: int, n_logs: int | None) -> dict:
    """Everything a §1 run directory holds for ONE protocol, as data — shared by the standalone
    :func:`write_run` and W1's plugin (``taniteval/taniteval/bench/plugins/nuscenes_ol.py``)."""
    tag = ev["tag"]
    utc = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gt_ctrl = _gt_control_block(ev)
    split = {"name": split_name, "n_scenes": n_scenes, "n_logs": n_logs,
             "n_samples_total": len(ev["rows"]), "n_scored": int(ev["scored"].sum()),
             "sample_rule": {Pipeline.UNIAD: "all samples; missing future steps masked (UniAD)",
                             Pipeline.VAD: "fut_valid_flag: 6 future samples (VAD)",
                             Pipeline.STP3: "2 past + 6 future in scene (ST-P3)"}[tag.pipeline]}
    floors = [a for a, v in ev["arms"].items() if v["kind"] == "floor"]
    arms, per_sample = {}, {}
    for arm, a in ev["arms"].items():
        k, res = a["kernel"], a["result"]
        rows = [SCORE_CSV_HEADER]
        for i, r in enumerate(ev["rows"]):
            for t in range(N_FUTURE):
                rows.append([r["sample_token"], r["scene_name"], int(ev["scored"][i]), t,
                             HORIZONS_S[t], f"{a['pred'][i, t, 0]:.6f}", f"{a['pred'][i, t, 1]:.6f}",
                             f"{ev['gt'][i, t, 0]:.6f}", f"{ev['gt'][i, t, 1]:.6f}",
                             int(k.valid[i, t]), f"{k.l2[i, t]:.6f}", int(k.obj_col[i, t]),
                             int(k.obj_box_col[i, t]), int(k.gt_box_col[i, t])])
        art = build_artifact(res, arm=arm, arm_kind=a["kind"], declared_inputs=a["declared_inputs"],
                             command_source=a["command_source"], split=split, gt_control=gt_ctrl,
                             nonstraight=_nonstraight_block(ev, arm),
                             input_geometry=a["input_geometry"], families=a["families"],
                             provenance={"harness": "taniteval/adapters/nuscenes_planning.py",
                                         "pins": PINS, "utc": utc, **(a.get("extra") or {})})
        sc = k.scored
        per_sample[arm] = np.array([
            (_REDUCERS[tag.reduction](k.l2[i])["headline"]["avg_1_2_3s"] if sc[i] else np.nan)
            for i in range(len(sc))])
        head = res.get("L2_m", "avg_1_2_3s")
        sub = {m: {h: res.get(m, h).value for h in HEADLINE_HORIZONS}
               for m in ("L2_m", "collision_box_pct", "collision_point_pct")
               if res.collision_available or m == "L2_m"}
        summ = {
            "kind": a["kind"], "status": "OK", "declared_inputs": list(a["declared_inputs"]),
            "headline": {"value": head.value, "column": "L2_m", "n": head.n_samples,
                         "statistic": (f"mean L2 over the 1/2/3 s columns, {tag.reduction.value} "
                                       f"reduction, {tag.pipeline.value} pipeline")},
            "submetrics": sub,
            "interval": {"status": "UNAVAILABLE", "reason": "no pre-registered cluster unit for "
                         "nuScenes (scene vs log); the published rows carry none", "n": head.n_samples},
            "families": {f: {kk: vv for kk, vv in v.items() if kk in ("status", "n", "reason")}
                         for f, v in a["families"].items()},
            "files": {"scores": f"scores/{arm}.csv", "artifact": f"artifacts/{arm}.json",
                      "criteria": f"criteria/{arm}.txt"},
            "statistics": {"four_family_supplements": a["families"],
                           "protocol_rows": res.to_dict()["rows"]},
            "paired": {},
        }
        if a["input_geometry"]:
            summ["modality"] = {"input_geometry": a["input_geometry"]}
        arms[arm] = {"csv_rows": rows, "artifact": art, "summary_arm": summ,
                     "meta": {"name": arm, "kind": a["kind"],
                              "declared_inputs": list(a["declared_inputs"]), "status": "OK"}}
    for arm in arms:
        for fl in floors:
            if fl == arm:
                arms[arm]["summary_arm"]["paired"][fl] = {"status": "SELF"}
                continue
            x, y = per_sample[arm], per_sample[fl]
            ok = ~np.isnan(x) & ~np.isnan(y)
            arms[arm]["summary_arm"]["paired"][fl] = {
                "status": "OK", "headline_delta": float(np.mean(x[ok] - y[ok])) if ok.any() else None,
                "n_common": int(ok.sum()),
                "note": "mean paired difference of per-sample avg-L2 (arm - floor); no interval "
                        "(UNAVAILABLE: cluster unit not pre-registered)"}
    pin = PINS[_pipe_key(tag.pipeline)]
    summary = {
        "schema": "taniteval.bench.summary/1", "run_id": None, "benchmark": "nuscenes_ol",
        "protocol": tag.protocol, "split": split_name, "claim_bearing": CLAIM_BEARING,
        "evidence_class": "MEASURED",
        "stamps": {"tier": "OPEN-LOOP-L2", "loop": {"planner": "OPEN"}, "closed_loop": False},
        "headline_metric": {"name": f"L2 avg 1/2/3 s ({tag.published_family})", "column": "L2_m",
                            "higher_is_better": False,
                            "statistic": f"{tag.reduction.value} reduction on {tag.pipeline.value}"},
        "floors": floors, "arms": {k: v["summary_arm"] for k, v in arms.items()},
        "controls": {"gt_control": gt_ctrl,
                     # recorded on EVERY VAD-pipeline run, pass or refuse — an audit that only
                     # speaks when it fails is indistinguishable from one that never ran
                     **({"vad_category_audit": ev["vad_category_audit"]}
                        if ev.get("vad_category_audit") is not None else {})},
        "provenance": {"protocol_tag": tag.to_dict(), "pins": PINS,
                       "claim_bearing_reason": CLAIM_BEARING_REASON, "split": split},
    }
    return {
        "tag": tag, "split": split, "gt_control": gt_ctrl, "floors": floors, "arms": arms,
        "summary": summary,
        "devkit": {"repo": f"{pin['repo']} (reference implementation; nuScenes has NO official "
                           f"planning devkit)", "sha": pin["commit"],
                   "patches": [{"name": "taniteval/adapters/nuscenes_planning.py: verbatim kernel + "
                                        "reduction port, cv2.fillPoly port (OpenCV 4.5.4), float64 "
                                        "accumulation", "kind": "harness"}]},
        "stamps": {"tier": "OPEN-LOOP-L2", "loop": {"planner": "OPEN"}, "evidence_class": "MEASURED",
                   "closed_loop": False},
    }


def _write_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(rows)


def write_run(ev: dict, run_dir: str | os.PathLike, *, split_name: str, n_scenes: int,
              n_logs: int | None, ckpt: Mapping | None = None, device: str = "cpu",
              wall_s: float = 0.0, command: Sequence[str] | None = None,
              repo_root: str | os.PathLike | None = None) -> dict:
    """STANDALONE writer of the BUILD_PLAN §1 run directory for ONE protocol (the W1 suite writes
    the same files through its plugin; both validate against bench/schema/*.schema.json)."""
    run_dir = Path(run_dir)
    for sub in ("scores", "artifacts", "criteria"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    repo = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    out = build_run_outputs(ev, split_name=split_name, n_scenes=n_scenes, n_logs=n_logs)
    utc = _dt.datetime.now(_dt.timezone.utc)
    ck = dict(ckpt or {"path": None, "sha256": None, "registry_key": None})
    tagname = "none" if not ck.get("path") else Path(str(ck["path"])).stem.lower()
    tagname = "".join(ch if (ch.isalnum() or ch in "_.") else "_" for ch in tagname) or "none"
    run_id = (f"{utc.strftime('%Y%m%dT%H%M%SZ')}-nuscenes_ol-{tagname}-"
              f"{hashlib.sha1(str(run_dir).encode()).hexdigest()[:6]}")
    for arm, a in out["arms"].items():
        _write_csv(run_dir / "scores" / f"{arm}.csv", a["csv_rows"])
        ap = run_dir / "artifacts" / f"{arm}.json"
        ap.write_text(json.dumps(a["artifact"], indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        crit = run_dir / "criteria" / f"{arm}.txt"
        try:
            cp = subprocess.run([sys.executable, str(repo / "tools" / "criteria_check.py"), str(ap)],
                                capture_output=True, text=True, timeout=120, encoding="utf-8",
                                errors="replace", cwd=str(repo))
            crit.write_text(f"rc={cp.returncode}\n{cp.stdout}\n[stderr]\n{cp.stderr}", encoding="utf-8")
        except Exception as e:                           # noqa: BLE001
            crit.write_text(f"criteria_check could not run: {type(e).__name__}: {e}\n", encoding="utf-8")
    summary = dict(out["summary"], run_id=run_id)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    bench_run = {
        "schema": "taniteval.bench.bench_run/1", "run_id": run_id,
        "utc": utc.strftime("%Y-%m-%dT%H:%M:%SZ"), "utc_end": None,
        "git_head": _git_head(repo), "command": list(command or []),
        "ckpt": {"path": ck.get("path"), "sha256": ck.get("sha256"), "registry_key": ck.get("registry_key")},
        "benchmark": "nuscenes_ol", "protocol": out["tag"].protocol, "devkit": out["devkit"],
        "split": {"name": split_name, "n_scenes": n_scenes, "n_logs": n_logs},
        "arms": [a["meta"] for a in out["arms"].values()],
        "device": {"requested": device if device in ("auto", "cpu", "cuda") else "cpu", "used": "cpu"},
        "wall_s": float(wall_s), "claim_bearing": CLAIM_BEARING, "status": "COMPLETE",
        "stamps": out["stamps"], "report": {"status": "REPORT_PENDING"},
        "protocol_tag": out["tag"].to_dict(), "claim_bearing_reason": CLAIM_BEARING_REASON,
        "writer": "standalone (adapters.nuscenes_planning.write_run)",
    }
    files = {}
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "bench_run.json":
            files[p.relative_to(run_dir).as_posix()] = {"bytes": p.stat().st_size,
                                                        "sha256": _sha256_file(p), "location": "repo"}
    bench_run["files"] = files
    bench_run["utc_end"] = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (run_dir / "bench_run.json").write_text(json.dumps(bench_run, indent=1, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    return {"bench_run": bench_run, "summary": summary, "run_dir": str(run_dir)}


def load_split(nuscenes_root: str | os.PathLike, split: str, version: str | None = None):
    """(Meta, scene names present, n_logs) — REFUSES a partial split and, via
    ``tanitad.data.nuscenes.load_tables``, absent metadata (with the human acquisition steps)."""
    version = version or ("v1.0-mini" if split.startswith("mini") else "v1.0-trainval")
    meta = Meta.load(nuscenes_root, version)
    scenes = devkit_split(split)
    present = [s for s in scenes if s in meta.scene_by_name]
    if len(present) != len(scenes):
        raise RefusedInput(f"split {split}: {len(scenes) - len(present)} of {len(scenes)} scenes "
                           f"are absent from {version} — refusing a partial split")
    logs = {meta.scene_by_name[s].get("log_token") for s in present}
    return meta, present, len(logs)


def run_nuscenes_ol(nuscenes_root: str | os.PathLike, protocol: str, split: str, arms: Sequence[str],
                    run_dir: str | os.PathLike, *, version: str | None = None,
                    model_arms: Mapping[str, dict] | None = None, limit: int | None = None,
                    ckpt: Mapping | None = None, device: str = "cpu",
                    command: Sequence[str] | None = None) -> dict:
    """End to end for ONE protocol: metadata -> arms -> the §1 run directory (standalone).

    Refuses (NuScenesTermsError, carrying the human acquisition steps) when the metadata is
    absent — the PI ACTION LIST in the W6 RESULT.md is the unblock."""
    t0 = time.time()
    protocol_tag(protocol)                                      # closed set, before any I/O
    meta, present, n_logs = load_split(nuscenes_root, split, version)
    ev = evaluate_arms(meta, present, protocol, arms, model_arms=model_arms, limit=limit)
    return write_run(ev, run_dir, split_name=split, n_scenes=len(present), n_logs=n_logs,
                     ckpt=ckpt, device=device, wall_s=time.time() - t0, command=command)


# --------------------------------------------------------------------------- #
# 11b. Model arms — our checkpoint on nuScenes CAM_FRONT                         #
# --------------------------------------------------------------------------- #
#: nuScenes model arms. ⛔ No command arm: nuScenes' command is the GT future thresholded at
#: +-2 m (the route-echo leak) — refused, not offered. A2 is the vision-only arm our binding
#: rule admits; A3 adds the measured t0 ego state (PI 2026-09-02: v0 at t0 is admissible).
MODEL_ARM_SPECS = {
    "A2_vision_pure": {"bridge_arm": "A2_vision_pure", "ego": False,
                       "declared": ["CAM_FRONT keyframe(s) -> cylindrical training frame via "
                                    "calib.pinhole_rectify (observed mask stamped)"]},
    "A3_ego_nocmd": {"bridge_arm": "A3_ego_nocmd", "ego": True,
                     "declared": ["CAM_FRONT keyframe(s) -> cylindrical training frame via "
                                  "calib.pinhole_rectify (observed mask stamped)",
                                  "ego_velocity[t0], ego_acceleration[t0] from ego_pose backward "
                                  "differences (causal; LIDAR_TOP pose chain)"]},
}
REFUSED_MODEL_ARMS = {
    "A1_ego_cmd": "the nuScenes 'driving command' is the GT future thresholded at +-2 m "
                  "(VAD converter :452-457, UniAD trajectory_api.py:272-280) - an echo of the label",
    "A1NT_ego_cmd_nearest": "same GT-derived command as A1",
}


def ego_state_backward(meta: Meta, lidar_sd: dict) -> dict:
    """(vx, vy, ax, ay) in the ego frame at t0 from the LIDAR_TOP pose chain, backward only."""
    def pose(sd):
        # ⚠️ the timestamp stays an INTEGER (microseconds) until it is DIFFERENCED: nuScenes
        # stamps are ~1.6e15 us, and `t0*1e-6 - t1*1e-6` loses ~4e-6 RELATIVE on a 50 ms gap
        # (MEASURED on the fixture: v0 3.9999847 for a true 4.0 m/s).
        ep = meta.by["ego_pose"][sd["ego_pose_token"]]
        R = _quat_to_rotmat(ep["rotation"])
        return np.asarray(ep["translation"][:2], float), R, int(sd["timestamp"])
    p0, R0, t0 = pose(lidar_sd)
    chain = [lidar_sd]
    while len(chain) < 3 and chain[-1].get("prev") and chain[-1]["prev"] in meta.by["sample_data"]:
        chain.append(meta.by["sample_data"][chain[-1]["prev"]])
    if len(chain) < 2:
        return {"ego_velocity": [0.0, 0.0], "ego_acceleration": [0.0, 0.0],
                "source": "no previous pose: zeros"}
    ps = [pose(sd) for sd in chain]
    v01 = (ps[0][0] - ps[1][0]) / max((ps[0][2] - ps[1][2]) * 1e-6, 1e-6)
    a = np.zeros(2)
    if len(ps) >= 3:
        v12 = (ps[1][0] - ps[2][0]) / max((ps[1][2] - ps[2][2]) * 1e-6, 1e-6)
        a = (v01 - v12) / max(0.5 * (ps[0][2] - ps[2][2]) * 1e-6, 1e-6)
    Rw = R0[:2, :2].T                                   # global -> ego(t0)
    return {"ego_velocity": (Rw @ v01).tolist(), "ego_acceleration": (Rw @ a).tolist(),
            "source": "ego_pose backward differences on the LIDAR_TOP chain (t <= t0)"}


def _pil_loader(path: str) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def make_model_arm(arm: str, *, nuscenes_root: str | os.PathLike, forward: Callable,
                   window_rows: int, construction: str = "ST", frame: str = "wide",
                   image_loader: Callable | None = None, n_stack: int = 3) -> dict:
    """A ``model_arms`` entry for :func:`evaluate_arms`.

    ``forward(rows_u8 [W, 3*n_stack, H, W], decl) -> knots [8, 2]`` in OUR ego frame (the real one
    is :func:`refc_forward`; tests pass a fake). Per sample: CAM_FRONT frames at or before t0 (on
    disk) -> :func:`history_slots` -> ``calib.pinhole_rectify`` into the training frame (stamped)
    -> the trainer's own ``stack_frames`` -> ``forward`` -> :func:`plan_to_lidar`."""
    if arm in REFUSED_MODEL_ARMS:
        raise RefusedInput(f"model arm {arm} refused: {REFUSED_MODEL_ARMS[arm]}")
    if arm not in MODEL_ARM_SPECS:
        raise RefusedInput(f"unknown model arm {arm!r}; admissible: {sorted(MODEL_ARM_SPECS)}")
    spec = MODEL_ARM_SPECS[arm]
    root = Path(nuscenes_root)
    fr = model_frame(frame)
    load = image_loader or _pil_loader
    stamps: list = []

    def plan_fn(meta: Meta, sample: dict) -> np.ndarray:
        _ensure_stack_on_path()
        import torch
        from tanitad.data.comma2k19 import stack_frames
        sd_cam = meta.key_sd.get((sample["token"], "CAM_FRONT"))
        if sd_cam is None:
            raise RefusedInput(f"sample {sample['token']} has no CAM_FRONT keyframe")
        intr = camera_intrinsics(meta, sd_cam)
        cams, sd = [], sd_cam
        while sd is not None and len(cams) < 40:
            if (root / sd["filename"]).is_file():
                cams.append((int(sd["timestamp"]), sd["filename"]))
            sd = meta.by["sample_data"].get(sd.get("prev", "")) if sd.get("prev") else None
        hs = history_slots(int(sd_cam["timestamp"]), cams, window_rows + n_stack - 1, construction)
        uniq = sorted(set(hs["files"]))
        imgs = np.stack([load(str(root / f)) for f in uniq])
        rect, stamp = rectify_cam_front(torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous(),
                                        intr, fr)
        src = [uniq.index(f) for f in hs["files"]]
        rows = stack_frames(rect[src].contiguous(), n_stack)
        decl = {"_arm": spec["bridge_arm"], "_declared": []}
        if spec["ego"]:
            es = ego_state_backward(meta, meta.lidar_sd(sample))
            decl.update({"_declared": ["ego_velocity[t0]", "ego_acceleration[t0]"],
                         "ego_velocity": es["ego_velocity"], "ego_acceleration": es["ego_acceleration"]})
        knots = np.asarray(forward(rows, decl), dtype=np.float64)
        cs = meta.by["calibrated_sensor"][meta.lidar_sd(sample)["calibrated_sensor_token"]]
        xy, _ = plan_to_lidar(knots, cs["translation"], cs["rotation"])
        if not stamps:
            stamps.append({**stamp, "history": {k: hs[k] for k in ("construction", "slots_s",
                                                                   "max_abs_time_error_s")}})
        return xy

    def geometry():
        return stamps[0] if stamps else None

    return {"plan_fn": plan_fn, "declared_inputs": list(spec["declared"]) +
            [f"history construction {construction} (window {window_rows} rows x {n_stack} frames)"],
            "command_source": "none", "geometry_fn": geometry, "arm_kind": "model"}


def refc_forward(ckpt: str, config: str | None = None, device: str = "cpu"):
    """The real forward: W1's PROMOTED E2 bridge (``taniteval.bench.navsim.bridge``, read-only
    use) — ``load_refcv4b`` + ``run_model``. Returns ``(forward, window_rows, provenance)``.
    ⚠️ UNTESTED END TO END here: no nuScenes images exist on this box (terms gate)."""
    from taniteval.bench.navsim import bridge as B
    model, cfg, targs, prov, arm_mod = B.load_refcv4b(ckpt, config)
    tr = arm_mod.trainer()
    steps, window = int(prov["decoder_steps"]), int(prov["window"])

    def forward(rows, decl):
        return B.run_model(model, tr, rows, decl, steps)["traj"]

    return forward, window, {"window_rows": window, "decoder_steps": steps,
                             "step": prov.get("step"), "horizons": prov.get("horizons")}


# --------------------------------------------------------------------------- #
# 12. CLI                                                                       #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    """The SPEC §2 literals, in-process (the pytest file is the real gate)."""
    gt = np.stack([[0.0, 2.5 * (k + 1)] for k in range(N_FUTURE)])[None]
    lin = gt + np.stack([[0.5 * (k + 1), 0.0] for k in range(N_FUTURE)])[None]
    k = kernel_stp3(lin, gt)
    a = score(k, variant_tag(Reduction.AT_T, Pipeline.STP3))
    b = score(k, variant_tag(Reduction.AVG_UP_TO_T, Pipeline.STP3))
    got = ([a.get("L2_m", h).value for h in HEADLINE_HORIZONS],
           [b.get("L2_m", h).value for h in HEADLINE_HORIZONS])
    want = ([1.0, 2.0, 3.0, 2.0], [0.75, 1.25, 1.75, 1.25])
    print(json.dumps({"at_t": got[0], "avg_up_to_t": got[1], "expected": want,
                      "ego_box_pixels": int(len(_RC)), "pass": got == want and len(_RC) == 32}))
    return 0 if (got == want and len(_RC) == 32) else 1


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="nuScenes open-loop planning harness (claim_bearing: false)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    g = sub.add_parser("geometry")
    g.add_argument("--frame", default="wide", choices=("wide", "rig_clean"))
    r = sub.add_parser("run")
    r.add_argument("--nuscenes-root", required=True)
    r.add_argument("--split", default="val")
    r.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    r.add_argument("--arms", default="GT,STOP,CV")
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=None)
    x = sub.add_parser("extract-list")
    x.add_argument("--nuscenes-root", required=True)
    x.add_argument("--split", default="val")
    x.add_argument("--channels", default="CAM_FRONT")
    x.add_argument("--sweeps", action="store_true")
    x.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return _selftest()
    if a.cmd == "geometry":
        _ensure_stack_on_path()
        from tanitad.data.calib import NUSCENES_CAM_FRONT_INTR_NOMINAL
        print(json.dumps(geometry_stamp(NUSCENES_CAM_FRONT_INTR_NOMINAL, model_frame(a.frame)),
                         indent=1))
        return 0
    if a.cmd == "run":
        out = run_nuscenes_ol(a.nuscenes_root, a.protocol, a.split,
                              [s for s in a.arms.split(",") if s], a.out, limit=a.limit,
                              command=sys.argv)
        print(json.dumps({"run_dir": out["run_dir"], "protocol": out["bench_run"]["protocol"],
                          "headline": {k: v["headline"]["value"]
                                       for k, v in out["summary"]["arms"].items()}}, indent=1))
        return 0
    if a.cmd == "extract-list":
        meta = Meta.load(a.nuscenes_root, "v1.0-mini" if a.split.startswith("mini") else "v1.0-trainval")
        n = write_extract_list(meta, devkit_split(a.split), a.channels.split(","), a.sweeps, a.out)
        print(json.dumps({"members": n, "out": a.out}))
        return 0
    return 2


def write_extract_list(meta: Meta, scenes: Sequence[str], channels: Sequence[str],
                       sweeps: bool, out: str | os.PathLike) -> int:
    """The exact tar member list (``samples/...`` [+ ``sweeps/...``]) for the listed scenes and
    channels — feed to ``tar -xzf <part>.tgz -T <list>`` so only those files land on disk."""
    toks = {s["token"] for name in scenes for s in meta.samples_of_scene[meta.scene_by_name[name]["token"]]}
    names = []
    for sd in meta.t.get("sample_data", []):
        ch = meta.channel_of_cs.get(sd.get("calibrated_sensor_token"))
        if ch not in channels or sd.get("sample_token") not in toks:
            continue
        if sd.get("is_key_frame") or sweeps:
            names.append(sd["filename"])
    Path(out).write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")
    return len(names)


if __name__ == "__main__":
    sys.exit(main())
