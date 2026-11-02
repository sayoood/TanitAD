"""refcv9 C1 -- the SCORER ADAPTER (REFCV9_DESIGN 4.2): ONE code path that turns a model's map output into the planes MAPSEL /
MAPPATH / ``nearfield_prec`` read, for refcv9's banded channels AND refcv8's 8-way raster -- so both arms are scored by the same
function and a difference between them is the models', not the readers'.

The planes are the ones MAPSEL's dump has always written (``pdrv`` = P(drivable surface), ``pedge``, ``psw`` = P(sidewalk), ``code`` =
the decided class code) at the 10 cm raster of the declared extent, PLUS ``unobs`` (True = the lift saw nothing there: NO EVIDENCE,
never "sidewalk"). Every value is ``uint8`` (probability * 255, rounded) in :func:`to_u8`, the layout the readers load.

* refcv8 (``map_hires_logits`` [B, 8, H, W]): ``p = softmax(z - log w)`` under the DECLARED prior-corrected rule, ``pdrv`` = the sum
  over the drivable CODES (1 2 3 4 6), ``code`` = ``argmax`` -- :func:`planes_from_logits8` is MAPSEL's ``own_planes`` arithmetic
  item for item (pinned equal by ``tests/test_map_planes.py``). ``unobs`` = ``~lift_valid_to_fine(map_hires_lift_valid)`` when the
  forward's lift validity is given, else None (a raster written before the flag existed carries none: ``unobs_source`` says so).
* refcv9 (``map_banded``): ``pdrv = sigmoid(drv)``, ``psw = sigmoid(sdw)``, ``pedge = sigmoid(edge_h)`` (the dilated, ~0.5 m wide
  heatmap: wider than refcv8's 1-cell edge class, which is the point of 2.1), each band resampled to the 10 cm raster by
  nearest-exact; ``code`` is a DERIVED partition for the legacy readers (priority: sidewalk if it beats drv, else drv, then the
  thin overlays hatch < xwalk < arrow < lane on the surface, edge anywhere off the sidewalk), thresholds per channel (default 0.5;
  the harness passes the FIT-split ones); ``unobs`` is the head's own mask resampled the same way.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.models import map_banded as MB

__all__ = ["MapPlanes", "planes_from_logits8", "planes_from_banded", "planes_from_output", "to_u8", "read_raster",
           "DRV_CODES", "EDGE_CODE", "SIDEWALK_CODE"]

DRV_CODES = MB.DRV_CODES
EDGE_CODE, SIDEWALK_CODE = MB.C_EDGE, MB.C_SIDEWALK


class MapPlanes(NamedTuple):
    pdrv: Tensor            # [B, H, W] float in [0, 1]
    pedge: Tensor
    psw: Tensor
    code: Tensor            # [B, H, W] long
    unobs: Tensor | None    # [B, H, W] bool, or None = this output carries no validity
    source: str             # "refcv8" | "refcv9"
    unobs_source: str       # "lift_valid" | "head" | "absent"


def planes_from_logits8(z: Tensor, class_weight, rule: str, lift_valid: Tensor | None = None) -> MapPlanes:
    """refcv8: ``own_planes``' arithmetic, per batch item (so every item is bit-identical to the single-item original)."""
    from tanitad.models import map_head_hires as MH
    if rule != "prior_corrected" or class_weight is None:
        raise RuntimeError(f"[map-planes] the declared decision rule is {rule!r} / class weights "
                           f"{'absent' if class_weight is None else 'present'}: defined for prior_corrected + weights")
    pd, pe, ps, cd = [], [], [], []
    for i in range(z.shape[0]):
        zi = z[i:i + 1].detach().float()
        s = zi - MH._decision_logw(class_weight, zi).view(1, -1, 1, 1)
        p = torch.softmax(s, 1)[0]
        cd.append(s.argmax(1)[0])
        pd.append(p[list(DRV_CODES)].sum(0))
        pe.append(p[EDGE_CODE])
        ps.append(p[SIDEWALK_CODE])
    un = None if lift_valid is None else ~MH.lift_valid_to_fine(lift_valid, tuple(z.shape[2:]))
    return MapPlanes(torch.stack(pd), torch.stack(pe), torch.stack(ps), torch.stack(cd), un, "refcv8",
                     "absent" if un is None else "lift_valid")


def _to_raster(x: Tensor, geom: tuple, x_max_m: float, y_half_m: float, fill) -> Tensor:
    """Per-band ``[B, rows, cols]`` -> the 10 cm raster ``[B, H, W]`` (nearest-exact per band, bands stacked along x)."""
    H, W = int(round(x_max_m / MB.FINE_CELL_M)), int(round(2.0 * y_half_m / MB.FINE_CELL_M))
    outs = []
    for t, g in zip(x, geom):
        r0, r1 = g.fine_rows
        r1 = min(r1, H)
        outs.append(F.interpolate(t[:, None].float(), size=(r1 - r0, W), mode="nearest-exact")[:, 0])
    r = torch.cat(outs, 1)
    if tuple(r.shape[1:]) != (H, W):
        raise ValueError(f"[map-planes] banded planes tile {tuple(r.shape[1:])}, the raster is {(H, W)}")
    return r


def planes_from_banded(mb: dict, *, x_max_m: float, y_half_m: float, thresholds: dict | None = None) -> MapPlanes:
    """refcv9: the banded channels (``out["perception"]["map_banded"]``) -> planes at the 10 cm raster."""
    geom = MB.band_geometry(x_max_m, y_half_m)
    ci = {c: i for i, c in enumerate(MB.CHANNELS)}
    thr = {c: 0.5 for c in MB.PROB_CHANNELS}
    thr.update(thresholds or {})
    prob = {c: _to_raster([torch.sigmoid(z[:, ci[c]].detach().float()) for z in mb["logits"]], geom, x_max_m, y_half_m, 0.0)
            for c in MB.PROB_CHANNELS}
    un = _to_raster([u.float() for u in mb["unobs"]], geom, x_max_m, y_half_m, 1.0) > 0.5
    drv, sw = prob["drv"] >= thr["drv"], (prob["sdw"] >= thr["sdw"]) & (prob["sdw"] > prob["drv"])
    code = torch.zeros(drv.shape, dtype=torch.long, device=drv.device)
    code[drv] = MB.C_DRIVABLE
    code[sw] = MB.C_SIDEWALK
    for name, cc in (("hatch", MB.C_HATCH), ("xwalk", MB.C_XWALK), ("arrow", MB.C_ARROW), ("lane", MB.C_LANE)):
        code[(prob[name] >= thr[name]) & (code == MB.C_DRIVABLE)] = cc
    code[(prob["edge_h"] >= thr["edge_h"]) & (code != MB.C_SIDEWALK)] = MB.C_EDGE
    return MapPlanes(prob["drv"], prob["edge_h"], prob["sdw"], code, un, "refcv9", "head")


def planes_from_output(pout: dict, *, class_weight=None, rule: str | None = None, x_max_m: float | None = None,
                       y_half_m: float | None = None, thresholds: dict | None = None) -> MapPlanes:
    """THE dispatch: ``pout`` = ``out["perception"]``. ``map_banded`` -> refcv9, ``map_hires_logits`` -> refcv8; both absent is an
    error (a scorer that finds no map must say so, not score an empty one)."""
    if pout.get("map_banded") is not None:
        if x_max_m is None or y_half_m is None:
            raise ValueError("[map-planes] refcv9 output: pass the declared extent (x_max_m, y_half_m)")
        return planes_from_banded(pout["map_banded"], x_max_m=x_max_m, y_half_m=y_half_m, thresholds=thresholds)
    if pout.get("map_hires_logits") is not None:
        return planes_from_logits8(pout["map_hires_logits"], class_weight, rule, pout.get("map_hires_lift_valid"))
    raise ValueError("[map-planes] the perception output carries neither `map_banded` nor `map_hires_logits`")


def to_u8(planes: MapPlanes, i: int = 0) -> tuple:
    """Item ``i`` -> ``(pdrv_u8, pedge_u8, psw_u8, code_u8, unobs_u8 | None)`` numpy ``[H, W]`` -- ``own_planes``' ``q`` for the
    probabilities, so a refcv8 item is byte-identical to MAPSEL's."""
    q = lambda t: t.mul(255.0).round().clamp_(0, 255).to(torch.uint8).cpu().numpy()   # noqa: E731
    un = None if planes.unobs is None else planes.unobs[i].to(torch.uint8).cpu().numpy()
    return q(planes.pdrv[i].clone()), q(planes.pedge[i].clone()), q(planes.psw[i].clone()), \
        planes.code[i].to(torch.uint8).cpu().numpy(), un


def read_raster(path) -> dict:
    """The READ side: a raster ``.npz`` of EITHER schema -> ``{pdrv, pedge, psw, code, gt | None, unobs | None, schema,
    unobs_source}``. A raster with no ``unobs`` plane (every refcv8 / pre-C1 raster) reports ``unobs`` None and
    ``unobs_source`` "absent" -- the reader decides what that means; it is never silently all-observed."""
    z = np.load(path)
    out = {k: z[k] for k in ("pdrv", "pedge", "psw", "code")}
    out["gt"] = z["gt"] if "gt" in z.files else None
    out["unobs"] = z["unobs"].astype(bool) if "unobs" in z.files else None
    out["unobs_source"] = "raster" if out["unobs"] is not None else "absent"
    out["schema"] = "refcv9" if out["unobs"] is not None else "refcv8"
    return out
