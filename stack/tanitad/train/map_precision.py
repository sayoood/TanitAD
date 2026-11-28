"""``--w-map-fp`` -- the 10 cm map head's PRECISION term in the planner's near-field corridor (refcv9 warm start).

THE DECISION (PI, 2026-10-27): "the map head ALSO trained, with a precision-targeted loss" (DRAFT_PREREG_REFCV9_WARM s1).

WHY (MEASURED, ``FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-26-refcv9-mapft/result/nearfield_prec_mapft.out``,
783 windows, corridor |y| < 3 m, own p_sidewalk >= 0.5 on GT-seen cells): at 0-5 m the own map's sidewalk flags are
0.261 precise (exact) and 0.62 of them sit on GT-DRIVABLE cells -- false positives right in front of the ego, which is
what makes the planner's map cost useless there. MAPFT's converged hard-label CE did not move it (H-R9-MAPFT-1).

THE TERM -- pure, on the head's own 10 cm logits and the SAME GT codes the CE reads
    L_fp = mean over the GT-DRIVABLE cells of the corridor of ( p_sidewalk + p_edge ),  p = softmax over the 8 classes
    corridor = the 10 cm cells whose CENTRE has x in [0, 20) m ahead and |y| < 3 m.
    * pooled over the batch's LABELLED windows (``map_fine_label``), never a mean of per-window means;
    * a batch with no GT-drivable corridor cell gives an exact 0 attached to the graph (the key is ALWAYS emitted, with
      its cell count, so G-LIVE reads the term on every training step);
    * NOT narrowed by the lift's validity mask: the term is defined over GT-labelled cells, and the planner's map
      cost reads every corridor cell of the raster, observed or not;
    * the CLASS IDS are READ from the label definition (``semantic_map_gt_fine.FINE_CLASSES`` = the code order of the
      10 cm GT, ``semantic_map_gt.CHANNELS[:8]``), never typed: "drivable" -> 1, "non-drivable edge" -> 5,
      "sidewalk / verge" -> 7 (``test_map_precision`` pins the three as literals against the names);
    * the GRID is the map head's (``semantic_map_gt_fine``: row 0 = x in [0, 0.1) m, the NEAREST row; col 0 = y in
      [-y_half, -y_half + 0.1) m, the RIGHT edge; 0.1 m cells). On the refcv8 extent (100 m x +-30 m -> 1000 x 600) the
      corridor is rows 0-199 x cols 270-329 = 12,000 cells; it is CROPPED before the softmax (8 x 200 x 60 per window,
      not 8 x 1000 x 600).
Added as ``w * L_fp`` to the 10 cm map term (``refc_v3_train._map_hires_loss``). ``--w-map-fp 0`` (the default) adds
no term, no key and no model attribute: the tip's graph, bit for bit.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from tanitad.data.semantic_map_gt_fine import FINE_CELL_M, FINE_CLASSES, N_FINE_CLASSES

__all__ = ["FLAG", "DEST", "LOSS_KEY", "COUNT_KEY", "MODEL_ATTR", "CORRIDOR_X_M", "CORRIDOR_Y_HALF_M",
           "DRIVABLE", "EDGE", "SIDEWALK", "corridor_ranges", "corridor_mask", "map_fp_loss", "is_on", "weight",
           "pin_args", "dvb_check", "register_dvb_levers", "stamp"]

FLAG = "--w-map-fp"
DEST = "w_map_fp"
#: the loss key the trainer emits on every training step under w > 0 (G-LIVE reads it), and its cell count
LOSS_KEY = "map_fp"
COUNT_KEY = "n_map_fp_cells"
#: the weight the LOSS reads (set on the model ONLY when w > 0: an absent attribute is the tip's model)
MODEL_ATTR = "_w_map_fp"
#: the corridor (the brief's literals): x in [0, 20) m ahead, |y| < 3 m, by cell CENTRE
CORRIDOR_X_M: tuple = (0.0, 20.0)
CORRIDOR_Y_HALF_M: float = 3.0
#: the class codes, READ from the label definition by NAME
DRIVABLE: int = FINE_CLASSES.index("drivable")
EDGE: int = FINE_CLASSES.index("non-drivable edge")
SIDEWALK: int = FINE_CLASSES.index("sidewalk / verge")


def corridor_ranges(out_hw: tuple, *, x_max_m: float, y_half_m: float, cell_m: float = FINE_CELL_M,
                    x_range_m: tuple = CORRIDOR_X_M, y_half_corr_m: float = CORRIDOR_Y_HALF_M) -> tuple:
    """-> ``((row0, row1), (col0, col1))``, half-open, of the corridor on the head's grid. REFUSES a grid that is not
    the declared extent at ``cell_m`` and an empty corridor. The rows / cols are selected by cell CENTRE
    (``x_i = (i + 0.5) * cell``, ``y_j = -y_half + (j + 0.5) * cell``) and are contiguous by construction."""
    h, w = int(out_hw[0]), int(out_hw[1])
    if h != int(round(float(x_max_m) / cell_m)) or w != int(round(2.0 * float(y_half_m) / cell_m)):
        raise ValueError(f"[map-fp] the grid {h} x {w} is not the extent {x_max_m} m x +-{y_half_m} m at {cell_m} m")
    xc = (torch.arange(h, dtype=torch.float64) + 0.5) * cell_m
    yc = -float(y_half_m) + (torch.arange(w, dtype=torch.float64) + 0.5) * cell_m
    rows = ((xc >= float(x_range_m[0])) & (xc < float(x_range_m[1]))).nonzero().flatten()
    cols = (yc.abs() < float(y_half_corr_m)).nonzero().flatten()
    if rows.numel() == 0 or cols.numel() == 0:
        raise ValueError(f"[map-fp] the corridor is empty on a {h} x {w} grid")
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    assert r1 - r0 == rows.numel() and c1 - c0 == cols.numel()          # contiguous
    return (r0, r1), (c0, c1)


def corridor_mask(out_hw: tuple, *, x_max_m: float, y_half_m: float, **kw) -> Tensor:
    """``[H, W]`` bool -- the corridor of :func:`corridor_ranges` as a full-grid mask (tests, diagnostics)."""
    (r0, r1), (c0, c1) = corridor_ranges(out_hw, x_max_m=x_max_m, y_half_m=y_half_m, **kw)
    m = torch.zeros(int(out_hw[0]), int(out_hw[1]), dtype=torch.bool)
    m[r0:r1, c0:c1] = True
    return m


def map_fp_loss(logits: Tensor, codes: Tensor, *, x_max_m: float, y_half_m: float) -> dict:
    """``logits`` ``[B, 8, H, W]`` and ``codes`` ``[B, H, W]`` uint8 (255 = not seen) on the head's grid -> ``{loss,
    n_cells}``: the mean of ``p_sidewalk + p_edge`` over the GT-drivable corridor cells (module docstring). A batch
    with no such cell returns an exact 0 attached to the graph, ``n_cells`` 0."""
    if logits.dim() != 4 or int(logits.shape[1]) != N_FINE_CLASSES:
        raise ValueError(f"[map-fp] logits must be [B, {N_FINE_CLASSES}, H, W], got {tuple(logits.shape)}")
    if tuple(codes.shape) != (logits.shape[0],) + tuple(logits.shape[2:]) or codes.dtype != torch.uint8:
        raise ValueError(f"[map-fp] codes must be [B, H, W] uint8 on the logits' grid {tuple(logits.shape[2:])}, got "
                         f"{tuple(codes.shape)} {codes.dtype}")
    (r0, r1), (c0, c1) = corridor_ranges(tuple(logits.shape[2:]), x_max_m=x_max_m, y_half_m=y_half_m)
    drv = codes[:, r0:r1, c0:c1].to(logits.device) == DRIVABLE
    n = int(drv.sum())
    if n == 0:
        return {"loss": logits.sum() * 0.0, "n_cells": 0}
    p = torch.softmax(logits[:, :, r0:r1, c0:c1].float(), dim=1)
    fp = p[:, SIDEWALK] + p[:, EDGE]
    return {"loss": fp[drv].sum() / n, "n_cells": n}


# ================================================================================================================= #
# argv, G-DVB, the stamp                                                                                            #
# ================================================================================================================= #
def weight(args) -> float:
    try:
        return float(getattr(args, DEST, 0.0) or 0.0)
    except (TypeError, ValueError):
        return float("nan")


def is_on(args) -> bool:
    return weight(args) > 0.0


def pin_args(args) -> None:
    """The weight's refusals, at argv time. ``--w-map-fp 0`` (the default): nothing is checked, nothing is built.
    w > 0 needs the 10 cm head (``--map-hires on``, ``--w-map-hires > 0``: the term reads its logits and the batch's
    ``map_fine``), and REFUSES a freeze that does not train the map head (the term would move nothing)."""
    w = weight(args)
    if not math.isfinite(w) or w < 0.0:
        raise SystemExit(f"[map-fp] ⛔ {FLAG} {getattr(args, DEST, None)!r}: must be a finite value >= 0")
    if w == 0.0:
        return
    bad = []
    if str(getattr(args, "map_hires", "off") or "off") != "on":
        bad.append("--map-hires is not `on`: the term reads the 10 cm head's logits")
    if not float(getattr(args, "w_map_hires", 0.0) or 0.0) > 0.0:
        bad.append("--w-map-hires is 0: the 10 cm map term (whose labelled windows the precision term reads) is off")
    from tanitad.train import map_only_ft as _mof
    if _mof.is_on(args):
        try:
            mods = _mof.trainable_modules(args)
        except SystemExit:
            mods = None                                  # a malformed --r9-train-only is refused by its own pin
        if mods is not None and not any(m in mods for m in _mof.MAP_ONLY_MODULES):
            bad.append(f"the declared freeze trains {list(mods)}, not the map head: the term would move nothing")
    if bad:
        raise SystemExit(f"[map-fp] ⛔ {FLAG} {w:g} refused:\n  - " + "\n  - ".join(bad))


def stamp(args, x_max_m: float, y_half_m: float, out_hw: tuple) -> dict:
    """The ``config.json["map_hires"]["map_fp"]`` block (absent when off)."""
    (r0, r1), (c0, c1) = corridor_ranges(out_hw, x_max_m=x_max_m, y_half_m=y_half_m)
    return {"flag": FLAG, "w": weight(args), "loss_key": LOSS_KEY, "count_key": COUNT_KEY,
            "term": "mean over GT-drivable corridor cells of (p_sidewalk + p_edge), softmax over the 8 classes",
            "corridor_x_m": list(CORRIDOR_X_M), "corridor_y_half_m": CORRIDOR_Y_HALF_M,
            "corridor_rows": [r0, r1], "corridor_cols": [c0, c1], "n_corridor_cells": (r1 - r0) * (c1 - c0),
            "classes": {"drivable": DRIVABLE, "edge": EDGE, "sidewalk": SIDEWALK},
            "lift_valid_narrowing": False}


def dvb_check(model, args) -> list:
    """G-DVB ``loss`` lever: the weight the LOSS reads (``model._w_map_fp``) is the declared one, the attribute is
    absent when the weight is 0, and a live weight has the 10 cm head it reads."""
    from tanitad.train.declared_vs_built import Mismatch
    w = weight(args)
    got = getattr(model, MODEL_ATTR, None)
    out = []
    if w > 0.0:
        try:
            ok = got is not None and abs(float(got) - w) <= 1e-12
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(Mismatch(FLAG, w, got, f"model.{MODEL_ATTR} (read by the loss)"))
        if getattr(model, "_map_hires", None) is None:
            out.append(Mismatch(FLAG, w, "no 10 cm head", "model._map_hires",
                                "a live precision weight with no head to read supervises nothing"))
    elif got is not None:
        out.append(Mismatch(FLAG, w, got, f"model.{MODEL_ATTR}", "a built weight the argv does not declare"))
    return out


def register_dvb_levers(register) -> None:
    """Called by the trainer at import (the registry is pinned two-way against its parser)."""
    register(DEST, "loss", dvb_check)
