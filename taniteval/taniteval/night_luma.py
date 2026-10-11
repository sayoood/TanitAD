"""NIGHT-LUMA -- the night stratum defined by BRIGHTNESS, not by the clock label.

SPEC_REFCV8_DRAFT section 8.10 (X10) (a), registered text:

    NIGHT-LUMA = the night stratum defined by mean luma < 30 (not the clock label).
    Report-only (c): no bar.

WHY (MEASURED, D3 ``D3_raw_plausibility/RESULT.md`` and ``raw/c5b_daynight_vs_luma.json``):
the corpus ``strata.daynight_clock`` is a CLOCK label. 60 % of "night" train clips are
brighter than the 5th percentile of "day"; luma AUC 0.74; only ~8 % of clips are actually
dark (mean luma < 30) against 46 % labelled night; one "day" clip has mean luma 12.6.

THE DEFINITION IS D3's, TO THE LETTER (``c5_frames.py::frame_stats`` / ``proc``), because
the threshold 30 and the "~8 %" were measured with exactly this statistic and a different
luma convention moves clips across the line:

  * 8-bit scale, 0..255.
  * per-frame luma = the plain MEAN OF THE THREE CHANNELS (``a.mean(2)``) -- NOT
    Rec.601 weights. ``weights="rec601"`` exists to reproduce the programme's other luma
    convention (``build_pai_substrate.py``) and is NOT the registered definition.
  * the black bottom STRIP (rig B, 27-37 rows: consecutive all-zero rows from the bottom)
    is excluded before averaging; D3 detects it on FRAME 0 and applies it to every frame
    (``strip_rows="first"``). Without the exclusion the all-zero rows are averaged in as
    black and a rig-B clip reads proportionally too dark (the test pins the exact ratio).
  * clip luma = the MEAN over frames of the per-frame mean (``mean_of_means``). Not the
    median, not the minimum.
  * night  <=>  clip luma  <  30  (STRICT: a clip at exactly 30.0 is not night).

A fully black frame 0 has no non-zero row, so the strip detector would swallow the whole
frame and D3's code would average an empty slice (NaN). Here it falls back to strip = 0 and
the clip reads 0.0, i.e. dark -- the darkest clip must never be the one that drops out.

LAYER: this is a function over FRAMES. It does not read the corpus; the clip census
(``night_luma_census``) takes clip-level luma already computed. Nothing here has been run
over the corpus.

Zero GPU, numpy only.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "NIGHT_LUMA_THRESHOLD", "RGB_MEAN", "REC601", "bottom_strip_rows", "frame_mean_luma",
    "clip_mean_luma", "is_night_luma", "night_luma_census", "stratum_mask",
]

#: SPEC_REFCV8_DRAFT section 8.10: "mean luma < 30" (8-bit scale).
NIGHT_LUMA_THRESHOLD: float = 30.0
RGB_MEAN = "rgb_mean"          # D3 / registered
REC601 = "rec601"              # NOT registered; reproduces build_pai_substrate.LUMA
_REC601_W = (0.299, 0.587, 0.114)


def _hwc(frames, layout: str) -> np.ndarray:
    """-> ``[T, H, W, 3]`` uint8-valued float64-safe view; accepts one frame or a clip."""
    a = np.asarray(frames)
    if layout == "CHW":
        if a.ndim not in (3, 4):
            raise ValueError(f"CHW frames must be [3,H,W] or [T,3,H,W], got {a.shape}")
        a = np.moveaxis(a, -3, -1)
    elif layout != "HWC":
        raise ValueError(f"layout must be 'HWC' or 'CHW', got {layout!r}")
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] RGB frames, got {a.shape}")
    if a.shape[0] == 0 or a.shape[1] == 0 or a.shape[2] == 0:
        raise ValueError(f"empty frames {a.shape}")
    return a


def bottom_strip_rows(frame) -> int:
    """Consecutive all-zero rows from the bottom of ONE ``[H, W, 3]`` frame (D3 ``strip_rows``).

    A frame that is all zero (every row) returns 0, not H: there is no strip to exclude,
    there is just a black frame (see the module docstring)."""
    a = np.asarray(frame)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"frame must be [H,W,3], got {a.shape}")
    zero_row = a.reshape(a.shape[0], -1).max(axis=1) == 0
    if zero_row.all():
        return 0
    n = 0
    for z in zero_row[::-1]:
        if not z:
            break
        n += 1
    return n


def _luma_plane(a: np.ndarray, weights: str) -> np.ndarray:
    x = a.astype(np.float64)
    if weights == RGB_MEAN:
        return x.mean(axis=-1)
    if weights == REC601:
        return x @ np.asarray(_REC601_W, dtype=np.float64)
    raise ValueError(f"weights must be {RGB_MEAN!r} or {REC601!r}, got {weights!r}")


def frame_mean_luma(frame, *, strip_rows: int = 0, weights: str = RGB_MEAN) -> float:
    """Mean luma (0..255) of ONE ``[H, W, 3]`` frame over the rows above the bottom ``strip_rows``."""
    a = _hwc(frame, "HWC")
    if a.shape[0] != 1:
        raise ValueError(f"frame_mean_luma takes one frame, got {a.shape[0]}")
    h = a.shape[1]
    s = int(strip_rows)
    if s < 0 or s >= h:
        raise ValueError(f"strip_rows {s} leaves no rows of a {h}-row frame")
    return float(_luma_plane(a[0, : h - s], weights).mean())


def clip_mean_luma(frames, *, strip_rows="first", weights: str = RGB_MEAN,
                   layout: str = "HWC") -> float:
    """Clip luma = mean over frames of the per-frame mean luma (D3 ``mean_of_means``).

    ``strip_rows``: ``"first"`` (D3: detected on frame 0, applied to all), ``"none"`` / 0,
    ``"per_frame"``, or an int."""
    a = _hwc(frames, layout)
    if strip_rows == "first":
        s_each = [bottom_strip_rows(a[0])] * a.shape[0]
    elif strip_rows == "per_frame":
        s_each = [bottom_strip_rows(f) for f in a]
    elif strip_rows == "none":
        s_each = [0] * a.shape[0]
    elif isinstance(strip_rows, (int, np.integer)):
        s_each = [int(strip_rows)] * a.shape[0]
    else:
        raise ValueError(f"strip_rows must be 'first'|'per_frame'|'none'|int, got {strip_rows!r}")
    means = [frame_mean_luma(f[None], strip_rows=s, weights=weights) for f, s in zip(a, s_each)]
    return float(np.mean(means))


def is_night_luma(mean_luma: float, threshold: float = NIGHT_LUMA_THRESHOLD) -> bool:
    """``mean_luma < threshold`` (strict). Refuses a non-finite luma: NaN must not read as day."""
    m = float(mean_luma)
    if not np.isfinite(m):
        raise ValueError(f"non-finite clip luma {mean_luma!r} cannot be labelled day or night")
    return bool(m < float(threshold))


def night_luma_census(clip_ids: Sequence, clip_luma: Sequence[float],
                      clock_night: Sequence[bool] | None = None, *,
                      threshold: float = NIGHT_LUMA_THRESHOLD) -> dict:
    """Clip-level census of the luma-defined night stratum, beside the clock label.

    ``clock_night`` (the corpus ``strata.daynight_clock == "night"``) is optional; with it the
    block carries the 2x2 cross-tab that is the finding (clock night vs actually dark)."""
    ids = [str(c) for c in clip_ids]
    luma = np.asarray(clip_luma, dtype=np.float64)
    if luma.shape != (len(ids),):
        raise ValueError(f"clip_luma must be [{len(ids)}], got {luma.shape}")
    if len(set(ids)) != len(ids):
        raise ValueError("clip_ids must be unique (one luma per clip)")
    if not np.isfinite(luma).all():
        raise ValueError(f"{int((~np.isfinite(luma)).sum())} clip(s) have non-finite luma")
    dark = luma < float(threshold)
    out = {"metric": "NIGHT-LUMA", "spec": "SPEC_REFCV8_DRAFT section 8.10 X10 (a)",
           "rule": f"clip mean luma < {float(threshold):g} (8-bit, mean of RGB, bottom strip excluded)",
           "n_clips": len(ids), "n_dark": int(dark.sum()),
           "share_dark": (float(dark.mean()) if ids else None), "report_only": True}
    if clock_night is not None:
        cn = np.asarray(clock_night, dtype=bool)
        if cn.shape != dark.shape:
            raise ValueError(f"clock_night must be [{len(ids)}], got {cn.shape}")
        out["n_clock_night"] = int(cn.sum())
        out["share_clock_night"] = float(cn.mean()) if ids else None
        out["crosstab_clips"] = {
            "clock_night_and_dark": int((cn & dark).sum()),
            "clock_night_and_not_dark": int((cn & ~dark).sum()),
            "clock_day_and_dark": int((~cn & dark).sum()),
            "clock_day_and_not_dark": int((~cn & ~dark).sum()),
        }
    return out


def stratum_mask(window_clip_id: Iterable, clip_luma_by_id: Mapping,
                 threshold: float = NIGHT_LUMA_THRESHOLD) -> dict:
    """Window-level night mask inherited from the clip label, with WINDOW n and CLIP n.

    Every window's clip must have a luma: a missing clip raises (a silent "not night" would
    shrink the stratum and flatter the daytime result)."""
    ids = np.asarray([str(c) for c in window_clip_id])
    lut = {str(k): float(v) for k, v in clip_luma_by_id.items()}
    missing = sorted({c for c in ids.tolist() if c not in lut})
    if missing:
        raise ValueError(f"{len(missing)} clip(s) have windows but no luma, e.g. {missing[:3]}")
    mask = np.asarray([is_night_luma(lut[c], threshold) for c in ids.tolist()], dtype=bool)
    clips_in = sorted(set(ids[mask].tolist()))
    return {"mask": mask, "n_windows": int(mask.sum()), "n_clips": len(clips_in),
            "n_windows_total": int(ids.size), "n_clips_total": int(len(set(ids.tolist()))),
            "clips": clips_in}
