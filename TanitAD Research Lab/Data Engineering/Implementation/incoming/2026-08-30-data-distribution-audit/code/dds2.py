"""DDS v2 — scored against the DESIGN'S OWN strata cells, plus competence terms.

v1 scored coverage over axes I chose. v2 scores it over the cell grid the PI's
own selection design specified (`2026-08-06-alpamayo-augmentation/DESIGN.md:49`:
equal-weight cells over road-class x day/night x country). That matters: the
score now measures deviation from a design someone actually chose, which is
defensible in a way an invented axis set is not.

Two COMPETENCE terms sit beside coverage, because coverage alone can be satisfied
by a broad-but-shallow subset:

  * `defect_regime`  — mass in fast_arterial + highway. 88.7 % of our oracle gap
    is longitudinal and that is where longitudinal control is exercised.
  * `stop_launch`    — share of clips containing a full stop->launch cycle.
    MEASURED on the 20 s trainable window: the corpus holds only 11.8 % (558
    clips), and 2.4 % stop-and-go. A thin competence a score should be able to
    see.

⛔ BOTH ENTER A GEOMETRIC MEAN, so a zero on either sinks the score. A subset that
covers every cell but contains no high-speed driving, or no stop-launch cycle at
all, is not a good 26 h no matter how broad it looks.

⚠️ Windows: every ego-derived input here is computed on the 20 s CAMERA CLIP, not
the 139 s egomotion span. See `measure_stop_coverage.py`.
"""
import numpy as np
import pandas as pd

MIN_PER_CELL = 5
DEFECT_BANDS = {"fast_arterial", "highway"}


def _eff_div(counts: pd.Series) -> float:
    p = counts[counts > 0] / counts.sum()
    if len(p) <= 1:
        return 0.0
    return float(np.exp(-(p * np.log(p)).sum()) / len(counts))


def dds2(subset: pd.DataFrame, design_cells: list[str],
         parent_defect_mass: float, parent_stop_launch: float) -> dict:
    """Score a candidate subset. NO TRAINING REQUIRED.

    `subset` needs: strata_cell, speed_band, has_stop_launch.
    `design_cells` is the FULL cell grid the design specified — coverage is
    measured against the design, not against whatever the subset happens to hold.
    """
    c = subset.strata_cell.value_counts().reindex(design_cells).fillna(0)
    coverage = float((c >= MIN_PER_CELL).sum() / len(design_cells))
    diversity = _eff_div(c)

    sb = subset.speed_band.value_counts(normalize=True)
    defect = float(sum(sb.get(b, 0.0) for b in DEFECT_BANDS))
    defect_ratio = defect / parent_defect_mass if parent_defect_mass > 0 else 0.0

    sl = float(subset.has_stop_launch.mean())
    sl_ratio = sl / parent_stop_launch if parent_stop_launch > 0 else 0.0

    terms = [coverage, diversity, max(defect_ratio, 1e-6), max(sl_ratio, 1e-6)]
    score = float(np.prod(terms) ** (1 / len(terms)))
    return {"dds2": round(score, 4), "coverage_vs_design": round(coverage, 4),
            "cell_diversity": round(diversity, 4),
            "defect_regime_mass": round(defect, 4),
            "defect_ratio": round(defect_ratio, 4),
            "stop_launch_share": round(sl, 4), "stop_launch_ratio": round(sl_ratio, 4),
            "n": int(len(subset)), "d": 4,
            "cells_covered": int((c >= MIN_PER_CELL).sum()),
            "cells_in_design": len(design_cells),
            "blind_axes": ["weather", "road_surface", "traffic_density"]}
