"""DDS — a Data Distribution Score that ranks candidate subsets BEFORE training.

PI: *"develop a DATA DISTRIBUTION METRIC OR SCORE … achieve the same or better
results with MUCH MUCH LESS DATA … by genius data curation, selection."*

WHY THE SCORE REWARDS DIVERSITY AND NOT REPRESENTATIVENESS — from the banked
primaries, not from taste:
  * Codevilla ICCV 2019: more data was WORSE — best at 10 h of a 100 h corpus,
    failure rising with volume, because "the diversity of the dataset does not
    grow fast enough compared to the main mode of demonstrations". Matching the
    parent's distribution would maximise exactly that main mode.
  * WOD-E2E: a 12 h benchmark curated for scenarios at <0.03 % frequency does
    the work of ~40,000 h accumulated by volume. Rare strata carry the value.
  ⇒ the target is NOT the parent's shape. It is COVERAGE of its strata, with
    rare strata counted, and mass in the regimes where our defect lives.

⛔ THE REFERENCE IS THE PARENT (306,152 clips), NOT B1. MEASURED 2026-08-30: B1
is not a neutral baseline — as a draw from the parent it gives chi-square 7,300
on df=24 (US expected 2,395, observed 302). Scoring against B1 would bake its
selection bias into the score.

⚠️ AXES THIS SCORE CAN SEE: speed regime (derived per clip from egomotion),
country and hour-of-day (catalogue join). ⛔ It CANNOT see weather, road type,
traffic density or surface — those columns DO NOT EXIST in the shipped metadata
(verified by two full column dumps) and must be derived by an instrument we do
not have. The score reports its blind axes rather than implying full coverage.
"""
import numpy as np
import pandas as pd

#: Mean-speed bands in m/s. The boundary at 14.0 is not arbitrary: it is the
#: exact hard gate at `physicalai_r0.py:100`, so the DESIGNED corpus scores a
#: structural zero on the top two bands and the score has a real known-bad arm.
SPEED_BANDS = [(0, 2), (2, 8), (8, 14), (14, 20), (20, 99)]
SPEED_LABELS = ["crawl", "slow_urban", "urban_arterial", "fast_arterial", "highway"]
#: Strata where the programme's largest measured defect lives. 88.7 % of the
#: oracle gap is LONGITUDINAL, and the two fast bands are where longitudinal
#: control is exercised — a subset with none of them cannot exhibit the defect,
#: let alone fix it.
DEFECT_BANDS = {"fast_arterial", "highway"}
MIN_PER_STRATUM = 5          # a stratum with fewer than this is not covered


def speed_band(v: float) -> str:
    for (lo, hi), lab in zip(SPEED_BANDS, SPEED_LABELS):
        if lo <= v < hi:
            return lab
    return SPEED_LABELS[-1]


def _effective_diversity(counts: pd.Series) -> float:
    """exp(Shannon entropy) / n_strata_in_parent — the EFFECTIVE fraction of
    strata actually exercised. Rewards spreading over strata, and unlike a raw
    count it is not satisfied by one token clip in each."""
    p = counts[counts > 0] / counts.sum()
    if len(p) <= 1:
        return 0.0
    return float(np.exp(-(p * np.log(p)).sum()) / len(counts))


def _coverage(counts: pd.Series) -> float:
    return float((counts >= MIN_PER_STRATUM).sum() / len(counts))


def dds(subset: pd.DataFrame, parent: pd.DataFrame,
        axes=("speed_band", "country", "hour_of_day")) -> dict:
    """Score a candidate subset. NO TRAINING REQUIRED — this is the point.

    `subset` / `parent` need the columns in `axes`. Returns per-axis detail plus
    a scalar `dds`, and always reports n and d.
    """
    per = {}
    for ax in axes:
        strata = sorted(parent[ax].dropna().unique())
        c = subset[ax].value_counts().reindex(strata).fillna(0)
        per[ax] = {"coverage": _coverage(c), "effective_diversity": _effective_diversity(c),
                   "n_strata": len(strata), "n_covered": int((c >= MIN_PER_STRATUM).sum())}
    # defect-regime mass: floored, so a subset cannot score well while omitting
    # the regime our largest defect lives in
    sb = subset["speed_band"].value_counts(normalize=True)
    defect_mass = float(sum(sb.get(b, 0.0) for b in DEFECT_BANDS))
    parent_defect = float(sum(
        parent["speed_band"].value_counts(normalize=True).get(b, 0.0) for b in DEFECT_BANDS))
    defect_ratio = defect_mass / parent_defect if parent_defect > 0 else 0.0

    cov = float(np.mean([per[a]["coverage"] for a in axes]))
    div = float(np.mean([per[a]["effective_diversity"] for a in axes]))
    # geometric mean: a ZERO on any term sinks the score. That is deliberate —
    # a subset missing the defect regime entirely must not be rescued by being
    # broad elsewhere, which is exactly how the r0 corpus would otherwise pass.
    score = float((cov * div * max(defect_ratio, 1e-6)) ** (1 / 3))
    return {"dds": round(score, 4), "coverage": round(cov, 4),
            "effective_diversity": round(div, 4),
            "defect_regime_mass": round(defect_mass, 4),
            "defect_ratio_vs_parent": round(defect_ratio, 4),
            "n": int(len(subset)), "d": len(axes), "per_axis": per,
            "blind_axes": ["weather", "road_type", "traffic_density", "surface"]}
