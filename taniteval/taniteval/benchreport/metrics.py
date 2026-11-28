"""Protocol metric sets and per-scene statistics for the W5 report (pure stdlib + numpy).

Everything here is DEFINITION, not measurement: which sub-metrics a protocol has, which are
multiplicative (a zero anywhere zeroes the scene), what the devkit CSV columns are called, and the
small per-scene statistics the report needs (scene means, zero rates, paired win/tie/loss).

The EPDMS / PDMS definitions are the devkit's own, as pinned in
``products/P7-TanitEval/CRITERIA_REGISTRY.json`` ``benchmarks.navsim.variants`` +
``EPDMS_submetrics`` (and E1's control C4: ``score = NC·DAC·DDC·TLC·(5EP+5TTC+2LK+2HC+2EC)/16``,
MEASURED to 1.1e-16 on 220 rows). ⚠️ The registry's ``EPDMS_v2.multipliers`` omits TLC (E1 integration
ask 2); the devkit, its docs and C4 all have FOUR multipliers, so four are used here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- the closed protocol set (BUILD_PLAN §1 + registry navsim.cross_protocol) -------------------
NAVSIM_PROTOCOLS = (
    "PDMS_v1_navtest",
    "EPDMS_v2_navhard_two_stage",
    "EPDMS_v2_navtest_single_stage",
    "EPDMS_v2_private_test_hard_two_stage",
    "EPDMS_v2_warmup_two_stage",
)
OTHER_PROTOCOLS = ("nuScenes_OL_L2_stp3", "nuScenes_OL_L2_uniad", "internal_t1")
KNOWN_PROTOCOLS = NAVSIM_PROTOCOLS + OTHER_PROTOCOLS


@dataclass(frozen=True)
class MetricSet:
    """One protocol family's sub-metric definition."""
    name: str
    multipliers: tuple
    weighted: dict
    denominator: float
    columns: dict                       # short name -> devkit CSV column STEM
    long_names: dict = field(default_factory=dict)
    higher_is_better: bool = True
    stages: tuple = ("stage_one", "stage_two")

    @property
    def submetrics(self) -> tuple:
        return tuple(self.multipliers) + tuple(self.weighted)


EPDMS_V2 = MetricSet(
    name="EPDMS_v2",
    multipliers=("NC", "DAC", "DDC", "TLC"),
    weighted={"EP": 5, "TTC": 5, "LK": 2, "HC": 2, "EC": 2},
    denominator=16.0,
    columns={"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
             "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
             "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
             "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"},
    long_names={"NC": "no at-fault collision", "DAC": "drivable-area compliance",
                "DDC": "driving-direction compliance", "TLC": "traffic-light compliance",
                "EP": "ego progress", "TTC": "time-to-collision within bound",
                "LK": "lane keeping", "HC": "history comfort", "EC": "extended (two-frame) comfort"},
)

PDMS_V1 = MetricSet(
    name="PDMS_v1",
    multipliers=("NC", "DAC"),
    weighted={"EP": 5, "TTC": 5, "C": 2},
    denominator=12.0,
    columns={"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
             "EP": "ego_progress", "TTC": "time_to_collision_within_bound", "C": "comfort"},
    long_names={"NC": "no at-fault collision", "DAC": "drivable-area compliance",
                "EP": "ego progress", "TTC": "time-to-collision within bound", "C": "comfort"},
    stages=("single",),
)

METRIC_SETS = {"EPDMS_v2": EPDMS_V2, "PDMS_v1": PDMS_V1}


def metric_set_for(protocol: str) -> MetricSet | None:
    if protocol.startswith("EPDMS_v2"):
        return EPDMS_V2
    if protocol.startswith("PDMS_v1"):
        return PDMS_V1
    return None


# --- the t0-speed bands (E2 RESULT.md §3 used exactly these edges) -------------------------------
SPEED_BANDS = (
    ("v_lt1", "v0 < 1 m/s", 0.0, 1.0),
    ("v_1_4", "1 ≤ v0 < 4 m/s", 1.0, 4.0),
    ("v_4_8", "4 ≤ v0 < 8 m/s", 4.0, 8.0),
    ("v_ge8", "v0 ≥ 8 m/s", 8.0, math.inf),
)

TIE_TOL = 1e-12          # E2 parse_scores.py: |d| <= 1e-12 is a tie


def band_of(v0: float) -> str:
    for bid, _lab, lo, hi in SPEED_BANDS:
        if lo <= v0 < hi:
            return bid
    raise ValueError(f"v0={v0!r} outside every band")


def mean(xs) -> float | None:
    xs = [float(x) for x in xs]
    if not xs:
        return None
    return math.fsum(xs) / len(xs)


def wtl(deltas, tol: float = TIE_TOL) -> dict:
    """Paired per-scene win / tie / loss of arm − floor (E2's rule: |d| <= tol is a tie)."""
    d = [float(x) for x in deltas]
    return {"wins": sum(1 for x in d if x > tol), "ties": sum(1 for x in d if abs(x) <= tol),
            "losses": sum(1 for x in d if x < -tol), "n": len(d), "tie_tol": tol}


def failing_submetrics(row: dict, mset: MetricSet) -> list:
    """Sub-metrics a scene did not get full marks on, as (name, value, kind).

    kind: 'zero-multiplier' (the scene score is 0), 'partial-multiplier' (NC/DDC 0.5),
    'zero' (a weighted term at 0), 'partial' (a weighted term in (0,1)).
    """
    out = []
    for k in mset.multipliers:
        v = row.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if v == 0:
            out.append((k, v, "zero-multiplier"))
        elif v < 1:
            out.append((k, v, "partial-multiplier"))
    for k in mset.weighted:
        v = row.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if v == 0:
            out.append((k, v, "zero"))
        elif v < 1:
            out.append((k, v, "partial"))
    return out
