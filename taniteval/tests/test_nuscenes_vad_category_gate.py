"""VAD collision selects agents by NAME, and the index-order audit is recorded — not trusted.

HISTORY, both steps MEASURED 2026-09-26:
1. ``55aa747`` — the index-based grid was REFUSED (collision UNAVAILABLE) on the 23-entry base metadata.
2. The pre-registered name-based fix (``…/raw/nuscenes/PREREG_VAD_NAME_BASED.md``, sha256 3ca3f905…)
   PASSED its external gate: VAD-protocol GT-collision floor **1.035 / 0.987 / 0.938 %** at 1/2/3 s vs
   PARA-Drive Table 8's **1.02 / 0.96 / 0.91 %** — within 3 % at every horizon, same decreasing shape.
   Selection is now ``vad_target_by_name``; the tests below pin it AND keep the refusal machinery honest.

⛔ WHY (MEASURED 2026-09-26, the harness's first contact with real nuScenes metadata). VAD selects
colliding agents by raw ``category.json`` INDEX — ``{2..8}`` pedestrian, ``{14..23}`` vehicle — which
is correct only for the 32-entry lidarseg ordering. The base ``v1.0-trainval_meta`` ships 23 entries,
and there those indices pick ``animal`` and ``vehicle.car`` as "human", barriers and traffic cones as
"vehicle", and miss adult/child pedestrians and every car, truck, bus and two-wheeler. The harness
reported a VAD-protocol GT-collision floor of **0.359 %** against PARA-Drive Table 8's published
**0.96 %** — 2.7x too low, in the direction dropping most road users predicts.

⭐ The audit that would have caught it (``vad_category_index_audit``) existed and was NEVER CALLED on
the benchmark path — the eighth "built, tested, unreachable from its caller" instance in this programme.

Expectations are LITERALS. The accept-path fixture is SYNTHETIC and labelled as such: it tests the
predicate's decision rule, and makes no claim about the real lidarseg file's order (guessing that order
is precisely the bug).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))       # <repo>/taniteval/tests
_TE = os.path.dirname(_HERE)                             # <repo>/taniteval
_REPO = os.path.dirname(_TE)                             # <repo>
for _p in (os.path.join(_REPO, "stack"), _TE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import nuscenes_planning as NP              # noqa: E402

REAL_CATEGORY = "D:/Archive/devbox-C/nuscenes/data/v1.0-trainval/category.json"

HUMAN7 = ["human.pedestrian.adult", "human.pedestrian.child", "human.pedestrian.construction_worker",
          "human.pedestrian.personal_mobility", "human.pedestrian.police_officer",
          "human.pedestrian.stroller", "human.pedestrian.wheelchair"]
VEHICLE10 = ["vehicle.bicycle", "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.car",
             "vehicle.construction", "vehicle.emergency.ambulance", "vehicle.emergency.police",
             "vehicle.motorcycle", "vehicle.trailer", "vehicle.truck"]


class _Meta:
    def __init__(self, names):
        self.t = {"category": [{"name": n} for n in names]}


def _intended_32():
    """SYNTHETIC: 32 slots where {2..8} are exactly the 7 pedestrian names and {14..23} exactly the
    10 vehicle names — the ordering VAD's indices ASSUME. Filler names are neither human. nor vehicle."""
    names = [f"filler.{i}" for i in range(32)]
    for k, i in enumerate(range(2, 9)):
        names[i] = HUMAN7[k]
    for k, i in enumerate(range(14, 24)):
        names[i] = VEHICLE10[k]
    return names


# ------------------------------------------------------------------ the predicate

def test_accepts_the_ordering_vad_assumes():
    """POSITIVE CONTROL: without it a guard that refuses EVERYTHING would pass every other test."""
    ok, reason = NP.vad_category_order_ok(NP.vad_category_index_audit(_Meta(_intended_32())))
    assert ok is True and reason == ""


def test_refuses_an_out_of_range_index():
    names = _intended_32()[:23]                                 # 23 entries: index 23 cannot exist
    ok, reason = NP.vad_category_order_ok(NP.vad_category_index_audit(_Meta(names)))
    assert ok is False and "[23]" in reason and "out of range" in reason


def test_refuses_when_a_pedestrian_index_selects_a_car():
    names = _intended_32()
    names[8] = "vehicle.car"                                    # the real 23-entry file does this
    ok, reason = NP.vad_category_order_ok(NP.vad_category_index_audit(_Meta(names)))
    assert ok is False and "non-human" in reason


@pytest.mark.skipif(not os.path.exists(REAL_CATEGORY), reason="nuScenes metadata not on this box")
def test_the_REAL_base_metadata_is_refused_and_says_why():
    """Pinned against the actual file the harness met on 2026-09-26."""
    cats = json.load(open(REAL_CATEGORY, encoding="utf-8"))
    names = [c["name"] for c in cats]
    assert len(names) == 23                                     # the base, not lidarseg
    audit = NP.vad_category_index_audit(_Meta(names))
    ok, reason = NP.vad_category_order_ok(audit)
    assert ok is False
    assert audit["indices_out_of_range"] == [23]
    assert set(audit["human_names_not_selected"]) == {"human.pedestrian.adult", "human.pedestrian.child"}
    assert "vehicle.truck" in audit["vehicle_names_not_selected"]
    assert "pedestrians MISSED" in reason and "selects by NAME" in reason


# ------------------------------------------- the name rule (pre-registered gates A and B)

#: the pre-registered effective set — a LITERAL, written before the run (PREREG §"Derived effective set")
EXPECTED_PEDESTRIANS = ["human.pedestrian.adult", "human.pedestrian.child",
                        "human.pedestrian.construction_worker", "human.pedestrian.police_officer"]
EXPECTED_VEHICLES = ["vehicle.bicycle", "vehicle.bus.bendy", "vehicle.bus.rigid", "vehicle.car",
                     "vehicle.construction", "vehicle.motorcycle", "vehicle.trailer", "vehicle.truck"]


def _effective(names, rule):
    """Apply the UNCHANGED detection-class filter, then a target rule; return (peds, vehs)."""
    sel = {n: rule(i, n) for i, n in enumerate(names) if NP.NAME_MAPPING.get(n) in NP.DET_CLASSES}
    return sorted(n for n, t in sel.items() if t == 2), sorted(n for n, t in sel.items() if t == 1)


def _index_rule(i, _n):
    return 1 if i in NP.VAD_VEHICLE_INDEX else (2 if i in NP.VAD_HUMAN_INDEX else 0)


def _name_rule(_i, n):
    return NP.vad_target_by_name(n)


def test_GATE_A_name_rule_equals_index_rule_on_the_ordering_vad_assumed():
    """On the ordering VAD's indices assume, the name rule reproduces verbatim VAD slot for slot."""
    names = _intended_32()
    for i, n in enumerate(names):
        assert _index_rule(i, n) == _name_rule(i, n), f"slot {i} {n!r} disagrees"


@pytest.mark.skipif(not os.path.exists(REAL_CATEGORY), reason="nuScenes metadata not on this box")
def test_GATE_B_real_metadata_selects_exactly_the_preregistered_12():
    names = [c["name"] for c in json.load(open(REAL_CATEGORY, encoding="utf-8"))]
    peds, vehs = _effective(names, _name_rule)
    assert peds == EXPECTED_PEDESTRIANS
    assert vehs == EXPECTED_VEHICLES
    assert "movable_object.barrier" not in peds + vehs
    assert "movable_object.trafficcone" not in peds + vehs


@pytest.mark.skipif(not os.path.exists(REAL_CATEGORY), reason="nuScenes metadata not on this box")
def test_MUTATION_the_index_rule_on_the_same_file_is_the_bug():
    """⛔ Deliberate regression: put the INDEX rule back and the selection must be the broken one —
    barriers and cones in, adult/child pedestrians and the common vehicles out. If this ever reads
    identical to the name rule, the fix is not what is making the difference."""
    names = [c["name"] for c in json.load(open(REAL_CATEGORY, encoding="utf-8"))]
    peds_i, vehs_i = _effective(names, _index_rule)
    assert (peds_i, vehs_i) != _effective(names, _name_rule)
    assert "human.pedestrian.adult" not in peds_i
    assert "vehicle.car" not in vehs_i
    assert "movable_object.barrier" in vehs_i or "movable_object.trafficcone" in vehs_i


# ------------------------------------------------------------------ the kernel

def _toy(n=4):
    rng = np.random.default_rng(0)
    gt = rng.normal(0, 3, (n, NP.N_FUTURE, 2)).astype(np.float32)
    pred = (gt + rng.normal(0, 0.5, gt.shape)).astype(np.float32)
    # per-SAMPLE flag, shape [N]: the real call site passes `scored` here (VAD's fut_valid rule
    # applied per sample), not a per-timestep mask
    return pred, gt, np.ones(n, bool)


def test_refused_kernel_reports_UNAVAILABLE_with_the_reason():
    pred, gt, fv = _toy()
    k = NP.kernel_vad(pred, gt, None, fv, unavailable_reason="because X")
    assert k.collision_available is False and k.collision_unavailable_reason == "because X"


def test_refusal_does_not_touch_L2():
    """⭐ The refusal must remove collision and ONLY collision: L2 bit-identical with and without it."""
    pred, gt, fv = _toy()
    occ = np.zeros((pred.shape[0], NP.N_FUTURE, NP.BEV, NP.BEV), np.uint8)
    with_occ = NP.kernel_vad(pred, gt, occ, fv)
    refused = NP.kernel_vad(pred, gt, None, fv, unavailable_reason="r")
    assert np.array_equal(with_occ.l2, refused.l2)
    assert with_occ.collision_available is True                 # the normal path still computes


def test_a_refused_collision_is_never_published_as_zero():
    """⛔ The trap a refusal can create: zero arrays averaged into a FAKE 0.0 % floor."""
    pred, gt, fv = _toy()
    k = NP.kernel_vad(pred, gt, None, fv, unavailable_reason="refused for a reason")
    block = NP._gt_control_block({"arms": {"GT": {"kernel": k}}, "scored": fv})
    assert block["status"] == "UNAVAILABLE"
    assert "gt_collision_box_pct" not in block
    assert block["reason"] == "refused for a reason"
