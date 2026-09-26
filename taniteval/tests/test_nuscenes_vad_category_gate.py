"""The VAD collision column must REFUSE when VAD's literal category indices select the wrong classes.

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
    assert "pedestrians MISSED" in reason and "L2 is unaffected" in reason


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
