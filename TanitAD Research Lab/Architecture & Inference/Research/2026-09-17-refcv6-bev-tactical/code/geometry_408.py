"""⛔ PI RULING R1 SAYS 408 x 1024. THE TRUNK REFUSES IT. MEASURED.

This was not the question I was sent to answer; it is what the first attempt to
build at the ruling's geometry returned, and it blocks R1 rather than this
package. It is banked as an instrument so the PI gets arithmetic, not an
assertion.

THE FACT
--------
``TimmTrunkConfig.__post_init__`` refuses any axis that 32 does not divide, and
``408 % 32 == 8``. ⭐ **THAT REFUSAL IS CORRECT AND SHOULD NOT BE RELAXED.**
timm's resnets round the spatial size UP at each stride-2 stage, so at 408 the
trunk really emits a **26**-row stride-16 map — while ``timm_trunk.py``'s
``s16_shape = (h // 16, w // 16)`` would DECLARE **25**. The declared shape is
what ``build_perception_branch`` sizes ``BEVLift`` and ``Box3DMemory`` from, so
relaxing the guard does not buy 408: it buys a positional table one row short of
the map it indexes. The guard names exactly this ("the stride-32 map is silently
mis-sized").

THE CHEAP FIX, AND WHY IT IS NOT A COMPROMISE
---------------------------------------------
**416 x 1024** (= 13 x 32) is legal, and on the PI's OWN stated criterion it is
BETTER than 408, not a concession:

* the nearest visible road moves CLOSER (the ruling's headline number), and
* the vertical field is LARGER,

while its stride-16 map is **26 x 64** — precisely the map timm would have
produced at 408 anyway. The costs are +2.0 % pixels and +2.0 % cache over 408,
which lands on a HARD HF ceiling the ruling already flags.

⛔ NO NUMBER IN THIS FILE IS TYPED. Every optical value is computed from
``trunk_shapes``' own frame, and the build/refuse verdict comes from
constructing the real trunk and running a real forward. The 408 row reproducing
the ruling's published ``VFOV 45.296 deg`` is the cross-check that this file's
formula is the same one the ruling used — without it, a disagreement here would
be unreadable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve()
while ROOT.name and not (ROOT / "stack").is_dir():
    ROOT = ROOT.parent
for _p in (str(ROOT / "stack"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.models import trunk_shapes as ts                    # noqa: E402
from tanitad.models.timm_trunk import (TimmResNetTrunk,          # noqa: E402
                                       TimmTrunkConfig)

#: ⚠️ The mount height is the ONE quantity not derivable from the frame. It is
#: BACK-SOLVED from the ruling's own published pair (408 -> 3.15 m) so the
#: comparison below is on the ruling's terms, and it is reported as such rather
#: than presented as a measurement of the rig. The corpus's MEASURED range is
#: 1.2131-1.6672 m, which brackets it.
RULING_H, RULING_NEAR_M = 408, 3.15

HEIGHTS = (256, 384, 408, 416, 448)
WIDTH = 1024


def optics(h: int, w: int, cam_z_m: float) -> dict:
    f = ts.frame_for_width(w, h)
    vfov = 2.0 * math.degrees(math.atan((h / 2.0) / f.f_ref))
    return {"height": h, "width": w, "f_ref": float(f.f_ref),
            "projection": str(f.projection),
            "vfov_deg": vfov,
            # the camera looks along the horizon: the lowest visible ray leaves
            # at vfov/2 below it, so the nearest road point is z / tan(vfov/2).
            "nearest_visible_road_m": cam_z_m / math.tan(math.radians(vfov / 2)),
            "divisible_by_32": (h % 32 == 0) and (w % 32 == 0),
            "megapixels_rel_408": (h * w) / (RULING_H * WIDTH)}


def build_probe(h: int, w: int, name: str = "resnet34.a1_in1k") -> dict:
    """⛔ CONSTRUCT AND RUN, never reason about it. `feature_hw` and the
    dataclass guard are two different checks and only a real forward says what
    the trunk ACTUALLY emits."""
    row: dict = {}
    try:
        row["feature_hw_s16"] = list(ts.feature_hw(ts.frame_for_width(w, h), 16))
    except Exception as exc:
        row["feature_hw_s16"] = None
        row["feature_hw_refusal"] = str(exc)[:160]
    try:
        t = TimmResNetTrunk(TimmTrunkConfig(model_name=name, image_hw=(h, w),
                                            pretrained=False))
    except Exception as exc:
        row["builds"] = False
        row["build_refusal"] = str(exc)[:200]
        return row
    row["builds"] = True
    row["declared_s16_shape"] = list(t.s16_shape)
    row["declared_s32_shape"] = list(t.grid_shape)
    with torch.no_grad():
        s16, s32, _ = t.forward_features(torch.zeros(1, t.k * 3, h, w))
    row["actual_s16_shape"] = list(s16.shape[2:])
    row["actual_s32_shape"] = list(s32.shape[2:])
    row["declared_matches_actual"] = (
        row["declared_s16_shape"] == row["actual_s16_shape"]
        and row["declared_s32_shape"] == row["actual_s32_shape"])
    return row


def ceil_s16(h: int) -> int:
    """What timm WOULD emit at height ``h`` — the number `h // 16` misses."""
    return -(-h // 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # back-solve the mount height from the ruling's own published pair
    vf408 = 2.0 * math.degrees(
        math.atan((RULING_H / 2.0) / ts.frame_for_width(WIDTH, RULING_H).f_ref))
    cam_z = RULING_NEAR_M * math.tan(math.radians(vf408 / 2))

    rows = []
    for h in HEIGHTS:
        r = {**optics(h, WIDTH, cam_z), **build_probe(h, WIDTH)}
        r["timm_would_emit_s16_rows"] = ceil_s16(h)
        r["floor_div_s16_rows"] = h // 16
        r["floor_div_understates_by"] = ceil_s16(h) - (h // 16)
        rows.append(r)

    by = {r["height"]: r for r in rows}
    verdict = {
        "camera_height_m_backsolved_from_ruling": cam_z,
        # ⭐ THE CROSS-CHECK. If this does not reproduce the ruling's own
        # 45.296 deg, every other number here is about a different formula.
        "reproduces_ruling_vfov_45_296": round(by[408]["vfov_deg"], 3) == 45.296,
        "reproduces_ruling_near_3_15": round(
            by[408]["nearest_visible_road_m"], 2) == 3.15,
        # the blocker
        "408_builds": by[408]["builds"],
        "408_build_refusal": by[408].get("build_refusal"),
        "408_floor_div_would_understate_s16_rows_by":
            by[408]["floor_div_understates_by"],
        # the proposal
        "416_builds": by[416]["builds"],
        "416_declared_matches_actual": by[416].get("declared_matches_actual"),
        "416_s16_equals_what_timm_would_emit_at_408": (
            by[416].get("actual_s16_shape", [None])[0]
            == by[408]["timm_would_emit_s16_rows"]),
        "416_is_better_on_the_rulings_own_criterion": (
            by[416]["nearest_visible_road_m"] < by[408]["nearest_visible_road_m"]
            and by[416]["vfov_deg"] > by[408]["vfov_deg"]),
        "416_pixel_cost_vs_408": by[416]["megapixels_rel_408"],
        # ⛔ the alternative that is NOT better, stated so the choice is a
        # comparison rather than a single candidate
        "384_nearest_visible_road_m": by[384]["nearest_visible_road_m"],
        "384_loses_ground_vs_408": (
            by[384]["nearest_visible_road_m"] > by[408]["nearest_visible_road_m"]),
    }
    Path(a.out).write_text(
        json.dumps({"rows": rows, "verdict": verdict,
                    "trunk": "resnet34.a1_in1k (shape probe; the channel count "
                             "differs on resnet101, the SHAPES do not)",
                    "torch": torch.__version__}, indent=2), encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
