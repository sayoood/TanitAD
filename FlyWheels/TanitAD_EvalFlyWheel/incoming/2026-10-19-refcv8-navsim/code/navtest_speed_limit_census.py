#!/usr/bin/env python3
"""navtest_speed_limit_census.py -- how many navtest tokens carry a nuPlan-map posted limit, and which refcv8 road-type
bin {30, 50, 100, 120} km/h a CONTAINING-WINDOW snap would give (my own reading of the ladder, written here as literals;
the model's `speed_max_bin` stays the authority -- this census is ESTIMATED until the bridge records what it fed).

    git -C C:/Users/Admin/tanitad-push show origin/agent/arch-inf-20260803:"FlyWheels/TanitAD_EvalFlyWheel/incoming/\
2026-09-23-refcv6-standard-tests/navsim/raw/inputs/speed_limits_navtest.json.gz" | python navtest_speed_limit_census.py out.json
"""
import collections, gzip, json, sys

ROAD_TYPE_KMH = (30, 50, 100, 120)          # literal (refcv8_train.R8_ROAD_TYPE_KMH)
d = json.loads(gzip.decompress(sys.stdin.buffer.read()))
vals = list((d.get("tokens", d) if isinstance(d, dict) else d).values())
st = collections.Counter(v.get("status") for v in vals)
kmh = collections.Counter(round(v["speed_limit_mps"] * 3.6, 1) for v in vals if v.get("status") == "limit")


def snap(k):
    for s in ROAD_TYPE_KMH:
        if k <= s + 1e-6:
            return s
    return "over_120"


bins = collections.Counter(snap(v["speed_limit_mps"] * 3.6) for v in vals if v.get("status") == "limit")
res = {"source": "refcv6 suite raw/inputs/speed_limits_navtest.json.gz (export_speed_limits.py, nuPlan map lane limits)",
       "n_tokens": len(vals), "status": dict(st), "limit_kmh_histogram": dict(sorted(kmh.items())),
       "containing_window_bin_kmh": {str(k): v for k, v in bins.items()}}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(res, indent=1))
print(json.dumps(res))
