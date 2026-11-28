#!/usr/bin/env python3
"""Dev box (navsim-crun venv, the same venv that built ulp_reference_navhard_devbox.json): map every warmup synthetic
token to its pickle file by loading the scene exactly as make_ulp_reference.py's lite mode does
(Scene.load_from_disk(file, None, no-sensors)) and reading scene_metadata.initial_token.  Writes the
{token: file} map and the {"tokens": [...]} list make_ulp_reference.py --synthetic-map / --tokens read.

    python nhwu_warmup_synmap.py <synthetic_scene_pickles dir> <map.json> <tokens.json>
"""
import json
import os
import sys
from pathlib import Path

from navsim.common.dataclasses import Scene, SensorConfig

d, map_out, tok_out = sys.argv[1], sys.argv[2], sys.argv[3]
m = {}
for f in sorted(x for x in os.listdir(d) if x.endswith(".pkl")):
    sc = Scene.load_from_disk(Path(d) / f, None, SensorConfig.build_no_sensors())
    t = sc.scene_metadata.initial_token
    if t in m:
        raise SystemExit(f"duplicate token {t}: {m[t]} and {f}")
    m[t] = f
json.dump(m, open(map_out, "w", encoding="utf-8"), indent=0, sort_keys=True)
json.dump({"tokens": sorted(m)}, open(tok_out, "w", encoding="utf-8"), indent=0)
print(json.dumps({"n_files": len(m)}))
