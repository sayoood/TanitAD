#!/usr/bin/env python3
"""W2 (E3): are E1's adapter gaps G1-G6 still open? A POSITIVE assertion per gap.

E1's `code/build_artifacts.py` docstring lists six things `taniteval/adapters/navsim.py` did
not cover, each filled locally in E1's own script. That list was written against the adapter
as it stood BEFORE this stream's updates, so "still open" is a claim to be MEASURED, not
inherited. Each gap below is checked by calling the adapter and asserting the key/behaviour
is THERE — and G1 is checked against a REAL devkit row off E1's banked navhard CSV when one
is present, with a synthetic row as the fallback (the fallback is stated in the output).

Run: PYTHONIOENCODING=utf-8 <venv>/python.exe code/e1_adapter_gaps_G1_G6.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = r"D:\Projects\TanitAD"
sys.path.insert(0, os.path.join(REPO, "taniteval", "adapters"))
import navsim as ad  # noqa: E402

LONG = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
        "DDC": "driving_direction_compliance", "TLC": "traffic_light_compliance",
        "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
        "LK": "lane_keeping", "HC": "history_comfort", "EC": "two_frame_extended_comfort"}

rows, results = [], {}


def check(gap: str, what: str, ok: bool, detail: str):
    results[gap] = {"closed": bool(ok), "what": what, "evidence": detail}
    rows.append(f"{gap}  {'CLOSED' if ok else 'OPEN  '}  {what}\n        {detail}")


# ---- G1: long, stage-suffixed devkit columns -> the adapter's short keys ---- #
real = sorted(glob.glob(os.path.join(
    REPO, "taniteval", "results", "bench", "navsim_v2", "*", "*", "raw", "*", "*.csv")))
row, src = None, "synthetic row (no banked devkit CSV found)"
for p in real:
    try:
        with open(p, newline="", encoding="utf-8") as f:
            r0 = next(csv.DictReader(f))
    except (StopIteration, OSError, UnicodeDecodeError):
        continue
    if any(k.endswith("_stage_one") for k in r0):
        row, src = r0, os.path.relpath(p, REPO)
        break
if row is None:
    row = {f"{l}_stage_one": "1.0" for l in LONG.values()}
    row.update({f"{l}_stage_two": "0.5" for l in LONG.values()})
    row["score"] = "0.44"

sub = ad.submetrics_from_row(row, variant="EPDMS_v2", stage="one")
missing = sub["_missing_terms"]
check("G1", "submetrics_from_row reads <long>_stage_one|_stage_two",
      not missing and sub.get("stage") == "one",
      f"source={src}; missing terms={missing}; TLC={sub.get('TLC')}")

# ---- G2-G6: the artifact keys the four blocking gates read ---------------- #
N, K, DT = 6, 8, 0.5
gt = np.zeros((N, K, 2), dtype=np.float64)
for i in range(N):
    for j in range(K):
        gt[i, j, 0] = 8.0 * DT * (j + 1)
        gt[i, j, 1] = 0.02 * (j + 1) ** 2 * (1 if i % 2 == 0 else -1)
win = ad.scenes_to_win(gt + 0.1, gt, frame="ego", origin_included=False, dt_s=DT,
                       ego_speed_mps=np.full(N, 8.0),
                       scene_tokens=[f"tok{i:04d}" for i in range(N)],
                       log_names=[f"log{i % 3}" for i in range(N)])

art = ad.build_artifact(
    win, tier="T1", epdms=0.1148, variant="EPDMS_v2", split="navhard_two_stage",
    n_boot=40,
    navsim_protocol="EPDMS_v2_navhard_two_stage", devkit_sha=ad.DEVKIT_SHA_V2,
    sensor_set="3-camera stitch cam_l0+cam_f0+cam_r0 -> 256x640 cylindrical",
    setting="perception-free, zero-shot from PhysicalAI-AV",
    ego_status_enforcement={
        "mechanism": "declared-input seam: only the declared t0 fields are copied",
        "declared_fields": ["ego_velocity[t0]", "driving_command[t0]"],
        "evidence": {"file": "raw/K5_K6_ego_mutation.json", "byte_identical": 20}},
    controls={"floor": {"name": "CV", "score": 0.1148}},
    interval={"status": "OK", "estimator": "navsim_log_cluster_bootstrap",
              "cluster_unit": "log_name", "resample_unit": "log_name",
              "aggregation": "two_stage_mapping_key_mean", "lo": 0.0825, "hi": 0.145,
              "n_clusters": 76, "n_boot": 2000})


def dig(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return False, None
        cur = cur[k]
    return True, cur


for gap, path in (("G2", "estimator.cluster_unit"),
                  ("G3", "protocol.ego_status_enforcement"),
                  ("G4", "protocol.sensor_set"),
                  ("G4b", "protocol.setting"),
                  ("G5", "protocol.navsim_protocol"),
                  ("G5b", "protocol.devkit_sha"),
                  ("G6", "protocol.corpus"),
                  ("G6b", "controls")):
    found, val = dig(art, path)
    check(gap, f"build_artifact emits `{path}`", found and val not in (None, "", {}),
          f"value={json.dumps(val, ensure_ascii=False)[:110] if found else 'ABSENT'}")

print("\n".join(rows))
open_gaps = [k for k, v in results.items() if not v["closed"]]
print(f"\n{len(results) - len(open_gaps)}/{len(results)} closed; open: {open_gaps or 'none'}")
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "raw", "e1_adapter_gaps_G1_G6.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump({"results": results, "open": open_gaps}, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("wrote", os.path.relpath(out, REPO))
raise SystemExit(1 if open_gaps else 0)
