#!/usr/bin/env python3
"""Exact JPEG sizes per camera from the LOCAL OpenScene-extracted navtest frame zips (central directories only; 48,584 frames, 136 logs, 4 cameras).
Used to turn frame counts into bytes for the 'frames only' floor. Writes raw/jpg_sizes_local.json."""
import os, zipfile, json, collections, statistics
A = "D:/Projects/TanitAD/data/refe_navtest/frames/"
by = collections.defaultdict(list)
for f in sorted(os.listdir(A)):
    if f.endswith(".zip"):
        cam = f.rsplit("_CAM_", 1)[1][:-4]
        with zipfile.ZipFile(A + f) as z:
            by["CAM_" + cam].extend(i.file_size for i in z.infolist() if not i.is_dir())
out = {c: {"n": len(v), "mean": round(sum(v) / len(v)), "median": int(statistics.median(v)), "p05": sorted(v)[int(.05 * len(v))], "p95": sorted(v)[int(.95 * len(v))], "total": sum(v)} for c, v in by.items()}
out["all4"] = {"n": sum(o["n"] for o in out.values()), "mean_per_frame_4cams": round(sum(o["total"] for o in out.values()) / (sum(o["n"] for o in out.values()) / 4))}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "jpg_sizes_local.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
