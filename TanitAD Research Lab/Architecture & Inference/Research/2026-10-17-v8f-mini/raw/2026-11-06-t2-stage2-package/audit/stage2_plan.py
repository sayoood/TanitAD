"""AGENT F stage 2: starts per card section 3 + 6b (launch scenes) and the byte budget for the 42 launch scenes. CPU only; reads the screening table and the download manifest.
Usage: python -I stage2_plan.py OUT.json"""
import json
import math
import os
import sys
from collections import defaultdict

ART = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2_scenes/raw"
st = json.load(open(f"{ART}/SCREEN_TABLE.json"))
dm = json.load(open(f"{ART}/DOWNLOAD_MANIFEST.json"))["files"]
MIN_F0 = 18                                           # card 6b: f0 >= 18 logged frames of history (policy.window 17 + 2 - 1)
launch = [s for s in st["scenes"] if s.get("selected") and str(s.get("slot", "")).startswith("launch")]
assert len(launch) == 42, len(launch)
per_scene = defaultdict(lambda: {"bytes": 0, "alloc_1MiB": 0, "files": []})
for k, v in dm.items():
    p = per_scene[v["scene"]]
    p["bytes"] += v["bytes"]
    p["alloc_1MiB"] += math.ceil(v["bytes"] / (1 << 20)) * (1 << 20)
    p["files"].append({"member": v["member"], "bytes": v["bytes"], "sha256": v["sha256"], "location": v["location"]})
out = {"rule": "start = max(round(10*(t_c - d)), 18) for d in (4.0, 2.0); equal starts collapse; a start must satisfy f0 + 60 < n_ticks (the dispatcher refuses otherwise)", "scenes": []}
n_starts, drop = 0, []
for s in launch:
    t_c = s["launch_events"][0]["t_c_s"]
    n = s["n_ticks"]
    starts = []
    for d in (4.0, 2.0):
        f = max(int(round(10 * (t_c - d))), MIN_F0)
        if f not in starts:
            starts.append(f)
    ok = [f for f in starts if f + 60 < n]
    if len(ok) < len(starts):
        drop.append((s["scene"], starts, n))
    n_starts += len(ok)
    pb = per_scene.get(s["scene"], {"bytes": 0, "alloc_1MiB": 0, "files": []})
    out["scenes"].append({"scene": s["scene"], "slot": s["slot"], "t_c_s": t_c, "n_ticks": n, "stride": s.get("stride"), "starts_raw": starts, "starts": ok, "lead": s["launch_events"][0]["lead"],
                          "bytes": pb["bytes"], "alloc_1MiB": pb["alloc_1MiB"], "n_files": len(pb["files"]), "files": pb["files"]})
tot = sum(x["bytes"] for x in out["scenes"])
alc = sum(x["alloc_1MiB"] for x in out["scenes"])
out["totals"] = {"n_scenes": len(launch), "n_starts": n_starts, "starts_dropped_by_6s_rule": drop, "bytes": tot, "GB": tot / 1e9, "GiB": tot / 2**30, "alloc_1MiB_GiB": alc / 2**30,
                 "scenes_with_no_manifest_entry": [x["scene"] for x in out["scenes"] if x["n_files"] == 0]}
json.dump(out, open(sys.argv[1], "w", encoding="utf-8", newline="\n"), indent=1)
print(json.dumps(out["totals"], indent=1))
by = defaultdict(int)
for x in out["scenes"]:
    for f in x["files"]:
        by[f["member"]] += f["bytes"]
print("bytes by member (GB):", {k: round(v / 1e9, 2) for k, v in by.items()})
print("starts per scene histogram:", {n: sum(1 for x in out["scenes"] if len(x["starts"]) == n) for n in (0, 1, 2)})
print("the 2 stage-1 scenes:", [(x["scene"][:8], x["starts_raw"], x["starts"], round(x["bytes"] / 1e9, 3)) for x in out["scenes"] if x["scene"][:8] in ("7c72937c", "00040136")])
