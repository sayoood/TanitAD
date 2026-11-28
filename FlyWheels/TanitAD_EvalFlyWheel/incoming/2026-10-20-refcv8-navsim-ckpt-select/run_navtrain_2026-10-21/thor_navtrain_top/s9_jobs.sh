#!/bin/bash
# S9 prep: the seven scorer jobs from make_score_jobs.py, then the SAME argv with the profile roots re-pointed under /home/nvidia/r8ns
# (brief: Thor writes only under /home/nvidia/r8ns; the runbook's thor profile defaults to /dev/shm). Nothing else in the argv changes.
set -u
N=/home/nvidia/r8ns/navtrain; OUT=/home/nvidia/r8ns/navtrain_sel; X=/home/nvidia/r8ns/code/navtrain
cd $X && python3 $X/make_score_jobs.py --layout-root $OUT --sample $N/NAVTRAIN_SAMPLE_TOKENS.json --driver $X/navtrain_cache_score.py --out $OUT/jobs.orig.jsonl
python3 - "$OUT/jobs.orig.jsonl" "$OUT/jobs.jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
extra = ["--exp", "/home/nvidia/r8ns/navtrain/nsexp/r8sel_navtrain", "--data", "/home/nvidia/r8ns/navtrain/nsdata/openscene",
         "--maps", "/home/nvidia/r8ns/navtrain/nsdata/maps"]
n = 0
with open(dst, "w") as fo:
    for ln in open(src):
        if not ln.strip():
            continue
        j = json.loads(ln); j["argv"] = j["argv"] + extra; fo.write(json.dumps(j) + "\n"); n += 1
print(f"ZZJOBSPATCHED n={n}")
PY
