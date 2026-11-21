#!/usr/bin/env bash
# navhard_two_stage at 416x1024 (refcv6's frame), t0 frame only (--keep 1): stage 2 (5,462
# synthetic renders) then stage 1 (450 ORIGINAL scenes, jpgs from the verified extraction root
# C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920, VERIFY.txt = COMPLETE 3375/3375).
set -u
P="$(cd "$(dirname "$0")/.." && pwd)"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="C:/Users/Admin/ev6/stack;C:/Users/Admin/ev6/taniteval" OMP_NUM_THREADS=4
B=C:/Users/Admin/tanitad-caches/refcv6-navsim-20260923
"$PY" "$P/code/frames416.py" --split navhard_two_stage --stage 2 --keep 1 \
   --data-root C:/Users/Admin/navsim-crun/data/openscene --out "$B/navhard_s2" > "$B/navhard_s2.log" 2>&1
echo "S2 rc=$? $(date -u +%FT%TZ)" >> "$B/navhard_build.done"
"$PY" "$P/code/frames416.py" --split navhard_two_stage --stage 1 --keep 1 \
   --data-root C:/Users/Admin/navsim-crun/data/openscene \
   --inputs C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navhard_two_stage/navsim_agent_inputs.json \
   --stage1-sensors C:/Users/Admin/tanitad-caches/navhard-stage1-frames-20260920 \
   --out "$B/navhard_s1" > "$B/navhard_s1.log" 2>&1
echo "S1 rc=$? $(date -u +%FT%TZ)" >> "$B/navhard_build.done"
echo "ZZNAVHARDFRAMESDONEZZ" >> "$B/navhard_build.done"
