#!/bin/bash
# run_a2_fits.sh -- the 9 stage-2 fits (R9PF / R9PF-CONSTZ / R9PF-SHUF x fit seeds 0/1/2), at most 6 concurrent, OMP_NUM_THREADS=4 each. CPU only.
# Judged on ARTIFACTS (the head file + its _FIT.json), never on an exit code through a pipe.  Markers ZZA2<..>ZZ in logs/run_a2.log.
set -u
W=C:/Users/Admin/refcv8_pf/refcv9_r9pf
source $W/code/env_r9pf.sh
LOG=$W/logs/run_a2.log
mk() { echo "ZZA2$1ZZ $(date +%FT%T%z) $2" >> "$LOG"; }
mk START "9 fits"
jobs_list=""
for S in 0 1 2; do for A in R9PF R9PF-CONSTZ R9PF-SHUF; do jobs_list="$jobs_list $A:$S"; done; done
for J in $jobs_list; do
  A=${J%%:*}; S=${J##*:}
  while [ "$(jobs -rp | wc -l)" -ge 6 ]; do sleep 5; done
  ( "$PY" -X utf8 "$W/code/a2_fit.py" --arm "$A" --fit-seed "$S" > "$W/logs/a2_fit_${A}_fs${S}.log" 2>&1
    if [ -s "$W/heads/${A}_fs${S}.pt" ] && [ -s "F:/refcv9_r9pf_scratch/fits/${A}_fs${S}_FIT.json" ]; then mk FITOK "${A}_fs${S}"; else mk FITFAIL "${A}_fs${S}"; fi ) &
done
wait
n=$(ls $W/heads/*.pt 2>/dev/null | wc -l)
mk END "heads present: $n"
