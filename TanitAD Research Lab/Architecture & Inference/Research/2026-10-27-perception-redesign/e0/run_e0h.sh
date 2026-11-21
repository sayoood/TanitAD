#!/bin/bash
# run_e0h.sh -- E0h chain: 10 runs, 4 at a time (24-core dev box, OMP_NUM_THREADS=6 per run).  Marker on completion: ZZE0Hdone-<n_ok>-<n_total>ZZ in F:/r9e0/logs/e0h_chain.log
# Resume = re-run this script: a run whose MAPPATH_RESULT.json exists is skipped (the ARTIFACT is the evidence, not an exit code).
export TMP=F:/c_overflow_2026-10-26/tmp OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8
HERE="$(cd "$(dirname "$0")" && pwd)"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
R=F:/r9e0/e0h/runs
mkdir -p $R /f/r9e0/logs
run1 () {
  tag=$1; shift
  [ -s $R/$tag/MAPPATH_RESULT.json ] && return 0
  mkdir -p $R/$tag
  $PY -X utf8 "$HERE/e0h_dose.py" "$@" --out $R/$tag > /f/r9e0/logs/e0h_$tag.log 2>&1 < /dev/null
}
export -f run1; export R PY HERE
cat <<'LIST' | xargs -P 4 -L 1 bash -c 'run1 "$@"' _
p1 --mode p1
all --mode all
p0.50 --mode dose --p 0.5
p0.70 --mode dose --p 0.7
p0.60 --mode dose --p 0.6
p0.80 --mode dose --p 0.8
p0.90 --mode dose --p 0.9
p0.95 --mode dose --p 0.95
p0.70_stamp0.5 --mode dose --p 0.7 --stamp-m 0.5
p0.70_stamp2.0 --mode dose --p 0.7 --stamp-m 2.0
LIST
ok=$(ls $R/*/MAPPATH_RESULT.json 2>/dev/null | wc -l)
echo "ZZE0Hdone-${ok}-10ZZ" >> /f/r9e0/logs/e0h_chain.log
