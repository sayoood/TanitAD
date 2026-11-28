#!/usr/bin/env bash
# C2 planted positive (AMENDMENT 7 sec. 4).  Run AFTER the main fit plan finished.  CPU only, 2 threads.  Last line: ZZ C2 DONE ZZ  or  ZZ C2 FAIL ZZ <why>
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8
PY=${PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}
BASE=${BASE:-F:/TanitAD-scratch/q5_416/run}
H=F:/TanitAD-scratch/q5_416/build/c2
MINFREE_KB=$((26*1000*1000))
fail() { echo "ZZ C2 FAIL ZZ $*"; exit 1; }
disk() { local kb; kb=$(df -k F:/ | tail -1 | awk '{print $4}'); [ "${kb:-0}" -ge "$MINFREE_KB" ] || fail "disk (F: free ${kb} KB, need >= ${MINFREE_KB} KB before $1)"; }
step() { disk "$1"; shift; "$@" || fail "$*"; }
# real inputs must exist (the main plan's work/ dirs)
for S in S1 S2; do
  [ -f "$BASE/work/${S}_416/fit_E_linear.npy" ] && [ -f "$BASE/work/${S}_256p/fit_V40+E_mlp.npy" ] || fail "real fits missing for $S under $BASE/work (main plan not finished)"
done
cd "$H" || fail "cd $H"
step make    "$PY" q5_c2_planted.py make --base "$BASE" --samples S1,S2
step prep    "$PY" q5_c2_planted.py prep --base "$BASE" --samples S1,S2
for S in S1 S2; do for M in linear mlp; do
  step "fit $S $M" "$PY" q5_c2_planted.py fit --base "$BASE" --sample "$S" --model "$M"
done; done
step read    "$PY" q5_c2_planted.py read --base "$BASE" --samples S1,S2
echo "RESULT: $BASE/c2/RESULT_C2.json  (feed to: q5_fill_controls.py --c2)"
echo "ZZ C2 DONE ZZ"
