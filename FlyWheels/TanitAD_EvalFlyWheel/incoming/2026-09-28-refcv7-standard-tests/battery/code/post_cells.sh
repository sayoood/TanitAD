#!/bin/sh
# Follow-up for the CPU cell diagnosis (2026-10-04): wait for `g0_cells_cpu_r7.py` to END (its JSON says
# finished / refused, its ABORTED marker exists, or its process is gone), then run the flip decomposition
# (`g0_cells_analyse.py`, zero GPU) and bank both artifacts into the package after a UUID scan.
#   nohup sh post_cells.sh <msys pid of the cells job> > /d/refcv7_eval_kit/battery/g0cells_step30000_cpu/post.out 2>&1 &
# Markers ZZCELLS...ZZ in <work dir>/post.log; assert on the banked JSON, never on an exit code.
JOBPID="$1"
PKG=/d/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery
PKGW='D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery'
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=/d/refcv7_eval_kit/battery/g0cells_step30000_cpu
WW='D:/refcv7_eval_kit/battery/g0cells_step30000_cpu'
export PYTHONPATH="C:/Users/Admin/ev7/stack;C:/Users/Admin/ev7/taniteval" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
LOG=$W/post.log
echo "ZZCELLSPOSTSTART $(date +%FT%T%z) watching pid $JOBPID" >> $LOG
while :; do
  if [ -s $W/cells.json.ABORTED.json ]; then echo "ZZCELLSABORTED $(date +%FT%T%z)" >> $LOG; break; fi
  if [ -s $W/cells.json ] && $PY -c "import json,sys; d=json.load(open(r'$WW/cells.json',encoding='utf-8')); sys.exit(0 if (d.get('finished') or d.get('refused')) else 1)" 2>/dev/null; then
    echo "ZZCELLSENDED $(date +%FT%T%z)" >> $LOG; break
  fi
  if ! kill -0 "$JOBPID" 2>/dev/null; then echo "ZZCELLSJOBGONE $(date +%FT%T%z)" >> $LOG; break; fi
  sleep 60
done
mkdir -p $PKG/raw/g0cells_step30000_cpu
if [ -s $W/cells.json ] && $PY -c "import json,sys; d=json.load(open(r'$WW/cells.json',encoding='utf-8')); sys.exit(0 if d.get('n_batches_done') else 1)" 2>/dev/null; then
  $PY "$PKGW/code/g0_cells_analyse.py" --cells "$WW/cells.json" --g0 D:/refcv7_eval_kit/battery/step30000/g0.json \
      --out "$WW/analysis.json" >> $LOG 2>&1
  [ -s $W/analysis.json ] && echo "ZZCELLSANALYSED $(date +%FT%T%z)" >> $LOG || echo "ZZCELLSANALYSISNOJSON" >> $LOG
fi
for f in $W/cells.json $W/analysis.json $W/cells.log $W/cells.json.ABORTED.json $LOG; do
  [ -s "$f" ] || continue
  if grep -Eqi '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' "$f"; then
    echo "ZZCELLSBANKREFUSED $f (UUID-shaped token)" >> $LOG
  else
    cp "$f" $PKG/raw/g0cells_step30000_cpu/
  fi
done
echo "ZZCELLSPOSTEND $(date +%FT%T%z)" >> $LOG
cp $LOG $PKG/raw/g0cells_step30000_cpu/ 2>/dev/null
