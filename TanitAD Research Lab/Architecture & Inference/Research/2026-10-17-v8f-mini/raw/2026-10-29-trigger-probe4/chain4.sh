#!/bin/bash
# chain4.sh -- the real-data chain of trigger probe 4 (PREREG_FROZEN.md).  Detached; resumable (every arm-seed / control is banked as it finishes; re-running skips finished work).
# 1. waits for ALL FIVE probe-3 markers (the brief: no real-data fit while probe 3 runs)  2. RAM guard >= 4 GB free  3. per-population workers in parallel (decision arms -> controls -> R-MAX beside arms)
# 4. strata + attention maps, eval-141, tables, pytest, manifest.   Logs: raw/chain4.log (START/END/rc per step), raw/<step>.log.   Completion marker: raw/CHAIN4.done (content = failed steps, empty = none).
export PATH=/usr/bin:/bin:$PATH
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
cd /e/Projects/TanitAD-artifacts/dagland/trigger_probe4 || exit 1
P3=/e/Projects/TanitAD-artifacts/dagland/trigger_probe3/raw
log() { echo "$(date -Iseconds) $*" >> raw/chain4.log; }
run() { name=$1; shift; log "START $name"; "$PY" -I "$@" >> raw/$name.log 2>&1; rc=$?; log "END $name rc=$rc"; [ $rc -ne 0 ] && echo "$name" >> raw/.failed_steps; return $rc; }
rm -f raw/.failed_steps
log "CHAIN4 started; waiting for probe-3 markers"
n=0
until [ -f $P3/CHAIN_c2band.done ] && [ -f $P3/CHAIN_mlpctl.done ] && [ -f $P3/CHAIN_evalmlp.done ] && [ -f $P3/CHAIN_mutate.done ] && [ -f $P3/CHAIN_pf.done ]; do sleep 30; n=$((n+1))
  if [ $n -gt 4320 ]; then log "GAVE UP: probe-3 markers still missing after 36 h: $(ls $P3/*.done 2>/dev/null | xargs -n1 basename | tr '
' ' ')"; echo waiting-timeout > raw/CHAIN4.done; exit 1; fi
done

log "all five probe-3 markers present"
run ramguard p4_run.py --stage ramguard || { log "RAM guard failed"; echo "ramguard" > raw/CHAIN4.done; exit 1; }
run check p4_run.py --stage check
worker() {
  pop=$1
  export P4_THREADS=3
  run arms_dec_$pop p4_run.py --stage arms --pop $pop --arms G8att,G4att
  run controls_$pop p4_run.py --stage controls --pop $pop
  run arms_max_$pop p4_run.py --stage arms --pop $pop --arms G8max,G4max
}
worker pstop &
worker pmove &
wait
export P4_THREADS=6
for pop in pstop pmove; do run post_$pop p4_run.py --stage post --pop $pop; done
for pop in pstop pmove; do run eval141_$pop p4_run.py --stage eval141 --pop $pop; done
run tables p4_run.py --stage tables
run pytest -m pytest -q -p no:cacheprovider test_trigger_probe4.py
cp raw/pytest.log raw/pytest_output_final.txt 2>/dev/null
run manifest make_manifest.py
log "CHAIN4-END"
cat raw/.failed_steps > raw/CHAIN4.done 2>/dev/null || : > raw/CHAIN4.done
