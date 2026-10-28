#!/usr/bin/env bash
# Lane A of the reader's checks, run sequentially on the FROZEN reader (dag_orc_read.py): selftest -> dry-run -> K-RC leg 2 (DAG-OACC) -> smoke (stand-ins) -> K-RC leg 1 (DAG2).  Each step logs to logs/ and the lane stops on the first failure.
# CPU only: CUDA_VISIBLE_DEVICES=-1.  Markers: ZZLANEA-DONE | ZZLANEA-STOP <step>.
cd "$(dirname "$0")" || exit 1
export CUDA_VISIBLE_DEVICES=-1
PYV=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
T0=$(date +%s)
step() {   # step <name> <command...>
  local n=$1; shift
  echo "$(date '+%H:%M:%S') START $n"
  "$@" > logs/final_$n.out 2>&1; local rc=$?
  echo "$(date '+%H:%M:%S') END $n rc=$rc ($(( $(date +%s) - T0 )) s since the lane started)"
  [ $rc = 0 ] || { echo "ZZLANEA-STOP $n rc=$rc"; exit $rc; }
}
step selftest $PYV -I dag_orc_selftest.py --quiet
step dry_run $PYV dag_orc_read.py --dry-run --out DRY_RUN.json
step rc_oacc $PYV dag_orc_read.py --rc-oacc --out RC_OACC.json
step smoke $PYV dag_orc_read.py --smoke-standins --out SMOKE.json
step rc_dag2 $PYV dag_orc_read.py --rc-dag2 --out RC_DAG2.json
echo "ZZLANEA-DONE"
