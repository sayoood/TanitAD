#!/bin/bash
# P6 lane-rank twin bank 3-C: the FULL re-roll (44,409 twins, one 10 s rollout each) on the pod CPUs.  PREREG_3C.md + addendum1.
#   bash run_full.sh preflight     checks only (nothing started)
#   bash run_full.sh launch        preflight, then DETACHES the worker (setsid -f nohup ... < /dev/null); returns at once
#   bash run_full.sh worker        (internal) 720 shards through `xargs -P $NPROC`, a progress log, the first-5-min ETA, a DONE marker
# PREFLIGHT REFUSES unless: the pod is not training (no GPU compute process) | the identity controls passed (controls/controls.json all_pass) | the fixed-tree twin
# parts are complete (PARTS_DONE + every twin's file present) | the input bank is the one the controls were pinned to | the output dir does not exist yet.
# Resume: re-run `launch`; finished shards (DONE marker) are skipped, partial shards resume from their audit file.
set -u
source /workspace/ops/bank3c/code/env_3c.sh
W=${W:-/workspace/ops/bank3c/full}                              # scratch: shards, logs, progress  (W / BANK / MC_* / EXPECT / OUTBANK are overridable ONLY for the dress rehearsal)
BANK=${BANK:-/workspace/data/refe_navtrain/train_grow_cmd}      # INPUT -- never modified
OUTBANK=${OUTBANK:-/workspace/data/refe_navtrain/train_grow_cmd_3c}   # created by `merge`
MC_TWIN=${MC_TWIN:-/workspace/data/refe_navtrain/metric_cache_twins_3c}
MC_LOGGED=${MC_LOGGED:-/workspace/data/refe_navtrain/metric_cache}
EXPECT=${EXPECT:-44409}                                          # lane-rank twins the input must hold
# ⚠ THE POD CONTAINER IS CAPPED AT 11.05 CPUs (cgroup cfs_quota 1,105,000 / period 100,000 us; `nproc` still says 96). MEASURED 2026-10-17: the same 74-twin control at 10 procs
# (probe) = 13.4 s mean per rollout, at 32 procs = 49 s: throughput does not grow past ~11 processes. 12 = quota + 1 for I/O wait. The brief's "~90 processes" buys nothing.
NPROC=${NPROC:-12}; NSHARDS=${NSHARDS:-720}; CHUNK=32
mkdir -p $W/logs $W/shards

preflight() {
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ] || { echo "REFUSING: a GPU compute process is running (training?)"; return 2; }
  grep -q '"all_pass": true' ${CONTROLS_JSON:-/workspace/ops/bank3c/controls/controls.json} 2>/dev/null || { echo "REFUSING: controls.json missing or not all_pass (3C-INVALID)"; return 2; }
  [ -f $W/PARTS_DONE ] || { echo "REFUSING: $W/PARTS_DONE missing -- run build_twin_parts_3c.sh first (the unfixed parts in metric_cache_twins_rejected are unusable)"; return 2; }
  [ ! -e $OUTBANK ] || { echo "REFUSING: $OUTBANK already exists"; return 2; }
  $PY - <<PYEOF || return 2
import json, sys
sys.path.insert(0, "$PKG/refe")
import online_labeller as OL
miss = n = 0
for ln in open("$BANK/targets_rank1.jsonl", encoding="utf-8"):
    if '"lane_rank"' not in ln: continue
    r = json.loads(ln); a = r["aug"]
    if a.get("kind") != "lane_rank": continue
    n += 1
    if not OL.parts_path("$MC_TWIN", r["log_name"], r["token"], a).is_file() or not OL.parts_path("$MC_LOGGED", r["log_name"], r["token"], None).is_file(): miss += 1
print("parts coverage:", n, "twins,", miss, "missing")
sys.exit(0 if (n == $EXPECT and miss == 0) else 3)
PYEOF
  echo "PREFLIGHT_OK"
}

case "${1:-}" in
  preflight) preflight; exit $? ;;
  launch)
    preflight || exit $?
    setsid -f nohup bash "$0" worker < /dev/null > $W/worker.out 2>&1
    echo "launched; progress:  python3 /workspace/ops/bank3c/code/watch_3c.py --out $W --started-epoch \$(cat $W/started_epoch)"; exit 0 ;;
  worker)
    [ -f $W/started_epoch ] || date -u +%s > $W/started_epoch
    EP=$(cat $W/started_epoch)
    ( # progress log every 120 s + the ETA after the first 5 minutes of THIS launch
      T0=$(date +%s)
      while [ ! -f $W/RUN_DONE ] && [ ! -f $W/RUN_INCOMPLETE ]; do
        python3 $CODE3C/watch_3c.py --out $W --total $EXPECT --started-epoch $EP >> $W/progress.log 2>&1
        if [ ! -f $W/ETA.txt ] && [ $(( $(date +%s) - T0 )) -ge 300 ]; then tail -n 1 $W/progress.log > $W/ETA.txt; fi
        sleep 120
      done ) &
    cd $PKG/refe
    export MC_TWIN MC_LOGGED W BANK NSHARDS CHUNK EXPECT
    seq 0 $((NSHARDS-1)) | xargs -P $NPROC -I{} bash -c '
      source /workspace/ops/bank3c/code/env_3c.sh; cd $PKG/refe
      i={}; s=$(printf "s%04d" $i)
      [ -f $W/shards/$s/DONE ] && exit 0
      run3c $PY $CODE3C/reroll_bank_3c.py run --bank $BANK --rank0 $RANK0 --out $W/shards/$s --mc $MC_LOGGED --mc-twin $MC_TWIN --mc-logged $MC_LOGGED \
        --log-shard $i/$NSHARDS --chunk $CHUNK --code-dir $CODE3C --refe-dir $PKG/refe --pkg-code-dir $PKG/code >> $W/logs/$s.log 2>&1'
    nd=$(ls $W/shards/*/DONE 2>/dev/null | wc -l)
    if [ "$nd" = "$NSHARDS" ]; then echo "ZZ3C_RUN_DONE shards=$nd" > $W/RUN_DONE; else echo "shards done $nd of $NSHARDS -- re-run launch to resume" > $W/RUN_INCOMPLETE; fi
    python3 $CODE3C/watch_3c.py --out $W --total $EXPECT --started-epoch $EP >> $W/progress.log 2>&1
    ;;
  *) echo "usage: run_full.sh preflight|launch"; exit 64 ;;
esac
