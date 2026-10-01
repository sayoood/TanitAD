#!/usr/bin/env bash
# ROLLBACK of code/switch_tangent.sh: stop the v3t supervisor and its trainer, restore the .PRE_tangent files, restart
# the supervisor that was live before the switch (refe/.PREV_SUPERVISOR_tangent: pod_train_v2.sh or pod_train_v3.sh).
# The run resumes from ckpt_last.pt as after any restart. code/rollback_yawloss.sh's procedure, line for line.
#   setsid nohup bash rollback_tangent.sh > /workspace/rollback_tangent.out 2>&1 &
set -u
RUN=/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3
PK=/workspace/refe-plan
LOG=/workspace/switch_tangent.log
say() { echo "$(date -u +%FT%TZ) ROLLBACK $*" >> "$LOG"; }
chk() { tr "\0" " " < "/proc/$1/cmdline" 2>/dev/null | grep -q -- "$2"; }
[ -s "$PK/refe/train.py.PRE_tangent" ] && [ -s "$PK/refe/model.py.PRE_tangent" ] || { say "ABORT: no .PRE_tangent files"; exit 1; }
PREV=$(cat "$PK/refe/.PREV_SUPERVISOR_tangent" 2>/dev/null)
case "$PREV" in pod_train_v2.sh|pod_train_v3.sh) ;; *) say "ABORT: no recorded previous supervisor ('$PREV')"; exit 1;; esac
for p in $(ps -eo pid=,args= | grep "[p]od_train_v3t.sh" | awk '{print $1}'); do chk "$p" "pod_train_v3t.sh" && kill "$p"; done
sleep 3
for p in $(ps -eo pid=,args= | grep "[p]od_train_v3t.sh" | awk '{print $1}'); do chk "$p" "pod_train_v3t.sh" && kill -9 "$p"; done
for p in $(ps -eo pid=,args= | grep "[t]rain.py --backbone vitl16" | awk '{print $1}'); do chk "$p" "train.py --backbone vitl16" && kill "$p"; done
for i in $(seq 1 90); do [ "$(ps -eo args | grep -c '[t]rain.py --backbone vitl16')" = 0 ] && break; sleep 1; done
for p in $(ps -eo pid=,args= | grep "[t]rain.py --backbone vitl16" | awk '{print $1}'); do chk "$p" "train.py --backbone vitl16" && kill -9 "$p"; done
sleep 2
[ "$(ps -eo args | grep -c '[t]rain.py --backbone vitl16')" = 0 ] || { say "ABORT: trainer still alive"; exit 1; }
cd "$PK/refe" || exit 1
cp -p train.py train.py.FAILED_tangent; cp -p model.py model.py.FAILED_tangent; cp -p measures.py measures.py.FAILED_tangent
cp -p train.py.PRE_tangent train.py && cp -p model.py.PRE_tangent model.py || { say "ABORT: restore failed"; exit 1; }
if [ -e measures.py.PRE_tangent ]; then cp -p measures.py.PRE_tangent measures.py; fi
say "restored: $(md5sum train.py model.py | awk '{print substr($1,1,12), $2}' | tr '\n' ' ')"
cd /workspace || exit 1
setsid nohup env GROW=1 THREADS=2 RUN="$RUN" STAGES=train bash "refe-plan/code/$PREV" \
  >> /workspace/pod_train.out 2>&1 < /dev/null &
say "$PREV supervisor restarted (pid $!)"
