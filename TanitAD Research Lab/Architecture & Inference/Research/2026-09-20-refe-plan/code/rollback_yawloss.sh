#!/usr/bin/env bash
# ROLLBACK of code/switch_yawloss.sh: stop the v3 supervisor and its trainer, restore the .PRE_yawloss files, restart
# the v2 supervisor (the live line before the switch). The run resumes from ckpt_last.pt as after any restart.
#   setsid nohup bash rollback_yawloss.sh > /workspace/rollback_yawloss.out 2>&1 &
set -u
RUN=/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3
PK=/workspace/refe-plan
LOG=/workspace/switch_yawloss.log
say() { echo "$(date -u +%FT%TZ) ROLLBACK $*" >> "$LOG"; }
chk() { tr "\0" " " < "/proc/$1/cmdline" 2>/dev/null | grep -q -- "$2"; }
[ -s "$PK/refe/train.py.PRE_yawloss" ] && [ -s "$PK/refe/model.py.PRE_yawloss" ] || { say "ABORT: no .PRE_yawloss files"; exit 1; }
for p in $(ps -eo pid=,args= | grep "[p]od_train_v3.sh" | awk '{print $1}'); do chk "$p" "pod_train_v3.sh" && kill "$p"; done
sleep 3
for p in $(ps -eo pid=,args= | grep "[p]od_train_v3.sh" | awk '{print $1}'); do chk "$p" "pod_train_v3.sh" && kill -9 "$p"; done
for p in $(ps -eo pid=,args= | grep "[t]rain.py --backbone vitl16" | awk '{print $1}'); do chk "$p" "train.py --backbone vitl16" && kill "$p"; done
for i in $(seq 1 90); do [ "$(ps -eo args | grep -c '[t]rain.py --backbone vitl16')" = 0 ] && break; sleep 1; done
for p in $(ps -eo pid=,args= | grep "[t]rain.py --backbone vitl16" | awk '{print $1}'); do chk "$p" "train.py --backbone vitl16" && kill -9 "$p"; done
sleep 2
[ "$(ps -eo args | grep -c '[t]rain.py --backbone vitl16')" = 0 ] || { say "ABORT: trainer still alive"; exit 1; }
cd "$PK/refe" || exit 1
cp -p train.py train.py.FAILED_yawloss; cp -p model.py model.py.FAILED_yawloss
cp -p train.py.PRE_yawloss train.py && cp -p model.py.PRE_yawloss model.py
say "restored: $(md5sum train.py model.py | awk '{print substr($1,1,12), $2}' | tr '\n' ' ')"
cd /workspace || exit 1
setsid nohup env GROW=1 THREADS=2 RUN="$RUN" STAGES=train bash refe-plan/code/pod_train_v2.sh \
  >> /workspace/pod_train.out 2>&1 < /dev/null &
say "v2 supervisor restarted (pid $!)"
