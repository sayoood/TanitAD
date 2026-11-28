#!/usr/bin/env bash
# LIVE SWITCH of the REFe run to the paper's scorer supervision (PI decision 2026-09-26: "B, the paper version"
# + "add NAVSIM-faithful drivable area labels and perform the switching of the trainer").
# The procedure of memory `live-run-switch-procedure` (2026-09-02): pod-side, setsid nohup, every step logged;
# kill the SUPERVISOR first by explicit, cmdline-verified PID; the trainer (TERM, 90 s, KILL); its loader workers;
# swap code only after the old trainer is dead (keep .PRE_onpolicy copies); start the v2 supervisor (fresh lock,
# the live launch line + OP_FLAGS); verify on the trainer's OWN lines.
#   setsid nohup bash switch_onpolicy.sh <wrapper pid> <supervisor pid> <trainer pid> > /workspace/switch_onpolicy.out 2>&1 &
set -u
WRAP=$1; SUP=$2; TR=$3
RUN=/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3
PK=/workspace/refe-plan
LOG=/workspace/switch_onpolicy.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
chk() { tr "\0" " " < "/proc/$1/cmdline" 2>/dev/null | grep -q -- "$2"; }
say "switcher up (pid $$): wrapper=$WRAP supervisor=$SUP trainer=$TR"
chk "$WRAP" "wait_then_train.sh" || { say "ABORT: $WRAP is not wait_then_train.sh"; exit 1; }
chk "$SUP" "refe-plan/code/pod_train.sh" || { say "ABORT: $SUP is not pod_train.sh"; exit 1; }
chk "$TR" "train.py --backbone vitl16" || { say "ABORT: $TR is not the trainer"; exit 1; }
for f in refe/train.py.NEW refe/model.py.NEW code/pod_train_v2.sh; do
  [ -s "$PK/$f" ] || { say "ABORT: missing $PK/$f"; exit 1; }
done
WK=$(ps -eo pid=,ppid= | awk -v p="$TR" '$2==p {print $1}' | tr '\n' ' ')
say "trainer's loader workers: $WK"
# 1) a FRESH checkpoint: written after this script started, size and mtime stable for 20 s -- or, with
#    NOW=1, the current one if it is younger than NOW_MAX_S (the resume recomputes the steps since it;
#    worth it when an epoch boundary is near: the next epoch then reads the on-policy sets)
T0=$(date +%s)
if [ "${NOW:-0}" = 1 ]; then
  age=$(( T0 - $(stat -c %Y "$RUN/ckpt_last.pt") ))
  [ "$age" -le "${NOW_MAX_S:-900}" ] && T0=0 && say "NOW mode: current checkpoint is ${age} s old -- switching on it"
fi
while :; do
  m=$(stat -c %Y "$RUN/ckpt_last.pt")
  if [ "$m" -gt "$T0" ]; then
    s1=$(stat -c %s "$RUN/ckpt_last.pt"); sleep 20
    s2=$(stat -c %s "$RUN/ckpt_last.pt"); m2=$(stat -c %Y "$RUN/ckpt_last.pt")
    [ "$s1" = "$s2" ] && [ "$m" = "$m2" ] && break
  fi
  sleep 10
done
say "fresh checkpoint: $(stat -c '%y %s' "$RUN/ckpt_last.pt"); last step line: $(grep -E '^ *step' "$RUN/train.log" | tail -1 | cut -c1-60)"
# 2) the wrapper and the SUPERVISOR first (else its loop relaunches the OLD line)
kill "$WRAP" "$SUP" 2>/dev/null; sleep 3
chk "$SUP" "pod_train.sh" && { kill -9 "$SUP"; say "supervisor needed KILL"; }
chk "$WRAP" "wait_then_train.sh" && { kill -9 "$WRAP"; say "wrapper needed KILL"; }
say "supervisor + wrapper stopped"
# 3) the trainer: TERM, up to 90 s, KILL; then its (orphaned) loader workers, cmdline-verified
kill "$TR" 2>/dev/null
for i in $(seq 1 90); do kill -0 "$TR" 2>/dev/null || break; sleep 1; done
kill -0 "$TR" 2>/dev/null && { kill -9 "$TR"; say "trainer needed KILL"; }
for w in $WK; do chk "$w" "train.py --backbone vitl16" && kill -9 "$w" && say "killed loader worker $w"; done
sleep 2
N=$(ps -eo args | grep -c "[t]rain.py --backbone vitl16")
[ "$N" = 0 ] || { say "ABORT: $N trainer processes still alive -- NOT starting v2"; exit 1; }
say "old trainer gone"
# 4) swap the code, keeping the exact pre-switch files
cd "$PK/refe" || exit 1
cp -p train.py train.py.PRE_onpolicy && cp -p model.py model.py.PRE_onpolicy
mv train.py.NEW train.py && mv model.py.NEW model.py
say "code swapped: $(md5sum train.py model.py | awk '{print substr($1,1,12), $2}' | tr '\n' ' ')"
# 5) the v2 supervisor: STAGES=train reuses batch_accum.txt; fresh lock; setsid so it outlives this script
cd /workspace || exit 1
setsid nohup env GROW=1 THREADS=2 RUN="$RUN" STAGES=train bash refe-plan/code/pod_train_v2.sh \
  >> /workspace/pod_train.out 2>&1 < /dev/null &
say "v2 supervisor started (pid $!)"
# 6) evidence from the trainer's own lines (the resume takes a few minutes: bank + compile)
OK=0
for i in $(seq 1 90); do
  if grep -q "DECLARED RECIPE CHANGE" "$RUN/train.log" && grep -q "SCORER: ON-POLICY" "$RUN/train.log"; then OK=1; break; fi
  grep -q "REFUSING TO RESUME" <(tail -30 "$RUN/train.log") && break
  sleep 10
done
say "evidence: $(grep -E 'SCORER: ON-POLICY|ON-POLICY scorer bank|RESUMED|DECLARED RECIPE CHANGE|REFUSING' "$RUN/train.log" | tail -5 | tr '\n' '|' | cut -c1-900)"
NEWTR=$(ps -eo pid,args | grep "[t]rain.py --backbone vitl16" | grep -- "--scorer-mode onpolicy" | head -1 | awk '{print $1}')
say "new trainer pid ${NEWTR:-NONE}; onpolicy flags on its cmdline: $(tr '\0' ' ' < /proc/${NEWTR:-0}/cmdline 2>/dev/null | grep -c -- '--scorer-mode onpolicy')"
[ "$OK" = 1 ] && say "ZZSWITCH_OK" || say "ZZSWITCH_CHECK_MANUALLY"
