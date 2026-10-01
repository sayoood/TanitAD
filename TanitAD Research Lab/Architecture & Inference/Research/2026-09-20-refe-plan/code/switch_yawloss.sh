#!/usr/bin/env bash
# LIVE SWITCH of the REFe run to the plain heading loss (M6, `--yaw-loss plain`), run ONLY after PREREG_M6_YAWLOSS
# reads ADOPT and on the PI's go (PI, 2026-09-27: "run the test now and in positive case pause and resume with the fix
# of heading"). Same procedure as code/switch_onpolicy.sh (2026-09-26): pod-side, setsid nohup, every step logged; a
# FRESH checkpoint first; the SUPERVISOR stopped before the trainer (else it relaunches the old line); the trainer TERM,
# 90 s, KILL, then its loader workers, all by explicit cmdline-verified PID; code swapped only after the old trainer is
# dead, keeping .PRE_yawloss copies; the v3 supervisor (fresh lock, the live line + `--yaw-loss plain --declare-change
# yaw_loss`); evidence from the trainer's OWN lines. Nothing else changes: same run dir, same bank, same schedule.
#   md5s of the shipped files are REQUIRED as env (the switch refuses files it cannot verify):
#   EXPECT_TRAIN_MD5=.. EXPECT_MODEL_MD5=.. EXPECT_MEASURES_MD5=.. EXPECT_V3_MD5=.. \
#     setsid nohup bash switch_yawloss.sh <supervisor pid> <trainer pid> > /workspace/switch_yawloss.out 2>&1 &
# Rollback (if the evidence step does not read ZZSWITCH_OK): code/rollback_yawloss.sh.
set -u
SUP=$1; TR=$2
RUN=/workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3
PK=/workspace/refe-plan
LOG=/workspace/switch_yawloss.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
chk() { tr "\0" " " < "/proc/$1/cmdline" 2>/dev/null | grep -q -- "$2"; }
say "switcher up (pid $$): supervisor=$SUP trainer=$TR"
chk "$SUP" "refe-plan/code/pod_train_v2.sh" || { say "ABORT: $SUP is not pod_train_v2.sh"; exit 1; }
chk "$TR" "train.py --backbone vitl16" || { say "ABORT: $TR is not the trainer"; exit 1; }
# the shipped files, verified by md5 BEFORE anything is stopped
for pair in "refe/train.py.NEW:${EXPECT_TRAIN_MD5:-}" "refe/model.py.NEW:${EXPECT_MODEL_MD5:-}" \
            "refe/measures.py.NEW:${EXPECT_MEASURES_MD5:-}" "code/pod_train_v3.sh:${EXPECT_V3_MD5:-}"; do
  f=${pair%%:*}; want=${pair##*:}
  [ -s "$PK/$f" ] || { say "ABORT: missing $PK/$f"; exit 1; }
  [ ${#want} -eq 32 ] || { say "ABORT: no expected md5 for $f"; exit 1; }
  got=$(md5sum "$PK/$f" | awk '{print $1}')
  [ "$got" = "$want" ] || { say "ABORT: md5 of $f is $got, expected $want"; exit 1; }
done
say "shipped files verified by md5"
WK=$(ps -eo pid=,ppid= | awk -v p="$TR" '$2==p {print $1}' | tr '\n' ' ')
say "trainer's loader workers: $WK"
# 1) a FRESH checkpoint (written after this script started, size and mtime stable for 20 s); NOW=1 accepts the current
#    one if it is younger than NOW_MAX_S (the resume then recomputes the steps since it)
T0=$(date +%s)
if [ "${NOW:-0}" = 1 ]; then
  age=$(( T0 - $(stat -c %Y "$RUN/ckpt_last.pt") ))
  [ "$age" -le "${NOW_MAX_S:-600}" ] && T0=0 && say "NOW mode: current checkpoint is ${age} s old -- switching on it"
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
# 2) the SUPERVISOR first
kill "$SUP" 2>/dev/null; sleep 3
chk "$SUP" "pod_train_v2.sh" && { kill -9 "$SUP"; say "supervisor needed KILL"; }
say "supervisor stopped"
# 3) the trainer: TERM, up to 90 s, KILL; then its (orphaned) loader workers, cmdline-verified
kill "$TR" 2>/dev/null
for i in $(seq 1 90); do kill -0 "$TR" 2>/dev/null || break; sleep 1; done
kill -0 "$TR" 2>/dev/null && { kill -9 "$TR"; say "trainer needed KILL"; }
for w in $WK; do chk "$w" "train.py --backbone vitl16" && kill -9 "$w" && say "killed loader worker $w"; done
sleep 2
N=$(ps -eo args | grep -c "[t]rain.py --backbone vitl16")
[ "$N" = 0 ] || { say "ABORT: $N trainer processes still alive -- NOT starting v3 (nothing swapped yet)"; exit 1; }
say "old trainer gone"
# 4) swap the code, keeping the exact pre-switch files
cd "$PK/refe" || exit 1
cp -p train.py train.py.PRE_yawloss && cp -p model.py model.py.PRE_yawloss || { say "ABORT: backup failed"; exit 1; }
mv train.py.NEW train.py && mv model.py.NEW model.py && mv measures.py.NEW measures.py || { say "ABORT: swap failed"; exit 1; }
say "code swapped: $(md5sum train.py model.py measures.py | awk '{print substr($1,1,12), $2}' | tr '\n' ' ')"
# 5) the v3 supervisor: STAGES=train reuses batch_accum.txt; its own fresh lock; setsid so it outlives this script
cd /workspace || exit 1
setsid nohup env GROW=1 THREADS=2 RUN="$RUN" STAGES=train bash refe-plan/code/pod_train_v3.sh \
  >> /workspace/pod_train.out 2>&1 < /dev/null &
say "v3 supervisor started (pid $!)"
# 6) evidence from the trainer's own lines (the resume takes a few minutes: bank + compile)
OK=0
for i in $(seq 1 90); do
  if grep -q "DECLARED RECIPE CHANGE" <(tail -400 "$RUN/train.log") && grep -q "MEASURE M6" <(tail -400 "$RUN/train.log"); then OK=1; break; fi
  grep -q "REFUSING" <(tail -40 "$RUN/train.log") && break
  sleep 10
done
say "evidence: $(grep -E 'MEASURE M6|DECLARED RECIPE CHANGE|RESUMED|REFUSING|ON-POLICY scorer bank' "$RUN/train.log" | tail -6 | tr '\n' '|' | cut -c1-1200)"
NEWTR=$(ps -eo pid,args | grep "[t]rain.py --backbone vitl16" | grep -- "--yaw-loss plain" | head -1 | awk '{print $1}')
say "new trainer pid ${NEWTR:-NONE}; yaw-loss flag on its cmdline: $(tr '\0' ' ' < /proc/${NEWTR:-0}/cmdline 2>/dev/null | grep -c -- '--yaw-loss plain')"
[ "$OK" = 1 ] && [ -n "${NEWTR:-}" ] && say "ZZSWITCH_OK" || say "ZZSWITCH_CHECK_MANUALLY"
