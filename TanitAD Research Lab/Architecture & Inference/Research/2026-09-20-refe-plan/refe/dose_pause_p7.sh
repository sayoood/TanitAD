#!/usr/bin/env bash
# Runs on the pod, detached.  Usage: dose_pause_p7.sh <p7_pod.sh pid> <train.py pid> S1ZD      (the one dose arm)
# Style of refe/rep_pause_p7.sh (PREREG_DOSE_P6).  Waits for P7's next ckpt_last.pt, stops P7 by explicit PID, confirms the GPU is empty, runs the dose arm's chain (dose_chain.sh S1ZD, 4.1 L40-h ESTIMATED), then ALWAYS resumes P7 (also when this script is killed or errors after P7 was stopped), unless /workspace/data/refe_dose/HOLD_P7 exists.
# Terminal lines in pause.log: PAUSE_STOPPED_EARLY (P7 not touched), P7_RESUMED or P7_HELD.  S1Z's lambda_z must already be calibrated: dose_chain.sh refuses otherwise.
set -u
CHAIN=${1:-}; TR=${2:-}; shift 2 2>/dev/null || true
ARMS="$*"
R=/workspace/data/refe_runs/p7_reserve_head; OUT=/workspace/data/refe_dose; L=$OUT/pause.log
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
early() { say "PAUSE_STOPPED_EARLY $*"; exit 1; }
[ -n "$CHAIN" ] && [ -n "$TR" ] && [ -n "$ARMS" ] || early "usage: dose_pause_p7.sh <p7_pod.sh pid> <train.py pid> S1ZD"
for a in $ARMS; do case "$a" in S1ZD) ;; *) early "unknown arm $a (S1ZD)" ;; esac; done
TF=$(grep -o "/root/\.[A-Za-z0-9_.]*" /workspace/data/refe_scorer1p6/after_arms.sh | head -1); [ -f "$TF" ] || early "no token file"
tr '\0' ' ' < /proc/$CHAIN/cmdline | grep -q 'p7_pod.sh' || early "pid $CHAIN is not the launch script"
tr '\0' ' ' < /proc/$TR/cmdline | grep -q 'train.py' || early "pid $TR is not the trainer"
STOPPED=0; RESUMED=0
resume_p7() {
  [ "$STOPPED" = 1 ] && [ "$RESUMED" = 0 ] || return 0
  RESUMED=1
  for i in $(seq 1 90); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = "0" ] && break; sleep 2; done
  if [ -e $OUT/HOLD_P7 ]; then say "P7_HELD ($OUT/HOLD_P7 exists)"; return 0; fi
  HF_TOKEN=$(cat "$TF") P7_LABEL_WORKERS=13 P7_WAIT_P6=1 setsid nohup bash /workspace/refe-plan-p7/refe/p7_pod.sh >> /workspace/data/refe_p7/p7_stdout.log 2>&1 < /dev/null &
  say "P7_RESUMED pid $!"
}
trap 'resume_p7' EXIT
m0=$(stat -c %Y "$R/ckpt_last.pt"); say "waiting for a checkpoint newer than mtime $m0 (arms: $ARMS)"; m=$m0
for i in $(seq 1 3600); do m=$(stat -c %Y "$R/ckpt_last.pt"); [ "$m" != "$m0" ] && break; kill -0 $TR 2>/dev/null || early "the trainer ended by itself"; sleep 1; done
[ "$m" != "$m0" ] || early "no new checkpoint within 60 min"
sleep 15
desc() { local c; for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
D="$(desc $TR | tr '\n' ' ')"
say "checkpoint mtime $m; stopping launch script $CHAIN, then trainer $TR; descendants: $D"
STOPPED=1
kill -TERM $CHAIN; kill -TERM $TR
for i in $(seq 1 60); do kill -0 $TR 2>/dev/null || break; sleep 1; done
kill -0 $TR 2>/dev/null && { say "trainer still alive after 60 s, SIGKILL"; kill -9 $TR; sleep 2; }
for p in $D; do kill -0 "$p" 2>/dev/null && kill -TERM "$p" 2>/dev/null; done
sleep 10
for p in $D; do kill -0 "$p" 2>/dev/null && { say "descendant $p still alive, SIGKILL"; kill -9 "$p" 2>/dev/null; }; done
empty=0; for i in $(seq 1 90); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = "0" ] && { empty=1; break; }; sleep 2; done
if [ "$empty" != 1 ]; then say "PAUSE_STOPPED_EARLY GPU still has a compute process after P7 was stopped"; exit 1; fi      # the EXIT trap resumes P7
for i in $(seq 1 240); do kill -0 $CHAIN 2>/dev/null || break; sleep 1; done
if kill -0 $CHAIN 2>/dev/null; then say "PAUSE_STOPPED_EARLY launch script $CHAIN did not exit"; exit 1; fi
say "left over P7 helpers: $(ps -eo pid,cmd | grep -E 'p2r_sync|p7_pod' | grep -v grep | cut -c1-80 | tr '\n' ';')"
say "P7 stopped, GPU empty"
# the dose arm, one chain invocation
for A in $ARMS; do
  LA=$(echo "$A" | tr 'A-Z' 'a-z')
  say "chain $A starting"
  DOSE_HF_TOKEN_FILE="$TF" timeout 25200 bash /workspace/refe-plan-dose/refe/dose_chain.sh "$A" >> $OUT/chain_$LA.out 2>&1 < /dev/null
  say "chain $A rc $? (last: $(grep -E 'ZZDOSE_|REFUSED' $OUT/chain_$LA.log 2>/dev/null | tail -1 | cut -c1-160))"
done
resume_p7
