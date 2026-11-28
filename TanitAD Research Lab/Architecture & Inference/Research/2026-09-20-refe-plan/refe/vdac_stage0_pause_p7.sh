#!/usr/bin/env bash
# Runs on the pod, detached.  Usage: vdac_stage0_pause_p7.sh <p7_pod.sh pid> <train.py pid> S0      (ONE stage per P7 pause)
# SIBLING of refe/vdac_pause_p7.sh (PREREG_VDAC_P6_A5 Stage 0; raw/2026-11-12-vdac-a5-build2/POD_PLAN.md 3.1), same stop / GPU-empty / ALWAYS-resume logic.
# Waits for P7's next ckpt_last.pt, stops P7 by explicit PID, confirms the GPU is empty, runs refe/vdac_stage0_pod.sh from /workspace/refe-plan-stage0
# (in-sample IoU forward of HA-s0 + the seed-0 initial head for S1-F; ~0.25 L40-h ESTIMATED; killed at 5,400 s), then ALWAYS resumes P7 (also when this
# script is killed or errors after P7 was stopped), unless /workspace/data/refe_vdac/HOLD_P7 exists. Log: /workspace/data/refe_vdac/stage0_pause.log.
# Terminal lines: PAUSE_STOPPED_EARLY, P7_RESUMED or P7_HELD. The Stage 0 READ (vdac_stage0.py read, then s1f) runs on the dev box afterwards.
set -u
CHAIN=${1:-}; TR=${2:-}; STAGEARG=${3:-}
R=/workspace/data/refe_runs/p7_reserve_head; OUT=/workspace/data/refe_vdac; L=$OUT/stage0_pause.log
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
early() { say "PAUSE_STOPPED_EARLY $*"; exit 1; }
[ -n "$CHAIN" ] && [ -n "$TR" ] && [ -n "$STAGEARG" ] || early "usage: vdac_stage0_pause_p7.sh <p7_pod.sh pid> <train.py pid> S0"
case "$STAGEARG" in
  S0) ARMS="" ;;
  *) early "unknown stage $STAGEARG (this wrapper runs S0 only; S1a / S1b use vdac_pause_p7.sh)" ;;
esac
TF=$(grep -o "/root/\.[A-Za-z0-9_.]*" /workspace/data/refe_scorer1p6/after_arms.sh | head -1); [ -f "$TF" ] || early "no token file"
[ -f /workspace/refe-plan-stage0/refe/vdac_stage0_pod.sh ] || early "no /workspace/refe-plan-stage0/refe/vdac_stage0_pod.sh (ship the Stage 0 package first); P7 was not touched"
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
m0=$(stat -c %Y "$R/ckpt_last.pt"); say "waiting for a checkpoint newer than mtime $m0 (stage $STAGEARG, arms: $ARMS)"; m=$m0
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
# Stage 0 (PREREG_VDAC_P6_A5): one invocation of the stage script; 5,400 s = 90 min kill bound against ~15 min ESTIMATED
say "stage0 starting (timeout 5400 s)"
timeout 5400 bash /workspace/refe-plan-stage0/refe/vdac_stage0_pod.sh >> $OUT/stage0_pause.out 2>&1 < /dev/null
src=$?
say "stage0 rc $src (last: $(grep -E 'ZZSTAGE0_' $OUT/stage0/stage0.log 2>/dev/null | tail -1 | cut -c1-160))"
resume_p7
