#!/usr/bin/env bash
# Runs on the pod, detached.  Usage: pause_p7_s1zsm.sh <p7_pod.sh pid> <train.py pid>      (PREREG_S1ZSM_P6 section 7: P7 is paused ONCE for the one S1ZSM chain)
# Style of refe/rep_pause_p7.sh (same P7 resume command, same cmdline checks, EXIT-trap resume).  Before P7 is touched it checks that the chain can start (the shipped package + manifest, the frozen card bytes, the navtest
# lists): a missing input is PAUSE_STOPPED_EARLY with P7 UNTOUCHED.  Then it waits for P7's next ckpt_last.pt, stops P7 by explicit PID (cmdline checked), confirms the GPU is empty, runs refe/s1zsm_chain.sh in the
# foreground (<= 6 h), then ALWAYS resumes P7 (also when this script is killed or errors after P7 was stopped), unless /workspace/data/refe_s1zsm/HOLD_P7 exists.
# Terminal lines in pause.log: PAUSE_STOPPED_EARLY (P7 not touched, or the GPU never emptied and P7 was resumed by the trap), P7_RESUMED or P7_HELD.
set -u
CHAIN=${1:-}; TR=${2:-}
R=/workspace/data/refe_runs/p7_reserve_head; OUT=/workspace/data/refe_s1zsm; L=$OUT/pause.log; PKG=/workspace/refe-plan-s1zsm
CARD_SHA=ac75b2891e794de800102a6d5a0f0353e8092e5f976f31b543ac7700811ad6b8
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
early() { say "PAUSE_STOPPED_EARLY $*"; exit 1; }
[ -n "$CHAIN" ] && [ -n "$TR" ] || early "usage: pause_p7_s1zsm.sh <p7_pod.sh pid> <train.py pid>"
# the chain's inputs, BEFORE P7 is touched
[ -f "$PKG/SHIP_MANIFEST.json" ] && [ -f "$PKG/refe/s1zsm_chain.sh" ] || early "the S1ZSM package is not shipped ($PKG/SHIP_MANIFEST.json, refe/s1zsm_chain.sh)"
[ -f "$OUT/PREREG_S1ZSM_P6.md" ] && [ "$(sha256sum "$OUT/PREREG_S1ZSM_P6.md" | cut -c1-64)" = "$CARD_SHA" ] || early "$OUT/PREREG_S1ZSM_P6.md is missing or is not the frozen card (sha256 ${CARD_SHA:0:12}...)"
[ -f "$OUT/navtest_ids/navtest_logs.json" ] && [ -f "$OUT/navtest_ids/navtest_tokens.txt.gz" ] || early "$OUT/navtest_ids/ is not staged"
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
m0=$(stat -c %Y "$R/ckpt_last.pt"); say "waiting for a checkpoint newer than mtime $m0"; m=$m0
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
# the S1ZSM chain, in the foreground of this script
S1ZSM_HF_TOKEN_FILE="$TF" timeout 21600 bash $PKG/refe/s1zsm_chain.sh >> $OUT/chain.out 2>&1 < /dev/null
say "S1ZSM chain rc $? (last: $(grep -E 'ZZS1ZSM_|REFUSED' $OUT/chain.log 2>/dev/null | tail -1 | cut -c1-160))"
resume_p7
