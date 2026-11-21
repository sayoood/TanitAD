#!/usr/bin/env bash
# Runs on the pod, detached. Run 2 (2026-11-10, after the G18 fix at 8ef2cdda). PI 2026-11-09: "Go ahead with the TTC and zero-tail arm first in Batch 2". Usage: pause_p7_s1z.sh <p7_pod.sh pid> <train.py pid>
# Waits for P7's next ckpt_last.pt, stops P7 by explicit PID, confirms the GPU is empty, (A) is skipped (FIT logits done 2026-11-10); runs
# (B) the S1Z chain, then (C) ALWAYS resumes P7 unless /workspace/data/refe_s1z/HOLD_P7 exists. Terminal lines in pause.log: PAUSE_STOPPED_EARLY, P7_RESUMED or P7_HELD.
set -u
CHAIN=$1; TR=$2
R=/workspace/data/refe_runs/p7_reserve_head; OUT=/workspace/data/refe_s1z; L=$OUT/pause.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
early() { say "PAUSE_STOPPED_EARLY $*"; exit 1; }
TF=$(grep -o "/root/\.[A-Za-z0-9_.]*" /workspace/data/refe_scorer1p6/after_arms.sh | head -1); [ -f "$TF" ] || early "no token file"
tr '\0' ' ' < /proc/$CHAIN/cmdline | grep -q 'p7_pod.sh' || early "pid $CHAIN is not the launch script"
tr '\0' ' ' < /proc/$TR/cmdline | grep -q 'train.py' || early "pid $TR is not the trainer"
m0=$(stat -c %Y "$R/ckpt_last.pt"); say "waiting for a checkpoint newer than mtime $m0"; m=$m0
for i in $(seq 1 3600); do m=$(stat -c %Y "$R/ckpt_last.pt"); [ "$m" != "$m0" ] && break; kill -0 $TR 2>/dev/null || early "the trainer ended by itself"; sleep 1; done
[ "$m" != "$m0" ] || early "no new checkpoint within 60 min"
sleep 15
desc() { local c; for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
D="$(desc $TR | tr '\n' ' ')"
say "checkpoint mtime $m; stopping launch script $CHAIN, then trainer $TR; descendants: $D"
kill -TERM $CHAIN; kill -TERM $TR
for i in $(seq 1 60); do kill -0 $TR 2>/dev/null || break; sleep 1; done
kill -0 $TR 2>/dev/null && { say "trainer still alive after 60 s, SIGKILL"; kill -9 $TR; sleep 2; }
for p in $D; do kill -0 "$p" 2>/dev/null && kill -TERM "$p" 2>/dev/null; done
sleep 10
for p in $D; do kill -0 "$p" 2>/dev/null && { say "descendant $p still alive, SIGKILL"; kill -9 "$p" 2>/dev/null; }; done
empty=0; for i in $(seq 1 90); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = "0" ] && { empty=1; break; }; sleep 2; done
[ "$empty" = 1 ] || early "GPU still has a compute process"
for i in $(seq 1 240); do kill -0 $CHAIN 2>/dev/null || break; sleep 1; done
kill -0 $CHAIN 2>/dev/null && early "launch script $CHAIN did not exit"
say "left over P7 helpers: $(ps -eo pid,cmd | grep -E 'p2r_sync|p7_pod' | grep -v grep | cut -c1-80 | tr '\n' ';')"
say "P7 stopped, GPU empty"
# (A) skipped in this run: the FIT arm logits finished on 2026-11-10 (arm_fit2, ZZFITARM_OK)
# (B) the S1Z chain, in the foreground of this script
S1Z_HF_TOKEN_FILE="$TF" timeout 21600 bash /workspace/refe-plan-s1z/refe/s1z_chain.sh >> $OUT/chain.out 2>&1 < /dev/null
say "S1Z chain rc $? (last: $(grep -E 'ZZS1Z_|REFUSED' $OUT/chain.log | tail -1 | cut -c1-160))"
for i in $(seq 1 90); do n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = "0" ] && break; sleep 2; done
# (C) resume P7 unless held
if [ -e $OUT/HOLD_P7 ]; then say "P7_HELD ($OUT/HOLD_P7 exists)"; exit 0; fi
HF_TOKEN=$(cat "$TF") P7_LABEL_WORKERS=13 P7_WAIT_P6=1 setsid nohup bash /workspace/refe-plan-p7/refe/p7_pod.sh >> /workspace/data/refe_p7/p7_stdout.log 2>&1 < /dev/null &
say "P7_RESUMED pid $!"
