#!/usr/bin/env bash
# Runs on the pod, detached. PI 2026-11-09: "Go for pausing as you recommend". Waits for P7's next ckpt_last.pt, stops P7's launch script and then its trainer by explicit PID,
# confirms the GPU is empty, writes the record for the resume, and launches the P6 label chain. Terminal lines in pause.log: PAUSE_STOPPED_EARLY or LABEL_CHAIN_STARTED.
set -u
CHAIN=45626
TR=84379
R=/workspace/data/refe_runs/p7_reserve_head
OUT=/workspace/data/p6base_labels
L=$OUT/pause.log
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
early() { say "PAUSE_STOPPED_EARLY $*"; exit 1; }
tr '\0' ' ' < /proc/$CHAIN/cmdline | grep -q 'p7_pod.sh' || early "pid $CHAIN is not the launch script"
tr '\0' ' ' < /proc/$TR/cmdline | grep -q 'train.py' || early "pid $TR is not the trainer"
m0=$(stat -c %Y "$R/ckpt_last.pt")
say "waiting for a checkpoint newer than mtime $m0"
m=$m0
for i in $(seq 1 1800); do
  m=$(stat -c %Y "$R/ckpt_last.pt")
  [ "$m" != "$m0" ] && break
  kill -0 $TR 2>/dev/null || early "the trainer ended by itself"
  sleep 1
done
[ "$m" != "$m0" ] || early "no new checkpoint within 30 min"
sleep 15
desc() { local c; for c in $(ps -o pid= --ppid "$1"); do echo "$c"; desc "$c"; done; }
D=$(desc $TR | tr '\n' ' ')
say "checkpoint mtime $m; stopping launch script $CHAIN, then trainer $TR; descendants: $D"
kill -TERM $CHAIN
kill -TERM $TR
for i in $(seq 1 60); do kill -0 $TR 2>/dev/null || break; sleep 1; done
kill -0 $TR 2>/dev/null && { say "trainer still alive after 60 s, SIGKILL"; kill -9 $TR; sleep 2; }
for p in $D; do kill -0 "$p" 2>/dev/null && kill -TERM "$p" 2>/dev/null; done
sleep 10
for p in $D; do kill -0 "$p" 2>/dev/null && { say "descendant $p still alive, SIGKILL"; kill -9 "$p" 2>/dev/null; }; done
empty=0
for i in $(seq 1 90); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$n" = "0" ]; then empty=1; break; fi
  sleep 2
done
[ "$empty" = 1 ] || early "GPU still has a compute process"
for i in $(seq 1 240); do kill -0 $CHAIN 2>/dev/null || break; sleep 1; done
kill -0 $CHAIN 2>/dev/null && early "launch script $CHAIN did not exit"
{ echo "paused_utc $(date -u +%FT%TZ)"; echo "ckpt_last_mtime $m"; ls -la "$R/ckpt_last.pt"; echo "last metrics line:"; tail -n 1 "$R/metrics.jsonl" | cut -c1-400; echo "resume: P7_LABEL_WORKERS=13 P7_WAIT_P6=1 with HF_TOKEN from <the pod's token file>, setsid nohup bash /workspace/refe-plan-p7/refe/p7_pod.sh"; } > "$OUT/PAUSED_P7.txt"
say "P7 stopped, GPU empty"
export PKG=/workspace/refe-plan-p6 MODEL=/workspace/data/refe_runs/p6_paper_faithful/model_final.pt
export TARGETS=/workspace/data/refe_navtrain/train_grow_cmd_3c_p7 IMAGES=/workspace/data/navtrain_pixels CALIB=/workspace/data/refe_navtrain/train_grow/calib_table.json
export OUT PY=/workspace/venv-teacher/bin/python NAVSIM_PY=/workspace/venv-teacher/bin/python STACK=
export N_TRAIN=60000 WORKERS=12 EXPECT_HELDOUT=1705 RELABEL_SCRIPT=/workspace/p6labels/p6_relabel_cached.py P6_METRIC_CACHE=/workspace/data/refe_navtrain/metric_cache
export PYTHONPATH=/workspace/dz/DriveZero/DriveRL/nuplan-devkit NUPLAN_MAPS_ROOT=/workspace/data/nuplan-maps/nuplan-maps-v1.0 NUPLAN_DATA_ROOT=/workspace/data OMP_NUM_THREADS=4
unset HF_TOKEN
cd /workspace/p6labels || early "no package directory"
setsid nohup bash p6_labels_chain.sh > "$OUT/chain.stdout" 2>&1 < /dev/null &
echo $! > "$OUT/chain.pid"
say "LABEL_CHAIN_STARTED pid $(cat "$OUT/chain.pid")"
