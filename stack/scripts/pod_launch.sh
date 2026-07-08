#!/usr/bin/env bash
# TanitAD pod launcher — idempotent, survives terminal disconnects.
# Usage:            bash scripts/pod_launch.sh
# Override sizing:  EPISODES=250 bash scripts/pod_launch.sh
# Details: RUNPOD_RUNBOOK.md §6 (RAM rule: ~0.18 GB/comma segment + 0.12 GB/clip,
# --episodes caps EACH corpus: ~50GB RAM -> 120, ~100GB -> 250, >=180GB -> 500)
set -e
EPISODES="${EPISODES:-120}"
RUN_ID="${RUN_ID:-p0-sB01-realmix}"
OUT="/workspace/experiments/${RUN_ID}"
LOG="/workspace/experiments/${RUN_ID}.log"

cd "$(dirname "$0")/.."                      # -> stack/
mkdir -p /workspace/experiments

# Guard against BOTH a live python AND a live runner loop — during the
# runner's 15 s restart sleep no python exists, and relaunching in that
# window spawned a second trainer (observed: duplicate step lines, two
# processes fighting over ckpt.pt).
if pgrep -f "train_worldmodel" >/dev/null 2>&1 \
   || pgrep -f "run_${RUN_ID}.sh" >/dev/null 2>&1; then
    echo "training (or its auto-restart runner) already running:"
    pgrep -af "train_worldmodel|run_${RUN_ID}.sh" || true
    echo "following its output (Ctrl-C stops WATCHING only):"
    exec tail -f "${LOG}"
fi

echo "--- RAM ---"; free -g | head -2
echo "--- episodes cap: ${EPISODES}/corpus (EPISODES=... to override) ---"

# F-5: batch 64 naive OOMs the 48 GB A40 (512 window-frames of stored
# activations). micro 16 x accum 4 = effective 64; checkpointing for headroom.
MICRO="${MICRO:-16}"
ACCUM="${ACCUM:-4}"
STEPS="${STEPS:-60000}"

# Container RAM ceiling can be far below `free` (host view) — size EPISODES to
# THIS number, not to free -g (suspected cause of the overnight kill):
LIMIT=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo max)
echo "--- container memory limit: ${LIMIT} (bytes; 'max' = unlimited) ---"

# In-container auto-restart: training resumes from ckpt.pt + episode cache,
# so a crash costs at most save_every steps. (A full POD interruption still
# needs a manual pod_launch — it also resumes.)
RUNNER="/workspace/experiments/run_${RUN_ID}.sh"
cat > "${RUNNER}" <<EOF
#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in \$(seq 1 20); do
  echo "[runner] attempt \${attempt}" >> ${LOG}
  python -u -m tanitad.train.train_worldmodel \\
    --config base250cam --data realmix \\
    --data-root /workspace/data/comma2k19 \\
    --sim-root  /workspace/data/physicalai \\
    --sim-frac 0.6 --episodes ${EPISODES} --steps ${STEPS} \\
    --batch-size ${MICRO} --accum ${ACCUM} --grad-checkpoint \\
    --out ${OUT} >> ${LOG} 2>&1 && break
  echo "[runner] exited nonzero; restarting in 15 s" >> ${LOG}
  sleep 15
done
echo "[runner] finished" >> ${LOG}
EOF
chmod +x "${RUNNER}"

# No tmux: detach via setsid+nohup (survives terminal resets), then follow the
# log as normal console output. Ctrl-C stops WATCHING only, never training.
touch "${LOG}"
nohup setsid bash "${RUNNER}" </dev/null >/dev/null 2>&1 &
echo $! > "/workspace/experiments/${RUN_ID}.pid"

echo "training launched detached (pid $(cat /workspace/experiments/${RUN_ID}.pid))"
echo "  stop it:      pkill -f train_worldmodel"
echo "  watch again:  tail -f ${LOG}   (or just re-run this script)"
echo "  NOTE: dataset build/cache-load runs before step 0 prints."
echo "--- live output (Ctrl-C detaches you, training continues) ---"
exec tail -f "${LOG}"
