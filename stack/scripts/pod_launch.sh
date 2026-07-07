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
command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; }

if pgrep -f "train_worldmodel" >/dev/null 2>&1; then
    echo "training already running:"; pgrep -af "train_worldmodel"
    echo "attach: tmux attach -t train   |   log: tail -f ${LOG}"
    exit 0
fi
tmux kill-session -t train 2>/dev/null || true

echo "--- RAM ---"; free -g | head -2
echo "--- episodes cap: ${EPISODES}/corpus (EPISODES=... to override) ---"

# F-5: batch 64 naive OOMs the 48 GB A40 (512 window-frames of stored
# activations). micro 16 x accum 4 = effective 64; checkpointing for headroom.
MICRO="${MICRO:-16}"
ACCUM="${ACCUM:-4}"

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
    --sim-frac 0.6 --episodes ${EPISODES} \\
    --batch-size ${MICRO} --accum ${ACCUM} --grad-checkpoint \\
    --out ${OUT} >> ${LOG} 2>&1 && break
  echo "[runner] exited nonzero; restarting in 15 s" >> ${LOG}
  sleep 15
done
EOF
chmod +x "${RUNNER}"
tmux new-session -d -s train "bash ${RUNNER}"

echo "launched in tmux session 'train'"
echo "  attach:  tmux attach -t train   (detach: Ctrl-b d)"
echo "  log:     tail -f ${LOG}"
echo "  NOTE: dataset build (video decode) runs 10-20 min before step 0 prints."
