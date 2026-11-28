#!/usr/bin/env bash
# r9prof_run.sh <tag> <argv.json> <tree> <steps>  -- profile run under the Thor GPU lock (copy of r9e1_run.sh, wrapper instead of the bare trainer)
set -u
R=/home/nvidia/r8ns/r9e1; TAG="$1"; ARGVJ="$2"; TREE="$3"; NSTEP="$4"
LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock; PY=/home/nvidia/venvs/tanitad-train/bin/python
mkdir -p "$R/logs" "$R/prof"
exec 9>"$LOCKF"; echo "[$(date -u +%T)] waiting for the Thor GPU lock (<= 20 min)"
flock -w 1200 9 || { echo "ZZR9P-LOCKBUSYZZ"; exit 3; }
NAPPS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')"
[ "$NAPPS" = 0 ] || { echo "ZZR9P-GPUNOTEMPTYZZ ($NAPPS)"; exit 4; }
"$PY" - "$ARGVJ" > "$R/logs/$TAG.args" <<'PYEOF'
import json, sys
for t in json.load(open(sys.argv[1]))["argv"]: print(t)
PYEOF
mapfile -t A < "$R/logs/$TAG.args"
echo "[$(date -u +%T)] gpu lock held, GPU empty; launching ${#A[@]} tokens"
T0=$(date +%s)
( cd "$TREE/stack" && export PYTHONPATH="$TREE/stack" R9_SCRIPTS="$TREE/stack/scripts" R9_PROF_STEPS="$NSTEP" OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_TOKEN_PATH=/nonexistent/hf_token &&
  unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_HUB_TOKEN &&
  "$PY" "$TREE/stack/scripts/r9_profile_run.py" "$R/prof/$TAG.json" -- "${A[@]}" > "$R/logs/$TAG.out" 2> "$R/logs/$TAG.err" 200>&- )
RC=$?
echo "[$(date -u +%T)] trainer exited rc=$RC wall=$(( $(date +%s) - T0 )) s"
echo "ZZR9P-DONE-${RC}ZZ"
