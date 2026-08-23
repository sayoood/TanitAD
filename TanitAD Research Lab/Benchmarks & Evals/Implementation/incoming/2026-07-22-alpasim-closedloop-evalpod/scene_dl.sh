#!/bin/bash
# Reads HF token from stdin (line 1), downloads ONE NuRec scene (26.04 release) detached.
# Token touches only tmpfs (/dev/shm, 600) and is deleted by the child before download.
IFS= read -r TOK
printf '%s' "$TOK" > /dev/shm/hftok
chmod 600 /dev/shm/hftok
SCENE=01d503d4-449b-46fc-8d78-9085e70d3554
nohup setsid bash -c '
  export HF_TOKEN="$(cat /dev/shm/hftok)"; rm -f /dev/shm/hftok
  export HF_HOME=/workspace/.hf HF_HUB_ENABLE_HF_TRANSFER=0
  echo "DL_START $(date -u +%H:%M:%S)"
  hf download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec --repo-type dataset --revision 26.04 \
    --include "sample_set/26.04_release/'"$SCENE"'/*" \
    --local-dir /workspace/scene_dl
  echo "DL_EXIT=$? $(date -u +%H:%M:%S)"
' > /workspace/scene_dl.log 2>&1 &
echo "DL_LAUNCHED pid=$!"
