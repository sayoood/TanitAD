#!/usr/bin/env bash
# Thor concurrency pilot — 10 densest Alpamayo chunks, built ALONGSIDE the live
# v6F-SW-30k training run (PID 25477). Authorised as a MEASURED pilot with a hard
# abort; protecting the trainer outranks completing the build.
#
# Deliberately conservative:
#   nice -n 19          lowest CPU priority — the trainer preempts it always
#   OMP_NUM_THREADS=3   torch spawns ~113 threads/process otherwise (CLAUDE.md)
#   V2_TORCH_THREADS=3  v2_compressed's own torch.set_num_threads knob
#   PAI_DECODE_THREADS=3
#   single process      no shard fan-out
#
# venv: tanitad-edge — NOT tanitad-train. The trainer runs from tanitad-train and
# that venv is never written to while it is training.
set -uo pipefail

BASE=/home/nvidia/w120pilot
export TANITAD_STACK=/home/nvidia/TanitAD/stack
export PYTHONPATH=/home/nvidia/TanitAD/stack:/home/nvidia/TanitAD/taniteval
export OMP_NUM_THREADS=3
export V2_TORCH_THREADS=3
export PAI_DECODE_THREADS=3
export PAI_DECODE_BATCH=8
# token read IN PLACE from the HF cache; never printed, never in argv.
# Both names: _hf_download reads HF_TOKEN, huggingface_hub reads either.
export HF_TOKEN="$(cat /home/nvidia/.cache/huggingface/token)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

mkdir -p "$BASE/pai_root/r0" "$BASE/out"

# ⛔ REQUIRED, and NOT optional plumbing. `physicalai._chunk_of_clip` reads
# <root>/r0/r0_selection.parquet to map clip_id -> chunk, which is how
# `intrinsics_for_clip` finds the chunk's camera_intrinsics parquet. Without it
# the first build raises FileNotFoundError *after* the chunk zip is already
# downloaded (MEASURED: 536 MB paid for, zero clips built).
#
# Correctness, not just liveness: with no per-clip intrinsics the code falls back
# to the corpus MEDIAN, which "reverts the crop to geometric-center -> horizon NOT
# rig-corrected". PhysicalAI front-wide has TWO rigs (cy ~543 / ~755), so a
# geometric-center crop is ~215 px wrong for rig B. The crash prevented a
# silently mis-cropped corpus.
# The selection parquet already carries exactly the two columns needed.
cp -f "$BASE/pilot_sel_top10.parquet" "$BASE/pai_root/r0/r0_selection.parquet"

nohup nice -n 19 /home/nvidia/venvs/tanitad-edge/bin/python -u \
  /home/nvidia/TanitAD/stack/scripts/v2_compressed.py build \
  --sel  "$BASE/pilot_sel_top10.parquet" \
  --root "$BASE/pai_root" \
  --out  "$BASE/out" \
  --hfov 120 --height 256 --width 640 \
  --projection-mode cylindrical --codec png \
  >> "$BASE/build.log" 2>&1 &

echo $! > "$BASE/build.pid"
sleep 2
echo "ZZLAUNCH-$(cat "$BASE/build.pid")-$(kill -0 "$(cat "$BASE/build.pid")" 2>/dev/null && echo alive || echo dead)ZZ"
