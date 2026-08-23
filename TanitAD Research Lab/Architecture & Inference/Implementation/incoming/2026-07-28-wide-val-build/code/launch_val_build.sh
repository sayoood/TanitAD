#!/bin/bash
# Wide-FOV v5 cache build — parity VAL split, 120deg 256x640 cylindrical, PNG
# lossless. The TRAIN build's launcher (…/2026-07-28-wide-fov-build/code/
# launch_build.sh) with THREE differences, each deliberate:
#
#   1. --only-clips  -> parity_val_clips.txt  (600 clips, key 0c5f7dac3b11)
#   2. --out         -> a directory NAMED FOR THE KEY IT WILL BE REGISTERED
#      UNDER. parity.register_v2_geometry_sibling() REFUSES a key that does not
#      appear in the cache path, because corpus_key_of() resolves by path
#      substring and a key nothing resolves to is an inert registration
#      (WIDE_FOV_BUILD.md §6.3). Building straight into the final name removes
#      a rename step that can be forgotten.
#   3. OMP/MKL 6 (the brief's numbers; the train build used 4). Set either way —
#      UNSET is the trap: torch spawns ~113 threads per process.
#
# ⚠️ NO DOWNLOAD IS EXPECTED. All 600 val clips' mp4 + timestamps were MEASURED
# present on pod2 (2026-07-27) and all 197 egomotion zips are cached, so with
# the reuse-probe fix in v2_compressed.py every chunk should log
# "all N clips already local — no download". If a shard starts downloading,
# the reuse probe is not matching the on-disk names — stop and look.
# Force the download path with PAI_NO_LOCAL_REUSE=1.
set -u
source /root/.hf_env          # HF_TOKEN — for the fallback download path only
export HF_TOKEN
export TANITAD_STACK=/workspace/wfov/stack_v5
export PYTHONPATH=/workspace/wfov/stack_v5
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PAI_DECODE_THREADS=4
SHARD="$1"; K="$2"
cd /workspace/wfov/stack_v5
exec python3 -u scripts/v2_compressed.py build \
  --sel  /workspace/data/physicalai_phase0/r0/r0_selection.parquet \
  --root /workspace/data/physicalai_phase0 \
  --out  /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
  --only-clips /workspace/wfov/paritysplit/parity_val_clips.txt \
  --hfov 120 --height 256 --width 640 \
  --projection-mode cylindrical --codec png \
  --shard "${SHARD}/${K}"
