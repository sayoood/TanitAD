#!/bin/bash
# Launch K detached, chunk-sharded v2-compressed-cache build workers. Parameterized
# via env so pod1 and pod3 share it. Each worker survives ssh logout
# (setsid+nohup+</dev/null); disjoint by chunk index -> no collision within a pod.
# Resumable (skips built clip_ids). Banks per chunk. Kill by explicit PID only.
#
# Two-pod split (clip-id disjoint): pod1 SEL=sel_bottom (0-4499), pod3 SEL=sel_top
# (4500-8999). Same OUT basename on both -> clean consolidation (clip-id-keyed).
PYBIN=${PYBIN:-python3}
SEL=${SEL:-/workspace/data/physicalai_v2/r0/r0_selection.parquet}
ROOT=${ROOT:-/workspace/data/physicalai_v2}
OUT=${OUT:-/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d}
LOG=${LOG:-$ROOT/logs}
K=${K:-5}                              # tune to the pod cgroup RAM cap
export HF_TOKEN=${HF_TOKEN:-$(cat /root/.cache/huggingface/token 2>/dev/null || grep -oE 'hf_[A-Za-z0-9]+' /workspace/TanitAD/Keys.txt 2>/dev/null | head -1)}
export PYTHONPATH=/workspace/TanitAD/stack
export CUDA_VISIBLE_DEVICES=""         # pure-CPU build
export V2_TORCH_THREADS=${V2_TORCH_THREADS:-20}   # cap torch intra-op (K*this < cores)
export PAI_DECODE_THREADS=${PAI_DECODE_THREADS:-4}
export PAI_DECODE_BATCH=${PAI_DECODE_BATCH:-16}
mkdir -p "$OUT" "$LOG"
: > "$LOG/worker_pids.txt"
for i in $(seq 0 $((K-1))); do
  setsid nohup $PYBIN /workspace/TanitAD/stack/scripts/v2_compressed.py build \
    --sel "$SEL" --root "$ROOT" --out "$OUT" --quality 90 --shard "$i/$K" \
    > "$LOG/worker_$i.log" 2>&1 < /dev/null &
  echo $! >> "$LOG/worker_pids.txt"
done
echo "launched $K workers ($PYBIN) SEL=$(basename $SEL) -> $OUT"
echo "pids: $(cat $LOG/worker_pids.txt | tr '\n' ' ')"
