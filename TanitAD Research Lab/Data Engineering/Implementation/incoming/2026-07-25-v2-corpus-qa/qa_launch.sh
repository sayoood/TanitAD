#!/bin/bash
# Detached, niced, bounded-worker v2 corpus QA scan. READ-ONLY on the cache dir.
# Env: PYBIN, WORKERS, TAG. Survives ssh logout (setsid+nohup+</dev/null).
PYBIN=${PYBIN:-python3}
WORKERS=${WORKERS:-12}
TAG=${TAG:-pod}
CACHE=${CACHE:-/workspace/data/physicalai_v2/epcache-physicalai-v2bal-4b7eeeac222d}
OUT=/workspace/tmp/qa_full_$TAG
rm -f "$OUT.json" "$OUT.csv" "$OUT.log"
export QA_LIB=/workspace/tmp/qa_lib
export PYTHONPATH=/workspace/TanitAD/stack
export CUDA_VISIBLE_DEVICES=""
setsid nohup nice -n 15 "$PYBIN" /workspace/tmp/v2_corpus_qa_scan.py \
  --cache "$CACHE" --out "$OUT" --workers "$WORKERS" \
  > "$OUT.log" 2>&1 < /dev/null &
echo "LAUNCHED pid=$! tag=$TAG workers=$WORKERS pybin=$PYBIN out=$OUT"
