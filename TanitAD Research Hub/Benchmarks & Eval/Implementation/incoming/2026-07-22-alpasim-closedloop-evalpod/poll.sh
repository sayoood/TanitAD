#!/bin/bash
# Poll re-eval progress for a tag. Usage: bash poll.sh <tag>
TAG="${1:-base}"
echo "=== master [$TAG] ==="; tail -5 /workspace/gate1_reeval_${TAG}.log 2>/dev/null
echo "=== run tail ==="; tail -12 /workspace/gate1_reeval_${TAG}_run.log 2>/dev/null
echo "=== renderer log tail ==="; tail -4 /workspace/reeval_renderer.log 2>/dev/null
echo "=== ports up ==="; ss -ltn | grep -oE ':(6011|6006|6007|6789) ' | sort -u | tr '\n' ' '; echo
echo "=== parquets ==="; ls -1 /workspace/gate1_reeval_${TAG}/rollouts/clipgt-*/*/metrics.parquet 2>/dev/null | wc -l
echo "=== DONE marker ==="; cat /workspace/gate1_reeval_${TAG}/DONE 2>/dev/null || echo "no DONE yet"
echo "=== gpu ==="; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | tr '\n' ';'; echo
