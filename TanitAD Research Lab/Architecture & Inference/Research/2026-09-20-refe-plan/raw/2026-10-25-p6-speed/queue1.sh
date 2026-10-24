#!/bin/bash
# lever queue 1 (recipe-preserving, one lever at a time). Runs detached; each run appends a JSON row to logs/runs.jsonl.
cd /root/tad/speed
W=/root/tad/speed/dinov3-vitl16
[ -f $W/model.safetensors ] || { mkdir -p $W && cp /workspace/data/backbones/dinov3-vitl16/* $W/; }
run() { tag=$1; shift
  if [ -e /root/tad/launch_b ]; then echo "ZZQ STOP launch_b exists" >> logs/queue1.log; exit 0; fi
  python3 -u bench_p6.py --weights $W --tag $tag --phases --reps 3 --n-micro 12 --json logs/runs.jsonl "$@" > logs/$tag.log 2>&1
  echo "ZZQ $tag rc=$? $(grep -o '"s_per_micro": [0-9.]*' logs/$tag.log)" >> logs/queue1.log; }
run base_rep2
run n2once --n2-once
run u8 --img u8
run early --order early
run parts --compile parts
run fusedqkv --fused-qkv
run cudnn --sdpa cudnn
run b2 --batch 2
run maxat --compile-mode max-autotune-no-cudagraphs
echo "ZZQ DONE" >> logs/queue1.log
