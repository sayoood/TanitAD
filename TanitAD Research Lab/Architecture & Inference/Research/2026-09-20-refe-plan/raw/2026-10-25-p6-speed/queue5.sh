#!/bin/bash
# queue 5: RECIPE-CHANGING combinations (PI decision list). Chained after queue 4.
cd /root/tad/speed
until grep -qE "ZZQ (DONE|STOP)" logs/queue4.log; do sleep 20; done
W=/root/tad/speed/dinov3-vitl16
stop() { [ -e /root/tad/launch_b ] && { echo "ZZQ STOP launch_b exists" >> logs/queue5.log; exit 0; }; }
tim() { tag=$1; shift; stop
  python3 -u bench_p6.py --weights $W --tag $tag --phases --reps 3 --n-micro 12 --json logs/runs.jsonl "$@" > logs/$tag.log 2>&1
  echo "ZZQ $tag rc=$? $(grep -o '"s_per_micro": [0-9.]*' logs/$tag.log) $(grep -o '"max_mem_gb": [0-9.]*' logs/$tag.log)" >> logs/queue5.log; }
tim C_lora12_bf16 --lora-from 12 --bf16-resid
tim C_lora12_b8 --lora-from 12 --batch 8
tim C_res448_lora12 --res 448x832 --lora-from 12
tim C_lora18_bf16 --lora-from 18 --bf16-resid
tim C_res448_bf16 --res 448x832 --bf16-resid
echo "ZZQ DONE" >> logs/queue5.log
