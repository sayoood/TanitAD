#!/bin/bash
# queue 3: kernel profile of the base, max-autotune repeat, inductor tuning knobs. Chained after queue 2.
cd /root/tad/speed
until grep -qE "ZZQ (DONE|STOP)" logs/queue2.log; do sleep 20; done
W=/root/tad/speed/dinov3-vitl16
stop() { [ -e /root/tad/launch_b ] && { echo "ZZQ STOP launch_b exists" >> logs/queue3.log; exit 0; }; }
tim() { tag=$1; shift; stop
  python3 -u bench_p6.py --weights $W --tag $tag --phases --reps 3 --n-micro 12 --json logs/runs.jsonl "$@" > logs/$tag.log 2>&1
  echo "ZZQ $tag rc=$? $(grep -o '"s_per_micro": [0-9.]*' logs/$tag.log)" >> logs/queue3.log; }
stop; python3 -u bench_p6.py --weights $W --tag prof_base --reps 1 --n-micro 4 --profile logs/prof_base.txt > logs/prof_base.log 2>&1; echo "ZZQ prof_base rc=$?" >> logs/queue3.log
tim base_rep3
tim maxat_rep2 --compile-mode max-autotune-no-cudagraphs
TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 tim maxat_cdt --compile-mode max-autotune-no-cudagraphs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True tim expseg
echo "ZZQ DONE" >> logs/queue3.log
