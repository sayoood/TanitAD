#!/bin/bash
# queue 2: equivalence dumps (step-2603 LoRA/heads, fixed 16-sample window = 4 micro x 4) + RECIPE-CHANGING timings.
cd /root/tad/speed
until grep -qE "ZZQ DONE|ZZQ STOP" logs/queue1.log; do sleep 20; done
W=/root/tad/speed/dinov3-vitl16; CK=/root/tad/rb2603/p6_paper_faithful/ckpt_last.pt
mkdir -p eq
stop() { [ -e /root/tad/launch_b ] && { echo "ZZQ STOP launch_b exists" >> logs/queue2.log; exit 0; }; }
eqr() { tag=$1; shift; stop
  python3 -u bench_p6.py --weights $W --ckpt $CK --equiv eq/$tag.pt --equiv-samples 16 "$@" > logs/$tag.log 2>&1
  echo "ZZQ $tag rc=$? $(grep -o 'gnorm [0-9.e+-]*' logs/$tag.log)" >> logs/queue2.log; }
tim() { tag=$1; shift; stop
  python3 -u bench_p6.py --weights $W --tag $tag --phases --reps 3 --n-micro 12 --json logs/runs.jsonl "$@" > logs/$tag.log 2>&1
  echo "ZZQ $tag rc=$? $(grep -o '"s_per_micro": [0-9.]*' logs/$tag.log)" >> logs/queue2.log; }
eqr eq_base_a
eqr eq_base_b
eqr eq_n2once --n2-once
eqr eq_u8 --img u8
eqr eq_early --order early
eqr eq_parts --compile parts
eqr eq_maxat --compile-mode max-autotune-no-cudagraphs
eqr eq_nocompile --compile none
eqr eq_bf16resid --bf16-resid
tim R_lora12 --lora-from 12
tim R_lora18 --lora-from 18
tim R_res448 --res 448x832
tim R_res384 --res 384x704
tim R_bf16resid --bf16-resid
echo "ZZQ DONE" >> logs/queue2.log
