#!/bin/bash
# queue 4: equivalence v2 (WTA winners + per-group) incl. the ULP-perturbation calibration arm. Chained after queue 3.
cd /root/tad/speed
until grep -qE "ZZQ (DONE|STOP)" logs/queue3.log; do sleep 20; done
W=/root/tad/speed/dinov3-vitl16; CK=/root/tad/rb2603/p6_paper_faithful/ckpt_last.pt
stop() { [ -e /root/tad/launch_b ] && { echo "ZZQ STOP launch_b exists" >> logs/queue4.log; exit 0; }; }
eqr() { tag=$1; shift; stop
  python3 -u bench_p6.py --weights $W --ckpt $CK --equiv eq/$tag.pt --equiv-samples 16 "$@" > logs/$tag.log 2>&1
  echo "ZZQ $tag rc=$? $(grep -o 'gnorm [0-9.e+-]*' logs/$tag.log)" >> logs/queue4.log; }
eqr e2_base_a
eqr e2_base_b
eqr e2_perturb6 --perturb-img 1e-6
eqr e2_perturb3 --perturb-img 1e-3
eqr e2_parts --compile parts
eqr e2_maxat --compile-mode max-autotune-no-cudagraphs
eqr e2_bf16resid --bf16-resid
eqr e2_n2once --n2-once
python3 bench_p6.py --compare eq/e2_base_a.pt eq/e2_base_b.pt eq/e2_perturb6.pt eq/e2_perturb3.pt eq/e2_parts.pt eq/e2_maxat.pt eq/e2_bf16resid.pt eq/e2_n2once.pt > logs/equiv2_table.jsonl 2>&1
echo "ZZQ DONE" >> logs/queue4.log
