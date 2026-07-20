#!/bin/bash
cd /root
export PYTHONPATH=/root/taniteval:/root/TanitAD/stack
panel(){ name=$1; shift; echo "=== PANEL $name START $(date -u '+%F %T')"; timeout -k 60 1800 "$@"; echo "=== PANEL $name RC=$? $(date -u '+%F %T')"; }
panel rollout    python3 -m taniteval.runner run --model flagship-30k
panel imagination python3 -m taniteval.runner imagination --model flagship-30k
panel hierarchy  python3 -m taniteval.runner hierarchy --model flagship-30k
panel diag       python3 -m taniteval.bench --model flagship-30k
panel planning   python3 -m taniteval.planning --model flagship-30k
panel ab_19v30   python3 -m taniteval.runner ab --a flagship-speed --b flagship-30k
panel ab_refb    python3 -m taniteval.runner ab --a refb --b flagship-30k
panel ab_refa    python3 -m taniteval.runner ab --a refa-dinov2 --b flagship-30k
panel report     python3 -m taniteval.runner report
echo SUITE_DONE
