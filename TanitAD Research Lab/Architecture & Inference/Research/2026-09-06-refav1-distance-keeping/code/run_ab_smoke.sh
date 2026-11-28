#!/usr/bin/env bash
# D-REFAV1-DK-COST P4 -- the ARMED vs UNARMED A/B, end to end on the real
# step-21109 checkpoint, on the 2 dev-box-slice episodes that carry a VIOLATING
# eval-grid window (2172a5a3: 1 window, gap 30.29 m at 21.60 m/s; 2602baaa:
# 2 windows, gap 41.31/44.48 m at 27.17/27.00 m/s).
#
# It is a PLUMBING + DIRECTION proof, not a driving result: n = 4 windows over
# 2 episodes cannot carry a family claim, and the record says so.
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad/dk/smoke"
WT="C:/Users/Admin/tanitad-wt"
S="C:/Users/Admin/refav1_eval_slice"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
LB="$WT/_lead_b1/b1_eval_lead_block.npz"
LAB="$WT/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz"

export PYTHONPATH="$WT/stack;$WT/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {   # $1 = tag, $2 = w_dk
  "$PY" "$WT/taniteval/tools/refav1_arm.py" \
    --ckpt "$S/ckpt_ep3/ckpt.pt" --config "$S/ckpt_ep3/config.json" \
    --cache "$SP/cache" --episodes "$SP/eps" \
    --labels "$LAB" --nav "$LAB" \
    --lead-block "$LB" --dk-gap-block "$LB" \
    --dk-w "$2" --dk-tau 1.5 --dk-d0 5.0 --dk-gap-source oracle_label \
    --device cuda --window-stride 40 --episodes-n 0 --no-navshuf \
    --dump-dir "$SP/dump_$1" --out "$SP/out_$1.json" --arm "dksmoke-$1" \
    >> "$SP/$1.log" 2>&1
  echo "EXIT_$1=$?" | tee -a "$SP/$1.log"
}

run off 0.0
run on 1e-5
echo "SMOKE_DONE"
