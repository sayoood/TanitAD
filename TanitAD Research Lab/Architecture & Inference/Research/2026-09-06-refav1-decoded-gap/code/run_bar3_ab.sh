#!/usr/bin/env bash
# D-REFAV1-DK-DECODED BAR-3 -- the four-rung gap-source ladder on the REAL
# step-21109 checkpoint. ONE VARIABLE MOVES PER RUN: --dk-gap-source.
#
#   off      w_dk = 0                      the PARITY control (shipped path)
#   oracle   gate ORACLE  gap ORACLE       the published CEILING
#   decgap   gate ORACLE  gap DECODED      isolates the GAP decode
#   decoded  gate DECODED gap DECODED      the VISION-ONLY arm
#
# Same seed, same window list, same weight/tau/d0. The window list is the
# ORACLE's own LEAD windows in the 17 episodes that carry a violating one --
# identical for every arm, so it is a MEASUREMENT choice and never an input.
#
# ⛔ NEITHER NAMED GPU IS TOUCHED: the A40 runs refcv5's training and Thor runs a
# sibling's eval. This is the dev-box RTX 4060 only.
set -u
R="C:/Users/Admin/dkhead_run"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
# md5-verified copies in the ISOLATED run tree: the tanitad-wt MIRROR is
# re-synced FROM the repo mid-run and DELETES repo-absent files, which has
# already killed one long run. Never read a live input from it.
LB="$R/inputs/b1_eval_lead_block.npz"
LAB="$R/inputs/s2_labels_v7.2_eval.jsonl.gz"
CK="C:/Users/Admin/refav1_eval_slice/ckpt_ep3"
# ⚠️ the RECALIBRATED bundle: the shrinkage lever was derived AFTER BAR-2 and
# is therefore POST-HOC, not pre-registered. It is the stronger head (BAR-2
# sensitivity 14/21 vs 12/21 at identical specificity), so BAR-3 is run on it
# and the status is stated rather than smuggled.
HEAD="$R/head/gap_head_bundle_recal.pt"
WL="$R/bar3_windows_sub.json"
OUT="$R/ab"

export PYTHONPATH="$R/stack;$R/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run () {   # $1 tag, rest = extra flags
  tag="$1"; shift
  "$PY" "$R/taniteval/tools/refav1_arm.py" \
    --ckpt "$CK/ckpt.pt" --config "$CK/config.json" \
    --cache "$OUT/cache" --episodes "$OUT/eps" \
    --labels "$LAB" --nav "$LAB" \
    --lead-block "$LB" --dk-gap-block "$LB" \
    --window-list "$WL" \
    --dk-tau 1.5 --dk-d0 5.0 \
    "$@" \
    --device cuda --episodes-n 0 --no-navshuf --seed 0 \
    --dump-dir "$OUT/dump_$tag" --out "$OUT/out_$tag.json" --arm "dk-$tag" \
    >> "$OUT/$tag.log" 2>&1
  echo "EXIT_$tag=$?" | tee -a "$OUT/$tag.log"
}

run off     --dk-w 0
run oracle  --dk-w 1e-5 --dk-gap-source oracle_label
run decgap  --dk-w 1e-5 --dk-gap-source decoded_gap --dk-gap-head "$HEAD"
run decoded --dk-w 1e-5 --dk-gap-source decoded     --dk-gap-head "$HEAD" \
            --dk-present-thr 0.5
echo "BAR3_DONE"
