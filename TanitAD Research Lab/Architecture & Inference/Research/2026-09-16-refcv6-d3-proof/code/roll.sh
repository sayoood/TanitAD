#!/bin/sh
# D3 rung-1 roller. ONE arm per invocation.
#
#   bash roll.sh <tag> <infer_seed> [extra refcv3_arm.py args...]
#
# STOP: the code tree is C:/Users/Admin/refcv5cmp/repo, NOT the branch tip.
#   MEASURED 2026-09-16: stack/tanitad/refs/refc.py differs from the tip by 321
#   content lines, so a tip roll would not be paired with the banked refcv5-v2
#   dump. refcv3_arm.py and v2_dataset.py are content-identical in all three
#   trees (line endings only).
# STOP: compare.sh step 1 re-syncs this tree FROM THE DEAD G: DRIVE. Never run it.
#   This script does not sync; it uses the tree as it stands.
# STOP: THE CALL STRUCTURE IS THE BANKED ONE, VERBATIM -- all seven arms, both nav
#   conditionings, --with-oracle-sel. MEASURED 2026-09-16 by code/provgate.py:
#   dropping --no-navshuf/--no-navzero (fed batch 3 rows -> 1) moved `os` by
#   max 1.18 m / mean 0.065 m on ep000-001 while g/ha/ha0/ha0_ext stayed
#   BIT-IDENTICAL -- the ddim sampler's noise draw follows the batch shape. A
#   cheaper call would therefore have un-paired every ablation from the bank.
set -e
# ⛔ `cmd | tee log` returns TEE's status, so a dead roll reads as SUCCESS and the next
# stage runs on a partial dump. MEASURED here 2026-09-16: a killed roll printed the
# queue's DONE marker. (`never-chain-the-lander-behind-a-pipe`.)
set -o pipefail
R=/c/Users/Admin/refcv5cmp
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
OUTDIR=/c/Users/Admin/d3_out
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export PYTHONPATH="C:\\Users\\Admin\\refcv5cmp\\repo\\stack;C:\\Users\\Admin\\refcv5cmp\\repo\\taniteval;C:\\Users\\Admin\\refcv5cmp\\repo"

TAG="$1"; shift
SEED="$1"; shift
mkdir -p "$OUTDIR"

cd "$R/repo"
"$PY" taniteval/tools/refcv3_arm.py \
  --ckpt /c/Users/Admin/refcv5v2_final/ckpt.pt \
  --config /c/Users/Admin/refcv5v2_final/config.json \
  --episodes "$R/data/eval" \
  --labels "$R/data/s2_labels_v7.2_eval.jsonl.gz" \
  --nav-source v72 --grid 2s --action-units steer --device cuda \
  --window-stride 5 --with-oracle-sel \
  --lead-block "$R/data/b1_eval_lead_block.npz" \
  --n-boot 2000 --seed 0 --infer-seed "$SEED" \
  --arm "$TAG" \
  --tiers "os=T1,os_navshuf=T1,os_navzero=T1,oracle_sel=T0" \
  --dump-dir "$OUTDIR/${TAG}_dump" --out "$OUTDIR/${TAG}.json" \
  "$@" 2>&1 | tee "$OUTDIR/${TAG}_roll.log"
