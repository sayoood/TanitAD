#!/bin/sh
# T-B mask roller. ONE arm per invocation.
#
#   bash maskroll.sh <tag> <mask_mode: lead|rand|zero> <infer_seed> [extra args...]
#
# Same call structure as code/roll.sh (all seven arms, both nav conditionings,
# --with-oracle-sel) so every mask arm is paired with the banked refcv5-v2 dump and
# with every other mask arm. The ONLY difference from roll.sh is that the eval
# dataset's `_window_u8` is wrapped by `run_mask_roll.py`.
set -e
# ⛔ `cmd | tee log` returns TEE's status, so a dead roll reads as SUCCESS and the next
# stage runs on a partial dump. MEASURED here 2026-09-16: a killed roll printed the
# queue's DONE marker. (`never-chain-the-lander-behind-a-pipe`.)
set -o pipefail
R=/c/Users/Admin/refcv5cmp
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
HERE=$(dirname "$0")
OUTDIR=/c/Users/Admin/d3_out
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export PYTHONPATH="C:\\Users\\Admin\\refcv5cmp\\repo\\stack;C:\\Users\\Admin\\refcv5cmp\\repo\\taniteval;C:\\Users\\Admin\\refcv5cmp\\repo"

TAG="$1"; shift
MODE="$1"; shift
SEED="$1"; shift
mkdir -p "$OUTDIR"

cd "$R/repo"
"$PY" "$HERE/run_mask_roll.py" --mask-mode "$MODE" --mask-seed "$SEED" -- \
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
