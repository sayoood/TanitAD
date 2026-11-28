#!/bin/bash
set -u
W=/c/Users/Admin/tanitad-selq-20260906
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export TANITAD_REPO="$W"; export PYTHONPATH="$W/stack;$W/taniteval"; export OMP_NUM_THREADS=6
cd "$W"
EPN="${1:-0}"; OUT="${2}"; DUMP="${3}"; STRIDE="${4:-1}"
mkdir -p "$(dirname "$OUT")"
"$PY" -X utf8 taniteval/tools/refcv3_arm.py \
  --ckpt /c/Users/Admin/refcv4b_final/ckpt_40284_FINAL.pt \
  --config /c/Users/Admin/refcv4b_final/config.json \
  --episodes /c/Users/Admin/refav1_eval_full/eps \
  --labels /c/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz \
  --grid 2s --arm refcv4b-40284-local --episodes-n "$EPN" \
  --window-stride "$STRIDE" \
  --out "$OUT" --dump-dir "$DUMP" \
  --device cuda --lru 6 --n-boot 200 --seed 0 \
  --no-navshuf --no-navzero --no-lead-block
echo "ARM_EXIT=$?"
