#!/usr/bin/env bash
# The gate on the CURRENT tip (5de9363 + the four unlanded gate files) with refcv6-r101-s0's own
# argv, dev box, CPU, RAM-gated at the brief's 8 GB floor. Each job's verdict is its evidence
# JSON; this script only sequences them (one heavy job at a time).
#   TREE=<tree> OUT=<gate dir> ./run_tip_evidence.sh <checks>
set -u
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
IN=C:/Users/Admin/lg0926/inputs
TREE="${TREE:?}"; OUT="${OUT:?}"; CHECKS="${1:?checks}"
PM=()
while IFS= read -r l; do [ -n "$l" ] && PM+=(--path-map "$l"); done < "$IN/pathmap.txt"
export PYTHONIOENCODING=utf-8
# ⛔ MSYS rewrites POSIX-looking args (`/home/nvidia/...=D:/...`) for native python
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
"$PY" "$TREE/stack/scripts/launch_gate.py" run --profile "${PROFILE:-refcv6}" --checks "$CHECKS" \
  --tree "$TREE" --commit 0000000000000000000000000000000000000000 \
  --argv-file "${ARGV_FILE:-$IN/argv_refcv6_r101_s0.json}" --out-dir "$OUT" "${PM[@]}" \
  --eval-loader C:/Users/Admin/ev6_battery/code/refcv6_loader.py --eval-kit D:/refcv6_eval_kit \
  --eval-stamps D:/refcv6_eval_kit/ckpt_final/config.json \
  --eval-remap-overrides "$IN/eval_remap_overrides.json" \
  --clock-reference "$IN/q4c_grid_vs_egolog_ALLTRAIN.json" \
  --clock-manifest train=C:/Users/Admin/qland/work/a16_switch/train_v2manifest.pt \
  --cpu-only --omp 4 --min-free-gb 8 --ram-wait-s 21600 \
  --key-file C:/Users/Admin/lg0926/keys/rehearsal.key ${EXTRA:-}
echo "ZZTIPRUN-$(basename "$OUT")-DONEZZ"
