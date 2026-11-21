#!/usr/bin/env bash
# G-EVAL ONLY, dev box, CPU -- the refcv7 eval loader under the launch gate's own check.
# A copy of the Master Mind's C:/Users/Admin/qland/work/refcv7/probe_cost/run_dry_devbox.sh (not edited) with:
#   TREE      -> C:/lgt/r7ldr (tip 37086c3 + the box A17 overlay == b711411 code, + the two new files)
#   --checks  -> G-EVAL
#   --out-dir -> $OUT (fresh per run)
#   --eval-loader -> the NEW stack/tanitad/eval/refcv7_loader.py
#   --eval-stamps -> $STAMPS (a config.json written for THIS argv; or the labelled STAND-IN)
# No --git-dir: the tree is deliberately NOT the commit, so no token can bind. Evidence only.
# usage: STAMPS=<config.json> OUT=<fresh dir> bash run_geval_devbox.sh
set -u
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=-1
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
TREE=C:/lgt/r7ldr
IN=C:/Users/Admin/lg0926/inputs
OUT=${OUT:?set OUT to a fresh dir under C:/lgt/}
STAMPS=${STAMPS:?set STAMPS to a config.json written for this argv}
PM=()
while IFS= read -r l; do l="${l%$'\r'}"; [ -n "$l" ] && PM+=(--path-map "$l"); done < C:/Users/Admin/qland/work/refcv7/probe_cost/pathmap_refcv7.txt
"$PY" "$TREE/stack/scripts/launch_gate_refcv7.py" run --tree "$TREE" \
  --commit 37086c399cf209ad2bca8e0ffe19242c5e1e5f56 \
  --argv-file C:/Users/Admin/qland/work/refcv7/probe_cost/launch_argv_intended.json \
  --out-dir "$OUT" "${PM[@]}" --checks G-EVAL \
  --eval-loader "$TREE/stack/tanitad/eval/refcv7_loader.py" \
  --eval-kit D:/refcv6_eval_kit --eval-stamps "$STAMPS" \
  --eval-remap-overrides "$IN/eval_remap_overrides.json" \
  --clock-reference "$IN/q4c_grid_vs_egolog_ALLTRAIN.json" \
  --cpu-only --omp 4 --min-free-gb 6 --ram-wait-s 3600 > "$OUT.log" 2>&1
echo "rc=$? (not the verdict: read $OUT/evidence/G-EVAL.json)" >> "$OUT.log"
