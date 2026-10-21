#!/usr/bin/env bash
# H-NT-DECIDE-1: waits for E1's DONE_A1.json (and for thor_build_a1.sh's BUILD_DONE), then writes the FINAL label
# release over E1's written token set (--e1-manifest) + the final clock reference, into /home/nvidia/r8ns/nt/labels/a1/.
# nice 19, 1 core. v2 (2026-10-21 ~23:10): every python step inside a KERNEL memory cap (the 22:50 global OOM),
# peak recorded by peak.py, and the clock reference GATED on the labels' EXIT.
set -u
NT=/home/nvidia/r8ns/nt
L=$NT/labels
W=$L/a1_build
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOGS=/home/nvidia/r8ns/navtrain/navsim_logs/trainval
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export PYTHONPATH="$W/tree/stack:$W/tree/stack/scripts" NT_TREE="$W/tree"
cd "$W" || exit 2
cap() { local M=$1; shift; systemd-run --user --scope -q -p MemoryMax="$M" -p MemorySwapMax=0 nice -n 19 "$@" 200>&-; }
while [ ! -s "$NT/DONE_A1.json" ] || [ ! -s logs/BUILD_DONE ]; do sleep 120 200>&-; done
cap 16G taskset -c 13 $PY code/peak.py labels_final logs -- $PY -X utf8 code/nt_labels.py --tree "$W/tree" \
  --logs-dir "$LOGS" --navtrain-filter "$W/inputs/navtrain_filter_v11.json" \
  --heldout "$W/tree/stack/tanitad/configs/refcv8_nt/nt_heldout.json" \
  --e1-manifest "$NT/MANIFEST_A1.jsonl" \
  --merge-clock /home/nvidia/data/refcv8_clip_clock_sidecar_ts33.jsonl --out "$L/a1" > logs/labels_final.log 2>&1
echo $? > logs/EXIT_labels_final
[ "$(cat logs/EXIT_labels_final)" = 0 ] && [ -s "$L/a1/v9_labels_nt_train.npz" ] || { echo "labels_final failed" > logs/FINAL_FAILED; exit 3; }
cap 8G taskset -c 13 $PY code/peak.py clockref_final logs -- $PY -X utf8 code/nt_clock_reference.py --logs-dir "$LOGS" \
  --token-table "$L/a1/token_table_nt.json" --e1-manifest "$NT/MANIFEST_A1.jsonl" \
  --out "$L/a1/nt_clock_reference_a1.json" > logs/clockref_final.log 2>&1
echo $? > logs/EXIT_clockref_final
md5sum "$L/a1/v9_labels_nt_train.npz" > logs/V9_MD5_FINAL 2>&1
echo DONE > logs/FINAL_DONE
