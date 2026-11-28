#!/usr/bin/env bash
# H-NT-DECIDE-1: waits for E1's DONE_A1.json (and for thor_build_a1.sh), then writes the FINAL label release over E1's
# written token set (--e1-manifest) + the final clock reference, into /home/nvidia/r8ns/nt/labels/a1/. nice 19, 1 core.
set -u
NT=/home/nvidia/r8ns/nt
L=$NT/labels
W=$L/a1_build
PY=/home/nvidia/venvs/tanitad-train/bin/python
LOGS=/home/nvidia/r8ns/navtrain/navsim_logs/trainval
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export PYTHONPATH="$W/tree/stack:$W/tree/stack/scripts"
cd "$W" || exit 2
while [ ! -s "$NT/DONE_A1.json" ] || [ ! -s logs/BUILD_DONE ]; do sleep 120 200>&-; done
nice -n 19 taskset -c 13 $PY -X utf8 code/nt_labels.py --tree "$W/tree" --logs-dir "$LOGS" \
  --navtrain-filter "$W/inputs/navtrain_filter_v11.json" --heldout "$W/tree/stack/tanitad/configs/refcv8_nt/nt_heldout.json" \
  --e1-manifest "$NT/MANIFEST_A1.jsonl" \
  --merge-clock /home/nvidia/data/refcv8_clip_clock_sidecar_ts33.jsonl --out "$L/a1" > logs/labels_final.log 2>&1
echo $? > logs/EXIT_labels_final
nice -n 19 taskset -c 13 $PY -X utf8 code/nt_clock_reference.py --logs-dir "$LOGS" \
  --token-table "$L/a1/token_table_nt.json" --e1-manifest "$NT/MANIFEST_A1.jsonl" \
  --out "$L/a1/nt_clock_reference_a1.json" > logs/clockref_final.log 2>&1
echo $? > logs/EXIT_clockref_final
md5sum "$L/a1/v9_labels_nt_train.npz" > logs/V9_MD5_FINAL 2>&1
echo DONE > logs/FINAL_DONE
