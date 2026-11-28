#!/usr/bin/env bash
# H-NT-DECIDE-1 E2: the provisional A1 label release WITH the FOLLOW-evidence (lead) pass (nt_labels.py 673af445,
# --lead anns, the default) into labels/a1_prov_lead/ -- beside a1_prov/ (v1, lead=None), which is NOT touched:
# phase A's controls and the joins read a1_prov. Purpose: measure the production lon distribution + wall/peak before
# thor_finalize_a1_v2.sh builds the FINAL release with the same code (~03:55). Same caps / core as labels_prov.
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
cap 16G taskset -c 11 $PY code/peak.py labels_prov_lead logs -- $PY -X utf8 code/nt_labels.py --tree "$W/tree" \
  --logs-dir "$LOGS" --navtrain-filter "$W/inputs/navtrain_filter_v11.json" \
  --heldout "$W/tree/stack/tanitad/configs/refcv8_nt/nt_heldout.json" --lead anns \
  --merge-clock /home/nvidia/data/refcv8_clip_clock_sidecar_ts33.jsonl --out "$L/a1_prov_lead" > logs/labels_prov_lead.log 2>&1
echo $? > logs/EXIT_labels_prov_lead
