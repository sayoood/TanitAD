#!/bin/bash
# run_e0bc_thor.sh -- E0b + E0c on Thor (CPU only; no GPU touched).  Marker: ZZE0BCdone-<rc>ZZ in /home/nvidia/r8ns/r9e0/e0bc.log.  Resume = re-run; result = /home/nvidia/r8ns/r9e0/E0BC_RESULT.json
export OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8
D=/home/nvidia/r8ns/r9e0
PY=/home/nvidia/venvs/tanitad-train/bin/python
mkdir -p $D/scratch
ls /home/nvidia/data/refcv6-b1-416x1024-train | sed 's/\.v2ep\.pt$//' > $D/scratch/ids.txt
$PY -X utf8 $D/e0bc_census.py --clip-list $D/scratch/ids.txt --clip-index /home/nvidia/data/physicalai-b1/clip_index.parquet \
   --ego-dir /home/nvidia/data/physicalai-b1/labels/egomotion --labels /home/nvidia/refcv8_v9labels/release_train4572/v9_labels_train.npz \
   --out $D/E0BC_RESULT.json > $D/e0bc_run.log 2>&1 < /dev/null
rc=$?
rm -f $D/scratch/ids.txt
echo "ZZE0BCdone-${rc}ZZ" >> $D/e0bc.log
