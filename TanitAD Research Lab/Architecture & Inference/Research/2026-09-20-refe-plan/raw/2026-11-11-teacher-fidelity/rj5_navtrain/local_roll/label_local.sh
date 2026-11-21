#!/bin/bash
# MM 2026-11-13: label the 16,133 local RJ5 rows with the UNCHANGED dual labeller (Tc20s + P5-R lambda 3.0), AFTER the TF-N5 A1 chain ends
# (it has RAM priority). Logs to F:/tanitad_scratch/rj5nt/label_local/label.log; end marker ZZRJ5_LABEL_DONE / ZZRJ5_LABEL_STOP.
export PATH="/usr/bin:/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 P5R_BLAS_THREADS=1 REFE_SIM_HZ=10 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export NUPLAN_MAPS_ROOT="E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
export P5C_PKG="F:/tanitad_scratch/n4nt/run_a1/w/pk" P5R_DIR="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r" P5R_LAMBDA_JSON="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r/raw/lambda_star.json"
export P5C_BANKDIR="F:/tanitad_scratch/rj5nt/label_local/bank" P5C_DBROOT="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
export PYTHONPATH="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
PY=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
DL="F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-11-11-p5c-readiness/a1/p5c_dual_labels.py"
O=F:/tanitad_scratch/rj5nt/label_local; L=$O/label.log; N5=F:/tanitad_scratch/n5nt/run/n5run/chain.log
mkdir -p $O/bank $O/labels
echo "$(date +%H:%M:%S) waiting for TF-N5 chain end" >> $L
until grep -qE "ZZN5_CHAIN (DONE|STOP)" $N5; do sleep 120; done
echo "$(date +%H:%M:%S) TF-N5 ended; prep" >> $L
cd F:/tanitad_scratch/rj5nt/code
$PY rj5_label.py prep --rj5 F:/tanitad_scratch/rj5nt/merge_local/rj5_rank0.jsonl.gz --out-bank $O/bank --logs-file F:/tanitad_scratch/rj5nt/logs_devbox_local.txt >> $L 2>&1 || { echo "$(date +%H:%M:%S) ZZRJ5_LABEL_STOP prep" >> $L; exit 1; }
$PY -B "$DL" label --shard 0 --nshard 2 --out $O/labels > $O/label_s0.log 2>&1 < /dev/null &
p0=$!
sleep 30
$PY -B "$DL" label --shard 1 --nshard 2 --out $O/labels > $O/label_s1.log 2>&1 < /dev/null &
p1=$!
wait $p0; r0=$?; wait $p1; r1=$?
echo "$(date +%H:%M:%S) label rcs $r0 $r1" >> $L
$PY rj5_label.py merge --rj5 F:/tanitad_scratch/rj5nt/merge_local/rj5_rank0.jsonl.gz --labels-dir $O/labels --out $O/rj5_labels_local.jsonl.gz --logs-file F:/tanitad_scratch/rj5nt/logs_devbox_local.txt >> $L 2>&1
rm_rc=$?
if [ $r0 -eq 0 ] && [ $r1 -eq 0 ] && [ $rm_rc -eq 0 ]; then echo "$(date +%H:%M:%S) ZZRJ5_LABEL_DONE" >> $L; else echo "$(date +%H:%M:%S) ZZRJ5_LABEL_STOP rcs $r0 $r1 merge $rm_rc" >> $L; fi
