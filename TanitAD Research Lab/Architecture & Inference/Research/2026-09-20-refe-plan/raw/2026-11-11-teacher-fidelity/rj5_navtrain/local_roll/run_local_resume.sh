#!/bin/bash
export PATH="/usr/bin:/bin:$PATH"
export PYTHONPATH="F:/tanitad_scratch/rj5nt/w/patch;C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit;C:/Users/Admin/dz/DriveZero/DriveRL/src;F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe;F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/code;F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
export NUPLAN_MAPS_ROOT="E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0" NUPLAN_DATA_ROOT="E:/Projects/TanitAD/data/nuplan" REFE_NUPLAN_DB_ROOT="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
export REFE_NAVTRAIN_YAML="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
export DZ_ROOT=C:/Users/Admin/dz/DriveZero/DriveRL DRIVERL_EVAL_ROOT=C:/Users/Admin/dz/DriveZero/DriveRL DRIVERL_EVAL_NUPLAN_ROOT=C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit DRIVERL_EVAL_CONFIG_PATH=C:/Users/Admin/dz/DriveZero/DriveRL/release/configs/driverl_teacher.yaml DRIVERL_EVAL_CHECKPOINT_PATH=C:/Users/Admin/dz/DriveZero/DriveRL/release/checkpoints/checkpoint_2400.pt
export REFE_SIM_HZ=10 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
unset DRIVERL_EVAL_ROUTE_LANE_RANK
cd F:/tanitad_scratch/rj5nt/code
# MM 2026-11-13 ~08:35Z: resume after TF-N4 A1 freed RAM. Batch 0 was started at procs 1 (tags b0000g0s0, shard 0/1); finish it at
# procs 1 so its shard files stay consistent, then every later batch at procs 2 (fresh tags per batch, so no cross-shard duplicates).
PYX=/c/Users/Admin/venvs/driverl-eval/Scripts/python.exe
COMMON="rj5_chain.py --bank F:/tanitad_scratch/rj5nt/bank_p7_rank0_slim.jsonl.gz --logs-file F:/tanitad_scratch/rj5nt/logs_devbox_local.txt --work F:/tanitad_scratch/rj5nt/run_local --db-dir F:/tanitad_scratch/rj5nt/db --held-db-dir E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/trainval --batch-logs 12 --arms RJ5 --control-every 20"
echo "$(date +%H:%M:%S) MM_RESUME phase1 procs=1 max-batches=1" >> F:/tanitad_scratch/rj5nt/run_local/chain.log
$PYX $COMMON --procs 1 --max-batches 1 >> F:/tanitad_scratch/rj5nt/run_local/chain.log 2>&1 < /dev/null
rc=$?
echo "$(date +%H:%M:%S) MM_RESUME phase1 rc=$rc" >> F:/tanitad_scratch/rj5nt/run_local/chain.log
if [ -f F:/tanitad_scratch/rj5nt/run_local/batches/batch_0000.done ]; then
  echo "$(date +%H:%M:%S) MM_RESUME phase2 procs=2" >> F:/tanitad_scratch/rj5nt/run_local/chain.log
  exec $PYX $COMMON --procs 2 >> F:/tanitad_scratch/rj5nt/run_local/chain.log 2>&1 < /dev/null
fi
echo "$(date +%H:%M:%S) MM_RESUME STOP batch 0 not done" >> F:/tanitad_scratch/rj5nt/run_local/chain.log
