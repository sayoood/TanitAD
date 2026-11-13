#!/bin/bash
# RJ5 navtrain bulk roll: 877 logs fetched from the public nuPlan bucket (PI 2026-11-13 "you can download"), half B, 2 at a time to F:/tanitad_scratch/rj5nt/dbB (pinned-run floor 20 GB on F:),
# rolled, deleted. Starts when the local labelling job has ended. Re-running resumes.
export PATH="/usr/bin:/bin:$PATH"
export PYTHONPATH="F:/tanitad_scratch/rj5nt/w/patch;C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit;C:/Users/Admin/dz/DriveZero/DriveRL/src;F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe;F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/code;F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval"
export NUPLAN_MAPS_ROOT="E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0" NUPLAN_DATA_ROOT="E:/Projects/TanitAD/data/nuplan" REFE_NUPLAN_DB_ROOT="E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
export REFE_NAVTRAIN_YAML="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
export DZ_ROOT=C:/Users/Admin/dz/DriveZero/DriveRL DRIVERL_EVAL_ROOT=C:/Users/Admin/dz/DriveZero/DriveRL DRIVERL_EVAL_NUPLAN_ROOT=C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit DRIVERL_EVAL_CONFIG_PATH=C:/Users/Admin/dz/DriveZero/DriveRL/release/configs/driverl_teacher.yaml DRIVERL_EVAL_CHECKPOINT_PATH=C:/Users/Admin/dz/DriveZero/DriveRL/release/checkpoints/checkpoint_2400.pt
export REFE_SIM_HZ=10 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
unset DRIVERL_EVAL_ROUTE_LANE_RANK
export REFE_CD_CACHE=F:/tanitad_scratch/rj5nt/cdcache
L=F:/tanitad_scratch/rj5nt/label_local/label.log
while ! grep -qE 'ZZRJ5_LABEL_(DONE|STOP)' "$L"; do sleep 60; done
while [ "$(/c/Users/Admin/venvs/driverl-eval/Scripts/python.exe -c 'import shutil;print(int(shutil.disk_usage("F:/").free/2**30*10))' | tr -d '\r')" -lt 215 ]; do sleep 300; done
echo "$(date -u +%T) label ended; starting bulk chain" >> F:/tanitad_scratch/rj5nt/run_fetch_B/launch.log
cd F:/tanitad_scratch/rj5nt/code
exec /c/Users/Admin/venvs/driverl-eval/Scripts/python.exe rj5_chain.py --bank F:/tanitad_scratch/rj5nt/bank_p7_rank0_slim.jsonl.gz --logs-file F:/tanitad_scratch/rj5nt/logs_fetch_B.txt --work F:/tanitad_scratch/rj5nt/run_fetch_B --db-dir F:/tanitad_scratch/rj5nt/dbB --fetch --fetch-script "F:/tanitad_scratch/rj5nt/w/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/code/fetch_navtrain_dbs.py" --procs 2 --batch-logs 2 --arms RJ5 --control-every 20 --min-free-gb 20.6 --est-gb-per-log 0.35 >> F:/tanitad_scratch/rj5nt/run_fetch_B/chain.log 2>&1 < /dev/null
