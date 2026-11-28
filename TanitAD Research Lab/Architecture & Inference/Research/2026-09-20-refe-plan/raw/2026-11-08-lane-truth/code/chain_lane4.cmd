@echo off
rem chain 4 = stage 2 + stage 3 of the lane-truth chain, no wait (the findstr wait in chain 2 could never match a forward-slash path; CRUISE was already done)
set DZ=C:/Users/Admin/dz/DriveZero/DriveRL
set VPY=C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe
set DRIVERL_EVAL_ROOT=%DZ%
set DRIVERL_EVAL_NUPLAN_ROOT=%DZ%/nuplan-devkit
set DRIVERL_EVAL_RELEASE_ID=driverl-teacher-u2400
set DRIVERL_EVAL_CONFIG_PATH=%DZ%/release/configs/driverl_teacher.yaml
set DRIVERL_EVAL_CHECKPOINT_PATH=%DZ%/release/checkpoints/checkpoint_2400.pt
set DRIVERL_EVAL_PYTHON=%VPY%
set DRIVERL_EVAL_DEVICE=cpu
set CUDA_VISIBLE_DEVICES=-1
set NUPLAN_DATA_ROOT=E:/Projects/TanitAD/data/nuplan
set NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0
set REFE_NUPLAN_DB_ROOT=E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set REFE_SIM_HZ=10
set DRIVERL_EVAL_ROUTE_LANE_RANK=
set "REFE=E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
set W=F:/scratch/lane_truth_work/run
set L=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/lane_truth
cd /d "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
"%VPY%" "%REFE%/onpolicy_relabel_teacher_lane.py" --queue %W%/q_ctrl2_real --files "props_r0_*" --rank 0 --out %W%/out_ctrl2_real --workers 4 --repair-last-heading > %L%/logs/ctrl2_real.log 2>&1
"%VPY%" "%REFE%/onpolicy_relabel_teacher_lane.py" --queue %W%/q_ctrl2_ph --files "props_r0_*" --rank 0 --out %W%/out_ctrl2_ph --workers 4 --repair-last-heading > %L%/logs/ctrl2_ph.log 2>&1
echo CHAIN2_DONE > %L%/logs/CHAIN2_DONE
set REFE_MAP_CACHE=0
set REFE_QUERY_CACHE=0
"%VPY%" "%REFE%/onpolicy_relabel_teacher_lane.py" --queue %W%/q_ctrl2_real --files "props_r0_*" --rank 0 --out %W%/out_ctrl3_nocache --workers 4 --repair-last-heading --limit 8 > %L%/logs/ctrl3_nocache.log 2>&1
set REFE_MAP_CACHE=
set REFE_QUERY_CACHE=
"%VPY%" "%REFE%/onpolicy_relabel_teacher_lane.py" --queue %W%/q_ctrl2_real --files "props_r0_*" --rank 0 --out %W%/out_ctrl3_norepair --workers 4 --limit 8 > %L%/logs/ctrl3_norepair.log 2>&1
echo CHAIN3_DONE > %L%/logs/CHAIN3_DONE
