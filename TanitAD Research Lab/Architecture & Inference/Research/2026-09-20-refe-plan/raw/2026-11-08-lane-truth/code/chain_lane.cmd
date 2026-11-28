@echo off
rem lane-truth chain: control (placeholder teacher) -> CRUISE 154 tokens -> other 904 tokens. CPU only, 4 workers.
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
"%VPY%" "%REFE%/onpolicy_relabel_teacher_lane.py" --queue %W%/q_ctrl_ph --files "props_r0_*" --rank 0 --out %W%/out_ctrl_ph --workers 4 --repair-last-heading > %L%/logs/ctrl_ph.log 2>&1
"%VPY%" "%REFE%/onpolicy_relabel_teacher_lane.py" --queue %W%/q_cruise --files "props_r0_*" --rank 0 --out %W%/out_cruise --workers 4 --repair-last-heading > %L%/logs/cruise.log 2>&1
"%VPY%" "%REFE%/onpolicy_relabel_teacher_lane.py" --queue %W%/q_other --files "props_r0_*" --rank 0 --out %W%/out_other --workers 4 --repair-last-heading > %L%/logs/other.log 2>&1
echo CHAIN_DONE > %L%/logs/CHAIN_DONE
