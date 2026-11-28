export DZ="C:/Users/Admin/dz/DriveZero/DriveRL"
export VPY="C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
export DRIVERL_EVAL_ROOT="$DZ" DRIVERL_EVAL_NUPLAN_ROOT="$DZ/nuplan-devkit" DRIVERL_EVAL_RELEASE_ID=driverl-teacher-u2400
export DRIVERL_EVAL_CONFIG_PATH="$DZ/release/configs/driverl_teacher.yaml" DRIVERL_EVAL_CHECKPOINT_PATH="$DZ/release/checkpoints/checkpoint_2400.pt"
export DRIVERL_EVAL_PYTHON="$VPY" DRIVERL_EVAL_DEVICE=cpu
export NUPLAN_DATA_ROOT="D:/Projects/TanitAD/data/nuplan" NUPLAN_MAPS_ROOT="D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
export REFE_NUPLAN_DB_ROOT="D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export REFE_SIM_HZ="${REFE_SIM_HZ:-10}"
unset DRIVERL_EVAL_ROUTE_LANE_RANK
