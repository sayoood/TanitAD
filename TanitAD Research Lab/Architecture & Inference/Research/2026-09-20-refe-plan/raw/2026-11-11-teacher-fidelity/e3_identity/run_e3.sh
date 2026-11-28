#!/usr/bin/env bash
# usage: run.sh OUT FILTER DBSUB LIMIT NWORKERS
OUT=$1; FILTER=$2; DBSUB=$3; LIMIT=$4; NW=$5
R=C:/Users/Admin/dz/DriveZero/DriveRL
export PYTHONPATH="$R/nuplan-devkit;$R/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= HYDRA_FULL_ERROR=1
export NUPLAN_DATA_ROOT=E:/Projects/TanitAD/data/nuplan
export NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0
export NUPLAN_EXP_ROOT=F:/tanitad_scratch/tf_e3/exp
LIM=(); [ "$LIMIT" != "0" ] && LIM=(scenario_filter.limit_total_scenarios=$LIMIT)
C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe $R/nuplan-devkit/nuplan/planning/script/run_simulation.py \
 +simulation=closed_loop_nonreactive_agents_driverl scenario_builder=nuplan \
 scenario_builder.db_files=E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/$DBSUB \
 scenario_filter=$FILTER scenario_filter.shuffle=false "${LIM[@]}" \
 worker=ray_distributed worker.use_distributed=false worker.threads_per_node=$NW worker.log_to_driver=false distributed_mode=SINGLE_NODE \
 number_of_gpus_allocated_per_simulation=0.0 number_of_cpus_allocated_per_simulation=1 max_callback_workers=0 disable_callback_parallelization=true \
 ~callback.simulation_log_callback \
 planner=driverl_nuplan_planner ego_controller=driverl_one_stage_controller \
 planner.driverl_nuplan_planner.device=cpu \
 planner.driverl_nuplan_planner.config_path=$R/release/configs/driverl_teacher.yaml \
 planner.driverl_nuplan_planner.checkpoint_path=$R/release/checkpoints/checkpoint_2400.pt \
 planner.driverl_nuplan_planner.route_goal_horizon_s=12.0 planner.driverl_nuplan_planner.route_goal_min_speed_mps=5.0 \
 planner.driverl_nuplan_planner.route_goal_pair_mode=legacy planner.driverl_nuplan_planner.tts_enabled=false \
 planner.driverl_nuplan_planner.tts_num_candidates=8 planner.driverl_nuplan_planner.tts_seed=42 \
 ego_controller.device=cpu ego_controller.nuplan_bicycle_max_acceleration=4.0 ego_controller.nuplan_bicycle_max_steering_rate=0.8 \
 output_dir=$OUT experiment_name=closed_loop_nonreactive_agents_driverl_e3_$(basename $OUT) exit_on_failure=true ~main_callback.metric_summary_callback
