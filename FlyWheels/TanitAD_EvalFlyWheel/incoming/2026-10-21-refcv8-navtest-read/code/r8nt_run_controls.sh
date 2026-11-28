#!/bin/bash
# r8nt_run_controls.sh -- score the two known-value controls (STOP official; refcv7-50,400 LEGAL seam re-scored) on the Thor CPU backend
# through the SAME scorer path the refcv8 arms will use (r8nt_score.py -> score_navtest8_thor.py -> W3 cmd_score), then check them.
# CPU only (CUDA_VISIBLE_DEVICES=""), nice 19 / ionice idle via thor_pool.py (<= 4 scorers, MemAvailable >= 35 GB to launch).
# Markers: QQCTRL <json> (from the check) then QQCTRLS_END.
set -u
X=/home/nvidia/r8ns/navtest_read; NX=/home/nvidia/r8ns/code/navtest_read; K=$X/controls; S=$X/controls/scores_navtest
export OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
mkdir -p $S
python3 $NX/r8nt_score.py jobs --prep $X/prep --scores $S --tag ctrl --arm STOP=STOP --arm R7=$K/seam_R7_VMAXOFF.npz --out $K/jobs.jsonl
cd $S && python3 /home/nvidia/navsim-thor/code/thor/thor_pool.py --jobs $K/jobs.jsonl --state $K/pool_state.json > $K/pool.out 2>&1
python3 $NX/r8nt_score.py finalize --prep $X/prep --scores $S --tag ctrl --arm STOP=STOP --arm R7=$K/seam_R7_VMAXOFF.npz \
    --order-ref $K/STOP_navtest.csv > $K/finalize.out 2>&1
/home/nvidia/venvs/navsim-cpu/bin/python $NX/r8nt_controls_check.py $S ctrl $K/STOP_navtest.csv $K/r7thor_s50400_R7_VMAXOFF.csv \
    $K/CONTROLS.json
echo QQCTRLS_END
