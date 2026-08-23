#!/bin/bash
set -x
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12:$CPATH
export OMP_NUM_THREADS=6
SC=/home/nvidia/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
AG=/home/nvidia/tanitad_cl/stack/experiments/alpasim-gsplat
L=/home/nvidia/rq_out/logs
ST=$L/status.txt
snap () { echo "--- $1 $(date -u +%FT%TZ) ---"; ps -eo pid,etime,args --sort=-rss | grep -E 'python' | grep -v grep | head -5; timeout 3 tegrastats --interval 1000 2>/dev/null | head -1; free -g | sed -n 2p; }
cd $AG
snap BEFORE_COST > $L/contention.log 2>&1
python3 rs_cost_probe.py --scene-dir $SC --out /home/nvidia/rq_out/rs_cost_probe.json --config chosen --frames 0,150,300,450 --reps 3 --loader-dir /home/nvidia/nurec-gsplat > $L/cost_probe.log 2>&1
echo "EXIT costprobe = $?" >> $ST
snap AFTER_COST >> $L/contention.log 2>&1
python3 rs_frame_offset.py --scene-dir $SC --out /home/nvidia/rq_out/rs_frame_offset.json --config chosen --k 3 --loader-dir /home/nvidia/nurec-gsplat > $L/frame_offset.log 2>&1
echo "EXIT frameoffset = $?" >> $ST
python3 rs_sweep.py --scene-dir $SC --out /home/nvidia/rq_out/rs_utgate_chosen --config chosen --panel utgate --n-frames-auto 12 --png-frame 0 --loader-dir /home/nvidia/nurec-gsplat --repo /home/nvidia/tanitad_cl > $L/utgate_chosen.log 2>&1
echo "EXIT utgate = $?" >> $ST
python3 rs_seam_control.py --scene-dir $SC --out /home/nvidia/rq_out/rs_seam_control.json --config chosen --loader-dir /home/nvidia/nurec-gsplat > $L/seam.log 2>&1
echo "EXIT seam = $?" >> $ST
python3 rs_sweep.py --scene-dir $SC --out /home/nvidia/rq_out/rs_batch_chosen --config chosen --panel batch --n-frames-auto 12 --png-frame 0 --loader-dir /home/nvidia/nurec-gsplat --repo /home/nvidia/tanitad_cl > $L/batch.log 2>&1
echo "EXIT batch = $?" >> $ST
python3 /home/nvidia/rq_out/rs_regression_check.py > $L/regression.log 2>&1
echo "EXIT regression = $?" >> $ST
cd /home/nvidia/nurec-gsplat
python3 render_probe.py --scene-dir $SC --frame 0 --layers background,road --rolling-shutter --out /home/nvidia/rq_out/rp_rs_check > $L/rp_rs_check.log 2>&1
echo "EXIT render_probe_rs = $?" >> $ST
snap END >> $L/contention.log 2>&1
echo RSE_ALLDONE >> $ST
