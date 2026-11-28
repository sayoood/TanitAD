#!/bin/bash
# Chained: waits for the phase-1 driver to finish so Thor is not asked to hold a fifth
# concurrent GPU job (four is what preceded the 18:51 reboot).
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-aarch64-gnu/include/python3.12:$CPATH
export OMP_NUM_THREADS=6
SC=/home/nvidia/nurec_scenes/sample_set/26.04_release/00040136-e651-4abd-991d-0655ccda9430
AG=/home/nvidia/tanitad_cl/stack/experiments/alpasim-gsplat
L=/home/nvidia/rq_out/logs
until grep -q RSE_ALLDONE $L/status.txt 2>/dev/null; do sleep 20; done
cd $AG
set -x
python3 rs_frame_offset.py --scene-dir $SC --out /home/nvidia/rq_out/rs_frame_offset_k10.json --config chosen --k 10 --loader-dir /home/nvidia/nurec-gsplat > $L/frame_offset_k10.log 2>&1
echo "EXIT frameoffset_k10 = $?" >> $L/status.txt
echo RSE2_ALLDONE >> $L/status.txt
