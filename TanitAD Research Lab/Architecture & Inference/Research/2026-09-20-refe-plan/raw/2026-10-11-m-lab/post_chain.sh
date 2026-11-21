#!/bin/bash
# waits for the main chain, then runs the chunk probe and the aggregation (no model turns involved)
cd "$(dirname "$0")"
until grep -q ZZMLAB_CHAIN_DONE chain.log 2>/dev/null; do sleep 20; done
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 REFE_SIM_HZ=10 NUPLAN_MAPS_ROOT=E:/Archive/devbox-C/navsim/data/maps NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="E:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896;$(pwd -W)/../.."
C:/Users/Admin/navsim-crun/venv/Scripts/python.exe mlab_chunk.py > chunk_probe.log 2>&1
C:/Users/Admin/navsim-crun/venv/Scripts/python.exe mlab_summarize.py > summarize.log 2>&1
echo ZZMLAB_POST_DONE $(date +%FT%T) >> chain.log
