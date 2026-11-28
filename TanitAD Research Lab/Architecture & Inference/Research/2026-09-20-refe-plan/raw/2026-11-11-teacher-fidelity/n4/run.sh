#!/bin/bash
# usage: run.sh script.py [args]   (NAVSIM venv python, pinned devkit 3e8291b, CPU only)
export PYTHONPATH="F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896;F:/tanitad_scratch/n4dac/w/refe"
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8
exec C:/Users/Admin/navsim-crun/venv/Scripts/python.exe "$@"
