@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=2"
set "MKL_NUM_THREADS=2"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\band_labels
C:\Users\Admin\venvs\tanitad\Scripts\python.exe build_strategic_geom.py --split train --progress F:\TanitAD-scratch\hwm_bands\a\prog_train.txt > F:\strat_out\band_labels\build_strategic_geom_train.log 2>&1
if exist F:\strat_out\band_labels\strategic_geom_train.npz echo DONE > F:\strat_out\band_labels\strategic_geom.done
