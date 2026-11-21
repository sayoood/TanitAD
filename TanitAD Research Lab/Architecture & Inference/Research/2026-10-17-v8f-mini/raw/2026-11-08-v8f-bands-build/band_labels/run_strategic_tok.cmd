@echo off
set "CUDA_VISIBLE_DEVICES=-1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=F:\TanitAD-scratch\ladder_tree\stack;F:\TanitAD-scratch\ladder_tree\taniteval"
cd /d E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\band_labels
C:\Users\Admin\venvs\tanitad\Scripts\python.exe build_strategic_tok.py --split eval > F:\strat_out\band_labels\build_strategic_tok_eval.log 2>&1
C:\Users\Admin\venvs\tanitad\Scripts\python.exe build_strategic_tok.py --split train --progress F:\TanitAD-scratch\hwm_bands\a\prog_tok_train.txt > F:\strat_out\band_labels\build_strategic_tok_train.log 2>&1
if exist F:\strat_out\band_labels\strategic_tok_train.npz echo DONE > F:\strat_out\band_labels\strategic_tok.done
