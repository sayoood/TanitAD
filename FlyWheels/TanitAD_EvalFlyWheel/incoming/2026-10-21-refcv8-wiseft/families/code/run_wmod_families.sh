#!/bin/sh
# Four-family paired NavSim reads (navtrain sample) of the R8-WISEFT-MOD arms (beside; never a bar).  2026-10-21 17:12 Berlin.  CPU only.
F=C:/Users/Admin/refcv8_pf/navsim_ckpt/families_navtrain; S=$F/seams; INP=C:/Users/Admin/refcv8_pf/navsim_ckpt/run/inputs/navtrain_sample_inputs.json.gz
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONPATH="D:/Projects/TanitAD/stack;D:/Projects/TanitAD/taniteval;D:/Projects/TanitAD/taniteval/tools" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
run() { "$PY" -X utf8 $F/code/paired_families_navsim.py --inputs $INP --seam-a "$1" --seam-b "$2" --name-a "$3" --name-b "$4" --label "$5" --out "$F/out/$6.json" > "$F/out/$6.log" 2>&1; echo "$6 rc=$?"; }
for s in "" _s1; do sd=s0; [ -n "$s" ] && sd=s1
  run $S/ema48852/bridge/seam_R8_LEGAL$s.npz $S/wiseft/wmod_PLAN/bridge/seam_R8_LEGAL$s.npz EMA48852_$sd WPLAN_$sd "R8-WISEFT-MOD theta_PLAN minus EMA-48,852, navtrain $sd (beside)" pf_WPLAN_minus_EMA_$sd
  run $S/ema48852/bridge/seam_R8_LEGAL$s.npz $S/wiseft/wmod_PERC/bridge/seam_R8_LEGAL$s.npz EMA48852_$sd WPERC_$sd "R8-WISEFT-MOD theta_PERC minus EMA-48,852, navtrain $sd (beside)" pf_WPERC_minus_EMA_$sd
done
run $S/wiseft/wise_l0p50/bridge/seam_R8_LEGAL.npz $S/wiseft/wmod_PLAN/bridge/seam_R8_LEGAL.npz W0p50_s0 WPLAN_s0 "theta_PLAN minus theta_0.5, navtrain s0 (beside)" pf_WPLAN_minus_W0p50_s0
echo ZZWMFAMEND
