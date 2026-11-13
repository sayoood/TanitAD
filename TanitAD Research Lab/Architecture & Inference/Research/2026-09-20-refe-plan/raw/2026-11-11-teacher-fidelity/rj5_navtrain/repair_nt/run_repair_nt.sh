#!/bin/bash
# REPAIR-D (TF-N4 A1 settings, pinned run_a1 code, read-only) on the RJ5 navtrain P5-R labels of the 198 local logs: builder tracked score,
# repair where DAC = 0. Descriptive dev-set construction for the REPAIR-D successor (navtrain only; no navtest token).
export PATH="/usr/bin:/bin:$PATH"
O=F:/tanitad_scratch/rj5nt/repair_nt; L=F:/tanitad_scratch/rj5nt/label_local/label.log; RUN=F:/tanitad_scratch/n4nt/run_a1
while ! grep -qE 'ZZRJ5_LABEL_(DONE|STOP)' "$L"; do sleep 60; done
grep -q ZZRJ5_LABEL_DONE "$L" || { echo "$(date -u +%T) ZZREPNT_STOP label not done" >> $O/repair.log; exit 1; }
PY=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 REFE_SIM_HZ=10 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1
export NUPLAN_MAPS_ROOT=E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0 TANITAD_NAVSIM_ROOT=F:/Archive/devbox-C/navsim
export N4_W=$RUN/w/pk N4_P5R=$RUN/w/p5r
export PYTHONPATH='F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;F:\tanitad_scratch\n4nt\run_a1\w\pk\refe'
echo "$(date -u +%T) inputs" >> $O/repair.log
/c/Users/Admin/venvs/driverl-eval/Scripts/python.exe $O/mk_inputs.py F:/tanitad_scratch/rj5nt/label_local/rj5_labels_local.jsonl.gz F:/tanitad_scratch/rj5nt/merge_local/rj5_rank0.jsonl.gz $O/nt_plans.npz $O/nt_rolls.jsonl >> $O/repair.log 2>&1 || { echo "$(date -u +%T) ZZREPNT_STOP inputs" >> $O/repair.log; exit 1; }
cd $RUN/n4
for s in 0 1; do
  $PY -B n4_nt_build.py derive --route navtrain --n2r-plans $O/nt_plans.npz --rolls $O/nt_rolls.jsonl --out $O/derive_s${s}of2.jsonl --shard $s/2 > $O/derive_s$s.log 2>&1 < /dev/null &
  eval "p$s=\$!"
done
wait $p0; r0=$?; wait $p1; r1=$?
if [ $r0 -eq 0 ] && [ $r1 -eq 0 ]; then echo "$(date -u +%T) ZZREPNT_DONE" >> $O/repair.log; else echo "$(date -u +%T) ZZREPNT_STOP rcs $r0 $r1" >> $O/repair.log; fi
