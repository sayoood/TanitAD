cd F:/tanitad_scratch/instr/val
while [ ! -f full_done.txt ]; do sleep 20; done
export PYTHONPATH="F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;F:\tanitad_scratch\instr\w\refe"
export CUDA_VISIBLE_DEVICES=-1
V=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
I=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/instruments
rm -f F:/tanitad_scratch/instr/val/rerun_r40.jsonl
$V $I/transp_score.py --refe F:/tanitad_scratch/instr/w/refe --plans T=plans_T_r40.npz Tc20s=plans_Tc20s_r40.npz Tw=plans_Tw_r40.npz --out rerun_r40.jsonl --controls --human-arm > rerun_r40.log 2>&1
unset PYTHONPATH
$V $I/validate_controls.py --rows "full_s*.jsonl" --tf1 "tf1rows/tf1_s[0-3].jsonl" tf1rows/tf1_s4r.jsonl --p5s F:/tanitad_scratch/tf_design/w/raw/p5s_cubic.npz --rerun rerun_r40.jsonl --out controls.json > controls.log 2>&1
echo FINISHED > finish_done.txt
