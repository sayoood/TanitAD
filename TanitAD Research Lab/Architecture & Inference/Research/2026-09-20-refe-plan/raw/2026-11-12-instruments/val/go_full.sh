cd F:/tanitad_scratch/instr/val
export PYTHONPATH="F:\Archive\devbox-C\navsim\navsim-3e8291bfa89ff247231e0227778840cd0a036896;F:\tanitad_scratch\instr\w\refe"
export CUDA_VISIBLE_DEVICES=-1
V=C:/Users/Admin/navsim-crun/venv/Scripts/python.exe
S=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/instruments/transp_score.py
for i in 0 1; do $V $S --refe F:/tanitad_scratch/instr/w/refe --plans T=plans_T.npz Tc20s=plans_Tc20s.npz Tw=plans_Tw.npz --out full_s$i.jsonl --shard $i/2 --controls --human-arm > full_s$i.log 2>&1 & done
wait
echo DONE > full_done.txt
