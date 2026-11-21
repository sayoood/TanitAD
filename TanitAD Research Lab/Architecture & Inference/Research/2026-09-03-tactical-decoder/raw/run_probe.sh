set -e
SCR="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8e7cfa33-c625-47cf-88aa-711db80ac113/scratchpad"
export PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/colab;C:/Users/Admin/tanitad-wt/taniteval"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
LAB="C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz"
cd "$SCR/tacdec"
"$PY" intent_logit_probe.py --ckpt "C:/Users/Admin/refav1_eval_slice/ckpt/ckpt.pt" \
  --cache "C:/Users/Admin/refav1_eval_slice/fp8" --episodes "C:/Users/Admin/refav1_eval_slice/eps" \
  --labels "$LAB" --nav "$LAB" --name incumbent \
  --arm-tool-dir "C:/Users/Admin/tanitad-wt/taniteval/tools" \
  --episodes-n 20 --window-stride 10 --device cuda --n-boot 10000 \
  --out "$SCR/tacdec/intent_probe_incumbent.json" \
  --npz "$SCR/tacdec/intent_probe_incumbent.npz" > "$SCR/tacdec/probe_incumbent.log" 2>&1
echo "INCUMBENT EXIT $?"
"$PY" intent_logit_probe.py --ckpt "C:/Users/Admin/refav1_eval_slice/ckpt_ep2/ckpt.pt" \
  --config "C:/Users/Admin/refav1_eval_slice/ckpt_ep2/config.json" \
  --cache "C:/Users/Admin/refav1_eval_slice/fp8" --episodes "C:/Users/Admin/refav1_eval_slice/eps" \
  --labels "$LAB" --nav "$LAB" --name ep2 \
  --arm-tool-dir "C:/Users/Admin/tanitad-wt/taniteval/tools" \
  --episodes-n 20 --window-stride 10 --device cuda --n-boot 10000 \
  --out "$SCR/tacdec/intent_probe_ep2.json" \
  --npz "$SCR/tacdec/intent_probe_ep2.npz" > "$SCR/tacdec/probe_ep2.log" 2>&1
echo "EP2 EXIT $?"
echo ZZALLDONEZZ
