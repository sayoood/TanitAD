set -e
SCR="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8e7cfa33-c625-47cf-88aa-711db80ac113/scratchpad"
export PYTHONPATH="C:/Users/Admin/tanitad-wt/stack;C:/Users/Admin/tanitad-wt/colab;C:/Users/Admin/tanitad-wt/taniteval"
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
LAB="C:/Users/Admin/tanitad-wt/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz"
BANKED="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-tactical-decoder/raw/turn_decomposition_A.json"
cd "$SCR/e0"
"$PY" e0_goalspace_probe.py \
  --ckpt "C:/Users/Admin/refav1_eval_slice/ckpt_ep2/ckpt.pt" \
  --config "C:/Users/Admin/refav1_eval_slice/ckpt_ep2/config.json" \
  --cache "C:/Users/Admin/refav1_eval_slice/fp8" \
  --episodes "C:/Users/Admin/refav1_eval_slice/eps" \
  --labels "$LAB" --nav "$LAB" --name ep2 \
  --arm-tool-dir "C:/Users/Admin/tanitad-wt/taniteval/tools" \
  --episodes-n 20 --window-stride 10 --device cuda \
  --banked-turn-decomposition "$BANKED" \
  --out "$SCR/e0/e0_goalspace_ep2.json" \
  --raw-out "$SCR/e0/e0_goalspace_ep2_rows.json" > "$SCR/e0/e0_ep2.log" 2>&1
echo "EP2 EXIT $?"
echo ZZE0DONEZZ
