#!/bin/sh
# L3+L4 — the chord/weight panel on BOTH banked step-1,000 checkpoints,
# the same 140 windows / 20 episodes / stride 10 as every banked refav1 read.
# Dev-box 4060, forward-only. Thor and the A40 pod are NOT contacted.
set -e
SCR="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8e7cfa33-c625-47cf-88aa-711db80ac113/scratchpad/costrepair"
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
WT="C:/Users/Admin/tanitad-wt"
SLICE="C:/Users/Admin/refav1_eval_slice"
CSP="$REPO/TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-refav1-cost-surface/raw"
LAB="$WT/_s2build/release/v72/s2_labels_v7.2_eval.jsonl.gz"
export PYTHONPATH="$WT/stack;$WT/colab;$WT/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
cd "$WT"

"$PY" "$SCR/cost_repair_probe.py" \
  --ckpt "$SLICE/ckpt_ep2/ckpt.pt" --config "$SLICE/ckpt_ep2/config.json" \
  --cache "$SLICE/fp8" --episodes "$SLICE/eps" --labels "$LAB" --nav "$LAB" \
  --name ep2 --arm-tool-dir "$WT/taniteval/tools" \
  --cost-surface-probe "$WT/taniteval/tools/cost_surface_probe.py" \
  --banked-dump "$SLICE/t1_dump_ep2" \
  --banked-cost-surface "$CSP/cost_surface_ep2.json" \
  --episodes-n 20 --window-stride 10 --gate-n 5 --device cuda \
  --out "$SCR/cost_repair_ep2.json" --raw-out "$SCR/cost_repair_ep2_rows.json" \
  > "$SCR/run_ep2.log" 2>&1
echo "EP2 EXIT $?"

"$PY" "$SCR/cost_repair_probe.py" \
  --ckpt "$SLICE/ckpt/ckpt.pt" \
  --cache "$SLICE/fp8" --episodes "$SLICE/eps" --labels "$LAB" --nav "$LAB" \
  --name incumbent --arm-tool-dir "$WT/taniteval/tools" \
  --cost-surface-probe "$WT/taniteval/tools/cost_surface_probe.py" \
  --banked-dump "$SLICE/t1_dump" \
  --banked-cost-surface "$CSP/cost_surface_incumbent.json" \
  --episodes-n 20 --window-stride 10 --gate-n 5 --device cuda \
  --out "$SCR/cost_repair_incumbent.json" \
  --raw-out "$SCR/cost_repair_incumbent_rows.json" \
  > "$SCR/run_incumbent.log" 2>&1
echo "INCUMBENT EXIT $?"
echo ZZCOSTREPAIRDONEZZ
