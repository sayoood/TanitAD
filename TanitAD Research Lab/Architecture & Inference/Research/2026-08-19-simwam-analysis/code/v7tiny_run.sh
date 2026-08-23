#!/usr/bin/env bash
# v7-tiny: v6's REAL trainer, tiny, ONE VARIABLE = the residual init scale.
# Not a reimplementation -- identical objective (o1,o2,o3,o4,o5,o6) by construction.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
SPD="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
CACHE="$SPD/sp2/cache/slotprobe-lead130-w120-256x640cyl"
export PYTHONPATH="$REPO/stack"; export PYTHONIOENCODING=utf-8
cd "$REPO/stack" || exit 1

common="--stage S-W --v2-cache $CACHE --no-require-parity
  --frame-h 256 --frame-w 640 --patch 16
  --enc-dim 128 --enc-depth 3 --enc-heads 4
  --pred-dim 256 --pred-depth 3 --pred-heads 4
  --readout-grid 4 --readout-dim 128
  --window 6 --horizons 1 2 4 --o1-k 4 --o5-k 6
  --d-tac 128 --d-str 64 --steps 2000 --batch 4 --v2-lru 6
  --log-every 100 --seed 0"

echo "[v7t] ARM 1/2: fixed (RESIDUAL_HEAD_INIT_SCALE=1e-3, the default)"
"$PY" scripts/train_v6_staged.py --out "$SPD/v7tiny_fixed" $common \
  > "$SPD/v7tiny_fixed.log" 2>&1
echo "[v7t] arm 1 done"

echo "[v7t] ARM 2/2: regress (DEFECT REINTRODUCED)"
TANITAD_RESIDUAL_INIT_SCALE=1.0 "$PY" scripts/train_v6_staged.py \
  --out "$SPD/v7tiny_regress" $common \
  > "$SPD/v7tiny_regress.log" 2>&1
echo "[v7t] arm 2 done"
echo "[v7t] DONE"
