#!/usr/bin/env bash
# H-RANK-22: O1 confined to the PREDICTOR (encoder detached for the O1 term only).
#
# This arm is BIT-IDENTICAL to `lewm_o1` except for the single flag
# --o1-detach-encoder, so any difference in participation or divergence is
# attributable to the gradient path and to nothing else.
#
# lewm_o1 (the arm this is matched to) MEASURED: participation 2.94, divergence 516.6
# lewm    (w_o1 = 0)                   MEASURED: participation 4.43, divergence 0.000
# The hypothesis predicts this arm lands near participation ~4.4 AND divergence >> 0.
set -u
SPD="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad"
M=/c/Users/Admin/tanitad-mirror/stack
export PYTHONPATH="$M"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=3
export HF_HUB_OFFLINE=1
cd "$M" || exit 1
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
nm=lewm_o1_detach

# ⛔ verify the fix is actually present in the code this launch will import,
# rather than trusting that the file was copied (a chain that runs stale code
# reproduces the bug it was written to remove).
if ! grep -q "o1_detach_encoder" "$M/scripts/train_v6_staged.py"; then
  echo "ZZDETACH-STALE-CODE-REFUSEDZZ"
  exit 1
fi
echo "ZZDETACH-CODE-VERIFIEDZZ"

echo "ZZ$nm-STARTZZ"
"$PY" scripts/train_v6_staged.py --out "$SPD/v7tiny_$nm" --stage S-W --v2-cache "$SPD/sp2/cache/slotprobe-lead130-w120-256x640cyl" --no-require-parity --frame-h 256 --frame-w 640 --patch 16 --enc-dim 128 --enc-depth 3 --enc-heads 4 --pred-dim 256 --pred-depth 3 --pred-heads 4 --readout-grid 4 --readout-dim 128 --window 6 --horizons 1 2 4 --o1-k 4 --o5-k 1 --d-tac 128 --d-str 64 --steps 2000 --batch 4 --v2-lru 6 --log-every 200 --seed 0 --spectrum-accum 43 --sigreg-slices 512 --o5-form l1 --sigreg-subspaces 32 --w-o5 1.0 --w-o6 0.1 --w-o2 0 --w-o3 0 --w-o1-ctrl 1.0 --w-o1-fact 1.0 --w-o1-scene 0.3 --o1-detach-encoder > "$SPD/v7tiny_$nm.log" 2>&1

if [ -s "$SPD/v7tiny_$nm/ckpt.pt" ]; then
  # confirm from the RUN'S OWN LOG that the flag was actually in force
  if grep -q '"o1_detach_encoder": true' "$SPD/v7tiny_$nm.log"; then
    echo "ZZ$nm-OK-FLAG-CONFIRMEDZZ"
  else
    echo "ZZ$nm-OK-BUT-FLAG-NOT-IN-LOGZZ"
  fi
else
  echo "ZZ$nm-MISSINGZZ"
  tail -5 "$SPD/v7tiny_$nm.log"
fi
