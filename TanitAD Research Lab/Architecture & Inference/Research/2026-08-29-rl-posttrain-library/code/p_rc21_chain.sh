#!/usr/bin/env bash
# P-RC21 autonomous chain — PI: "go ahead automatically if you are ready".
# Every stage banks its own artifacts; a kill at any point keeps what ran.
set -u
CK="C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"
WANT_MD5="8f10d6f934f4199e11ddc7352e074939"
OUT="C:/Users/Admin/tanitad-data/rl-pilot"
REPO="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/stack"

echo "[chain] waiting for ckpt transfer ..."
until [ "$(stat -c %s "$CK" 2>/dev/null)" = "1250838325" ]; do sleep 20; done
GOT=$(md5sum "$CK" | awk '{print $1}')
echo "[chain] ckpt md5 $GOT (want $WANT_MD5)"
if [ "$GOT" != "$WANT_MD5" ]; then
  echo "ZZCHAIN-MD5-REFUSEDZZ"; exit 3
fi
if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -qi python; then
  echo "ZZCHAIN-GPU-BUSY-REFUSEDZZ"; exit 4
fi

echo "[chain] P1 launch (grpo, 2000 steps, cap 2h) $(date)"
timeout 7200 "$PY" stack/scripts/rl_pilot_refc21.py \
  --ckpt "$CK" \
  --train-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74" \
  --train-agents "$OUT/pilot_train_agents.jsonl" \
  --val-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836" \
  --val-agents "$OUT/pilot_val_agents.jsonl" \
  --out "$OUT/p1-grpo" --steps 2000 --batch 2 --reward default --seed 0 \
  > "$OUT/p1-grpo.log" 2>&1
P1=$?
if [ -s "$OUT/p1-grpo/pilot_summary.json" ]; then echo "ZZP1-OK-${P1}ZZ"; else echo "ZZP1-NOSUMMARY-${P1}ZZ"; fi

echo "[chain] P2-reg launch (progress-only, 300 steps, cap 30min) $(date)"
timeout 1800 "$PY" stack/scripts/rl_pilot_refc21.py \
  --ckpt "$CK" \
  --train-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74" \
  --train-agents "$OUT/pilot_train_agents.jsonl" \
  --val-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836" \
  --val-agents "$OUT/pilot_val_agents.jsonl" \
  --out "$OUT/p2-reg" --steps 300 --batch 2 --reward hackable --seed 0 \
  > "$OUT/p2-reg.log" 2>&1
P2=$?
if [ -s "$OUT/p2-reg/pilot_summary.json" ]; then echo "ZZP2-OK-${P2}ZZ"; else echo "ZZP2-NOSUMMARY-${P2}ZZ"; fi
echo "ZZCHAIN-DONEZZ"
