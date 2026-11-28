#!/usr/bin/env bash
# P-RC21 AMENDMENT 1 — the anchor sweep. Pre-registered before any arm ran.
set -u
CK="C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"
O="C:/Users/Admin/tanitad-data/rl-pilot"
R="G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
cd "$R" || exit 1; export PYTHONPATH="$R/stack"
if nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -qi python; then
  echo "ZZGPU-BUSY-REFUSEDZZ"; exit 4
fi
run () {
  nm=$1; w=$2; rew=$3; st=$4
  echo "[sweep] $nm w_anchor=$w reward=$rew steps=$st $(date +%H:%M:%S)"
  timeout 3600 "$PY" stack/scripts/rl_pilot_refc21.py --ckpt "$CK" \
    --train-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74" \
    --train-agents "$O/pilot_train_agents.jsonl" \
    --val-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836" \
    --val-agents "$O/pilot_val_agents.jsonl" \
    --out "$O/rerun-$nm" --steps "$st" --batch 2 --reward "$rew" \
    --w-anchor "$w" --seed 0 > "$O/rerun-$nm.log" 2>&1
  if [ -s "$O/rerun-$nm/pilot_summary.json" ]; then echo "ZZ${nm}-OKZZ"; else echo "ZZ${nm}-NOSUMMARYZZ"; fi
}


run t1-w1-prox   1.0   proximity  2000
run t2-w10-prox  10.0  proximity  2000
run treg         0     hackable   300
echo "ZZRERUN-DONEZZ"
