#!/usr/bin/env bash
# n=2 per cell could not separate the >=GT bar from this rig's own run-to-run noise:
# the bar-OFF cell alone spans 145 points of dR3 across two seeds. This adds seeds
# 2,3,4 to BOTH settings, taking each cell to n=5.
# ⭐ The hypothesis it tests is now a VARIANCE one, not a mean one: bar-ON read
# +42.26/+38.61 (spread 3.66) while bar-OFF read +221.18/+76.02 (spread 145.15).
# The bar may be STABILISING the arm rather than shifting it. n=2 cannot tell.
set -u
CK="C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"
OUT="C:/Users/Admin/tanitad-data/rl-pilot"
REPO="C:/Users/Admin/tanitad-rlrun"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/stack"
[ "$(stat -c %s "$CK" 2>/dev/null)" = "1250838325" ] || { echo "ZZGATE-CKPTZZ"; exit 2; }
grep -q -- "--gt-bar" "$REPO/stack/scripts/rl_pilot_refc21.py" || { echo "ZZGATE-NO-FLAGZZ"; exit 6; }
B=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci "pyth""on")
[ "${B:-1}" -gt 0 ] && { echo "ZZGATE-GPU-BUSYZZ"; exit 4; }
COMMON=( --ckpt "$CK"
  --train-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74"
  --train-agents "$OUT/pilot_train_agents.jsonl"
  --val-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836"
  --val-agents "$OUT/pilot_val_agents.jsonl"
  --steps 2000 --batch 2 --reward default )
for s in 2 3 4; do
  for mode in on off; do
    tag="$([ "$mode" = on ] && echo gtbar || echo off)-s$s"
    extra=(); [ "$mode" = on ] && extra=(--gt-bar)
    echo "[seeds] $tag $(date -u +%FT%TZ)"
    timeout 3600 "$PY" stack/scripts/rl_pilot_refc21.py "${COMMON[@]}" --out "$OUT/$tag" --seed "$s" "${extra[@]}" > "$OUT/$tag.log" 2>&1
    rc=$?
    if [ -s "$OUT/$tag/pilot_summary.json" ]; then echo "ZZ${tag}-OK-${rc}ZZ"; else echo "ZZ${tag}-NOSUMMARY-${rc}ZZ"; fi
  done
done
echo "ZZSEEDS-DONEZZ"
