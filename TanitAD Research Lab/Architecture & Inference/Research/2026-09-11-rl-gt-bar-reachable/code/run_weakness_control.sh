#!/usr/bin/env bash
# ⭐ THE CONTROL THAT DECIDES WHETHER THE >=GT BAR IS SMART OR MERELY WEAK.
#
# The bar admits ~14.7 % of sampled anchors. So the obvious sceptical reading of
# "bar-ON drifts less" is trivial: fewer positive gradients means a smaller total
# update, and a model that changes less drifts less. That would make the finding an
# artifact of update MAGNITUDE, not of update QUALITY.
#
# ⛔ The one piece of evidence already against it is that bar-ON also produces MORE
# fan-reward improvement (mean +0.1012 vs +0.0522), which a merely-weaker update
# should not -- but those two distributions OVERLAP, so it is suggestive and not a
# discriminator.
#
# ⭐ THE DISCRIMINATOR, using only flags that already exist: run bar-OFF at 300 steps,
# which is 15 % of 2,000 and therefore matches the bar's admitted fraction of positive
# gradients. If "less update" is the whole story, this arm should drift like a bar-ON
# arm (~36-44 %). If it still drifts like a full bar-OFF arm (~76-221 %), then the
# bar is selecting BETTER gradients rather than merely fewer.
#
# ⛔ This is not a perfect matching -- 300 full-batch steps is not identical to 2,000
# steps at 15 % admission, because the optimiser state and schedule differ. It is a
# cheap ONE-DIRECTIONAL test: a bar-OFF-short arm that still drifts badly RULES OUT
# the weakness explanation; one that drifts little does NOT confirm it.
set -u
CK="C:/Users/Admin/tanitad-data/models/refc-base-30k/ckpt.pt"
OUT="C:/Users/Admin/tanitad-data/rl-pilot"
REPO="C:/Users/Admin/tanitad-rlrun"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=6
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/stack"
[ "$(stat -c %s "$CK" 2>/dev/null)" = "1250838325" ] || { echo "ZZGATE-CKPTZZ"; exit 2; }
B=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -ci "pyth""on")
[ "${B:-1}" -gt 0 ] && { echo "ZZGATE-GPU-BUSYZZ"; exit 4; }
COMMON=( --ckpt "$CK"
  --train-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74"
  --train-agents "$OUT/pilot_train_agents.jsonl"
  --val-epdir "C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836"
  --val-agents "$OUT/pilot_val_agents.jsonl"
  --batch 2 --reward default )
for s in 0 1 2; do
  tag="offshort-s$s"
  echo "[weak] $tag $(date -u +%FT%TZ)"
  timeout 1800 "$PY" stack/scripts/rl_pilot_refc21.py "${COMMON[@]}" --steps 300 --seed "$s" --out "$OUT/$tag" > "$OUT/$tag.log" 2>&1
  rc=$?
  if [ -s "$OUT/$tag/pilot_summary.json" ]; then echo "ZZ${tag}-OK-${rc}ZZ"; else echo "ZZ${tag}-NOSUMMARY-${rc}ZZ"; fi
done
echo "ZZWEAK-DONEZZ"
