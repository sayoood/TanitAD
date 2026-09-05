#!/usr/bin/env bash
# THE COMBINED ARM - "make it drive", all three levers at once.
#   --cost-metric ccos            the goal term decides (5.4e6x the `cos` decision)
#   --seed-kappa-ladder ...       the candidate set spans the road's own curvature
#   --kamm-mu 0.7                 |kappa| <= mu*g/v^2 replaces the constant clip
# ⚠️ THIS IS A THREE-VARIABLE ARM AND IS NOT ATTRIBUTABLE ON ITS OWN. Attribution
# comes from the ONE-VARIABLE arms banked beside it (wk15 / wk151 for W_KAPPA,
# ccosh_w000 for the hold branch, l3ladder for the pool, kamm07 for the cap).
# It answers a different question: what is the BEST refav1 can do on this panel
# with the levers this package built. Both outcomes are already committed in
# PREREG_COST_GEOMETRY.md §3 (it beats ha0_ext on the four families while acting,
# or it does not and the report says which lever is blocked on what).
set -u
SP="C:/Users/Admin/AppData/Local/Temp/claude/G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/f407bc82-7969-457c-a947-6be2014fee89/scratchpad"
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_margin/p4/fp8
EPS=C:/Users/Admin/refav1_margin/p4/eps

for i in $(seq 1 900); do
  n=$(powershell.exe -NoProfile -Command \
      "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*refav1' + '_arm.py*' }).Count" \
      2>/dev/null | tr -d '\r ')
  n=$(printf '%s' "${n:-9}" | tr -dc '0-9')
  dD=$(grep -c "ZZQUEUED-DONE" "$SP/queueDv2.log" 2>/dev/null)
  dD=$(printf '%s' "${dD:-0}" | tr -dc '0-9')
  [ $((i % 15)) -eq 0 ] && echo "ZZWAITF-${i}-n${n}-d${dD}ZZ $(date -u +%FT%TZ)"
  if [ "${dD:-0}" -ge 1 ] 2>/dev/null && [ "${n:-9}" -le 2 ] 2>/dev/null; then
    echo "ZZWAITF-CLEAR-${i}ZZ $(date -u +%FT%TZ)"; break
  fi
  sleep 20
done

tag=combined
echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
  --no-lead-block --cost-metric ccos \
  --cost-weights 0.0,0.0,64.29715042415070 \
  --seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04 \
  --kamm-mu 0.7 \
  --plan-seed 0 \
  --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
  --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
echo "ZZQUEUEF-DONE-$(date -u +%FT%TZ)ZZ"
