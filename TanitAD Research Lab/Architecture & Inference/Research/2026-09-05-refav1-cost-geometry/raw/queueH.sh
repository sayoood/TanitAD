#!/usr/bin/env bash
# THE ARM THE L3 NULL ASKS FOR: the seed ladder TOGETHER WITH a cost that has a
# reason to prefer its rungs.
# MEASURED (RESULT.md 7.7): with W_KAPPA = 0 the ladder's 10 extra candidates are
# NEVER chosen - not one of 40 windows realises a rung magnitude - because the cost
# is indifferent between them and the canonical seed. And wk15 produced
# intermediate curvatures (0.0115-0.0530) with NO ladder at all, purely because the
# penalty rewarded them. So the informative combination is ladder + W_KAPPA, not
# ladder alone and not ladder + a CONSTRAINT (a cap cannot make a rung attractive).
# Two variables against ccos_argmax, one against wk15: the candidate set.
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
  [ $((i % 15)) -eq 0 ] && echo "ZZWAITH-${i}-n${n}ZZ $(date -u +%FT%TZ)"
  [ "${n:-9}" -le 2 ] 2>/dev/null && { echo "ZZWAITH-CLEAR-${i}ZZ $(date -u +%FT%TZ)"; break; }
  sleep 20
done

tag=wk15_ladder
rm -rf "$OUT/dump_$tag"
echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
  --no-lead-block --cost-metric ccos \
  --cost-weights 0.0,15.11245,64.29715042415070 \
  --seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04 \
  --plan-seed 0 \
  --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
  --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
echo "ZZQUEUEH-DONE-$(date -u +%FT%TZ)ZZ"
