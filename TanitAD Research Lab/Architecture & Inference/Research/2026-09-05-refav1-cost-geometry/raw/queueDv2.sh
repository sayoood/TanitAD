#!/usr/bin/env bash
# L3 lane - THE SUSTAINED-CURVATURE SEED LADDER, one variable against the banked
# `ccos_argmax`: the search's iteration-0 candidate set. Same metric (ccos), same
# weights (0, 0, 64.297), same seed, same panel. `canonical_controls` and the
# goal field are UNTOUCHED, so this is NOT a vocabulary change and the arm is
# comparable window-for-window with every other v7.0 arm here.
# Rungs 0.002 / 0.005 / 0.01 / 0.02 / 0.04 = R 500 / 200 / 100 / 50 / 25 m,
# chosen to span the corpus's measured road curvature (R 100-1000 m) plus one
# rung either side; both signs are added by plan(), so +10 candidates.
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
  # count only python.exe: a shell whose own command line names the tool would
  # SELF-MATCH (the pgrep -f trap; it bit once already this session)
  n=$(powershell.exe -NoProfile -Command \
      "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*refav1' + '_arm.py*' }).Count" \
      2>/dev/null | tr -d '\r ')
  n=$(printf '%s' "${n:-9}" | tr -dc '0-9')
  dA=$(grep -c "ZZQUEUEA-DONE" "$SP/queueA.log" 2>/dev/null)
  dA=$(printf '%s' "${dA:-0}" | tr -dc '0-9')
  [ $((i % 15)) -eq 0 ] && echo "ZZWAITD-${i}-n${n}-a${dA}ZZ $(date -u +%FT%TZ)"
  if [ "${dA:-0}" -ge 1 ] 2>/dev/null && [ "${n:-9}" -le 2 ] 2>/dev/null; then
    echo "ZZWAITD-CLEAR-${i}ZZ $(date -u +%FT%TZ)"; break
  fi
  sleep 20
done

tag=l3ladder
echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
  --no-lead-block --cost-metric ccos \
  --cost-weights 0.0,0.0,64.29715042415070 \
  --seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04 \
  --plan-seed 0 \
  --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
  --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
echo "ZZQUEUED-DONE-$(date -u +%FT%TZ)ZZ"
