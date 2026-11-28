#!/usr/bin/env bash
# L4 lane - THE FRICTION-CIRCLE (Kamm) CURVATURE CAP, one variable against the
# banked `ccos_argmax`: |kappa| <= mu*g/v^2 inside `_clip`, on the candidate's
# OWN speed profile, instead of the constant kappa_max = 0.2. Same metric,
# same weights, same seed, same panel, same vocabulary.
# MEASURED reason (raw/feas_audit.txt, v0 >= 2 m/s, n = 27; GROUND-TRUTH control
# reads envelope 0.0000 / kamm_over 0.0000, so the block is admissible):
# refav1's plans are ENVELOPE-feasible by construction (envelope_rate 0.0000,
# max|kappa| exactly 0.2000) yet 29.6 % leave the mu = 0.7 friction circle,
# 42.1 % at v0 >= 5 m/s, with peak_g up to 3.262 g against a ground truth of
# 0.373. At 20 m/s the cap is 0.0172 1/m -- 11.6x tighter than the clip.
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
  dB=$(grep -c "ZZQUEUEB2-DONE" "$SP/queueB2.log" 2>/dev/null)
  dB=$(printf '%s' "${dB:-0}" | tr -dc '0-9')
  [ $((i % 15)) -eq 0 ] && echo "ZZWAITE-${i}-n${n}-b${dB}ZZ $(date -u +%FT%TZ)"
  if [ "${dB:-0}" -ge 1 ] 2>/dev/null && [ "${n:-9}" -le 2 ] 2>/dev/null; then
    echo "ZZWAITE-CLEAR-${i}ZZ $(date -u +%FT%TZ)"; break
  fi
  sleep 20
done

tag=kamm07
echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
"$PY" "$M/taniteval/tools/refav1_arm.py" \
  --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
  --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
  --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
  --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
  --no-lead-block --cost-metric ccos \
  --cost-weights 0.0,0.0,64.29715042415070 \
  --kamm-mu 0.7 \
  --plan-seed 0 \
  --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
  --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
echo "ZZQUEUEE-DONE-$(date -u +%FT%TZ)ZZ"
