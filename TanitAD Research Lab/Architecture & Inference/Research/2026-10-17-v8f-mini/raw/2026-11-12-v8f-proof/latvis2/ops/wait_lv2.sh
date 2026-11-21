#!/usr/bin/env bash
# wait_lv2.sh : dev-box waiter (script polling, no model turns; bounded 6 h). Card PREREG_V8F_PROOF_ADD38 with note N1.
# Once the pod CPU job has ended: pull the four token dumps with sha256, read G and RR (two seeds each) and their pair. It starts nothing on the pod.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
H=/f/TanitAD-scratch/v8proof/latvis2; D=$H/dump; R=$H/read
REF=C:/Users/Admin/v8proof_c/ctxroll/LATENTS_CTX_G_s0_R1.npz
mkdir -p "$D" "$R"; cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof/runs/lv2_z; grep -a 'rc \|ZZ LV2\|SHA ' lv2_all.log | cut -c1-260; echo \"LAST \$(ls -t lv2_*_s?.log 2>/dev/null | head -n 1): \$(tail -q -n 1 \$(ls -t lv2_*_s?.log 2>/dev/null | head -n 1) 2>/dev/null | cut -c1-160)\"")
  echo "[$(ts)] $(echo "$S" | tail -n 2 | tr '\n' ';' | cut -c1-360)"
  echo "$S" | grep -q "ZZ LV2 POD DONE" && break
  [ "$k" -gt 180 ] && { echo "ZZ LV2 TIME LIMIT (6 h); reading what exists"; break; }
  sleep 120
done
echo "$S" > "$H/ops/pod_lv2_tokens.out"
echo "$S" | grep "^   SHA " | while read -r _ a s1 s2 _; do
  for pr in "DUMP_LV2_$a.npz:$s1" "DUMP_LV2_$a.npz.json:$s2"; do f=${pr%%:*}; s=${pr##*:}
    [ -f "$D/$f" ] && [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && continue
    timeout 600 bash $C /workspace/v8proof/runs/lv2_z/$f "$D/$f" < /dev/null; [ "$(sha256sum "$D/$f" | cut -c1-64)" = "$s" ] && echo "[$(ts)] pulled $f" || echo "[$(ts)] ZZ SHA MISMATCH $f"
  done
done
for a in G_s0 G_s1 RR_s0 RR_s1; do timeout 120 bash $C /workspace/v8proof/runs/lv2_z/lv2_$a.log "$H/ops/pod_lv2_$a.log" < /dev/null; done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
for A in G RR; do
  [ -f "$D/DUMP_LV2_${A}_s0.npz" ] && [ -f "$D/DUMP_LV2_${A}_s1.npz" ] || { echo "[$(ts)] $A: dumps of both seeds not present, no read"; continue; }
  "$PY" lv2_read.py read --dump "$D/DUMP_LV2_${A}_s0.npz" "$D/DUMP_LV2_${A}_s1.npz" --arm $A --seeds 0,1 --out-dir "$R/$A" --ref-windows "$REF" > "$R/$A.log" 2>&1
  echo "[$(ts)] read $A rc $? : $(tail -n 1 "$R/$A.log" | cut -c1-300)"
done
if [ -d "$R/G" ] && [ -d "$R/RR" ]; then
  "$PY" lv2_read.py pair --a-dir "$R/RR" --b-dir "$R/G" --out-dir "$R/pair" > "$R/pair.log" 2>&1
  echo "[$(ts)] pair RR minus G rc $? : $(tail -n 1 "$R/pair.log" | cut -c1-300)"
fi
echo "ZZ LV2 READS DONE $(ts)"
