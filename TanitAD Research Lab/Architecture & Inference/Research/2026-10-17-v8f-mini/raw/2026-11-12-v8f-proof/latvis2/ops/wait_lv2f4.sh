#!/usr/bin/env bash
# wait_lv2f4.sh : dev-box waiter (script polling, no model turns; bounded 6 h). Card PREREG_V8F_PROOF_ADD38, note N4.
# Once the pod CPU job has ended: pull the four forecast dumps of the new seeds and their k5 probe outputs with sha256, then read G and RR in two scopes
# (new seeds 2,3 alone; four seeds 0,1,2,3) with the unchanged readers, and the pairs RR minus G. It starts nothing on the pod.
P=/c/Users/Admin/v8pod/pssh.sh; C=/c/Users/Admin/v8pod/pcat.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
H=/f/TanitAD-scratch/v8proof/latvis2; D=$H/dump; R=$H/read_n4
REF=C:/Users/Admin/v8proof_c/ctxroll/LATENTS_CTX_G_s0_R1.npz
mkdir -p "$D" "$R"; cd $H || exit 1
ts() { date -u +%H:%M:%SZ; }
pod() { timeout "$1" bash "$P" "$2" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
k=0
while :; do
  k=$((k+1))
  S=$(pod 60 "cd /workspace/v8proof/runs/lv2_z; grep -a 'rc \|ZZ LV2F4\|SHA \|tool sha' lv2f4_all.log | cut -c1-330")
  echo "[$(ts)] $(echo "$S" | tail -n 2 | tr '\n' ';' | cut -c1-420)"
  echo "$S" | grep -q "ZZ LV2F4 POD DONE" && break
  [ "$k" -gt 180 ] && { echo "ZZ LV2F4 TIME LIMIT (6 h); reading what exists"; break; }
  sleep 120
done
echo "$S" > "$H/ops/pod_lv2_forecast4.out"
echo "$S" | grep "^   SHA " | while read -r _ a s1 s2 s3 _; do
  for pr in "DUMP_LV2F_$a.npz:$s1:$D" "DUMP_LV2F_$a.npz.json:$s2:$D" "k5_probe_$a.out:$s3:$H/ops"; do f=$(echo "$pr" | cut -d: -f1); s=$(echo "$pr" | cut -d: -f2); to=$(echo "$pr" | cut -d: -f3-)
    [ -f "$to/$f" ] && [ "$(sha256sum "$to/$f" | cut -c1-64)" = "$s" ] && continue
    timeout 600 bash $C /workspace/v8proof/runs/lv2_z/$f "$to/$f" < /dev/null; [ "$(sha256sum "$to/$f" | cut -c1-64)" = "$s" ] && echo "[$(ts)] pulled $f" || echo "[$(ts)] ZZ SHA MISMATCH $f"
  done
done
for a in G_s2 G_s3 RR_s2 RR_s3; do timeout 120 bash $C /workspace/v8proof/runs/lv2_z/lv2f_$a.log "$H/ops/pod_lv2f_$a.log" < /dev/null; done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONIOENCODING=utf-8
dl() { local a=$1 s o=""; shift; for s in "$@"; do o="$o $D/DUMP_LV2F_${a}_s$s.npz"; done; echo $o; }
kl() { local a=$1 s o=""; shift; for s in "$@"; do o="$o ops/k5_probe_${a}_s$s.out"; done; echo $o; }
for A in G RR; do
  ok=1; for s in 0 1 2 3; do [ -f "$D/DUMP_LV2F_${A}_s$s.npz" ] && [ -s "ops/k5_probe_${A}_s$s.out" ] || ok=0; done
  [ $ok = 1 ] || { echo "[$(ts)] $A: dumps or probe outputs of the four seeds not all present, no read"; continue; }
  "$PY" lv2_forecast_read.py read --dump $(dl $A 2 3) --arm $A --seeds 2,3 --out-dir "$R/${A}_new" --ref-windows "$REF" --k5-parts $(kl $A 2 3) > "$R/${A}_new.log" 2>&1
  echo "[$(ts)] forecast read $A new seeds rc $? : $(tail -n 1 "$R/${A}_new.log" | cut -c1-320)"
  "$PY" lv2_forecast_read.py read --dump $(dl $A 0 1 2 3) --arm $A --seeds 0,1,2,3 --out-dir "$R/${A}_all" --ref-windows "$REF" --k5-parts $(kl $A 0 1 2 3) > "$R/${A}_all.log" 2>&1
  echo "[$(ts)] forecast read $A four seeds rc $? : $(tail -n 1 "$R/${A}_all.log" | cut -c1-320)"
  "$PY" lv2_read.py read --dump $(dl $A 2 3) --arm $A --seeds 2,3 --out-dir "$R/first_${A}_new" --ref-windows "$REF" --k5-parts $(kl $A 2 3) > "$R/first_${A}_new.log" 2>&1
  echo "[$(ts)] first read $A new seeds rc $? : $(tail -n 1 "$R/first_${A}_new.log" | cut -c1-200)"
  "$PY" lv2_read.py read --dump $(dl $A 0 1 2 3) --arm $A --seeds 0,1,2,3 --out-dir "$R/first_${A}_all" --ref-windows "$REF" --k5-parts $(kl $A 0 1 2 3) > "$R/first_${A}_all.log" 2>&1
  echo "[$(ts)] first read $A four seeds rc $? : $(tail -n 1 "$R/first_${A}_all.log" | cut -c1-200)"
done
for sc in new all; do
  if [ -d "$R/G_$sc" ] && [ -d "$R/RR_$sc" ]; then
    "$PY" lv2_forecast_read.py pair --a-dir "$R/RR_$sc" --b-dir "$R/G_$sc" --out-dir "$R/pair_$sc" > "$R/pair_$sc.log" 2>&1
    echo "[$(ts)] pair RR minus G ($sc) rc $? : $(tail -n 1 "$R/pair_$sc.log" | cut -c1-320)"
  fi
done
echo "ZZ LV2F4 READS DONE $(ts)"
