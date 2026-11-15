#!/usr/bin/env bash
# wait_leadsup.sh : dev-box waiter for card PREREG_V8F_PROOF_ADD24. Polls the three pod chains every 5 min (bounded, 10 h), pulls the T1-kin dump of each
# finished arm, and when all six are pulled runs the cost read (pairs LS:G, LSH:G, LSD:G on seeds 0 and 1, spread arm G with seeds 0 to 5).
# Exits non-zero on a STOP marker or on the time limit. It starts nothing on the pod.
P=/c/Users/Admin/v8pod/pssh.sh
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
W=F:/TanitAD-scratch/v8proof
PULL=/c/Users/Admin/v8proof_c/pull
export PULL_ROOT=$PULL RUNS_ROOT=/c/Users/Admin/v8proof_c/runs
cd /f/TanitAD-scratch/v8proof || exit 1
ARMS="LS_s0 LS_s1 LSD_s0 LSD_s1 LSH_s0 LSH_s1"
state() { timeout 60 bash "$P" "cd /workspace/v8proof/chains; for c in leadsup_a leadsup_b leadsup_c; do if [ -e \$c/\${c}_STOP.txt ]; then echo \"\$c STOP \$(head -c 200 \$c/\${c}_STOP.txt)\"; elif [ -e \$c/\${c}_DONE.txt ]; then echo \"\$c DONE\"; else echo \"\$c RUN \$(grep -c -- 'ARM DONE' \$c/chain.log)\"; fi; done; grep -h -o -- 'ARM DONE [A-Za-z0-9_]*' leadsup_a/chain.log leadsup_b/chain.log leadsup_c/chain.log | sed 's/ARM DONE //'" 2>/dev/null | sed '/hf_/d' | tr -d '\r'; }
queue_of() { case "$1" in LS_s*) echo leadsup_a;; LSD_s*) echo leadsup_b;; *) echo leadsup_c;; esac; }
pull_one() { [ -f "$PULL/$1/PULLED.txt" ] && return 0; Q=$(queue_of "$1") bash spatial/q3/pull_spatial_t1.sh "$1" 2>&1 | tail -n 2; }
k=0
while :; do
  k=$((k+1))
  S=$(state)
  echo "[$(date -u +%H:%M:%SZ)] $(echo "$S" | tr '\n' ';' | cut -c1-300)"
  if echo "$S" | grep -q -- ' STOP '; then echo "ZZ LEADSUP STOP: $(echo "$S" | grep -- ' STOP ' | head -n 3 | tr '\n' ';')"; exit 2; fi
  for a in $ARMS; do
    if echo "$S" | grep -q -x -- "$a"; then pull_one "$a"; fi
  done
  n=0; for a in $ARMS; do [ -f "$PULL/$a/PULLED.txt" ] && n=$((n+1)); done
  [ "$n" = 6 ] && break
  [ "$k" -gt 120 ] && { echo "ZZ LEADSUP TIME LIMIT (10 h), pulled $n of 6"; exit 3; }
  sleep 300
done
echo "[$(date -u +%H:%M:%SZ)] all six arms pulled; cost read"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8
OUT=$W/spatial/leadsup/read/out_cost; mkdir -p "$OUT"
cd /f/TanitAD-scratch/v8proof/spatial/q3 || exit 1
"$PY" q3_read.py read --t1-root "$W/hiergap/pull,C:/Users/Admin/v8proof_c/pull" --pairs LS:G,LSH:G,LSD:G,LS:LSD,LSH:LSD --seeds 0,1 --spread-arm G --spread-seeds 0,1,2,3,4,5 --same-sign --out-dir "$OUT" > "$OUT.log" 2>&1
echo "read rc $? : $(tail -n 2 "$OUT.log" | tr '\n' ' ' | cut -c1-300)"
grep -E "ALL_PASS|Q3: |ade6_m\||speed_mae|along_mae|cross_mae|heading_mae|lon_err|lat_err|joint_err|ade_tac_m \|" "$OUT/VERDICT_Q3.md" | cut -c1-260
echo "ZZ LEADSUP COST READ DONE $(date -u +%H:%M:%SZ)"
