M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_margin/p4/fp8
EPS=C:/Users/Admin/refav1_margin/p4/eps
# wait for EXPLICIT PIDs to be gone (never pgrep -f; never a success marker)
wait_pids () {
  local pids="$1"
  for i in $(seq 1 900); do
    n=$(powershell.exe -NoProfile -Command \
        "@(Get-Process -Id $pids -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r ')
    n=$(printf '%s' "${n:-1}" | tr -dc '0-9')
    [ "${n:-1}" = "0" ] && { echo "ZZWAIT-CLEAR-${i}ZZ $(date -u +%FT%TZ)"; return 0; }
    [ $((i % 15)) -eq 0 ] && echo "ZZWAIT-${i}-n${n}ZZ $(date -u +%FT%TZ)"
    sleep 20
  done
  echo "ZZWAIT-TIMEOUTZZ"; return 1
}
run_wk () {                      # $1 = tag, $2 = W_KAPPA
  local tag="$1" wk="$2"
  echo "ZZARM-${tag}-START-$(date -u +%FT%TZ)ZZ"
  "$PY" "$M/taniteval/tools/refav1_arm.py" \
    --ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt \
    --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json \
    --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL" \
    --device cuda --episodes-n 0 --window-stride 16 --no-navshuf \
    --no-lead-block --cost-metric ccos \
    --cost-weights "0.0,${wk},64.29715042415070" \
    --plan-seed 0 \
    --dump-dir "$OUT/dump_${tag}" --out "$OUT/rec_${tag}.json" \
    --arm "refav1-21109-p4-${tag}" >> "$OUT/${tag}.log" 2>&1
  echo "ZZARM-${tag}-EXIT-$?-$(date -u +%FT%TZ)ZZ"
}
