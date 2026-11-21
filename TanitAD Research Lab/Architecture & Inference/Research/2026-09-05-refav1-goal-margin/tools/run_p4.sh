#!/usr/bin/env bash
# P4 — does refav1 EXECUTE a turn where the road turns?
#
# 3 arms on a turn-DENSE 8-episode panel (40 windows, 17 with |gt_kappa|>0.04):
#   A  cos   + argmax  (the SHIPPED DEFAULT)          -> predicted: curvature 0
#   B  ccos  + argmax                                  -> predicted: non-zero
#   C  ccos  + argmax @ plan-seed 1  (SEED REPLICATE, H-ESTIM-SEED-1)
# Everything else identical, so A vs B isolates the cost metric and B vs C the
# planner's own run-to-run noise.
#
# ⛔ WAITS for the sibling refav1 arms to finish before touching the GPU — the
# dev box is shared and an 8 GB card is already at 6.3 GB. The wait polls for
# the ABSENCE of the sibling process, not for a success marker (a wait keyed on
# success polls forever when the thing it waits for crashes).
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_margin/p4out"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

# ---- wait for the GPU (max 60 min) --------------------------------------- #
for i in $(seq 1 360); do
  live=$(powershell.exe -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | Where-Object { \$_.CommandLine -like '*refav1_arm.py*' }).Count" 2>/dev/null | tr -d '\r ')
  free=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d '\r ')
  echo "[wait $i] sibling refav1_arm procs=$live gpu_used_MiB=$free  $(date -u +%FT%TZ)"
  if [ "${live:-1}" = "0" ]; then echo "[wait] GPU clear of sibling arms"; break; fi
  sleep 10
done

LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz
CACHE=C:/Users/Admin/refav1_margin/p4/fp8
EPS=C:/Users/Admin/refav1_margin/p4/eps

common=(--ckpt C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt
        --config C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json
        --cache "$CACHE" --episodes "$EPS" --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n 0 --window-stride 16 --no-navshuf
        --no-lead-block
        --cost-weights 0.0,0.0,64.29715042415070)

run () { local name="$1"; shift
  echo "=== ARM $name  $(date -u +%FT%TZ) ==="
  "$PY" "$M/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
      --dump-dir "$OUT/dump_$name" --out "$OUT/rec_$name.json" \
      --arm "refav1-21109-p4-$name" >> "$OUT/$name.log" 2>&1
  echo "    exit=$? $(date -u +%FT%TZ)"; }

run cos_argmax    --cost-metric cos  --plan-seed 0
run ccos_argmax   --cost-metric ccos --plan-seed 0
run ccos_seed1    --cost-metric ccos --plan-seed 1
echo "P4_ALL_ARMS_DONE $(date -u +%FT%TZ)"
