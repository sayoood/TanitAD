#!/usr/bin/env bash
# The paired decision-rule A/B on the freed dev-box 4060.
#
# ⭐ WHY THESE SETTINGS
#  --cost-metric ccos + zeroed weights : the ONLY configuration in which the
#      planner can execute a turn at all (D-REFAV1-DRIVE-GATE2: under the
#      default `cos` a correctly decoded TURN is refused on 38/38 real windows,
#      so an A/B there would compare two zeros).
#  --window-stride 40                  : the banked panel's grid, so these arms
#      are comparable to dump_ccos_comp window-for-window.
#  --dump-dir                          : dumps persist, so a killed run is
#      recovered with --analyze-only and the GPU time is never lost.
#  arm A carries NO flag               : the paired CONTROL, and (verified
#      bit-identical to a zero bias on the real ckpt) the legacy path.
set -u
M="C:/Users/Admin/tanitad-wt"
OUT="C:/Users/Admin/refav1_drive/ab"
PY="C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
export PYTHONPATH="$M/stack;$M/taniteval"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
mkdir -p "$OUT"

CKPT=C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt
CFG=C:/Users/Admin/refav1_eval_slice/ckpt_ep3/config.json
CACHE=C:/Users/Admin/refav1_eval_full/fp8
EPS=C:/Users/Admin/refav1_eval_full/eps
NEP=${NEP:-30}

# ⚠️ --labels/--nav ARE NOT OPTIONAL FOR THIS EXPERIMENT. Without them the arm
# runs `labels=OFF nav=OFF` and the TACTICAL family is unavailable — and the
# tactical family is precisely what a decision-rule change is supposed to move.
# A first launch omitted them and was killed 3 min in rather than spending an
# hour of GPU on a panel that could not answer the question.
LBL=C:/Users/Admin/navcomp/data/s2_labels_v7.2_eval.jsonl.gz

common=(--ckpt "$CKPT" --config "$CFG" --cache "$CACHE" --episodes "$EPS"
        --labels "$LBL" --nav "$LBL"
        --device cuda --episodes-n "$NEP" --window-stride 40 --no-navshuf
        --no-lead-block --cost-metric ccos
        --cost-weights 0.0,0.0,64.29715042415070)

run () {  # name, extra-args...
  local name="$1"; shift
  echo "=== ARM $name  $(date -u +%FT%TZ) ==="
  "$PY" "$M/taniteval/tools/refav1_arm.py" "${common[@]}" "$@" \
      --dump-dir "$OUT/dump_$name" --out "$OUT/rec_$name.json" \
      --arm "refav1-21109-ab-$name" \
      >> "$OUT/$name.log" 2>&1
  echo "    exit=$? $(date -u +%FT%TZ)"
}

run argmax
run prior050 --lat-logit-bias "0.197442,0,0,0,1.191905,0.976514,1.669661,1.434659"
echo "AB_DONE $(date -u +%FT%TZ)"
