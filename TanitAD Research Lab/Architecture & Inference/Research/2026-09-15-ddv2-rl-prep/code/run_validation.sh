#!/bin/sh
# PREREG_DDV2_RL_VALIDATION.md, executed in order. Dev-box RTX 4060 only.
# Each stage is followed by a check on its ARTIFACTS; a failed check stops the chain.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WT=/c/Users/Admin/tanitad-wt-rl-ddv2
OUT=/c/Users/Admin/tanitad-caches/ddv2rl-20260915
PKG="$WT/TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-ddv2-rl-prep"
HELD=$OUT/heldout41_eps
BASE_CK=/c/Users/Admin/refcv5v2_final/ckpt.pt
BASE_CFG=/c/Users/Admin/refcv5v2_final/config.json
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export PYTHONPATH="C:\\Users\\Admin\\tanitad-wt-rl-ddv2\\stack;C:\\Users\\Admin\\tanitad-wt-rl-ddv2\\taniteval;C:\\Users\\Admin\\tanitad-wt-rl-ddv2"
cd "$WT" || exit 2

gpu_free() {
  foreign=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -i -E "python|torch" || true)
  if [ -n "$foreign" ]; then
    echo "STOP: a CUDA compute process is running: $foreign"
    exit 3
  fi
}
stamp() { echo "== $1  $(date -u +%Y-%m-%dT%H:%M:%SZ)"; }

for SPEC in "rl 0" "norl 0" "rl 1"; do
  set -- $SPEC
  A=$1; S=$2; D="$OUT/arm-$A-s$S"
  gpu_free
  stamp "train $A s$S"
  "$PY" stack/scripts/ddv2_rl_refcv5.py train --arm "$A" --seed "$S" --steps 600 --batch 4 --out-dir "$D" > "$D.log" 2>&1
  "$PY" "$PKG/code/check_arm.py" "$D" --steps 600 --arm "$A" || { echo "ARTIFACT CHECK FAILED: $A s$S"; exit 4; }
done

for CK in base base_repeat arm-rl-s0 arm-norl-s0 arm-rl-s1; do
  case "$CK" in
    base|base_repeat) C="$BASE_CK"; G="$BASE_CFG" ;;
    *) C="$OUT/$CK/ckpt.pt"; G="$OUT/$CK/config.json" ;;
  esac
  gpu_free
  stamp "heldout T0 $CK"
  "$PY" stack/scripts/ddv2_rl_refcv5.py heldout --ckpt "$C" --config "$G" --stride 10 --out "$OUT/heldout_$CK.json" > "$OUT/heldout_$CK.log" 2>&1
  "$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); n=r['n_windows']; print('heldout', sys.argv[1], 'windows', n); sys.exit(0 if n > 0 else 1)" "$OUT/heldout_$CK.json" || { echo "HELDOUT CHECK FAILED: $CK"; exit 5; }
done

for CK in base arm-rl-s0 arm-norl-s0 arm-rl-s1; do
  case "$CK" in
    base) C="$BASE_CK"; G="$BASE_CFG" ;;
    *) C="$OUT/$CK/ckpt.pt"; G="$OUT/$CK/config.json" ;;
  esac
  gpu_free
  stamp "T1 roll $CK"
  rm -rf "$OUT/t1_${CK}_dump"
  "$PY" taniteval/tools/refcv3_arm.py --ckpt "$C" --config "$G" --episodes "$HELD" \
      --labels /c/Users/Admin/refcv5cmp/data/s2_labels_v7.2_eval.jsonl.gz --nav-source v72 \
      --grid 2s --action-units steer --device cuda --window-stride 5 \
      --lead-block /c/Users/Admin/refcv5cmp/data/b1_eval_lead_block.npz \
      --n-boot 2000 --seed 0 --infer-seed 0 --dump-dir "$OUT/t1_${CK}_dump" --out "$OUT/t1_$CK.json" \
      --tiers "os=T1,os_navshuf=T1,os_navzero=T1" > "$OUT/t1_$CK.log" 2>&1
  "$PY" -c "import json,sys,os; p=sys.argv[1]; ok=os.path.exists(p) and os.path.getsize(p)>10000; print('t1', p, 'ok', ok); sys.exit(0 if ok else 1)" "$OUT/t1_$CK.json" || { echo "T1 CHECK FAILED: $CK"; exit 6; }
done
stamp "ALL DONE"
