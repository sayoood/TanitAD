#!/usr/bin/env bash
# THE OWED T1 ROLLS — the pre-registration required them and Stage A did not run them.
#
# ⛔ WHY THIS MATTERS MORE THAN ANY T0 NUMBER IN THE PACKAGE. `PREREG_DDV2_RL_VALIDATION.md`
# §13.3's harm guard **H-DDV2RL-2** is a T1 statistic, and 2026-09-15 read **FAIL-HARM**:
# both RL seeds worse than the cold start on T1 ADE (+0.083 [0.056, 0.115] and
# +0.081 [0.051, 0.116]). Fixing that harm is what L1 exists for. Without T1, the L1
# package cannot say whether it did — and §13.3 says the guard is then reported
# **unevaluable, never passed**.
#
# ⭐ THE DROP ORDER IS PRE-REGISTERED, SO THIS IS NOT A JUDGEMENT CALL. §14: if the
# cumulative passes 2.85 h, drop (1) Stage B, (2) L1-NORL-s0's T1 roll, (3) L1-RL-s1's.
#   * (1) Stage B is already NOT RUN.
#   * (2) L1-NORL-s0's T1 roll is DROPPED here — it feeds only §13.4 "NORL − BASE under
#         the new form", which is **reported with a direction but NO criterion**.
#   * (3) L1-RL-s1's roll is KEPT, because H-DDV2RL-2 needs BOTH RL seeds against BASE.
# ⇒ dropping exactly item (2) makes the harm guard EVALUABLE inside the budget:
#   3 rolls x ~9.6 min = ~29 min => cumulative ~2.75 h < 2.85 h.
#
# Invocation copied from the 2026-09-15 package's own `run_validation.sh` so the two
# packages' T1 numbers are comparable; only the checkpoints differ.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WT=/c/Users/Admin/tanitad-wt-rl-ddv2
OLD=/c/Users/Admin/tanitad-caches/ddv2rl-20260915
OUT=/c/Users/Admin/tanitad-caches/ddv2rl-l1-20260917
HELD=$OLD/heldout41_eps
BASE_CK=/c/Users/Admin/refcv5v2_final/ckpt.pt
BASE_CFG=/c/Users/Admin/refcv5v2_final/config.json
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6          # ⛔ CLAUDE.md: torch spawns ~113 threads/process
export PYTHONPATH="C:\\Users\\Admin\\tanitad-wt-rl-ddv2\\stack;C:\\Users\\Admin\\tanitad-wt-rl-ddv2\\taniteval;C:\\Users\\Admin\\tanitad-wt-rl-ddv2"
cd "$WT" || exit 2

gpu_free() {   # ⛔ never add GPU load to a box that is training
  foreign=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
            | grep -i -E "python|torch" || true)
  if [ -n "$foreign" ]; then echo "ZZSTOP-GPU-BUSY $foreignZZ"; exit 3; fi
}

for CK in base l1-rl-s0 l1-rl-s1; do
  case "$CK" in
    base) C="$BASE_CK"; G="$BASE_CFG" ;;
    *)    C="$OUT/$CK/ckpt.pt"; G="$OUT/$CK/config.json" ;;
  esac
  [ -f "$C" ] || { echo "ZZABORT-NO-CKPT $CKZZ"; exit 4; }
  gpu_free
  echo "== T1 roll $CK  $(date -u +%H:%M:%SZ)"
  rm -rf "$OUT/t1_${CK}_dump"
  "$PY" taniteval/tools/refcv3_arm.py --ckpt "$C" --config "$G" --episodes "$HELD" \
      --labels /c/Users/Admin/refcv5cmp/data/s2_labels_v7.2_eval.jsonl.gz --nav-source v72 \
      --grid 2s --action-units steer --device cuda --window-stride 5 \
      --lead-block /c/Users/Admin/refcv5cmp/data/b1_eval_lead_block.npz \
      --n-boot 2000 --seed 0 --infer-seed 0 --dump-dir "$OUT/t1_${CK}_dump" \
      --out "$OUT/t1_$CK.json" \
      --tiers "os=T1,os_navshuf=T1,os_navzero=T1" > "$OUT/t1_$CK.log" 2>&1
  RC=$?
  # ⛔ THE ARTIFACT IS THE EVIDENCE, NEVER THE EXIT CODE.
  SZ=$(stat -c %s "$OUT/t1_$CK.json" 2>/dev/null || echo 0)
  echo "ZZT1 $CK rc=$RC bytes=$SZZZ"
  [ "${SZ:-0}" -gt 10000 ] || { echo "ZZT1-INCOMPLETE $CKZZ"; tail -6 "$OUT/t1_$CK.log"; exit 5; }
done
echo "ZZT1-ALL-DONEZZ"
