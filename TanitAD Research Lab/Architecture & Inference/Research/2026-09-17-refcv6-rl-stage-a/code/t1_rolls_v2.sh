#!/usr/bin/env bash
# ⭐ v2. `t1_rolls.sh` (v1) is KEPT BESIDE THIS ONE ON PURPOSE: it is the script that
# actually ran the first attempt, and a banked artifact says what ran. v1 died in two
# ways worth preserving — a `$SZZZ` marker/variable collision that swallowed the real
# error, and a `gpu_free()` guard that on Windows/WDDM refuses on any process holding a
# graphics context (it refused on a 19.6 MB pytest). Both are diagnosed in
# `PREREG_DDV2_RL_VALIDATION.md` §16.2. v2 is what produced the T1 numbers.
# THE OWED T1 ROLLS — the pre-registration required them and Stage A did not run them.
#
# ⛔ WHY THIS MATTERS MORE THAN ANY T0 NUMBER IN THE PACKAGE. `PREREG_DDV2_RL_VALIDATION.md`
# §13.3's harm guard **H-DDV2RL-2** is a T1 statistic, and 2026-09-15 read **FAIL-HARM**:
# both RL seeds worse than the cold start on T1 ADE (+0.083 [0.056, 0.115] and
# +0.081 [0.051, 0.116]). Fixing that harm is what L1 exists for. Without T1 the L1
# package cannot say whether it did — and §13.3 says the guard is then reported
# **unevaluable, never passed**.
#
# ⭐ THE DROP ORDER IS PRE-REGISTERED, SO THIS IS NOT A JUDGEMENT CALL. §14: if the
# cumulative passes 2.85 h, drop (1) Stage B, (2) L1-NORL-s0's T1 roll, (3) L1-RL-s1's.
#   * (1) Stage B is already NOT RUN.
#   * (2) L1-NORL-s0's T1 roll is DROPPED here — it feeds only §13.4 "NORL − BASE under
#         the new form", which is reported with a direction but NO criterion.
#   * (3) L1-RL-s1's roll is KEPT, because H-DDV2RL-2 needs BOTH RL seeds against BASE.
# ⇒ dropping exactly item (2) makes the harm guard EVALUABLE inside the budget.
#
# ⛔ NOTHING IN THE INVOCATION IS REDUCED TO FIT. It is copied verbatim from the
# 2026-09-15 `run_validation.sh` so the two packages' T1 numbers stay comparable; a
# shrunk roll would be a different measurement wearing the same name. Only the
# checkpoints differ.
set -u
PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
WT=/c/Users/Admin/tanitad-wt-rl-ddv2
OLD=/c/Users/Admin/tanitad-caches/ddv2rl-20260915
OUT=/c/Users/Admin/tanitad-caches/ddv2rl-l1-20260917
HELD=$OLD/heldout41_eps
BASE_CK=/c/Users/Admin/refcv5v2_final/ckpt.pt
BASE_CFG=/c/Users/Admin/refcv5v2_final/config.json
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export PYTHONPATH="C:\\Users\\Admin\\tanitad-wt-rl-ddv2\\stack;C:\\Users\\Admin\\tanitad-wt-rl-ddv2\\taniteval;C:\\Users\\Admin\\tanitad-wt-rl-ddv2"
cd "$WT" || exit 2

# ⛔ Never add load to a box that is training — but MEASURE THE RIGHT THING.
# ⚠️ The inherited guard grepped `nvidia-smi --query-compute-apps` for "python|torch". On
# Linux that lists CUDA compute clients; on Windows/WDDM it lists EVERY process with a
# graphics context — explorer.exe, SearchHost.exe, the NVIDIA overlay, Edge. MEASURED
# 2026-09-17: it refused on a pytest process holding 19.6 MB and would have refused
# forever. A guard that cannot tell a trainer from a text editor is a stop button, not a
# safety guard.
# ⛔ And it watched the wrong resource: the first T1 attempt died of HOST RAM (free
# 0.57 GB of 31.8, a sibling python at 4.95 GB), not of GPU — T1 itself peaks ~1.5 GB of
# an 8 GB card. So refuse on FREE MEMORY, tight enough that a real trainer still blocks
# us (arm 3 held ~3.5 GB of GPU, which pushes `used` well past the bar).
box_free() {
  local stat gu hf
  stat=$("$PY" /c/Users/Admin/qland/boxstat.py 2>/dev/null) || {
    echo "ZZSTOP-PROBE-FAILED box state unreadable -- INCONCLUSIVE, never clearZZ"; exit 3; }
  gu=${stat%% *}; hf=${stat##* }
  case "$gu$hf" in *[!0-9]*) echo "ZZSTOP-PROBE-SHAPE '$stat'ZZ"; exit 3 ;; esac
  [ "$gu" -le 2500 ] || { echo "ZZSTOP-GPU-BUSY used=${gu}MiB (desktop baseline ~1550)ZZ"; exit 3; }
  [ "$hf" -ge 8 ]    || { echo "ZZSTOP-HOST-TIGHT free=${hf}GB (the first attempt OOMd at 0.57)ZZ"; exit 3; }
  echo "   box ok: gpu_used=${gu}MiB host_free=${hf}GB"
}

for CK in base l1-rl-s0 l1-rl-s1; do
  case "$CK" in
    base) C="$BASE_CK"; G="$BASE_CFG" ;;
    *)    C="$OUT/$CK/ckpt.pt"; G="$OUT/$CK/config.json" ;;
  esac
  [ -f "$C" ] || { echo "ZZABORT-NO-CKPT ${CK}ZZ"; exit 4; }
  box_free
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
  # ⛔ BRACE THE NAME. `$SZZZ` parsed as the variable SZZZ, not $SZ followed by the ZZ
  # marker, and `set -u` then killed the script BEFORE it could report the real failure.
  # My own marker convention collided with variable interpolation — the same family as a
  # monitor filter that matches its own echoed command text.
  echo "ZZT1 $CK rc=$RC bytes=${SZ}ZZ"
  [ "${SZ:-0}" -gt 10000 ] || { echo "ZZT1-INCOMPLETE ${CK}ZZ"; tail -8 "$OUT/t1_$CK.log"; exit 5; }
done
echo "ZZT1-ALL-DONEZZ"
