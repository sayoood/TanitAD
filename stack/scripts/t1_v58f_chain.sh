#!/bin/bash
# ===========================================================================
# E1.4 — the T1 (action-closed-loop) rows for v5f + v5.8f.   pod5, 2026-08-11
#
# WHAT THIS PRODUCES (the PI's blocker for the v5.8f release row):
#   /workspace/experiments/t1-v58f/t1_v5f_30k.json        arm 1 (frozen trunk)
#   /workspace/experiments/t1-v58f/t1_stage_a_repaired.json arm 2 (repaired)
#   /workspace/experiments/t1-v58f/t1_summary.json        both + PAIRED CIs
#   …/dump_v5f_30k/ep*.npz, …/dump_stage_a/ep*.npz        per-window dumps
#
# Both arms roll the SAME grid (same corpus, --episodes, --window-stride,
# --window, --horizon-k) with --with-t0-open-loop --with-hold-action, so the
# T0-vs-T1 gap AND the hold-action control are measured in the same run —
# that triplet is what made MODEL_REGISTRY §1.12's action-echo finding
# admissible, and a cross-arm PAIRED bootstrap is only valid on one grid.
#
# ⛔ NO GIT IN THIS CHAIN. Pods have no git credentials: `git fetch` HANGS and
# a `checkout -B` after a failed fetch RESETS the tree to an ancient HEAD,
# destroying the file-shipped fix (CLAUDE.md). The orchestrator file-ships
# t1_eval.py / t1_summary.py; this script GREP-VERIFIES the shipped code is
# really present and REFUSES rather than running a stale instrument.
#
# ⛔ ONE TRAINER PER POD: waits for COTRAIN_EXIT before touching the GPU.
# ===========================================================================
set -u
LOG=${T1_LOG:-/tmp/t1_v58f.log}
REPO=${T1_REPO:-/workspace/TanitAD}
OUT=${T1_OUT:-/workspace/experiments/t1-v58f}
CORPUS=${T1_CORPUS:-/workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl}
CKPT_V5F=${T1_CKPT_V5F:-/workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt}
CKPT_STAGEA=${T1_CKPT_STAGEA:-/workspace/experiments/stage-a-predictor/ckpt_stage_a.pt}
# the §1.12 grid defaults; env-overridable so the chain can be time-boxed
# WITHOUT editing the file — whatever is used is stamped into every JSON.
EPISODES=${T1_EPISODES:-40}
STRIDE=${T1_STRIDE:-1}
CHUNK=${T1_CHUNK:-16}
NBOOT=${T1_NBOOT:-2000}
TOOL=$REPO/taniteval/tools/t1_eval.py
SUMM=$REPO/taniteval/tools/t1_summary.py

exec >>"$LOG" 2>&1
echo "=== t1_v58f_chain start $(date -u +%FT%TZ) ==="

# --- 1. the GPU is not ours until the co-train run releases it -------------
until grep -q 'COTRAIN_EXIT' /tmp/cotrain.log 2>/dev/null; do sleep 180; done
echo "[t1] COTRAIN_EXIT seen: $(grep COTRAIN_EXIT /tmp/cotrain.log | tail -1)"

# --- 2. VERIFY THE SHIPPED INSTRUMENT (a missing flag = a stale file) ------
fail() { echo "T1_EXIT=$1"; exit 1; }
[ -f "$TOOL" ] || fail SYNC_FAILED_NO_T1_EVAL
[ -f "$SUMM" ] || fail SYNC_FAILED_NO_T1_SUMMARY
for tok in -- --v2-val-cache --grounding-readout --window-stride \
           run_rollout_ext roll_closed_grounding implied_controls V2RawEp; do
  [ "$tok" = "--" ] && continue
  grep -q -- "$tok" "$TOOL" || { echo "[t1] MISSING '$tok' in $TOOL"; \
    fail SYNC_FAILED_FLAG_MISSING; }
done
grep -q -- "--paired" "$SUMM" || fail SYNC_FAILED_SUMMARY_STALE
python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" "$TOOL" \
  || fail SYNC_FAILED_T1_EVAL_UNPARSEABLE
echo "[t1] instrument verified: adapter flags + functions present in $TOOL"

# --- 3. inputs ------------------------------------------------------------
[ -d "$CORPUS" ] || fail NO_CORPUS
mkdir -p "$OUT" || fail NO_OUT_DIR
cd "$REPO/stack" || fail NO_STACK_DIR
export PYTHONPATH=$REPO/stack           # or trainers/tools die: no tanitad
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6}   # torch spawns ~113 threads/proc
echo "[t1] grid: episodes=$EPISODES window-stride=$STRIDE chunk=$CHUNK " \
     "corpus=$CORPUS"

# --- 3b. PRE-FLIGHT: the adapter's own CPU tests --------------------------
# ⭐ Worth the ~20 s: the pod HAS torchvision, so test_real_v2_provider_path —
# which SKIPS on the dev box — actually drives build_v2_providers -> V2RawEp
# here, before any GPU time. A failure means the shipped adapter is broken;
# refuse rather than produce a T1 row from it.
if python3 -c "import pytest" >/dev/null 2>&1; then
  if python3 -m pytest -q "$REPO/stack/tests/test_t1_v2_adapter.py"; then
    echo "[t1] adapter CPU tests PASS (incl. the real v2-provider path)"
  else
    fail ADAPTER_TESTS_FAILED
  fi
else
  echo "[t1] pytest unavailable — adapter CPU tests skipped (grep verify only)"
fi

run_arm() {                       # $1 = name, $2 = ckpt, $3 = marker
  local name=$1 ckpt=$2 marker=$3
  if [ ! -f "$ckpt" ]; then
    echo "${marker}=SKIPPED_NO_CKPT:$ckpt"
    return 1
  fi
  echo "[t1] === arm $name  ckpt=$ckpt  $(date -u +%FT%TZ) ==="
  python3 -u "$TOOL" \
    --arm "$name" \
    --ckpt "$ckpt" \
    --grounding-readout \
    --v2-val-cache "$CORPUS" \
    --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
    --v2-subframe 176x624 \
    --episodes "$EPISODES" --window-stride "$STRIDE" --chunk "$CHUNK" \
    --with-t0-open-loop --with-hold-action \
    --n-boot "$NBOOT" \
    --dump-dir "$OUT/dump_${name//-/_}" \
    --out "$OUT/t1_${name//-/_}.json"
  local rc=$?
  echo "${marker}=$rc"
  return $rc
}

# --- 4. the two arms, same grid -------------------------------------------
run_arm v5f-30k "$CKPT_V5F" T1_V5F_EXIT
rc_a=$?
run_arm stage-a-repaired "$CKPT_STAGEA" T1_STAGEA_EXIT
rc_b=$?

# --- 5. the combined record + the CROSS-ARM PAIRED bootstrap --------------
args=()
[ $rc_a -eq 0 ] && args+=(--arm "v5f-30k=$OUT/t1_v5f_30k.json")
[ $rc_b -eq 0 ] && args+=(--arm "stage-a-repaired=$OUT/t1_stage_a_repaired.json")
if [ ${#args[@]} -eq 0 ]; then
  echo "T1_SUMMARY_EXIT=SKIPPED_NO_ARMS"
  echo "T1_EXIT=NO_ARMS_PRODUCED"
  exit 1
fi
[ $rc_a -eq 0 ] && [ $rc_b -eq 0 ] && \
  args+=(--paired "stage-a-repaired,v5f-30k")
python3 -u "$SUMM" "${args[@]}" --n-boot "$NBOOT" \
  --out "$OUT/t1_summary.json"
rc_s=$?
echo "T1_SUMMARY_EXIT=$rc_s"

# --- 6. one marker the watcher greps for ----------------------------------
if [ $rc_a -eq 0 ] && [ $rc_b -eq 0 ] && [ $rc_s -eq 0 ]; then
  echo "T1_EXIT=0"
else
  echo "T1_EXIT=PARTIAL:v5f=$rc_a,stage_a=$rc_b,summary=$rc_s"
fi
echo "=== t1_v58f_chain end $(date -u +%FT%TZ) ==="
