#!/usr/bin/env bash
# Reproduce every arm of this package. CPU ONLY -- the dev-box GPU was held at
# ~100 % by another package while this ran, and adding GPU load is refused by
# CLAUDE.md ("never add GPU/RAM load to a box that is training").
#
#   bash code/run_all.sh
#
# ⚠️ Peak RSS is ~19 GB for the `forward` arm at batch 1 (MEASURED). The arms
# are SEQUENTIAL for that reason; do not parallelise them on a 32 GB box.
set -u
WT="${TANITAD_WT:-C:/Users/Admin/tanitad-wt-e2e}"
PY="${TANITAD_PY:-C:/Users/Admin/venvs/tanitad/Scripts/python.exe}"
PKG="$WT/TanitAD Research Lab/Architecture & Inference/Research/2026-09-17-refcv6-e2e-1024"
CACHE="${E2E_CACHE:-D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl}"
ANCH="${E2E_ANCHORS:-D:/Projects/TanitAD-artifacts/hf-refcv5v2/anchors.pt}"
V7="${E2E_V7:-C:/Users/Admin/refcv5cmp/data/s2_labels_v7.2_eval.jsonl.gz}"

export TANITAD_WT="$WT" PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES="" \
       OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" \
       E2E_CACHE="$CACHE" E2E_ANCHORS="$ANCH" E2E_V7="$V7"
mkdir -p "$PKG/logs" "$PKG/raw"

run () {                       # run <logname> <cmd...>
  local name="$1"; shift
  echo "=== $name ===" | tee -a "$PKG/logs/run_all.log"
  "$@" > "$PKG/logs/$name.log" 2>&1
  local rc=$?                  # NOT through a pipe: $? after `| tee` is tee's
  echo "$name EXIT=$rc" | tee -a "$PKG/logs/run_all.log"
  tail -3 "$PKG/logs/$name.log" | tee -a "$PKG/logs/run_all.log"
}

COMMON=(--cache "$CACHE" --anchors "$ANCH" --v7-labels "$V7")

run assembly   "$PY" "$PKG/code/e2e_1024.py" --arm assembly   "${COMMON[@]}"
run forward    "$PY" "$PKG/code/e2e_1024.py" --arm forward    --clips 0 --tag full139 "${COMMON[@]}"
run perception "$PY" "$PKG/code/e2e_1024.py" --arm perception --clips 0 "${COMMON[@]}"
run tactical   "$PY" "$PKG/code/e2e_1024.py" --arm tactical   --clips 5 "${COMMON[@]}"
run identity   "$PY" "$PKG/code/e2e_1024.py" --arm identity   "${COMMON[@]}"
run forensics  "$PY" "$PKG/code/zero_grad_forensics.py"
run pretrained "$PY" "$PKG/code/pretrained_guard.py"
run joinkey    "$PY" "$PKG/code/join_frame_key.py"
run cost       "$PY" "$PKG/code/cost_model.py"
run summarize  "$PY" "$PKG/code/summarize.py"
echo "ALL DONE -- see $PKG/raw/summary.json"
