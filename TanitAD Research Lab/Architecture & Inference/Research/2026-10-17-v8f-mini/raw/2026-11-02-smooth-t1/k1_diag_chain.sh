#!/usr/bin/env bash
# SMOOTH-T1 K1 diagnostic (supplementary to the frozen card; the card's K1 is read as written elsewhere).  Two DAG-OACC s0 rollouts, banked settings, CPU, 4 threads in total:
#   K1U_DAGOACC_s0   the UNPATCHED tip tool (sha256 61e140ed..., the banked tree's tool) at 1 thread, no --smooth   -> vs K1OFF_DAGOACC_s0 (patched, --smooth off, 1 thread): is the patch neutral?
#   K1P3_DAGOACC_s0  the PATCHED tool, --smooth off, 3 threads                                                       -> vs K1OFF (1 thread): how big is thread-count noise on its own?
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/smooth_t1
S=/f/TanitAD-scratch/smooth_t1
OUT=$S/out_diag
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
export CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/k1_diag.out"; }
C_D0=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt
run_one() {   # tag tree threads [extra args]
  local tag=$1 tree=$2 thr=$3; shift 3
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZK1D-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"; local t0=$(date +%s)
  log "ZZK1D-RUN $tag threads=$thr tool=$(sha256sum "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-16)"
  OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr "$PY" -I "$tree/stack/scripts/v8f_t1_eval.py" run --ckpt "$C_D0" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu "$@" > "$PK/logs/$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then printf '{"tag": "%s", "threads": %s, "tool_sha256": "%s", "wall_s": %s}\n' "$tag" "$thr" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"; log "ZZK1D-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"; else log "ZZK1D-RUN-FAIL $tag rc=$rc"; fi
}
run_one K1U_DAGOACC_s0 "$S/tree_dag_unp" 1 &
run_one K1P3_DAGOACC_s0 "$S/tree_dag" 3 --smooth off &
wait
log "ZZK1D-DONE"
