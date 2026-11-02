#!/usr/bin/env bash
# SMOOTH-T1 thread-noise REPLICATE (supplementary to the frozen card; decides nothing).  The four sg72 rollouts again, tool/ckpts/args exactly as smooth_t1_chain.sh, at 2 threads each
# (the card's rollouts ran at 1 thread; the banked Y at 16).  Purpose: how much of the X rows / the outcome moves under float32 thread-order noise alone.  Out: out_t2/<tag>.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/smooth_t1
S=/f/TanitAD-scratch/smooth_t1
OUT=$S/out_t2
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
THR=2
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$THR MKL_NUM_THREADS=$THR OPENBLAS_NUM_THREADS=$THR PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/threadrep.out"; }
want=$(sed -n 's/^PREREG_FROZEN.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA.txt"); have=$(sha256sum "$PK/PREREG_FROZEN.md" | cut -c1-64)
[ "$want" = "$have" ] || { log "ZZTR-STOP prereg hash mismatch"; exit 2; }
run_one() {   # tag tree ckpt
  local tag=$1 tree=$2 ckpt=$3
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZTR-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"; local t0=$(date +%s)
  log "ZZTR-RUN $tag threads=$THR"
  "$PY" -I "$tree/stack/scripts/v8f_t1_eval.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu --smooth sg72 > "$PK/logs/threadrep_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then printf '{"tag": "%s", "smooth": "sg72", "threads": %s, "tool_sha256": "%s", "wall_s": %s}\n' "$tag" "$THR" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"; log "ZZTR-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"; else log "ZZTR-RUN-FAIL $tag rc=$rc"; fi
}
DAG=$S/tree_dag; LI=$S/tree_li
C_D0=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; C_D1=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
C_S0=$ART/dagland/soup/ckpt/M2b_s0_soup/ckpt_last.pt; C_S1=$ART/dagland/soup/ckpt/M2b_s1_soup/ckpt_last.pt
run_one SMOOTH_DAGOACC_s0 "$DAG" "$C_D0" &
run_one SMOOTH_DAGOACC_s1 "$DAG" "$C_D1" &
wait
run_one SMOOTH_SOUP_s0 "$LI" "$C_S0" &
run_one SMOOTH_SOUP_s1 "$LI" "$C_S1" &
wait
log "ZZTR-DONE"
