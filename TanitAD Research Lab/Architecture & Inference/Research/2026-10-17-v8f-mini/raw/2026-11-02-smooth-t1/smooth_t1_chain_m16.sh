#!/usr/bin/env bash
# SMOOTH-T1 MATCHED-THREAD (16) rerun, one run at a time, same design (Master Mind 2026-11-02: the banked Y ran at 16 threads; K1/K2/K5 failed at 1 thread). Derived from smooth_t1_chain.sh.
# SMOOTH-T1 rollout chain (dev-box CPU, zero training).  Gate: PREREG_FROZEN.md must hash to FROZEN_SHA.txt.  Then five T1-kin rollouts of the PATCHED tool:
#   K1OFF_DAGOACC_s0   --smooth off  (known answer K1: bit-for-bit the banked DAGOACC_s0 dump and ticks)
#   SMOOTH_{DAGOACC,SOUP}_s{0,1}   --smooth sg72 (arm X), the banked rollout settings otherwise unchanged
# Wave A (4 procs x 1 thread) then wave B (the rest).  Each run writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch, an incomplete out dir is MOVED to _incomplete_*.
# Then extract (19 rows per run) and read.  Markers: ZZSMT-*.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/smooth_t1
S=/f/TanitAD-scratch/smooth_t1
OUT=$S/out_m16
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=${SMT_THREADS:-16} MKL_NUM_THREADS=${SMT_THREADS:-16} OPENBLAS_NUM_THREADS=${SMT_THREADS:-16} PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain_m16.out"; }

want=$(sed -n 's/^PREREG_FROZEN.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA.txt"); have=$(sha256sum "$PK/PREREG_FROZEN.md" | cut -c1-64)
if [ -z "$want" ] || [ "$want" != "$have" ]; then log "ZZSMT-STOP prereg does not hash to FROZEN_SHA.txt ($have vs $want)"; exit 2; fi
log "ZZSMT-START gate ok $have threads=${SMT_THREADS:-16}"

run_one() {   # tag tree ckpt smooth
  local tag=$1 tree=$2 ckpt=$3 smooth=$4
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZSMT-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZSMT-RUN $tag smooth=$smooth"
  "$PY" -I "$tree/stack/scripts/v8f_t1_eval.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu --smooth "$smooth" > "$PK/logs/m16_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "smooth": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tree": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$smooth" "$ckpt" \
       "$(sha256sum "$ckpt" | cut -c1-64)" "$tree" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" "${SMT_THREADS:-16}" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZSMT-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZSMT-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

DAG=$S/tree_dag; LI=$S/tree_li
C_D0=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; C_D1=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
C_S0=$ART/dagland/soup/ckpt/M2b_s0_soup/ckpt_last.pt; C_S1=$ART/dagland/soup/ckpt/M2b_s1_soup/ckpt_last.pt

run_one K1OFF_DAGOACC_s0 "$DAG" "$C_D0" off
run_one SMOOTH_DAGOACC_s0 "$DAG" "$C_D0" sg72
run_one SMOOTH_DAGOACC_s1 "$DAG" "$C_D1" sg72
run_one SMOOTH_SOUP_s0 "$LI" "$C_S0" sg72
run_one SMOOTH_SOUP_s1 "$LI" "$C_S1" sg72
log "ZZSMT-ROLLOUTS-DONE"
for t in K1OFF_DAGOACC_s0 SMOOTH_DAGOACC_s0 SMOOTH_DAGOACC_s1 SMOOTH_SOUP_s0 SMOOTH_SOUP_s1; do [ -f "$OUT/done_$t.txt" ] || { log "ZZSMT-STOP missing done_$t"; exit 3; }; done
if [ -f "$PK/code/smooth_t1_read.py" ]; then
  log "ZZSMT-READ start"
  "$PY" -I "$PK/code/smooth_t1_read.py" --out-dir "$OUT" --raw "$PK/raw_m16" > "$PK/logs/read_m16.out" 2>&1; log "ZZSMT-READ rc=$?"
else
  log "ZZSMT-READ skipped: code/smooth_t1_read.py not present (run it by hand)"
fi
log "ZZSMT-CHAIN-DONE"
