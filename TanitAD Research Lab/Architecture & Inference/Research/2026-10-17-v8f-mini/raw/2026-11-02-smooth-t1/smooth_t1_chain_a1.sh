#!/usr/bin/env bash
# SMOOTH-T1 AMENDMENT_1 chain (dev-box CPU, zero training).  Gate: PREREG_FROZEN.md and AMENDMENT_1.md must hash to FROZEN_SHA.txt / AMENDMENT_1_SHA.txt.
# Rollouts (16 threads, ONE AT A TIME, same patched tool / trees / checkpoints / arguments as out_m16, but --smooth off):
#   OFFREP_DAGOACC_s0  Y''  (a second --smooth off rollout of DAG-OACC s0)
#   OFF_DAGOACC_s1, OFF_SOUP_s0, OFF_SOUP_s1   Y' of the other three arm-seeds   (Y' of DAG-OACC s0 = K1OFF_DAGOACC_s0, already in out_m16)
# Then code/a1_post.sh (extract, nondeterminism bound, reader, tables, results, manifest) if present.  A finished run is skipped on re-launch; an incomplete out dir is MOVED aside.  Markers: ZZA1-*.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/smooth_t1
S=/f/TanitAD-scratch/smooth_t1
OUT=$S/out_m16
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain_a1.out"; }

want=$(sed -n 's/^PREREG_FROZEN.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA.txt"); have=$(sha256sum "$PK/PREREG_FROZEN.md" | cut -c1-64)
if [ -z "$want" ] || [ "$want" != "$have" ]; then log "ZZA1-STOP prereg does not hash to FROZEN_SHA.txt ($have vs $want)"; exit 2; fi
want2=$(sed -n 's/^AMENDMENT_1.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/AMENDMENT_1_SHA.txt"); have2=$(sha256sum "$PK/AMENDMENT_1.md" | cut -c1-64)
if [ -z "$want2" ] || [ "$want2" != "$have2" ]; then log "ZZA1-STOP amendment does not hash to AMENDMENT_1_SHA.txt ($have2 vs $want2)"; exit 2; fi
log "ZZA1-START gate ok prereg=$have amendment=$have2 threads=16"

run_one() {   # tag tree ckpt   (always --smooth off)
  local tag=$1 tree=$2 ckpt=$3
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZA1-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZA1-RUN $tag smooth=off"
  "$PY" -I "$tree/stack/scripts/v8f_t1_eval.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu --smooth off > "$PK/logs/a1_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "smooth": "off", "ckpt": "%s", "ckpt_sha256": "%s", "tree": "%s", "tool_sha256": "%s", "threads": 16, "wall_s": %s}\n' "$tag" "$ckpt" \
       "$(sha256sum "$ckpt" | cut -c1-64)" "$tree" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZA1-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZA1-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

DAG=$S/tree_dag; LI=$S/tree_li
C_D0=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; C_D1=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
C_S0=$ART/dagland/soup/ckpt/M2b_s0_soup/ckpt_last.pt; C_S1=$ART/dagland/soup/ckpt/M2b_s1_soup/ckpt_last.pt

run_one OFFREP_DAGOACC_s0 "$DAG" "$C_D0"
run_one OFF_DAGOACC_s1 "$DAG" "$C_D1"
run_one OFF_SOUP_s0 "$LI" "$C_S0"
run_one OFF_SOUP_s1 "$LI" "$C_S1"
log "ZZA1-ROLLOUTS-DONE"
for t in K1OFF_DAGOACC_s0 SMOOTH_DAGOACC_s0 SMOOTH_DAGOACC_s1 SMOOTH_SOUP_s0 SMOOTH_SOUP_s1 OFFREP_DAGOACC_s0 OFF_DAGOACC_s1 OFF_SOUP_s0 OFF_SOUP_s1; do [ -f "$OUT/done_$t.txt" ] || { log "ZZA1-STOP missing done_$t"; exit 3; }; done
if [ -f "$PK/code/a1_post.sh" ]; then
  log "ZZA1-POST start"
  bash "$PK/code/a1_post.sh" > "$PK/logs/a1_post.out" 2>&1; log "ZZA1-POST rc=$?"
else
  log "ZZA1-POST skipped: code/a1_post.sh not present (run it by hand)"
fi
log "ZZA1-CHAIN-DONE"
