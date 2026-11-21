#!/usr/bin/env bash
# KAPPA-T1 rollout chain (H-KAPPA-T1-1 Part 2, lever (a) selected by G-KAPPA).  Dev-box CPU, zero training, 16 threads, ONE run at a time.
#   SREP_<arm>_s<seed>  --smooth sg72                 (Y = S rerun fresh; known answer KA1: equals out_m16/SMOOTH_<arm>_s<seed>)
#   FEAS_<arm>_s<seed>  --smooth sg72 --feasible box  (X = S + the box projection)
# Gate: PREREG_FROZEN.md must hash to FROZEN_SHA.txt and raw/GATE_KAPPA.json must select lever a.  Each run writes done_<tag>.txt only on rc 0; a finished run is
# skipped on re-launch; an incomplete out dir is MOVED to _incomplete_*.  Then the read (code/kappa_t1_read.py).  Markers: ZZKAP-*.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/kappa_t1
S=/f/TanitAD-scratch/kappa_t1
OUT=$S/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TH=16
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }

want=$(sed -n 's/^PREREG_FROZEN.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA.txt"); have=$(sha256sum "$PK/PREREG_FROZEN.md" | cut -c1-64)
if [ -z "$want" ] || [ "$want" != "$have" ]; then log "ZZKAP-STOP prereg does not hash to FROZEN_SHA.txt ($have vs $want)"; exit 2; fi
lever=$("$PY" -I -c "import json,sys;d=json.load(open(sys.argv[1]))['decision'];print(d['lever'] if d['controls_pass'] else 'none')" "$PK/raw/GATE_KAPPA.json")
if [ "$lever" != "a" ]; then log "ZZKAP-STOP gate lever is '$lever', this chain runs lever a only"; exit 2; fi
TOOL_SHA=260bd17505726a297afe4691fafe55b06f20af7c41bface7dacf947f5279de93
for t in tree_dag tree_li; do
  got=$(sha256sum "$S/$t/stack/scripts/v8f_t1_eval.py" | cut -c1-64)
  [ "$got" = "$TOOL_SHA" ] || { log "ZZKAP-STOP $t tool sha $got != $TOOL_SHA"; exit 2; }
done
log "ZZKAP-START gate ok $have lever=$lever threads=$TH"

run_one() {   # tag tree ckpt extra-flags...
  local tag=$1 tree=$2 ckpt=$3; shift 3
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZKAP-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZKAP-RUN $tag flags=$*"
  "$PY" -I "$tree/stack/scripts/v8f_t1_eval.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu "$@" > "$PK/logs/run_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tree": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$ckpt" \
       "$(sha256sum "$ckpt" | cut -c1-64)" "$tree" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZKAP-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZKAP-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

DAG=$S/tree_dag; LI=$S/tree_li
C_D0=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; C_D1=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
C_S0=$ART/dagland/soup/ckpt/M2b_s0_soup/ckpt_last.pt; C_S1=$ART/dagland/soup/ckpt/M2b_s1_soup/ckpt_last.pt

run_one SREP_DAGOACC_s0 "$DAG" "$C_D0" --smooth sg72
run_one FEAS_DAGOACC_s0 "$DAG" "$C_D0" --smooth sg72 --feasible box
run_one SREP_DAGOACC_s1 "$DAG" "$C_D1" --smooth sg72
run_one FEAS_DAGOACC_s1 "$DAG" "$C_D1" --smooth sg72 --feasible box
run_one SREP_SOUP_s0 "$LI" "$C_S0" --smooth sg72
run_one FEAS_SOUP_s0 "$LI" "$C_S0" --smooth sg72 --feasible box
run_one SREP_SOUP_s1 "$LI" "$C_S1" --smooth sg72
run_one FEAS_SOUP_s1 "$LI" "$C_S1" --smooth sg72 --feasible box
log "ZZKAP-ROLLOUTS-DONE"
for t in SREP_DAGOACC_s0 FEAS_DAGOACC_s0 SREP_DAGOACC_s1 FEAS_DAGOACC_s1 SREP_SOUP_s0 FEAS_SOUP_s0 SREP_SOUP_s1 FEAS_SOUP_s1; do
  [ -f "$OUT/done_$t.txt" ] || { log "ZZKAP-STOP missing done_$t"; exit 3; }
done
log "ZZKAP-READ start"
"$PY" -I "$PK/code/kappa_t1_read.py" --out-dir "$OUT" --raw "$PK/raw" > "$PK/logs/read.out" 2>&1; log "ZZKAP-READ rc=$?"
log "ZZKAP-CHAIN-DONE"
