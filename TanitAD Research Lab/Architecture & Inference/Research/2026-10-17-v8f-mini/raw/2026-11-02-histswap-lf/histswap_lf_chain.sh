#!/usr/bin/env bash
# HISTSWAP-LF rollout chain (H-HISTSWAP-LF-1).  Dev-box CPU, zero training, 16 threads, ONE run at a time, STARTED ONLY AFTER the HISTSWAP chain has written its marker (it owns the CPU until then).
#   SREP5_DAGOACC_s0                 --smooth sg72                          (KA1: a fresh --histswap off rerun with the LF tool; must equal kappa_t1's SREP_DAGOACC_s0 bit for bit)
#   HCV_{DAGOACC,SOUP}_s{0,1}        --smooth sg72 --histswap cv_exec      (HCV, deployable; Y = kappa_t1's SREP_*, reused)
#   HLV_{DAGOACC,SOUP}_s{0,1}        --smooth sg72 --histswap cv_logged    (HLVL, ORACLE level only)
# Gate 1: wait (poll in here, never in a model turn) for histswap/raw/HISTSWAP.done or HISTSWAP.fail, at most 4 h.  Gate 2: PREREG_HISTSWAP_LF_FROZEN.md must hash to FROZEN_SHA.txt; both trees carry the card's tool.
# Each run writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch again); an incomplete out dir is MOVED to _incomplete_*.  Then the read (code/histswap_lf_read.py).
# Markers: ZZHLF-* in logs/chain.out; raw/HLF.done on success, raw/HLF.fail (with the reason) on any nonzero rc.  Priority order: KA1 rerun, HCV (4 cells), HLVL (4 cells).
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/histswap_lf
S=/f/TanitAD-scratch/histswap_lf
OUT=$S/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
GATE=/e/Projects/TanitAD-artifacts/dagland/histswap/raw
TH=16
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZHLF-FAIL $*"; echo "$(date +%FT%T) $*" > "$PK/raw/HLF.fail"; exit "${2:-3}"; }
for m in HLF.done HLF.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

log "ZZHLF-WAIT for $GATE/HISTSWAP.done|HISTSWAP.fail (the HISTSWAP chain owns the CPU until then)"
waited=0
until [ -f "$GATE/HISTSWAP.done" ] || [ -f "$GATE/HISTSWAP.fail" ]; do
  sleep 20; waited=$((waited+20))
  [ $waited -ge 14400 ] && fail "no HISTSWAP marker after 4 h" 4
done
log "ZZHLF-GATE-OPEN after ${waited}s: $(ls "$GATE" | grep '^HISTSWAP\.' | tr '\n' ' ')"

want=$(sed -n 's/^PREREG_HISTSWAP_LF_FROZEN.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA.txt"); have=$(sha256sum "$PK/PREREG_HISTSWAP_LF_FROZEN.md" | cut -c1-64)
if [ -z "$want" ] || [ "$want" != "$have" ]; then fail "prereg does not hash to FROZEN_SHA.txt ($have vs $want)" 2; fi
TOOL_SHA=$(sed -n 's/^tool code\/v8f_t1_eval_histswap_lf.py sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA.txt")
[ -n "$TOOL_SHA" ] || fail "no tool sha in FROZEN_SHA.txt" 2
for t in tree_dag tree_li; do
  got=$(sha256sum "$S/$t/stack/scripts/v8f_t1_eval_histswap_lf.py" | cut -c1-64)
  [ "$got" = "$TOOL_SHA" ] || fail "$t tool sha $got != $TOOL_SHA" 2
done
for t in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do [ -f "/f/TanitAD-scratch/kappa_t1/out/done_SREP_$t.txt" ] || fail "reused Y missing: kappa_t1 done_SREP_$t.txt" 2; done
log "ZZHLF-START prereg ok $have tool $TOOL_SHA threads=$TH"

run_one() {   # tag tree ckpt extra-flags...
  local tag=$1 tree=$2 ckpt=$3; shift 3
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZHLF-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZHLF-RUN $tag flags=$*"
  "$PY" -I "$tree/stack/scripts/v8f_t1_eval_histswap_lf.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu "$@" > "$PK/logs/run_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tree": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$ckpt" \
       "$(sha256sum "$ckpt" | cut -c1-64)" "$tree" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval_histswap_lf.py" | cut -c1-64)" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZHLF-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZHLF-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

DAG=$S/tree_dag; LI=$S/tree_li
declare -A CK TREE
CK[DAGOACC_s0]=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; CK[DAGOACC_s1]=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
CK[SOUP_s0]=$ART/dagland/soup/ckpt/M2b_s0_soup/ckpt_last.pt; CK[SOUP_s1]=$ART/dagland/soup/ckpt/M2b_s1_soup/ckpt_last.pt
TREE[DAGOACC_s0]=$DAG; TREE[DAGOACC_s1]=$DAG; TREE[SOUP_s0]=$LI; TREE[SOUP_s1]=$LI
nfail=0
run_one SREP5_DAGOACC_s0 "$DAG" "${CK[DAGOACC_s0]}" --smooth sg72 --histswap off || nfail=$((nfail+1))
for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do
  run_one "HCV_$as" "${TREE[$as]}" "${CK[$as]}" --smooth sg72 --histswap cv_exec || nfail=$((nfail+1))
done
for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do
  run_one "HLV_$as" "${TREE[$as]}" "${CK[$as]}" --smooth sg72 --histswap cv_logged || nfail=$((nfail+1))
done
log "ZZHLF-ROLLOUTS-DONE nfail=$nfail"
missing=""
for t in SREP5_DAGOACC_s0 HCV_DAGOACC_s0 HCV_DAGOACC_s1 HCV_SOUP_s0 HCV_SOUP_s1 HLV_DAGOACC_s0 HLV_DAGOACC_s1 HLV_SOUP_s0 HLV_SOUP_s1; do
  [ -f "$OUT/done_$t.txt" ] || missing="$missing $t"
done
[ -z "$missing" ] || fail "missing done markers:$missing" 3
log "ZZHLF-READ start"
"$PY" -I "$PK/code/histswap_lf_read.py" --out-dir "$OUT" --raw "$PK/raw" > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZHLF-READ rc=$rc"
[ $rc -eq 0 ] || fail "read rc=$rc (see logs/read.out)" "$rc"
echo "$(date +%FT%T) HISTSWAP_LF_T1.json written" > "$PK/raw/HLF.done"
log "ZZHLF-CHAIN-DONE"
