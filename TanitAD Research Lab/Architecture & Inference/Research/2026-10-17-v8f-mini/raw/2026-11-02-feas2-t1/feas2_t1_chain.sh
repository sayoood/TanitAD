#!/usr/bin/env bash
# FEAS2-T1 rollout chain (H-FEAS2-T1-1).  Dev-box CPU, zero training, 16 threads, ONE run at a time.
#   SREP2_DAGOACC_s0                         --smooth sg72                   (KA1: a fresh rerun of ONE arm-seed; must equal kappa_t1's SREP_DAGOACC_s0 bit for bit)
#   {KBOX,ABOX,KABOX}_<arm>_s<seed>          --smooth sg72 --feasible {kbox,abox,kabox}      (12 arms; Y = kappa_t1's SREP_*, reused)
# Gate: PREREG_FROZEN.md must hash to FROZEN_SHA.txt; both trees carry the card's tool.  Each run writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch
# (RESUME = launch again); an incomplete out dir is MOVED to _incomplete_*.  Then the read (code/feas2_t1_read.py).  Markers: ZZFEAS2-* in logs/chain.out;
# raw/CHAIN.done on success, raw/CHAIN.fail (with the reason) on any nonzero rc.  Priority order: KA1 rerun, then DAG-OACC s0, s1, SOUP s0, s1, each x kbox, abox, kabox.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/feas2_t1
S=/f/TanitAD-scratch/feas2_t1
OUT=$S/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TH=16
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZFEAS2-FAIL $*"; echo "$(date +%FT%T) $*" > "$PK/raw/CHAIN.fail"; exit "${2:-3}"; }
rm -f "$PK/raw/CHAIN.done" "$PK/raw/CHAIN.fail"

want=$(sed -n 's/^PREREG_FROZEN.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA.txt"); have=$(sha256sum "$PK/PREREG_FROZEN.md" | cut -c1-64)
if [ -z "$want" ] || [ "$want" != "$have" ]; then fail "prereg does not hash to FROZEN_SHA.txt ($have vs $want)" 2; fi
TOOL_SHA=4b66cc86cc5afe185736a65000c7f8cb1aa0c89d22bcd32770c99ad7971596df
for t in tree_dag tree_li; do
  got=$(sha256sum "$S/$t/stack/scripts/v8f_t1_eval.py" | cut -c1-64)
  [ "$got" = "$TOOL_SHA" ] || fail "$t tool sha $got != $TOOL_SHA" 2
done
for t in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do [ -f "/f/TanitAD-scratch/kappa_t1/out/done_SREP_$t.txt" ] || fail "reused Y missing: kappa_t1 done_SREP_$t.txt" 2; done
log "ZZFEAS2-START prereg ok $have tool $TOOL_SHA threads=$TH"

run_one() {   # tag tree ckpt extra-flags...
  local tag=$1 tree=$2 ckpt=$3; shift 3
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZFEAS2-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZFEAS2-RUN $tag flags=$*"
  "$PY" -I "$tree/stack/scripts/v8f_t1_eval.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu "$@" > "$PK/logs/run_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tree": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$ckpt" \
       "$(sha256sum "$ckpt" | cut -c1-64)" "$tree" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval.py" | cut -c1-64)" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZFEAS2-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZFEAS2-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

DAG=$S/tree_dag; LI=$S/tree_li
declare -A CK TREE
CK[DAGOACC_s0]=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; CK[DAGOACC_s1]=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
CK[SOUP_s0]=$ART/dagland/soup/ckpt/M2b_s0_soup/ckpt_last.pt; CK[SOUP_s1]=$ART/dagland/soup/ckpt/M2b_s1_soup/ckpt_last.pt
TREE[DAGOACC_s0]=$DAG; TREE[DAGOACC_s1]=$DAG; TREE[SOUP_s0]=$LI; TREE[SOUP_s1]=$LI
nfail=0
run_one SREP2_DAGOACC_s0 "$DAG" "${CK[DAGOACC_s0]}" --smooth sg72 || nfail=$((nfail+1))
for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do
  for m in kbox abox kabox; do
    P=$(echo "$m" | tr a-z A-Z)
    run_one "${P}_$as" "${TREE[$as]}" "${CK[$as]}" --smooth sg72 --feasible "$m" || nfail=$((nfail+1))
  done
done
log "ZZFEAS2-ROLLOUTS-DONE nfail=$nfail"
missing=""
for t in SREP2_DAGOACC_s0 $(for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do for p in KBOX ABOX KABOX; do echo ${p}_$as; done; done); do
  [ -f "$OUT/done_$t.txt" ] || missing="$missing $t"
done
[ -z "$missing" ] || fail "missing done markers:$missing" 3
log "ZZFEAS2-READ start"
"$PY" -I "$PK/code/feas2_t1_read.py" --out-dir "$OUT" --raw "$PK/raw" > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZFEAS2-READ rc=$rc"
[ $rc -eq 0 ] || fail "read rc=$rc (see logs/read.out)" "$rc"
echo "$(date +%FT%T) FEAS2_T1.json written" > "$PK/raw/CHAIN.done"
log "ZZFEAS2-CHAIN-DONE"
