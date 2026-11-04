#!/usr/bin/env bash
# LEVELPROBE Stage-2 rollout chain (H-LEVELPROBE-1).  Dev-box CPU, zero training, 16 threads, ONE run at a time.  Reads PREREG_LEVELPROBE.md section C: ONLY the as-written arms run here.
#   HT0_{DAGOACC,SOUP}_s{0,1}   --smooth sg72 --histswap cv_t0     (HLVL-T0, ORACLE: the logged speed at t0 held constant; Y = kappa_t1's SREP_*, reused)
#   LPKA1_DAGOACC_s0            --smooth sg72                      (KA1: a fresh default-path rerun with the NEW tool must equal the reused Y bit for bit)
#   HPR_*                       --histswap cv_pred --level-file F  ONLY IF $PK/AMEND_A1_FROZEN.txt exists (an approved, hashed amendment) AND the level file named in it exists and hashes as pinned.  Never otherwise.
# Deadline: every rollout must have FINISHED by 05:00 on 2026-11-04 (another read needs this CPU from ~05:15); a run that would START after 04:45 is skipped, and the chain says so.
# Each run writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch again); an incomplete out dir is MOVED to _incomplete_*.  Then the read (code/levelprobe_read.py).
# Markers: ZZLP-* in logs/chain.out; raw/LP.done on success, raw/LP.fail (with the reason) on any nonzero rc.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/levelprobe
S=/f/TanitAD-scratch/levelprobe_scratch
OUT=$S/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TH=16
DEADLINE_START=$(date -d "2026-11-04 04:45:00" +%s)
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZLP-FAIL $*"; echo "$(date +%FT%T) $*" > "$PK/raw/LP.fail"; exit "${2:-3}"; }
for m in LP.done LP.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

want=$(sed -n 's/^PREREG_LEVELPROBE.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA.txt"); have=$(sha256sum "$PK/PREREG_LEVELPROBE.md" | cut -c1-64)
if [ -z "$want" ] || [ "$want" != "$have" ]; then fail "prereg does not hash to FROZEN_SHA.txt ($have vs $want)" 2; fi
TOOL_SHA=$(sed -n 's/^tool code\/v8f_t1_eval_levelprobe.py sha256 \([0-9a-f]*\) .*$/\1/p' "$PK/FROZEN_SHA.txt" | tail -1)
[ -n "$TOOL_SHA" ] || fail "no tool sha in FROZEN_SHA.txt" 2
for t in tree_dag tree_li; do
  got=$(sha256sum "$S/$t/stack/scripts/v8f_t1_eval_levelprobe.py" | cut -c1-64)
  [ "$got" = "$TOOL_SHA" ] || fail "$t tool sha $got != $TOOL_SHA" 2
done
for t in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do [ -f "/f/TanitAD-scratch/kappa_t1/out/done_SREP_$t.txt" ] || fail "reused Y missing: kappa_t1 done_SREP_$t.txt" 2; done
log "ZZLP-START prereg ok $have tool $TOOL_SHA threads=$TH"

run_one() {   # tag tree ckpt extra-flags...
  local tag=$1 tree=$2 ckpt=$3; shift 3
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZLP-SKIP $tag (done)"; return 0; }
  if [ "$(date +%s)" -ge "$DEADLINE_START" ]; then log "ZZLP-SKIP-DEADLINE $tag (would start after 04:45)"; return 9; fi
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZLP-RUN $tag flags=$*"
  "$PY" -I "$tree/stack/scripts/v8f_t1_eval_levelprobe.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu "$@" > "$PK/logs/run_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tree": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$ckpt" \
       "$(sha256sum "$ckpt" | cut -c1-64)" "$tree" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval_levelprobe.py" | cut -c1-64)" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZLP-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZLP-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

DAG=$S/tree_dag; LI=$S/tree_li
declare -A CK TREE
CK[DAGOACC_s0]=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; CK[DAGOACC_s1]=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
CK[SOUP_s0]=$ART/dagland/soup/ckpt/M2b_s0_soup/ckpt_last.pt; CK[SOUP_s1]=$ART/dagland/soup/ckpt/M2b_s1_soup/ckpt_last.pt
TREE[DAGOACC_s0]=$DAG; TREE[DAGOACC_s1]=$DAG; TREE[SOUP_s0]=$LI; TREE[SOUP_s1]=$LI
nfail=0; nskip=0
for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do
  run_one "HT0_$as" "${TREE[$as]}" "${CK[$as]}" --smooth sg72 --histswap cv_t0; rc=$?
  [ $rc -eq 9 ] && nskip=$((nskip+1)) || { [ $rc -ne 0 ] && nfail=$((nfail+1)); }
done
run_one LPKA1_DAGOACC_s0 "$DAG" "${CK[DAGOACC_s0]}" --smooth sg72; rc=$?
[ $rc -eq 9 ] && nskip=$((nskip+1)) || { [ $rc -ne 0 ] && nfail=$((nfail+1)); }

# HLVL-PRED: gated on an approved, hashed amendment.  NOT part of the as-written card (PREREG section C).
if [ -f "$PK/AMEND_A1_FROZEN.txt" ]; then
  LF=$(sed -n 's/^level_file \(.*\) sha256 [0-9a-f]*$/\1/p' "$PK/AMEND_A1_FROZEN.txt" | head -1); LFS=$(sed -n 's/^level_file .* sha256 \([0-9a-f]*\)$/\1/p' "$PK/AMEND_A1_FROZEN.txt" | head -1)
  if [ -n "$LF" ] && [ -f "$LF" ] && [ "$(sha256sum "$LF" | cut -c1-64)" = "$LFS" ]; then
    for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do
      run_one "HPR_$as" "${TREE[$as]}" "${CK[$as]}" --smooth sg72 --histswap cv_pred --level-file "$LF"; rc=$?
      [ $rc -eq 9 ] && nskip=$((nskip+1)) || { [ $rc -ne 0 ] && nfail=$((nfail+1)); }
    done
  else
    log "ZZLP-PRED-SKIP amendment frozen but its level file is missing or does not hash as pinned"
  fi
else
  log "ZZLP-PRED-NOT-RUN no AMEND_A1_FROZEN.txt (HLVL-PRED is undefined as written, PREREG section C)"
fi
log "ZZLP-ROLLOUTS-DONE nfail=$nfail nskip=$nskip"
[ "$nskip" -eq 0 ] || fail "$nskip rollouts skipped by the 04:45 start deadline" 9
missing=""
for t in HT0_DAGOACC_s0 HT0_DAGOACC_s1 HT0_SOUP_s0 HT0_SOUP_s1 LPKA1_DAGOACC_s0; do
  [ -f "$OUT/done_$t.txt" ] || missing="$missing $t"
done
[ -z "$missing" ] || fail "missing done markers:$missing" 3
log "ZZLP-READ start"
"$PY" -I "$PK/code/levelprobe_read.py" --out-dir "$OUT" --raw "$PK/raw" > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZLP-READ rc=$rc"
[ $rc -eq 0 ] || fail "read rc=$rc (see logs/read.out)" "$rc"
echo "$(date +%FT%T) LEVELPROBE.json written" > "$PK/raw/LP.done"
log "ZZLP-CHAIN-DONE"
