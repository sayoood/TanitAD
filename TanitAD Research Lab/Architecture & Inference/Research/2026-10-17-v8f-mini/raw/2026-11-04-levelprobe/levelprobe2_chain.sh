#!/usr/bin/env bash
# LEVELPROBE2 rollout chain (H-LEVELPROBE-2, PREREG_LEVELPROBE2.md).  Dev-box CPU, zero training, 16 threads, ONE run at a time.
#   LP2KA_DAGOACC_s0 (L_T0 level file; must equal HT0_DAGOACC_s0 bit for bit), then M6_* (MEAN6), F3_* (FUT3), F1_* (FUT1) on the four cells, all `--smooth sg72 --histswap cv_pred --level-file`.
# Deadline: every rollout must have FINISHED by 05:00 on 2026-11-04; a run that would START after 04:45 is skipped and the chain fails saying so.  RESUME = launch again (finished runs are skipped).
# Markers: ZZLP2-* in logs/chain2.out; raw/LP2.done on success (raw/LEVELPROBE2.json) / raw/LP2.fail (reason inside).
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
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain2.out"; }
fail() { log "ZZLP2-FAIL $*"; echo "$(date +%FT%T) $*" > "$PK/raw/LP2.fail"; exit "${2:-3}"; }
for m in LP2.done LP2.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

want=$(sed -n 's/^PREREG_LEVELPROBE2.md sha256 \([0-9a-f]*\)$/\1/p' "$PK/FROZEN_SHA2.txt"); have=$(sha256sum "$PK/PREREG_LEVELPROBE2.md" | cut -c1-64)
if [ -z "$want" ] || [ "$want" != "$have" ]; then fail "prereg does not hash to FROZEN_SHA2.txt ($have vs $want)" 2; fi
TOOL_SHA=$(sed -n 's/^tool code\/v8f_t1_eval_levelprobe.py sha256 \([0-9a-f]*\) .*$/\1/p' "$PK/FROZEN_SHA.txt" | tail -1)
[ -n "$TOOL_SHA" ] || fail "no tool sha in FROZEN_SHA.txt" 2
for t in tree_dag tree_li; do
  got=$(sha256sum "$S/$t/stack/scripts/v8f_t1_eval_levelprobe.py" | cut -c1-64)
  [ "$got" = "$TOOL_SHA" ] || fail "$t tool sha $got != $TOOL_SHA" 2
done
for t in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do [ -f "/f/TanitAD-scratch/kappa_t1/out/done_SREP_$t.txt" ] || fail "reused Y missing: kappa_t1 done_SREP_$t.txt" 2; done
log "ZZLP2-START prereg ok $have tool $TOOL_SHA threads=$TH"

run_one() {   # tag tree ckpt extra-flags...
  local tag=$1 tree=$2 ckpt=$3; shift 3
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZLP2-SKIP $tag (done)"; return 0; }
  if [ "$(date +%s)" -ge "$DEADLINE_START" ]; then log "ZZLP2-SKIP-DEADLINE $tag (would start after 04:45)"; return 9; fi
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZLP2-RUN $tag flags=$*"
  "$PY" -I "$tree/stack/scripts/v8f_t1_eval_levelprobe.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu "$@" > "$PK/logs/run_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tree": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$ckpt" \
       "$(sha256sum "$ckpt" | cut -c1-64)" "$tree" "$(sha256sum "$tree/stack/scripts/v8f_t1_eval_levelprobe.py" | cut -c1-64)" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZLP2-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZLP2-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

DAG=$S/tree_dag; LI=$S/tree_li
declare -A CK TREE
CK[DAGOACC_s0]=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; CK[DAGOACC_s1]=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
CK[SOUP_s0]=$ART/dagland/soup/ckpt/M2b_s0_soup/ckpt_last.pt; CK[SOUP_s1]=$ART/dagland/soup/ckpt/M2b_s1_soup/ckpt_last.pt
TREE[DAGOACC_s0]=$DAG; TREE[DAGOACC_s1]=$DAG; TREE[SOUP_s0]=$LI; TREE[SOUP_s1]=$LI
lvl() { sed -n "s#^level_file \(.*/L_$1.npz\) sha256 [0-9a-f]*\$#\1#p" "$PK/FROZEN_SHA2.txt" | tail -1; }
lvs() { sed -n "s#^level_file .*/L_$1.npz sha256 \([0-9a-f]*\)\$#\1#p" "$PK/FROZEN_SHA2.txt" | tail -1; }
for k in T0 MEAN6 FUT1 FUT3; do
  f=$(lvl $k); [ -n "$f" ] && [ -f "$f" ] && [ "$(sha256sum "$f" | cut -c1-64)" = "$(lvs $k)" ] || fail "level file L_$k missing or does not hash as pinned in FROZEN_SHA2.txt" 2
done
nfail=0; nskip=0
step() { run_one "$@"; local rc=$?; if [ $rc -eq 9 ]; then nskip=$((nskip+1)); elif [ $rc -ne 0 ]; then nfail=$((nfail+1)); fi; }
step LP2KA_DAGOACC_s0 "$DAG" "${CK[DAGOACC_s0]}" --smooth sg72 --histswap cv_pred --level-file "$(lvl T0)"
for pair in "M6 MEAN6" "F3 FUT3" "F1 FUT1"; do
  set -- $pair
  for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do
    step "$1_$as" "${TREE[$as]}" "${CK[$as]}" --smooth sg72 --histswap cv_pred --level-file "$(lvl $2)"
  done
done
log "ZZLP2-ROLLOUTS-DONE nfail=$nfail nskip=$nskip"
[ "$nskip" -eq 0 ] || fail "$nskip rollouts skipped by the 04:45 start deadline" 9
missing=""
for t in LP2KA_DAGOACC_s0 LPKA1_DAGOACC_s0 HT0_DAGOACC_s0; do [ -f "$OUT/done_$t.txt" ] || missing="$missing $t"; done
for p in M6 F3 F1; do for as in DAGOACC_s0 DAGOACC_s1 SOUP_s0 SOUP_s1; do [ -f "$OUT/done_${p}_$as.txt" ] || missing="$missing ${p}_$as"; done; done
[ -z "$missing" ] || fail "missing done markers:$missing" 3
log "ZZLP2-READ start"
"$PY" -I "$PK/code/levelprobe2_read.py" --out-dir "$OUT" --raw "$PK/raw" > "$PK/logs/read2.out" 2>&1; rc=$?
log "ZZLP2-READ rc=$rc"
[ $rc -eq 0 ] || fail "read rc=$rc (see logs/read2.out)" "$rc"
echo "$(date +%FT%T) LEVELPROBE2.json written" > "$PK/raw/LP2.done"
log "ZZLP2-CHAIN-DONE"
