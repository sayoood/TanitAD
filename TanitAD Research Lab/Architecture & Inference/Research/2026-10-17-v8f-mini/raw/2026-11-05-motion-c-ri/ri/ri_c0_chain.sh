#!/usr/bin/env bash
# REINDEXED production chain: the C0 checkpoints (RECOVER-FT arm C0, s0 and s1) over ALL 2,399 eval windows with --scene reindexed (RIC0_*) and with --scene reindexed --feasible kbox (RIC0C_*, the reader's composite form).
# Dev-box CPU, ZERO training, 12 threads, ONE rollout at a time.  Order (priority): RIC0_s0, RIC0_s1, RIC0C_s0, RIC0C_s1.  Flags = the reader's (recover_read.py run_flags + rnext_read_chain.sh run_tool):
#   --smooth sg72 --scene reindexed [--feasible kbox]  --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks <out>/ticks --dump-heads --device cpu
# Gates: PINS.txt (tool, patch, receipt script, tests) and the unit-test / mutation / smoke logs must hold; both C0 checkpoints must hash to their SHA256SUMS.txt line.  Each run writes done_<tag>.txt only on rc 0; a finished run
# is skipped on re-launch (RESUME = launch ri_c0_chain.cmd again via WMI Win32_Process.Create); an incomplete out dir is MOVED to _incomplete_*.  After the four rollouts code/ri_receipts.py gates the receipts and writes
# raw/RI_RECEIPTS.json; raw/RI.done is written only when every rollout and every receipt check passed, raw/RI.fail (reason inside) otherwise.  No other rollout is waited for (the dev box has 24 logical CPUs; 12 threads here).
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/motion_c/ri
PKW=E:/Projects/TanitAD-artifacts/dagland/motion_c/ri
S=/f/TanitAD-scratch/motion_c/ri
SW=F:/TanitAD-scratch/motion_c/ri
OUT=$S/out
OUTW=$SW/out
TOOLW=$SW/tree_ri/stack/scripts/v8f_t1_eval_ri.py
TOOL=$S/tree_ri/stack/scripts/v8f_t1_eval_ri.py
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
RDB=/e/Projects/TanitAD-artifacts/v8f-recover
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
TH=12
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZRI-FAIL $*"; echo "$(date +%FT%T) $*" > "$PK/raw/RI.fail"; exit "${2:-3}"; }
if [ -f "$PK/raw/RI.running" ]; then
  op=$(cat "$PK/raw/RI.running" | tr -d '\r\n ')
  if [ -n "$op" ] && kill -0 "$op" 2>/dev/null; then log "ZZRI-ALREADY-RUNNING pid $op: not starting a second chain"; exit 9; fi
  mv "$PK/raw/RI.running" "$PK/raw/_stale_RI.running_$(date +%H%M%S)"
fi
echo $$ > "$PK/raw/RI.running"
trap 'echo "exited $(date +%FT%T)" > "$PK/raw/RI.running"' EXIT
for m in RI.done RI.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt"; }
for f in code/v8f_t1_eval_ri.py code/apply_ri_patch.py code/ri_receipts.py tests/test_v8f_t1_ri.py; do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
TOOL_SHA=$(pin code/v8f_t1_eval_ri.py)
got=$(sha256sum "$TOOL" | cut -c1-64)
[ "$got" = "$TOOL_SHA" ] || fail "scratch tree tool sha $got != $TOOL_SHA" 2
last=$(grep -v '^[[:space:]]*$' "$PK/logs/pytest_ri.log" | tail -n 1)
case "$last" in *"12 passed"*) case "$last" in *failed*|*error*) fail "unit tests not green: $last" 2;; esac;; *) fail "unit tests not green: $last" 2;; esac
grep -q "^MUTATIONS ALL RED" "$PK/logs/mutations.log" || fail "the mutation log does not carry MUTATIONS ALL RED" 2
grep -q "^SMOKERI OK" "$PK/logs/smoke_ri.out" || fail "the smoke log does not carry SMOKERI OK" 2
for r in M2b_s0_recft_C0 M2b_s1_recft_C0; do
  [ -f "$RDB/$r/ckpt_last.pt" ] || fail "checkpoint $r is absent from $RDB" 5
  got=$(sha256sum "$RDB/$r/ckpt_last.pt" | cut -c1-64)
  sums=$(awk '$2 ~ /ckpt_last.pt$/ {print $1}' "$RDB/$r/SHA256SUMS.txt" | head -n 1)
  [ -n "$sums" ] && [ "$got" = "$sums" ] || fail "$r ckpt sha256 $got != SHA256SUMS.txt $sums (a partial copy?)" 6
done
[ -f "$ART/sfresh_inputs/inputs/native_tracks.pt" ] || fail "input native_tracks.pt absent" 5
[ -d "$ART/sfresh_inputs/cache/eval/clips" ] || fail "eval cache absent" 5
for f in eval_grid_windows.json join_eval141.npz; do [ -f "$DATA/$f" ] || fail "input $f absent from the data dir" 5; done
log "ZZRI-START tool $TOOL_SHA unit tests: $last; mutations all red; smoke ok; both C0 checkpoints match SHA256SUMS.txt; threads=$TH"

run_one() {   # tag run extra-flags...
  local tag=$1 run=$2; shift 2
  local ck="$ART/v8f-recover/$run/ckpt_last.pt"
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZRI-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZRI-RUN $tag flags=$*"
  "$PY" -I "$TOOLW" run --ckpt "$ck" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUTW/$tag" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal \
     --dump-ticks "$OUTW/$tag/ticks" --dump-heads --device cpu "$@" < /dev/null > "$PK/logs/run_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$ck" \
       "$(sha256sum "$RDB/$run/ckpt_last.pt" | cut -c1-64)" "$TOOLW" "$TOOL_SHA" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZRI-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZRI-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

nfail=0
run_one RIC0_s0  M2b_s0_recft_C0 --smooth sg72 --scene reindexed || nfail=$((nfail+1))
run_one RIC0_s1  M2b_s1_recft_C0 --smooth sg72 --scene reindexed || nfail=$((nfail+1))
run_one RIC0C_s0 M2b_s0_recft_C0 --smooth sg72 --scene reindexed --feasible kbox || nfail=$((nfail+1))
run_one RIC0C_s1 M2b_s1_recft_C0 --smooth sg72 --scene reindexed --feasible kbox || nfail=$((nfail+1))
log "ZZRI-ROLLOUTS-DONE nfail=$nfail"
missing=""
for t in RIC0_s0 RIC0_s1 RIC0C_s0 RIC0C_s1; do [ -f "$OUT/done_$t.txt" ] || missing="$missing $t"; done
[ -z "$missing" ] || fail "missing done markers:$missing (see logs/run_<tag>.out)" 3
"$PY" -I "$PKW/code/ri_receipts.py" --out-dir "$OUTW" --raw "$PKW/raw" > "$PK/logs/receipts.out" 2>&1; rc=$?
log "ZZRI-RECEIPTS rc=$rc"
cat "$PK/logs/receipts.out" | tee -a "$PK/logs/chain.out"
[ $rc -eq 0 ] || fail "receipt gate rc=$rc (see logs/receipts.out and raw/RI_RECEIPTS.json)" "$rc"
echo "$(date +%FT%T) RI_RECEIPTS.json written; four rollouts done" > "$PK/raw/RI.done"
log "ZZRI-CHAIN-DONE"
