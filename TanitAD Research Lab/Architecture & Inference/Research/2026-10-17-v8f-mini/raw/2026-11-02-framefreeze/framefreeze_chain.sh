#!/usr/bin/env bash
# FRAMEFREEZE control chain (RECOVER-FT card section 7): the base-model control rollouts of --scene hold_now.  Dev-box CPU, ZERO training, 16 threads, ONE run at a time.
#   FZY_DAGOACC_s{0,1}   --smooth sg72 --scene hold_now                           (FZ-Y: the freeze's own effect on the init; Y = KAPPA-T1's SREP_*, reused)
#   FZH_DAGOACC_s{0,1}   --smooth sg72 --histswap cv_logged --scene hold_now      (FZ-HLVL: the known-answer positive control)
# then code/framefreeze_read.py -> raw/FZ_CONTROLS.json (KFZ1..KFZ4 + the freeze's own effect on all 19 rows).  A failed KFZ3 / KFZ4 is a FINDING written into the JSON (the RECOVER-FT read REFUSES the frame-freeze
# contrast); raw/FZ.done is written whenever the JSON was written; raw/FZ.fail (reason inside) only when a gate, a rollout or the read itself failed.
# Gate 1: wait (poll in here, never in a model turn) for dagland/recover_ft/FROZEN_SHA.txt, at most 12 h; its first line must be '<card> sha256 <hash>' and the card must hash to it.  Gate 2: PINS.txt (tool, patch,
# reader) and the unit-test log (7 passed) must hold; the tool must be the one in the scratch tree.  Gate 3: no other tool rollout / extractor is running (CPU shared), at most 3 h.
# Each run writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch again); an incomplete out dir is MOVED to _incomplete_*.  Priority order: FZY_s0, FZH_s0, FZY_s1, FZH_s1.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/framefreeze
S=/f/TanitAD-scratch/framefreeze
OUT=$S/out
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
CARD=/e/Projects/TanitAD-artifacts/dagland/recover_ft
TH=16
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs" "$PK/raw"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZFZ-FAIL $*"; echo "$(date +%FT%T) $*" > "$PK/raw/FZ.fail"; exit "${2:-3}"; }
for m in FZ.done FZ.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done

log "ZZFZ-WAIT for $CARD/FROZEN_SHA.txt (the RECOVER-FT card freezes the frame-freeze reading thresholds)"
waited=0
until [ -f "$CARD/FROZEN_SHA.txt" ]; do
  sleep 20; waited=$((waited+20))
  [ $waited -ge 43200 ] && fail "no RECOVER-FT FROZEN_SHA.txt after 12 h" 4
done
log "ZZFZ-GATE-OPEN after ${waited}s"
CARD_LINE=$(head -n 1 "$CARD/FROZEN_SHA.txt" | tr -d '\r')
card_file=$(echo "$CARD_LINE" | awk 'NF==3 && $2=="sha256" {print $1}'); card_sha=$(echo "$CARD_LINE" | awk 'NF==3 && $2=="sha256" {print $3}')
[ -n "$card_file" ] && [ -n "$card_sha" ] || fail "first line of the card's FROZEN_SHA.txt is not '<file> sha256 <hash>': $CARD_LINE" 2
have=$(sha256sum "$CARD/$card_file" | cut -c1-64)
[ "$have" = "$card_sha" ] || fail "$card_file hashes to $have, not the frozen $card_sha" 2

pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt"; }
for f in code/v8f_t1_eval_framefreeze.py code/apply_fz_patch.py code/framefreeze_read.py; do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64)
  [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
TOOL_SHA=$(pin code/v8f_t1_eval_framefreeze.py)
got=$(sha256sum "$S/tree_dag/stack/scripts/v8f_t1_eval_framefreeze.py" | cut -c1-64)
[ "$got" = "$TOOL_SHA" ] || fail "scratch tree tool sha $got != $TOOL_SHA" 2
last=$(grep -v '^[[:space:]]*$' "$PK/logs/pytest_framefreeze.log" | tail -n 1)
case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green: $last" 2;; esac;; *) fail "unit tests not green: $last" 2;; esac
for t in DAGOACC_s0 DAGOACC_s1; do [ -f "/f/TanitAD-scratch/kappa_t1/out/done_SREP_$t.txt" ] || fail "reused Y missing: kappa_t1 done_SREP_$t.txt" 2; done
log "ZZFZ-START card ok $card_file $card_sha tool $TOOL_SHA unit tests: $last threads=$TH"

busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0
while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do
  sleep 30; w=$((w+30))
  [ $w -ge 10800 ] && { log "ZZFZ-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding (CPU contention only slows, it does not change the result)"; break; }
done
[ $w -gt 0 ] && log "ZZFZ-CPU-WAIT ${w}s for another rollout or extractor to finish"

run_one() {   # tag ckpt extra-flags...
  local tag=$1 ckpt=$2; shift 2
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZFZ-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZFZ-RUN $tag flags=$*"
  "$PY" -I "$S/tree_dag/stack/scripts/v8f_t1_eval_framefreeze.py" run --ckpt "$ckpt" --tag "$tag" \
     --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$OUT/$tag" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal \
     --dump-ticks "$OUT/$tag/ticks" --dump-heads --device cpu --freeze-card-ref "$CARD_LINE" "$@" > "$PK/logs/run_$tag.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tool": "%s", "tool_sha256": "%s", "card_line": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$ckpt" \
       "$(sha256sum "$ckpt" | cut -c1-64)" "$S/tree_dag/stack/scripts/v8f_t1_eval_framefreeze.py" "$TOOL_SHA" "$CARD_LINE" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZFZ-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZFZ-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}

CK0=$ART/v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt; CK1=$ART/v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt
nfail=0
run_one FZY_DAGOACC_s0 "$CK0" --smooth sg72 --scene hold_now || nfail=$((nfail+1))
run_one FZH_DAGOACC_s0 "$CK0" --smooth sg72 --histswap cv_logged --scene hold_now || nfail=$((nfail+1))
run_one FZY_DAGOACC_s1 "$CK1" --smooth sg72 --scene hold_now || nfail=$((nfail+1))
run_one FZH_DAGOACC_s1 "$CK1" --smooth sg72 --histswap cv_logged --scene hold_now || nfail=$((nfail+1))
log "ZZFZ-ROLLOUTS-DONE nfail=$nfail"
missing=""
for t in FZY_DAGOACC_s0 FZH_DAGOACC_s0 FZY_DAGOACC_s1 FZH_DAGOACC_s1; do [ -f "$OUT/done_$t.txt" ] || missing="$missing $t"; done
[ -z "$missing" ] || fail "missing done markers:$missing" 3
log "ZZFZ-READ start"
"$PY" -I "$PK/code/framefreeze_read.py" --out-dir "$OUT" --raw "$PK/raw" > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZFZ-READ rc=$rc"
[ $rc -eq 0 ] || fail "read rc=$rc (see logs/read.out)" "$rc"
[ -f "$PK/raw/FZ_CONTROLS.json" ] || fail "the read returned 0 but raw/FZ_CONTROLS.json is absent" 3
echo "$(date +%FT%T) FZ_CONTROLS.json written" > "$PK/raw/FZ.done"
log "ZZFZ-CHAIN-DONE"
