#!/usr/bin/env bash
# RECOVER-FT T1-kin READ chain (H-RECOVER-FT-1 + H-RECOVER-COMPOSITE-1).  Dev-box CPU, ZERO training, 16 threads, ONE rollout at a time.  The Master Mind starts it AFTER pull_recft.sh finished.
#   usage:  recover_read_chain.sh --dry-run     lists the 15 rollouts, the gates and what is present; waits for nothing, runs nothing.
#           recover_read_chain.sh               (normal; start through recover_read_chain.cmd = WMI Win32_Process.Create so it survives the caller)
# Gate 0  the composite card hashes to FROZEN_SHA.txt, the RECOVER-FT card to its own FROZEN_SHA.txt; the tool in the scratch tree, the reader and the unit-test log are the pinned ones (PINS.txt).
# Gate 1  WAIT (in here, never in a model turn; at most 6 h) for the six pulled runs in E:/Projects/TanitAD-artifacts/v8f-recover/: ckpt_last.pt, summary.json, argv.json, SHA256SUMS.txt (pull_recft.sh writes it LAST per run,
#         so its presence means the copy finished) and recft_markers/CHAIN_RECOVER.done; each ckpt's sha256 must equal SHA256SUMS.txt and the box-2 RUN_<tag>.done marker's evidence.ckpt_sha256 when the marker carries one.
# Gate 2  no other tool rollout / extractor is running (CPU shared), at most 3 h.   Gate 3  a 6-window PRE-FLIGHT (a checkpoint of each seed loads in the dev-box eval tree and the flags are accepted).
# Rollouts (list = code/recover_read.py --list-runs, the single source): R_s0 C0_s0 KA1R_s0 R_s1 C0_s1 C0REP_s0 C0REP_s1 FZR_s0 FZC0_s0 FZR_s1 FZC0_s1 RC_s0 C0C_s0 RC_s1 C0C_s1, all --smooth sg72;
#         FZ* = --scene hold_now; RC / C0C = --feasible kbox; KA1R_s0 = a bit-identity rerun of R_s0.  Each writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch again);
#         an incomplete out dir is MOVED to _incomplete_*, never deleted.  Then the read (code/recover_read.py --mode production): controls first, rows after.
# Markers: ZZRREAD-* in logs/chain.out; raw/RREAD.done on success (raw/RECOVER_READ.json written), raw/RREAD.fail (reason inside) otherwise; a REFUSED control is a result to report, not to patch around.
set -u
PK=/e/Projects/TanitAD-artifacts/dagland/recover_read
S=/f/TanitAD-scratch/recover_read
OUT=$S/out
TREE=$S/tree_dag
TOOL=$TREE/stack/scripts/v8f_t1_eval_framefreeze.py
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
RD=$ART/v8f-recover
RDB=/e/Projects/TanitAD-artifacts/v8f-recover
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
FT=/e/Projects/TanitAD-artifacts/dagland/recover_ft
FZ=/e/Projects/TanitAD-artifacts/dagland/framefreeze
TH=16
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT" "$PK/logs" "$PK/raw" "$S/smoke"
log() { echo "$(date +%FT%T) $*" | tee -a "$PK/logs/chain.out"; }
fail() { log "ZZRREAD-FAIL $*"; echo "$(date +%FT%T) $*" > "$PK/raw/RREAD.fail"; exit "${2:-3}"; }
table() { "$PY" -I "$PK/code/recover_read.py" --list-runs | tr -d '\r'; }
pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt" | head -n 1; }
RUNS="M2b_s0_recft_R M2b_s0_recft_C0 M2b_s0_recft_C0rep M2b_s1_recft_R M2b_s1_recft_C0 M2b_s1_recft_C0rep"
TAGS="R_0:M2b_s0_recft_R C0_0:M2b_s0_recft_C0 R_1:M2b_s1_recft_R C0_1:M2b_s1_recft_C0 C0rep_0:M2b_s0_recft_C0rep C0rep_1:M2b_s1_recft_C0rep"

if [ $DRY -eq 1 ]; then
  echo "RECOVER-READ chain DRY RUN (no wait, no rollout)"
  echo "scratch $S   out $OUT   pulled runs $RDB   tool $TOOL   threads $TH"
  n=0; while IFS=$'\t' read -r tag seed ckrun scene feas flags; do
    n=$((n+1)); printf '%2d  %-9s seed %s  ckpt %-20s %s\n' "$n" "$tag" "$seed" "$ckrun" "$flags"
  done < <(table)
  echo "estimated wall (ESTIMATED from the earlier chains' done_*.txt, 16 threads, idle box): 11 logged-frame rollouts x ~160 s (R/C0/C0rep x2, KA1R, RC/C0C x4: FEAS2 KBOX 153-154 s, HISTSWAP-LF 162 s) = ~29 min;"
  echo "  4 hold_now rollouts x 340-470 s (FRAMEFREEZE FZY/FZH incl. the receipt check) = ~27 min; pre-flight ~1.2 min; read: 15 extractions (~90 s each, 2 workers) ~11 min + rows ~5 min   => about 75 min (+-15)"
  for m in "composite card:$PK/PREREG_COMPOSITE_FROZEN.md" "reader:$PK/code/recover_read.py" "tool in scratch tree:$TOOL"; do
    f=${m#*:}; if [ -f "$f" ]; then echo "present  ${m%%:*}  $(sha256sum "$f" | cut -c1-16)"; else echo "ABSENT   ${m%%:*}  $f"; fi
  done
  for t in $TAGS; do r=${t#*:}; if [ -f "$RDB/$r/SHA256SUMS.txt" ] && [ -f "$RDB/$r/ckpt_last.pt" ]; then echo "present  pulled $r"; else echo "ABSENT   pulled $r (the chain waits for it, 6 h max)"; fi; done
  [ -f "$RDB/recft_markers/CHAIN_RECOVER.done" ] && echo "present  CHAIN_RECOVER.done" || echo "ABSENT   recft_markers/CHAIN_RECOVER.done"
  echo "after the read: raw/RREAD.done and raw/RECOVER_READ.json (success) or raw/RREAD.fail"
  exit 0
fi

for m in RREAD.done RREAD.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done
log "ZZRREAD-START threads=$TH"
# ---- gate 0
first=$(head -n 1 "$PK/FROZEN_SHA.txt" | tr -d '\r'); cf=$(echo "$first" | awk '{print $1}'); cs=$(echo "$first" | awk '{print $3}')
[ "$(sha256sum "$PK/$cf" | cut -c1-64)" = "$cs" ] || fail "composite card $cf does not hash to FROZEN_SHA.txt" 2
CARD_LINE=$(head -n 1 "$FT/FROZEN_SHA.txt" | tr -d '\r'); ff=$(echo "$CARD_LINE" | awk '{print $1}'); fs=$(echo "$CARD_LINE" | awk '{print $3}')
[ "$(sha256sum "$FT/$ff" | cut -c1-64)" = "$fs" ] || fail "RECOVER-FT card $ff does not hash to its FROZEN_SHA.txt" 2
for f in code/recover_read.py tests/test_recover_read.py; do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64); [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
TOOL_SHA=$(sed -n 's|^code/v8f_t1_eval_framefreeze.py sha256 \([0-9a-f]*\)$|\1|p' "$FZ/PINS.txt" | head -n 1)
[ -n "$TOOL_SHA" ] && [ "$(sha256sum "$TOOL" | cut -c1-64)" = "$TOOL_SHA" ] || fail "scratch tree tool is not the pinned FRAMEFREEZE tool ($TOOL_SHA)" 2
last=$(grep -v '^[[:space:]]*$' "$PK/logs/pytest_recover_read.log" | tail -n 1)
case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green: $last" 2;; esac;; *) fail "unit tests not green: $last" 2;; esac
log "ZZRREAD-GATE0 cards ok $cs / $fs, tool $TOOL_SHA, reader and tests pinned, unit tests: $last"
# ---- gate 1: the pulled runs
waited=0
ready() { for t in $TAGS; do r=${t#*:}; for f in ckpt_last.pt summary.json argv.json SHA256SUMS.txt gate_selfcheck.json; do [ -f "$RDB/$r/$f" ] || return 1; done; done; [ -f "$RDB/recft_markers/CHAIN_RECOVER.done" ]; }
until ready; do
  if [ -f "$RDB/recft_markers/CHAIN_RECOVER.STOPPED" ] || ls "$RDB/recft_markers"/RUN_*.FAILED > /dev/null 2>&1; then fail "the box-2 chain did not complete (CHAIN_RECOVER.STOPPED or a RUN_*.FAILED marker is pulled)" 5; fi
  sleep 30; waited=$((waited+30)); [ $waited -ge 21600 ] && fail "the six pulled runs are not complete after 6 h" 4
done
log "ZZRREAD-GATE1 pulled runs present after ${waited}s"
for t in $TAGS; do
  tg=${t%%:*}; r=${t#*:}; got=$(sha256sum "$RDB/$r/ckpt_last.pt" | cut -c1-64)
  sums=$(awk '$2 ~ /ckpt_last.pt$/ {print $1}' "$RDB/$r/SHA256SUMS.txt" | head -n 1)
  [ "$got" = "$sums" ] || fail "$r ckpt sha256 $got != SHA256SUMS.txt $sums (a partial copy?)" 6
  mk=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('evidence',{}).get('ckpt_sha256') or '')" "$ART/v8f-recover/recft_markers/RUN_$tg.done" 2>/dev/null | tr -d '\r')
  if [ -z "$mk" ]; then log "ZZRREAD-MARKER $tg UNVERIFIED (RUN_$tg.done carries no ckpt sha)"; elif [ "$mk" != "$got" ]; then fail "$r ckpt sha256 $got != box-2 marker $mk" 6; else log "ZZRREAD-MARKER $tg ok ${got:0:16}"; fi
done
# ---- gate 2: CPU
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZRREAD-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZRREAD-CPU-WAIT ${w}s"

run_tool() {   # tag ckpt outdir extra-flags... (stdin closed so a tool never eats the table)
  local tag=$1 ckpt=$2 od=$3; shift 3
  "$PY" -I "$TOOL" run --ckpt "$ckpt" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$od" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks "$od/ticks" --dump-heads --device cpu "$@" < /dev/null
}
# ---- gate 3: pre-flight (6 windows): R_s0 with kbox, C0_s1 with hold_now; each must exit 0 and write its ticks file
pre() {   # tag run flags...
  local tag=$1 run=$2; shift 2; rm -rf "$S/smoke/$tag"; mkdir -p "$S/smoke/$tag"
  run_tool "$tag" "$RD/$run/ckpt_last.pt" "$S/smoke/$tag" --max-windows 6 "$@" > "$PK/logs/smoke_$tag.out" 2>&1; local rc=$?
  [ $rc -eq 0 ] && [ -f "$S/smoke/$tag/ticks/ticks_$tag.npz" ] || fail "pre-flight $tag rc=$rc (a checkpoint does not load in the dev-box eval tree, or a flag is refused): see logs/smoke_$tag.out" 7
  log "ZZRREAD-PREFLIGHT-OK $tag"
}
pre PRE_R_s0_kbox M2b_s0_recft_R --smooth sg72 --scene logged --feasible kbox
pre PRE_C0_s1_hold M2b_s1_recft_C0 --smooth sg72 --scene hold_now --freeze-card-ref "$CARD_LINE"

run_one() {   # tag ckrun flags...
  local tag=$1 ck=$2; shift 2
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZRREAD-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZRREAD-RUN $tag flags=$*"
  run_tool "$tag" "$RD/$ck/ckpt_last.pt" "$OUT/$tag" "$@" > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tool_sha256": "%s", "card_line": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$RD/$ck/ckpt_last.pt" \
       "$(sha256sum "$RDB/$ck/ckpt_last.pt" | cut -c1-64)" "$TOOL_SHA" "$CARD_LINE" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZRREAD-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZRREAD-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed ckrun scene feas flags <&3; do
  extra=(); [ "$scene" = "hold_now" ] && extra=(--freeze-card-ref "$CARD_LINE")
  # shellcheck disable=SC2086
  run_one "$tag" "$ckrun" $flags "${extra[@]}" || nfail=$((nfail+1))
done 3< <(table)
log "ZZRREAD-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing" 3
log "ZZRREAD-READ start"
"$PY" -I "$PK/code/recover_read.py" --mode production --out-dir "$OUT" --raw "$PK/raw" --recover-dir "$RD" > "$PK/logs/read.out" 2>&1; rc=$?
log "ZZRREAD-READ rc=$rc"
[ $rc -eq 0 ] || fail "read rc=$rc (a control may have REFUSED the read: see logs/read.out and raw/RECOVER_READ.json 'refused')" "$rc"
[ -f "$PK/raw/RECOVER_READ.json" ] || fail "the read returned 0 but raw/RECOVER_READ.json is absent" 3
echo "$(date +%FT%T) RECOVER_READ.json written" > "$PK/raw/RREAD.done"
log "ZZRREAD-CHAIN-DONE"
