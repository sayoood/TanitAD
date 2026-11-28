#!/usr/bin/env bash
# RECOVER-NEXT dev-box READ chain for ONE arm (H-RECOVER-NEXT card section 3).  Dev-box CPU, ZERO training, 16 threads, ONE rollout at a time.  The Master Mind starts it AFTER pull_rnext.sh finished.
#   usage:  rnext_read_chain.sh <ARM> --dry-run     ARM in RAPR | RTL | RP01 | RAPRP01 | RPT (or env RNEXT_ARM): lists the 7 rollouts to roll, the 8 reused, the gates, what is present and the estimated wall; waits for nothing, runs nothing.
#           rnext_read_chain.sh <ARM>               (normal; start through rnext_read_chain_<ARM>.cmd = WMI Win32_Process.Create so it survives the caller)
# Gate 0  the three cards hash to their FROZEN_SHA.txt (composite, RECOVER-FT, RECOVER-NEXT); the tool in the scratch tree is the pinned FRAMEFREEZE tool; the reader, combiner and the three test files are the pinned ones
#         (PINS.txt); the three unit-test logs are green; raw/KA_RARM_R.json says PASS and was produced by the pinned reader (the --r-arm R known answer).
# Gate 1  the 8 REUSED rollouts' done markers exist (C:/Users/Admin/rr_scratch/out, read only); WAIT (in here, never in a model turn; at most 6 h) for the arm's two pulled runs in E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{0,1}_recft_<ARM>/:
#         ckpt_last.pt summary.json argv.json SHA256SUMS.txt (pull_rnext.sh writes it LAST per run) gate_selfcheck.json, and rnext_markers/RUN_<ARM>_{0,1}.done (a pulled RUN_<ARM>_*.FAILED stops the chain); each ckpt's sha256 must equal
#         SHA256SUMS.txt AND the box-2 marker's evidence.ckpt_sha256 (a marker without one is refused: pull_rnext.sh already treats that as a mismatch).
# Gate 2  no other tool rollout / extractor is running (CPU shared; another arm's chain waits here), at most 3 h.   Gate 3  a 6-window PRE-FLIGHT (the arm's s0 checkpoint with kbox, its s1 checkpoint with hold_now).
# Rollouts (list = code/recover_read.py --r-arm <ARM> --list-runs, the single source): <ARM>_s0 KA1<ARM>_s0 <ARM>_s1 FZ<ARM>_s0 FZ<ARM>_s1 <ARM>C_s0 <ARM>C_s1, all --smooth sg72; FZ* = --scene hold_now; <ARM>C = --feasible kbox;
#         KA1<ARM>_s0 = a bit-identity rerun of <ARM>_s0.  Each writes done_<tag>.txt only on rc 0; a finished run is skipped on re-launch (RESUME = launch again); an incomplete out dir is MOVED to _incomplete_*, never deleted.
#         Then the read (code/recover_read.py --mode production --r-arm <ARM>): controls first, rows after -> raw/RECOVER_READ_<ARM>.json; then the combiner (code/recover_next_combine.py) -> raw/RECOVER_NEXT.json from every per-arm JSON that exists.
# Markers: ZZRNEXT-* in logs/chain_<ARM>.out; raw/RNEXT_<ARM>.done on success, raw/RNEXT_<ARM>.fail (reason inside) otherwise; a REFUSED control is a result to report, not to patch around.
# WRITES: only $PK (this package) and $W (default C:/Users/Admin/rr_scratch/rnext, a new directory).  Nothing on F: (full) and nothing in the existing rr_scratch/out.
set -u
ARM=${RNEXT_ARM:-}
for t in "$@"; do case "$t" in RAPR|RTL|RP01|RAPRP01|RPT) ARM=$t;; esac; done
case "$ARM" in RAPR|RTL|RP01|RAPRP01|RPT) ;; *) echo "usage: rnext_read_chain.sh <RAPR|RTL|RP01|RAPRP01|RPT> [--dry-run]"; exit 64;; esac
DRY=0; for t in "$@"; do [ "$t" = "--dry-run" ] && DRY=1; done
PK=/e/Projects/TanitAD-artifacts/dagland/recover_next/read
NX=/e/Projects/TanitAD-artifacts/dagland/recover_next
RRP=/e/Projects/TanitAD-artifacts/dagland/recover_read
S=/f/TanitAD-scratch/recover_read                      # the eval tree: READ ONLY (F: is full)
TREE=$S/tree_dag
TOOL=$TREE/stack/scripts/v8f_t1_eval_framefreeze.py
W=${RNEXT_W:-/c/Users/Admin/rr_scratch/rnext}          # write root (a new directory on C:)
OUT=$W/out_$ARM
REUSE=/c/Users/Admin/rr_scratch/out                    # the finished C0 / C0rep / FZ-C0 / C0C rollouts: READ ONLY
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
ART=E:/Projects/TanitAD-artifacts
RD=$ART/v8f-recover
RDB=/e/Projects/TanitAD-artifacts/v8f-recover
DATA="$ART/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
FT=/e/Projects/TanitAD-artifacts/dagland/recover_ft
FZ=/e/Projects/TanitAD-artifacts/dagland/framefreeze
TH=16
LOGF=$PK/logs/chain_$ARM.out
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH PYTHONDONTWRITEBYTECODE=1
log() { echo "$(date +%FT%T) $*" | tee -a "$LOGF"; }
fail() { log "ZZRNEXT-FAIL $*"; echo "$(date +%FT%T) $*" > "$PK/raw/RNEXT_$ARM.fail"; exit "${2:-3}"; }
table() { "$PY" -I "$PK/code/recover_read.py" --r-arm "$ARM" --list-runs | tr -d '\r'; }
pin() { sed -n "s|^$1 sha256 \([0-9a-f]*\)\$|\1|p" "$PK/PINS.txt" | head -n 1; }
TAGS="${ARM}_0:M2b_s0_recft_${ARM} ${ARM}_1:M2b_s1_recft_${ARM}"
REUSED_TAGS="C0_s0 C0_s1 C0REP_s0 C0REP_s1 FZC0_s0 FZC0_s1 C0C_s0 C0C_s1"
COMPARATORS="M2b_s0_recft_R M2b_s0_recft_C0 M2b_s0_recft_C0rep M2b_s1_recft_R M2b_s1_recft_C0 M2b_s1_recft_C0rep"

if [ $DRY -eq 1 ]; then
  echo "RECOVER-NEXT read chain DRY RUN, arm $ARM (no wait, no rollout, nothing written)"
  echo "package $PK   write root $W   out $OUT   reused rollouts (read only) $REUSE   pulled runs $RDB   tool $TOOL   threads $TH"
  echo "rollouts to ROLL (7):"
  n=0; while IFS=$'\t' read -r tag seed ckrun scene feas flags; do
    n=$((n+1)); st="not yet rolled"; [ -f "$OUT/done_$tag.txt" ] && st="done (skipped on launch)"
    printf '  %d  %-12s seed %s  ckpt %-22s %-48s %s\n' "$n" "$tag" "$seed" "$ckrun" "$flags" "$st"
  done < <(table)
  echo "rollouts REUSED, not re-rolled (8; done markers in $REUSE):"
  for t in $REUSED_TAGS; do if [ -f "$REUSE/done_$t.txt" ]; then echo "  present  $t"; else echo "  ABSENT   $t  (the chain refuses to start: gate 1)"; fi; done
  echo "inputs:"
  for m in "composite card:$RRP/PREREG_COMPOSITE_FROZEN.md" "RECOVER-NEXT card:$NX/PREREG_RECOVER_NEXT_FROZEN.md" "reader:$PK/code/recover_read.py" "combiner:$PK/code/recover_next_combine.py" "tool in scratch tree:$TOOL" \
           "known answer raw/KA_RARM_R.json:$PK/raw/KA_RARM_R.json" "R read (for the combiner):$RRP/raw/RECOVER_READ.json"; do
    f=${m#*:}; if [ -f "$f" ]; then echo "  present  ${m%%:*}  $(sha256sum "$f" | cut -c1-16)"; else echo "  ABSENT   ${m%%:*}  $f"; fi
  done
  for t in $TAGS; do r=${t#*:}; tg=${t%%:*}
    if [ -f "$RDB/$r/SHA256SUMS.txt" ] && [ -f "$RDB/$r/ckpt_last.pt" ] && [ -f "$RDB/$r/gate_selfcheck.json" ] && [ -f "$RDB/rnext_markers/RUN_$tg.done" ]; then echo "  present  pulled $r + marker RUN_$tg.done"; else echo "  ABSENT   pulled $r (the chain waits for it, 6 h max; pull_rnext.sh writes it)"; fi; done
  for r in $COMPARATORS; do [ -f "$RDB/$r/ckpt_last.pt" ] && echo "  present  comparator $r" || echo "  ABSENT   comparator $r"; done
  echo "estimated wall for this arm (idle box, 16 threads): rollouts ~28 min = MEASURED walls of the same seven rollout kinds on the RECOVER-FT chain (done_*.txt: logged 234 / 219 / 241 s, hold_now 255 / 417 s, kbox 156 / 158 s);"
  echo "  gates + sha checks ~3 min, pre-flight ~1.5 min, 7 extractions (~90 s each, 2 workers) ~5.5 min, file controls + rows ~7 min (ESTIMATED), combiner seconds  =>  about 45 min (+-12) per arm; three arms ~2.3 h if run back to back"
  echo "after the read: raw/RECOVER_READ_$ARM.json, raw/RECOVER_NEXT.json and raw/RNEXT_$ARM.done (success) or raw/RNEXT_$ARM.fail"
  exit 0
fi

mkdir -p "$OUT" "$PK/logs" "$PK/raw" "$W/smoke"
for m in RNEXT_$ARM.done RNEXT_$ARM.fail; do [ -f "$PK/raw/$m" ] && mv "$PK/raw/$m" "$PK/raw/_old_${m}_$(date +%H%M%S)"; done
log "ZZRNEXT-START arm=$ARM threads=$TH"
# ---- gate 0
for pair in "$RRP:PREREG_COMPOSITE_FROZEN.md" "$FT:PREREG_RECOVER_FT_FROZEN.md" "$NX:PREREG_RECOVER_NEXT_FROZEN.md"; do
  d=${pair%%:*}; line=$(head -n 1 "$d/FROZEN_SHA.txt" | tr -d '\r'); cf=$(echo "$line" | awk '{print $1}'); cs=$(echo "$line" | awk '{print $3}')
  [ "$(sha256sum "$d/$cf" | cut -c1-64)" = "$cs" ] || fail "card $d/$cf does not hash to its FROZEN_SHA.txt" 2
done
CARD_LINE=$(head -n 1 "$FT/FROZEN_SHA.txt" | tr -d '\r')
for f in code/recover_read.py code/recover_next_combine.py code/ka_rarm_r.py tests/test_recover_read.py tests/test_recover_read_arms.py tests/test_recover_next_combine.py; do
  want=$(pin "$f"); got=$(sha256sum "$PK/$f" | cut -c1-64); [ -n "$want" ] && [ "$want" = "$got" ] || fail "$f sha $got != PINS.txt ($want)" 2
done
TOOL_SHA=$(sed -n 's|^code/v8f_t1_eval_framefreeze.py sha256 \([0-9a-f]*\)$|\1|p' "$FZ/PINS.txt" | head -n 1)
[ -n "$TOOL_SHA" ] && [ "$(sha256sum "$TOOL" | cut -c1-64)" = "$TOOL_SHA" ] || fail "scratch tree tool is not the pinned FRAMEFREEZE tool ($TOOL_SHA)" 2
for lg in pytest_recover_read pytest_recover_read_arms pytest_recover_next_combine; do
  last=$(grep -v '^[[:space:]]*$' "$PK/logs/$lg.log" | tail -n 1)
  case "$last" in *" passed"*) case "$last" in *failed*|*error*) fail "unit tests not green ($lg): $last" 2;; esac;; *) fail "unit tests not green ($lg): $last" 2;; esac
done
RSHA=$(pin code/recover_read.py)
"$PY" -I -c "
import json,sys
ka=json.load(open(sys.argv[1]+'/raw/KA_RARM_R.json')); rd=json.load(open(sys.argv[1]+'/raw/RECOVER_READ_R.json'))
sys.exit(0 if ka['PASS'] is True and ka['n_differences']==0 and rd['reader_sha256']==sys.argv[2] else 1)" "$PK" "$RSHA" || fail "raw/KA_RARM_R.json is not a PASS of the pinned reader (the --r-arm R known answer)" 2
log "ZZRNEXT-GATE0 cards ok, tool $TOOL_SHA, reader/combiner/tests pinned, unit tests green, known answer PASS (reader ${RSHA:0:12})"
# ---- gate 1: the reused rollouts and the pulled runs
for t in $REUSED_TAGS; do [ -f "$REUSE/done_$t.txt" ] || fail "reused rollout $t has no done marker in $REUSE" 5; done
for r in $COMPARATORS; do [ -f "$RDB/$r/ckpt_last.pt" ] || fail "comparator checkpoint $r is absent from $RDB" 5; done
waited=0
ready() { for t in $TAGS; do r=${t#*:}; tg=${t%%:*}; for f in ckpt_last.pt summary.json argv.json SHA256SUMS.txt gate_selfcheck.json; do [ -f "$RDB/$r/$f" ] || return 1; done; [ -f "$RDB/rnext_markers/RUN_$tg.done" ] || return 1; done; return 0; }
until ready; do
  if ls "$RDB/rnext_markers"/RUN_${ARM}_*.FAILED > /dev/null 2>&1; then fail "a RUN_${ARM}_*.FAILED marker is pulled: the box-2 run did not finish" 5; fi
  sleep 30; waited=$((waited+30)); [ $waited -ge 21600 ] && fail "the two pulled $ARM runs are not complete after 6 h" 4
done
log "ZZRNEXT-GATE1 pulled runs present after ${waited}s"
for t in $TAGS; do
  tg=${t%%:*}; r=${t#*:}; got=$(sha256sum "$RDB/$r/ckpt_last.pt" | cut -c1-64)
  sums=$(awk '$2 ~ /ckpt_last.pt$/ {print $1}' "$RDB/$r/SHA256SUMS.txt" | head -n 1)
  [ "$got" = "$sums" ] || fail "$r ckpt sha256 $got != SHA256SUMS.txt $sums (a partial copy?)" 6
  mk=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1])).get('evidence',{}).get('ckpt_sha256') or '')" "$RDB/rnext_markers/RUN_$tg.done" 2>/dev/null | tr -d '\r')
  [ -n "$mk" ] || fail "RUN_$tg.done carries no evidence.ckpt_sha256" 6
  [ "$mk" = "$got" ] || fail "$r ckpt sha256 $got != box-2 marker $mk" 6
  log "ZZRNEXT-MARKER $tg ok ${got:0:16}"
done
# ---- gate 2: CPU
busy() { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match 'v8f_t1_eval_[a-z_0-9]+[.]py\S*\s+run|extract_smooth[.]py' -and \$_.ProcessId -ne \$PID }).Count" | tr -d '\r '; }
w=0; while [ "$(busy)" != "0" ] && [ -n "$(busy)" ]; do sleep 30; w=$((w+30)); [ $w -ge 10800 ] && { log "ZZRNEXT-BUSY-TIMEOUT another rollout or extractor still runs after 3 h; proceeding"; break; }; done
[ $w -gt 0 ] && log "ZZRNEXT-CPU-WAIT ${w}s"

run_tool() {   # tag ckpt outdir extra-flags... (stdin closed so a tool never eats the table)
  local tag=$1 ckpt=$2 od=$3; shift 3
  "$PY" -I "$TOOL" run --ckpt "$ckpt" --tag "$tag" --tracks "$ART/sfresh_inputs/inputs/native_tracks.pt" --eval-cache "$ART/sfresh_inputs/cache" \
     --eval-grid "$DATA/eval_grid_windows.json" --eval-join "$DATA/join_eval141.npz" --out-dir "$od" \
     --replan-steps 10 --nav deadreckon --schedule periodic --limit causal --dump-ticks "$od/ticks" --dump-heads --device cpu "$@" < /dev/null
}
# ---- gate 2b: pulled-file controls (KP1 KR1 KR2-argv KR3) in seconds, before any rollout (code/precheck_pulled.py = the reader's own functions)
"$PY" -I "$PK/code/precheck_pulled.py" --r-arm "$ARM" > "$PK/logs/precheck_$ARM.out" 2>&1 || fail "pulled-file controls refuse $ARM before any rollout: see logs/precheck_$ARM.out" 8
log "ZZRNEXT-PRECHECK-OK $ARM"
# ---- gate 3: pre-flight (6 windows): the arm's s0 checkpoint with kbox, its s1 checkpoint with hold_now; each must exit 0 and write its ticks file
pre() {   # tag run flags...
  local tag=$1 run=$2; shift 2; rm -rf "$W/smoke/$tag"; mkdir -p "$W/smoke/$tag"
  run_tool "$tag" "$RD/$run/ckpt_last.pt" "$W/smoke/$tag" --max-windows 6 "$@" > "$PK/logs/smoke_$tag.out" 2>&1; local rc=$?
  [ $rc -eq 0 ] && [ -f "$W/smoke/$tag/ticks/ticks_$tag.npz" ] || fail "pre-flight $tag rc=$rc (a checkpoint does not load in the dev-box eval tree, or a flag is refused): see logs/smoke_$tag.out" 7
  log "ZZRNEXT-PREFLIGHT-OK $tag"
}
pre PRE_${ARM}_s0_kbox M2b_s0_recft_${ARM} --smooth sg72 --scene logged --feasible kbox
pre PRE_${ARM}_s1_hold M2b_s1_recft_${ARM} --smooth sg72 --scene hold_now --freeze-card-ref "$CARD_LINE"

run_one() {   # tag ckrun flags...
  local tag=$1 ck=$2; shift 2
  [ -f "$OUT/done_$tag.txt" ] && { log "ZZRNEXT-SKIP $tag"; return 0; }
  [ -d "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/_incomplete_${tag}_$(date +%H%M%S)"
  mkdir -p "$OUT/$tag"
  local t0=$(date +%s)
  log "ZZRNEXT-RUN $tag flags=$*"
  run_tool "$tag" "$RD/$ck/ckpt_last.pt" "$OUT/$tag" "$@" > "$PK/logs/run_$tag.out" 2>&1; local rc=$?
  if [ $rc -eq 0 ]; then
    printf '{"tag": "%s", "flags": "%s", "ckpt": "%s", "ckpt_sha256": "%s", "tool_sha256": "%s", "card_line": "%s", "threads": %s, "wall_s": %s}\n' "$tag" "$*" "$RD/$ck/ckpt_last.pt" \
       "$(sha256sum "$RDB/$ck/ckpt_last.pt" | cut -c1-64)" "$TOOL_SHA" "$CARD_LINE" "$TH" "$(( $(date +%s) - t0 ))" > "$OUT/done_$tag.txt"
    log "ZZRNEXT-RUN-DONE $tag wall=$(( $(date +%s) - t0 ))s"
  else
    log "ZZRNEXT-RUN-FAIL $tag rc=$rc"
  fi
  return $rc
}
nfail=0
while IFS=$'\t' read -r tag seed ckrun scene feas flags <&3; do
  extra=(); [ "$scene" = "hold_now" ] && extra=(--freeze-card-ref "$CARD_LINE")
  # shellcheck disable=SC2086
  run_one "$tag" "$ckrun" $flags "${extra[@]}" || nfail=$((nfail+1))
done 3< <(table)
log "ZZRNEXT-ROLLOUTS-DONE nfail=$nfail"
missing=""; while IFS=$'\t' read -r tag rest; do [ -f "$OUT/done_$tag.txt" ] || missing="$missing $tag"; done < <(table)
[ -z "$missing" ] || fail "missing done markers:$missing" 3
log "ZZRNEXT-READ start"
"$PY" -I "$PK/code/recover_read.py" --mode production --r-arm "$ARM" --out-dir "$OUT" --reuse-dir "$REUSE" --work-dir "$W/work" --raw "$PK/raw" --recover-dir "$RD" > "$PK/logs/read_$ARM.out" 2>&1; rc=$?
log "ZZRNEXT-READ rc=$rc"
[ $rc -eq 0 ] || fail "read rc=$rc (a control may have REFUSED the read: see logs/read_$ARM.out and raw/RECOVER_READ_$ARM.json 'refused')" "$rc"
[ -f "$PK/raw/RECOVER_READ_$ARM.json" ] || fail "the read returned 0 but raw/RECOVER_READ_$ARM.json is absent" 3
"$PY" -I "$PK/code/recover_next_combine.py" --raw "$PK/raw" > "$PK/logs/combine_$ARM.out" 2>&1; rc=$?
log "ZZRNEXT-COMBINE rc=$rc"
[ $rc -eq 0 ] && [ -f "$PK/raw/RECOVER_NEXT.json" ] || fail "combiner rc=$rc: see logs/combine_$ARM.out" 3
echo "$(date +%FT%T) RECOVER_READ_$ARM.json and RECOVER_NEXT.json written" > "$PK/raw/RNEXT_$ARM.done"
log "ZZRNEXT-CHAIN-DONE arm=$ARM"
