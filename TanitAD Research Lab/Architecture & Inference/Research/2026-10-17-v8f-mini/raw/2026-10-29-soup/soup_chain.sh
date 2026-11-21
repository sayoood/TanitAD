#!/bin/bash
# PREREG_V8F_SOUP section 13a: the DEV-BOX CPU part, as ONE detached, SEQUENTIAL chain.  ZERO training, ZERO GPU, no network and no remote shell of any kind (soup_selftest scans this file, the lib, the launcher and the
# waker for remote-shell commands and finds none, and runs the whole chain with every such command trapped).  Adapted from E:/Projects/TanitAD-artifacts/launch_intent_t1/lit1_chain.sh (read-only): same guards, same
# done records, same resume rule; the differences: there is nothing to wait for (every checkpoint exists), the steps are the rendered argv/PLAN.txt, and the reader's PREREG and its sha256 come from the ENVIRONMENT.
#
#   for each step of argv/PLAN.txt, in order, ONE torch process at a time:   T1-kin causal, T1-kin logged, fresh-age T0, standard T0 of every soup; then the standard T0 of DAG-OACC_s and A_s (K-T0DEV)
#     before: RAM >= 2048 MB, the output drive >= 5 GB, C: >= 2 GB (it WAITS, never deletes); the tool tree re-hashed; the checkpoint re-hashed against its pin
#     after:  the tree and the checkpoint re-hashed; the outputs checked (>= 141 dumps for a rollout); a done record naming the checkpoint sha256 (before / after), the tool digest and the command sha256
#   then: every done record re-verified, the soup reader's manifest (runs_soup.json), the reader (--prereg and --prereg-sha from SOUP_PREREG / SOUP_PREREG_SHA; REFUSED at the start if either is unset).
#
#   bash soup_chain.sh --dry-run     every gate, the plan, the 28 commands (each parsed by the pinned tool's own argparse; nothing runs) and the reader command.  Ends ZZSOUP-DRY-OK (or ZZSOUP-STOP / ZZSOUP-DRY-NOT-READY).
#   bash soup_chain.sh --write-pins  rewrite PINS.txt (a deliberate act: the chain STOPs on any difference from PINS.txt)
#   bash soup_chain.sh               the real chain.  RESUME = run the same command again: a finished step is skipped by a done record naming THIS checkpoint, THIS tool digest and THIS command (and its outputs are all
#                                    there); an incomplete earlier attempt is MOVED to <OUT>/_incomplete_*, never deleted.  The previous chain.out is rotated to chain.out.prev_<time> so the waker sees only this run's markers.
# Markers (own log, <LOG>/chain.out, ANCHORED at line start):
#   ZZSOUP-STEP-DONE <n>/<N> <tag> (<kind>)                  one per step (done or skipped)
#   ZZSOUP-DONE <n>/<N> steps, reader rc <rc> ...            finished.  reader rc 0 = an outcome written | 2 = REFUSED (a control failed; nothing written) | 3 = UNPLACED | other = the reader crashed
#   ZZSOUP-STOP <why>                                        refused.  Fix the named cause and re-run the SAME command.  Also touches <LOG>/STOP.
#   ZZSOUP-DRY-OK | ZZSOUP-DRY-NOT-READY <why>               the dry run's end markers.
# Environment (a WMI-launched process does not inherit the caller's shell: put these in soup_chain.cmd):
#   SOUP_PREREG  SOUP_PREREG_SHA     REQUIRED: the promoted PREREG_V8F_SOUP and its sha256 (the reader's --prereg / --prereg-sha; the file must hash to it)
#   SOUP_OUT (F:/TanitAD-scratch/soup_t1_out)  SOUP_LOG (<this dir>/logs)  SOUP_TREE (F:/TanitAD-scratch/launch_intent_t1/tree_li)  SOUP_PLAN (argv/PLAN.txt)  SOUP_PINS (PINS.txt)  SOUP_SPEC  SOUP_BUILDER  SOUP_READER
#   SOUP_LI_READER  SOUP_LI_READER_SHA  SOUP_LI_RUNS  SOUP_BANKED_LI_READ  SOUP_ORC_DIR  SOUP_ARGV_DIR     SOUP_RAM_MIN_MB (2048)  SOUP_F_MIN_GB (5)  SOUP_C_MIN_MB (2048)  SOUP_E_MIN_MB (200)
#   SOUP_GUARD_WAIT_S (7200)  SOUP_POLL_S (60)     SOUP_WORLD (TEST ONLY: a throw-away directory carrying SOUP_TESTWORLD.marker; refused anywhere else, see soup_chain_lib.py world_ok)
set -u
shopt -u patsub_replacement 2>/dev/null || true      # bash 5.2: '&' in a replacement string would be expanded ("Architecture & Inference" is in the tool paths)
export PATH="$PATH:/usr/bin:/mingw64/bin"
MODE=real
case "${1:-}" in --dry-run) MODE=dry ;; --write-pins) MODE=pins ;; "") ;; *) echo "usage: soup_chain.sh [--dry-run | --write-pins]   (configuration = SOUP_* environment variables, see the header)"; exit 2 ;; esac
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); Hm=$(cygpath -m "$H")
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LIB="$Hm/soup_chain_lib.py"
m() { cygpath -m "$1"; }
u() { cygpath -u "$1"; }
SPEC=${SOUP_SPEC:-$Hm/soups.json}; PLAN=${SOUP_PLAN:-$Hm/argv/PLAN.txt}; ARGVDIR=${SOUP_ARGV_DIR:-$Hm/argv}; PINS=${SOUP_PINS:-$Hm/PINS.txt}
BUILDER=${SOUP_BUILDER:-$Hm/soup_build.py}; READER=${SOUP_READER:-$Hm/read/soup_read.py}
LIREADER=${SOUP_LI_READER:-$Hm/read/ref/launch_intent_read.py}; LIREADER_SHA=${SOUP_LI_READER_SHA:-fb3d2d7d945b54209b62d5f906d4dcea30925a24f0b807459b8cbf3d7f0747a9}
LIRUNS=${SOUP_LI_RUNS:-F:/TanitAD-scratch/launch_intent_t1_out/runs.json}; BANKED=${SOUP_BANKED_LI_READ:-F:/TanitAD-scratch/launch_intent_t1_out/read_A1/LAUNCH_INTENT_READ.json}
ORCDIR=${SOUP_ORC_DIR:-E:/Projects/TanitAD-artifacts/dagorc_rebase/read}
TREE=${SOUP_TREE:-F:/TanitAD-scratch/launch_intent_t1/tree_li}; OUT=${SOUP_OUT:-F:/TanitAD-scratch/soup_t1_out}; LOG=${SOUP_LOG:-$H/logs}
PREREG=${SOUP_PREREG:-}; PREREG_SHA=${SOUP_PREREG_SHA:-}
RAMMIN=${SOUP_RAM_MIN_MB:-2048}; FMIN=${SOUP_F_MIN_GB:-5}; CMIN=${SOUP_C_MIN_MB:-2048}; EMIN=${SOUP_E_MIN_MB:-200}; GUARDWAIT_S=${SOUP_GUARD_WAIT_S:-7200}; POLL=${SOUP_POLL_S:-60}
WORLD=${SOUP_WORLD:-}; TESTWORLD=0
if [ -n "$WORLD" ]; then   # TEST ONLY: honoured iff the directory is a valid throw-away world (soup_chain_lib.py world_ok); anything else is refused before anything is read
  SOUP_WORLD="$WORLD" "$PY" -I "$LIB" world < /dev/null > /dev/null 2>&1 || { echo "ZZSOUP-STOP SOUP_WORLD=$WORLD is not a valid throw-away world: refused, nothing was run"; exit 1; }
  TESTWORLD=1; export SOUP_WORLD="$WORLD"
fi
fsha() { sha256sum < "$(u "$1")" | cut -c1-64; }
fmt() { "$PY" -I "$LIB" fmt -- "$@" < /dev/null; }
cmd_sha() { printf '%s\0' "$@" | sha256sum | cut -c1-64; }
TREEm=$(m "$TREE"); TREEu=$(u "$TREE"); OUTm=$(m "$OUT"); OUTu=$(u "$OUT"); LOGu=$(u "$LOG"); LOGm=$(m "$LOG")
SPECm=$(m "$SPEC"); PLANm=$(m "$PLAN"); PINSm=$(m "$PINS"); BUILDERm=$(m "$BUILDER"); READERm=$(m "$READER"); LIREADERm=$(m "$LIREADER"); LIRUNSm=$(m "$LIRUNS"); BANKEDm=$(m "$BANKED"); ORCDIRm=$(m "$ORCDIR"); ARGVDIRm=$(m "$ARGVDIR")
TMPLARGS=(); for f in "$(u "$ARGVDIR")"/*.tmpl; do [ -e "$f" ] && TMPLARGS+=(--tmpl "$(m "$f")"); done
EXPT=(); [ "$TESTWORLD" = 1 ] && EXPT=(--expect-tree-digest none)      # a test world's stub tree has no production digest; real mode compares with the pinned 579183a3...
LIBARGS=(--plan "$PLANm" --tree "$TREEm" --out "$OUTm")
PINARGS=("${LIBARGS[@]}" --spec "$SPECm" --builder "$BUILDERm" --lib "$LIB" --chain "$Hm/soup_chain.sh" --reader "$READERm" --li-reader "$LIREADERm" --banked "$BANKEDm" --li-runs "$LIRUNSm" "${TMPLARGS[@]}" "${EXPT[@]}")

# ------------------------------------------------------------------------------------------------ the log, the lock, say / stop
DRYBAD=""
if [ "$MODE" != real ]; then
  say() { echo "$(date '+%m-%d %H:%M:%S') ${MODE^^}: $*"; }
  stop() { say "$*"; echo "ZZSOUP-STOP $*"; exit 1; }
else
  mkdir -p "$LOGu"
  say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOGu/chain.out"; }
  stop() { say "STOP: $*"; echo "ZZSOUP-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }
  rm -f "$LOGu/STOP"
  if ! mkdir "$LOGu/chain.lock" 2>/dev/null; then      # ONE instance only (two instances would run two torch processes at once)
    op=$(cat "$LOGu/chain.lock/winpid" 2>/dev/null)
    if [ -n "$op" ] && tasklist //FI "PID eq $op" //NH 2>/dev/null | grep -q " $op "; then echo "ZZSOUP-STOP another instance of this chain is running (winpid $op, $LOGm/chain.lock)"; exit 1; fi
    rm -rf "$LOGu/chain.lock"; mkdir "$LOGu/chain.lock" || { echo "ZZSOUP-STOP cannot take $LOGm/chain.lock"; exit 1; }
  fi
  cat /proc/$$/winpid > "$LOGu/chain.lock/winpid" 2>/dev/null; cp "$LOGu/chain.lock/winpid" "$LOGu/chain.winpid" 2>/dev/null
  trap 'rm -rf "$LOGu/chain.lock"' EXIT
  [ -s "$LOGu/chain.out" ] && mv "$LOGu/chain.out" "$LOGu/chain.out.prev_$(date +%m%d%H%M%S)"
  say "=== soup_chain start (mode real; test world $TESTWORLD; tool tree $TREEm; plan $PLANm; OUT $OUTm; reader $READERm)"
fi
nostop() { [ "$MODE" = real ] || return 0; [ -e "$LOGu/STOP" ] && stop "the STOP file $LOGm/STOP is present: the chain was asked to stop"; return 0; }
cfree_mb() { df -m /c | awk 'NR==2{print $4}'; }
efree_mb() { df -m /e | awk 'NR==2{print $4}'; }
CF0=$(cfree_mb); [ "$CF0" -ge "$CMIN" ] || stop "C: has $CF0 MB free (< $CMIN MB): the pagefile is on C: and a full C: killed torch start-ups before"
say "C: gate: $CF0 MB free >= $CMIN MB"
if [ "$TESTWORLD" = 0 ]; then EF0=$(efree_mb); [ "$EF0" -ge "$EMIN" ] || stop "E: has $EF0 MB free (< $EMIN MB): the chain's logs land there"; say "E: gate: $EF0 MB free >= $EMIN MB"; fi

# ------------------------------------------------------------------------------------------------ 0  the PREREG: both from the environment, and the file is the pinned one
PRE_WHY=""
if [ -z "$PREREG" ] || [ -z "$PREREG_SHA" ]; then PRE_WHY="SOUP_PREREG and SOUP_PREREG_SHA must BOTH be set (the promoted PREREG_V8F_SOUP's path and sha256; soup_chain.cmd has the lines): got PREREG='${PREREG}' SHA='${PREREG_SHA}'"
elif ! [[ "$PREREG_SHA" =~ ^[0-9a-f]{64}$ ]]; then PRE_WHY="SOUP_PREREG_SHA is not a 64-hex sha256: '$PREREG_SHA'"
elif [ ! -f "$(u "$PREREG")" ]; then PRE_WHY="SOUP_PREREG $PREREG is not a file"
elif [ "$(fsha "$PREREG")" != "$PREREG_SHA" ]; then PRE_WHY="the PREREG at $PREREG hashes to $(fsha "$PREREG"), not to SOUP_PREREG_SHA $PREREG_SHA"
fi
if [ -n "$PRE_WHY" ]; then [ "$MODE" = dry ] && { say "a real run would STOP here: $PRE_WHY"; DRYBAD="PREREG: $PRE_WHY"; } || { [ "$MODE" = pins ] || stop "$PRE_WHY"; }; else PREREGm=$(m "$PREREG"); say "PREREG $PREREGm sha256 ${PREREG_SHA:0:16}... (the pin; the reader gets --prereg and --prereg-sha from it)"; fi

# ------------------------------------------------------------------------------------------------ 1  the tool tree, the plan, readiness, pins
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TREEu/stack")      # OMP/MKL 8 = the banked rollouts' threads
export PYTHONPATH="$TREEm/stack;$TREEm/taniteval;$TREEm/stack/scripts"
tree_digest() { "$PY" -I "$LIB" treedigest --tree "$TREEm" < /dev/null; }
TOOLD=$(tree_digest); [ -n "$TOOLD" ] || stop "cannot compute the tool tree digest of $TREEm"
PL=$("$PY" -I "$LIB" plan "${LIBARGS[@]}" < /dev/null 2>&1) || stop "the plan $PLANm is invalid: $(echo "$PL" | tr '\n' ' ' | cut -c1-500)"
mapfile -t PLANL <<< "$PL"; NSTEPS=${#PLANL[@]}
say "plan: $NSTEPS steps from $PLANm (sha256 $(fsha "$PLAN" | cut -c1-16)...), tool tree $TREEm digest ${TOOLD:0:16}..."
RDY=$("$PY" -I "$LIB" ready "${LIBARGS[@]}" --spec "$SPECm" < /dev/null 2>&1) || stop "a checkpoint is not ready (section 13a): $(echo "$RDY" | tr '\n' ' ' | cut -c1-600)"
say "readiness: $RDY"
[ -f "$(u "$READER")" ] && [ -f "$(u "$BUILDER")" ] && [ -f "$(u "$LIREADER")" ] || stop "a reader file is missing (soup reader $READERm, builder $BUILDERm, LAUNCH-INTENT reader $LIREADERm)"
[ "$(fsha "$LIREADER")" = "$LIREADER_SHA" ] || stop "the LAUNCH-INTENT reader $LIREADERm hashes to $(fsha "$LIREADER"), not to the pinned $LIREADER_SHA"
if [ "$MODE" = pins ]; then
  "$PY" -I "$LIB" pins-write "${PINARGS[@]}" --write "$PINSm" < /dev/null || stop "cannot write $PINSm"; exit 0
fi
PV=$("$PY" -I "$LIB" pins-verify "${PINARGS[@]}" --pins "$PINSm" < /dev/null 2>&1) || stop "PINS.txt: $(echo "$PV" | tr '\n' ';' | cut -c1-900) -- nothing was started"
say "PINS.txt verified ($(grep -vc '^#' "$(u "$PINS")") pins: the plan, the templates, the spec, the builder, the lib, this script, the readers, the banked LAUNCH-INTENT files, every checkpoint, the tool tree digest)"
RSHA=$(fsha "$READER"); BSHA=$(fsha "$BUILDER")
MINI=$TREEm/stack/scripts/v8f_mini_eval.py; MINISHA=$(fsha "$MINI")
pin_check() { [ "$(tree_digest)" = "$TOOLD" ] || stop "the tool tree changed ($1)"; [ "$(fsha "$MINI")" = "$MINISHA" ] || stop "the T0 tool v8f_mini_eval.py changed ($1)"; }
pin_check_end() {
  pin_check "$1"
  [ "$(fsha "$READER")" = "$RSHA" ] || stop "the soup reader changed ($1)"
  [ "$(fsha "$BUILDER")" = "$BSHA" ] || stop "the soup builder changed ($1)"
  [ -z "$PREREG" ] || [ "$(fsha "$PREREG")" = "$PREREG_SHA" ] || stop "the PREREG changed ($1)"
}

# ------------------------------------------------------------------------------------------------ 2  the dry run
reader_cmd() {
  RCMD=("$PY" -I "$READERm" --manifest "$OUTm/runs_soup.json" --out "$OUTm/read/SOUP_READ.json" --prereg "${PREREGm:-UNSET}" --prereg-sha "${PREREG_SHA:-UNSET}" --li-reader "$LIREADERm" --li-reader-sha "$LIREADER_SHA"
        --orc-dir "$ORCDIRm" --banked-li-read "$BANKEDm" --soup-build "$BUILDERm" --soup-build-sha "$BSHA" --spec "$SPECm" --work-dir "$OUTm/_tmp")
}
step_cmd() { mapfile -t CMD < <("$PY" -I "$LIB" argv "${LIBARGS[@]}" --idx "$1" < /dev/null); CMD=("${CMD[@]%$'\r'}"); [ "${#CMD[@]}" -ge 8 ] || stop "step $1: the plan rendered ${#CMD[@]} tokens"; }
if [ "$MODE" = dry ]; then
  say "plan header: $(head -1 "$(u "$PLAN")" | cut -c1-200)"
  say "== each command below is the plan's, parsed by the pinned tool's own argparse; nothing is run"
  for line in "${PLANL[@]}"; do
    IFS=$'\t' read -r idx kind tag ckpt <<< "$line"; step_cmd "$idx"
    pr=$("$PY" -I "$LIB" parsecheck --tool "$(m "${CMD[1]}")" -- "${CMD[@]:2}" < /dev/null 2>&1 | head -1 | cut -c1-12)
    say "  [$idx $kind] parse: ${pr:-?}  $(fmt "${CMD[@]}" | cut -c1-400)"
    case "$pr" in PARSE-OK*) ;; *) DRYBAD="${DRYBAD:+$DRYBAD; }step $tag does not parse" ;; esac
  done
  reader_cmd; say "== reader: $(fmt "${RCMD[@]}")"
  say "   -> $OUTm/read/SOUP_READ.json (rc 0 outcome | 2 REFUSED, nothing written | 3 UNPLACED); stdout -> $LOGm/soup_read.out.  NOTHING was run, written, launched or waited for."
  if [ -z "$DRYBAD" ]; then echo "ZZSOUP-DRY-OK"; exit 0; fi
  echo "ZZSOUP-DRY-NOT-READY $DRYBAD"; exit 1
fi

# ------------------------------------------------------------------------------------------------ 3  the steps.  One step = one process.  Done records name the checkpoint, the tool digest and the command.
mkdir -p "$OUTu" "$OUTu/ticks" "$OUTu/argv" "$OUTu/t0fresh" "$OUTu/t0std" "$OUTu/t0dev" "$OUTu/read" "$OUTu/_tmp"
export TMP="$OUTm/_tmp" TEMP="$OUTm/_tmp"        # keep the temp files off C:
guards_wait() {   # guards_wait <what>: RAM, the output drive and C: BEFORE every step; it waits (never deletes), up to GUARDWAIT_S
  local w=0 what=$1 out t0; t0=$(date +%s)
  while :; do
    out=$("$PY" -I "$LIB" guards --ram-min-mb "$RAMMIN" --disk "${OUTm:0:3}" --disk-min-gb "$FMIN" --c-min-mb "$CMIN" 2>&1 < /dev/null) && { [ $w -gt 0 ] && say "$what: guards hold again ($out)"; return 0; }
    nostop; [ $w -eq 0 ] && say "$what: waiting, a guard is LOW: $out"; w=$((w + 1))
    [ $(( $(date +%s) - t0 )) -ge "$GUARDWAIT_S" ] && stop "a guard stayed LOW for $GUARDWAIT_S s before $what: $out"
    sleep "$POLL"
  done
}
aside() {   # aside <path>...: MOVE an earlier attempt's output into a fresh _incomplete_<time> directory (never delete)
  local w f; w="$OUTu/_incomplete_$(date +%m%d%H%M%S)_$$_$RANDOM"; mkdir -p "$w"
  for f in "$@"; do [ -e "$(u "$f")" ] && mv "$(u "$f")" "$w/"; done
  say "moved the output of an earlier attempt to $(m "$w")"
}
run_step() {   # run_step <idx> <kind> <tag> <ckpt>
  local idx=$1 kind=$2 tag=$3 ck=$4 sha sha2 pinned cs t0 rc att why o any f
  step_cmd "$idx"; cs=$(cmd_sha "${CMD[@]}")
  [ -f "$(u "$ck")" ] || stop "step $tag: no checkpoint at $ck"
  sha=$(fsha "$ck"); pinned=$("$PY" -I "$LIB" pinsha --pins "$PINSm" --ckpt "$ck" "${LIBARGS[@]}" < /dev/null)
  [ -n "$pinned" ] && [ "$sha" = "$pinned" ] || stop "step $tag: $ck hashes to ${sha:0:16}, PINS.txt has '${pinned:0:16}' (before $tag)"
  if why=$("$PY" -I "$LIB" stepdone "${LIBARGS[@]}" --idx "$idx" --ckpt-sha "$sha" --tool-digest "$TOOLD" --cmd-sha "$cs" < /dev/null 2>&1); then say "skip $tag (complete, same checkpoint ${sha:0:12}, same tool digest, same command)"; return 0; fi
  say "step $tag is not done ($why)"
  mapfile -t OUTS < <("$PY" -I "$LIB" outputs "${LIBARGS[@]}" --idx "$idx" < /dev/null); OUTS=("${OUTS[@]%$'\r'}")
  any=0; for f in "${OUTS[@]}"; do [ -e "$(u "$f")" ] && any=1; done
  [ $any -eq 1 ] && aside "${OUTS[@]}"
  guards_wait "$tag"
  nostop; pin_check "before $tag"
  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed since the pins were verified (before $tag)"
  for att in 1 2; do
    case "$kind" in causal|logged) printf '%s\n' "${CMD[@]}" > "$OUTu/argv/$tag.argv" ;; esac
    t0=$(date +%s)
    say "start $tag [$kind] (attempt $att; ckpt $(basename "$(dirname "$ck")") ${sha:0:12}): $(fmt "${CMD[@]}")"
    "${CMD[@]}" > "$LOGu/$tag.out" 2>&1 < /dev/null; rc=$?
    say "end $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
    if [ $rc -eq 139 ] && [ $att -eq 1 ]; then say "$tag: native crash (rc 139): one retry; the crashed attempt's output is moved aside and never read"; mv "$LOGu/$tag.out" "$LOGu/$tag.segv.out"; aside "${OUTS[@]}"; continue; fi
    break
  done
  pin_check "after $tag"
  sha2=$(fsha "$ck"); [ "$sha2" = "$sha" ] || stop "$ck changed during $tag (${sha:0:16} -> ${sha2:0:16})"
  if [ $rc -ne 0 ]; then stop "step $tag failed (rc $rc): $(tail -2 "$LOGu/$tag.out" | tr '\n' ' ' | cut -c1-300)"; fi
  o=$("$PY" -I "$LIB" stepcheck "${LIBARGS[@]}" --idx "$idx" < /dev/null 2>&1) || stop "step $tag: $o"
  "$PY" -I "$LIB" stepwrite "${LIBARGS[@]}" --idx "$idx" --ckpt-sha "$sha" --before "$sha" --after "$sha2" --tool-digest "$TOOLD" --cmd-sha "$cs" --wall "$(( $(date +%s) - t0 ))" < /dev/null || stop "step $tag: cannot write the done record"
  say "done $tag ($o)"
}
say "== sequential: $NSTEPS steps, one process at a time, OMP/MKL $OMP_NUM_THREADS threads; tool digest ${TOOLD:0:16}..."
n=0
for line in "${PLANL[@]}"; do
  IFS=$'\t' read -r idx kind tag ckpt <<< "$line"
  nostop
  run_step "$idx" "$kind" "$tag" "$ckpt"
  n=$((n + 1)); echo "ZZSOUP-STEP-DONE $n/$NSTEPS $tag ($kind)" >> "$LOGu/chain.out"; tail -1 "$LOGu/chain.out"
done
pin_check_end "end of the steps"
n=0; for line in "${PLANL[@]}"; do
  IFS=$'\t' read -r idx kind tag ckpt <<< "$line"; step_cmd "$idx"
  "$PY" -I "$LIB" stepdone "${LIBARGS[@]}" --idx "$idx" --ckpt-sha "$(fsha "$ckpt")" --tool-digest "$TOOLD" --cmd-sha "$(cmd_sha "${CMD[@]}")" < /dev/null > /dev/null 2>&1 && n=$((n + 1))
done
[ "$n" -eq "$NSTEPS" ] || stop "$n of $NSTEPS steps have a verifying done record"
say "== all $n steps verified; the manifest"
"$PY" -I "$LIB" manifest "${LIBARGS[@]}" --spec "$SPECm" --li-runs "$LIRUNSm" --write "$OUTm/runs_soup.json" > "$LOGu/manifest.out" 2>&1 < /dev/null || stop "manifest: $(tr '\n' ' ' < "$LOGu/manifest.out" | cut -c1-500)"
say "$(tail -1 "$LOGu/manifest.out")"
[ -n "$PREREG" ] && [ -n "$PREREG_SHA" ] || stop "SOUP_PREREG / SOUP_PREREG_SHA vanished before the reader"
reader_cmd
guards_wait "the reader"; nostop; pin_check_end "before the reader"
say "== reader: $(fmt "${RCMD[@]}")"
env -u TANITEVAL_STACK_OVERRIDE -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${RCMD[@]}" > "$LOGu/soup_read.out" 2>&1 < /dev/null; RRC=$?
say "reader rc=$RRC: $(grep -a -E '"OUTCOME"|REFUSED|UNPLACED|wrote ' "$LOGu/soup_read.out" | head -3 | tr '\n' ' ' | cut -c1-400)"
pin_check_end "after the reader"
echo "ZZSOUP-DONE $n/$NSTEPS steps, reader rc $RRC ($(case $RRC in 0) echo 'outcome written' ;; 2) echo 'REFUSED, nothing written' ;; 3) echo 'UNPLACED, written' ;; *) echo 'reader crashed' ;; esac)): $OUTm/read/SOUP_READ.json  log $LOGm/soup_read.out" >> "$LOGu/chain.out"
tail -1 "$LOGu/chain.out"
