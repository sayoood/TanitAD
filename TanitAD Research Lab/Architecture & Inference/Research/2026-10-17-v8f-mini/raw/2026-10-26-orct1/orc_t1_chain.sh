#!/bin/bash
# PREREG_V8F_ORECOVER (Project Steering/PREREG_V8F_ORECOVER.md + Amendments 1-4), the DEV-BOX CPU part, as ONE detached chain (ZERO training, ZERO GPU, box 2 is never touched):
#   for each of the four runs, in the PREREG's order (section 5):  ORC s0, ORC-0 s0, ORC s1, ORC-0 s1
#     1. wait until box 2's pull of that run to E:/Projects/TanitAD-artifacts/v8f-orecover/M2b_s<S>_<orc|orc0>/ is COMPLETE  (orct1_pullcheck.py: SHA256SUMS.txt lists exactly the eight files the box-2
#        chain hashes after its last scp, each file exists, is non-empty and re-hashes to the listed sha256, summary.json done).  By CONTENT: never on a half-pulled file, never on a stale SHA256SUMS.txt.
#     2. two T1-kin rollouts on eval-141, plain (no post-map), CPU:
#          v8f_t1_eval.py run --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks --dump-heads     (decisive)
#          v8f_t1_eval.py run ... --limit logged ...                                                                                            (K4 only, PREREG Amendment 2)
#        = the T1-HEADS rollouts' tool tree (E:/.../v8f-t1heads-run/tool_81aa3e36 = 81aa3e36), eval cache / tracks / grid / join, OMP/MKL 8 threads, default --batch 32.
#   two lanes (A: every --limit causal rollout, B: every --limit logged rollout), so AT MOST 2 rollout processes at once; CUDA_VISIBLE_DEVICES=-1.
#   after all four runs: orct1_manifest.py writes the reader manifest runs.json, then orc_read.py runs with every required argument and its output goes to <OUT>/read/ORC_READ.json.
#
#   bash orc_t1_chain.sh --dry-run     prints every guard result, path, pull state, the EXACT command of every rollout, the manifest it would write and the reader command; touches nothing (no file, no dir, no log).
#   bash orc_t1_chain.sh               the real chain.  RESUME = re-run the same command: a finished rollout (json + ticks + 141 episode dumps + a done record naming THIS checkpoint's sha256) is skipped,
#                                      the pins are re-asserted, an incomplete rollout of an earlier attempt is MOVED to <OUT>/_incomplete_*/ (never deleted) and redone.
# Markers (own log, <LOG>/chain.out):   ZZORCT1-DONE <n>/8 rollouts, reader rc <rc> ...   |   ZZORCT1-STOP <reason>   (STOP also touches <LOG>/STOP; the other lane stops at its next step).
#   The reader's own rc is reported, not hidden: 0 = an outcome row was written, 2 = REFUSED (a control failed; nothing written), 3 = UNPLACED.  A REFUSED read is a finished chain with a verdict "fix and re-run".
# Environment (all optional):
#   ORCT_TOOL_TREE   the tool tree (default = the T1-HEADS one)          ORCT_READER_TREE  the tree holding orc_read.py + t1hd_read.py + t1k_read.py + the PREREG (default = the oc overlay)
#   ORC_ART          where box 2's chain pulls (the chain's own variable)  ORCT_OUT / ORCT_LOG  rollout+read output (E:) / chain log
#   ORCT_B2_OUT      OPTIONAL: the box-2 chain's log (chain_v8f_orc.out).  If set, a ZZV8FMINI-STOP line that appears AFTER this chain started stops this chain (it would wait for pulls that never come).
#   ORCT_M2B_SHA_JSON OPTIONAL: the box-2 chain's measured ckpt_sha256.json: its M2b_s0 / M2b_s1 values must equal the runs' recorded init checkpoint, and are passed to the reader.  Default: from the runs' own records.
#   ORCT_MAX_WAIT_H  longest wait for ONE pull, hours (default 40)         ORCT_C_MIN_MB  C: free MB needed before a rollout (default 2000; the pagefile is on C:)
#   ORCT_SMOKE=1     plumbing test only: 24 windows, tags SMOKE_*, outputs under orct1/smoke, marker ZZORCT1-SMOKE-DONE, stand-in run(s) from ORCT_RUNS (default orc:0), the manifest replicates that one run.
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
DRY=0
case "${1:-}" in --dry-run) DRY=1 ;; "") ;; *) echo "usage: orc_t1_chain.sh [--dry-run]   (configuration = ORCT_* environment variables, see the header)"; exit 2 ;; esac
H=/c/Users/Admin/orct1; Hm=C:/Users/Admin/orct1
SMOKE=${ORCT_SMOKE:-0}
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
TT=${ORCT_TOOL_TREE:-E:/Projects/TanitAD-artifacts/v8f-t1heads-run/tool_81aa3e36}
RT=${ORCT_READER_TREE:-E:/Projects/TanitAD-artifacts/rt4/tree}      # A4: a fresh export of the tip with the A4 reader (oc/tree holds the A3 one)
SF=${ORCT_INPUTS:-/c/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/bfe78918-15e8-439b-8566-593ca3297bd8/scratchpad/sfresh}
if [ "$SMOKE" = 1 ]; then
  TAGP="[SMOKE] "; PFX=SMOKE_; MAXW=24; NDUMP=1; POLL=${ORCT_POLL_S:-2}; DONE_MARK=ZZORCT1-SMOKE-DONE
  ART=${ORC_ART:-/e/Projects/TanitAD-artifacts/orct1/smoke_art}; OUT=${ORCT_OUT:-$H/smoke/out}; LOG=${ORCT_LOG:-$H/smoke/logs}
  IFS=, read -ra RUNS <<< "${ORCT_RUNS:-orc:0}"; REP=${RUNS[0]}
else
  TAGP=""; PFX=""; MAXW=0; NDUMP=141; POLL=60; DONE_MARK=ZZORCT1-DONE; REP=""
  ART=${ORC_ART:-/e/Projects/TanitAD-artifacts/v8f-orecover}; OUT=${ORCT_OUT:-/e/Projects/TanitAD-artifacts/v8f-orecover-t1}; LOG=${ORCT_LOG:-$H/logs}
  RUNS=(orc:0 orc0:0 orc:1 orc0:1)         # FOUR runs, ALWAYS ALL FOUR, in the PREREG's order; no run is conditional on another's read (ORCT_RUNS is honoured in SMOKE only)
fi
RUNSCSV=$(IFS=,; echo "${RUNS[*]}")
m() { cygpath -m "$1"; }
TTu=$(cygpath -u "$TT"); TTm=$(m "$TT"); RTu=$(cygpath -u "$RT"); RTm=$(m "$RT"); ARTu=$(cygpath -u "$ART"); ARTm=$(m "$ART")
OUTu=$(cygpath -u "$OUT"); OUTm=$(m "$OUT"); LOGu=$(cygpath -u "$LOG"); LOGm=$(m "$LOG"); SFu=$(cygpath -u "$SF"); SFm=$(m "$SF")
TOOL="$TTu/stack/scripts/v8f_t1_eval.py"
READER="$RTm/$PK/raw/2026-10-26-orcread/orc_read.py"; T1HD="$RTm/$PK/raw/2026-10-26-t1hdread/t1hd_read.py"; T1K="$RTm/$PK/raw/2026-10-26-t1kin-hacc/t1k_read.py"; PREREG="$RTm/Project Steering/PREREG_V8F_ORECOVER.md"
GRID="$TTm/$PK/raw/eval_grid_windows.json"; JOIN="$TTm/$PK/raw/join_eval141.npz"; TRACKS="$SFm/inputs/native_tracks.pt"; CACHEm="$SFm/cache"
MAXWAIT_S=$(( ${ORCT_MAX_WAIT_H:-40} * 3600 )); CMIN=${ORCT_C_MIN_MB:-2000}
# PREREG_V8F_ORECOVER Amendment 4: the T0 eval at --ages fresh, on CPU (~32 s per run, measured in SFRESH), is K4's comparator. K4 compares a logged rollout's tick 0 (ages 0/0)
# with "the arm's own T0 eval"; box 2's standard eval is --ages cycle, which differs from tick 0 by construction (M2b s0, S pos@30: 7.0231 cycle vs 0.9774 fresh). The manifest
# carries it as `t0_fresh` (K4 only); `t0` stays box 2's standard eval (R-T0, T0-CKPT, the families). Both are required, so neither switch exists any more.
[ "${ORCT_T0_FRESH:-1}" = 1 ] || { echo "Amendment 4: the fresh-age T0 eval is K4's comparator; ORCT_T0_FRESH=${ORCT_T0_FRESH} is refused"; exit 2; }
[ -z "${ORCT_T0:-}" ] || { echo "Amendment 4: ORCT_T0 is gone (t0 is box 2's standard eval; K4 reads t0_fresh)"; exit 2; }
T0F=1
V9="$SFm/inputs/v9"; P10="$SFm/inputs/p10/p10_prior.pt"; T0DIR="$OUTm/t0fresh"; KIN="$RTm/$PK/raw/eval141_kin.npz"      # the T1 tool tree has no eval141_kin.npz; the reader tree (a full export) has it

if [ "$DRY" = 1 ]; then
  say() { echo "$(date '+%m-%d %H:%M:%S') ${TAGP}DRY: $*"; }
  stop() { say "$*"; echo "ZZORCT1-STOP $*"; exit 1; }
else
  mkdir -p "$LOGu" "$OUTu" "$OUTu/ticks"
  say() { echo "$(date '+%m-%d %H:%M:%S') ${TAGP}$*" | tee -a "$LOGu/chain.out"; }
  stop() { say "STOP: $*"; echo "ZZORCT1-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }
  rm -f "$LOGu/STOP"
  say "=== orc_t1_chain start (tool tree $TTm, reader tree $RTm, ART $ARTm, OUT $OUTm; runs $RUNSCSV)"
fi
nostop() { [ "$DRY" = 1 ] && return 0; [ -e "$LOGu/STOP" ] && exit 1; return 0; }
fmt() { local o="" a; for a in "$@"; do case "$a" in *[!A-Za-z0-9_./:=,@%+-]*) o="$o '$a'" ;; *) o="$o $a" ;; esac; done; echo "${o# }"; }
rundir() { echo "M2b_s${2}_${1}"; }                                   # rundir <arm> <seed>: the directory box 2's chain pulls into
armU() { if [ "$1" = orc ]; then echo ORC; else echo ORC0; fi; }
tagof() { echo "${PFX}$(armU "$1")_s${2}_m10_${3}_heads"; }          # tagof <arm> <seed> <limit>: the rollout's --tag (T1-HEADS naming)
for r in "${RUNS[@]}"; do case "$r" in orc:[01]|orc0:[01]) ;; *) stop "bad run spec '$r' (arm:seed, arm in orc/orc0, seed 0/1)" ;; esac; done
[ "$SMOKE" = 1 ] || [ "${ORCT_RUNS:-}" = "" ] || say "NOTE: ORCT_RUNS is ignored outside SMOKE (the PREREG runs ALL FOUR)"

# ------------------------------------------------------------------------------------------------ 0a  the flag guards (grep on the tool, each with a same-breath control; then the tool's own --help)
[ -f "$TOOL" ] || stop "no tool at $TTm/stack/scripts/v8f_t1_eval.py"
c_pos=$(grep -cE 'add_argument\("--replan-steps"' "$TOOL"); c_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$TOOL")
[ "$c_pos" -ge 1 ] && [ "$c_neg" -eq 0 ] || stop "grep control broken on $TTm (positive control --replan-steps = $c_pos, must be >= 1; negative control = $c_neg, must be 0)"
l_pos=$(grep -cE '^LIMIT_MODES *= *\(.*"logged"' "$TOOL"); l_neg=$(grep -cE '^LIMIT_MODES *= *\(.*"zzz-no-such-limit"' "$TOOL")
[ "$l_pos" -ge 1 ] && [ "$l_neg" -eq 0 ] || stop "grep control broken on LIMIT_MODES of $TTm (positive control logged = $l_pos, must be >= 1; negative control = $l_neg, must be 0)"
n_heads=$(grep -cE 'add_argument\("--dump-heads"' "$TOOL"); n_ticks=$(grep -cE 'add_argument\("--dump-ticks"' "$TOOL"); n_caus=$(grep -cE '^LIMIT_MODES *= *\(.*"causal"' "$TOOL")
say "guards: add_argument --dump-heads = $n_heads, --dump-ticks = $n_ticks, causal in LIMIT_MODES = $n_caus  (controls: --replan-steps = $c_pos, no-such-flag = $c_neg, logged = $l_pos, no-such-limit = $l_neg) in $TTm"
[ "$n_heads" -ge 1 ] || stop "REFUSED: $TTm has no --dump-heads (grep $n_heads; the control read $c_pos, so the grep works): a rollout without the head columns cannot be read"
[ "$n_ticks" -ge 1 ] || stop "REFUSED: $TTm has no --dump-ticks (grep $n_ticks; the control read $c_pos, so the grep works)"
[ "$n_caus" -ge 1 ] || stop "REFUSED: $TTm has no causal bucket (LIMIT_MODES grep $n_caus; the control read $l_pos, so the grep works): the decisive rollout is --limit causal"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack")
export PYTHONPATH="$TTm/stack;$TTm/taniteval;$TTm/stack/scripts"
HELP=$("$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --help 2>&1 < /dev/null | tr -s '[:space:]' ' ')
echo "$HELP" | grep -q -e '--dump-heads' || stop "REFUSED: the tool's own --help lists no --dump-heads"
echo "$HELP" | grep -qE '\{logged,[^}]*causal' || stop "REFUSED: the tool's own --help lists no causal choice for --limit"
say "tool --help agrees (--dump-heads, --limit {logged,...,causal,...})"

# ------------------------------------------------------------------------------------------------ 0b  inputs and pins
md5chk() { [ "$(md5sum < "$1" 2>/dev/null | cut -c1-32)" = "$2" ] || stop "md5 of $1 is not $2"; }
md5chk "$GRID" 34ed8ac56365eeff72e42581bde2f5c4
md5chk "$JOIN" 532fcd644c0e0835e847b2f1cee011ad
md5chk "$TRACKS" 189f4e68913c0163912a509203963f8d
[ "$(ls "$SFu/cache/eval/clips" 2>/dev/null | wc -l)" = 141 ] || stop "eval cache $CACHEm is not 141 clips (the T1-HEADS inputs live in the session scratchpad; ORCT_INPUTS names another copy)"
for f in "$READER" "$T1HD" "$T1K" "$PREREG"; do [ -f "$f" ] || stop "reader input missing: $f"; done
T0IN=("$V9/v9_labels_eval139.npz" "$V9/lead_eval139.npz" "$V9/refcv6_speed_max_v8_eval.jsonl" "$P10" "$TTu/stack/scripts/v8f_mini_eval.py" "$KIN")
if [ "$T0F" = 1 ]; then
  for f in "${T0IN[@]}"; do [ -s "$f" ] || stop "fresh-T0 input missing: $f"; done
  md5chk "$V9/v9_labels_eval139.npz" 6b5c7f207cffc3b7eb3cd527fd433599          # the md5 the box-2 chain asserts for the v9 eval release
  md5chk "$KIN" c92ee56db60b972ec84518a4a73d9f83                               # eval141_kin.npz as the SFRESH fresh evals used it (sfresh/code == the oc export)
  grep -cE 'add_argument\("--ages"' "${T0IN[4]}" | grep -qv '^0$' || stop "REFUSED: the tool tree's v8f_mini_eval.py has no --ages (the fresh-age T0 eval needs it)"
fi
tree_digest() {   # one sha256 over every .py the tool can import (stack/tanitad, taniteval) + the tool
  ( cd "$TTu" && { find stack/tanitad taniteval -name '*.py' -type f; echo stack/scripts/v8f_t1_eval.py; } | LC_ALL=C sort -u | tr '\n' '\0' | xargs -0 sha256sum | sha256sum | cut -c1-64 )
}
keyfiles="stack/scripts/v8f_t1_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/floors_kin.py taniteval/taniteval/four_families.py taniteval/taniteval/ci.py taniteval/tools/t1_eval.py"
reader_sha() { sha256sum < "$(cygpath -u "$1")" | cut -c1-64; }
TOOLD=$(tree_digest)
say "inputs OK (grid, join, tracks md5; 141-clip eval cache); tool tree digest ${TOOLD:0:12}"
write_pins() {
  { echo "tree=$TTm"; echo "digest=$TOOLD"; for f in $keyfiles; do echo "file=$f $(sha256sum < "$TTu/$f" | cut -c1-64)"; done
    echo "reader=orc_read.py $(reader_sha "$READER")"; echo "reader=t1hd_read.py $(reader_sha "$T1HD")"; echo "reader=t1k_read.py $(reader_sha "$T1K")"; echo "reader=PREREG $(reader_sha "$PREREG")"
    [ "$T0F" = 1 ] && t0_pins; return 0; }
}
t0_pins() { local f; for f in "${T0IN[@]}"; do echo "t0in=$(basename "$f") $(sha256sum < "$f" | cut -c1-64)"; done; }
if [ "$DRY" = 1 ]; then
  say "pins that would be written to $OUTm/PINS.txt:"; write_pins | sed 's/^/    /'
else
  if [ ! -e "$OUTu/PINS.txt" ]; then write_pins > "$OUTu/PINS.txt"; say "pins written (tool digest ${TOOLD:0:12}; reader scripts + PREREG sha256)"; fi
  [ "$(grep ^tree= "$OUTu/PINS.txt" | cut -c6-)" = "$TTm" ] || stop "PINS.txt belongs to another TOOL TREE ($(grep ^tree= "$OUTu/PINS.txt")); delete it deliberately to re-pin"
  if [ "$T0F" = 1 ] && ! grep -q '^t0in=' "$OUTu/PINS.txt"; then t0_pins >> "$OUTu/PINS.txt"; say "pins: the fresh-T0 inputs appended"; fi
fi
pin_check() {   # pin_check <when>
  [ "$DRY" = 1 ] && return 0
  [ "$(grep ^digest= "$OUTu/PINS.txt" | cut -c8-)" = "$(tree_digest)" ] || stop "the tool tree changed ($1)"
  local nm
  for nm in orc_read.py:"$READER" t1hd_read.py:"$T1HD" t1k_read.py:"$T1K" PREREG:"$PREREG"; do
    [ "$(grep "^reader=${nm%%:*} " "$OUTu/PINS.txt" | cut -d' ' -f2)" = "$(reader_sha "${nm#*:}")" ] || stop "reader input ${nm%%:*} changed ($1)"
  done
  if [ "$T0F" = 1 ]; then
    local f; for f in "${T0IN[@]}"; do [ "$(grep "^t0in=$(basename "$f") " "$OUTu/PINS.txt" | cut -d' ' -f2)" = "$(sha256sum < "$f" | cut -c1-64)" ] || stop "fresh-T0 input $(basename "$f") changed ($1)"; done
  fi
}
pin_check "start"
cfree_mb() { df -m /c | awk 'NR==2{print $4}'; }

# ------------------------------------------------------------------------------------------------ 1  the rollouts
t1cmd() {   # t1cmd <ckpt> <tag> <limit>  ->  T1CMD (the array IS the command that runs; --dry-run prints it)
  T1CMD=("$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --ckpt "$1" --tag "$2" --tracks "$TRACKS" --eval-cache "$CACHEm" --eval-grid "$GRID" --eval-join "$JOIN" --out-dir "$OUTm"
         --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit "$3" --dump-ticks "$OUTm/ticks" --dump-heads --device cpu)
  [ "$MAXW" -gt 0 ] && T1CMD+=(--max-windows "$MAXW")
  return 0
}
b2count() { if [ -n "${ORCT_B2_OUT:-}" ] && [ -f "$ORCT_B2_OUT" ]; then grep -c '^ZZV8FMINI-STOP' "$ORCT_B2_OUT"; else echo 0; fi; }
B2STOP0=$(b2count)
PULL_SHA=""
wait_pull() {   # wait_pull <arm> <seed>: block until the pull is COMPLETE (sets PULL_SHA); stops on STOP, on a box-2 STOP that appeared after we started, on timeout
  local arm=$1 s=$2 t0 n=0 out rc
  t0=$(date +%s)
  while :; do
    nostop
    out=$("$PY" "$Hm/orct1_pullcheck.py" "$ARTm" "$arm" "$s" 2>&1 < /dev/null); rc=$?
    if [ $rc -eq 0 ]; then PULL_SHA=$(echo "$out" | cut -d' ' -f2); say "pull of $(rundir "$arm" "$s") is COMPLETE: ckpt sha256 ${PULL_SHA:0:16}... ($(echo "$out" | cut -d' ' -f3-))"; return 0; fi
    [ $((n % 30)) -eq 0 ] && say "waiting for the pull of $(rundir "$arm" "$s") ($(( ($(date +%s) - t0) / 60 )) min): $out"
    n=$((n + 1))
    [ "$(b2count)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP after this chain started ($ORCT_B2_OUT): no more pulls are coming ($(grep '^ZZV8FMINI-STOP' "$ORCT_B2_OUT" | tail -1 | cut -c1-200))"
    [ $(( $(date +%s) - t0 )) -gt "$MAXWAIT_S" ] && stop "timeout: the pull of $(rundir "$arm" "$s") is still not complete after $((MAXWAIT_S / 3600)) h ($out)"
    sleep "$POLL"
  done
}
n_dumps() { ls "$OUTu/dump_$1" 2>/dev/null | grep -c '^ep.*\.npz$'; }
rollout_done() {   # rollout_done <tag> <ckpt sha256>: finished, AND by THIS checkpoint (a re-pulled different checkpoint must not be skipped)
  [ -s "$OUTu/t1v8f_$1.json" ] && [ -s "$OUTu/ticks/ticks_$1.npz" ] && [ -s "$OUTu/done_$1.txt" ] || return 1
  [ "$(n_dumps "$1")" -ge "$NDUMP" ] || return 1
  grep -q "\"ckpt_sha256\": \"$2\"" "$OUTu/done_$1.txt" && grep -q "\"tool_digest\": \"$TOOLD\"" "$OUTu/done_$1.txt"
}
t1run() {   # t1run <arm> <seed> <limit> <ckpt sha256>
  local arm=$1 s=$2 lim=$3 sha=$4 tag ck t0 rc att w
  tag=$(tagof "$arm" "$s" "$lim"); ck="$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt"
  if rollout_done "$tag" "$sha"; then say "skip $tag (complete, same checkpoint ${sha:0:12}, same tool)"; return 0; fi
  if [ -e "$OUTu/t1v8f_$tag.json" ] || [ -e "$OUTu/ticks/ticks_$tag.npz" ] || [ -d "$OUTu/dump_$tag" ] || [ -e "$OUTu/done_$tag.txt" ]; then   # an earlier attempt: MOVE, never delete
    mkdir -p "$OUTu/_incomplete_$(date +%m%d%H%M%S)"; w=$(ls -d "$OUTu"/_incomplete_* | tail -1)
    for f in "$OUTu/t1v8f_$tag.json" "$OUTu/ticks/ticks_$tag.npz" "$OUTu/dump_$tag" "$OUTu/done_$tag.txt"; do [ -e "$f" ] && mv "$f" "$w/"; done
    say "moved the output of an earlier attempt of $tag to $w"
  fi
  w=0; while [ "$(cfree_mb)" -lt "$CMIN" ]; do nostop; [ $w -eq 0 ] && say "$tag: waiting for C: >= $CMIN MB free (now $(cfree_mb) MB; the pagefile is on C:)"; w=$((w + 1)); [ $w -ge 120 ] && stop "C: under $CMIN MB free for 2 h"; sleep 60; done
  nostop; pin_check "before $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$(rundir "$arm" "$s")/ckpt_last.pt changed since its pull was verified (before $tag)"
  t1cmd "$ck" "$tag" "$lim"
  for att in 1 2; do
    t0=$(date +%s)
    say "start $tag (attempt $att; ckpt $(rundir "$arm" "$s") ${sha:0:12}, limit $lim): $(fmt "${T1CMD[@]}")"
    "${T1CMD[@]}" > "$LOGu/$tag.out" 2>&1 < /dev/null; rc=$?
    say "end $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
    [ $rc -eq 139 ] && [ $att -eq 1 ] && { say "$tag: native crash (rc 139, the T1-HEADS causal s0 first launch did the same): one retry; the crashed attempt's output is never read"; mv "$LOGu/$tag.out" "$LOGu/$tag.segv.out"; rm -f "$OUTu/done_$tag.txt"; continue; }
    break
  done
  pin_check "after $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$(rundir "$arm" "$s")/ckpt_last.pt changed during $tag"
  { [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ] && [ -s "$OUTu/ticks/ticks_$tag.npz" ] && [ "$(n_dumps "$tag")" -ge "$NDUMP" ]; } || stop "rollout $tag failed (rc $rc; json $([ -s "$OUTu/t1v8f_$tag.json" ] && echo yes || echo NO), ticks $([ -s "$OUTu/ticks/ticks_$tag.npz" ] && echo yes || echo NO), dumps $(n_dumps "$tag")/$NDUMP): $(tail -2 "$LOGu/$tag.out" | tr '\n' ' ' | cut -c1-300)"
  printf '{"tag": "%s", "limit": "%s", "ckpt_sha256": "%s", "tool_digest": "%s", "n_dumps": %s, "wall_s": %s, "smoke": %s}\n' "$tag" "$lim" "$sha" "$TOOLD" "$(n_dumps "$tag")" "$(( $(date +%s) - t0 ))" "$([ "$SMOKE" = 1 ] && echo true || echo false)" > "$OUTu/done_$tag.txt"
}
t0tag() { echo "${PFX}M2b-${1}_s${2}-fresh"; }                    # t0tag <arm> <seed>: the fresh-age T0 eval's --tag (files eval_<tag>.json / .npz)
t0cmd() {   # t0cmd <ckpt> <tag>  ->  T0CMD: v8f_mini_eval.py run --ages fresh, the SFRESH argv (CPU, ~32 s) with this run's checkpoint
  T0CMD=("$PY" "$TTm/stack/scripts/v8f_mini_eval.py" run --ckpt "$1" --tag "$2" --ages fresh --v9-eval "$V9/v9_labels_eval139.npz" --lead-eval "$V9/lead_eval139.npz"
         --speed-eval "$V9/refcv6_speed_max_v8_eval.jsonl" --kin-mode yaw --p10 "$P10" --out-dir "$T0DIR" --eval-cache "$CACHEm" --tracks "$TRACKS" --eval-grid "$GRID" --eval-kin "$KIN" --eval-join "$JOIN" --project --families)
  return 0
}
t0run() {   # t0run <arm> <seed> <ckpt sha256>: only when ORCT_T0_FRESH=1; runs inside the logged lane, so it never adds a third process
  local arm=$1 s=$2 sha=$3 tag ck t0 rc w
  tag=$(t0tag "$arm" "$s"); ck="$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt"
  if [ -s "$OUTu/t0fresh/eval_$tag.json" ] && [ -s "$OUTu/t0fresh/eval_$tag.npz" ] && [ -s "$OUTu/done_t0f_$tag.txt" ] && grep -q "\"ckpt_sha256\": \"$sha\"" "$OUTu/done_t0f_$tag.txt"; then
    say "skip fresh-T0 eval $tag (complete, same checkpoint ${sha:0:12})"; return 0
  fi
  mkdir -p "$OUTu/t0fresh"
  if [ -e "$OUTu/t0fresh/eval_$tag.json" ] || [ -e "$OUTu/t0fresh/eval_$tag.npz" ]; then
    mkdir -p "$OUTu/_incomplete_$(date +%m%d%H%M%S)"; w=$(ls -d "$OUTu"/_incomplete_* | tail -1)
    for f in "$OUTu"/t0fresh/eval_$tag.* "$OUTu"/t0fresh/eval_${tag}_proj.npz "$OUTu/done_t0f_$tag.txt"; do [ -e "$f" ] && mv "$f" "$w/"; done
    say "moved the output of an earlier attempt of the fresh-T0 eval $tag to $w"
  fi
  nostop; pin_check "before $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$(rundir "$arm" "$s")/ckpt_last.pt changed since its pull was verified (before $tag)"
  t0cmd "$ck" "$tag"; t0=$(date +%s)
  say "start fresh-T0 eval $tag: $(fmt "${T0CMD[@]}")"
  "${T0CMD[@]}" > "$LOGu/$tag.out" 2>&1 < /dev/null; rc=$?
  say "end fresh-T0 eval $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
  pin_check "after $tag"
  { [ $rc -eq 0 ] && [ -s "$OUTu/t0fresh/eval_$tag.json" ] && [ -s "$OUTu/t0fresh/eval_$tag.npz" ]; } || stop "fresh-T0 eval $tag failed (rc $rc): $(tail -2 "$LOGu/$tag.out" | tr '\n' ' ' | cut -c1-300)"
  printf '{"tag": "%s", "ckpt_sha256": "%s", "wall_s": %s, "smoke": %s}\n' "$tag" "$sha" "$(( $(date +%s) - t0 ))" "$([ "$SMOKE" = 1 ] && echo true || echo false)" > "$OUTu/done_t0f_$tag.txt"
}
lane() {   # lane <limit>: every run's rollout of that limit, in the PREREG's order, each as soon as its pull is complete (the logged lane also runs the optional fresh-T0 eval of the run)
  local lim=$1 spec arm s
  for spec in "${RUNS[@]}"; do
    arm=${spec%%:*}; s=${spec##*:}
    nostop; wait_pull "$arm" "$s"; t1run "$arm" "$s" "$lim" "$PULL_SHA"
    [ "$T0F" = 1 ] && [ "$lim" = logged ] && t0run "$arm" "$s" "$PULL_SHA"
  done
  say "lane $lim done"
}

# ------------------------------------------------------------------------------------------------ 2  the manifest and the reader
mk_reader_cmd() {   # mk_reader_cmd <m2b sha s0> <m2b sha s1>  ->  RDCMD
  RDCMD=("$PY" -I "$READER" --manifest "$OUTm/runs.json" --out "$OUTm/read/ORC_READ.json" --tool-root "$TTm" --t1hd-read "$T1HD" --t1k-read "$T1K"
         --eval-cache "$CACHEm" --tracks "$TRACKS" --eval-grid "$GRID" --eval-join "$JOIN" --m2b-sha "0=$1" --m2b-sha "1=$2" --prereg "$PREREG" --key-built-p ego_perturb.ego_perturb_p)
}
MANI=("$PY" "$Hm/orct1_manifest.py" manifest --art "$ARTm" --out "$OUTm" --pfx "$PFX" --runs "$RUNSCSV" --n-dump-min "$NDUMP")
M2BS=("$PY" "$Hm/orct1_manifest.py" m2bsha --art "$ARTm" --runs "$RUNSCSV")
[ -n "${ORCT_M2B_SHA_JSON:-}" ] && M2BS+=(--b2-json "$(m "$ORCT_M2B_SHA_JSON")")
[ "$T0F" = 1 ] && MANI+=(--t0-fresh-dir "$T0DIR")
if [ "$SMOKE" = 1 ]; then MANI+=(--smoke-replicate "$REP" --label "SMOKE: ONE stand-in run replicated into all four slots; NOT an O-RECOVER read"); M2BS+=(--smoke-replicate "$REP"); fi

# ------------------------------------------------------------------------------------------------ --dry-run: print and leave
if [ "$DRY" = 1 ]; then
  say "C: free $(cfree_mb) MB (a rollout waits for >= $CMIN MB); threads OMP/MKL $OMP_NUM_THREADS, CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES', max 2 rollout processes (lane causal + lane logged), poll $POLL s, max wait $((MAXWAIT_S / 3600)) h per pull"
  say "box-2 stop watcher: $([ -n "${ORCT_B2_OUT:-}" ] && echo "$ORCT_B2_OUT (ZZV8FMINI-STOP lines now: $B2STOP0)" || echo 'OFF (ORCT_B2_OUT unset)')"
  say "== the four runs in the PREREG's order, and the state of each pull right now"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; say "  $(rundir "$arm" "$s")  $("$PY" "$Hm/orct1_pullcheck.py" "$ARTm" "$arm" "$s" 2>&1 < /dev/null | cut -c1-220)"; done
  say "== rollouts (lane A = causal, lane B = logged; lane order = the run order; each starts when ITS pull is complete; ckpt = <ART>/<run>/ckpt_last.pt)"
  for lim in causal logged; do
    for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}
      t1cmd "$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt" "$(tagof "$arm" "$s" "$lim")" "$lim"; say "  [lane $lim] $(fmt "${T1CMD[@]}")"
    done
  done
  if [ "$T0F" = 1 ]; then
    say "== fresh-T0 eval (ORCT_T0_FRESH=1; runs in the logged lane after that run's logged rollout; the manifest's t0_fresh, K4's comparator; t0 stays box 2's standard eval: Amendment 4)"
    for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; t0cmd "$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt" "$(t0tag "$arm" "$s")"; say "  $(fmt "${T0CMD[@]}")"; done
  else
    say "== fresh-T0 eval: OFF (ORCT_T0_FRESH unset).  The manifest's t0 = box 2's standard (--ages cycle) eval, the PREREG as written; K4 will then FAIL by construction (see HANDOFF)"
  fi
  say "  each writes $OUTm/t1v8f_<tag>.json, $OUTm/ticks/ticks_<tag>.npz, $OUTm/dump_<tag>/ep*.npz (>= $NDUMP), then $OUTm/done_<tag>.txt (checkpoint sha256 + tool digest)"
  say "== manifest: $(fmt "${MANI[@]}" --write "$OUTm/runs.json")"
  "${MANI[@]}" --dry-run | sed 's/^/    /'      # the SAME argv the real run writes with (it carries --t0-fresh-dir: Amendment 4)
  say "== --m2b-sha: $(fmt "${M2BS[@]}")   (the runs' own summary.json / selfcheck.json levers.init_ckpt.sha256, all equal per seed; the READER compares them with the PREREG's 601cb435... / 6104fb15...)"
  mk_reader_cmd '<0: M2b_s0 init sha256>' '<1: M2b_s1 init sha256>'
  say "== reader: $(fmt "${RDCMD[@]}")"
  say "   -> $OUTm/read/ORC_READ.json (rc 0 outcome written | 2 REFUSED, nothing written | 3 UNPLACED); reader stdout/stderr -> $LOGm/orc_read.out"
  say "== markers in $LOGm/chain.out: $DONE_MARK <n>/$(( ${#RUNS[@]} * 2 )) rollouts, reader rc <rc> ...  |  ZZORCT1-STOP <why>.  Nothing was written, launched or waited for."
  echo "ZZORCT1-DRY-OK"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ the real run
say "== lanes: A = --limit causal (decisive), B = --limit logged (K4 only), $(( ${#RUNS[@]} )) runs each, OMP/MKL $OMP_NUM_THREADS threads each, max 2 processes"
lane causal & PA=$!
lane logged & PB=$!
wait $PA $PB
[ -e "$LOGu/STOP" ] && { say "a lane stopped: no DONE"; exit 1; }
pin_check "end"
n=0; for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; for lim in causal logged; do [ -s "$OUTu/done_$(tagof "$arm" "$s" "$lim").txt" ] && n=$((n + 1)); done; done
NEED=$(( ${#RUNS[@]} * 2 ))
[ "$n" -eq "$NEED" ] || stop "$n of $NEED rollouts have a done record"
if [ "$T0F" = 1 ]; then for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; [ -s "$OUTu/done_t0f_$(t0tag "$arm" "$s").txt" ] || stop "no done record for the fresh-T0 eval of $(rundir "$arm" "$s")"; done; fi
say "== all $n rollouts done; manifest"
"${MANI[@]}" --write "$OUTm/runs.json" > "$LOGu/manifest.out" 2>&1 < /dev/null || stop "manifest: $(tr '\n' ' ' < "$LOGu/manifest.out" | cut -c1-400)"
say "$(tail -1 "$LOGu/manifest.out")"
SHAS=$("${M2BS[@]}" 2> "$LOGu/m2bsha.err" < /dev/null) || stop "m2b sha: $(tr '\n' ' ' < "$LOGu/m2bsha.err" | cut -c1-300)"
S0=$(echo "$SHAS" | tr ' ' '\n' | grep '^0=' | cut -c3-); S1=$(echo "$SHAS" | tr ' ' '\n' | grep '^1=' | cut -c3-)
[ ${#S0} -eq 64 ] && [ ${#S1} -eq 64 ] || stop "m2b sha values malformed: '$SHAS'"
mk_reader_cmd "$S0" "$S1"
pin_check "before the reader"
say "== reader: $(fmt "${RDCMD[@]}")"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${RDCMD[@]}" > "$LOGu/orc_read.out" 2>&1 < /dev/null; RRC=$?
say "reader rc=$RRC: $(grep -a -E '"OUTCOME"|REFUSED|UNPLACED|wrote ' "$LOGu/orc_read.out" | head -3 | tr '\n' ' ' | cut -c1-400)"
pin_check "after the reader"
echo "$DONE_MARK $n/$NEED rollouts, reader rc $RRC ($([ $RRC -eq 0 ] && echo 'outcome row written' || { [ $RRC -eq 2 ] && echo 'REFUSED, nothing written' || { [ $RRC -eq 3 ] && echo UNPLACED || echo 'reader crashed'; }; })): $OUTm/read/ORC_READ.json  log $LOGm/orc_read.out" >> "$LOGu/chain.out"
tail -1 "$LOGu/chain.out"
