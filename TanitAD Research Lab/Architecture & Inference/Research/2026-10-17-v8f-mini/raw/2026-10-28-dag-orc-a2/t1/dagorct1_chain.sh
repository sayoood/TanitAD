#!/bin/bash
# PREREG_V8F_DAG_ORC, RE-BASED ON DAG-OACC by Amendment 1 (A1-2026-10-27): the DEV-BOX CPU part, as ONE detached, SEQUENTIAL chain.  ZERO training, ZERO GPU, box 2 is never contacted.  BUILT AND DRY-RUN, NOT LAUNCHED.
# DERIVED from E:/Projects/TanitAD-artifacts/dagoacc_t1/dagoacc_t1_chain.sh (the chain that rolled DAG-OACC out; sections 0a-0c, the rollout function, the done records, the disk gates, the markers) and, for the two K4 lanes,
# from dag2t1/dag2_t1_chain.sh (the chain that ran PREREG_V8F_DAGGER2's --limit logged rollouts and fresh-age T0 evals).  NOTES.md lists, file:line, what was copied verbatim and what differs.
#
#   for each of the four runs, in the box-2 chain's order:   A s0, REP s0, A s1, REP s1     (M2b_s<S>_dagoaccorc / M2b_s<S>_dagoaccorcrep)
#     1. WAIT, by script polling (60 s sleep, no model turns), until the box-2 chain's log (DORT1_B2_OUT, read-only) shows that run's ' pulled to ' line AND a PASSing 'run record' line AND a PASSing K-REACH line;
#        a FAIL line, a ZZDAGOACCORC-CONTROL-FAIL, a ZZV8FMINI-STOP after the pull but before both controls, or ANY new ZZV8FMINI-STOP after this chain started  =>  ZZDAGORCT1-STOP.  A run that failed its controls is never read.
#     2. the pull must be COMPLETE BY CONTENT (dagorct1_lib.py pullcheck): SHA256SUMS.txt lists exactly the 8 names, every file re-hashes, the checkpoint's sha256 equals the sha256 the box-2 log printed, summary done, and the
#        run's own records say it IS this arm and seed (argv / init / bank / w_acc / perturbation / launch-gate token).  The checkpoint is re-hashed against the pulled SHA256SUMS.txt BEFORE and AFTER each step.
#     3. ONE process at a time (never two torch rollouts at once): T1-kin --limit causal (decisive), T1-kin --limit logged (K4 only), then the fresh-age T0 eval (K4's comparator).
#   after the 12 steps: the box-2 chain's ZZV8FMINI-DONE (after its last pull), every pull re-verified, the manifest runs.json (CTRL = the banked DAG-OACC; K-BUILD recorded), then the PRE-REGISTERED reader
#   dag_orc_read.py (sha256-pinned, never edited) in its real-run mode -> <OUT>/read/DAGORC_READ.json.
#
#   bash dagorct1_chain.sh --dry-run     prints every guard result, pull state, the EXACT command of every step (stand-in checkpoints = DAG-OACC's banked ones), the manifest and the reader command; runs NO rollout
#   bash dagorct1_chain.sh --smoke       REAL end-to-end proof of the argv on DAG-OACC_s0's checkpoint: causal + logged with --max-windows (default 8) and the full fresh T0 eval, into <H>/smoke/ (never into OUT)
#   bash dagorct1_chain.sh --write-pins  rewrite PINS.txt (a deliberate act; the chain STOPs on any difference from PINS.txt)
#   bash dagorct1_chain.sh               the real chain.  RESUME = run the same command again (a finished step is skipped by a done record naming THIS checkpoint's sha256, the tool digest and the command's sha256;
#                                        an incomplete earlier attempt is MOVED to <OUT>/_incomplete_*, never deleted).  The previous chain.out is rotated to chain.out.prev_<time> so the waker sees only this run's markers.
# Markers (own log, <LOG>/chain.out, ANCHORED at line start):
#   ZZDAGORCT1-DONE <n>/12 steps, reader rc <rc> ...   finished.  reader rc 0 = an outcome row written | 2 = REFUSED (a control failed; nothing written) | 3 = UNPLACED | 4 = BLOCKED (a required slot incomplete)
#   ZZDAGORCT1-STOP <why>                              refused.  Fix the named cause and re-run the SAME command.  Also touches <LOG>/STOP.
#   ZZDAGORCT1-DRY-OK / ZZDAGORCT1-SMOKE-OK            the dry run's / smoke's end marker.
# Environment (all optional; a WMI-launched process does not inherit the caller's shell, so put them in the .cmd):
#   DORT1_B2_OUT (THE box-2 chain log, one variable; default C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out; the r1 log chain_v8f_dagoaccorc.out is REFUSED)  DORT1_ART (box 2's pull dir = its DAGOACCORC_ART; default E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc)
#   DORT1_OUT (default E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc-t1)  DORT1_LOG  DORT1_TREE (the tool tree DAG-OACC was rolled out with)  DORT1_READ_DIR (the reader's directory)  DORT1_INPUTS  DORT1_TIP_MIRROR
#   DORT1_EXPECT_COMMIT (optional 40-hex: the fixed trainer commit every run's launch-gate token must bind)  DORT1_REFUSE_COMMITS (extra refused commits; the r1 commit e7ea20d1 is always refused)
#   DORT1_COMPOSE_PKG / DORT1_EVIDENCE_COMMIT (where the flag-off bit-identity evidence lives, for K-BUILD; default the launch commit's tree)  DORT1_MAX_WAIT_H (12, per wait)  DORT1_C_MIN_MB (2048)  DORT1_E_MIN_MB (1500)  DORT1_E_ROLL_MB (200)  DORT1_SMOKE_WINDOWS (8)  DORT1_TEST_CFREE_MB / DORT1_TEST_EFREE_MB (--dry-run only)
set -u
shopt -u patsub_replacement 2>/dev/null || true       # bash 5.2: '&' in a replacement string would be expanded ("Architecture & Inference" is in the tool paths)
export PATH="/usr/bin:/mingw64/bin:$PATH"
MODE=real
case "${1:-}" in --dry-run) MODE=dry ;; --smoke) MODE=smoke ;; --write-pins) MODE=pins ;; "") ;; *) echo "usage: dagorct1_chain.sh [--dry-run | --smoke | --write-pins]   (configuration = DORT1_* environment variables, see the header)"; exit 2 ;; esac
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); Hm=$(cygpath -m "$H")
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LIB="$Hm/dagorct1_lib.py"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
AR=/e/Projects/TanitAD-artifacts
TT=${DORT1_TREE:-E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a}
RD=${DORT1_READ_DIR:-E:/Projects/TanitAD-artifacts/dagorc_rebase/read}; READER="$RD/dag_orc_read.py"
READER_SHA=2e5f9f1778e104615d4480f51c6246703c9e3bb8f6c0eeaa3fc0c5621bd8adcf                    # dag_orc_read.py as pre-registered (pinned; never edited)
SF=${DORT1_INPUTS:-$AR/sfresh_inputs}
CTRL_T1=$AR/v8f-dagoacc-t1; CTRL_ART=$AR/v8f-dagoacc                                           # CTRL = DAG-OACC, BANKED (read, never re-run)
ART=${DORT1_ART:-$AR/v8f-dag-orc-oacc}; OUT=${DORT1_OUT:-$AR/v8f-dag-orc-oacc-t1}; LOG=${DORT1_LOG:-$H/logs}
B2OUT=${DORT1_B2_OUT:-C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out}      # THE ONE VARIABLE: every wait keys on this log's pull + PASSing run record + PASSing K-REACH lines (r2 = the re-run at the fixed commit; r1's log is refused)
MIRROR=${DORT1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}
NDUMP=141; POLL=60; PFX=""; MAXW=0; STANDIN=0
RUNS=(dagoaccorc:0 dagoaccorcrep:0 dagoaccorc:1 dagoaccorcrep:1)      # ALWAYS ALL FOUR, in the box-2 chain's order; no run is conditional on another's read
TMPL=$H/argv
if [ "$MODE" = smoke ]; then OUT=$H/smoke/out; LOG=$H/smoke/logs; PFX=SMOKE_; MAXW=${DORT1_SMOKE_WINDOWS:-8}; STANDIN=1; ART=$CTRL_ART; RUNS=(dagoacc:0); fi
if [ "$MODE" = dry ]; then STANDIN=1; fi
m() { cygpath -m "$1"; }
TTu=$(cygpath -u "$TT"); TTm=$(m "$TT"); ARTm=$(m "$ART"); OUTu=$(cygpath -u "$OUT"); OUTm=$(m "$OUT"); LOGu=$(cygpath -u "$LOG"); LOGm=$(m "$LOG"); B2OUTm=$(m "$B2OUT"); RDm=$(m "$RD")
case "$(basename "$B2OUT")" in chain_v8f_dagoaccorc.out) [ "$MODE" = real ] && { echo "ZZDAGORCT1-STOP DORT1_B2_OUT is the r1 log (chain_v8f_dagoaccorc.out): r1 STOPPED on K-REACH and is REFUSED; use the r2 log"; exit 1; } ;; esac
SFu=$(cygpath -u "$SF"); SFm=$(m "$SF"); CTRL_T1m=$(m "$CTRL_T1"); CTRL_ARTm=$(m "$CTRL_ART"); MIRRORm=$(m "$MIRROR")
TOOL="$TTu/stack/scripts/v8f_t1_eval.py"; TOOL0="$TTu/stack/scripts/v8f_mini_eval.py"
GRID="$TTm/$PK/raw/eval_grid_windows.json"; JOIN="$TTm/$PK/raw/join_eval141.npz"; KIN="$TTm/$PK/raw/eval141_kin.npz"; TRACKS="$SFm/inputs/native_tracks.pt"; CACHEm="$SFm/cache"
V9="$SFm/inputs/v9"; P10="$SFm/inputs/p10/p10_prior.pt"; T0DIR="$OUTm/t0fresh"
MAXWAIT_S=$(( ${DORT1_MAX_WAIT_H:-12} * 3600 )); CMIN=${DORT1_C_MIN_MB:-2048}; EMIN=${DORT1_E_MIN_MB:-1500}; EROLL=${DORT1_E_ROLL_MB:-200}; FLOORMB=1024      # FLOORMB = the box-2 chain's 1 GiB floor on E: for every pull
fmt() { local o="" a; for a in "$@"; do case "$a" in *[!A-Za-z0-9_./:=,@%+-]*) o="$o '$a'" ;; *) o="$o $a" ;; esac; done; echo "${o# }"; }
rundir() { echo "M2b_s${2}_${1}"; }                                         # rundir <arm> <seed>
etag() { echo "M2b-${1}_s${2}"; }                                           # box 2's eval tag (eval_<tag>.json)
tstem() { case "$1" in dagoaccorc) echo DAGORC ;; dagoaccorcrep) echo DAGORCREP ;; *) echo DAGOACC ;; esac; }
tagof() { echo "${PFX}$(tstem "$1")_s${2}_m10_${3}_heads"; }                # tagof <arm> <seed> <limit>: the rollout's --tag
t0tag() { echo "${PFX}$(etag "$1" "$2")-fresh"; }                           # the fresh-age T0 eval's --tag
ckpt_of() { if [ "$STANDIN" = 1 ]; then echo "$CTRL_ARTm/M2b_s${2}_dagoacc/ckpt_last.pt"; else echo "$ARTm/$(rundir "$1" "$2")/ckpt_last.pt"; fi; }      # STANDIN = DAG-OACC's banked checkpoints (dry-run / smoke ONLY)
cfree_mb() { if [ -n "${DORT1_TEST_CFREE_MB:-}" ] && [ "$MODE" = dry ]; then echo "$DORT1_TEST_CFREE_MB"; else df -m /c | awk 'NR==2{print $4}'; fi; }
efree_mb() { if [ -n "${DORT1_TEST_EFREE_MB:-}" ] && [ "$MODE" = dry ]; then echo "$DORT1_TEST_EFREE_MB"; else df -m /e | awk 'NR==2{print $4}'; fi; }

if [ "$MODE" = dry ] || [ "$MODE" = pins ]; then
  say() { echo "$(date '+%m-%d %H:%M:%S') DRY: $*"; }
  stop() { say "$*"; echo "ZZDAGORCT1-STOP $*"; exit 1; }
else
  mkdir -p "$LOGu" "$OUTu" "$OUTu/ticks" "$OUTu/argv" "$OUTu/_tmp"
  say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOGu/chain.out"; }
  stop() { say "STOP: $*"; echo "ZZDAGORCT1-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }
  rm -f "$LOGu/STOP"
  # ONE instance only (two instances would run two torch rollouts at once)
  if ! mkdir "$LOGu/chain.lock" 2>/dev/null; then
    op=$(cat "$LOGu/chain.lock/winpid" 2>/dev/null)
    if [ -n "$op" ] && tasklist //FI "PID eq $op" //NH 2>/dev/null | grep -q " $op "; then echo "ZZDAGORCT1-STOP another instance of this chain is running (winpid $op, $LOGu/chain.lock)"; exit 1; fi
    rm -rf "$LOGu/chain.lock"; mkdir "$LOGu/chain.lock" || { echo "ZZDAGORCT1-STOP cannot take $LOGu/chain.lock"; exit 1; }
  fi
  cat /proc/$$/winpid > "$LOGu/chain.lock/winpid" 2>/dev/null; cp "$LOGu/chain.lock/winpid" "$LOGu/chain.winpid" 2>/dev/null
  trap 'rm -rf "$LOGu/chain.lock"' EXIT
  [ -s "$LOGu/chain.out" ] && mv "$LOGu/chain.out" "$LOGu/chain.out.prev_$(date +%m%d%H%M%S)"
  say "=== dagorct1_chain start (mode $MODE; tree $TTm; reader $(m "$READER"); ART $ARTm; OUT $OUTm; box-2 log $B2OUTm; runs ${RUNS[*]})"
fi
nostop() { [ "$MODE" = real ] || [ "$MODE" = smoke ] || return 0; [ -e "$LOGu/STOP" ] && exit 1; return 0; }
stops_now() { "$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" meta 2>/dev/null < /dev/null | tr ' ' '\n' | grep '^stops=' | cut -d= -f2; }
pending_now() { "$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" meta 2>/dev/null < /dev/null | tr ' ' '\n' | grep '^pending=' | cut -d= -f2; }

# ------------------------------------------------------------------------------------------------ 0  gates, tree guards, pins, inputs  (sections 0a-0c of the DAG-OACC chain)
CF0=$(cfree_mb)
[ "$CF0" -ge "$CMIN" ] || stop "C: has $CF0 MB free (< $CMIN MB): the registered gate for a dev-box T1 chain is C: >= 2 GB free (the pagefile is on C:; a full C: killed torch start-ups before)"
say "C: gate: $CF0 MB free >= $CMIN MB"
EF0=$(efree_mb)
[ "$EF0" -ge "$EMIN" ] || stop "E: has $EF0 MB free (< $EMIN MB): the chain writes ~1.3 GB (4 runs x 2 rollouts x ~160 MB + ticks + fresh T0) to $OUTm and the box-2 chain's pulls need a 1 GiB floor on E:; free E: first (PI-only deletes) or set DORT1_OUT to another drive"
say "E: gate: $EF0 MB free >= $EMIN MB"
need_e_mb() { local p=0; [ "$MODE" = real ] && p=$(pending_now); p=${p:-0}; if [ "$p" -gt 0 ]; then echo $(( EROLL + FLOORMB + 100 * p )); else echo "$EROLL"; fi; }      # while box-2 pulls are still to come, leave THEIR 1 GiB floor + ~100 MB each
cmin_wait() { local w=0 what=$1; while [ "$(cfree_mb)" -lt "$CMIN" ]; do nostop; [ $w -eq 0 ] && say "$what: waiting for C: >= $CMIN MB free (now $(cfree_mb) MB)"; w=$((w + 1)); [ $w -ge 120 ] && stop "C: under $CMIN MB free for 2 h"; sleep 60; done; }
emin_wait() { local w=0 what=$1 need; need=$(need_e_mb); while [ "$(efree_mb)" -lt "$need" ]; do nostop; [ $w -eq 0 ] && say "$what: waiting for E: >= $need MB free (now $(efree_mb) MB; $EROLL MB + the box-2 pull floor while pulls are pending); never deletes"; w=$((w + 1)); [ $w -ge 120 ] && stop "E: under $need MB free for 2 h"; sleep 60; need=$(need_e_mb); done; }

[ -f "$TOOL" ] && [ -f "$TOOL0" ] || stop "no tool at $TTm/stack/scripts/v8f_t1_eval.py or v8f_mini_eval.py"
TRN="$TTu/stack/scripts/v8f_mini_train.py"
t_pos=$(grep -cE 'add_argument\("--init-ckpt"' "$TRN"); t_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$TRN")
[ "$t_pos" -ge 1 ] && [ "$t_neg" -eq 0 ] || stop "grep control broken on the trainer of $TTm (positive control --init-ckpt = $t_pos, must be >= 1; negative control = $t_neg, must be 0)"
c_pos=$(grep -cE 'add_argument\("--replan-steps"' "$TOOL"); c_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$TOOL")
[ "$c_pos" -ge 1 ] && [ "$c_neg" -eq 0 ] || stop "grep control broken on $TTm (positive control --replan-steps = $c_pos, must be >= 1; negative control = $c_neg, must be 0)"
n_heads=$(grep -cE 'add_argument\("--dump-heads"' "$TOOL"); n_ticks=$(grep -cE 'add_argument\("--dump-ticks"' "$TOOL"); n_caus=$(grep -cE '^LIMIT_MODES *= *\(.*"causal"' "$TOOL"); n_log=$(grep -cE '^LIMIT_MODES *= *\(.*"logged"' "$TOOL")
n_cfg=$(grep -cE 'M\.V8FMiniConfig\(\*\*ck\["cfg"\]\)' "$TOOL"); n_mw=$(grep -cE 'add_argument\("--max-windows"' "$TOOL"); n_ages=$(grep -cE 'add_argument\("--ages"' "$TOOL0")
say "guards: --dump-heads = $n_heads, --dump-ticks = $n_ticks, causal = $n_caus, logged = $n_log, model built from the checkpoint's own cfg = $n_cfg, --max-windows = $n_mw, v8f_mini_eval --ages = $n_ages  (controls: --replan-steps = $c_pos, trainer --init-ckpt = $t_pos, no-such-flag = $c_neg/$t_neg) in $TTm"
for v in n_heads n_ticks n_caus n_log n_cfg n_mw n_ages; do [ "${!v}" -ge 1 ] || stop "REFUSED: guard $v = ${!v} in $TTm (the controls read $c_pos / $t_pos, so the grep works)"; done
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack")      # OMP/MKL 8 = DAG-OACC's rollouts (same reduction order)
export PYTHONPATH="$TTm/stack;$TTm/taniteval;$TTm/stack/scripts"
[ "$MODE" = dry ] || [ "$MODE" = pins ] || export TMP="$OUTm/_tmp" TEMP="$OUTm/_tmp"        # keep the temp files off C:
tool_help() { "$PY" "$1" run --help 2>&1 < /dev/null | tr -s '[:space:]' ' '; }
HELP=$(tool_help "$TTm/stack/scripts/v8f_t1_eval.py"); echo "$HELP" | grep -q 'usage:' || { say "the tool's --help printed no usage (began: ${HELP:0:200}); one retry in 30 s"; sleep 30; HELP=$(tool_help "$TTm/stack/scripts/v8f_t1_eval.py"); }
echo "$HELP" | grep -q 'usage:' && echo "$HELP" | grep -q -e '--dump-heads' && echo "$HELP" | grep -qE '\{logged,[^}]*causal' && echo "$HELP" | grep -q -e '--max-windows' || stop "REFUSED: the T1 tool's own --help does not agree (began: ${HELP:0:300})"
HELP0=$(tool_help "$TTm/stack/scripts/v8f_mini_eval.py"); echo "$HELP0" | grep -q 'usage:' && echo "$HELP0" | grep -q -e '--ages' && echo "$HELP0" | grep -q -e '--families' || stop "REFUSED: v8f_mini_eval.py's own --help does not agree (began: ${HELP0:0:300})"
say "tool --help agrees (v8f_t1_eval: --dump-heads, --limit {logged,...,causal,...}, --max-windows; v8f_mini_eval: --ages, --families)"

md5chk() { [ "$(md5sum < "$1" 2>/dev/null | cut -c1-32)" = "$2" ] || stop "md5 of $1 is not $2"; }
md5chk "$GRID" 34ed8ac56365eeff72e42581bde2f5c4; md5chk "$JOIN" 532fcd644c0e0835e847b2f1cee011ad; md5chk "$TRACKS" 189f4e68913c0163912a509203963f8d; md5chk "$KIN" c92ee56db60b972ec84518a4a73d9f83
md5chk "$V9/v9_labels_eval139.npz" 6b5c7f207cffc3b7eb3cd527fd433599
for f in "$V9/lead_eval139.npz" "$V9/refcv6_speed_max_v8_eval.jsonl" "$P10" "$READER"; do [ -f "$f" ] || stop "input missing: $f"; done
[ "$(ls "$SFu/cache/eval/clips" 2>/dev/null | wc -l)" = 141 ] || stop "eval cache $CACHEm is not 141 clips (the T1-HEADS inputs are $AR/sfresh_inputs)"
tree_digest() { ( cd "$TTu" && { find stack/tanitad taniteval -name '*.py' -type f; echo stack/scripts/v8f_t1_eval.py; } | LC_ALL=C sort -u | tr '\n' '\0' | xargs -0 sha256sum | sha256sum | cut -c1-64 ); }
fsha() { sha256sum < "$(cygpath -u "$1")" | cut -c1-64; }
keyfiles="stack/scripts/v8f_t1_eval.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/floors_kin.py taniteval/taniteval/four_families.py taniteval/taniteval/ci.py taniteval/tools/t1_eval.py stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_dagger.py"
readerfiles="dag_orc_read.py AMEND1.md PREREG_V8F_DAG_ORC.md P0_RF0.json RC_DAG2.json RC_OACC.json ref/dag2_read.py ref/dag_read.py ref/enumerate_cells.py ref/readA_code/rr_lib.py ref/readA_code/readA.py ref/readA_code/build_report.py ref/b1_eval_lead_block_k30.npz"
ctrlfiles="v8f-dagoacc-t1/runs.json v8f-dagoacc-t1/done_DAGOACC_s0_m10_causal_heads.txt v8f-dagoacc-t1/done_DAGOACC_s1_m10_causal_heads.txt v8f-dagoacc-t1/argv/DAGOACC_s0_m10_causal_heads.argv v8f-dagoacc-t1/argv/DAGOACC_s1_m10_causal_heads.argv v8f-dagoacc-t1/read/DAGOACC_READ.json v8f-dagoacc-t1/PINS.txt v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt"
B2DIR=$(dirname "$B2OUT"); B2DIR=$(dirname "$B2DIR"); B2SH=$B2DIR/$(basename "$B2OUT" .out).sh      # box2_ship/chain_v8f_dagoaccorc_r2.sh beside logs/
b2files="$(basename "$B2SH") v8f_dagoaccorc_chain_checks.py gate_dagoaccorc_assert.py"      # the files whose output / format this chain parses; pinned on first sight in <OUT>/PINS_BOX2.txt
TOOLD=$(tree_digest)
write_pins() {
  echo "# PINS of dagorct1_chain.sh (DAG-ORC re-based on DAG-OACC, dev-box T1-kin chain).  The chain STOPs on ANY difference.  Rewrite only deliberately: bash dagorct1_chain.sh --write-pins"
  echo "tree=$TTm"; echo "digest=$TOOLD"; for f in $keyfiles; do echo "file=$f $(sha256sum < "$TTu/$f" | cut -c1-64)"; done
  echo "reader=$(m "$READER") $(fsha "$READER")"
  for f in $readerfiles; do echo "readerfile=$f $(fsha "$RD/$f")"; done
  for f in $ctrlfiles; do echo "ctrl=$f $(fsha "$AR/$f")"; done
  for f in rollout_causal.tmpl rollout_logged.tmpl t0fresh.tmpl reader.tmpl; do echo "tmpl=$f $(fsha "$TMPL/$f")"; done
  echo "lib=dagorct1_lib.py $(fsha "$LIB")"
  echo "md5=$(basename "$GRID") 34ed8ac56365eeff72e42581bde2f5c4"; echo "md5=$(basename "$JOIN") 532fcd644c0e0835e847b2f1cee011ad"; echo "md5=native_tracks.pt 189f4e68913c0163912a509203963f8d"; echo "md5=$(basename "$KIN") c92ee56db60b972ec84518a4a73d9f83"
}
if [ "$MODE" = pins ]; then write_pins > "$H/PINS.txt"; echo "wrote $H/PINS.txt ($(grep -vc '^#' "$H/PINS.txt") pins)"; exit 0; fi
PINS=$H/PINS.txt; [ -f "$PINS" ] || stop "no PINS.txt at $PINS (bash dagorct1_chain.sh --write-pins writes it)"
pins_verify() {   # every pin recomputed; ONE difference stops the chain
  local k a b got bad=0 n=0
  while IFS=' ' read -r k a b; do
    case "$k" in
      "#"*|"") continue ;;
      tree=*) [ "${k#tree=}" = "$TTm" ] || { say "PIN: tree is $TTm, pinned ${k#tree=}"; bad=$((bad+1)); } ;;
      digest=*) [ "${k#digest=}" = "$TOOLD" ] || { say "PIN: tool tree digest ${TOOLD:0:12} != pinned ${k:7:12}"; bad=$((bad+1)); } ;;
      file=*) got=$(sha256sum < "$TTu/${k#file=}" 2>/dev/null | cut -c1-64); [ "$got" = "$a" ] || { say "PIN: ${k#file=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      reader=*) got=$(fsha "$READER"); [ "$got" = "$a" ] && [ "$got" = "$READER_SHA" ] || { say "PIN: the reader is ${got:0:12}, pinned ${a:0:12} (hard pin ${READER_SHA:0:12})"; bad=$((bad+1)); } ;;
      readerfile=*) got=$(fsha "$RD/${k#readerfile=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: reader file ${k#readerfile=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      ctrl=*) got=$(fsha "$AR/${k#ctrl=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: banked CTRL file ${k#ctrl=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      tmpl=*) got=$(fsha "$TMPL/${k#tmpl=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: argv template ${k#tmpl=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      lib=*) got=$(fsha "$LIB"); [ "$got" = "$a" ] || { say "PIN: dagorct1_lib.py is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      md5=*) ;;      # md5 inputs are checked by md5chk above
      *) say "PIN: unknown pin line '$k'"; bad=$((bad+1)) ;;
    esac
    n=$((n+1))
  done < "$PINS"
  [ $bad -eq 0 ] || stop "PINS.txt: $bad difference(s) (listed above): nothing was started"
  say "PINS.txt verified ($n pins: tool tree digest ${TOOLD:0:12}, the reader ${READER_SHA:0:12}, its 13 support files, the banked CTRL files, the box-2 chain files whose output is parsed, the argv templates, the lib)"
}
pins_verify
b2_pin() {   # b2_pin <when>: the box-2 chain's files (the printer of the run-record / K-REACH lines) are pinned ON FIRST SIGHT in <OUT>/PINS_BOX2.txt; a later difference means the log format may have changed
  [ "$MODE" = real ] || return 0
  local f got pf=$OUTu/PINS_BOX2.txt a k
  for f in $b2files; do
    [ -f "$(cygpath -u "$B2DIR")/$f" ] || continue
    got=$(sha256sum < "$(cygpath -u "$B2DIR")/$f" | cut -c1-64)
    a=$(grep -a "^$f " "$pf" 2>/dev/null | cut -d' ' -f2)
    if [ -z "$a" ]; then echo "$f $got" >> "$pf"; say "box-2 chain file $f pinned on first sight: ${got:0:16}... ($1)"
    elif [ "$a" != "$got" ]; then stop "the box-2 chain file $f changed since it was first seen (${a:0:12} -> ${got:0:12}) ($1): the log lines this chain parses may differ; re-review dagorct1_lib.py boxlog, delete $pf deliberately to re-pin"; fi
  done
}
CTRL_DIG=$(grep '^digest=' "$CTRL_T1/PINS.txt" | head -1 | cut -c8-)
[ "$CTRL_DIG" = "$TOOLD" ] || stop "the tool tree digest ${TOOLD:0:12} is not the digest DAG-OACC's rollouts were made with (${CTRL_DIG:0:12}, $CTRL_T1/PINS.txt): CTRL and the new arms would be scored by two tools"
say "tool identity: the digest ${TOOLD:0:12} IS the digest of the tree that rolled DAG-OACC out ($CTRL_T1/PINS.txt)"
pin_check() { [ "$(tree_digest)" = "$TOOLD" ] || stop "the tool tree changed ($1)"; [ "$(fsha "$READER")" = "$READER_SHA" ] || stop "the reader changed ($1)"; }

# the reader's own static check and its evidence files (cheap): a REFUSED read three hours from now is found now
SC=$("$PY" -I "$(m "$READER")" --static-check 2>&1 < /dev/null) || stop "the reader's --static-check failed: $(echo "$SC" | tr '\n' ' ' | cut -c1-500)"
say "reader --static-check: $(echo "$SC" | tail -1)"
RCE=$("$PY" -I "$LIB" rcevidence --reader "$RDm/dag_orc_read.py" "$RDm/RC_DAG2.json" "$RDm/RC_OACC.json" 2>&1 < /dev/null) || stop "$RCE"
say "$RCE"

# ------------------------------------------------------------------------------------------------ 1  argv templates and the commands (the tool-call arrays ARE the commands; --dry-run prints them)
render() {   # render <template> -> array R  (placeholders @KEY@ from the shell variables below; no '#' lines, no leftover @KEY@)
  R=(); local line
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}; case "$line" in "#"*|"") continue ;; esac
    line=${line//@PY@/"$PY"}; line=${line//@TREE@/"$TTm"}; line=${line//@CKPT@/"$R_CKPT"}; line=${line//@TAG@/"$R_TAG"}; line=${line//@TRACKS@/"$TRACKS"}; line=${line//@CACHE@/"$CACHEm"}
    line=${line//@GRID@/"$GRID"}; line=${line//@JOIN@/"$JOIN"}; line=${line//@KIN@/"$KIN"}; line=${line//@OUT@/"$R_OUT"}; line=${line//@V9@/"$V9"}; line=${line//@P10@/"$P10"}; line=${line//@T0DIR@/"$R_T0DIR"}
    line=${line//@READER@/"$(m "$READER")"}; line=${line//@RDIR@/"$RDm"}
    R+=("$line")
  done < "$1"
  local t; for t in "${R[@]}"; do case "$t" in *@[A-Z0-9_]*@*) stop "template $1: unresolved placeholder in '$t'" ;; esac; done
}
t1cmd() {   # t1cmd <ckpt> <tag> <limit>  ->  T1CMD:  argv/rollout_<limit>.tmpl = DAG-OACC's banked rollout argv except --ckpt, --tag, --out-dir, --dump-ticks (+ --limit for the logged lane)
  R_CKPT=$1; R_TAG=$2; R_OUT=$OUTm; R_T0DIR=$T0DIR; render "$TMPL/rollout_$3.tmpl"; T1CMD=("${R[@]}")
  [ "$MAXW" -gt 0 ] && T1CMD+=(--max-windows "$MAXW")
  [ "${TRACE:-0}" = 1 ] && T1CMD=("$PY" "$(m "$H")/trace_imports.py" "$(m "$H")/smoke/trace_$2.json" "${T1CMD[@]:1}")      # SMOKE ONLY: the same argv, run through the import tracer (records the exact runtime import set)
  return 0
}
t0cmd() {   # t0cmd <ckpt> <tag>  ->  T0CMD:  argv/t0fresh.tmpl = the command that ran for PREREG_V8F_DAGGER2's fresh-age evals
  R_CKPT=$1; R_TAG=$2; R_OUT=$OUTm; R_T0DIR=$T0DIR; render "$TMPL/t0fresh.tmpl"; T0CMD=("${R[@]}")
  [ "${TRACE:-0}" = 1 ] && T0CMD=("$PY" "$(m "$H")/trace_imports.py" "$(m "$H")/smoke/trace_$2.json" "${T0CMD[@]:1}")
  return 0
}
mk_reader_cmd() { R_CKPT=""; R_TAG=""; R_OUT=$OUTm; R_T0DIR=$T0DIR; render "$TMPL/reader.tmpl"; RDCMD=("${R[@]}"); }
# template proofs (the same in --dry-run and a real start): the templates, rendered with the parameters of a run that ALREADY HAPPENED, reproduce its recorded command token for token
tmpl_proof() {
  local saved_out=$OUTm saved_pfx=$PFX saved_maxw=$MAXW; PFX=""; MAXW=0      # the proofs are about the full-run commands (the smoke adds --max-windows after rendering)
  OUTm=$CTRL_T1m; t1cmd "$CTRL_ARTm/M2b_s0_dagoacc/ckpt_last.pt" DAGOACC_s0_m10_causal_heads causal; OUTm=$saved_out
  local want got; want=$(cat "$CTRL_T1/argv/DAGOACC_s0_m10_causal_heads.argv"); got=$(printf '%s\n' "${T1CMD[@]}")
  [ "$want" = "$got" ] || stop "TEMPLATE PROOF FAILED: argv/rollout_causal.tmpl rendered for DAG-OACC s0 differs from $CTRL_T1m/argv/DAGOACC_s0_m10_causal_heads.argv"
  say "template proof 1: rollout_causal.tmpl rendered with DAG-OACC s0's parameters == its banked rollout argv ($(echo "$want" | wc -l) tokens, byte for byte)"
  local d2=/e/Projects/TanitAD-artifacts/dag2t1/logs/chain.out d2t=E:/Projects/TanitAD-artifacts/v8f-dagger2-t1 d2a=E:/Projects/TanitAD-artifacts/v8f-dagger2 line
  OUTm=$d2t; t1cmd "$d2a/M2b_s0_dag2/ckpt_last.pt" DAG2_s0_m10_logged_heads logged; OUTm=$saved_out
  line=$(grep -a 'start DAG2_s0_m10_logged_heads (attempt 1' "$d2" | head -1); line=${line#*): }
  [ "$line" = "$(fmt "${T1CMD[@]}")" ] || stop "TEMPLATE PROOF FAILED: argv/rollout_logged.tmpl rendered for DAGGER2's DAG2 s0 logged lane differs from the command recorded in $d2"
  say "template proof 2: rollout_logged.tmpl rendered with DAGGER2's DAG2 s0 parameters == the logged-lane command that RAN (dag2t1/logs/chain.out)"
  local saved_t0=$T0DIR; T0DIR=$d2t/t0fresh; t0cmd "$d2a/M2b_s0_dag2/ckpt_last.pt" M2b-dag2_s0-fresh; T0DIR=$saved_t0
  line=$(grep -a 'start fresh-T0 eval M2b-dag2_s0-fresh:' "$d2" | head -1); line=${line#*: }
  [ "$line" = "$(fmt "${T0CMD[@]}")" ] || stop "TEMPLATE PROOF FAILED: argv/t0fresh.tmpl rendered for DAGGER2's DAG2 s0 differs from the fresh-T0 command recorded in $d2"
  say "template proof 3: t0fresh.tmpl rendered with DAGGER2's DAG2 s0 parameters == the fresh-age T0 command that RAN (dag2t1/logs/chain.out)"
  PFX=$saved_pfx; MAXW=$saved_maxw
}
tmpl_proof
TRACE=0; [ "$MODE" = smoke ] && TRACE=1      # the smoke (and only the smoke) runs both tools through trace_imports.py, AFTER the template proofs (which compare un-wrapped commands)

# ------------------------------------------------------------------------------------------------ 2  one step = one process.  Done records name the checkpoint, the tool digest and the command.
n_dumps() { ls "$OUTu/dump_$1" 2>/dev/null | grep -c '^ep.*\.npz$'; }
NEEDD=$NDUMP; [ "$MAXW" -gt 0 ] && NEEDD=1
cmd_sha() { printf '%s\0' "$@" | sha256sum | cut -c1-64; }
rollout_done() {   # rollout_done <tag> <ckpt sha256> <cmd sha>
  [ -s "$OUTu/t1v8f_$1.json" ] && [ -s "$OUTu/ticks/ticks_$1.npz" ] && [ -s "$OUTu/done_$1.txt" ] && [ -s "$OUTu/argv/$1.argv" ] || return 1
  [ "$(n_dumps "$1")" -ge "$NEEDD" ] || return 1
  grep -q "\"ckpt_sha256\": \"$2\"" "$OUTu/done_$1.txt" && grep -q "\"tool_digest\": \"$TOOLD\"" "$OUTu/done_$1.txt" && grep -q "\"cmd_sha256\": \"$3\"" "$OUTu/done_$1.txt"
}
t1run() {   # t1run <arm> <seed> <limit> <ckpt sha256>
  local arm=$1 s=$2 lim=$3 sha=$4 tag ck t0 rc att w f cs
  tag=$(tagof "$arm" "$s" "$lim"); ck=$(ckpt_of "$arm" "$s")
  t1cmd "$ck" "$tag" "$lim"; cs=$(cmd_sha "${T1CMD[@]}")
  if rollout_done "$tag" "$sha" "$cs"; then say "skip $tag (complete, same checkpoint ${sha:0:12}, same tool, same command)"; return 0; fi
  if [ -e "$OUTu/t1v8f_$tag.json" ] || [ -e "$OUTu/ticks/ticks_$tag.npz" ] || [ -d "$OUTu/dump_$tag" ] || [ -e "$OUTu/done_$tag.txt" ] || [ -e "$OUTu/argv/$tag.argv" ]; then   # an earlier attempt: MOVE, never delete
    mkdir -p "$OUTu/_incomplete_$(date +%m%d%H%M%S)"; w=$(ls -d "$OUTu"/_incomplete_* | tail -1)
    for f in "$OUTu/t1v8f_$tag.json" "$OUTu/ticks/ticks_$tag.npz" "$OUTu/dump_$tag" "$OUTu/done_$tag.txt" "$OUTu/argv/$tag.argv"; do [ -e "$f" ] && mv "$f" "$w/"; done
    say "moved the output of an earlier attempt of $tag to $w"
  fi
  cmin_wait "$tag"; emin_wait "$tag"
  nostop; pin_check "before $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$ck changed since its pull was verified (before $tag)"
  mkdir -p "$OUTu/argv"; printf '%s\n' "${T1CMD[@]}" > "$OUTu/argv/$tag.argv"
  for att in 1 2; do
    t0=$(date +%s)
    say "start $tag (attempt $att; ckpt $(basename "$(dirname "$ck")") ${sha:0:12}, limit $lim): $(fmt "${T1CMD[@]}")"
    "${T1CMD[@]}" > "$LOGu/$tag.out" 2>&1 < /dev/null; rc=$?
    say "end $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
    [ $rc -eq 139 ] && [ $att -eq 1 ] && { say "$tag: native crash (rc 139): one retry; the crashed attempt's output is never read"; mv "$LOGu/$tag.out" "$LOGu/$tag.segv.out"; rm -f "$OUTu/done_$tag.txt"; continue; }
    break
  done
  pin_check "after $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$ck changed during $tag"
  { [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ] && [ -s "$OUTu/ticks/ticks_$tag.npz" ] && [ "$(n_dumps "$tag")" -ge "$NEEDD" ]; } || stop "rollout $tag failed (rc $rc; json $([ -s "$OUTu/t1v8f_$tag.json" ] && echo yes || echo NO), ticks $([ -s "$OUTu/ticks/ticks_$tag.npz" ] && echo yes || echo NO), dumps $(n_dumps "$tag")/$NEEDD): $(tail -2 "$LOGu/$tag.out" | tr '\n' ' ' | cut -c1-300)"
  printf '{"tag": "%s", "limit": "%s", "ckpt_sha256": "%s", "ckpt_sha256_before": "%s", "ckpt_sha256_after": "%s", "tool_digest": "%s", "cmd_sha256": "%s", "n_dumps": %s, "wall_s": %s, "smoke": %s, "max_windows": %s}\n' \
    "$tag" "$lim" "$sha" "$sha" "$sha" "$TOOLD" "$cs" "$(n_dumps "$tag")" "$(( $(date +%s) - t0 ))" "$([ "$MODE" = smoke ] && echo true || echo false)" "$([ "$MAXW" -gt 0 ] && echo "$MAXW" || echo null)" > "$OUTu/done_$tag.txt"
}
t0run() {   # t0run <arm> <seed> <ckpt sha256>: the fresh-age T0 eval (K4's comparator)
  local arm=$1 s=$2 sha=$3 tag ck t0 rc w f cs
  tag=$(t0tag "$arm" "$s"); ck=$(ckpt_of "$arm" "$s")
  t0cmd "$ck" "$tag"; cs=$(cmd_sha "${T0CMD[@]}")
  if [ -s "$OUTu/t0fresh/eval_$tag.json" ] && [ -s "$OUTu/t0fresh/eval_$tag.npz" ] && [ -s "$OUTu/done_t0f_$tag.txt" ] && grep -q "\"ckpt_sha256\": \"$sha\"" "$OUTu/done_t0f_$tag.txt" && grep -q "\"cmd_sha256\": \"$cs\"" "$OUTu/done_t0f_$tag.txt"; then
    say "skip fresh-T0 eval $tag (complete, same checkpoint ${sha:0:12}, same command)"; return 0
  fi
  mkdir -p "$OUTu/t0fresh"
  if [ -e "$OUTu/t0fresh/eval_$tag.json" ] || [ -e "$OUTu/t0fresh/eval_$tag.npz" ]; then
    mkdir -p "$OUTu/_incomplete_$(date +%m%d%H%M%S)"; w=$(ls -d "$OUTu"/_incomplete_* | tail -1)
    for f in "$OUTu"/t0fresh/eval_$tag.* "$OUTu"/t0fresh/eval_${tag}_proj.npz "$OUTu/done_t0f_$tag.txt"; do [ -e "$f" ] && mv "$f" "$w/"; done
    say "moved the output of an earlier attempt of the fresh-T0 eval $tag to $w"
  fi
  cmin_wait "$tag"; emin_wait "$tag"
  nostop; pin_check "before $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$ck changed since its pull was verified (before $tag)"
  t0=$(date +%s)
  say "start fresh-T0 eval $tag: $(fmt "${T0CMD[@]}")"
  "${T0CMD[@]}" > "$LOGu/$tag.out" 2>&1 < /dev/null; rc=$?
  say "end fresh-T0 eval $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
  pin_check "after $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$ck changed during $tag"
  { [ $rc -eq 0 ] && [ -s "$OUTu/t0fresh/eval_$tag.json" ] && [ -s "$OUTu/t0fresh/eval_$tag.npz" ]; } || stop "fresh-T0 eval $tag failed (rc $rc): $(tail -2 "$LOGu/$tag.out" | tr '\n' ' ' | cut -c1-300)"
  printf '{"tag": "%s", "ckpt_sha256": "%s", "tool_digest": "%s", "cmd_sha256": "%s", "wall_s": %s, "smoke": %s}\n' "$tag" "$sha" "$TOOLD" "$cs" "$(( $(date +%s) - t0 ))" "$([ "$MODE" = smoke ] && echo true || echo false)" > "$OUTu/done_t0f_$tag.txt"
}

# ------------------------------------------------------------------------------------------------ 3  waiting for a run: the box-2 chain's LOG says PASS, then the pull is verified by content
B2STOP0=$(stops_now); B2STOP0=${B2STOP0:-0}
PULL_SHA=""
LAUNCH_COMMIT=$(cat "$OUTu/LAUNCH_COMMIT.txt" 2>/dev/null)
GATED="stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py"      # the model / config the checkpoint is built from and the data module the harness calls: a trainer change reaching either is a STOP
commit_gate() {   # commit_gate <arm> <seed>: all four runs share ONE launch commit; it is not a refused one; the rollout harness is unchanged at it
  local g=$(rundir "$1" "$2") c out rc
  c=$("$PY" -I "$LIB" tokencommit --dir "$ARTm/$g" 2>&1 < /dev/null) || stop "$g: launch-gate token: $c"
  [ -z "${DORT1_EXPECT_COMMIT:-}" ] || [ "$c" = "$DORT1_EXPECT_COMMIT" ] || stop "$g was built at ${c:0:12}, DORT1_EXPECT_COMMIT is ${DORT1_EXPECT_COMMIT:0:12}"
  if [ -n "$LAUNCH_COMMIT" ]; then [ "$c" = "$LAUNCH_COMMIT" ] || stop "$g was built at ${c:0:12}, the earlier run(s) at ${LAUNCH_COMMIT:0:12}: one launch commit is required"; return 0; fi
  LAUNCH_COMMIT=$c; echo "$c" > "$OUTu/LAUNCH_COMMIT.txt"
  say "launch commit (from $g's token): $c  (not the refused r1 commit e7ea20d1; later runs must bind the same)"
  out=$("$PY" -I "$LIB" rolloutcode --tree "$TTm" --git-dir "$MIRRORm" --launch "$c" --gate $GATED 2>&1 < /dev/null); rc=$?
  echo "$out" | while IFS= read -r l; do say "  rolloutcode: $l"; done
  [ $rc -eq 0 ] || stop "ROLLOUT-CODE: the launch commit ${c:0:12} changes code the T1 / T0 harness reaches in $GATED, or it is not in the mirror $MIRRORm (rc $rc): the unchanged tool tree is no longer the code the checkpoints were trained with. Decide, then re-run."
  out=$("$PY" -I "$LIB" kbuild --art "$ARTm" --git-dir "$MIRRORm" --launch "$c" ${DORT1_COMPOSE_PKG:+--compose-pkg "$DORT1_COMPOSE_PKG"} ${DORT1_EVIDENCE_COMMIT:+--evidence-commit "$DORT1_EVIDENCE_COMMIT"} 2>&1 < /dev/null)
  say "  K-BUILD early status: $(echo "$out" | "$PY" -I -c "
import json,sys
try:
    b=json.load(sys.stdin); print('changed files', sorted({h['file'].split('/')[-1] for h in b['changed']}), '| hunks', len(b['changed']), '| bit_identity_test', sorted({h['bit_identity_test'][:60] for h in b['changed']}), '| default_off', sorted({h['default_off'] for h in b['changed']}))
except Exception as e:
    print('NOT BUILT YET (the manifest step will REFUSE until the flag-off bit-identity evidence is in the mirror: set DORT1_EVIDENCE_COMMIT / DORT1_COMPOSE_PKG):', str(e)[:200])
" 2>&1 | cut -c1-400)"
}
wait_run() {   # wait_run <arm> <seed>: sets PULL_SHA.  STOPs on a FAIL, a WRONG pull, a new box-2 STOP, a timeout.
  local arm=$1 s=$2 g t0 n=0 out rc pc prc lsha tag
  g=$(rundir "$arm" "$s"); tag=$(etag "$arm" "$s"); t0=$(date +%s); local aw0=0
  while :; do
    nostop
    [ -f "$B2OUT" ] && b2_pin "waiting for $g"
    out=$("$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" run "$arm" "$s" 2>&1 < /dev/null); rc=$?
    [ $rc -eq 3 ] && stop "the box-2 chain's record of $g FAILED a control (a run that failed its controls is never read): $out"
    case "$out" in "WAIT AWAITING-CONTROLS"*) [ $aw0 -eq 0 ] && aw0=$(date +%s); [ $(( $(date +%s) - aw0 )) -gt 900 ] && stop "the log shows $g pulled but no PASSing run-record + K-REACH lines 15 min later ($out): the line format may have changed; re-review dagorct1_lib.py boxlog" ;; *) aw0=0 ;; esac
    if [ $rc -eq 0 ]; then
      lsha=$(echo "$out" | cut -d' ' -f2)
      pc=$("$PY" -I "$LIB" pullcheck --dir "$ARTm/$g" --tag "$tag" --identity "$arm" "$s" --expect-sha "$lsha" 2>&1 < /dev/null); prc=$?
      if [ $prc -eq 0 ]; then PULL_SHA=$(echo "$pc" | cut -d' ' -f2); commit_gate "$arm" "$s"; say "$g: the box-2 log shows pulled + run record PASS + K-REACH PASS; the pull is COMPLETE by content: ckpt sha256 ${PULL_SHA:0:16}... == the pull line of $(basename "$B2OUT") == SHA256SUMS.txt (re-hashed from the file) ($(echo "$pc" | cut -d' ' -f3-))"; return 0; fi
      [ $prc -eq 3 ] && stop "the pulled $g is WRONG, waiting cannot fix it: $pc"
      out="the log shows PASS but the pull is not complete yet: $pc"
    fi
    [ $((n % 30)) -eq 0 ] && say "waiting for $g ($(( ($(date +%s) - t0) / 60 )) min): $out"
    n=$((n + 1))
    [ "$(stops_now)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP after this chain started ($B2OUTm): no more pulls are coming ($(grep -a '^ZZV8FMINI-STOP' "$B2OUT" | tail -1 | cut -c1-200))"
    [ $(( $(date +%s) - t0 )) -gt "$MAXWAIT_S" ] && stop "timeout: $g is still not complete after $((MAXWAIT_S / 3600)) h ($out)"
    sleep "$POLL"
  done
}
step_run() {   # step_run <arm> <seed>: this run's three steps, sequentially
  local arm=$1 s=$2
  nostop; wait_run "$arm" "$s"
  t1run "$arm" "$s" causal "$PULL_SHA"
  t1run "$arm" "$s" logged "$PULL_SHA"
  t0run "$arm" "$s" "$PULL_SHA"
  say "run $(rundir "$arm" "$s"): causal, logged and fresh-T0 steps done"
}

# ------------------------------------------------------------------------------------------------ 4  after the 12 steps: the box-2 DONE marker, every pull re-verified, the manifest, the reader
b2_done_wait() {
  local t0=$(date +%s) n=0
  while :; do
    "$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" done < /dev/null > /dev/null 2>&1 && { say "box-2 chain DONE after its last pull: $(grep -a '^ZZV8FMINI-DONE' "$B2OUT" | tail -1 | cut -c1-200)"; return 0; }
    nostop
    [ "$(stops_now)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP before its DONE ($B2OUTm): $(grep -a '^ZZV8FMINI-STOP' "$B2OUT" | tail -1 | cut -c1-200)"
    [ $((n % 30)) -eq 0 ] && say "waiting for the box-2 chain's ZZV8FMINI-DONE ($(( ($(date +%s) - t0) / 60 )) min)"
    n=$((n + 1)); [ $(( $(date +%s) - t0 )) -gt 7200 ] && stop "no ZZV8FMINI-DONE in $B2OUTm 2 h after the last pull"
    sleep "$POLL"
  done
}
verify_pulls() {   # every pull is STILL complete, the log STILL says PASS, and the checkpoint is STILL the one every rollout scored
  local spec arm s out rc sha rec pc
  for spec in "${RUNS[@]}"; do
    arm=${spec%%:*}; s=${spec##*:}
    out=$("$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" run "$arm" "$s" 2>&1 < /dev/null) || stop "after the rollouts the box-2 log no longer shows $(rundir "$arm" "$s") as PASS: $out"
    pc=$("$PY" -I "$LIB" pullcheck --dir "$ARTm/$(rundir "$arm" "$s")" --tag "$(etag "$arm" "$s")" --identity "$arm" "$s" --expect-sha "$(echo "$out" | cut -d' ' -f2)" 2>&1 < /dev/null) || stop "after the rollouts the pull of $(rundir "$arm" "$s") is no longer COMPLETE: $pc"
    sha=$(echo "$pc" | cut -d' ' -f2)
    for rec in "done_$(tagof "$arm" "$s" causal).txt" "done_$(tagof "$arm" "$s" logged).txt" "done_t0f_$(t0tag "$arm" "$s").txt"; do
      grep -q "\"ckpt_sha256\": \"$sha\"" "$OUTu/$rec" || stop "$rec does not name the pulled checkpoint $sha"
    done
  done
}

# ------------------------------------------------------------------------------------------------ --dry-run: print and leave
if [ "$MODE" = dry ]; then
  say "C: free $(cfree_mb) MB (>= $CMIN MB at start and before every step); E: free $(efree_mb) MB (>= $EMIN MB at start; before every step $EROLL MB, plus the box-2 pull floor $FLOORMB MB + ~100 MB per pull still pending: now pending=$(pending_now))"
  say "threads: OMP/MKL $OMP_NUM_THREADS (= DAG-OACC's rollouts), CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES', ONE torch process at a time (sequential), poll $POLL s, max wait $((MAXWAIT_S / 3600)) h per wait; outputs under $OUTm (nothing on C:)"
  say "box-2 log: $B2OUTm  ($([ -f "$B2OUT" ] && wc -l < "$B2OUT" || echo "absent, 0") lines; ZZV8FMINI-STOP lines now: $B2STOP0; $("$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" meta < /dev/null))"
  say "== the four runs in order, the box-2 log's state of each, and the pull's state (a pull directory that is not there yet reads INCOMPLETE)"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; g=$(rundir "$arm" "$s")
    say "  $g  log: $("$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" run "$arm" "$s" 2>&1 < /dev/null | cut -c1-160)  | pull: $("$PY" -I "$LIB" pullcheck --dir "$ARTm/$g" --tag "$(etag "$arm" "$s")" --identity "$arm" "$s" 2>&1 < /dev/null | cut -c1-200)"
  done
  for rd in "$ART"/*_r1_kreach_stop; do [ -d "$rd" ] && say "  REFUSED r1 directory present (ignored, never read): $(m "$rd")"; done
  c1=$("$PY" -I "$LIB" tokencommit --dir "$ARTm/$(rundir dagoaccorc 0)" 2>&1 < /dev/null | head -1)
  say "== the launch commit the pulled tokens bind NOW: ${c1:-none}  (REFUSED = the r1 commit e7ea20d1, whose A s0 failed K-REACH; the r2 pulls must bind ONE other commit, DORT1_EXPECT_COMMIT optionally names it)"
  say "== does a trainer change reach the rollout harness?  rollout code (static import closure of v8f_t1_eval.py + v8f_mini_eval.py + t1_eval.py in the tool tree) vs the r1 build e7ea20d1, shown as the worked example; the real chain runs this gate on the r2 commit after the first pull and STOPs if the gated files ($GATED) are touched:"
  "$PY" -I "$LIB" rolloutcode --tree "$TTm" --git-dir "$MIRRORm" --launch e7ea20d1287599409d1758971ded62b2fc4ef375 --gate $GATED < /dev/null 2>&1 | while IFS= read -r l; do say "    $l"; done
  say "== STAND-INS for this dry run: every step below uses DAG-OACC's BANKED checkpoint (v8f-dagoacc/M2b_s<S>_dagoacc) in place of the run it will score; its sha256 is verified against the pulled SHA256SUMS.txt and the pinned value"
  for s in 0 1; do
    d=$CTRL_ARTm/M2b_s${s}_dagoacc; sa=$("$PY" -I "$LIB" pullcheck --dir "$d" --tag "M2b-dagoacc_s${s}" 2>&1 < /dev/null); sb=$(sha256sum < "$(cygpath -u "$d/ckpt_last.pt")" | cut -c1-64)
    pin=$(grep -a -c "$sb" "$H/PINS.txt"); lst=$(grep -a ' \*\?ckpt_last.pt$' "$(cygpath -u "$d")/SHA256SUMS.txt" | cut -c1-64)
    [ "$sb" = "$lst" ] || stop "stand-in s$s: ckpt_last.pt $sb != its SHA256SUMS.txt $lst"
    say "  stand-in s$s: $(echo "$sa" | cut -c1-120) | re-hashed ${sb:0:16}... == SHA256SUMS.txt == the pull check's; named in PINS.txt: $pin line(s)"
  done
  say "== the 12 steps that WOULD run, sequentially, in this order (ckpt = the stand-in; the real ckpt is <ART>/<run>/ckpt_last.pt, hash-verified before and after every step)"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; ck=$(ckpt_of "$arm" "$s")
    say "-- $(rundir "$arm" "$s")  (real ckpt: $ARTm/$(rundir "$arm" "$s")/ckpt_last.pt)"
    t1cmd "$ck" "$(tagof "$arm" "$s" causal)" causal; say "  [1 causal ] $(fmt "${T1CMD[@]}")"
    t1cmd "$ck" "$(tagof "$arm" "$s" logged)" logged; say "  [2 logged ] $(fmt "${T1CMD[@]}")"
    t0cmd "$ck" "$(t0tag "$arm" "$s")"; say "  [3 T0fresh] $(fmt "${T0CMD[@]}")"
  done
  say "  each rollout writes $OUTm/t1v8f_<tag>.json, ticks/ticks_<tag>.npz, dump_<tag>/ep*.npz (>= $NDUMP), argv/<tag>.argv, then done_<tag>.txt (ckpt sha256 before/after, tool digest, command sha256); each fresh eval writes t0fresh/eval_<tag>.{json,npz} and done_t0f_<tag>.txt"
  say "== after the 12 steps: box-2 ZZV8FMINI-DONE (after the last pull), every pull re-verified, then the manifest"
  say "== manifest (CTRL = the banked DAG-OACC: $CTRL_T1m/runs.json arms.DAGOACC; K-BUILD from the four launch-gate tokens + git objects in $MIRRORm), exactly as it will be written (excerpt):"
  "$PY" -I "$LIB" manifest --art "$ARTm" --out "$OUTm" --ctrl-runs "$CTRL_T1m/runs.json" --git-dir "$MIRRORm" --dry-run < /dev/null 2>&1 | python -c "
import json,sys
m=json.load(sys.stdin); b=m.get('build',{})
print('  arms:', {k: sorted(v) for k,v in m['arms'].items()})
print('  A s0 entry:', json.dumps(m['arms']['A']['0'])[:900])
print('  K-BUILD: launch', b.get('launch_commit'), 'base', b.get('base_commit'), '| files', {f.split('/')[-1]: (v['sha256_launch'][:12], v['sha256_base'][:12]) for f,v in b.get('files',{}).items()})
print('  K-BUILD changed hunks:', len(b.get('changed',[])), '| bit_identity_test', sorted({h['bit_identity_test'] for h in b.get('changed',[])}), '| default_off', sorted({h['default_off'] for h in b.get('changed',[])}), '| NOT_BUILT:', b.get('NOT_BUILT'))
" | while IFS= read -r l; do say "$l"; done
  mk_reader_cmd; say "== reader (pre-registered, sha256 ${READER_SHA:0:12}, real-run mode; OMP/MKL 2 threads, as DAG-OACC's chain): $(fmt "${RDCMD[@]}")"
  say "   -> $OUTm/read/DAGORC_READ.json (rc 0 outcome written | 2 REFUSED, nothing written | 3 UNPLACED | 4 BLOCKED, a required slot incomplete); reader stdout/stderr -> $LOGm/dagorc_read.out"
  say "== pre-flight, run now (reads banked inputs only, ~15 s): the reader's own --dry-run (static + K-P0 + CTRL's run-record controls)"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PY" -I "$(m "$READER")" --dry-run 2>&1 < /dev/null | sed 's/^/    /' | tail -24
  say "== markers in $LOGm/chain.out (anchored): ZZDAGORCT1-DONE <n>/$(( ${#RUNS[@]} * 3 )) steps, reader rc <rc> ...  |  ZZDAGORCT1-STOP <why>.  NOTHING was run, written, launched or waited for (no rollout)."
  echo "ZZDAGORCT1-DRY-OK"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ --smoke: a REAL end-to-end run of the argv on DAG-OACC_s0's checkpoint, a handful of windows, into <H>/smoke/ only
if [ "$MODE" = smoke ]; then
  sha=$(sha256sum < "$(cygpath -u "$(ckpt_of dagoacc 0)")" | cut -c1-64)
  [ "$sha" = 67265ec1666fae3a6d51e924ee06dfb68b2cd0630afc62e42f3a46182b8bc45f ] || stop "smoke: the stand-in checkpoint is not DAG-OACC_s0's banked one ($sha)"
  say "SMOKE: stand-in = DAG-OACC_s0's banked checkpoint ${sha:0:16}...; causal and logged lanes with --max-windows $MAXW, the fresh T0 eval in full; outputs ONLY under $OUTm (never the chain's OUT); NOT a read"
  t1run dagoacc 0 causal "$sha"; t1run dagoacc 0 logged "$sha"; t0run dagoacc 0 "$sha"
  say "SMOKE json check: $("$PY" -I -c "
import json,sys
for t in ('SMOKE_DAGOACC_s0_m10_causal_heads','SMOKE_DAGOACC_s0_m10_logged_heads'):
    j=json.load(open('$OUTm/t1v8f_'+t+'.json'))['t1v8f']; print(t, {k: j.get(k) for k in ('limit','replan_steps','nav','scene','schedule','max_windows','n_windows')})
j=json.load(open('$OUTm/t0fresh/eval_SMOKE_M2b-dagoacc_s0-fresh.json')); print('fresh T0 ages', j.get('ages'))
")"
  echo "ZZDAGORCT1-SMOKE-OK steps=3 ckpt=${sha:0:16} max_windows=$MAXW out=$OUTm" | tee -a "$LOGu/chain.out"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ the real run
[ -f "$B2OUT" ] && say "box-2 log present" || say "the box-2 log $B2OUTm does not exist yet (the r2 chain has not started): waiting for it, by script polling"
say "== sequential: ${#RUNS[@]} runs x (causal, logged, fresh T0) = $(( ${#RUNS[@]} * 3 )) steps, one process at a time, OMP/MKL $OMP_NUM_THREADS threads; box-2 ZZV8FMINI-STOP lines already in the log: $B2STOP0"
for spec in "${RUNS[@]}"; do step_run "${spec%%:*}" "${spec##*:}"; done
pin_check "end"
n=0; for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; for r in "done_$(tagof "$arm" "$s" causal).txt" "done_$(tagof "$arm" "$s" logged).txt" "done_t0f_$(t0tag "$arm" "$s").txt"; do [ -s "$OUTu/$r" ] && n=$((n + 1)); done; done
NEED=$(( ${#RUNS[@]} * 3 ))
[ "$n" -eq "$NEED" ] || stop "$n of $NEED steps have a done record"
say "== all $n steps done; box-2 DONE marker"
b2_done_wait
verify_pulls
say "== manifest"
"$PY" -I "$LIB" manifest --art "$ARTm" --out "$OUTm" --ctrl-runs "$CTRL_T1m/runs.json" --git-dir "$MIRRORm" --n-dump-min "$NDUMP" --write "$OUTm/runs.json" > "$LOGu/manifest.out" 2>&1 < /dev/null || stop "manifest: $(tr '\n' ' ' < "$LOGu/manifest.out" | cut -c1-700)"
say "$(tail -1 "$LOGu/manifest.out")"
mk_reader_cmd
pin_check "before the reader"
say "== reader: $(fmt "${RDCMD[@]}")"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${RDCMD[@]}" > "$LOGu/dagorc_read.out" 2>&1 < /dev/null; RRC=$?
say "reader rc=$RRC: $(grep -a -E '"OUTCOME"|REFUSED|BLOCKED|UNPLACED|wrote ' "$LOGu/dagorc_read.out" | head -3 | tr '\n' ' ' | cut -c1-400)"
pin_check "after the reader"
echo "ZZDAGORCT1-DONE $n/$NEED steps, reader rc $RRC ($(case $RRC in 0) echo 'outcome row written' ;; 2) echo 'REFUSED, nothing written' ;; 3) echo 'UNPLACED, written' ;; 4) echo 'BLOCKED, a required slot incomplete' ;; *) echo 'reader crashed' ;; esac)): $OUTm/read/DAGORC_READ.json  log $LOGm/dagorc_read.out" >> "$LOGu/chain.out"
tail -1 "$LOGu/chain.out"
