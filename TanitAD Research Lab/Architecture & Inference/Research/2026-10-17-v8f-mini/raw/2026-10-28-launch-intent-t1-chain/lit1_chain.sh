#!/bin/bash
# PREREG_V8F_LAUNCH_INTENT (draft v2, MM_REDESIGN_2026-10-28): the DEV-BOX CPU part, as ONE detached, SEQUENTIAL chain.  ZERO training, ZERO GPU, box 2 is never contacted by this file (lit1_selftest scans it for remote-shell commands and finds none).
# BUILT, DRY-RUN AND SELF-TESTED, NOT LAUNCHED.  Adapted from E:/Projects/TanitAD-artifacts/dagorct1/dagorct1_chain.sh (read-only): same shape, four differences (NOTES.md): the box-2 log is chain_v8f_launchintent.out and a run is READY
# on its pull line + a PASSing run record + a PASSing K-SHARE line; runs are processed in PULL ORDER; the tool tree is the pinned tree + ONE file (the launch commit's model: the pinned tree REJECTS a LAUNCH-INTENT checkpoint's cfg);
# a RAM guard (>= 4 GB) and a disk guard (>= 5 GB on the output drive) run before every rollout.
#
#   for each of the four runs (A_0 A_1 A_rep,0 A_rep,1), in the order their pulls land:
#     1. READY: the box-2 log shows '<run> pulled to <ART>/<run> (checkpoint sha256 H) [<label>, control <ctl>]' AND 'LAUNCH-INTENT run record: PASS' AND 'LAUNCH-INTENT K-SHARE: PASS' in that run's section.
#     2. the pull is COMPLETE BY CONTENT and its run record is RE-VERIFIED from the pulled files (lit1_lib.py pullcheck: 8 names re-hashed; ckpt sha256 equal in THREE places; lever 1.58926 present in summary + selfcheck;
#        K-SHARE recomputed; K-MASK; K-INIT; bank; token binds).  The tokens must agree on ONE commit, the reader sha256 and the PREREG sha256, and the reader file / PREREG file on disk must hash to them.
#     3. ONE torch process at a time: T1-kin --limit causal (decisive), --limit logged (K4), fresh-age T0 eval (K4's comparator).  Checkpoint re-hashed before and after every step; done records carry ckpt sha256 before / after,
#        the tool digest and the command sha256.  Before EVERY step: RAM >= 4096 MB available and >= 5 GB free on the output drive (it WAITS, never deletes).
#   after the 12 steps: the box-2 ZZV8FMINI-DONE (after the last pull), every pull re-verified, the manifest runs.json (CTRL = banked DAG-OACC, REP = banked DAG-ORC r2's REP; K-BUILD from git objects), then the reader.
#
#   bash lit1_chain.sh --dry-run     prints every guard, the plan, the EXACT 12 commands (each parsed by the pinned tool's own argparse; nothing runs) and the reader command; builds the LAUNCH-INTENT tree under F:/ if absent
#   bash lit1_chain.sh --write-pins  rewrite PINS.txt (a deliberate act; the chain STOPs on any difference from PINS.txt)
#   bash lit1_chain.sh               the real chain.  RESUME = run the same command again (a finished step is skipped by a done record naming THIS checkpoint's sha256, the tool digest and the command's sha256; an incomplete
#                                    earlier attempt is MOVED to <OUT>/_incomplete_*, never deleted).  The previous chain.out is rotated to chain.out.prev_<time> so the waker sees only this run's markers.
# Markers (own log, <LOG>/chain.out, ANCHORED at line start):
#   ZZLIT1-STAGE-DONE <k>/4 <run> (<label>): causal, logged, t0fresh done      one per run, in the order the runs were processed (4/4 = the last)
#   ZZLIT1-DONE <n>/12 steps, reader rc <rc> ...                               finished.  reader rc 0 = an outcome row written | 2 = REFUSED (a control failed; nothing written) | 3 = UNPLACED | 4 = BLOCKED
#   ZZLIT1-STOP <why>                                                          refused.  Fix the named cause and re-run the SAME command.  Also touches <LOG>/STOP.
#   ZZLIT1-DRY-OK                                                              the dry run's end marker.
# Environment (all optional; a WMI-launched process does not inherit the caller's shell, so put them in the .cmd):
#   LIT1_B2_OUT (box 2's chain log; default C:/Users/Admin/box2_ship/logs/chain_v8f_launchintent.out; the predecessors' logs are REFUSED)   LIT1_ART (box 2's pull dir; default E:/Projects/TanitAD-artifacts/v8f-launch-intent)
#   LIT1_OUT (default F:/TanitAD-scratch/launch_intent_t1_out)   LIT1_LOG   LIT1_TREE_LI (default F:/TanitAD-scratch/launch_intent_t1/tree_li)   LIT1_READER (default dagland/launch_intent_v2/read/launch_intent_read.py = the LANDED reader, sha256 5e950dcf...)   LIT1_PREREG (default landed/PREREG_V8F_LAUNCH_INTENT.md = the LANDED PREREG at 77234276, sha256 469bac1f...; both sha256 are PINNED, LIT1_EXPECT_PREREG_SHA / LIT1_EXPECT_READER_SHA override)
#   LIT1_ORC_DIR (the reader's --orc-dir)   LIT1_EXPECT_COMMIT (40-hex: the commit every token must bind)   LIT1_MIRROR   LIT1_MAX_WAIT_H (12)   LIT1_RAM_MIN_MB (4096)   LIT1_F_MIN_GB (5)   LIT1_C_MIN_MB (2048)
#   LIT1_B2DONE_WAIT_S (7200)   LIT1_GUARD_WAIT_S (7200)   LIT1_POLL_S (60)   LIT1_WORLD (TEST ONLY: a directory inside F:/TanitAD-scratch/launch_intent_t1; refused anywhere else; skips the real-input guards, see NOTES)
set -u
shopt -u patsub_replacement 2>/dev/null || true       # bash 5.2: '&' in a replacement string would be expanded ("Architecture & Inference" is in the tool paths)
export PATH="/usr/bin:/mingw64/bin:$PATH"
MODE=real
case "${1:-}" in --dry-run) MODE=dry ;; --write-pins) MODE=pins ;; "") ;; *) echo "usage: lit1_chain.sh [--dry-run | --write-pins]   (configuration = LIT1_* environment variables, see the header)"; exit 2 ;; esac
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); Hm=$(cygpath -m "$H")
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
LIB="$Hm/lit1_lib.py"
AR=/e/Projects/TanitAD-artifacts
BASE=E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a                                      # the pinned tool tree that rolled CTRL and REP out: READ ONLY
TREELI=${LIT1_TREE_LI:-F:/TanitAD-scratch/launch_intent_t1/tree_li}                          # BASE's code + the launch commit's model (see NOTES.md)
V2=E:/Projects/TanitAD-artifacts/dagland/launch_intent_v2
READER=${LIT1_READER:-$V2/read/launch_intent_read.py}; PREREG=${LIT1_PREREG:-$H/landed/PREREG_V8F_LAUNCH_INTENT.md}; ORCDIR=${LIT1_ORC_DIR:-E:/Projects/TanitAD-artifacts/dagorc_rebase/read}
SF=${LIT1_INPUTS:-$AR/sfresh_inputs}
CTRL_T1=$AR/v8f-dagoacc-t1; CTRL_ART=$AR/v8f-dagoacc; REP_OUT=F:/TanitAD-scratch/dagorc_t1_out; REP_ART=$AR/v8f-dag-orc-oacc      # CTRL = DAG-OACC, REP = DAG-ORC r2's REP: BANKED (read, never re-run)
ART=${LIT1_ART:-$AR/v8f-launch-intent}; OUT=${LIT1_OUT:-F:/TanitAD-scratch/launch_intent_t1_out}; LOG=${LIT1_LOG:-$H/logs}
B2OUT=${LIT1_B2_OUT:-C:/Users/Admin/box2_ship/logs/chain_v8f_launchintent.out}
MIRROR=${LIT1_MIRROR:-C:/Users/Admin/tanitad-push}
LAUNCH_COMMIT=b09ac89fc9d6e81baf3977954ae638919b979e71
NDUMP=141; POLL=${LIT1_POLL_S:-60}; MINAGE=3
RAMMIN=${LIT1_RAM_MIN_MB:-4096}; FMIN=${LIT1_F_MIN_GB:-5}; CMIN=${LIT1_C_MIN_MB:-2048}; EMIN=${LIT1_E_MIN_MB:-200}
MAXWAIT_S=${LIT1_MAX_WAIT_S:-$(( ${LIT1_MAX_WAIT_H:-12} * 3600 ))}; B2DONE_S=${LIT1_B2DONE_WAIT_S:-7200}; GUARDWAIT_S=${LIT1_GUARD_WAIT_S:-7200}
RUNS=(launchintent:0 launchintent:1 launchintentrep:0 launchintentrep:1)      # the box-2 chain's order; they are PROCESSED in pull order
WORLD=${LIT1_WORLD:-}; TESTWORLD=0
if [ -n "$WORLD" ]; then   # TEST ONLY: honoured iff the directory resolves inside F:/TanitAD-scratch/launch_intent_t1 (lit1_lib.py world_ok); anything else is refused before anything is read
  LIT1_WORLD="$WORLD" "$PY" -I "$LIB" world < /dev/null > /dev/null 2>&1 || { echo "ZZLIT1-STOP LIT1_WORLD=$WORLD is not a directory inside F:/TanitAD-scratch/launch_intent_t1: refused, nothing was run"; exit 1; }
  TESTWORLD=1; export LIT1_WORLD="$WORLD"      # the world's paths come from the LIT1_* variables the test sets (LIT1_TREE_LI, LIT1_READER, LIT1_PREREG, LIT1_ART, LIT1_OUT, LIT1_LOG, LIT1_B2_OUT)
fi
case "$(basename "$B2OUT")" in chain_v8f_dagoaccorc.out|chain_v8f_dagoaccorc_r2.out|chain_v8f_dagoacc.out|chain_v8f_dag2.out) [ "$MODE" = real ] && { echo "ZZLIT1-STOP LIT1_B2_OUT is a SIBLING chain's log ($(basename "$B2OUT")): this chain reads chain_v8f_launchintent.out"; exit 1; } ;; esac
m() { cygpath -m "$1"; }
TTu=$(cygpath -u "$TREELI"); TTm=$(m "$TREELI"); BASEm=$(m "$BASE"); ARTm=$(m "$ART"); OUTu=$(cygpath -u "$OUT"); OUTm=$(m "$OUT"); LOGu=$(cygpath -u "$LOG"); LOGm=$(m "$LOG"); B2OUTm=$(m "$B2OUT"); MIRRORm=$(m "$MIRROR")
SFm=$(m "$SF"); CTRL_T1m=$(m "$CTRL_T1"); REP_OUTm=$(m "$REP_OUT"); READERm=$(m "$READER"); PREREGm=$(m "$PREREG"); ORCDIRm=$(m "$ORCDIR")
PKS="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
fmt() { "$PY" -I "$LIB" fmt -- "$@" < /dev/null; }
rundir() { echo "M2b_s${2}_${1}"; }                                        # rundir <arm> <seed>
etag() { echo "M2b-${1}_s${2}"; }
label() { case "$1:$2" in launchintent:0) echo A_0 ;; launchintent:1) echo A_1 ;; launchintentrep:0) echo A_rep,0 ;; launchintentrep:1) echo A_rep,1 ;; esac; }
tstem() { case "$1" in launchintent) echo LI ;; launchintentrep) echo LIREP ;; esac; }
tagof() { echo "$(tstem "$1")_s${2}_m10_${3}_heads"; }                    # tagof <arm> <seed> <limit>: the rollout's --tag
t0tag() { echo "$(etag "$1" "$2")-fresh"; }
ckpt_of() { echo "$ARTm/$(rundir "$1" "$2")/ckpt_last.pt"; }
fsha() { sha256sum < "$(cygpath -u "$1")" | cut -c1-64; }

if [ "$MODE" = dry ] || [ "$MODE" = pins ]; then
  say() { echo "$(date '+%m-%d %H:%M:%S') DRY: $*"; }
  stop() { say "$*"; echo "ZZLIT1-STOP $*"; exit 1; }
else
  mkdir -p "$LOGu" "$OUTu" "$OUTu/ticks" "$OUTu/argv" "$OUTu/_tmp"
  say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$LOGu/chain.out"; }
  stop() { say "STOP: $*"; echo "ZZLIT1-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }
  rm -f "$LOGu/STOP"
  if ! mkdir "$LOGu/chain.lock" 2>/dev/null; then      # ONE instance only (two instances would run two torch rollouts at once)
    op=$(cat "$LOGu/chain.lock/winpid" 2>/dev/null)
    if [ -n "$op" ] && tasklist //FI "PID eq $op" //NH 2>/dev/null | grep -q " $op "; then echo "ZZLIT1-STOP another instance of this chain is running (winpid $op, $LOGu/chain.lock)"; exit 1; fi
    rm -rf "$LOGu/chain.lock"; mkdir "$LOGu/chain.lock" || { echo "ZZLIT1-STOP cannot take $LOGu/chain.lock"; exit 1; }
  fi
  cat /proc/$$/winpid > "$LOGu/chain.lock/winpid" 2>/dev/null; cp "$LOGu/chain.lock/winpid" "$LOGu/chain.winpid" 2>/dev/null
  trap 'rm -rf "$LOGu/chain.lock"' EXIT
  [ -s "$LOGu/chain.out" ] && mv "$LOGu/chain.out" "$LOGu/chain.out.prev_$(date +%m%d%H%M%S)"
  say "=== lit1_chain start (mode real; test world $TESTWORLD; tool tree $TTm; base tree $BASEm; reader $READERm; ART $ARTm; OUT $OUTm; box-2 log $B2OUTm; runs ${RUNS[*]})"
fi
nostop() { [ "$MODE" = real ] || return 0; [ -e "$LOGu/STOP" ] && exit 1; return 0; }
cfree_mb() { df -m /c | awk 'NR==2{print $4}'; }
efree_mb() { df -m /e | awk 'NR==2{print $4}'; }
CF0=$(cfree_mb)
[ "$CF0" -ge "$CMIN" ] || stop "C: has $CF0 MB free (< $CMIN MB): the pagefile is on C: and a full C: killed torch start-ups before"
say "C: gate: $CF0 MB free >= $CMIN MB"
if [ "$TESTWORLD" = 0 ]; then EF0=$(efree_mb); [ "$EF0" -ge "$EMIN" ] || stop "E: has $EF0 MB free (< $EMIN MB): the chain's logs and the box-2 pulls land there"; say "E: gate: $EF0 MB free >= $EMIN MB (rollout outputs go to $OUTm, not E:)"; fi
OUTDRIVE="${OUTm:0:3}"      # e.g. F:/

# ------------------------------------------------------------------------------------------------ 0  real-input guards (NOT run in a test world: lit1_selftest covers each at lib level, and --dry-run runs them on the real inputs)
TOOL="$TTu/stack/scripts/v8f_t1_eval.py"; TOOL0="$TTu/stack/scripts/v8f_mini_eval.py"
KB_STATE="n/a"
md5chk() { [ "$(md5sum < "$1" 2>/dev/null | cut -c1-32)" = "$2" ] || stop "md5 of $1 is not $2"; }
real_guards() {
  [ -f "$BASE/stack/scripts/v8f_t1_eval.py" ] || stop "no pinned tool tree at $BASEm"
  local rawp="$BASE/$PKS/raw" t_pos t_neg c_pos c_neg v
  t_pos=$(grep -cE 'add_argument\("--init-ckpt"' "$BASE/stack/scripts/v8f_mini_train.py"); t_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$BASE/stack/scripts/v8f_mini_train.py")
  [ "$t_pos" -ge 1 ] && [ "$t_neg" -eq 0 ] || stop "grep control broken on the pinned trainer (positive $t_pos, negative $t_neg)"
  c_pos=$(grep -cE 'add_argument\("--replan-steps"' "$BASE/stack/scripts/v8f_t1_eval.py"); c_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$BASE/stack/scripts/v8f_t1_eval.py")
  [ "$c_pos" -ge 1 ] && [ "$c_neg" -eq 0 ] || stop "grep control broken on the T1 tool (positive $c_pos, negative $c_neg)"
  local n_heads n_ticks n_caus n_log n_cfg n_mw n_ages
  n_heads=$(grep -cE 'add_argument\("--dump-heads"' "$TOOL"); n_ticks=$(grep -cE 'add_argument\("--dump-ticks"' "$TOOL"); n_caus=$(grep -cE '^LIMIT_MODES *= *\(.*"causal"' "$TOOL"); n_log=$(grep -cE '^LIMIT_MODES *= *\(.*"logged"' "$TOOL")
  n_cfg=$(grep -cE 'M\.V8FMiniConfig\(\*\*ck\["cfg"\]\)' "$TOOL"); n_mw=$(grep -cE 'add_argument\("--max-windows"' "$TOOL"); n_ages=$(grep -cE 'add_argument\("--ages"' "$TOOL0")
  say "guards: --dump-heads = $n_heads, --dump-ticks = $n_ticks, causal = $n_caus, logged = $n_log, model built from the checkpoint's own cfg = $n_cfg, --max-windows = $n_mw, v8f_mini_eval --ages = $n_ages  (controls: $c_pos / $t_pos / $c_neg / $t_neg) in $TTm"
  for v in n_heads n_ticks n_caus n_log n_cfg n_mw n_ages; do [ "${!v}" -ge 1 ] || stop "REFUSED: guard $v = ${!v} in $TTm"; done
  md5chk "$rawp/eval_grid_windows.json" 34ed8ac56365eeff72e42581bde2f5c4; md5chk "$rawp/join_eval141.npz" 532fcd644c0e0835e847b2f1cee011ad; md5chk "$SF/inputs/native_tracks.pt" 189f4e68913c0163912a509203963f8d; md5chk "$rawp/eval141_kin.npz" c92ee56db60b972ec84518a4a73d9f83
  md5chk "$SF/inputs/v9/v9_labels_eval139.npz" 6b5c7f207cffc3b7eb3cd527fd433599
  for v in "$SF/inputs/v9/lead_eval139.npz" "$SF/inputs/v9/refcv6_speed_max_v8_eval.jsonl" "$SF/inputs/p10/p10_prior.pt"; do [ -f "$v" ] || stop "input missing: $v"; done
  [ "$(ls "$SF/cache/eval/clips" 2>/dev/null | wc -l)" = 141 ] || stop "eval cache $SFm/cache is not 141 clips"
}
tree_ensure() {   # the LAUNCH-INTENT tool tree: built (or verified) here; a tree that does not verify is MOVED ASIDE and rebuilt, never deleted
  local o
  o=$("$PY" -I "$LIB" treebuild --dest "$TTm" --base "$BASEm" --mirror "$MIRRORm" 2>&1 < /dev/null) || stop "cannot build the LAUNCH-INTENT tool tree at $TTm: $(echo "$o" | tr '\n' ' ' | cut -c1-500)"
  say "LAUNCH-INTENT tool tree $TTm: $(echo "$o" | tail -1) (= the pinned tree's code, ONE file replaced: the launch commit's model; why: NOTES.md)"
  o=$("$PY" -I "$LIB" treecheck --tree "$TTm" --base "$BASEm" --mirror "$MIRRORm" 2>&1 < /dev/null) || stop "the LAUNCH-INTENT tool tree does not verify: $(echo "$o" | tr '\n' ' ' | cut -c1-600)"
  say "tree check: $(echo "$o" | tail -1); differs from the pinned base in exactly: $(echo "$o" | grep -A1 'differs_from_base' | tr -d ' \n')"
}
if [ "$TESTWORLD" = 1 ]; then say "TESTWORLD: real-input guards, tree build, template proofs, banked-pin checks and the reader's static check are SKIPPED (lit1_selftest only; real mode never honours LIT1_WORLD)"; else
  tree_ensure
  [ -f "$TOOL" ] && [ -f "$TOOL0" ] || stop "no tool at $TTm/stack/scripts/v8f_t1_eval.py or v8f_mini_eval.py"
  real_guards
fi
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack")      # OMP/MKL 8 = the banked rollouts' threads (same reduction order)
export PYTHONPATH="$TTm/stack;$TTm/taniteval;$TTm/stack/scripts"
[ "$MODE" = dry ] || [ "$MODE" = pins ] || export TMP="$OUTm/_tmp" TEMP="$OUTm/_tmp"        # keep the temp files off C:
tree_digest() { "$PY" -I "$LIB" treedigest --tree "$TTm" < /dev/null; }
TOOLD=$(tree_digest)
[ -n "$TOOLD" ] || stop "cannot compute the tool tree digest of $TTm"

PINS=$H/PINS.txt
write_pins() {
  echo "# PINS of lit1_chain.sh (PREREG_V8F_LAUNCH_INTENT, dev-box T1-kin chain).  The chain STOPs on ANY difference.  Rewrite only deliberately: bash lit1_chain.sh --write-pins"
  echo "basetree=$BASEm"; echo "basedigest=$("$PY" -I "$LIB" treedigest --tree "$BASEm" < /dev/null)"; echo "litree=$TTm $TOOLD"
  for f in stack/scripts/v8f_t1_eval.py stack/scripts/v8f_mini_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/floors_kin.py taniteval/taniteval/four_families.py taniteval/taniteval/ci.py taniteval/tools/t1_eval.py; do echo "basefile=$f $(fsha "$BASE/$f")"; done
  for f in v8f-dagoacc-t1/runs.json v8f-dagoacc-t1/done_DAGOACC_s0_m10_causal_heads.txt v8f-dagoacc-t1/done_DAGOACC_s1_m10_causal_heads.txt v8f-dagoacc-t1/argv/DAGOACC_s0_m10_causal_heads.argv v8f-dagoacc-t1/argv/DAGOACC_s1_m10_causal_heads.argv v8f-dagoacc-t1/PINS.txt v8f-dagoacc/M2b_s0_dagoacc/ckpt_last.pt v8f-dagoacc/M2b_s1_dagoacc/ckpt_last.pt; do echo "ctrl=$f $(fsha "$AR/$f")"; done
  echo "rep=$REP_OUTm/runs.json $(fsha "$REP_OUT/runs.json")"
  for s in 0 1; do for lim in causal logged; do echo "rep=$REP_OUTm/done_DAGORCREP_s${s}_m10_${lim}_heads.txt $(fsha "$REP_OUT/done_DAGORCREP_s${s}_m10_${lim}_heads.txt")"; done; echo "rep=$(m "$REP_ART")/M2b_s${s}_dagoaccorcrep/ckpt_last.pt $(fsha "$REP_ART/M2b_s${s}_dagoaccorcrep/ckpt_last.pt")"; echo "rep=$(m "$REP_ART")/M2b_s${s}_dagoaccorcrep/launch_gate_token.json $(fsha "$REP_ART/M2b_s${s}_dagoaccorcrep/launch_gate_token.json")"; done
  echo "compose=E:/Projects/TanitAD-artifacts/dagorc_t1/kbuild_r2_launch_8c24823f.json $(fsha "$AR/dagorc_t1/kbuild_r2_launch_8c24823f.json")"
  for f in rollout_causal.tmpl rollout_logged.tmpl t0fresh.tmpl reader.tmpl; do echo "tmpl=$f $(fsha "$H/argv/$f")"; done
  echo "lib=lit1_lib.py $(fsha "$H/lit1_lib.py")"; echo "lib=lit1_tree.py $(fsha "$H/lit1_tree.py")"
  echo "md5=eval_grid_windows.json 34ed8ac56365eeff72e42581bde2f5c4"; echo "md5=join_eval141.npz 532fcd644c0e0835e847b2f1cee011ad"; echo "md5=native_tracks.pt 189f4e68913c0163912a509203963f8d"; echo "md5=eval141_kin.npz c92ee56db60b972ec84518a4a73d9f83"
}
if [ "$MODE" = pins ]; then write_pins > "$H/PINS.txt"; echo "wrote $H/PINS.txt ($(grep -vc '^#' "$H/PINS.txt") pins)"; exit 0; fi
pins_verify() {   # every pin recomputed; ONE difference stops the chain
  local k a b got bad=0 n=0 rel
  [ -f "$PINS" ] || stop "no PINS.txt at $PINS (bash lit1_chain.sh --write-pins writes it)"
  while IFS=' ' read -r k a b; do
    case "$k" in
      "#"*|"") continue ;;
      basetree=*) [ "${k#basetree=}" = "$BASEm" ] || { say "PIN: base tree is $BASEm, pinned ${k#basetree=}"; bad=$((bad+1)); } ;;
      basedigest=*) got=$("$PY" -I "$LIB" treedigest --tree "$BASEm" < /dev/null); [ "${k#basedigest=}" = "$got" ] && [ "$got" = "6a4b47c8acfadf07ffc82ab174cccce78aa0404f6a4f89231f331013bbcda3d9" ] || { say "PIN: base digest ${got:0:12}, pinned ${k:11:12}"; bad=$((bad+1)); } ;;
      litree=*) [ "$a" = "$TOOLD" ] || { say "PIN: LAUNCH-INTENT tree digest ${TOOLD:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      basefile=*) got=$(fsha "$BASE/${k#basefile=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: ${k#basefile=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      ctrl=*) got=$(fsha "$AR/${k#ctrl=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: banked CTRL file ${k#ctrl=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      rep=*) got=$(fsha "${k#rep=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: banked REP file ${k#rep=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      compose=*) got=$(fsha "${k#compose=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: the DAG-ORC K-BUILD record is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      tmpl=*) got=$(fsha "$H/argv/${k#tmpl=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: argv template ${k#tmpl=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      lib=*) got=$(fsha "$H/${k#lib=}" 2>/dev/null); [ "$got" = "$a" ] || { say "PIN: ${k#lib=} is ${got:0:12}, pinned ${a:0:12}"; bad=$((bad+1)); } ;;
      md5=*) ;;      # md5 inputs are checked by md5chk above
      *) say "PIN: unknown pin line '$k'"; bad=$((bad+1)) ;;
    esac
    n=$((n+1))
  done < "$PINS"
  [ $bad -eq 0 ] || stop "PINS.txt: $bad difference(s) (listed above): nothing was started"
  say "PINS.txt verified ($n pins: base tree digest 6a4b47c8..., LAUNCH-INTENT tree ${TOOLD:0:12}, the banked CTRL and REP files, the DAG-ORC K-BUILD record, the argv templates, the two lib files)"
}
b2_pin() {   # b2_pin <when>: the box-2 chain's files (the printers of the log lines this chain parses) are pinned ON FIRST SIGHT in <OUT>/PINS_BOX2.txt; a later difference means the log format may have changed
  [ "$MODE" = real ] || return 0
  local f got a pf=$OUTu/PINS_BOX2.txt B2D; B2D=$(dirname "$(dirname "$B2OUT")")
  for f in chain_v8f_launchintent.sh v8f_launchintent_chain_checks.py gate_launchintent_assert.py; do
    [ -f "$(cygpath -u "$B2D")/$f" ] || continue
    got=$(sha256sum < "$(cygpath -u "$B2D")/$f" | cut -c1-64); a=$(grep -a "^$f " "$pf" 2>/dev/null | cut -d' ' -f2)
    if [ -z "$a" ]; then echo "$f $got" >> "$pf"; say "box-2 chain file $f pinned on first sight: ${got:0:16}... ($1)"
    elif [ "$a" != "$got" ]; then stop "the box-2 chain file $f changed since it was first seen (${a:0:12} -> ${got:0:12}) ($1): the log lines this chain parses may differ; re-review lit1_lib.py parse_boxlog, delete $pf deliberately to re-pin"; fi
  done
}
pin_check() { [ "$(tree_digest)" = "$TOOLD" ] || stop "the tool tree changed ($1)"; }
RSHA=""; PSHA=""; [ -f "$READER" ] && RSHA=$(fsha "$READER"); [ -f "$PREREG" ] && PSHA=$(fsha "$PREREG")
# the LANDED pre-registration and reader (coordinator 2026-10-28: PREREG_V8F_LAUNCH_INTENT landed at 77234276db2c, ls-remote verified; the reader landed unchanged): their sha256 are PINNED here, so a different file at
# LIT1_PREREG / LIT1_READER stops the chain before anything is read.  The PREREG default is landed/ in this directory, extracted from the git object 77234276:Project Steering/PREREG_V8F_LAUNCH_INTENT.md (landed/PROVENANCE.txt).
# A test world pins nothing unless LIT1_EXPECT_PREREG_SHA / LIT1_EXPECT_READER_SHA is set; the value 'none' disables a pin in a real run (the self-test's stub files).
EXP_P=${LIT1_EXPECT_PREREG_SHA-469bac1f3918b31b875e7c06b637980f6acb194a69ded3294c180aec95b1f65a}; EXP_R=${LIT1_EXPECT_READER_SHA-5e950dcff2993653dffbe9b4f01ea17eb4d7b1721b7c41a669e0dc43c559fc35}
if [ "$TESTWORLD" = 1 ]; then EXP_P=${LIT1_EXPECT_PREREG_SHA:-}; EXP_R=${LIT1_EXPECT_READER_SHA:-}; fi
[ "$EXP_P" != none ] || EXP_P=""; [ "$EXP_R" != none ] || EXP_R=""      # 'none' = no pin (an empty environment value cannot be passed on Windows)
[ -z "$EXP_P" ] || [ "$PSHA" = "$EXP_P" ] || stop "the PREREG at $PREREGm hashes to ${PSHA:-ABSENT}, the landed PREREG_V8F_LAUNCH_INTENT is $EXP_P (LIT1_PREREG / LIT1_EXPECT_PREREG_SHA)"
[ -z "$EXP_R" ] || [ "$RSHA" = "$EXP_R" ] || stop "the reader at $READERm hashes to ${RSHA:-ABSENT}, the landed reader is $EXP_R (LIT1_READER / LIT1_EXPECT_READER_SHA)"
if [ "$TESTWORLD" = 0 ]; then pins_verify; fi
if [ "$MODE" = real ]; then
  # the reader (a PARAMETER): present, its own static check on the PREREG, and its sha256 recorded; the tokens must bind exactly these bytes (checked at the first pull)
  [ -f "$READER" ] || stop "no reader at $READERm (LIT1_READER)"; [ -f "$PREREG" ] || stop "no PREREG at $PREREGm (LIT1_PREREG)"
  if [ "$TESTWORLD" = 0 ]; then
    SC=$("$PY" -I "$READERm" --static-check --prereg "$PREREGm" --prereg-sha "$PSHA" 2>&1 < /dev/null) || stop "the reader's --static-check failed on $PREREGm: $(echo "$SC" | tr '\n' ' ' | cut -c1-600)"
    say "reader $READERm sha256 ${RSHA:0:16}...; PREREG $PREREGm sha256 ${PSHA:0:16}...; reader --static-check: $(echo "$SC" | tail -1)"
  fi
fi
pin_check_end() { pin_check "$1"; [ "$(fsha "$READER")" = "$RSHA" ] || stop "the reader changed ($1)"; }

# ------------------------------------------------------------------------------------------------ 1  the commands (rendered by lit1_lib.py from argv/*.tmpl; --dry-run prints and parses them)
mkcmd() {   # mkcmd <kind> <ckpt> <tag> -> array CMD
  mapfile -t CMD < <("$PY" -I "$LIB" render "$1" --ckpt "$2" --tag "$3" --tree "$TTm" --out "$OUTm" --reader "$READERm" --prereg "$PREREGm" --prereg-sha "${PSHA:-}" --orc-dir "$ORCDIRm" < /dev/null)
  CMD=("${CMD[@]%$'\r'}"); [ "${#CMD[@]}" -ge 8 ] || stop "template $1 rendered ${#CMD[@]} tokens"
}
if [ "$TESTWORLD" = 0 ]; then
  tmpl_proof() {   # the templates, rendered with the parameters of runs that ALREADY HAPPENED (against the BASE tree they ran from), reproduce the recorded commands token for token
    local want got line d2=$AR/dagorc_t1/logs/chain.out.prev_1028061116 ck=$AR/v8f-dag-orc-oacc/M2b_s0_dagoaccorcrep/ckpt_last.pt sv_t=$TTm sv_o=$OUTm
    TTm=$BASEm; OUTm=$CTRL_T1m; mkcmd causal "$CTRL_ART/M2b_s0_dagoacc/ckpt_last.pt" DAGOACC_s0_m10_causal_heads; want=$(cat "$CTRL_T1/argv/DAGOACC_s0_m10_causal_heads.argv"); got=$(printf '%s\n' "${CMD[@]}")
    [ "$want" = "$got" ] || stop "TEMPLATE PROOF FAILED: rollout_causal.tmpl rendered for DAG-OACC s0 differs from its banked argv"
    say "template proof 1: rollout_causal.tmpl rendered with DAG-OACC s0's parameters == its banked rollout argv ($(echo "$want" | wc -l) tokens, byte for byte)"
    OUTm=$REP_OUTm
    mkcmd causal "$ck" DAGORCREP_s0_m10_causal_heads; line=$(grep -a 'start DAGORCREP_s0_m10_causal_heads (attempt 1' "$d2" | head -1); line=${line#*): }
    [ "$line" = "$(fmt "${CMD[@]}")" ] || stop "TEMPLATE PROOF FAILED: causal template for DAG-ORC r2's REP s0 differs from the command that RAN ($d2)"
    mkcmd logged "$ck" DAGORCREP_s0_m10_logged_heads; line=$(grep -a 'start DAGORCREP_s0_m10_logged_heads (attempt 1' "$d2" | head -1); line=${line#*): }
    [ "$line" = "$(fmt "${CMD[@]}")" ] || stop "TEMPLATE PROOF FAILED: logged template differs from the command that RAN for REP s0"
    mkcmd t0fresh "$ck" M2b-dagoaccorcrep_s0-fresh; line=$(grep -a 'start fresh-T0 eval M2b-dagoaccorcrep_s0-fresh' "$d2" | head -1); line=${line#*: }
    [ "$line" = "$(fmt "${CMD[@]}")" ] || stop "TEMPLATE PROOF FAILED: t0fresh template differs from the command that RAN for REP s0"
    say "template proofs 2-4: the causal, logged and fresh-T0 templates rendered with DAG-ORC r2's REP s0 parameters == the three commands that RAN (dagorc_t1/logs/chain.out.prev_1028061116), token for token"
    TTm=$sv_t; OUTm=$sv_o
  }
  tmpl_proof
fi
n_dumps() { ls "$OUTu/dump_$1" 2>/dev/null | grep -c '^ep.*\.npz$'; }
cmd_sha() { printf '%s\0' "$@" | sha256sum | cut -c1-64; }

# ------------------------------------------------------------------------------------------------ 2  one step = one process.  Done records name the checkpoint, the tool digest and the command.
guards_wait() {   # guards_wait <what>: RAM >= RAMMIN MB available and >= FMIN GB free on the output drive BEFORE every rollout; it waits (never deletes), up to GUARDWAIT_S
  local w=0 what=$1 out t0; t0=$(date +%s)
  while :; do
    out=$("$PY" -I "$LIB" guards --ram-min-mb "$RAMMIN" --disk "$OUTDRIVE" --disk-min-gb "$FMIN" 2>&1 < /dev/null) && { [ $w -gt 0 ] && say "$what: guards hold again ($out)"; return 0; }
    nostop; [ $w -eq 0 ] && say "$what: waiting, a guard is LOW: $out"; w=$((w + 1))
    [ $(( $(date +%s) - t0 )) -ge "$GUARDWAIT_S" ] && stop "a guard stayed LOW for $GUARDWAIT_S s before $what: $out"
    sleep "$POLL"
  done
}
rollout_done() {   # rollout_done <tag> <ckpt sha256> <cmd sha>
  [ -s "$OUTu/t1v8f_$1.json" ] && [ -s "$OUTu/ticks/ticks_$1.npz" ] && [ -s "$OUTu/done_$1.txt" ] && [ -s "$OUTu/argv/$1.argv" ] || return 1
  [ "$(n_dumps "$1")" -ge "$NDUMP" ] || return 1
  grep -q "\"ckpt_sha256\": \"$2\"" "$OUTu/done_$1.txt" && grep -q "\"ckpt_sha256_before\": \"$2\"" "$OUTu/done_$1.txt" && grep -q "\"ckpt_sha256_after\": \"$2\"" "$OUTu/done_$1.txt" && grep -q "\"tool_digest\": \"$TOOLD\"" "$OUTu/done_$1.txt" && grep -q "\"cmd_sha256\": \"$3\"" "$OUTu/done_$1.txt"
}
aside() {   # aside <file-or-dir>...: MOVE an earlier attempt's output into a fresh _incomplete_<time> directory (never delete)
  local w f; w="$OUTu/_incomplete_$(date +%m%d%H%M%S)_$$_$RANDOM"; mkdir -p "$w"
  for f in "$@"; do [ -e "$f" ] && mv "$f" "$w/"; done
  say "moved the output of an earlier attempt to $w"
}
t1run() {   # t1run <arm> <seed> <limit> <ckpt sha256>
  local arm=$1 s=$2 lim=$3 sha=$4 tag ck t0 rc att cs
  tag=$(tagof "$arm" "$s" "$lim"); ck=$(ckpt_of "$arm" "$s")
  mkcmd "$lim" "$ck" "$tag"; cs=$(cmd_sha "${CMD[@]}")
  if rollout_done "$tag" "$sha" "$cs"; then say "skip $tag (complete, same checkpoint ${sha:0:12}, same tool, same command)"; return 0; fi
  if [ -e "$OUTu/t1v8f_$tag.json" ] || [ -e "$OUTu/ticks/ticks_$tag.npz" ] || [ -d "$OUTu/dump_$tag" ] || [ -e "$OUTu/done_$tag.txt" ] || [ -e "$OUTu/argv/$tag.argv" ]; then
    aside "$OUTu/t1v8f_$tag.json" "$OUTu/ticks/ticks_$tag.npz" "$OUTu/dump_$tag" "$OUTu/done_$tag.txt" "$OUTu/argv/$tag.argv"
  fi
  guards_wait "$tag"
  nostop; pin_check "before $tag"
  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed since its pull was verified (before $tag)"
  mkdir -p "$OUTu/argv"; printf '%s\n' "${CMD[@]}" > "$OUTu/argv/$tag.argv"
  for att in 1 2; do
    t0=$(date +%s)
    say "start $tag (attempt $att; ckpt $(basename "$(dirname "$ck")") ${sha:0:12}, limit $lim): $(fmt "${CMD[@]}")"
    "${CMD[@]}" > "$LOGu/$tag.out" 2>&1 < /dev/null; rc=$?
    say "end $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
    [ $rc -eq 139 ] && [ $att -eq 1 ] && { say "$tag: native crash (rc 139): one retry; the crashed attempt's output is never read"; mv "$LOGu/$tag.out" "$LOGu/$tag.segv.out"; rm -f "$OUTu/done_$tag.txt"; continue; }
    break
  done
  pin_check "after $tag"
  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed during $tag"
  { [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ] && [ -s "$OUTu/ticks/ticks_$tag.npz" ] && [ "$(n_dumps "$tag")" -ge "$NDUMP" ]; } || stop "rollout $tag failed (rc $rc; json $([ -s "$OUTu/t1v8f_$tag.json" ] && echo yes || echo NO), ticks $([ -s "$OUTu/ticks/ticks_$tag.npz" ] && echo yes || echo NO), dumps $(n_dumps "$tag")/$NDUMP): $(tail -2 "$LOGu/$tag.out" | tr '\n' ' ' | cut -c1-300)"
  printf '{"tag": "%s", "limit": "%s", "ckpt_sha256": "%s", "ckpt_sha256_before": "%s", "ckpt_sha256_after": "%s", "tool_digest": "%s", "cmd_sha256": "%s", "n_dumps": %s, "wall_s": %s, "smoke": false, "max_windows": null}\n' \
    "$tag" "$lim" "$sha" "$sha" "$sha" "$TOOLD" "$cs" "$(n_dumps "$tag")" "$(( $(date +%s) - t0 ))" > "$OUTu/done_$tag.txt"
}
t0run() {   # t0run <arm> <seed> <ckpt sha256>: the fresh-age T0 eval (K4's comparator)
  local arm=$1 s=$2 sha=$3 tag ck t0 rc cs
  tag=$(t0tag "$arm" "$s"); ck=$(ckpt_of "$arm" "$s")
  mkcmd t0fresh "$ck" "$tag"; cs=$(cmd_sha "${CMD[@]}")
  if [ -s "$OUTu/t0fresh/eval_$tag.json" ] && [ -s "$OUTu/t0fresh/eval_$tag.npz" ] && [ -s "$OUTu/done_t0f_$tag.txt" ] && grep -q "\"ckpt_sha256\": \"$sha\"" "$OUTu/done_t0f_$tag.txt" && grep -q "\"ckpt_sha256_before\": \"$sha\"" "$OUTu/done_t0f_$tag.txt" \
     && grep -q "\"ckpt_sha256_after\": \"$sha\"" "$OUTu/done_t0f_$tag.txt" && grep -q "\"tool_digest\": \"$TOOLD\"" "$OUTu/done_t0f_$tag.txt" && grep -q "\"cmd_sha256\": \"$cs\"" "$OUTu/done_t0f_$tag.txt"; then
    say "skip fresh-T0 eval $tag (complete, same checkpoint ${sha:0:12}, same tool, same command)"; return 0
  fi
  mkdir -p "$OUTu/t0fresh"
  if [ -e "$OUTu/t0fresh/eval_$tag.json" ] || [ -e "$OUTu/t0fresh/eval_$tag.npz" ] || [ -e "$OUTu/done_t0f_$tag.txt" ]; then
    aside "$OUTu"/t0fresh/eval_$tag.* "$OUTu/t0fresh/eval_${tag}_proj.npz" "$OUTu/done_t0f_$tag.txt"
  fi
  guards_wait "$tag"
  nostop; pin_check "before $tag"
  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed since its pull was verified (before $tag)"
  t0=$(date +%s)
  say "start fresh-T0 eval $tag: $(fmt "${CMD[@]}")"
  "${CMD[@]}" > "$LOGu/$tag.out" 2>&1 < /dev/null; rc=$?
  say "end fresh-T0 eval $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
  pin_check "after $tag"
  [ "$(fsha "$ck")" = "$sha" ] || stop "$ck changed during $tag"
  { [ $rc -eq 0 ] && [ -s "$OUTu/t0fresh/eval_$tag.json" ] && [ -s "$OUTu/t0fresh/eval_$tag.npz" ]; } || stop "fresh-T0 eval $tag failed (rc $rc): $(tail -2 "$LOGu/$tag.out" | tr '\n' ' ' | cut -c1-300)"
  printf '{"tag": "%s", "ckpt_sha256": "%s", "ckpt_sha256_before": "%s", "ckpt_sha256_after": "%s", "tool_digest": "%s", "cmd_sha256": "%s", "wall_s": %s, "smoke": false}\n' "$tag" "$sha" "$sha" "$sha" "$TOOLD" "$cs" "$(( $(date +%s) - t0 ))" > "$OUTu/done_t0f_$tag.txt"
}

# ------------------------------------------------------------------------------------------------ 3  the box-2 chain's LOG says READY (pull + run record PASS + K-SHARE PASS), then the pull is verified by content
b2_stops() { "$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" meta 2>/dev/null < /dev/null | tr ' ' '\n' | grep '^stops=' | cut -d= -f2; }
B2STOP0=$(b2_stops); B2STOP0=${B2STOP0:-0}
LAUNCH_SEEN=$(cat "$OUTu/LAUNCH_COMMIT.txt" 2>/dev/null)
EARLY_DONE=0
commit_gate() {   # commit_gate <arm> <seed>: the tokens present agree on ONE commit, the reader sha256 and the PREREG sha256; the files on disk ARE those bytes; the commit descends from / equals the pinned flag commit
  local g=$(rundir "$1" "$2") j c r p out rc
  j=$("$PY" -I "$LIB" commits --art "$ARTm" 2>&1 < /dev/null) || stop "$g: the pulled launch-gate tokens do not bind ONE commit / reader sha / PREREG sha: $j"
  c=$(echo "$j" | "$PY" -I -c "import json,sys; d=json.load(sys.stdin); print(d['commits'][0], d['reader_sha256'][0], d['prereg_sha256'][0])"); set -- $c; c=$1; r=$2; p=$3
  [ -z "${LIT1_EXPECT_COMMIT:-}" ] || [ "$c" = "$LIT1_EXPECT_COMMIT" ] || stop "$g was built at ${c:0:12}, LIT1_EXPECT_COMMIT is ${LIT1_EXPECT_COMMIT:0:12}"
  [ "$r" = "$RSHA" ] || stop "the tokens bind reader sha256 ${r:0:16}..., the reader at $READERm is ${RSHA:0:16}...: this chain will not read with a different reader (LIT1_READER)"
  [ "$p" = "$PSHA" ] || stop "the tokens bind PREREG sha256 ${p:0:16}..., the PREREG at $PREREGm is ${PSHA:0:16}... (LIT1_PREREG)"
  if [ -n "$LAUNCH_SEEN" ]; then [ "$c" = "$LAUNCH_SEEN" ] || stop "$g was built at ${c:0:12}, the earlier run(s) at ${LAUNCH_SEEN:0:12}: one launch commit is required"; return 0; fi
  LAUNCH_SEEN=$c; echo "$c" > "$OUTu/LAUNCH_COMMIT.txt"
  say "launch commit (from the tokens, all in agreement): $c; reader sha256 ${r:0:16}... and PREREG sha256 ${p:0:16}... equal the files on disk"
  [ "$TESTWORLD" = 0 ] || return 0
  if git -c safe.directory='*' -C "$MIRROR" cat-file -e "$c^{commit}" 2>/dev/null; then
    git -c safe.directory='*' -C "$MIRROR" merge-base --is-ancestor "$LAUNCH_COMMIT" "$c" 2>/dev/null && say "  ancestry: ${c:0:12} equals or descends from the pinned flag commit ${LAUNCH_COMMIT:0:12}" || stop "the tokens' commit ${c:0:12} does not descend from the pinned flag commit ${LAUNCH_COMMIT:0:12}"
  else say "  ancestry UNVERIFIED: ${c:0:12} is not in the mirror $MIRRORm (the tokens' trainer / model sha256 equal the pinned launch bytes, checked by pullcheck)"; fi
  out=$("$PY" -I "$LIB" rolloutcode --mirror "$MIRRORm" --launch "$c" --base "$BASEm" 2>&1 < /dev/null); rc=$?
  say "  rolloutcode (the 45 files the harness loads, pinned tree vs ${c:0:12}): $(echo "$out" | tail -1)"; [ $rc -eq 0 ] || stop "ROLLOUT-CODE: a file the T1 / T0 harness loads differs between the pinned tree and the launch commit beyond the proven overlay: $(echo "$out" | tr '\n' ' ' | cut -c1-600)"
  out=$("$PY" -I "$LIB" kbuild --mirror "$MIRRORm" --launch "$c" --reader "$READERm" 2>&1 < /dev/null | tail -1)
  say "  K-BUILD, the reader's own verdict on the record the manifest will carry: ${out:0:600}"
  case "$out" in "READER-K-BUILD PASS"*) ;; *) say "  K-BUILD-WARNING: the reader will REFUSE the read at its K-BUILD control as the record stands; the rollouts are still taken (they are valid data); fix PREREG / reader or the record, then re-run to read" ;; esac
}
PULL_SHA=""
process_run() {   # process_run <arm> <seed> <log sha>: 0 = the three steps done, 1 = the pull is not complete yet
  local arm=$1 s=$2 lsha=$3 g pc prc
  g=$(rundir "$arm" "$s")
  pc=$("$PY" -I "$LIB" pullcheck --dir "$ARTm/$g" --arm "$arm" --seed "$s" --expect-sha "$lsha" --min-age "$MINAGE" 2>&1 < /dev/null); prc=$?
  [ $prc -eq 3 ] && stop "the pulled $g is WRONG, waiting cannot fix it: $pc"
  [ $prc -ne 0 ] && { PROC_MSG="the log shows PASS but the pull is not complete yet: $pc"; return 1; }
  PULL_SHA=$(echo "$pc" | cut -d' ' -f2)
  commit_gate "$arm" "$s"
  say "$g ($(label "$arm" "$s")): the box-2 log shows pulled + run record PASS + K-SHARE PASS; the pull is COMPLETE by content: ckpt sha256 ${PULL_SHA:0:16}... == the pull line == SHA256SUMS.txt == the file; $(echo "$pc" | cut -d' ' -f3-)"
  t1run "$arm" "$s" causal "$PULL_SHA"
  t1run "$arm" "$s" logged "$PULL_SHA"
  t0run "$arm" "$s" "$PULL_SHA"
  return 0
}

# ------------------------------------------------------------------------------------------------ 4  after the 12 steps: the box-2 DONE marker, every pull re-verified, the manifest, the reader
b2_done_wait() {
  local t0=$(date +%s) n=0
  while :; do
    "$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" done < /dev/null > /dev/null 2>&1 && { say "box-2 chain DONE after its last pull: $(grep -a '^ZZV8FMINI-DONE' "$B2OUT" | tail -1 | cut -c1-200)"; return 0; }
    nostop
    [ "$(b2_stops)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP before its DONE ($B2OUTm): $(grep -a '^ZZV8FMINI-STOP' "$B2OUT" | tail -1 | cut -c1-200)"
    [ $((n % 30)) -eq 0 ] && say "waiting for the box-2 chain's ZZV8FMINI-DONE ($(( ($(date +%s) - t0) / 60 )) min)"
    n=$((n + 1)); [ $(( $(date +%s) - t0 )) -gt "$B2DONE_S" ] && stop "no ZZV8FMINI-DONE (after the last pull) in $B2OUTm $B2DONE_S s after the last run was processed"
    sleep "$POLL"
  done
}
verify_pulls() {   # every pull is STILL complete, the log STILL says PASS, and the checkpoint is STILL the one every rollout scored
  local spec arm s out sha rec pc
  for spec in "${RUNS[@]}"; do
    arm=${spec%%:*}; s=${spec##*:}
    out=$("$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" run "$arm" "$s" 2>&1 < /dev/null) || stop "after the rollouts the box-2 log no longer shows $(rundir "$arm" "$s") as PASS: $out"
    pc=$("$PY" -I "$LIB" pullcheck --dir "$ARTm/$(rundir "$arm" "$s")" --arm "$arm" --seed "$s" --expect-sha "$(echo "$out" | cut -d' ' -f2)" --min-age "$MINAGE" 2>&1 < /dev/null) || stop "after the rollouts the pull of $(rundir "$arm" "$s") is no longer COMPLETE: $pc"
    sha=$(echo "$pc" | cut -d' ' -f2)
    for rec in "done_$(tagof "$arm" "$s" causal).txt" "done_$(tagof "$arm" "$s" logged).txt" "done_t0f_$(t0tag "$arm" "$s").txt"; do
      grep -q "\"ckpt_sha256\": \"$sha\"" "$OUTu/$rec" || stop "$rec does not name the pulled checkpoint $sha"
    done
  done
}

# ------------------------------------------------------------------------------------------------ --dry-run: print and leave
if [ "$MODE" = dry ]; then
  say "C: free $(cfree_mb) MB (>= $CMIN MB); output drive ${OUTDRIVE}: $("$PY" -I "$LIB" guards --ram-min-mb "$RAMMIN" --disk "$OUTDRIVE" --disk-min-gb "$FMIN" < /dev/null) (guards before EVERY rollout: RAM >= $RAMMIN MB, >= $FMIN GB free)"
  say "threads: OMP/MKL $OMP_NUM_THREADS (= the banked rollouts'), CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES', ONE torch process at a time (sequential), poll $POLL s, max wait $((MAXWAIT_S / 3600)) h; outputs under $OUTm"
  say "== the tool trees: pinned base $BASEm (digest 6a4b47c8... = the one CTRL and REP were rolled out with), LAUNCH-INTENT tree $TTm (digest ${TOOLD:0:16}...)"
  say "   cfg reach of the PINNED tree: $("$PY" -I "$LIB" cfgprobe --tree "$BASEm" < /dev/null 2>&1 | tail -1)"
  say "   cfg reach of the LAUNCH-INTENT tree: $("$PY" -I "$LIB" cfgprobe --tree "$TTm" < /dev/null 2>&1 | tail -1)"
  say "   rolloutcode vs the pinned flag commit ${LAUNCH_COMMIT:0:12}: $("$PY" -I "$LIB" rolloutcode --mirror "$MIRRORm" --launch "$LAUNCH_COMMIT" --base "$BASEm" < /dev/null 2>&1 | tail -1)"
  say "== box-2 log: $B2OUTm ($([ -f "$B2OUT" ] && wc -l < "$B2OUT" || echo "absent, 0") lines; ZZV8FMINI-STOP lines now: $B2STOP0; $("$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" meta < /dev/null))"
  "$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" table < /dev/null 2>&1 | while IFS= read -r l; do say "   $l"; done
  say "== reader $READERm: $([ -f "$READER" ] && echo "present, sha256 ${RSHA:-$(fsha "$READER")}" || echo ABSENT)  PREREG $PREREGm: $([ -f "$PREREG" ] && echo "present, sha256 $(fsha "$PREREG")" || echo ABSENT)"
  if [ -f "$READER" ] && [ -f "$PREREG" ]; then say "   reader --static-check: $("$PY" -I "$READERm" --static-check --prereg "$PREREGm" --prereg-sha "$(fsha "$PREREG")" < /dev/null 2>&1 | tail -1)"; fi
  say "== STAND-INS for this dry run: every command below names the run's REAL pull path (<ART>/<run>/ckpt_last.pt, which does not exist yet); nothing is run; each argv is parsed by the pinned tool's own argparse"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; ck=$(ckpt_of "$arm" "$s")
    say "-- $(rundir "$arm" "$s") ($(label "$arm" "$s"); trainer --seed $(( s + $( [ "$arm" = launchintentrep ] && echo 10 || echo 0 ) )); control $(case "$arm:$s" in launchintent:*) echo "DAG-OACC_s$s" ;; *) echo "REP_s$s" ;; esac))"
    for step in "causal:T1-kin causal (decisive)" "logged:T1-kin logged (K4)" "t0fresh:fresh-age T0 (K4)"; do k=${step%%:*}
      case $k in causal|logged) tg=$(tagof "$arm" "$s" "$k"); tool="$BASE/stack/scripts/v8f_t1_eval.py" ;; t0fresh) tg=$(t0tag "$arm" "$s"); tool="$BASE/stack/scripts/v8f_mini_eval.py" ;; esac
      mkcmd "$k" "$ck" "$tg"; pr=$("$PY" -I "$LIB" parsecheck --tool "$(m "$tool")" -- "${CMD[@]:2}" < /dev/null 2>&1 | head -1 | cut -c1-12)
      say "  [${step#*:}] parse: ${pr:-?}  $(fmt "${CMD[@]}")"
    done
  done
  say "  each rollout writes $OUTm/t1v8f_<tag>.json, ticks/ticks_<tag>.npz, dump_<tag>/ep*.npz (>= $NDUMP), argv/<tag>.argv, then done_<tag>.txt (ckpt sha256 before / after, tool digest, command sha256); each fresh eval writes t0fresh/eval_<tag>.{json,npz} and done_t0f_<tag>.txt"
  say "== after the 12 steps: box-2 ZZV8FMINI-DONE (after the last pull), every pull re-verified, then the manifest (CTRL = banked DAG-OACC, REP = banked DAG-ORC r2 REP; K-BUILD from git objects at the tokens' commit):"
  out=$("$PY" -I "$LIB" kbuild --mirror "$MIRRORm" --launch "$LAUNCH_COMMIT" --reader "$READERm" 2>&1 < /dev/null | tail -1); say "   K-BUILD at ${LAUNCH_COMMIT:0:12}, the reader's own k_build: ${out:0:500}"
  mkcmd reader "" ""; say "== reader: $(fmt "${CMD[@]}")"
  say "   -> $OUTm/read/LAUNCH_INTENT_READ.json (rc 0 outcome written | 2 REFUSED, nothing written | 3 UNPLACED | 4 BLOCKED); reader stdout/stderr -> $LOGm/launch_intent_read.out"
  say "== markers in $LOGm/chain.out (anchored): ZZLIT1-STAGE-DONE <k>/4 <run> ... (one per run) | ZZLIT1-DONE <n>/12 steps, reader rc <rc> | ZZLIT1-STOP <why>.  NOTHING was run, written, launched or waited for (no rollout)."
  echo "ZZLIT1-DRY-OK"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ the real run
[ -f "$B2OUT" ] && say "box-2 log present" || say "the box-2 log $B2OUTm does not exist yet (the box-2 chain has not started): waiting for it, by script polling"
say "== sequential: ${#RUNS[@]} runs x (causal, logged, fresh T0) = $(( ${#RUNS[@]} * 3 )) steps, one process at a time, OMP/MKL $OMP_NUM_THREADS threads; box-2 ZZV8FMINI-STOP lines already in the log: $B2STOP0; tool digest ${TOOLD:0:16}..."
COMPLETE=(); K=0; t_wait0=$(date +%s); npoll=0; PROC_MSG=""
while [ "$K" -lt "${#RUNS[@]}" ]; do
  nostop
  [ -f "$B2OUT" ] && b2_pin "waiting"
  out=$("$PY" -I "$LIB" boxlog --log "$B2OUTm" --art "$ARTm" next --complete "$(IFS=,; echo "${COMPLETE[*]:-}")" 2>&1 < /dev/null); rc=$?
  [ $rc -eq 3 ] && stop "the box-2 chain's record FAILED a control (a run that failed its controls is never read): $out"
  if [ $rc -eq 0 ]; then
    set -- $out; arm=$2; s=$3; lsha=$4; g=$(rundir "$arm" "$s")
    if process_run "$arm" "$s" "$lsha"; then
      COMPLETE+=("$g"); K=$((K + 1)); t_wait0=$(date +%s)
      echo "ZZLIT1-STAGE-DONE $K/${#RUNS[@]} $g ($(label "$arm" "$s")): causal, logged, t0fresh done (ckpt ${PULL_SHA:0:12})" >> "$LOGu/chain.out"; tail -1 "$LOGu/chain.out"
      continue
    fi
    out=$PROC_MSG
  fi
  [ $((npoll % 30)) -eq 0 ] && say "waiting ($(( ($(date +%s) - t_wait0) / 60 )) min since the last run finished; $K/${#RUNS[@]} runs done): $out"
  npoll=$((npoll + 1))
  [ "$(b2_stops)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP after this chain started ($B2OUTm): no more pulls are coming ($(grep -a '^ZZV8FMINI-STOP' "$B2OUT" | tail -1 | cut -c1-200))"
  [ $(( $(date +%s) - t_wait0 )) -gt "$MAXWAIT_S" ] && stop "timeout: no run became ready in $((MAXWAIT_S / 3600)) h ($out)"
  sleep "$POLL"
done
pin_check_end "end"
n=0; for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; for r in "done_$(tagof "$arm" "$s" causal).txt" "done_$(tagof "$arm" "$s" logged).txt" "done_t0f_$(t0tag "$arm" "$s").txt"; do [ -s "$OUTu/$r" ] && n=$((n + 1)); done; done
NEED=$(( ${#RUNS[@]} * 3 ))
[ "$n" -eq "$NEED" ] || stop "$n of $NEED steps have a done record"
say "== all $n steps done; waiting for the box-2 DONE marker"
b2_done_wait
verify_pulls
say "== manifest"
MC=$(cat "$OUTu/LAUNCH_COMMIT.txt")
"$PY" -I "$LIB" manifest --art "$ARTm" --out "$OUTm" --ctrl-runs "$CTRL_T1m/runs.json" --rep-runs "$REP_OUTm/runs.json" --mirror "$MIRRORm" --launch "$MC" --tree "$TTm" --tool-digest "$TOOLD" --n-dump-min "$NDUMP" --write "$OUTm/runs.json" > "$LOGu/manifest.out" 2>&1 < /dev/null || stop "manifest: $(tr '\n' ' ' < "$LOGu/manifest.out" | cut -c1-700)"
say "$(tail -1 "$LOGu/manifest.out")"
mkdir -p "$OUTu/read"; mkcmd reader "" ""
pin_check_end "before the reader"
say "== reader: $(fmt "${CMD[@]}")"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${CMD[@]}" > "$LOGu/launch_intent_read.out" 2>&1 < /dev/null; RRC=$?
say "reader rc=$RRC: $(grep -a -E '"OUTCOME"|REFUSED|BLOCKED|UNPLACED|wrote ' "$LOGu/launch_intent_read.out" | head -3 | tr '\n' ' ' | cut -c1-400)"
pin_check_end "after the reader"
echo "ZZLIT1-DONE $n/$NEED steps, reader rc $RRC ($(case $RRC in 0) echo 'outcome row written' ;; 2) echo 'REFUSED, nothing written' ;; 3) echo 'UNPLACED, written' ;; 4) echo 'BLOCKED, a required slot incomplete' ;; *) echo 'reader crashed' ;; esac)): $OUTm/read/LAUNCH_INTENT_READ.json  log $LOGm/launch_intent_read.out" >> "$LOGu/chain.out"
tail -1 "$LOGu/chain.out"
