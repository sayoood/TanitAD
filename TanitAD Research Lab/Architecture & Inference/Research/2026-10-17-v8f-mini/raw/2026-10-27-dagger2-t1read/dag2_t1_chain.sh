#!/bin/bash
# PREREG_V8F_DAGGER2 (Project Steering/PREREG_V8F_DAGGER2.md @ 1446333d = as amended by Amendment 1, A1-2026-10-27, sha256 c4239e45...): the DEV-BOX CPU part, as ONE detached chain (ZERO training, ZERO GPU, box 2 is never touched).
# DERIVED from dag_t1_chain.sh (the chain that ran PREREG_V8F_DAGGER's four runs end to end) by mk_dag2_helpers.py (chain.diff).  BUILT AND SELF-TESTED, NOT LAUNCHED.
#   for each of the four runs, in the PREREG's order:  DAG2 s0, DAG2-0 s0, DAG2 s1, DAG2-0 s1      (arms dag2 / dag20; box 2 pulls them to <ART>/M2b_s<S>_dag2 and <ART>/M2b_s<S>_dag20)
#     1. wait until the pull is COMPLETE by content (dag2t1_pullcheck.py: SHA256SUMS.txt lists EXACTLY the 8 names, every file re-hashes, summary.json done, and the run's own records say it IS
#        this arm and seed: init = DAG_s<S>, dag2 names all four dagger flags with the iteration-2 bank, dag20 only --dagger-p 0).  A dag20 directory holding DAG2's run is a STOP.
#     2. two T1-kin rollouts on eval-141, plain, CPU, DAG's exact settings and tool tree:  --limit causal (decisive) and --limit logged (K4 only), tags DAG2_s<S>_m10_<limit>_heads / DAG20_...
#     3. the fresh-age T0 eval of the checkpoint on CPU (K4's comparator).  Box 2 runs only the standard eval (R-T0).
#   two lanes (A: every causal rollout, B: every logged rollout + the fresh evals), so AT MOST 2 rollout processes at once; CUDA_VISIBLE_DEVICES=-1.
#   ORC-0 is not used: PREREG_V8F_DAGGER2 change 3 fixes RF (no replicate of DAG2-0 is trained).  DAG's banked outputs (v8f-dagger-t1/, dagreread/) are READ, never re-run: they are the DAG - P10 baseline of the
#   newly-worse clause.  The chain checks at start that they are present and are the pinned files (dag2t1_manifest.py dagbase).
#   after all four runs: the bank gate (both pulled <ART>/bank_s<S>/ COMPLETE: 11 files each, bank.json == the --dagger-bank-sha256 box 2's gate declared for DAG2 s<S>), dag2t1_manifest.py writes runs.json,
#   then dag2_read.py (which runs the DAGGER reader on DAG2 / DAG2-0 and Read A's rows) writes DAG2_READ.json.
#
#   bash dag2_t1_chain.sh --dry-run    prints every guard result, path, pull state, the EXACT command of every rollout / eval, the manifest and the reader command; touches nothing.
#   bash dag2_t1_chain.sh              the real chain.  RESUME = re-run the same command (a finished rollout is skipped by a done record naming THIS checkpoint's sha256 and the tool digest;
#                                      an incomplete earlier attempt is MOVED to <OUT>/_incomplete_*, never deleted).
# Markers (own log, <LOG>/chain.out):  ZZDAG2T1-DONE <n>/8 rollouts, reader rc <rc> ...  |  ZZDAG2T1-STOP <reason>   (STOP also touches <LOG>/STOP; the other lane stops at its next step).
#   Reader rc: 0 = an outcome row was written (FIXED / PARTIAL / NO EFFECT / HARMS), 2 = REFUSED (a control failed; nothing written), 3 = UNPLACED (defensive: unreachable since Amendment 1 made the table complete), 4 = BLOCKED (written).
#
# WHICH TREE:  DAG2T1_TREE = the tool tree DAG's rollouts used (default E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a), so DAG and DAG2 are scored by ONE tool.  The chain asserts it carries the DAgger
#   trainer (lever greps) and is IDENTICAL to the tree that rolled DAG out (the tree= line of <DAG_T1>/PINS.txt; same directory = trivially identical; another directory = file-by-file parity).
# DISK (MEASURED 2026-10-27): DAG's T1 outputs are 1.3 GB (8 rollouts x ~150 MB + ticks + T0).  E: has ~1.7 GB free: the chain gates on E: (DAG2T1_E_MIN_MB, default 1500 at start, 200 before each rollout) in
#   addition to the registered C: >= 2048 MB gate.  If E: is short the chain WAITS (up to 2 h) for space, then STOPs; it never deletes anything.
# Environment (all optional):
#   DAG2T1_TREE  DAG2T1_READER (dag2_read.py)  DAG2T1_DAG_READER (dag_read.py, pinned)  DAG2T1_READERS (dagreread/reader_src)  DAG2T1_READA_CODE (dag2t1/readA_code)  DAG2T1_LEAD_BLOCK
#   DAG2T1_ART (box 2's pull dir, default E:/Projects/TanitAD-artifacts/v8f-dagger2)   DAG2T1_OUT / DAG2T1_LOG   DAG2T1_DAG_ART / DAG2T1_DAG_T1 (DAG's banked pull / T1 outputs)
#   DAG2T1_PREREG2 (PREREG_V8F_DAGGER2.md; sha256-pinned)   DAG2T1_B2_OUT (box-2 chain log; a ZZV8FMINI-STOP after this chain started stops it)   DAG2T1_MAX_WAIT_H (40)   DAG2T1_BANK_WAIT_H (12)
#   DAG2T1_C_MIN_MB (2048)   DAG2T1_E_MIN_MB (1500)   DAG2T1_M2B_SHA_JSON (box-2 ckpt_sha256.json cross-check)   DAG2T1_TEST_CFREE_MB / DAG2T1_TEST_EFREE_MB (--dry-run only)
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
DRY=0
case "${1:-}" in --dry-run) DRY=1 ;; "") ;; *) echo "usage: dag2_t1_chain.sh [--dry-run]   (configuration = DAG2T1_* environment variables, see the header)"; exit 2 ;; esac
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); Hm=$(cygpath -m "$H")
SMOKE=0
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
TT=${DAG2T1_TREE:-E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a}
READER=${DAG2T1_READER:-$Hm/dag2_read.py}                                                       # dag2_read.py (this directory)
DAGR=${DAG2T1_DAG_READER:-C:/Users/Admin/dagread/dag_read.py}                                    # the DAGGER reader dag2_read imports (sha256-pinned inside dag2_read.py)
RSRC=${DAG2T1_READERS:-/e/Projects/TanitAD-artifacts/dagreread/reader_src}                       # Read A's reader sources (the files committed at bdf8d561)
RCODE=${DAG2T1_READA_CODE:-$Hm/readA_code}                                                       # Read A's driver code (rr_lib.py, readA.py, build_report.py; sha256-pinned copies of dagreread/code/)
LEADBLK=${DAG2T1_LEAD_BLOCK:-$Hm/inputs/b1_eval_lead_block_k30.npz}                                 # a sha256-verified copy (7a879baa...) of the repo blob 1d4f4fbb:<v8f-mini>/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz, staged next to this chain (the old default lived in a session scratchpad)
SF=${DAG2T1_INPUTS:-/e/Projects/TanitAD-artifacts/sfresh_inputs}                                 # on E: (the Master Mind's md5-verified copy of the T1-HEADS inputs; nothing of this chain lives on C:)
DAG_ART=${DAG2T1_DAG_ART:-/e/Projects/TanitAD-artifacts/v8f-dagger}; DAG_T1=${DAG2T1_DAG_T1:-/e/Projects/TanitAD-artifacts/v8f-dagger-t1}     # DAG's banked pull / T1 outputs (READ only)
REL=${DAG2T1_TRAIN_RELEASE:-/e/Projects/TanitAD-artifacts/_s2build-copy-20260919/release/v72/s2_labels_v7.2_train.jsonl.gz}
PREREG_SHA=${DAG2T1_PREREG_SHA:-a4df4b8f44d9a322a1907338c990dee09d03fb35e1e136e6c60eb72a790f60f9}   # PREREG_V8F_DAGGER.md (landed text; the DAGGER reader checks its table cells in it)
PREREG2=${DAG2T1_PREREG2:-$Hm/inputs/PREREG_V8F_DAGGER2.md}; PREREG2_SHA=c4239e4590ee26e30ce6f525987aca82e3fb8ff034b7264321e2d0faa724f1c6   # PREREG_V8F_DAGGER2.md @ 1446333d (Amendment 1): the staged blob copy; dagland/'s copy is the as-landed 1d4f4fbb text
TAGP=""; PFX=""; MAXW=0; NDUMP=141; POLL=60; DONE_MARK=ZZDAG2T1-DONE; REP=""
ART=${DAG2T1_ART:-/e/Projects/TanitAD-artifacts/v8f-dagger2}; OUT=${DAG2T1_OUT:-/e/Projects/TanitAD-artifacts/v8f-dagger2-t1}; LOG=${DAG2T1_LOG:-$H/logs}
RUNS=(dag2:0 dag20:0 dag2:1 dag20:1)     # FOUR runs, ALWAYS ALL FOUR, in the PREREG's order (DAG2 s0, DAG2-0 s0, DAG2 s1, DAG2-0 s1); no run is conditional on another's read
RUNSCSV=$(IFS=,; echo "${RUNS[*]}")
m() { cygpath -m "$1"; }
TTu=$(cygpath -u "$TT"); TTm=$(m "$TT"); ARTu=$(cygpath -u "$ART"); ARTm=$(m "$ART"); OUTu=$(cygpath -u "$OUT"); OUTm=$(m "$OUT"); LOGu=$(cygpath -u "$LOG"); LOGm=$(m "$LOG")
SFu=$(cygpath -u "$SF"); SFm=$(m "$SF"); DAG_ARTm=$(m "$DAG_ART"); DAG_T1m=$(m "$DAG_T1"); DAG_T1u=$(cygpath -u "$DAG_T1"); RELm=$(m "$REL"); READERm=$(m "$READER"); DAGRm=$(m "$DAGR"); RSRCm=$(m "$RSRC")
RCODEm=$(m "$RCODE"); LEADBLKm=$(m "$LEADBLK"); PREREG2m=$(m "$PREREG2")
TOOL="$TTu/stack/scripts/v8f_t1_eval.py"
T1HD="$TTm/$PK/raw/2026-10-26-t1hdread/t1hd_read.py"; T1K="$TTm/$PK/raw/2026-10-26-t1kin-hacc/t1k_read.py"; PREREG="$TTm/Project Steering/PREREG_V8F_DAGGER.md"
GRID="$TTm/$PK/raw/eval_grid_windows.json"; JOIN="$TTm/$PK/raw/join_eval141.npz"; KIN="$TTm/$PK/raw/eval141_kin.npz"; TRACKS="$SFm/inputs/native_tracks.pt"; CACHEm="$SFm/cache"
V9="$SFm/inputs/v9"; P10="$SFm/inputs/p10/p10_prior.pt"; T0DIR="$OUTm/t0fresh"; IDS="$OUTm/ids"
MAXWAIT_S=$(( ${DAG2T1_MAX_WAIT_H:-40} * 3600 )); BANKWAIT_S=$(( ${DAG2T1_BANK_WAIT_H:-12} * 3600 )); CMIN=${DAG2T1_C_MIN_MB:-2048}

if [ "$DRY" = 1 ]; then
  say() { echo "$(date '+%m-%d %H:%M:%S') ${TAGP}DRY: $*"; }
  stop() { say "$*"; echo "ZZDAG2T1-STOP $*"; exit 1; }
else
  mkdir -p "$LOGu" "$OUTu" "$OUTu/ticks"
  say() { echo "$(date '+%m-%d %H:%M:%S') ${TAGP}$*" | tee -a "$LOGu/chain.out"; }
  stop() { say "STOP: $*"; echo "ZZDAG2T1-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }
  rm -f "$LOGu/STOP"
  say "=== dag2_t1_chain start (tree $TTm, reader $READERm, ART $ARTm, OUT $OUTm; runs $RUNSCSV)"
fi
nostop() { [ "$DRY" = 1 ] && return 0; [ -e "$LOGu/STOP" ] && exit 1; return 0; }
fmt() { local o="" a; for a in "$@"; do case "$a" in *[!A-Za-z0-9_./:=,@%+-]*) o="$o '$a'" ;; *) o="$o $a" ;; esac; done; echo "${o# }"; }
rundir() { echo "M2b_s${2}_${1}"; }                                   # rundir <arm> <seed>: the directory box 2's chain pulls into (M2b_s0_dag2, M2b_s0_dag20, ...)
armU() { if [ "$1" = dag2 ]; then echo DAG2; else echo DAG20; fi; }
tagof() { echo "${PFX}$(armU "$1")_s${2}_m10_${3}_heads"; }          # tagof <arm> <seed> <limit>: the rollout's --tag (T1-HEADS naming)
cfree_mb() {   # the TEST override exists only in --dry-run / SMOKE: a real run always reads the real C:
  if [ -n "${DAG2T1_TEST_CFREE_MB:-}" ] && { [ "$DRY" = 1 ] || [ "$SMOKE" = 1 ]; }; then echo "$DAG2T1_TEST_CFREE_MB"; else df -m /c | awk 'NR==2{print $4}'; fi
}
for r in "${RUNS[@]}"; do case "$r" in dag2:[01]|dag20:[01]) ;; *) stop "bad run spec '$r' (arm:seed, arm in dag2/dag20, seed 0/1)" ;; esac; done
[ "${DAG2T1_RUNS:-}" = "" ] || say "NOTE: DAG2T1_RUNS is ignored (the PREREG runs ALL FOUR, in its order)"
[ -z "${DAG2T1_B2_FRESH:-}" ] || stop "DAG2T1_B2_FRESH='$DAG2T1_B2_FRESH' is set, but that switch was REMOVED (decision O1, 2026-10-26): box 2 runs no fresh-age eval and a SHA256SUMS.txt that carries one is refused; the fresh-age T0 eval is a dev-box CPU step of this chain. Unset it."
CF0=$(cfree_mb)
[ "$CF0" -ge "$CMIN" ] || stop "C: has $CF0 MB free (< $CMIN MB): PREREG_V8F_DAGGER section 8 gates this chain on C: >= 2 GB free (the pagefile is on C:; a full C: killed torch start-ups before; inherited by PREREG_V8F_DAGGER2)"
say "C: gate: $CF0 MB free >= $CMIN MB"
EMIN=${DAG2T1_E_MIN_MB:-1500}; EROLL=${DAG2T1_E_ROLL_MB:-200}
efree_mb() { if [ -n "${DAG2T1_TEST_EFREE_MB:-}" ] && [ "$DRY" = 1 ]; then echo "$DAG2T1_TEST_EFREE_MB"; else df -m /e | awk 'NR==2{print $4}'; fi; }
EF0=$(efree_mb)
[ "$EF0" -ge "$EMIN" ] || stop "E: has $EF0 MB free (< $EMIN MB): DAG's T1 outputs were 1.3 GB (MEASURED) and this chain writes the same amount to $OUTm; free E: first (PI-only deletes)"
say "E: gate: $EF0 MB free >= $EMIN MB (the outputs are ~1.3 GB, MEASURED for DAG)"

# ------------------------------------------------------------------------------------------------ 0a  the tree: the DAgger lever must be in it, the tool flags must exist (grep + same-breath control; then the tool's own --help)
[ -f "$TOOL" ] || stop "no tool at $TTm/stack/scripts/v8f_t1_eval.py"
TRN="$TTu/stack/scripts/v8f_mini_train.py"; DGR="$TTu/stack/tanitad/data/v8f_dagger.py"
[ -f "$TRN" ] || stop "REFUSED: $TTm has no stack/scripts/v8f_mini_train.py (this must be the DAgger tip, not a tool-only tree)"
t_pos=$(grep -cE 'add_argument\("--init-ckpt"' "$TRN"); t_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$TRN")
[ "$t_pos" -ge 1 ] && [ "$t_neg" -eq 0 ] || stop "grep control broken on the trainer of $TTm (positive control --init-ckpt = $t_pos, must be >= 1; negative control = $t_neg, must be 0)"
for fl in --dagger-p --dagger-bank --dagger-bank-sha256 --dagger-seed; do
  n=$(grep -cE "add_argument\(\"$fl\"" "$TRN")
  [ "$n" -ge 1 ] || stop "REFUSED: $TTm is not the DAgger tip: the trainer has no $fl (grep $n; the control read $t_pos, so the grep works). Use the landed DAgger tip (or di/tree until it lands), not rt4/tree"
done
[ -f "$DGR" ] && [ "$(grep -cE '^class DaggerSwap' "$DGR")" -ge 1 ] || stop "REFUSED: $TTm has no stack/tanitad/data/v8f_dagger.py with class DaggerSwap: not the DAgger tip"
say "tree carries the DAgger trainer: --dagger-p / --dagger-bank / --dagger-bank-sha256 / --dagger-seed and class DaggerSwap (control --init-ckpt = $t_pos, no-such-flag = $t_neg) in $TTm"
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
[ "$DRY" = 1 ] || { mkdir -p "$OUTu/_tmp"; export TMP="$OUTm/_tmp" TEMP="$OUTm/_tmp"; }        # keep the temp files off C: (about 2 GB free)
tool_help() { "$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --help 2>&1 < /dev/null | tr -s '[:space:]' ' '; }
HELP=$(tool_help)
echo "$HELP" | grep -q 'usage:' || { say "the tool's --help printed no usage (a crashed import under memory pressure? began: ${HELP:0:200}); one retry in 30 s"; sleep 30; HELP=$(tool_help); }
echo "$HELP" | grep -q 'usage:' || stop "REFUSED: the tool's --help did not run (it printed no usage; began: ${HELP:0:300}): a crash is not 'the flag is missing', and the chain does not guess"
echo "$HELP" | grep -q -e '--dump-heads' || stop "REFUSED: the tool's own --help lists no --dump-heads (the --help output began: ${HELP:0:300})"
echo "$HELP" | grep -qE '\{logged,[^}]*causal' || stop "REFUSED: the tool's own --help lists no causal choice for --limit (the --help output began: ${HELP:0:300})"
say "tool --help agrees (--dump-heads, --limit {logged,...,causal,...})"

# ------------------------------------------------------------------------------------------------ 0b  the tool tree is the tree that rolled DAG out (one tool scores DAG and DAG2)
DAG_TT=${DAG2T1_DAG_TOOL_TREE:-}
[ -n "$DAG_TT" ] || DAG_TT=$(grep '^tree=' "$DAG_T1/PINS.txt" 2>/dev/null | head -1 | cut -c6-)
[ -n "$DAG_TT" ] && [ -d "$(cygpath -u "$DAG_TT")" ] || stop "DAG's tool tree is unknown (no tree= line in $DAG_T1/PINS.txt; set DAG2T1_DAG_TOOL_TREE), so tool identity cannot be checked"
if [ "$(m "$DAG_TT")" = "$TTm" ]; then
  PARITY_LINE="IDENTICAL: the tool tree IS the tree that rolled DAG out ($TTm)"
else
  PAR=$("$PY" -I "$Hm/dag2t1_manifest.py" parity --tree "$TTm" --ref "$(m "$DAG_TT")" 2>&1 < /dev/null); prc=$?
  PARITY_LINE=$(echo "$PAR" | head -1)
  [ $prc -eq 0 ] || stop "PARITY REFUSED: the T1 / T0 tools of $TTm are not the tools that rolled DAG out ($(m "$DAG_TT")): $(echo "$PAR" | tr '\n' ' ' | cut -c1-500)"
fi
say "tool identity with the tree that rolled DAG out: $PARITY_LINE"
if [ -d "$(cygpath -u "${DAG2T1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}")" ]; then
  TIPC=$("$PY" -I "$Hm/dag2t1_manifest.py" treetip --tree "$TTm" --git-dir "$(m "${DAG2T1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}")" --tip "${DAG2T1_TIP:-1d4f4fbb}" 2>&1 < /dev/null) || stop "tree vs tip: $(echo "$TIPC" | tr '\n' ' ' | cut -c1-500)"
  say "$TIPC"
else
  say "tree vs tip check SKIPPED: no tip mirror at ${DAG2T1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}"
fi

# ------------------------------------------------------------------------------------------------ 0c  inputs and pins
md5chk() { [ "$(md5sum < "$1" 2>/dev/null | cut -c1-32)" = "$2" ] || stop "md5 of $1 is not $2"; }
md5chk "$GRID" 34ed8ac56365eeff72e42581bde2f5c4
md5chk "$JOIN" 532fcd644c0e0835e847b2f1cee011ad
md5chk "$TRACKS" 189f4e68913c0163912a509203963f8d
md5chk "$KIN" c92ee56db60b972ec84518a4a73d9f83                           # eval141_kin.npz as the SFRESH / ORC fresh evals used it
[ "$(ls "$SFu/cache/eval/clips" 2>/dev/null | wc -l)" = 141 ] || stop "eval cache $CACHEm is not 141 clips (the T1-HEADS inputs are E:/Projects/TanitAD-artifacts/sfresh_inputs; DAG2T1_INPUTS names another copy; is the copy finished?)"
for f in "$READER" "$DAGR" "$T1HD" "$T1K" "$PREREG" "$PREREG2" "$REL" "$LEADBLK"; do [ -f "$f" ] || stop "input missing: $f (the lead block lives in the repo: git -C <mirror> show 1d4f4fbb:<v8f-mini>/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz)"; done
[ "$(sha256sum < "$PREREG2" | cut -c1-64)" = "$PREREG2_SHA" ] || stop "PREREG_V8F_DAGGER2.md at $PREREG2 is not the pinned text (sha256 ${PREREG2_SHA:0:12}...): the PREREG is binding, nothing may differ"
SC=$("$PY" -I "$Hm/dag2_read.py" --static-check --tool-root "$TTm" --dag-read "$DAGRm" --readers "$RSRCm" --reada-code "$RCODEm" --lead-block "$LEADBLKm" --prereg2 "$PREREG2m" 2>&1 < /dev/null) || stop "dag2_read static check: $(echo "$SC" | tr '\n' ' ' | cut -c1-500)"
say "dag2_read static check: $(echo "$SC" | tail -1) (pinned reader sources, Read A driver code, DAG's banked reads, the lead block, PREREG_V8F_DAGGER2 outcome-table cells verbatim)"
[ "$(sha256sum < "$PREREG" | cut -c1-64)" = "$PREREG_SHA" ] || stop "the tree's PREREG_V8F_DAGGER.md is not the landed text the reader was built for (sha256 ${PREREG_SHA:0:12}...): an amendment landed? re-pin deliberately (DAG2T1_PREREG_SHA) after re-reading the reader"
T0IN=("$V9/v9_labels_eval139.npz" "$V9/lead_eval139.npz" "$V9/refcv6_speed_max_v8_eval.jsonl" "$P10" "$TTu/stack/scripts/v8f_mini_eval.py" "$KIN")
for f in "${T0IN[@]}"; do [ -s "$f" ] || stop "fresh-T0 input missing: $f"; done
md5chk "$V9/v9_labels_eval139.npz" 6b5c7f207cffc3b7eb3cd527fd433599          # the md5 the box-2 chain asserts for the v9 eval release
grep -cE 'add_argument\("--ages"' "${T0IN[4]}" | grep -qv '^0$' || stop "REFUSED: the tree's v8f_mini_eval.py has no --ages (the fresh-age T0 eval needs it)"
tree_digest() {   # one sha256 over every .py the tool can import (stack/tanitad, taniteval) + the tool
  ( cd "$TTu" && { find stack/tanitad taniteval -name '*.py' -type f; echo stack/scripts/v8f_t1_eval.py; } | LC_ALL=C sort -u | tr '\n' '\0' | xargs -0 sha256sum | sha256sum | cut -c1-64 )
}
keyfiles="stack/scripts/v8f_t1_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/floors_kin.py taniteval/taniteval/four_families.py taniteval/taniteval/ci.py taniteval/tools/t1_eval.py stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_dagger.py"
fsha() { sha256sum < "$(cygpath -u "$1")" | cut -c1-64; }
TOOLD=$(tree_digest)
say "inputs OK (grid, join, kin, tracks, v9 md5; 141-clip eval cache; PREREG = the landed text); tool tree digest ${TOOLD:0:12}"
t0_pins() { local f; for f in "${T0IN[@]}"; do echo "t0in=$(basename "$f") $(sha256sum < "$f" | cut -c1-64)"; done; }
write_pins() {
  { echo "tree=$TTm"; echo "digest=$TOOLD"; for f in $keyfiles; do echo "file=$f $(sha256sum < "$TTu/$f" | cut -c1-64)"; done
    echo "reader=dag2_read.py $(fsha "$READER")"; echo "reader=dag_read.py $(fsha "$DAGR")"; echo "prereg2=$(fsha "$PREREG2")"; echo "reader=t1hd_read.py $(fsha "$T1HD")"; echo "reader=t1k_read.py $(fsha "$T1K")"; echo "reader=PREREG $(fsha "$PREREG")"
    echo "dag_tool_tree=${DAG_TT:-unknown}"; echo "parity=$PARITY_LINE"; echo "release=$(fsha "$REL")"; t0_pins; }
}
if [ "$DRY" = 1 ]; then
  say "pins that would be written to $OUTm/PINS.txt:"; write_pins | sed 's/^/    /'
else
  if [ ! -e "$OUTu/PINS.txt" ]; then write_pins > "$OUTu/PINS.txt"; say "pins written (tool digest ${TOOLD:0:12}; reader scripts, PREREG, release sha256)"; fi
  [ "$(grep ^tree= "$OUTu/PINS.txt" | cut -c6-)" = "$TTm" ] || stop "PINS.txt belongs to another TOOL TREE ($(grep ^tree= "$OUTu/PINS.txt")); delete it deliberately to re-pin"
fi
pin_check() {   # pin_check <when>
  [ "$DRY" = 1 ] && return 0
  [ "$(grep ^digest= "$OUTu/PINS.txt" | cut -c8-)" = "$(tree_digest)" ] || stop "the tool tree changed ($1)"
  local nm f
  for nm in dag2_read.py:"$READER" dag_read.py:"$DAGR" t1hd_read.py:"$T1HD" t1k_read.py:"$T1K" PREREG:"$PREREG"; do
    [ "$(grep "^reader=${nm%%:*} " "$OUTu/PINS.txt" | cut -d' ' -f2)" = "$(fsha "${nm#*:}")" ] || stop "reader input ${nm%%:*} changed ($1)"
  done
  for f in "${T0IN[@]}"; do [ "$(grep "^t0in=$(basename "$f") " "$OUTu/PINS.txt" | cut -d' ' -f2)" = "$(sha256sum < "$f" | cut -c1-64)" ] || stop "fresh-T0 input $(basename "$f") changed ($1)"; done
}
pin_check "start"

# ------------------------------------------------------------------------------------------------ 0d  DAG's BANKED outputs (read, never re-run) and the two id lists (fail now, not after hours of rollouts)
DB=$("$PY" -I "$Hm/dag2t1_manifest.py" dagbase --dag-art "$DAG_ARTm" --dag-t1 "$DAG_T1m" 2>&1 < /dev/null) || stop "DAG's banked outputs: $(echo "$DB" | tr '\n' ' ' | cut -c1-500)"
say "$DB"
if [ "$DRY" = 1 ]; then
  I0=$("$PY" -I "$Hm/dag2t1_manifest.py" ids --grid "$GRID" --release "$RELm" --out-dir "$IDS" --dry-run 2>&1 < /dev/null) || stop "id lists: $(echo "$I0" | tr '\n' ' ' | cut -c1-500)"
  say "$I0"
else
  I0=$("$PY" -I "$Hm/dag2t1_manifest.py" ids --grid "$GRID" --release "$RELm" --out-dir "$IDS" 2>&1 < /dev/null) || stop "id lists: $(echo "$I0" | tr '\n' ' ' | cut -c1-500)"
  say "$I0"
fi
cmin_wait() {   # cmin_wait <what>: block (up to 2 h) until C: has CMIN MB again
  local w=0 what=$1
  while [ "$(cfree_mb)" -lt "$CMIN" ]; do nostop; [ $w -eq 0 ] && say "$what: waiting for C: >= $CMIN MB free (now $(cfree_mb) MB; the pagefile is on C:)"; w=$((w + 1)); [ $w -ge 120 ] && stop "C: under $CMIN MB free for 2 h"; sleep 60; done
}

emin_wait() {   # emin_wait <what>: block (up to 2 h) until E: has EROLL MB again; never deletes anything
  local w=0 what=$1
  while [ "$(efree_mb)" -lt "$EROLL" ]; do nostop; [ $w -eq 0 ] && say "$what: waiting for E: >= $EROLL MB free (now $(efree_mb) MB)"; w=$((w + 1)); [ $w -ge 120 ] && stop "E: under $EROLL MB free for 2 h"; sleep 60; done
}
# ------------------------------------------------------------------------------------------------ 1  the rollouts
t1cmd() {   # t1cmd <ckpt> <tag> <limit>  ->  T1CMD (the array IS the command that runs; --dry-run prints it)
  T1CMD=("$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --ckpt "$1" --tag "$2" --tracks "$TRACKS" --eval-cache "$CACHEm" --eval-grid "$GRID" --eval-join "$JOIN" --out-dir "$OUTm"
         --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit "$3" --dump-ticks "$OUTm/ticks" --dump-heads --device cpu)
  [ "$MAXW" -gt 0 ] && T1CMD+=(--max-windows "$MAXW")
  return 0
}
b2count() { if [ -n "${DAG2T1_B2_OUT:-}" ] && [ -f "$DAG2T1_B2_OUT" ]; then grep -c '^ZZV8FMINI-STOP' "$DAG2T1_B2_OUT"; else echo 0; fi; }
B2STOP0=$(b2count)
PULL_SHA=""
wait_pull() {   # wait_pull <arm> <seed>: block until the pull is COMPLETE (sets PULL_SHA); stops on STOP, on a WRONG run, on a box-2 STOP that appeared after we started, on timeout
  local arm=$1 s=$2 t0 n=0 out rc
  t0=$(date +%s)
  while :; do
    nostop
    out=$("$PY" "$Hm/dag2t1_pullcheck.py" run "$ARTm" "$arm" "$s" 2>&1 < /dev/null); rc=$?
    if [ $rc -eq 0 ]; then PULL_SHA=$(echo "$out" | cut -d' ' -f2); say "pull of $(rundir "$arm" "$s") is COMPLETE: ckpt sha256 ${PULL_SHA:0:16}... ($(echo "$out" | cut -d' ' -f3-))"; return 0; fi
    [ $rc -eq 3 ] && stop "the pulled $(rundir "$arm" "$s") is WRONG, waiting cannot fix it: $out"
    [ $((n % 30)) -eq 0 ] && say "waiting for the pull of $(rundir "$arm" "$s") ($(( ($(date +%s) - t0) / 60 )) min): $out"
    n=$((n + 1))
    [ "$(b2count)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP after this chain started ($DAG2T1_B2_OUT): no more pulls are coming ($(grep '^ZZV8FMINI-STOP' "$DAG2T1_B2_OUT" | tail -1 | cut -c1-200))"
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
  cmin_wait "$tag"; emin_wait "$tag"
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
  printf '{"tag": "%s", "limit": "%s", "ckpt_sha256": "%s", "tool_digest": "%s", "n_dumps": %s, "wall_s": %s, "smoke": %s}\n' "$tag" "$lim" "$sha" "$TOOLD" "$(n_dumps "$tag")" "$(( $(date +%s) - t0 ))" "false" > "$OUTu/done_$tag.txt"
}
t0tag() { echo "${PFX}M2b-${1}_s${2}-fresh"; }                    # t0tag <arm> <seed>: the fresh-age T0 eval's --tag (files eval_<tag>.json / .npz)
t0cmd() {   # t0cmd <ckpt> <tag>  ->  T0CMD: v8f_mini_eval.py run --ages fresh, the SFRESH / orct1 argv (CPU, ~21-36 s) with this run's checkpoint
  T0CMD=("$PY" "$TTm/stack/scripts/v8f_mini_eval.py" run --ckpt "$1" --tag "$2" --ages fresh --v9-eval "$V9/v9_labels_eval139.npz" --lead-eval "$V9/lead_eval139.npz"
         --speed-eval "$V9/refcv6_speed_max_v8_eval.jsonl" --kin-mode yaw --p10 "$P10" --out-dir "$T0DIR" --eval-cache "$CACHEm" --tracks "$TRACKS" --eval-grid "$GRID" --eval-kin "$KIN" --eval-join "$JOIN" --project --families)
  return 0
}
t0run() {   # t0run <arm> <seed> <ckpt sha256>: runs inside the logged lane, so it never adds a third process
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
  cmin_wait "$tag"
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
lane() {   # lane <limit>: every run's rollout of that limit, in the PREREG's order, each as soon as its pull is complete (the logged lane also runs the fresh-T0 eval of the run)
  local lim=$1 spec arm s
  for spec in "${RUNS[@]}"; do
    arm=${spec%%:*}; s=${spec##*:}
    nostop; wait_pull "$arm" "$s"; t1run "$arm" "$s" "$lim" "$PULL_SHA"
    [ "$lim" = logged ] && t0run "$arm" "$s" "$PULL_SHA"
  done
  say "lane $lim done"
}

# ------------------------------------------------------------------------------------------------ 2  the bank gate, the manifest and the reader
bank_gate() {   # the reader loads every bank shard npz (clip names, K-B3): wait until both bank directories are COMPLETE by content (the 11-name SHA256SUMS.txt, re-hashed; bank.json == the sha256 the trainer gate declared), STOP on a WRONG bank or a timeout
  local s t0=$(date +%s) n out rc
  [ "$SMOKE" = 1 ] && { say "SMOKE: the bank gate is skipped"; return 0; }
  for s in 0 1; do
    n=0
    while :; do
      out=$("$PY" "$Hm/dag2t1_pullcheck.py" bank "$ARTm" "$s" --expect-shards 4 2>&1 < /dev/null); rc=$?
      [ $rc -eq 3 ] && stop "the pulled bank s$s is WRONG, waiting cannot fix it: $out"
      [ $rc -eq 0 ] && { say "bank s$s is COMPLETE: bank.json sha256 $(echo "$out" | cut -d' ' -f2 | cut -c1-16)... ($(echo "$out" | cut -d' ' -f3-))"; break; }
      [ $((n % 30)) -eq 0 ] && say "waiting for the bank s$s directory ($(( ($(date +%s) - t0) / 60 )) min): $out"
      n=$((n + 1)); nostop
      [ $(( $(date +%s) - t0 )) -gt "$BANKWAIT_S" ] && stop "the bank s$s directory is still incomplete after $((BANKWAIT_S / 3600)) h, so dag_read would REFUSE: $out"
      sleep "$POLL"
    done
  done
}
mk_reader_cmd() {   # mk_reader_cmd <init sha s0> <init sha s1>  ->  RDCMD   (the checkpoint DAG2 / DAG2-0 started from = DAG_s<S>, from the runs' own records)
  RDCMD=("$PY" -I "$READERm" --manifest "$OUTm/runs.json" --out "$OUTm/read/DAG2_READ.json" --tool-root "$TTm" --t1hd-read "$T1HD" --t1k-read "$T1K"
         --eval-cache "$CACHEm" --tracks "$TRACKS" --eval-grid "$GRID" --eval-join "$JOIN" --init-sha "0=$1" --init-sha "1=$2" --prereg "$PREREG" --prereg2 "$PREREG2m"
         --dag-read "$DAGRm" --readers "$RSRCm" --reada-code "$RCODEm" --lead-block "$LEADBLKm" --dag-t1 "$DAG_T1m")
}
MANI=("$PY" -I "$Hm/dag2t1_manifest.py" manifest --art "$ARTm" --out "$OUTm" --ids-dir "$IDS" --runs "$RUNSCSV" --n-dump-min "$NDUMP" --t0-fresh-dir "$T0DIR" --dag-art "$DAG_ARTm")
M2BS=("$PY" -I "$Hm/dag2t1_manifest.py" initsha --art "$ARTm" --dag-art "$DAG_ARTm" --runs "$RUNSCSV")
[ -n "${DAG2T1_M2B_SHA_JSON:-}" ] && M2BS+=(--b2-json "$(m "$DAG2T1_M2B_SHA_JSON")")

# ------------------------------------------------------------------------------------------------ --dry-run: print and leave
if [ "$DRY" = 1 ]; then
  say "C: free $(cfree_mb) MB (the chain needs >= $CMIN MB, at start and before every rollout); threads OMP/MKL $OMP_NUM_THREADS, CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES', max 2 rollout processes (lane causal + lane logged), poll $POLL s, max wait $((MAXWAIT_S / 3600)) h per pull, box-2 fresh eval: none accepted (decision O1: a SHA256SUMS.txt that carries one is refused)"
  say "box-2 stop watcher: $([ -n "${DAG2T1_B2_OUT:-}" ] && echo "$DAG2T1_B2_OUT (ZZV8FMINI-STOP lines now: $B2STOP0)" || echo 'OFF (DAG2T1_B2_OUT unset)')"
  say "== the four runs in the PREREG's order, and the state of each pull right now"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; say "  $(rundir "$arm" "$s")  $("$PY" "$Hm/dag2t1_pullcheck.py" run "$ARTm" "$arm" "$s" 2>&1 < /dev/null | cut -c1-220)"; done
  for s in 0 1; do say "  bank_s$s  $("$PY" "$Hm/dag2t1_pullcheck.py" bank "$ARTm" "$s" --expect-shards 4 2>&1 < /dev/null | cut -c1-260)"; done
  say "== rollouts (lane A = causal, lane B = logged; lane order = the run order; each starts when ITS pull is complete; ckpt = <ART>/<run>/ckpt_last.pt)"
  for lim in causal logged; do
    for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}
      t1cmd "$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt" "$(tagof "$arm" "$s" "$lim")" "$lim"; say "  [lane $lim] $(fmt "${T1CMD[@]}")"
    done
  done
  say "== fresh-age T0 eval (runs in the logged lane after that run's logged rollout; the manifest's t0_fresh, K4's comparator; t0 stays box 2's standard eval)"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; t0cmd "$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt" "$(t0tag "$arm" "$s")"; say "  $(fmt "${T0CMD[@]}")"; done
  say "  each rollout writes $OUTm/t1v8f_<tag>.json, $OUTm/ticks/ticks_<tag>.npz, $OUTm/dump_<tag>/ep*.npz (>= $NDUMP), then $OUTm/done_<tag>.txt (checkpoint sha256 + tool digest)"
  say "== DAG banked outputs ($DAG_T1m, $DAG_ARTm, dagreread/) are READ for the DAG - P10 baseline of the newly-worse clause; NOT re-run.  RF is fixed (PREREG change 3): no ORC-0, no replicate"
  say "== manifest: $(fmt "${MANI[@]}" --write "$OUTm/runs.json")"
  "${MANI[@]}" --dry-run | sed 's/^/    /'      # the SAME argv the real run writes with
  say "== --init-sha: $(fmt "${M2BS[@]}")   (the runs' own summary.json / selfcheck.json levers.init_ckpt.sha256: DAG2 and DAG2-0 equal per seed and equal to DAG_s<S>'s ckpt, d186fd0b... / 1d12ec2e...)"
  mk_reader_cmd '<0: DAG_s0 ckpt sha256>' '<1: DAG_s1 ckpt sha256>'
  say "== bank gate: both <ART>/bank_s<S>/ directories must be COMPLETE before the reader runs: SHA256SUMS.txt over EXACTLY the 11 names (PROVENANCE.json, bank.json, 4 shard provenance, 4 shard npz, bank_stage.log), every file re-hashed, bank.json == the --dagger-bank-sha256 in <ART>/M2b_s<S>_dag/gate/pin.txt (wait up to $((BANKWAIT_S / 3600)) h; WRONG = STOP)"
  say "== reader: $(fmt "${RDCMD[@]}")"
  say "   -> $OUTm/read/DAG2_READ.json (rc 0 outcome written | 2 REFUSED, nothing written | 3 UNPLACED | 4 BLOCKED); reader stdout/stderr -> $LOGm/dag2_read.out"
  say "== markers in $LOGm/chain.out: $DONE_MARK <n>/$(( ${#RUNS[@]} * 2 )) rollouts, reader rc <rc> ...  |  ZZDAG2T1-STOP <why>.  Nothing was written, launched or waited for."
  echo "ZZDAG2T1-DRY-OK"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ the real run
say "== lanes: A = --limit causal (decisive), B = --limit logged (K4 only) + the fresh-age T0 evals, $(( ${#RUNS[@]} )) runs each, OMP/MKL $OMP_NUM_THREADS threads each, max 2 processes"
lane causal & PA=$!
lane logged & PB=$!
wait $PA $PB
[ -e "$LOGu/STOP" ] && { say "a lane stopped: no DONE"; exit 1; }
pin_check "end"
n=0; for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; for lim in causal logged; do [ -s "$OUTu/done_$(tagof "$arm" "$s" "$lim").txt" ] && n=$((n + 1)); done; done
NEED=$(( ${#RUNS[@]} * 2 ))
[ "$n" -eq "$NEED" ] || stop "$n of $NEED rollouts have a done record"
for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; [ -s "$OUTu/done_t0f_$(t0tag "$arm" "$s").txt" ] || stop "no done record for the fresh-T0 eval of $(rundir "$arm" "$s")"; done
say "== all $n rollouts and all fresh-T0 evals done; bank gate"
bank_gate
say "== manifest"
"${MANI[@]}" --write "$OUTm/runs.json" > "$LOGu/manifest.out" 2>&1 < /dev/null || stop "manifest: $(tr '\n' ' ' < "$LOGu/manifest.out" | cut -c1-600)"
say "$(tail -1 "$LOGu/manifest.out")"
SHAS=$("${M2BS[@]}" 2> "$LOGu/m2bsha.err" < /dev/null) || stop "m2b sha: $(tr '\n' ' ' < "$LOGu/m2bsha.err" | cut -c1-300)"
S0=$(echo "$SHAS" | tr ' ' '\n' | grep '^0=' | cut -c3-); S1=$(echo "$SHAS" | tr ' ' '\n' | grep '^1=' | cut -c3-)
[ ${#S0} -eq 64 ] && [ ${#S1} -eq 64 ] || stop "m2b sha values malformed: '$SHAS'"
mk_reader_cmd "$S0" "$S1"
pin_check "before the reader"
say "== reader: $(fmt "${RDCMD[@]}")"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${RDCMD[@]}" > "$LOGu/dag2_read.out" 2>&1 < /dev/null; RRC=$?
say "reader rc=$RRC: $(grep -a -E '"OUTCOME"|REFUSED|UNPLACED|BLOCKED|wrote ' "$LOGu/dag2_read.out" | head -3 | tr '\n' ' ' | cut -c1-400)"
pin_check "after the reader"
echo "$DONE_MARK $n/$NEED rollouts, reader rc $RRC ($(case $RRC in 0) echo 'outcome row written' ;; 2) echo 'REFUSED, nothing written' ;; 3) echo UNPLACED ;; 4) echo 'BLOCKED, written' ;; *) echo 'reader crashed' ;; esac)): $OUTm/read/DAG2_READ.json  log $LOGm/dag2_read.out" >> "$LOGu/chain.out"
tail -1 "$LOGu/chain.out"
