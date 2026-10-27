#!/bin/bash
# PREREG_V8F_FLATDAG (Project Steering/PREREG_V8F_FLATDAG.md @ 9b91b654, raw-blob sha256 8713d0a3...): the DEV-BOX CPU part, as ONE detached chain (ZERO training, ZERO GPU, box 2 is never touched).  BUILT AND SELF-TESTED, NOT LAUNCHED.
# DERIVED from dag2_t1_chain.sh (the chain that ran PREREG_V8F_DAGGER2's eight rollouts end to end): sections 0a-0d are the same guards; the rollout and read sections are FLAT-DAG's (chain.diff is the change).
#   PREREG section 5, verbatim: "Dev-box CPU: the 4 new T1-kin rollouts (DAG's are banked) and the reader."  So:
#   for each of the four runs:  FLAT-DAG s0, FLAT-DAG-0 s0, FLAT-DAG s1, FLAT-DAG-0 s1      (arms flatdag / flatdag0; box 2 pulls them to <ART>/M2b_s<S>_flatdag and <ART>/M2b_s<S>_flatdag0)
#     1. wait until the pull is COMPLETE by content (flatdag_t1_pullcheck.py: SHA256SUMS.txt lists EXACTLY the 9 names, every file re-hashes, summary.json done, and the run's own records say it IS this arm and seed:
#        init = FLAT_s<S>, --o-interface cut named, flatdag names all four dagger flags with FLAT's OWN bank (dagger_bank_flat), flatdag0 only --dagger-p 0).  A flatdag0 directory holding FLAT-DAG's run is a STOP.
#     2. ONE T1-kin rollout on eval-141, plain, CPU, the decisive settings (--limit causal) and the SAME tool tree that rolled DAG and FLAT out (B-2 of PREREG_V8F_DAG_REREAD): FLAT's rollout argv except
#        --ckpt / --tag / --out-dir / --dump-ticks (the reader's control C-ARGV compares the argv this chain writes to <OUT>/argv/<tag>.argv with dagreread/argv/FLAT_s<S>_m10_causal_heads.argv).
#   NO logged-bucket rollout and NO fresh-age T0 eval: they feed only the T1-HEADS reducer's K4, which is not among PREREG_V8F_FLATDAG's controls (C-F1..C-F4); the reader reports K4 as NOT RUN beside H-T1-HEADS.
#   two lanes (one per seed), so AT MOST 2 rollout processes at once; CUDA_VISIBLE_DEVICES=-1.
#   DAG's banked outputs (v8f-dagger-t1/) and FLAT's (dagreread/packed, raw/B2_T1kin_DAG_vs_FLAT.json, argv/) are READ, never re-run.
#   after all four rollouts: the box-2 chain's ZZV8FMINI-DONE (FDT1_B2_OUT, read-only), every pull re-verified, the bank gate (both pulled <ART>/bank_s<S>/ COMPLETE: 11 files each, bank.json == the --dagger-bank-sha256 box 2's gate
#   declared for FLAT-DAG s<S>), flatdag_t1_manifest.py writes runs.json, then flatdag_read.py writes <OUT>/read/FLATDAG_READ.json.
#
#   bash flatdag_t1_chain.sh --dry-run    prints every guard result, path, pull state, the EXACT command of every rollout and the reader command; touches nothing.
#   bash flatdag_t1_chain.sh              the real chain.  RESUME = re-run the same command (a finished rollout is skipped by a done record naming THIS checkpoint's sha256 and the tool digest;
#                                         an incomplete earlier attempt is MOVED to <OUT>/_incomplete_*, never deleted).
# Markers (own log, <LOG>/chain.out):  ZZFLATDAGT1-DONE <n>/4 rollouts, reader rc <rc> ...  |  ZZFLATDAGT1-STOP <reason>   (STOP also touches <LOG>/STOP; the other lane stops at its next step).
#   Reader rc: 0 = an outcome row was written (HIERARCHY BEATS FLAT / FLAT BETTER / NOT ESTABLISHED), 2 = REFUSED (a control failed; nothing written), 3 = UNPLACED (defensive: unreachable, the table is complete over its cells), 4 = BLOCKED (written).
#
# WHICH TREE:  FDT1_TREE = the tool tree DAG's AND FLAT's T1-kin rollouts used (default E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a: FLAT's B-2 rollout used it, rollout_lane.sh; its model is built from the checkpoint's own
#   cfg, `V8FMiniConfig(**ck["cfg"])` (v8f_t1_eval.py:506), so an `o_interface: cut` checkpoint is rolled out with the cut, v8f_mini.py:764).  The chain asserts it carries the DAgger trainer (lever greps) and is IDENTICAL to the tree that rolled DAG out.
# DISK: the four rollouts write ~150 MB each to E:.  The chain gates on C: >= 2048 MB (the registered gate) and E: >= 1500 MB at start (200 MB before each rollout); if short it WAITS (up to 2 h) and then STOPs; it never deletes anything.
# Environment (all optional):
#   FDT1_TREE  FDT1_READER (flatdag_read.py)  FDT1_DAG_READER (dag_read.py, pinned)  FDT1_DAG2_READER (dag2_read.py, pinned; flatdag_read imports it)  FDT1_READERS (dagreread/reader_src)  FDT1_READA_CODE  FDT1_LEAD_BLOCK
#   FDT1_ART (box 2's pull dir, default E:/Projects/TanitAD-artifacts/v8f-flatdag)  FDT1_OUT / FDT1_LOG  FDT1_DAG_ART / FDT1_DAG_T1 (DAG's banked pull / T1 outputs)  FDT1_DAGREREAD (FLAT's banked B-2 package)
#   FDT1_B2_OUT (box-2 chain log, default C:/Users/Admin/box2_ship/logs/chain_v8f_flatdag.out: a ZZV8FMINI-STOP after this chain started stops it; the reader waits for its ^ZZV8FMINI-DONE; set FDT1_B2_OUT=off to skip)
#   FDT1_MAX_WAIT_H (40)  FDT1_C_MIN_MB (2048)  FDT1_E_MIN_MB (1500)  FDT1_TEST_CFREE_MB / FDT1_TEST_EFREE_MB (--dry-run only)
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
DRY=0
case "${1:-}" in --dry-run) DRY=1 ;; "") ;; *) echo "usage: flatdag_t1_chain.sh [--dry-run]   (configuration = FDT1_* environment variables, see the header)"; exit 2 ;; esac
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); Hm=$(cygpath -m "$H")
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
AR=/e/Projects/TanitAD-artifacts
TT=${FDT1_TREE:-E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a}
READER=${FDT1_READER:-$Hm/flatdag_read.py}
DAGR=${FDT1_DAG_READER:-C:/Users/Admin/dagread/dag_read.py}                                      # the DAGGER reader flatdag_read imports (sha256-pinned inside flatdag_read.py)
D2R=${FDT1_DAG2_READER:-$AR/dag2t1/dag2_read.py}                                                 # dag2_read.py: Read A's driver class and helpers, imported (sha256-pinned inside flatdag_read.py)
RSRC=${FDT1_READERS:-$AR/dagreread/reader_src}                                                   # Read A's reader sources (the files committed at bdf8d561)
RCODE=${FDT1_READA_CODE:-$Hm/readA_code}                                                         # Read A's driver code (rr_lib.py, readA.py, build_report.py; sha256-pinned copies of dagreread/code/)
LEADBLK=${FDT1_LEAD_BLOCK:-$Hm/inputs/b1_eval_lead_block_k30.npz}                                 # sha256-verified copy (7a879baa...) of the repo blob <v8f-mini>/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz
SF=${FDT1_INPUTS:-$AR/sfresh_inputs}                                                             # on E: (the Master Mind's md5-verified copy of the T1-HEADS inputs)
DAG_ART=${FDT1_DAG_ART:-$AR/v8f-dagger}; DAG_T1=${FDT1_DAG_T1:-$AR/v8f-dagger-t1}               # DAG's banked pull / T1 outputs (READ only)
DAGRR=${FDT1_DAGREREAD:-$AR/dagreread}                                                           # FLAT's banked B-2 package (packed dumps, B2 table, rollout argv): READ only
FLATCK=${FDT1_FLAT_DIR:-$AR/v8f-mini-flat}
REL=${FDT1_TRAIN_RELEASE:-$AR/_s2build-copy-20260919/release/v72/s2_labels_v7.2_train.jsonl.gz}
PREREG_SHA=${FDT1_PREREG_SHA:-a4df4b8f44d9a322a1907338c990dee09d03fb35e1e136e6c60eb72a790f60f9}  # PREREG_V8F_DAGGER.md (the tree's landed text; the DAGGER reader checks its table cells in it)
PREREGF=${FDT1_PREREGF:-$Hm/inputs/PREREG_V8F_FLATDAG.md}; PREREGF_SHA=8713d0a38d8dfe1283f03e3865421c8f066c1700a8339fd9d59bdd5868810c46      # PREREG_V8F_FLATDAG.md @ 9b91b654 (raw blob)
PREREGR=$Hm/inputs/PREREG_V8F_DAG_REREAD.md; PREREGR_SHA=7acaa4dd25373d554d9d2cdc68d0c9b94687167e343cceec0e8cf0939d7657fa                       # Read A's rows (section 2) and C1-C4 (section 4), as at 9b91b654
PREREGD=$Hm/inputs/PREREG_V8F_DAGGER_9b91b654.md; PREREGD_SHA=c99a881e0d010af8604a21dfb949491ff2aaeab8d60cc7077c4777e5197b2945                 # DAGGER's R-ADE / R-T0 guards, K-B0..K-B5, COVERAGE, with Amendments 1 and 2
MAXW=0; NDUMP=141; POLL=60; DONE_MARK=ZZFLATDAGT1-DONE; TAGP=""
ART=${FDT1_ART:-$AR/v8f-flatdag}; OUT=${FDT1_OUT:-$AR/v8f-flatdag-t1}; LOG=${FDT1_LOG:-$H/logs}
B2OUT=${FDT1_B2_OUT-C:/Users/Admin/box2_ship/logs/chain_v8f_flatdag.out}; [ "$B2OUT" = off ] && B2OUT=""
RUNS=(flatdag:0 flatdag0:0 flatdag:1 flatdag0:1)     # FOUR runs, ALWAYS ALL FOUR, in the PREREG's order; no run is conditional on another's read
RUNSCSV=$(IFS=,; echo "${RUNS[*]}")
m() { cygpath -m "$1"; }
TTu=$(cygpath -u "$TT"); TTm=$(m "$TT"); ARTu=$(cygpath -u "$ART"); ARTm=$(m "$ART"); OUTu=$(cygpath -u "$OUT"); OUTm=$(m "$OUT"); LOGu=$(cygpath -u "$LOG"); LOGm=$(m "$LOG")
SFu=$(cygpath -u "$SF"); SFm=$(m "$SF"); DAG_ARTm=$(m "$DAG_ART"); DAG_T1m=$(m "$DAG_T1"); DAGRRm=$(m "$DAGRR"); RELm=$(m "$REL"); READERm=$(m "$READER"); DAGRm=$(m "$DAGR"); D2Rm=$(m "$D2R"); RSRCm=$(m "$RSRC")
RCODEm=$(m "$RCODE"); LEADBLKm=$(m "$LEADBLK"); PREREGFm=$(m "$PREREGF"); FLATCKm=$(m "$FLATCK")
TOOL="$TTu/stack/scripts/v8f_t1_eval.py"
T1HD="$TTm/$PK/raw/2026-10-26-t1hdread/t1hd_read.py"; T1K="$TTm/$PK/raw/2026-10-26-t1kin-hacc/t1k_read.py"; PREREG="$TTm/Project Steering/PREREG_V8F_DAGGER.md"
GRID="$TTm/$PK/raw/eval_grid_windows.json"; JOIN="$TTm/$PK/raw/join_eval141.npz"; KIN="$TTm/$PK/raw/eval141_kin.npz"; TRACKS="$SFm/inputs/native_tracks.pt"; CACHEm="$SFm/cache"
IDS="$OUTm/ids"
MAXWAIT_S=$(( ${FDT1_MAX_WAIT_H:-40} * 3600 )); CMIN=${FDT1_C_MIN_MB:-2048}

if [ "$DRY" = 1 ]; then
  say() { echo "$(date '+%m-%d %H:%M:%S') ${TAGP}DRY: $*"; }
  stop() { say "$*"; echo "ZZFLATDAGT1-STOP $*"; exit 1; }
else
  mkdir -p "$LOGu" "$OUTu" "$OUTu/ticks" "$OUTu/argv"
  say() { echo "$(date '+%m-%d %H:%M:%S') ${TAGP}$*" | tee -a "$LOGu/chain.out"; }
  stop() { say "STOP: $*"; echo "ZZFLATDAGT1-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }
  rm -f "$LOGu/STOP"
  say "=== flatdag_t1_chain start (tree $TTm, reader $READERm, ART $ARTm, OUT $OUTm; runs $RUNSCSV)"
fi
nostop() { [ "$DRY" = 1 ] && return 0; [ -e "$LOGu/STOP" ] && exit 1; return 0; }
fmt() { local o="" a; for a in "$@"; do case "$a" in *[!A-Za-z0-9_./:=,@%+-]*) o="$o '$a'" ;; *) o="$o $a" ;; esac; done; echo "${o# }"; }
rundir() { echo "M2b_s${2}_${1}"; }                                   # rundir <arm> <seed>: the directory box 2's chain pulls into (M2b_s0_flatdag, M2b_s0_flatdag0, ...)
armU() { if [ "$1" = flatdag ]; then echo FLATDAG; else echo FLATDAG0; fi; }
tagof() { echo "$(armU "$1")_s${2}_m10_causal_heads"; }               # tagof <arm> <seed>: the rollout's --tag (T1-HEADS naming; never FLAT_s.. which is the banked FLAT rollout)
cfree_mb() {   # the TEST override exists only in --dry-run: a real run always reads the real C:
  if [ -n "${FDT1_TEST_CFREE_MB:-}" ] && [ "$DRY" = 1 ]; then echo "$FDT1_TEST_CFREE_MB"; else df -m /c | awk 'NR==2{print $4}'; fi
}
for r in "${RUNS[@]}"; do case "$r" in flatdag:[01]|flatdag0:[01]) ;; *) stop "bad run spec '$r' (arm:seed, arm in flatdag/flatdag0, seed 0/1)" ;; esac; done
[ "${FDT1_RUNS:-}" = "" ] || say "NOTE: FDT1_RUNS is ignored (the PREREG runs ALL FOUR, in its order)"
CF0=$(cfree_mb)
[ "$CF0" -ge "$CMIN" ] || stop "C: has $CF0 MB free (< $CMIN MB): the registered gate for this dev-box chain is C: >= 2 GB free (the pagefile is on C:; a full C: killed torch start-ups before; PREREG_V8F_DAGGER section 8, inherited)"
say "C: gate: $CF0 MB free >= $CMIN MB"
EMIN=${FDT1_E_MIN_MB:-1500}; EROLL=${FDT1_E_ROLL_MB:-200}
efree_mb() { if [ -n "${FDT1_TEST_EFREE_MB:-}" ] && [ "$DRY" = 1 ]; then echo "$FDT1_TEST_EFREE_MB"; else df -m /e | awk 'NR==2{print $4}'; fi; }
EF0=$(efree_mb)
[ "$EF0" -ge "$EMIN" ] || stop "E: has $EF0 MB free (< $EMIN MB): the four rollouts write ~0.6 GB to $OUTm; free E: first (PI-only deletes)"
say "E: gate: $EF0 MB free >= $EMIN MB"

# ------------------------------------------------------------------------------------------------ 0a  the tree: the DAgger lever must be in it, the tool flags must exist (grep + same-breath control; then the tool's own --help)
[ -f "$TOOL" ] || stop "no tool at $TTm/stack/scripts/v8f_t1_eval.py"
TRN="$TTu/stack/scripts/v8f_mini_train.py"; DGR="$TTu/stack/tanitad/data/v8f_dagger.py"
[ -f "$TRN" ] || stop "REFUSED: $TTm has no stack/scripts/v8f_mini_train.py (this must be the DAgger tip, not a tool-only tree)"
t_pos=$(grep -cE 'add_argument\("--init-ckpt"' "$TRN"); t_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$TRN")
[ "$t_pos" -ge 1 ] && [ "$t_neg" -eq 0 ] || stop "grep control broken on the trainer of $TTm (positive control --init-ckpt = $t_pos, must be >= 1; negative control = $t_neg, must be 0)"
for fl in --dagger-p --dagger-bank --dagger-bank-sha256 --dagger-seed --o-interface; do
  n=$(grep -cE "add_argument\(\"$fl\"" "$TRN")
  [ "$n" -ge 1 ] || stop "REFUSED: $TTm is not the FLAT-DAG tip: the trainer has no $fl (grep $n; the control read $t_pos, so the grep works)"
done
[ -f "$DGR" ] && [ "$(grep -cE '^class DaggerSwap' "$DGR")" -ge 1 ] || stop "REFUSED: $TTm has no stack/tanitad/data/v8f_dagger.py with class DaggerSwap: not the DAgger tip"
say "tree carries the DAgger + cut trainer: --dagger-p / --dagger-bank / --dagger-bank-sha256 / --dagger-seed / --o-interface and class DaggerSwap (control --init-ckpt = $t_pos, no-such-flag = $t_neg) in $TTm"
c_pos=$(grep -cE 'add_argument\("--replan-steps"' "$TOOL"); c_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$TOOL")
[ "$c_pos" -ge 1 ] && [ "$c_neg" -eq 0 ] || stop "grep control broken on $TTm (positive control --replan-steps = $c_pos, must be >= 1; negative control = $c_neg, must be 0)"
l_pos=$(grep -cE '^LIMIT_MODES *= *\(.*"logged"' "$TOOL"); l_neg=$(grep -cE '^LIMIT_MODES *= *\(.*"zzz-no-such-limit"' "$TOOL")
[ "$l_pos" -ge 1 ] && [ "$l_neg" -eq 0 ] || stop "grep control broken on LIMIT_MODES of $TTm (positive control logged = $l_pos, must be >= 1; negative control = $l_neg, must be 0)"
n_heads=$(grep -cE 'add_argument\("--dump-heads"' "$TOOL"); n_ticks=$(grep -cE 'add_argument\("--dump-ticks"' "$TOOL"); n_caus=$(grep -cE '^LIMIT_MODES *= *\(.*"causal"' "$TOOL")
n_cfg=$(grep -cE 'M\.V8FMiniConfig\(\*\*ck\["cfg"\]\)' "$TOOL")
say "guards: add_argument --dump-heads = $n_heads, --dump-ticks = $n_ticks, causal in LIMIT_MODES = $n_caus, the model is built from the checkpoint's own cfg (V8FMiniConfig(**ck[\"cfg\"])) = $n_cfg  (controls: --replan-steps = $c_pos, no-such-flag = $c_neg, logged = $l_pos, no-such-limit = $l_neg) in $TTm"
[ "$n_heads" -ge 1 ] || stop "REFUSED: $TTm has no --dump-heads (grep $n_heads; the control read $c_pos, so the grep works): a rollout without the head columns cannot be read"
[ "$n_ticks" -ge 1 ] || stop "REFUSED: $TTm has no --dump-ticks (grep $n_ticks; the control read $c_pos, so the grep works)"
[ "$n_caus" -ge 1 ] || stop "REFUSED: $TTm has no causal bucket (LIMIT_MODES grep $n_caus; the control read $l_pos, so the grep works): the decisive rollout is --limit causal"
[ "$n_cfg" -ge 1 ] || stop "REFUSED: the tool of $TTm does not build the model from the checkpoint's cfg (grep $n_cfg): an o_interface: cut checkpoint could be rolled out as a full one"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 TANITEVAL_STACK_OVERRIDE=$(cygpath -w "$TTu/stack")
export PYTHONPATH="$TTm/stack;$TTm/taniteval;$TTm/stack/scripts"
[ "$DRY" = 1 ] || { mkdir -p "$OUTu/_tmp"; export TMP="$OUTm/_tmp" TEMP="$OUTm/_tmp"; }        # keep the temp files off C:
tool_help() { "$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --help 2>&1 < /dev/null | tr -s '[:space:]' ' '; }
HELP=$(tool_help)
echo "$HELP" | grep -q 'usage:' || { say "the tool's --help printed no usage (a crashed import under memory pressure? began: ${HELP:0:200}); one retry in 30 s"; sleep 30; HELP=$(tool_help); }
echo "$HELP" | grep -q 'usage:' || stop "REFUSED: the tool's --help did not run (it printed no usage; began: ${HELP:0:300}): a crash is not 'the flag is missing', and the chain does not guess"
echo "$HELP" | grep -q -e '--dump-heads' || stop "REFUSED: the tool's own --help lists no --dump-heads (the --help output began: ${HELP:0:300})"
echo "$HELP" | grep -qE '\{logged,[^}]*causal' || stop "REFUSED: the tool's own --help lists no causal choice for --limit (the --help output began: ${HELP:0:300})"
say "tool --help agrees (--dump-heads, --limit {logged,...,causal,...})"

# ------------------------------------------------------------------------------------------------ 0b  the tool tree is the tree that rolled DAG (and FLAT) out
DAG_TT=${FDT1_DAG_TOOL_TREE:-}
[ -n "$DAG_TT" ] || DAG_TT=$(grep '^tree=' "$DAG_T1/PINS.txt" 2>/dev/null | head -1 | cut -c6-)
[ -n "$DAG_TT" ] && [ -d "$(cygpath -u "$DAG_TT")" ] || stop "DAG's tool tree is unknown (no tree= line in $DAG_T1/PINS.txt; set FDT1_DAG_TOOL_TREE), so tool identity cannot be checked"
if [ "$(m "$DAG_TT")" = "$TTm" ]; then
  PARITY_LINE="IDENTICAL: the tool tree IS the tree that rolled DAG out ($TTm), and FLAT's B-2 rollout used it too (dagreread/rollout_lane.sh)"
else
  PAR=$("$PY" -I "$Hm/flatdag_t1_manifest.py" parity --tree "$TTm" --ref "$(m "$DAG_TT")" 2>&1 < /dev/null); prc=$?
  PARITY_LINE=$(echo "$PAR" | head -1)
  [ $prc -eq 0 ] || stop "PARITY REFUSED: the T1 / T0 tools of $TTm are not the tools that rolled DAG out ($(m "$DAG_TT")): $(echo "$PAR" | tr '\n' ' ' | cut -c1-500)"
fi
say "tool identity with the tree that rolled DAG out: $PARITY_LINE"
if [ -d "$(cygpath -u "${FDT1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}")" ]; then      # ADVISORY: the tool tree is pinned by identity with the tree that rolled DAG and FLAT out, not by a tip
  TIPC=$("$PY" -I "$Hm/flatdag_t1_manifest.py" treetip --tree "$TTm" --git-dir "$(m "${FDT1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}")" --tip "${FDT1_TIP:-9b91b654}" 2>&1 < /dev/null) && say "$TIPC" || say "tree vs tip (ADVISORY, does not stop the chain): $(echo "$TIPC" | tr '\n' ' ' | cut -c1-400)"
else
  say "tree vs tip check SKIPPED: no tip mirror at ${FDT1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}"
fi

# ------------------------------------------------------------------------------------------------ 0c  inputs and pins
md5chk() { [ "$(md5sum < "$1" 2>/dev/null | cut -c1-32)" = "$2" ] || stop "md5 of $1 is not $2"; }
md5chk "$GRID" 34ed8ac56365eeff72e42581bde2f5c4
md5chk "$JOIN" 532fcd644c0e0835e847b2f1cee011ad
md5chk "$TRACKS" 189f4e68913c0163912a509203963f8d
[ "$(ls "$SFu/cache/eval/clips" 2>/dev/null | wc -l)" = 141 ] || stop "eval cache $CACHEm is not 141 clips (the T1-HEADS inputs are $AR/sfresh_inputs; FDT1_INPUTS names another copy; is the copy finished?)"
for f in "$READER" "$DAGR" "$D2R" "$T1HD" "$T1K" "$PREREG" "$PREREGF" "$PREREGR" "$PREREGD" "$REL" "$LEADBLK"; do [ -f "$f" ] || stop "input missing: $f"; done
[ "$(sha256sum < "$PREREGF" | cut -c1-64)" = "$PREREGF_SHA" ] || stop "PREREG_V8F_FLATDAG.md at $PREREGF is not the pinned text (sha256 ${PREREGF_SHA:0:12}...): the PREREG is binding, nothing may differ"
[ "$(sha256sum < "$PREREGR" | cut -c1-64)" = "$PREREGR_SHA" ] || stop "PREREG_V8F_DAG_REREAD.md at $PREREGR is not the pinned text (sha256 ${PREREGR_SHA:0:12}...)"
[ "$(sha256sum < "$PREREGD" | cut -c1-64)" = "$PREREGD_SHA" ] || stop "PREREG_V8F_DAGGER.md at $PREREGD is not the pinned text (sha256 ${PREREGD_SHA:0:12}...)"
SC=$("$PY" -I "$Hm/flatdag_read.py" --static-check --tool-root "$TTm" --dag-read "$DAGRm" --dag2-read "$D2Rm" --readers "$RSRCm" --reada-code "$RCODEm" --lead-block "$LEADBLKm" --prereg-flatdag "$PREREGFm" \
     --prereg-reread "$(m "$PREREGR")" --prereg-dagger "$(m "$PREREGD")" --dag-t1 "$DAG_T1m" --dagreread "$DAGRRm" 2>&1 < /dev/null) || stop "flatdag_read static check: $(echo "$SC" | tr '\n' ' ' | cut -c1-500)"
say "flatdag_read static check: $(echo "$SC" | tail -1) (pinned reader sources, Read A driver code, DAG's and FLAT's banked reads, the lead block, PREREG_V8F_FLATDAG section 2 and 4 cells verbatim)"
[ "$(sha256sum < "$PREREG" | cut -c1-64)" = "$PREREG_SHA" ] || stop "the tree's PREREG_V8F_DAGGER.md is not the landed text the reader was built for (sha256 ${PREREG_SHA:0:12}...): an amendment landed? re-pin deliberately (FDT1_PREREG_SHA) after re-reading the reader"
tree_digest() {   # one sha256 over every .py the tool can import (stack/tanitad, taniteval) + the tool
  ( cd "$TTu" && { find stack/tanitad taniteval -name '*.py' -type f; echo stack/scripts/v8f_t1_eval.py; } | LC_ALL=C sort -u | tr '\n' '\0' | xargs -0 sha256sum | sha256sum | cut -c1-64 )
}
keyfiles="stack/scripts/v8f_t1_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/floors_kin.py taniteval/taniteval/four_families.py taniteval/taniteval/ci.py taniteval/tools/t1_eval.py stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_dagger.py"
fsha() { sha256sum < "$(cygpath -u "$1")" | cut -c1-64; }
TOOLD=$(tree_digest)
say "inputs OK (grid, join, tracks md5; 141-clip eval cache; PREREG_V8F_FLATDAG / DAG_REREAD / DAGGER = the pinned texts); tool tree digest ${TOOLD:0:12}"
write_pins() {
  { echo "tree=$TTm"; echo "digest=$TOOLD"; for f in $keyfiles; do echo "file=$f $(sha256sum < "$TTu/$f" | cut -c1-64)"; done
    echo "reader=flatdag_read.py $(fsha "$READER")"; echo "reader=dag_read.py $(fsha "$DAGR")"; echo "reader=dag2_read.py $(fsha "$D2R")"; echo "prereg_flatdag=$(fsha "$PREREGF")"
    echo "reader=t1hd_read.py $(fsha "$T1HD")"; echo "reader=t1k_read.py $(fsha "$T1K")"; echo "reader=PREREG $(fsha "$PREREG")"
    echo "dag_tool_tree=${DAG_TT:-unknown}"; echo "parity=$PARITY_LINE"; echo "release=$(fsha "$REL")"; }
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
  local nm
  for nm in flatdag_read.py:"$READER" dag_read.py:"$DAGR" dag2_read.py:"$D2R" t1hd_read.py:"$T1HD" t1k_read.py:"$T1K" PREREG:"$PREREG"; do
    [ "$(grep "^reader=${nm%%:*} " "$OUTu/PINS.txt" | cut -d' ' -f2)" = "$(fsha "${nm#*:}")" ] || stop "reader input ${nm%%:*} changed ($1)"
  done
}
pin_check "start"

# ------------------------------------------------------------------------------------------------ 0d  the BANKED outputs (read, never re-run) and the two id lists (fail now, not after hours of rollouts)
FB=$("$PY" -I "$Hm/flatdag_t1_manifest.py" flatbase --dagreread "$DAGRRm" 2>&1 < /dev/null) || stop "FLAT's banked B-2 material: $(echo "$FB" | tr '\n' ' ' | cut -c1-500)"
say "$FB"
DB=$("$PY" -I "$Hm/flatdag_t1_manifest.py" dagbase --dag-art "$DAG_ARTm" --dag-t1 "$DAG_T1m" 2>&1 < /dev/null) || stop "DAG's banked outputs: $(echo "$DB" | tr '\n' ' ' | cut -c1-500)"
say "$DB"
if [ "$DRY" = 1 ]; then
  I0=$("$PY" -I "$Hm/flatdag_t1_manifest.py" ids --grid "$GRID" --release "$RELm" --out-dir "$IDS" --dry-run 2>&1 < /dev/null) || stop "id lists: $(echo "$I0" | tr '\n' ' ' | cut -c1-500)"
else
  I0=$("$PY" -I "$Hm/flatdag_t1_manifest.py" ids --grid "$GRID" --release "$RELm" --out-dir "$IDS" 2>&1 < /dev/null) || stop "id lists: $(echo "$I0" | tr '\n' ' ' | cut -c1-500)"
fi
say "$I0"
cmin_wait() {   # cmin_wait <what>: block (up to 2 h) until C: has CMIN MB again
  local w=0 what=$1
  while [ "$(cfree_mb)" -lt "$CMIN" ]; do nostop; [ $w -eq 0 ] && say "$what: waiting for C: >= $CMIN MB free (now $(cfree_mb) MB; the pagefile is on C:)"; w=$((w + 1)); [ $w -ge 120 ] && stop "C: under $CMIN MB free for 2 h"; sleep 60; done
}
emin_wait() {   # emin_wait <what>: block (up to 2 h) until E: has EROLL MB again; never deletes anything
  local w=0 what=$1
  while [ "$(efree_mb)" -lt "$EROLL" ]; do nostop; [ $w -eq 0 ] && say "$what: waiting for E: >= $EROLL MB free (now $(efree_mb) MB)"; w=$((w + 1)); [ $w -ge 120 ] && stop "E: under $EROLL MB free for 2 h"; sleep 60; done
}

# ------------------------------------------------------------------------------------------------ 1  the rollouts
t1cmd() {   # t1cmd <ckpt> <tag>  ->  T1CMD (the array IS the command that runs; --dry-run prints it).  FLAT's B-2 rollout argv (dagreread/rollout_lane.sh) except --ckpt, --tag, --out-dir, --dump-ticks.
  T1CMD=("$PY" "$TTm/stack/scripts/v8f_t1_eval.py" run --ckpt "$1" --tag "$2" --tracks "$TRACKS" --eval-cache "$CACHEm" --eval-grid "$GRID" --eval-join "$JOIN" --out-dir "$OUTm"
         --replan-steps 10 --nav deadreckon --scene logged --schedule periodic --limit causal --dump-ticks "$OUTm/ticks" --dump-heads --device cpu)
  [ "$MAXW" -gt 0 ] && T1CMD+=(--max-windows "$MAXW")
  return 0
}
b2count() { if [ -n "$B2OUT" ] && [ -f "$B2OUT" ]; then grep -c '^ZZV8FMINI-STOP' "$B2OUT"; else echo 0; fi; }
B2STOP0=$(b2count)
PULL_SHA=""
wait_pull() {   # wait_pull <arm> <seed>: block until the pull is COMPLETE (sets PULL_SHA); stops on STOP, on a WRONG run, on a box-2 STOP that appeared after we started, on timeout
  local arm=$1 s=$2 t0 n=0 out rc
  t0=$(date +%s)
  while :; do
    nostop
    out=$("$PY" "$Hm/flatdag_t1_pullcheck.py" run "$ARTm" "$arm" "$s" 2>&1 < /dev/null); rc=$?
    if [ $rc -eq 0 ]; then PULL_SHA=$(echo "$out" | cut -d' ' -f2); say "pull of $(rundir "$arm" "$s") is COMPLETE: ckpt sha256 ${PULL_SHA:0:16}... ($(echo "$out" | cut -d' ' -f3-))"; return 0; fi
    [ $rc -eq 3 ] && stop "the pulled $(rundir "$arm" "$s") is WRONG, waiting cannot fix it: $out"
    [ $((n % 30)) -eq 0 ] && say "waiting for the pull of $(rundir "$arm" "$s") ($(( ($(date +%s) - t0) / 60 )) min): $out"
    n=$((n + 1))
    [ "$(b2count)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP after this chain started ($B2OUT): no more pulls are coming ($(grep '^ZZV8FMINI-STOP' "$B2OUT" | tail -1 | cut -c1-200))"
    [ $(( $(date +%s) - t0 )) -gt "$MAXWAIT_S" ] && stop "timeout: the pull of $(rundir "$arm" "$s") is still not complete after $((MAXWAIT_S / 3600)) h ($out)"
    sleep "$POLL"
  done
}
n_dumps() { ls "$OUTu/dump_$1" 2>/dev/null | grep -c '^ep.*\.npz$'; }
rollout_done() {   # rollout_done <tag> <ckpt sha256>: finished, AND by THIS checkpoint (a re-pulled different checkpoint must not be skipped)
  [ -s "$OUTu/t1v8f_$1.json" ] && [ -s "$OUTu/ticks/ticks_$1.npz" ] && [ -s "$OUTu/done_$1.txt" ] && [ -s "$OUTu/argv/$1.argv" ] || return 1
  [ "$(n_dumps "$1")" -ge "$NDUMP" ] || return 1
  grep -q "\"ckpt_sha256\": \"$2\"" "$OUTu/done_$1.txt" && grep -q "\"tool_digest\": \"$TOOLD\"" "$OUTu/done_$1.txt"
}
t1run() {   # t1run <arm> <seed> <ckpt sha256>
  local arm=$1 s=$2 sha=$3 tag ck t0 rc att w f
  tag=$(tagof "$arm" "$s"); ck="$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt"
  if rollout_done "$tag" "$sha"; then say "skip $tag (complete, same checkpoint ${sha:0:12}, same tool)"; return 0; fi
  if [ -e "$OUTu/t1v8f_$tag.json" ] || [ -e "$OUTu/ticks/ticks_$tag.npz" ] || [ -d "$OUTu/dump_$tag" ] || [ -e "$OUTu/done_$tag.txt" ] || [ -e "$OUTu/argv/$tag.argv" ]; then   # an earlier attempt: MOVE, never delete
    mkdir -p "$OUTu/_incomplete_$(date +%m%d%H%M%S)"; w=$(ls -d "$OUTu"/_incomplete_* | tail -1)
    for f in "$OUTu/t1v8f_$tag.json" "$OUTu/ticks/ticks_$tag.npz" "$OUTu/dump_$tag" "$OUTu/done_$tag.txt" "$OUTu/argv/$tag.argv"; do [ -e "$f" ] && mv "$f" "$w/"; done
    say "moved the output of an earlier attempt of $tag to $w"
  fi
  cmin_wait "$tag"; emin_wait "$tag"
  nostop; pin_check "before $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$(rundir "$arm" "$s")/ckpt_last.pt changed since its pull was verified (before $tag)"
  t1cmd "$ck" "$tag"
  mkdir -p "$OUTu/argv"; printf '%s\n' "${T1CMD[@]}" > "$OUTu/argv/$tag.argv"          # the reader's C-ARGV compares this with FLAT's banked rollout argv
  for att in 1 2; do
    t0=$(date +%s)
    say "start $tag (attempt $att; ckpt $(rundir "$arm" "$s") ${sha:0:12}, limit causal): $(fmt "${T1CMD[@]}")"
    "${T1CMD[@]}" > "$LOGu/$tag.out" 2>&1 < /dev/null; rc=$?
    say "end $tag rc=$rc wall=$(( $(date +%s) - t0 ))s"
    [ $rc -eq 139 ] && [ $att -eq 1 ] && { say "$tag: native crash (rc 139, the T1-HEADS causal s0 first launch did the same): one retry; the crashed attempt's output is never read"; mv "$LOGu/$tag.out" "$LOGu/$tag.segv.out"; rm -f "$OUTu/done_$tag.txt"; continue; }
    break
  done
  pin_check "after $tag"
  [ "$(sha256sum < "$(cygpath -u "$ck")" | cut -c1-64)" = "$sha" ] || stop "$(rundir "$arm" "$s")/ckpt_last.pt changed during $tag"
  { [ $rc -eq 0 ] && [ -s "$OUTu/t1v8f_$tag.json" ] && [ -s "$OUTu/ticks/ticks_$tag.npz" ] && [ "$(n_dumps "$tag")" -ge "$NDUMP" ]; } || stop "rollout $tag failed (rc $rc; json $([ -s "$OUTu/t1v8f_$tag.json" ] && echo yes || echo NO), ticks $([ -s "$OUTu/ticks/ticks_$tag.npz" ] && echo yes || echo NO), dumps $(n_dumps "$tag")/$NDUMP): $(tail -2 "$LOGu/$tag.out" | tr '\n' ' ' | cut -c1-300)"
  printf '{"tag": "%s", "limit": "causal", "ckpt_sha256": "%s", "ckpt_sha256_before": "%s", "ckpt_sha256_after": "%s", "tool_digest": "%s", "n_dumps": %s, "wall_s": %s, "smoke": false}\n' "$tag" "$sha" "$sha" "$sha" "$TOOLD" "$(n_dumps "$tag")" "$(( $(date +%s) - t0 ))" > "$OUTu/done_$tag.txt"
}
lane() {   # lane <seed>: this seed's two rollouts, FLAT-DAG then FLAT-DAG-0, each as soon as ITS pull is complete
  local s=$1 arm
  for arm in flatdag flatdag0; do
    nostop; wait_pull "$arm" "$s"; t1run "$arm" "$s" "$PULL_SHA"
  done
  say "lane s$s done"
}

# ------------------------------------------------------------------------------------------------ 2  the box-2 DONE marker, the bank gate, the manifest and the reader
b2_done_wait() {   # the box-2 chain's own end marker: ^ZZV8FMINI-DONE in its log (read-only).  Off when FDT1_B2_OUT=off.
  local t0=$(date +%s) n=0
  [ -n "$B2OUT" ] || { say "box-2 DONE watch is OFF (FDT1_B2_OUT=off)"; return 0; }
  while :; do
    [ -f "$B2OUT" ] && [ "$(grep -c '^ZZV8FMINI-DONE' "$B2OUT")" -ge 1 ] && { say "box-2 chain DONE: $(grep '^ZZV8FMINI-DONE' "$B2OUT" | tail -1 | cut -c1-200)"; return 0; }
    nostop
    [ "$(b2count)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP ($B2OUT) before its DONE: $(grep '^ZZV8FMINI-STOP' "$B2OUT" | tail -1 | cut -c1-200)"
    [ $((n % 30)) -eq 0 ] && say "waiting for the box-2 chain's ZZV8FMINI-DONE in $B2OUT ($(( ($(date +%s) - t0) / 60 )) min)"
    n=$((n + 1))
    [ $(( $(date +%s) - t0 )) -gt 7200 ] && stop "no ZZV8FMINI-DONE in $B2OUT 2 h after all four pulls were complete"
    sleep "$POLL"
  done
}
verify_pulls() {   # every pull is STILL complete and still the checkpoint the rollout scored
  local spec arm s out rc sha rec
  for spec in "${RUNS[@]}"; do
    arm=${spec%%:*}; s=${spec##*:}
    out=$("$PY" "$Hm/flatdag_t1_pullcheck.py" run "$ARTm" "$arm" "$s" 2>&1 < /dev/null); rc=$?
    [ $rc -eq 0 ] || stop "after the rollouts the pull of $(rundir "$arm" "$s") is no longer COMPLETE: $out"
    sha=$(echo "$out" | cut -d' ' -f2); rec=$(grep -o '"ckpt_sha256": "[0-9a-f]*"' "$OUTu/done_$(tagof "$arm" "$s").txt" | head -1 | cut -d'"' -f4)
    [ "$sha" = "$rec" ] || stop "$(rundir "$arm" "$s")/ckpt_last.pt is $sha now, the rollout scored $rec"
  done
}
bank_gate() {   # the reader loads every bank shard npz (clip names, K-B3): wait until both bank directories are COMPLETE by content, STOP on a WRONG bank or a timeout
  local s t0=$(date +%s) n out rc
  for s in 0 1; do
    n=0
    while :; do
      out=$("$PY" "$Hm/flatdag_t1_pullcheck.py" bank "$ARTm" "$s" --expect-shards 4 2>&1 < /dev/null); rc=$?
      [ $rc -eq 3 ] && stop "the pulled bank s$s is WRONG, waiting cannot fix it: $out"
      [ $rc -eq 0 ] && { say "bank s$s is COMPLETE: bank.json sha256 $(echo "$out" | cut -d' ' -f2 | cut -c1-16)... ($(echo "$out" | cut -d' ' -f3-))"; break; }
      [ $((n % 30)) -eq 0 ] && say "waiting for the bank s$s directory ($(( ($(date +%s) - t0) / 60 )) min): $out"
      n=$((n + 1)); nostop
      [ $(( $(date +%s) - t0 )) -gt 43200 ] && stop "the bank s$s directory is still incomplete after 12 h, so the reader would REFUSE: $out"
      sleep "$POLL"
    done
  done
}
mk_reader_cmd() {   # mk_reader_cmd <init sha s0> <init sha s1>  ->  RDCMD   (the checkpoint FLAT-DAG / FLAT-DAG-0 started from = FLAT_s<S>, from the runs' own records)
  RDCMD=("$PY" -I "$READERm" --manifest "$OUTm/runs.json" --out "$OUTm/read/FLATDAG_READ.json" --tool-root "$TTm" --t1hd-read "$T1HD" --t1k-read "$T1K"
         --eval-cache "$CACHEm" --tracks "$TRACKS" --eval-grid "$GRID" --eval-join "$JOIN" --init-sha "0=$1" --init-sha "1=$2" --prereg "$PREREG" --prereg-flatdag "$PREREGFm"
         --prereg-reread "$(m "$PREREGR")" --prereg-dagger "$(m "$PREREGD")" --dag-read "$DAGRm" --dag2-read "$D2Rm" --readers "$RSRCm" --reada-code "$RCODEm" --lead-block "$LEADBLKm"
         --dag-t1 "$DAG_T1m" --dagreread "$DAGRRm" --flat-dir "$FLATCKm")
}
MANI=("$PY" -I "$Hm/flatdag_t1_manifest.py" manifest --art "$ARTm" --out "$OUTm" --ids-dir "$IDS" --runs "$RUNSCSV" --n-dump-min "$NDUMP")
M2BS=("$PY" -I "$Hm/flatdag_t1_manifest.py" initsha --art "$ARTm" --runs "$RUNSCSV" --flat-dir "$FLATCKm")

# ------------------------------------------------------------------------------------------------ --dry-run: print and leave
if [ "$DRY" = 1 ]; then
  say "C: free $(cfree_mb) MB (the chain needs >= $CMIN MB, at start and before every rollout); threads OMP/MKL $OMP_NUM_THREADS, CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES', max 2 rollout processes (one lane per seed), poll $POLL s, max wait $((MAXWAIT_S / 3600)) h per pull"
  say "box-2 watcher: $([ -n "$B2OUT" ] && echo "$B2OUT (ZZV8FMINI-STOP lines now: $B2STOP0; ZZV8FMINI-DONE lines now: $([ -f "$B2OUT" ] && { grep -c '^ZZV8FMINI-DONE' "$B2OUT" || true; } || echo 'log absent'))" || echo 'OFF (FDT1_B2_OUT=off)')"
  say "== the four runs in the PREREG's order, and the state of each pull right now"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; say "  $(rundir "$arm" "$s")  $("$PY" "$Hm/flatdag_t1_pullcheck.py" run "$ARTm" "$arm" "$s" 2>&1 < /dev/null | cut -c1-220)"; done
  for s in 0 1; do say "  bank_s$s  $("$PY" "$Hm/flatdag_t1_pullcheck.py" bank "$ARTm" "$s" --expect-shards 4 2>&1 < /dev/null | cut -c1-260)"; done
  say "== rollouts (lane s0 and lane s1; lane order = FLAT-DAG then FLAT-DAG-0; each starts when ITS pull is complete; ckpt = <ART>/<run>/ckpt_last.pt)"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}
    t1cmd "$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt" "$(tagof "$arm" "$s")"; say "  [lane s$s] $(fmt "${T1CMD[@]}")"
  done
  say "  each rollout writes $OUTm/t1v8f_<tag>.json, $OUTm/ticks/ticks_<tag>.npz, $OUTm/dump_<tag>/ep*.npz (>= $NDUMP), $OUTm/argv/<tag>.argv, then $OUTm/done_<tag>.txt (checkpoint sha256 + tool digest)"
  say "== NOT run here: a logged-bucket rollout and a fresh-age T0 eval (they feed only K4, which is not a PREREG_V8F_FLATDAG control); DAG's and FLAT's banked outputs are READ, not re-run"
  say "== after the lanes: box-2 ZZV8FMINI-DONE, every pull re-verified, bank gate (both banks COMPLETE, 11 files each, bank.json == the sha256 box 2's gate declared)"
  say "== manifest: $(fmt "${MANI[@]}" --write "$OUTm/runs.json")"
  "${MANI[@]}" --dry-run | sed 's/^/    /'      # the SAME argv the real run writes with
  say "== --init-sha: $(fmt "${M2BS[@]}")   (the runs' own summary.json / selfcheck.json levers.init_ckpt.sha256: FLAT-DAG and FLAT-DAG-0 equal per seed and equal to FLAT_s<S>'s checkpoint, 8496934a... / 6fdfc9b0..., and to the re-hashed v8f-mini-flat/M2b_s<S>_flat.pt)"
  mk_reader_cmd '<0: FLAT_s0 ckpt sha256>' '<1: FLAT_s1 ckpt sha256>'
  say "== reader: $(fmt "${RDCMD[@]}")"
  say "   -> $OUTm/read/FLATDAG_READ.json (rc 0 outcome written | 2 REFUSED, nothing written | 3 UNPLACED | 4 BLOCKED); reader stdout/stderr -> $LOGm/flatdag_read.out"
  say "== markers in $LOGm/chain.out: $DONE_MARK <n>/${#RUNS[@]} rollouts, reader rc <rc> ...  |  ZZFLATDAGT1-STOP <why>.  Nothing was written, launched or waited for."
  echo "ZZFLATDAGT1-DRY-OK"
  exit 0
fi

# ------------------------------------------------------------------------------------------------ the real run
say "== lanes: one per seed, --limit causal (decisive), ${#RUNS[@]} rollouts in all, OMP/MKL $OMP_NUM_THREADS threads each, max 2 processes"
lane 0 & PA=$!
lane 1 & PB=$!
wait $PA $PB
[ -e "$LOGu/STOP" ] && { say "a lane stopped: no DONE"; exit 1; }
pin_check "end"
n=0; for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; [ -s "$OUTu/done_$(tagof "$arm" "$s").txt" ] && n=$((n + 1)); done
NEED=${#RUNS[@]}
[ "$n" -eq "$NEED" ] || stop "$n of $NEED rollouts have a done record"
say "== all $n rollouts done; box-2 DONE marker"
b2_done_wait
verify_pulls
say "== bank gate"
bank_gate
say "== manifest"
"${MANI[@]}" --write "$OUTm/runs.json" > "$LOGu/manifest.out" 2>&1 < /dev/null || stop "manifest: $(tr '\n' ' ' < "$LOGu/manifest.out" | cut -c1-600)"
say "$(tail -1 "$LOGu/manifest.out")"
SHAS=$("${M2BS[@]}" 2> "$LOGu/m2bsha.err" < /dev/null) || stop "init sha: $(tr '\n' ' ' < "$LOGu/m2bsha.err" | cut -c1-300)"
S0=$(echo "$SHAS" | tr ' ' '\n' | grep '^0=' | cut -c3-); S1=$(echo "$SHAS" | tr ' ' '\n' | grep '^1=' | cut -c3-)
[ ${#S0} -eq 64 ] && [ ${#S1} -eq 64 ] || stop "init sha values malformed: '$SHAS'"
mk_reader_cmd "$S0" "$S1"
pin_check "before the reader"
say "== reader: $(fmt "${RDCMD[@]}")"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${RDCMD[@]}" > "$LOGu/flatdag_read.out" 2>&1 < /dev/null; RRC=$?
say "reader rc=$RRC: $(grep -a -E '"OUTCOME"|REFUSED|UNPLACED|BLOCKED|wrote ' "$LOGu/flatdag_read.out" | head -3 | tr '\n' ' ' | cut -c1-400)"
pin_check "after the reader"
echo "$DONE_MARK $n/$NEED rollouts, reader rc $RRC ($(case $RRC in 0) echo 'outcome row written' ;; 2) echo 'REFUSED, nothing written' ;; 3) echo UNPLACED ;; 4) echo 'BLOCKED, written' ;; *) echo 'reader crashed' ;; esac)): $OUTm/read/FLATDAG_READ.json  log $LOGm/flatdag_read.out" >> "$LOGu/chain.out"
tail -1 "$LOGu/chain.out"
