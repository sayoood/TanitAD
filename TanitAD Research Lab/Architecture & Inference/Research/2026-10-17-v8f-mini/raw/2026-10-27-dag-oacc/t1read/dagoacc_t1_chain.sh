#!/bin/bash
# PREREG_V8F_DAG_OACC (Project Steering/PREREG_V8F_DAG_OACC.md @ 95063711, raw-blob sha256 bd722b7a...): the DEV-BOX CPU part, as ONE detached chain (ZERO training, ZERO GPU, box 2 is never touched).  BUILT AND SELF-TESTED, NOT LAUNCHED.
# DERIVED from flatdag_t1_chain.sh (chain.diff is the change; that chain derived from dag2_t1_chain.sh, which ran PREREG_V8F_DAGGER2's eight rollouts end to end): sections 0a-0c are the same guards; the rollout and read sections are DAG-OACC's.
#   PREREG section 5, verbatim: "Dev-box CPU: the 2 T1-kin rollouts (DAG's are banked) and the reader."  So:
#   for each of the two runs:  DAG-OACC s0, DAG-OACC s1      (arm dagoacc; box 2 pulls them to <ART>/M2b_s<S>_dagoacc; ONE arm, its control is the BANKED DAG_s<S>, no new control is trained)
#     1. wait until the pull is COMPLETE by content (dagoacc_t1_pullcheck.py: SHA256SUMS.txt lists EXACTLY the 8 names, every file re-hashes, summary.json done, and the run's own records say it IS this arm and seed (C-O1):
#        init = M2b_s<S>, DAG's recipe and DAG's ITERATION-1 bank sha256 identical, --w-acc named once with the w_acc stage's value, the argv == DAG's banked argv except --out and --w-acc).  A DAG run in a dagoacc directory is a STOP.
#     2. ONE T1-kin rollout on eval-141, plain, CPU, the decisive settings (--limit causal) and the SAME tool tree that rolled DAG out (B-2 of PREREG_V8F_DAG_REREAD): the registered rollout argv except
#        --ckpt / --tag / --out-dir / --dump-ticks (the reader's control C-ARGV compares the argv this chain writes to <OUT>/argv/<tag>.argv with dagreread/argv/FLAT_s<S>_m10_causal_heads.argv, which equals DAG's in every setting flag).
#   NO logged-bucket rollout and NO fresh-age T0 eval: they feed only the T1-HEADS reducer's K4, which is not among PREREG_V8F_DAG_OACC's controls (C-O1..C-O4; Read A's C1-C4 need neither); the reader reports K4 as NOT RUN.
#   NO bank: DAG-OACC trains on DAG's BANKED iteration-1 bank (<DAG_ART>/bank_s<S>, read only); its sha256 identity is C-O1 (checked by the pull check, the manifest and the reader).
#   two lanes (one per seed), so AT MOST 2 rollout processes at once; CUDA_VISIBLE_DEVICES=-1.
#   DAG's banked outputs (v8f-dagger-t1/ dumps and ticks, v8f-dagger/ checkpoints and argv) are READ, never re-run.
#   after both rollouts: the box-2 chain's ZZV8FMINI-DONE (DOT1_B2_OUT, read-only), every pull re-verified, the w_acc record (box-2 stage waccdag, C-O3 input) and DAG's two banked banks verified, dagoacc_t1_manifest.py writes runs.json,
#   then dagoacc_read.py writes <OUT>/read/DAGOACC_READ.json.
# AMENDMENT 1 (A1-2026-10-27, written by the Master Mind before any DAG-OACC rollout or value) is IN FORCE: AMEND1.md is pinned by sha256 below (this chain STOPs on any difference, and so does the reader: --amend1), section 4's DOMINANT now needs
#   the crawl rows to pass, LONGITUDINAL FIXED is "not DOMINANT", and the 255 cells the PREREG left unplaced are placed.  Nothing else in this chain changed (rollouts, controls, bars, order).
#
#   bash dagoacc_t1_chain.sh --dry-run    prints every guard result, path, pull state, the EXACT command of every rollout and the reader command; touches nothing.
#   bash dagoacc_t1_chain.sh              the real chain.  RESUME = re-run the same command (a finished rollout is skipped by a done record naming THIS checkpoint's sha256 and the tool digest;
#                                         an incomplete earlier attempt is MOVED to <OUT>/_incomplete_*, never deleted).
# Markers (own log, <LOG>/chain.out):  ZZDAGOACCT1-DONE <n>/2 rollouts, reader rc <rc> ...  |  ZZDAGOACCT1-STOP <reason>   (STOP also touches <LOG>/STOP; the other lane stops at its next step).
#   Reader rc: 0 = an outcome row was written, 2 = REFUSED (a control failed; nothing written), 3 = UNPLACED (written; DEFENSIVE and UNREACHABLE after Amendment 1: section 4 as amended places every cell (enumerate_cells.py: 0 unplaced), so seeing it is a BUG in decide()).
#
# WHICH TREE:  DOT1_TREE = the tool tree DAG's AND FLAT's T1-kin rollouts used (default E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a); its model is built from the checkpoint's own cfg, `V8FMiniConfig(**ck["cfg"])`
#   (v8f_t1_eval.py:506), whose config class carries w_acc (v8f_mini.py:187), so a DAG-OACC checkpoint (cfg.w_acc = W) loads.  The chain asserts it carries the DAgger + w_acc trainer (lever greps) and is IDENTICAL to the tree that rolled DAG out.
# DISK: the two rollouts write ~150 MB each to E:.  The chain gates on C: >= 2048 MB (the registered gate) and E: >= 1500 MB at start (200 MB before each rollout); if short it WAITS (up to 2 h) and then STOPs; it never deletes anything.
# Environment (all optional):
#   DOT1_AMEND1 (AMEND1.md; default the E: copy under dagland/dagoacc_amend1/; sha256-pinned, a different file STOPs)
#   DOT1_TREE  DOT1_READER (dagoacc_read.py)  DOT1_DAG_READER (dag_read.py, pinned)  DOT1_DAG2_READER (dag2_read.py, pinned; dagoacc_read imports it)  DOT1_READERS (dagreread/reader_src)  DOT1_READA_CODE  DOT1_LEAD_BLOCK
#   DOT1_ART (box 2's pull dir, default E:/Projects/TanitAD-artifacts/v8f-dagoacc)  DOT1_OUT / DOT1_LOG  DOT1_DAG_ART / DOT1_DAG_T1 (DAG's banked pull / T1 outputs)  DOT1_DAGREREAD (the registered rollout argv)
#   DOT1_B2_OUT (box-2 chain log, default C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out: a ZZV8FMINI-STOP after this chain started stops it; the reader waits for its ^ZZV8FMINI-DONE; set DOT1_B2_OUT=off to skip)
#   DOT1_MAX_WAIT_H (40)  DOT1_C_MIN_MB (2048)  DOT1_E_MIN_MB (1500)  DOT1_TEST_CFREE_MB / DOT1_TEST_EFREE_MB (--dry-run only)
set -u
export PATH="/usr/bin:/mingw64/bin:$PATH"
DRY=0
case "${1:-}" in --dry-run) DRY=1 ;; "") ;; *) echo "usage: dagoacc_t1_chain.sh [--dry-run]   (configuration = DOT1_* environment variables, see the header)"; exit 2 ;; esac
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); Hm=$(cygpath -m "$H")
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
AR=/e/Projects/TanitAD-artifacts
TT=${DOT1_TREE:-E:/Projects/TanitAD-artifacts/dagt1_tree_b43c7a9a}
READER=${DOT1_READER:-$Hm/dagoacc_read.py}
DAGR=${DOT1_DAG_READER:-C:/Users/Admin/dagread/dag_read.py}                                      # the DAGGER reader dagoacc_read imports (sha256-pinned inside dag2_read.py's PINS)
D2R=${DOT1_DAG2_READER:-$AR/dag2t1/dag2_read.py}                                                 # dag2_read.py: Read A's driver class and helpers, imported (sha256-pinned inside dagoacc_read.py)
RSRC=${DOT1_READERS:-$AR/dagreread/reader_src}                                                   # Read A's reader sources (the files committed at bdf8d561)
RCODE=${DOT1_READA_CODE:-$Hm/readA_code}                                                         # Read A's driver code (rr_lib.py, readA.py, build_report.py; sha256-pinned copies of dagreread/code/)
LEADBLK=${DOT1_LEAD_BLOCK:-$Hm/inputs/b1_eval_lead_block_k30.npz}                                 # sha256-verified copy (7a879baa...) of the repo blob <v8f-mini>/raw/2026-10-22-dk/b1_eval_lead_block_k30.npz
SF=${DOT1_INPUTS:-$AR/sfresh_inputs}                                                             # on E: (the Master Mind's md5-verified copy of the T1-HEADS inputs)
DAG_ART=${DOT1_DAG_ART:-$AR/v8f-dagger}; DAG_T1=${DOT1_DAG_T1:-$AR/v8f-dagger-t1}               # DAG's banked pull / T1 outputs (READ only)
DAGRR=${DOT1_DAGREREAD:-$AR/dagreread}                                                           # the registered rollout argv (dagreread/argv/FLAT_s<S>_m10_causal_heads.argv): READ only
PREREGO=${DOT1_PREREGO:-$Hm/inputs/PREREG_V8F_DAG_OACC.md}; PREREGO_SHA=bd722b7a1a9ae419fdd6ebaf437f5657396664670ea4b4f1074420a80e7f169d      # PREREG_V8F_DAG_OACC.md @ 95063711 (raw blob)
PREREGR=$Hm/inputs/PREREG_V8F_DAG_REREAD.md; PREREGR_SHA=7acaa4dd25373d554d9d2cdc68d0c9b94687167e343cceec0e8cf0939d7657fa                       # Read A's rows (section 2) and C1-C4 (section 4), as at 9b91b654
AMEND1=${DOT1_AMEND1:-E:/Projects/TanitAD-artifacts/dagland/dagoacc_amend1/AMEND1.md}; AMEND1_SHA=44d7e879e5349e5791d1a6ad9ff18559cfd72ea63ce3ef634dcde16e9a33373e   # Amendment 1 (A1-2026-10-27) to PREREG_V8F_DAG_OACC.md; repo path <v8f-mini>/raw/2026-10-27-dagoacc-amend1/AMEND1.md
OFFLINE=$Hm/inputs/wacc_offline_DAG_s0.json; OFFLINE_SHA=f30aa20978eb5d54727eb5feeb804c13dfe9d6fb7f213ebfd813e1143010a901                        # C-O3's offline reference (w_acc 49.6969), pinned
MAXW=0; NDUMP=141; POLL=60; DONE_MARK=ZZDAGOACCT1-DONE; TAGP=""
ART=${DOT1_ART:-$AR/v8f-dagoacc}; OUT=${DOT1_OUT:-$AR/v8f-dagoacc-t1}; LOG=${DOT1_LOG:-$H/logs}
B2OUT=${DOT1_B2_OUT-C:/Users/Admin/box2_ship/logs/chain_v8f_dagoacc_r2.out}; [ "$B2OUT" = off ] && B2OUT=""
RUNS=(dagoacc:0 dagoacc:1)     # TWO runs, ALWAYS BOTH, in the PREREG's order (s0 then s1); no run is conditional on another's read
RUNSCSV=$(IFS=,; echo "${RUNS[*]}")
m() { cygpath -m "$1"; }
TTu=$(cygpath -u "$TT"); TTm=$(m "$TT"); ARTu=$(cygpath -u "$ART"); ARTm=$(m "$ART"); OUTu=$(cygpath -u "$OUT"); OUTm=$(m "$OUT"); LOGu=$(cygpath -u "$LOG"); LOGm=$(m "$LOG")
SFu=$(cygpath -u "$SF"); SFm=$(m "$SF"); DAG_ARTm=$(m "$DAG_ART"); DAG_T1m=$(m "$DAG_T1"); DAGRRm=$(m "$DAGRR"); READERm=$(m "$READER"); DAGRm=$(m "$DAGR"); D2Rm=$(m "$D2R"); RSRCm=$(m "$RSRC")
RCODEm=$(m "$RCODE"); LEADBLKm=$(m "$LEADBLK"); PREREGOm=$(m "$PREREGO"); OFFLINEm=$(m "$OFFLINE"); AMEND1m=$(m "$AMEND1")
TOOL="$TTu/stack/scripts/v8f_t1_eval.py"
T1HD="$TTm/$PK/raw/2026-10-26-t1hdread/t1hd_read.py"; T1K="$TTm/$PK/raw/2026-10-26-t1kin-hacc/t1k_read.py"
GRID="$TTm/$PK/raw/eval_grid_windows.json"; JOIN="$TTm/$PK/raw/join_eval141.npz"; KIN="$TTm/$PK/raw/eval141_kin.npz"; TRACKS="$SFm/inputs/native_tracks.pt"; CACHEm="$SFm/cache"
MAXWAIT_S=$(( ${DOT1_MAX_WAIT_H:-40} * 3600 )); CMIN=${DOT1_C_MIN_MB:-2048}

if [ "$DRY" = 1 ]; then
  say() { echo "$(date '+%m-%d %H:%M:%S') ${TAGP}DRY: $*"; }
  stop() { say "$*"; echo "ZZDAGOACCT1-STOP $*"; exit 1; }
else
  mkdir -p "$LOGu" "$OUTu" "$OUTu/ticks" "$OUTu/argv"
  say() { echo "$(date '+%m-%d %H:%M:%S') ${TAGP}$*" | tee -a "$LOGu/chain.out"; }
  stop() { say "STOP: $*"; echo "ZZDAGOACCT1-STOP $*" >> "$LOGu/chain.out"; touch "$LOGu/STOP"; exit 1; }
  rm -f "$LOGu/STOP"
  say "=== dagoacc_t1_chain start (tree $TTm, reader $READERm, ART $ARTm, OUT $OUTm; runs $RUNSCSV)"
fi
nostop() { [ "$DRY" = 1 ] && return 0; [ -e "$LOGu/STOP" ] && exit 1; return 0; }
fmt() { local o="" a; for a in "$@"; do case "$a" in *[!A-Za-z0-9_./:=,@%+-]*) o="$o '$a'" ;; *) o="$o $a" ;; esac; done; echo "${o# }"; }
rundir() { echo "M2b_s${2}_${1}"; }                                   # rundir <arm> <seed>: the directory box 2's chain pulls into (M2b_s0_dagoacc, ...)
tagof() { echo "DAGOACC_s${2}_m10_causal_heads"; }                    # tagof <arm> <seed>: the rollout's --tag (never DAG_s.. which is the banked DAG rollout)
cfree_mb() {   # the TEST override exists only in --dry-run: a real run always reads the real C:
  if [ -n "${DOT1_TEST_CFREE_MB:-}" ] && [ "$DRY" = 1 ]; then echo "$DOT1_TEST_CFREE_MB"; else df -m /c | awk 'NR==2{print $4}'; fi
}
for r in "${RUNS[@]}"; do case "$r" in dagoacc:[01]) ;; *) stop "bad run spec '$r' (arm:seed, arm dagoacc, seed 0/1)" ;; esac; done
[ "${DOT1_RUNS:-}" = "" ] || say "NOTE: DOT1_RUNS is ignored (the PREREG runs BOTH seeds, in its order)"
CF0=$(cfree_mb)
[ "$CF0" -ge "$CMIN" ] || stop "C: has $CF0 MB free (< $CMIN MB): the registered gate for this dev-box chain is C: >= 2 GB free (the pagefile is on C:; a full C: killed torch start-ups before; PREREG_V8F_DAGGER section 8, inherited)"
say "C: gate: $CF0 MB free >= $CMIN MB"
EMIN=${DOT1_E_MIN_MB:-1500}; EROLL=${DOT1_E_ROLL_MB:-200}
efree_mb() { if [ -n "${DOT1_TEST_EFREE_MB:-}" ] && [ "$DRY" = 1 ]; then echo "$DOT1_TEST_EFREE_MB"; else df -m /e | awk 'NR==2{print $4}'; fi; }
EF0=$(efree_mb)
[ "$EF0" -ge "$EMIN" ] || stop "E: has $EF0 MB free (< $EMIN MB): the two rollouts write ~0.3 GB to $OUTm; free E: first (PI-only deletes)"
say "E: gate: $EF0 MB free >= $EMIN MB"

# ------------------------------------------------------------------------------------------------ 0a  the tree: the DAgger + w_acc levers must be in it, the tool flags must exist (grep + same-breath control; then the tool's own --help)
[ -f "$TOOL" ] || stop "no tool at $TTm/stack/scripts/v8f_t1_eval.py"
TRN="$TTu/stack/scripts/v8f_mini_train.py"; DGR="$TTu/stack/tanitad/data/v8f_dagger.py"
[ -f "$TRN" ] || stop "REFUSED: $TTm has no stack/scripts/v8f_mini_train.py (this must be the DAgger tip, not a tool-only tree)"
t_pos=$(grep -cE 'add_argument\("--init-ckpt"' "$TRN"); t_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$TRN")
[ "$t_pos" -ge 1 ] && [ "$t_neg" -eq 0 ] || stop "grep control broken on the trainer of $TTm (positive control --init-ckpt = $t_pos, must be >= 1; negative control = $t_neg, must be 0)"
for fl in --dagger-p --dagger-bank --dagger-bank-sha256 --dagger-seed --w-acc; do
  n=$(grep -cE "add_argument\(\"$fl\"" "$TRN")
  [ "$n" -ge 1 ] || stop "REFUSED: $TTm is not the DAG-OACC tip: the trainer has no $fl (grep $n; the control read $t_pos, so the grep works)"
done
[ -f "$DGR" ] && [ "$(grep -cE '^class DaggerSwap' "$DGR")" -ge 1 ] || stop "REFUSED: $TTm has no stack/tanitad/data/v8f_dagger.py with class DaggerSwap: not the DAgger tip"
say "tree carries the DAgger + w_acc trainer: --dagger-p / --dagger-bank / --dagger-bank-sha256 / --dagger-seed / --w-acc and class DaggerSwap (control --init-ckpt = $t_pos, no-such-flag = $t_neg) in $TTm"
c_pos=$(grep -cE 'add_argument\("--replan-steps"' "$TOOL"); c_neg=$(grep -cE 'add_argument\("--zzz-no-such-flag-zzz"' "$TOOL")
[ "$c_pos" -ge 1 ] && [ "$c_neg" -eq 0 ] || stop "grep control broken on $TTm (positive control --replan-steps = $c_pos, must be >= 1; negative control = $c_neg, must be 0)"
l_pos=$(grep -cE '^LIMIT_MODES *= *\(.*"logged"' "$TOOL"); l_neg=$(grep -cE '^LIMIT_MODES *= *\(.*"zzz-no-such-limit"' "$TOOL")
[ "$l_pos" -ge 1 ] && [ "$l_neg" -eq 0 ] || stop "grep control broken on LIMIT_MODES of $TTm (positive control logged = $l_pos, must be >= 1; negative control = $l_neg, must be 0)"
n_heads=$(grep -cE 'add_argument\("--dump-heads"' "$TOOL"); n_ticks=$(grep -cE 'add_argument\("--dump-ticks"' "$TOOL"); n_caus=$(grep -cE '^LIMIT_MODES *= *\(.*"causal"' "$TOOL")
n_cfg=$(grep -cE 'M\.V8FMiniConfig\(\*\*ck\["cfg"\]\)' "$TOOL"); n_wacc=$(grep -cE '^ +w_acc: float' "$TTu/stack/tanitad/models/v8f_mini.py")
say "guards: add_argument --dump-heads = $n_heads, --dump-ticks = $n_ticks, causal in LIMIT_MODES = $n_caus, the model is built from the checkpoint's own cfg (V8FMiniConfig(**ck[\"cfg\"])) = $n_cfg, the config class declares w_acc = $n_wacc  (controls: --replan-steps = $c_pos, no-such-flag = $c_neg, logged = $l_pos, no-such-limit = $l_neg) in $TTm"
[ "$n_heads" -ge 1 ] || stop "REFUSED: $TTm has no --dump-heads (grep $n_heads; the control read $c_pos, so the grep works): a rollout without the head columns cannot be read"
[ "$n_ticks" -ge 1 ] || stop "REFUSED: $TTm has no --dump-ticks (grep $n_ticks; the control read $c_pos, so the grep works)"
[ "$n_caus" -ge 1 ] || stop "REFUSED: $TTm has no causal bucket (LIMIT_MODES grep $n_caus; the control read $l_pos, so the grep works): the decisive rollout is --limit causal"
[ "$n_cfg" -ge 1 ] || stop "REFUSED: the tool of $TTm does not build the model from the checkpoint's cfg (grep $n_cfg): a w_acc checkpoint could be rolled out with a default config"
[ "$n_wacc" -ge 1 ] || stop "REFUSED: the config class of $TTm declares no w_acc (grep $n_wacc): a DAG-OACC checkpoint's cfg (w_acc = W) would not load (V8FMiniConfig(**ck[\"cfg\"]) raises on an unknown key)"
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

# ------------------------------------------------------------------------------------------------ 0b  the tool tree is the tree that rolled DAG out
DAG_TT=${DOT1_DAG_TOOL_TREE:-}
[ -n "$DAG_TT" ] || DAG_TT=$(grep '^tree=' "$DAG_T1/PINS.txt" 2>/dev/null | head -1 | cut -c6-)
[ -n "$DAG_TT" ] && [ -d "$(cygpath -u "$DAG_TT")" ] || stop "DAG's tool tree is unknown (no tree= line in $DAG_T1/PINS.txt; set DOT1_DAG_TOOL_TREE), so tool identity cannot be checked"
if [ "$(m "$DAG_TT")" = "$TTm" ]; then
  PARITY_LINE="IDENTICAL: the tool tree IS the tree that rolled DAG out ($TTm), and FLAT's B-2 rollout used it too (dagreread/rollout_lane.sh)"
else
  PAR=$("$PY" -I "$Hm/dagoacc_t1_manifest.py" parity --tree "$TTm" --ref "$(m "$DAG_TT")" 2>&1 < /dev/null); prc=$?
  PARITY_LINE=$(echo "$PAR" | head -1)
  [ $prc -eq 0 ] || stop "PARITY REFUSED: the T1 / T0 tools of $TTm are not the tools that rolled DAG out ($(m "$DAG_TT")): $(echo "$PAR" | tr '\n' ' ' | cut -c1-500)"
fi
say "tool identity with the tree that rolled DAG out: $PARITY_LINE"
if [ -d "$(cygpath -u "${DOT1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}")" ]; then      # ADVISORY: the tool tree is pinned by identity with the tree that rolled DAG and FLAT out, not by a tip
  TIPC=$("$PY" -I "$Hm/dagoacc_t1_manifest.py" treetip --tree "$TTm" --git-dir "$(m "${DOT1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}")" --tip "${DOT1_TIP:-9b91b654}" 2>&1 < /dev/null) && say "$TIPC" || say "tree vs tip (ADVISORY, does not stop the chain): $(echo "$TIPC" | tr '\n' ' ' | cut -c1-400)"
else
  say "tree vs tip check SKIPPED: no tip mirror at ${DOT1_TIP_MIRROR:-C:/Users/Admin/tanitad-push}"
fi

# ------------------------------------------------------------------------------------------------ 0c  inputs and pins
md5chk() { [ "$(md5sum < "$1" 2>/dev/null | cut -c1-32)" = "$2" ] || stop "md5 of $1 is not $2"; }
md5chk "$GRID" 34ed8ac56365eeff72e42581bde2f5c4
md5chk "$JOIN" 532fcd644c0e0835e847b2f1cee011ad
md5chk "$TRACKS" 189f4e68913c0163912a509203963f8d
[ "$(ls "$SFu/cache/eval/clips" 2>/dev/null | wc -l)" = 141 ] || stop "eval cache $CACHEm is not 141 clips (the T1-HEADS inputs are $AR/sfresh_inputs; DOT1_INPUTS names another copy; is the copy finished?)"
for f in "$READER" "$DAGR" "$D2R" "$T1HD" "$T1K" "$PREREGO" "$PREREGR" "$AMEND1" "$OFFLINE" "$LEADBLK"; do [ -f "$f" ] || stop "input missing: $f"; done
[ "$(sha256sum < "$PREREGO" | cut -c1-64)" = "$PREREGO_SHA" ] || stop "PREREG_V8F_DAG_OACC.md at $PREREGO is not the pinned text (sha256 ${PREREGO_SHA:0:12}...): the PREREG is binding, nothing may differ"
[ "$(sha256sum < "$PREREGR" | cut -c1-64)" = "$PREREGR_SHA" ] || stop "PREREG_V8F_DAG_REREAD.md at $PREREGR is not the pinned text (sha256 ${PREREGR_SHA:0:12}...)"
[ "$(sha256sum < "$AMEND1" | cut -c1-64)" = "$AMEND1_SHA" ] || stop "AMEND1.md at $AMEND1 is not the pinned text (sha256 ${AMEND1_SHA:0:12}...): Amendment 1 is binding, nothing may differ"
[ "$(sha256sum < "$OFFLINE" | cut -c1-64)" = "$OFFLINE_SHA" ] || stop "C-O3's offline reference $OFFLINE is not the pinned wacc_offline_DAG_s0.json (sha256 ${OFFLINE_SHA:0:12}...)"
SC=$("$PY" -I "$Hm/dagoacc_read.py" --static-check --tool-root "$TTm" --dag-read "$DAGRm" --dag2-read "$D2Rm" --readers "$RSRCm" --reada-code "$RCODEm" --lead-block "$LEADBLKm" --prereg-oacc "$PREREGOm" \
     --prereg-reread "$(m "$PREREGR")" --amend1 "$AMEND1m" --offline-wacc "$OFFLINEm" --dag-t1 "$DAG_T1m" --dag-art "$DAG_ARTm" --dagreread "$DAGRRm" 2>&1 < /dev/null) || stop "dagoacc_read static check: $(echo "$SC" | tr '\n' ' ' | cut -c1-500)"
say "dagoacc_read static check: $(echo "$SC" | tail -1) (pinned reader sources, Read A driver code, DAG's banked reads and argv, the lead block, PREREG_V8F_DAG_OACC section 2 and 4 cells verbatim, AMEND1.md pinned and its rewritten rows verbatim)"
tree_digest() {   # one sha256 over every .py the tool can import (stack/tanitad, taniteval) + the tool
  ( cd "$TTu" && { find stack/tanitad taniteval -name '*.py' -type f; echo stack/scripts/v8f_t1_eval.py; } | LC_ALL=C sort -u | tr '\n' '\0' | xargs -0 sha256sum | sha256sum | cut -c1-64 )
}
keyfiles="stack/scripts/v8f_t1_eval.py stack/tanitad/models/v8f_mini.py stack/tanitad/data/v8f_mini_data.py taniteval/taniteval/floors_kin.py taniteval/taniteval/four_families.py taniteval/taniteval/ci.py taniteval/tools/t1_eval.py stack/scripts/v8f_mini_train.py stack/tanitad/data/v8f_dagger.py"
fsha() { sha256sum < "$(cygpath -u "$1")" | cut -c1-64; }
TOOLD=$(tree_digest)
say "inputs OK (grid, join, tracks md5; 141-clip eval cache; PREREG_V8F_DAG_OACC / AMEND1 / DAG_REREAD / C-O3's offline reference = the pinned texts); tool tree digest ${TOOLD:0:12}"
write_pins() {
  { echo "tree=$TTm"; echo "digest=$TOOLD"; for f in $keyfiles; do echo "file=$f $(sha256sum < "$TTu/$f" | cut -c1-64)"; done
    echo "reader=dagoacc_read.py $(fsha "$READER")"; echo "reader=dag_read.py $(fsha "$DAGR")"; echo "reader=dag2_read.py $(fsha "$D2R")"; echo "prereg_oacc=$(fsha "$PREREGO")"; echo "amend1=$(fsha "$AMEND1")"
    echo "reader=t1hd_read.py $(fsha "$T1HD")"; echo "reader=t1k_read.py $(fsha "$T1K")"
    echo "dag_tool_tree=${DAG_TT:-unknown}"; echo "parity=$PARITY_LINE"; }
}
if [ "$DRY" = 1 ]; then
  say "pins that would be written to $OUTm/PINS.txt:"; write_pins | sed 's/^/    /'
else
  if [ ! -e "$OUTu/PINS.txt" ]; then write_pins > "$OUTu/PINS.txt"; say "pins written (tool digest ${TOOLD:0:12}; reader scripts, PREREG sha256)"; fi
  [ "$(grep ^tree= "$OUTu/PINS.txt" | cut -c6-)" = "$TTm" ] || stop "PINS.txt belongs to another TOOL TREE ($(grep ^tree= "$OUTu/PINS.txt")); delete it deliberately to re-pin"
fi
pin_check() {   # pin_check <when>
  [ "$DRY" = 1 ] && return 0
  [ "$(grep ^digest= "$OUTu/PINS.txt" | cut -c8-)" = "$(tree_digest)" ] || stop "the tool tree changed ($1)"
  local nm
  for nm in dagoacc_read.py:"$READER" dag_read.py:"$DAGR" dag2_read.py:"$D2R" t1hd_read.py:"$T1HD" t1k_read.py:"$T1K"; do
    [ "$(grep "^reader=${nm%%:*} " "$OUTu/PINS.txt" | cut -d' ' -f2)" = "$(fsha "${nm#*:}")" ] || stop "reader input ${nm%%:*} changed ($1)"
  done
}
pin_check "start"

# ------------------------------------------------------------------------------------------------ 0d  the BANKED outputs (read, never re-run): DAG's checkpoints / dumps / read, DAG's iteration-1 banks (C-O1's bank identity), the registered rollout argv
RA=$(for s in 0 1; do [ -s "$DAGRR/argv/FLAT_s${s}_m10_causal_heads.argv" ] && echo ok || echo missing; done | sort -u | tr '\n' ' ')
[ "$RA" = "ok " ] || stop "the registered rollout argv $DAGRRm/argv/FLAT_s<S>_m10_causal_heads.argv is missing ($RA): C-ARGV has nothing to compare with"
say "the registered T1-kin rollout argv (dagreread/argv/FLAT_s0 / s1, equal to DAG's in every setting flag) present: read, not re-run"
DB=$("$PY" -I "$Hm/dagoacc_t1_manifest.py" dagbase --dag-art "$DAG_ARTm" --dag-t1 "$DAG_T1m" 2>&1 < /dev/null) || stop "DAG's banked outputs: $(echo "$DB" | tr '\n' ' ' | cut -c1-500)"
say "$DB"
for s in 0 1; do
  BK=$("$PY" -I "$Hm/dagoacc_t1_pullcheck.py" dagbank "$DAG_ARTm" "$s" 2>&1 < /dev/null); brc=$?
  [ $brc -eq 0 ] || stop "DAG's banked iteration-1 bank s$s: $BK"
  say "$BK"
done
cmin_wait() {   # cmin_wait <what>: block (up to 2 h) until C: has CMIN MB again
  local w=0 what=$1
  while [ "$(cfree_mb)" -lt "$CMIN" ]; do nostop; [ $w -eq 0 ] && say "$what: waiting for C: >= $CMIN MB free (now $(cfree_mb) MB; the pagefile is on C:)"; w=$((w + 1)); [ $w -ge 120 ] && stop "C: under $CMIN MB free for 2 h"; sleep 60; done
}
emin_wait() {   # emin_wait <what>: block (up to 2 h) until E: has EROLL MB again; never deletes anything
  local w=0 what=$1
  while [ "$(efree_mb)" -lt "$EROLL" ]; do nostop; [ $w -eq 0 ] && say "$what: waiting for E: >= $EROLL MB free (now $(efree_mb) MB)"; w=$((w + 1)); [ $w -ge 120 ] && stop "E: under $EROLL MB free for 2 h"; sleep 60; done
}

# ------------------------------------------------------------------------------------------------ 1  the rollouts
t1cmd() {   # t1cmd <ckpt> <tag>  ->  T1CMD (the array IS the command that runs; --dry-run prints it).  The registered rollout argv (dagreread/rollout_lane.sh) except --ckpt, --tag, --out-dir, --dump-ticks.
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
    out=$("$PY" "$Hm/dagoacc_t1_pullcheck.py" run "$ARTm" "$arm" "$s" --dag-art "$DAG_ARTm" 2>&1 < /dev/null); rc=$?
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
  mkdir -p "$OUTu/argv"; printf '%s\n' "${T1CMD[@]}" > "$OUTu/argv/$tag.argv"          # the reader's C-ARGV compares this with the registered rollout argv
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
lane() {   # lane <seed>: this seed's ONE rollout, as soon as ITS pull is complete
  local s=$1 arm
  for arm in dagoacc; do
    nostop; wait_pull "$arm" "$s"; t1run "$arm" "$s" "$PULL_SHA"
  done
  say "lane s$s done"
}

# ------------------------------------------------------------------------------------------------ 2  the box-2 DONE marker, the w_acc record, the manifest and the reader
b2_done_wait() {   # the box-2 chain's own end marker: ^ZZV8FMINI-DONE in its log (read-only).  Off when DOT1_B2_OUT=off.
  local t0=$(date +%s) n=0
  [ -n "$B2OUT" ] || { say "box-2 DONE watch is OFF (DOT1_B2_OUT=off)"; return 0; }
  while :; do
    [ -f "$B2OUT" ] && [ "$(grep -c '^ZZV8FMINI-DONE' "$B2OUT")" -ge 1 ] && { say "box-2 chain DONE: $(grep '^ZZV8FMINI-DONE' "$B2OUT" | tail -1 | cut -c1-200)"; return 0; }
    nostop
    [ "$(b2count)" -gt "$B2STOP0" ] && stop "the box-2 chain wrote a ZZV8FMINI-STOP ($B2OUT) before its DONE: $(grep '^ZZV8FMINI-STOP' "$B2OUT" | tail -1 | cut -c1-200)"
    [ $((n % 30)) -eq 0 ] && say "waiting for the box-2 chain's ZZV8FMINI-DONE in $B2OUT ($(( ($(date +%s) - t0) / 60 )) min)"
    n=$((n + 1))
    [ $(( $(date +%s) - t0 )) -gt 7200 ] && stop "no ZZV8FMINI-DONE in $B2OUT 2 h after both pulls were complete"
    sleep "$POLL"
  done
}
verify_pulls() {   # every pull is STILL complete and still the checkpoint the rollout scored
  local spec arm s out rc sha rec
  for spec in "${RUNS[@]}"; do
    arm=${spec%%:*}; s=${spec##*:}
    out=$("$PY" "$Hm/dagoacc_t1_pullcheck.py" run "$ARTm" "$arm" "$s" --dag-art "$DAG_ARTm" 2>&1 < /dev/null); rc=$?
    [ $rc -eq 0 ] || stop "after the rollouts the pull of $(rundir "$arm" "$s") is no longer COMPLETE: $out"
    sha=$(echo "$out" | cut -d' ' -f2); rec=$(grep -o '"ckpt_sha256": "[0-9a-f]*"' "$OUTu/done_$(tagof "$arm" "$s").txt" | head -1 | cut -d'"' -f4)
    [ "$sha" = "$rec" ] || stop "$(rundir "$arm" "$s")/ckpt_last.pt is $sha now, the rollout scored $rec"
  done
}
wacc_gate() {   # C-O3's input: the box-2 w_acc stage's record is pulled and self-consistent (the reader then compares it with the offline reference), and DAG's banked banks are still the declared ones
  local out rc s
  out=$("$PY" "$Hm/dagoacc_t1_pullcheck.py" wacc "$ARTm" 2>&1 < /dev/null); rc=$?
  [ $rc -eq 0 ] || stop "the w_acc record (C-O3 input) is not usable: $out"
  say "w_acc record: $out (the reader compares it with the offline reference $OFFLINEm, 1e-4 relative)"
  for s in 0 1; do
    out=$("$PY" "$Hm/dagoacc_t1_pullcheck.py" dagbank "$DAG_ARTm" "$s" 2>&1 < /dev/null); rc=$?
    [ $rc -eq 0 ] || stop "DAG's iteration-1 bank s$s is no longer COMPLETE: $out"
  done
}
mk_reader_cmd() {   # mk_reader_cmd <init sha s0> <init sha s1>  ->  RDCMD   (the checkpoint DAG-OACC started from = M2b_s<S>, from the runs' own records)
  RDCMD=("$PY" -I "$READERm" --manifest "$OUTm/runs.json" --out "$OUTm/read/DAGOACC_READ.json" --tool-root "$TTm" --t1hd-read "$T1HD" --t1k-read "$T1K"
         --eval-cache "$CACHEm" --tracks "$TRACKS" --eval-grid "$GRID" --eval-join "$JOIN" --init-sha "0=$1" --init-sha "1=$2" --prereg-oacc "$PREREGOm"
         --prereg-reread "$(m "$PREREGR")" --amend1 "$AMEND1m" --offline-wacc "$OFFLINEm" --dag-read "$DAGRm" --dag2-read "$D2Rm" --readers "$RSRCm" --reada-code "$RCODEm" --lead-block "$LEADBLKm"
         --dag-t1 "$DAG_T1m" --dag-art "$DAG_ARTm" --dagreread "$DAGRRm")
}
MANI=("$PY" -I "$Hm/dagoacc_t1_manifest.py" manifest --art "$ARTm" --out "$OUTm" --dag-art "$DAG_ARTm" --dag-t1 "$DAG_T1m" --inputs "$Hm/inputs" --runs "$RUNSCSV" --n-dump-min "$NDUMP")
M2BS=("$PY" -I "$Hm/dagoacc_t1_manifest.py" initsha --art "$ARTm" --runs "$RUNSCSV")

# ------------------------------------------------------------------------------------------------ --dry-run: print and leave
if [ "$DRY" = 1 ]; then
  say "C: free $(cfree_mb) MB (the chain needs >= $CMIN MB, at start and before every rollout); threads OMP/MKL $OMP_NUM_THREADS, CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES', max 2 rollout processes (one lane per seed), poll $POLL s, max wait $((MAXWAIT_S / 3600)) h per pull"
  say "box-2 watcher: $([ -n "$B2OUT" ] && echo "$B2OUT (ZZV8FMINI-STOP lines now: $B2STOP0; ZZV8FMINI-DONE lines now: $([ -f "$B2OUT" ] && { grep -c '^ZZV8FMINI-DONE' "$B2OUT" || true; } || echo 'log absent'))" || echo 'OFF (DOT1_B2_OUT=off)')"
  say "== the two runs in the PREREG's order, and the state of each pull right now"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}; say "  $(rundir "$arm" "$s")  $("$PY" "$Hm/dagoacc_t1_pullcheck.py" run "$ARTm" "$arm" "$s" --dag-art "$DAG_ARTm" 2>&1 < /dev/null | cut -c1-220)"; done
  say "  w_acc record  $("$PY" "$Hm/dagoacc_t1_pullcheck.py" wacc "$ARTm" 2>&1 < /dev/null | cut -c1-200)"
  say "== rollouts (lane s0 and lane s1; each starts when ITS pull is complete; ckpt = <ART>/<run>/ckpt_last.pt)"
  for spec in "${RUNS[@]}"; do arm=${spec%%:*}; s=${spec##*:}
    t1cmd "$ARTm/$(rundir "$arm" "$s")/ckpt_last.pt" "$(tagof "$arm" "$s")"; say "  [lane s$s] $(fmt "${T1CMD[@]}")"
  done
  say "  each rollout writes $OUTm/t1v8f_<tag>.json, $OUTm/ticks/ticks_<tag>.npz, $OUTm/dump_<tag>/ep*.npz (>= $NDUMP), $OUTm/argv/<tag>.argv, then $OUTm/done_<tag>.txt (checkpoint sha256 + tool digest)"
  say "== NOT run here: a logged-bucket rollout and a fresh-age T0 eval (they feed only K4, which is not a PREREG_V8F_DAG_OACC control); DAG's banked outputs are READ, not re-run; there is no bank to wait for (DAG's iteration-1 bank is banked)"
  say "== after the lanes: box-2 ZZV8FMINI-DONE, every pull re-verified, w_acc record + DAG's two banked banks re-verified"
  say "== manifest: $(fmt "${MANI[@]}" --write "$OUTm/runs.json")"
  "${MANI[@]}" --dry-run | sed 's/^/    /'      # the SAME argv the real run writes with
  say "== --init-sha: $(fmt "${M2BS[@]}")   (the runs' own summary.json / selfcheck.json levers.init_ckpt.sha256: equal per seed and equal to M2b_s<S>'s checkpoint, 601cb435... / 6104fb15...)"
  mk_reader_cmd '<0: M2b_s0 ckpt sha256>' '<1: M2b_s1 ckpt sha256>'
  say "== reader: $(fmt "${RDCMD[@]}")"
  say "   -> $OUTm/read/DAGOACC_READ.json (rc 0 outcome written | 2 REFUSED, nothing written | 3 UNPLACED: defensive, unreachable after Amendment 1, a BUG if seen); reader stdout/stderr -> $LOGm/dagoacc_read.out"
  say "== markers in $LOGm/chain.out: $DONE_MARK <n>/${#RUNS[@]} rollouts, reader rc <rc> ...  |  ZZDAGOACCT1-STOP <why>.  Nothing was written, launched or waited for."
  echo "ZZDAGOACCT1-DRY-OK"
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
say "== w_acc record and DAG's banked banks"
wacc_gate
say "== manifest"
"${MANI[@]}" --write "$OUTm/runs.json" > "$LOGu/manifest.out" 2>&1 < /dev/null || stop "manifest: $(tr '\n' ' ' < "$LOGu/manifest.out" | cut -c1-600)"
say "$(tail -1 "$LOGu/manifest.out")"
SHAS=$("${M2BS[@]}" 2> "$LOGu/m2bsha.err" < /dev/null) || stop "init sha: $(tr '\n' ' ' < "$LOGu/m2bsha.err" | cut -c1-300)"
S0=$(echo "$SHAS" | tr ' ' '\n' | grep '^0=' | cut -c3-); S1=$(echo "$SHAS" | tr ' ' '\n' | grep '^1=' | cut -c3-)
[ ${#S0} -eq 64 ] && [ ${#S1} -eq 64 ] || stop "init sha values malformed: '$SHAS'"
mk_reader_cmd "$S0" "$S1"
pin_check "before the reader"
say "== reader: $(fmt "${RDCMD[@]}")"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${RDCMD[@]}" > "$LOGu/dagoacc_read.out" 2>&1 < /dev/null; RRC=$?
say "reader rc=$RRC: $(grep -a -E '"OUTCOME"|REFUSED|UNPLACED|wrote ' "$LOGu/dagoacc_read.out" | head -3 | tr '\n' ' ' | cut -c1-400)"
pin_check "after the reader"
echo "$DONE_MARK $n/$NEED rollouts, reader rc $RRC ($(case $RRC in 0) echo 'outcome row written' ;; 2) echo 'REFUSED, nothing written' ;; 3) echo 'UNPLACED, written (BUG: unreachable after Amendment 1)' ;; *) echo 'reader crashed' ;; esac)): $OUTm/read/DAGOACC_READ.json  log $LOGm/dagoacc_read.out" >> "$LOGu/chain.out"
tail -1 "$LOGu/chain.out"
