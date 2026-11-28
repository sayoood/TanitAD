#!/bin/bash
# d1_restart.sh [DRY|REAL] <tip40> <D1REC (12 or 40 hex)> -- MM 4a3d05, 2026-11-23 (derived from restart_e3.sh's kit-at-tip pattern). THE EARLY-D1 RESTART THAT KEEPS E3'S TREE.
# At E3's pause (PAUSED.json, step 30,000) D1 (arm K2DC02_d1) is gated and trained from ckpt_full_30000.pt:  R9_STUDY=e3 bash r9gate3_restart.sh <tip> "K2DC02_d1" <REC12>.
# r9gate3_restart.sh:21 refuses any SHA that is not the remote tip, and r9w_ship.sh:88 moves the old $R/launch_tree to launch_tree_aside_<HHMMSS> (NO commit in the name) and overwrites
# $R/launch_tree.commit (r9w_ship.sh:99). So if the tip moved since E3 launched, E3's token (manifest COMMIT, a tree digest) would no longer match the tree at $R/launch_tree.
# THIS SCRIPT first PRESERVES E3's tree by RENAME under a commit-named path ($R/launch_tree_e3_<c12>, + .commit + .fp = a cheap fingerprint recorded NOW), then runs the D1 restart.
# resume_e3.sh restores it (D1's tree moves aside, never deleted). Nothing is ever deleted here.
# Guards (DRY lists every failing one and changes nothing; it only READS Thor):
#   NOPAUSED (PAUSED.json present, E3 not done), NOCKPT (ckpt_full_30000.pt present), NOMANIFEST, CHAINLIVE (no gate3 chain with cwd $R), SUPLIVE (no sup_refcv7 on $R/launch_tree), GPULOCK (REAL only),
#   NOTIP (the remote tip is not <tip40>), TREEMISMATCH (E3's tree is not at $R/launch_tree and not yet preserved), PRESEXISTS, NOAPPROVAL / NOARM (D1's cap + argv at the tip),
#   CLOSUREMOVED (a record input changed D1REC..tip: D1's records no longer fit), NOD1RECORDS (queue 28 did not bind D1 at D1REC: names the D1-only rebind).
# D1's records: queue 28 bound them at REC a49c964032cd (K2DC02_d1-RECORDS-READY line in $R/logs/recq28_<c12>.status). r9gate3_restart.sh:29 stamps $R/gate3/REC_C12 with whatever REC12 it is
# given (no check against the old stamp), so a NEW REC for D1 is allowed once its records exist: R9Q28_D1_ONLY=1 (r9_recqueue28.sh:59,73) binds only D1 at that REC from a shipped rectree_<c12>.
# Ends ZZD1RESTART-OKZZ / ZZD1RESTART-DRYENDZZ / ZZD1RESTART-<GUARD>ZZ.
export PATH="/usr/bin:/bin:$PATH"
trap '' INT
MODE=${1:?usage: d1_restart.sh DRY|REAL <tip40> <D1REC>}; case "$MODE" in DRY|REAL) ;; *) echo "usage: d1_restart.sh DRY|REAL <tip40> <D1REC>"; exit 2;; esac
TIP=${2:?tip40}; [ ${#TIP} = 40 ] || { echo "tip must be 40 chars"; exit 2; }
DREC=${3:?D1REC}; case "${#DREC}" in 12|40) ;; *) echo "D1REC must be 12 or 40 hex chars"; exit 2;; esac
S=${R9K_LOG:-C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad}; OUT=$S/D1_RESTART.txt
GD=C:/Users/Admin/tanitad-push/.git; BR=agent/arch-inf-20260803
V=K2DC02; D1=${V}_d1; R=/home/nvidia/r8ns/r9e3; OUTD=$R/runs/r9e3-K2DC02; RD=stack/ops/runs.d; LOCKF=/home/nvidia/refcv7_post/thor_gpu.lock; CAP=10.5
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT"; }
th() { timeout 90 ssh -n -o ConnectTimeout=20 tanitad-thor "$1" 2>/dev/null | tr -d '\r'; }
FAILS=0
bad() { say "ZZD1RESTART-$1ZZ $2"; if [ "$MODE" = DRY ]; then FAILS=$((FAILS+1)); else exit "${3:-5}"; fi; }
# the E3 tree fingerprint: sorted "path size" of every non-bytecode file under stack + taniteval, + sha256 of the supervisor and the trainer. SAME STRING in resume_e3.sh (a test pins it).
FPSH="{ find stack taniteval -type f ! -name '*.pyc' ! -name '*.pyo' ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' -exec stat -c '%n %s' {} + | LC_ALL=C sort; sha256sum stack/ops/sup_refcv7.sh stack/scripts/refc_v3_train.py; } | sha256sum | cut -c1-64"
say "START (pid $$) mode $MODE tip ${TIP:0:12} D1REC ${DREC:0:12}"
# ---- E3 state on Thor ------------------------------------------------------------------------------------------------------------
[ -n "$(th "test -e $OUTD/PAUSED.json && echo y")" ] || bad NOPAUSED "no $OUTD/PAUSED.json: E3 has not paused (restart D1 only after the one-shot pause)"
[ -z "$(th "grep -l '\"done\"[[:space:]]*:[[:space:]]*true' $OUTD/summary.json 2>/dev/null")" ] || bad NOPAUSED "$OUTD/summary.json is done: E3 finished, D1 is not an early branch any more"
[ -n "$(th "test -s $OUTD/ckpt_full_30000.pt && echo y")" ] || bad NOCKPT "no $OUTD/ckpt_full_30000.pt (D1's --wsd-branch-from source)"
[ -n "$(th "test -s $R/runs.d/r9e3_$V.env && echo y")" ] || bad NOMANIFEST "no $R/runs.d/r9e3_$V.env"
MC=$(th "sed -n 's/^export COMMIT=//p' $R/runs.d/r9e3_$V.env"); LC=$(th "cat $R/launch_tree.commit 2>/dev/null")
[ ${#MC} = 40 ] || bad NOMANIFEST "E3's manifest COMMIT '${MC:0:12}' is not 40 chars"
PRES=$R/launch_tree_e3_${MC:0:12}
NCH=$(th "n=0; for p in \$(ps -eo pid,args | awk '/[k]it\\/r9gate3_chain.sh/ {print \$1}'); do [ \"\$(readlink /proc/\$p/cwd)\" = $R ] && n=\$((n+1)); done; echo \$n")
[ "$NCH" = 0 ] || bad CHAINLIVE "$NCH gate3 chain process(es) with cwd $R (E3's chain must have ended at the pause)" 7
NSUP=$(th "ps -eo pid,args | awk -v p=$R/launch_tree 'index(\$0,p) && /[s]up_refcv7/' | wc -l")
[ "$NSUP" = 0 ] || bad SUPLIVE "$NSUP sup_refcv7 process(es) run from $R/launch_tree: the pause has not released the arm" 7
[ "$MODE" = DRY ] || [ -n "$(th "flock -n $LOCKF true && echo FREE")" ] || bad GPULOCK "the Thor GPU lock is held" 7
# E3's tree: in place (launch_tree.commit == COMMIT), or already preserved (this script ran before)
STATE=""
if [ -n "$MC" ] && [ "$LC" = "$MC" ]; then
  [ -z "$(th "test -e $PRES && echo y")" ] || bad PRESEXISTS "$PRES exists while $R/launch_tree is still E3's tree: inspect, never overwritten" 9
  STATE=INPLACE
elif [ -n "$MC" ] && [ -n "$(th "test -d $PRES && test -s $PRES.fp && test \"\$(cat $PRES.commit)\" = $MC && echo y")" ]; then
  [ "$(th "cd $PRES && $FPSH")" = "$(th "cat $PRES.fp")" ] || bad TREEMISMATCH "$PRES content does not match its recorded fingerprint" 9
  STATE=PRESERVED
else bad TREEMISMATCH "E3's tree (${MC:0:12}) is neither at $R/launch_tree (commit ${LC:0:12}) nor preserved at $PRES" 9; fi
say "E3 tree state: ${STATE:-none} (E3 COMMIT ${MC:0:12}, launch_tree.commit ${LC:0:12}, preserved path $PRES)"
# ---- the tip, the closure, D1's cost + argv, D1's records ----------------------------------------------------------------------------
git --git-dir=$GD fetch -q origin $BR || bad NOTIP "git fetch origin $BR failed" 10
RT=$(git --git-dir=$GD ls-remote origin $BR | cut -c1-40)
[ "$RT" = "$TIP" ] || bad NOTIP "the remote tip is ${RT:0:12}, not ${TIP:0:12} (r9gate3_restart.sh:21 would refuse): pass the current tip" 10
git --git-dir=$GD cat-file -e "$TIP^{commit}" 2>/dev/null || bad NOTIP "tip ${TIP:0:12} not in the mirror" 10
REC=$(git --git-dir=$GD rev-parse "$DREC^{commit}" 2>/dev/null); [ ${#REC} = 40 ] || bad NOD1RECORDS "D1REC $DREC is not a commit in the mirror" 6
RC12=${REC:0:12}
if [ ${#REC} = 40 ]; then
  # the SAME record-input list as prep_q28.sh / restart_e3.sh (a test pins it)
  CLOSED=1
  git --git-dir=$GD diff --quiet $REC $TIP -- $RD/r9e3_$V.argv.json $RD/r9e3_${V}_d1.argv.json $RD/r9e3_$V.pi_cost_approval.json $RD/r9e3_${V}_d1.pi_cost_approval.json \
      stack/scripts/map_hires_overfit.py stack/scripts/g_box_overfit.py stack/scripts/refc_v3_train.py stack/tanitad taniteval || CLOSED=0
  NC=$(git --git-dir=$GD diff $REC $TIP -- stack/ops/r9e1_bind.sh | grep -E '^[+-][^+-]' | grep -v -E '^[+-][[:space:]]*#' | wc -l)
  [ "$NC" = 0 ] || CLOSED=0
  [ "$CLOSED" = 1 ] || bad CLOSUREMOVED "a record input changed $RC12..${TIP:0:12} (model/harness closure or r9e1_bind.sh non-comment lines): D1's records at $RC12 do not fit the tip. D1-ONLY REBIND FIRST at a commit whose closure equals the tip's: ship rectree (R9W_ROOT=$R/rectree_<c12> bash r9w_ship.sh tree <REC40>; the kit's r9w_ship.sh), then on Thor  cd $R && R9Q28_D1_ONLY=1 setsid nohup bash kit/r9_recqueue28.sh <REC40>  -> ZZRECQ28-DONEZZ, then re-run this script with that REC" 6
fi
QS=$R/logs/recq28_$RC12.status
[ "$(th "grep -c '${D1}-RECORDS-READY at $RC12' $QS 2>/dev/null")" -ge 1 ] 2>/dev/null || bad NOD1RECORDS "queue 28 has no '${D1}-RECORDS-READY at $RC12' line in $QS: D1 has no records at that REC (D1-only rebind: R9Q28_D1_ONLY=1 bash kit/r9_recqueue28.sh <REC40> after shipping rectree_<c12>)" 6
git --git-dir=$GD show "$TIP:$RD/r9e3_$D1.pi_cost_approval.json" 2>/dev/null | grep -q "\"max_s_per_step\": $CAP," || bad NOAPPROVAL "no $RD/r9e3_$D1.pi_cost_approval.json with the $CAP cap at $TIP" 14
OKA=$(git --git-dir=$GD show "$TIP:$RD/r9e3_$D1.argv.json" 2>/dev/null | C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -c "import json,sys; a=json.load(sys.stdin)['argv']; v=lambda k: a[a.index(k) + 1]; print(1 if a.count('--steps') == 1 and v('--steps') == '36000' and a.count('--wsd-branch-from') == 1 and v('--wsd-branch-from').endswith('/r9e3/runs/r9e3-K2DC02/ckpt_full_30000.pt') and v('--wsd-branch-decay') == '6000' and '--save-at' not in a and a.count('--r9-map-banded') == 1 else 0)" 2>/dev/null | tr -d '\r')
[ "$OKA" = 1 ] || bad NOARM "r9e3_$D1.argv.json at $TIP is not the D1 argv (--steps 36000, --wsd-branch-from ckpt_full_30000.pt, decay 6000, no --save-at, banded map)" 14
git --git-dir=$GD cat-file -e "$TIP:stack/ops/r9e1_gate3/r9gate3_zg_env.sh" 2>/dev/null || bad NOZGENV "no r9gate3_zg_env.sh at $TIP" 14
if [ "$MODE" = DRY ]; then say "ZZD1RESTART-DRYENDZZ tip ${TIP:0:12}: $FAILS failing guard(s); E3 tree $STATE; nothing preserved, shipped or restarted"; exit 0; fi
# ---- REAL: 1. kit at the tip (verified blob by blob)  2. PRESERVE E3's tree  3. the D1 restart  4. wait for the suite ------------------------
D=/c/Users/Admin/r9restart_d1_${TIP:0:7}
[ -e "$D" ] && { say "ZZD1RESTART-KITEXISTSZZ $D already exists"; exit 9; }
mkdir -p "$D" && git --git-dir=$GD -c core.autocrlf=false archive $TIP stack/ops | tar -x -C "$D" || { say "ZZD1RESTART-KITZZ archive"; exit 9; }
BAD=0
while read -r mode typ blob path; do
  [ "$(git --git-dir=$GD hash-object --no-filters "$D/$path")" = "$blob" ] || { say "blob mismatch $path"; BAD=1; }
done < <(git --git-dir=$GD ls-tree -r $TIP stack/ops/r9e1_gate3)
[ "$BAD" = 0 ] || { say "ZZD1RESTART-KITZZ blob mismatch"; exit 9; }
if [ "$STATE" = INPLACE ]; then
  FP=$(th "cd $R/launch_tree && $FPSH"); [ ${#FP} = 64 ] || { say "ZZD1RESTART-FPZZ fingerprint of $R/launch_tree failed"; exit 9; }
  th "echo $FP > $PRES.fp && echo $MC > $PRES.commit && echo ok" | grep -q ok || { say "ZZD1RESTART-PRESERVEZZ could not write the sidecars"; exit 9; }
  th "mv $R/launch_tree $PRES && test -d $PRES && echo moved" | grep -q moved || { say "ZZD1RESTART-PRESERVEZZ rename failed"; exit 9; }
  say "PRESERVED E3's tree: $R/launch_tree -> $PRES (commit ${MC:0:12}, fp ${FP:0:16}); sidecars $PRES.commit $PRES.fp"
fi
say "kit $D verified; restarting [$D1] REC12 $RC12 study e3 at ${TIP:0:12}"
R9_STUDY=e3 R9_PRECHECK_ARMS="$D1" bash "$D/stack/ops/r9e1_gate3/r9gate3_restart.sh" "$TIP" "$D1" $RC12 > "$D/restart.out" 2>&1
tail -3 "$D/restart.out" | cut -c1-240 | tee -a "$OUT"
if grep -q "ZZR9GATE3-RESTART-OKZZ ${TIP:0:12}" "$D/restart.out"; then
  C12=${TIP:0:12}; t0=$(date +%s)
  while :; do
    ST=$(th "awk '/ START commit $C12 /{f=1} f' $R/gate3/CHAIN_STATUS.txt | grep -E ' (SUITE-STARTED|STOP-[A-Z0-9-]+) ' | tail -1")
    case "$ST" in
      *" STOP-"*) say "ZZD1RESTART-CHAINSTOPZZ $(echo "$ST" | cut -c1-200)"; exit 11;;
      *SUITE-STARTED*) say "suite started: $(echo "$ST" | cut -c1-120)"; break;;
    esac
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { say "ZZD1RESTART-NOSUITEZZ no SUITE-STARTED in 90 min"; exit 12; }
    sleep 60
  done
  say "ZZD1RESTART-OKZZ ${TIP:0:12} [$D1]: D1 chain armed (gate then 6,000 steps follow detached). E3's tree is preserved at $PRES; after D1's ALLDONE run resume_e3.sh DRY then REAL"
  exit 0
fi
# a refusal BEFORE the ship leaves no launch_tree: put E3's tree back (rename, never a copy over something)
if [ "$STATE" = INPLACE ] && [ -z "$(th "test -e $R/launch_tree && echo y")" ]; then
  th "mv $PRES $R/launch_tree && echo back" | grep -q back && say "restart refused before the ship: E3's tree is back at $R/launch_tree (preserve sidecars kept)"
fi
say "ZZD1RESTART-REFUSEDZZ (see $D/restart.out; if launch_tree is now D1's tree, E3's is at $PRES and resume_e3.sh restores it)"; exit 8
