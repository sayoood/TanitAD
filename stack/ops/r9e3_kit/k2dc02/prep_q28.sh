#!/bin/bash
# prep_q28.sh <REC SHA40> -- MM 4a3d05, 2026-11-23 (derived from prep_q27.sh). Readies E3's (arm K2DC02, the refcv9 full run) launch-gate RECORDS and the e3 root on Thor WHILE Thor TRAINS another arm
# (K2DC02cl): no idle guard (the queue waits for the GPU lock, it never forces it), only disk guards. Steps:
#   guards (disk on Thor / F:, REC exists on the remote and is an ancestor of the remote tip, the record inputs at REC equal the tip's)
#   -> e3 ROOT SETUP (E3_STUDY_NOTE Q5 items 3-5 + the early-D1 pause stamp; every step idempotent, never overwrites a differing file):
#        kit (r9w_ship.sh kit under R9W_ROOT=/home/nvidia/r8ns/r9e3; skipped when r9w_gsuite_precheck.py already matches),
#        gate3/zerograd_controls.env (= the committed zerograd_controls.env.derived at REC: the shape r9gate3_zg_env.sh derives under refcv9-full; Q4),
#        gate3/REC_C12 (= the REC12 stamp), gate3/PAUSE_AT_STEP (early D1; R9E3_PAUSE_AT, default 30000, "none" = no pause stamp)
#   -> ship rectree_<c12> (r9w_ship.sh tree) under the e3 root -> ship r9_recqueue28.sh (LF, md5) -> R9Q28_DRY=1 (all dry-runs, no GPU) -> REAL queue 28 DETACHED
#      (it sits on the Thor GPU lock until K2DC02cl's chain releases it; ~4 x 81 min of GPU once it has it, ESTIMATED from queue 27's measured 79.6 / 82.8 min).
# Log: $R9K_LOG/PREP_Q28.txt (default: the scratchpad k2dc folder). Markers ZZPREPQ28-STARTEDZZ or ZZPREPQ28-STOP-<why>ZZ.
export PATH="/usr/bin:/bin:$PATH"
REC=${1:?usage: prep_q28.sh <REC SHA40>}; C12=${REC:0:12}
S=${R9K_LOG:-C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc}
OUT=$S/PREP_Q28.txt; R3=/home/nvidia/r8ns/r9e3; GD=C:/Users/Admin/tanitad-push/.git; BR=agent/arch-inf-20260803
KITD=C:/Users/Admin/refcv8_pf/refcv9_warm/kit; SHIP=$KITD/r9w_ship.sh; RQ="$(cd "$(dirname "$0")" && pwd)/r9_recqueue28.sh"
STAGE=/f/c_overflow_2026-10-26/tmp/r9gate3_ship_e3; PAUSE_AT="${R9E3_PAUSE_AT:-30000}"
RD=stack/ops/runs.d; ZGD=stack/ops/r9e1_gate3/zerograd_controls.env.derived
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT"; }
stop() { say "ZZPREPQ28-STOP-$1ZZ $2"; exit "${3:-8}"; }
th() { timeout 90 ssh -n -o BatchMode=yes -o ConnectTimeout=20 tanitad-thor "$1" 2>&1 | tr -d '\r'; }
[ ${#REC} = 40 ] || stop USAGE "REC must be a 40-char sha" 2
case "$PAUSE_AT" in none) ;; *[!0-9]*|0*|"") stop PAUSEARG "R9E3_PAUSE_AT '$PAUSE_AT' must be a positive integer or none" 2;; esac
# ---- guards ----------------------------------------------------------------------------------------------------------------
git --git-dir=$GD fetch -q origin $BR || stop FETCH "git fetch origin $BR failed" 8
TIP=$(git --git-dir=$GD ls-remote origin refs/heads/$BR | cut -c1-40)
[ ${#TIP} = 40 ] || stop TIP "could not read the remote tip" 8
git --git-dir=$GD cat-file -e "$REC^{commit}" 2>/dev/null || stop NOREC "REC $C12 is not a commit in the mirror" 2
git --git-dir=$GD cat-file -e "$TIP^{commit}" 2>/dev/null || stop NOTIP "remote tip ${TIP:0:12} not in the mirror after the fetch" 8
git --git-dir=$GD merge-base --is-ancestor $REC $TIP || stop NOTREMOTE "REC $C12 is not an ancestor of the remote tip ${TIP:0:12} (not on the remote branch)" 2
git --git-dir=$GD diff --quiet $REC $TIP -- $RD/r9e3_K2DC02.argv.json $RD/r9e3_K2DC02_d1.argv.json $RD/r9e3_K2DC02.pi_cost_approval.json $RD/r9e3_K2DC02_d1.pi_cost_approval.json \
    stack/scripts/map_hires_overfit.py stack/scripts/g_box_overfit.py stack/scripts/refc_v3_train.py stack/tanitad taniteval || stop RECINPUT "a record input changed $C12..${TIP:0:12}: MM decision (a closure-touching land voids the records)" 6
NC=$(git --git-dir=$GD diff $REC $TIP -- stack/ops/r9e1_bind.sh | grep -E '^[+-][^+-]' | grep -v -E '^[+-][[:space:]]*#' | wc -l)
[ "$NC" = 0 ] || stop RECINPUT "r9e1_bind.sh has $NC non-comment changed lines $C12..${TIP:0:12}" 6
for f in $RD/r9e3_K2DC02.argv.json $RD/r9e3_K2DC02_d1.argv.json $RD/r9e3_K2DC02.pi_cost_approval.json $RD/r9e3_K2DC02_d1.pi_cost_approval.json $ZGD stack/ops/r9e1_gate3/r9gate3_study.sh; do
  git --git-dir=$GD cat-file -e "$REC:$f" 2>/dev/null || stop NOFILE "no $f at $C12" 2; done
FREE=$(th "df -BG --output=avail / | tail -1 | tr -dc 0-9")
[ "${FREE:-0}" -ge 8 ] 2>/dev/null || stop DISK "Thor / has ${FREE} GB free (< 8; the rectree is ~0.5 GB, E3 itself needs far more: sweep my own dead smoke checkpoints first)" 8
FF=$(df -BG /f | awk 'NR==2 {gsub("G","",$4); print $4}')
[ "${FF:-0}" -ge 10 ] 2>/dev/null || stop FDISK "F: has ${FF} GB free (< 10)" 8
say "guards OK: REC $C12 (remote tip ${TIP:0:12}), Thor / ${FREE} GB free, F: ${FF} GB free"
# ---- e3 root setup (idempotent; a differing existing file STOPS, it is never overwritten) -------------------------------------------
th "mkdir -p $R3/gate3 $R3/runs.d $R3/kit $R3/logs && echo MKOK" | grep -qx MKOK || stop ROOT "could not create the e3 root dirs on Thor" 8
WP=$(md5sum $KITD/r9w_gsuite_precheck.py | cut -c1-32)
if [ "$(th "md5sum $R3/kit/r9w_gsuite_precheck.py 2>/dev/null | cut -c1-32")" = "$WP" ]; then say "e3 kit: r9w_gsuite_precheck.py already matches ($WP): kit ship skipped"
else
  KO=$(R9W_ROOT=$R3 R9W_SHIP_STAGE=$STAGE bash $SHIP kit 2>&1 < /dev/null); say "e3 kit ship: $(echo "$KO" | tail -n 2 | cut -c1-200 | tr '\n' '|')"
  echo "$KO" | grep -q 'ZZR9WSHIP-KIT-OKZZ' || stop KIT "r9w_ship.sh kit did not end ZZR9WSHIP-KIT-OKZZ" 8
  [ -e "$STAGE/r9w_kit.tar" ] && { sz=$(du -k "$STAGE/r9w_kit.tar" | cut -f1); rm -f -- "$STAGE/r9w_kit.tar"; say "DELLOG $STAGE/r9w_kit.tar (${sz} KB; re-creatable from the kit folder)"; }
fi
ZLOC=$(git --git-dir=$GD show "$REC:$ZGD" | tr -d '\r' | md5sum | cut -c1-32)
ZTH=$(th "test -s $R3/gate3/zerograd_controls.env && md5sum $R3/gate3/zerograd_controls.env | cut -c1-32")
if [ -z "$ZTH" ]; then
  git --git-dir=$GD show "$REC:$ZGD" | tr -d '\r' | timeout 90 ssh -o ConnectTimeout=20 tanitad-thor "cat > $R3/gate3/zerograd_controls.env" || stop ZGENV "could not write the zero-gradient controls env" 8
  [ "$(th "md5sum $R3/gate3/zerograd_controls.env | cut -c1-32")" = "$ZLOC" ] || stop ZGENV "zerograd_controls.env md5 differs on Thor" 8
  say "zerograd_controls.env installed from $C12:$ZGD (md5 $ZLOC)"
elif [ "$ZTH" = "$ZLOC" ]; then say "zerograd_controls.env already present, md5 $ZLOC"
else stop ZGENV "$R3/gate3/zerograd_controls.env exists with md5 $ZTH != the committed derived one $ZLOC: MM decision (never overwritten)" 15; fi
RS=$(th "cat $R3/gate3/REC_C12 2>/dev/null")
if [ -z "$RS" ]; then th "echo $C12 > $R3/gate3/REC_C12" > /dev/null; [ "$(th "cat $R3/gate3/REC_C12")" = "$C12" ] || stop RECSTAMP "REC_C12 stamp not written" 8; say "REC_C12 stamp written: $C12"
elif [ "$RS" = "$C12" ]; then say "REC_C12 stamp already $C12"
else stop RECSTAMP "REC_C12 is $RS on Thor, not $C12: one stamp serves every arm of the root. MM decision" 15; fi
if [ "$PAUSE_AT" = none ]; then say "early-D1 pause stamp skipped (R9E3_PAUSE_AT=none): E3 will run to the end without pausing"
else
  PS=$(th "cat $R3/gate3/PAUSE_AT_STEP 2>/dev/null")
  if [ -z "$PS" ]; then th "echo $PAUSE_AT > $R3/gate3/PAUSE_AT_STEP" > /dev/null; [ "$(th "cat $R3/gate3/PAUSE_AT_STEP")" = "$PAUSE_AT" ] || stop PAUSESTAMP "PAUSE_AT_STEP not written" 8; say "PAUSE_AT_STEP stamp written: $PAUSE_AT (the chain pauses K2DC02 ONCE after ckpt_full_${PAUSE_AT}.pt + its eval row, never a *_d1 arm)"
  elif [ "$PS" = "$PAUSE_AT" ]; then say "PAUSE_AT_STEP stamp already $PAUSE_AT"
  else stop PAUSESTAMP "PAUSE_AT_STEP is $PS on Thor, not $PAUSE_AT. MM decision" 15; fi
fi
# ---- rectree + queue -----------------------------------------------------------------------------------------------------------
[ -n "$(th "test -d $R3/rectree_$C12/launch_tree && echo y")" ] && stop EXISTS "$R3/rectree_$C12/launch_tree already exists (inspect, do not overwrite)" 8
SO=$(R9W_ROOT=$R3/rectree_$C12 R9W_SHIP_STAGE=$STAGE bash $SHIP tree $REC 2>&1 < /dev/null); say "rectree ship: $(echo "$SO" | tail -n 2 | cut -c1-200 | tr '\n' '|')"
echo "$SO" | grep -q 'ZZR9WSHIP-TREE-OKZZ' || stop RECTREE "r9w_ship.sh tree did not end ZZR9WSHIP-TREE-OKZZ" 8
for p in "$STAGE/launch_tree_$C12.tar"; do [ -e "$p" ] && { sz=$(du -m "$p" | cut -f1); rm -f -- "$p"; say "DELLOG $p (${sz} MB; re-creatable from git at $C12)"; }; done
[ -d "$STAGE/x_$C12" ] && { sz=$(du -sm "$STAGE/x_$C12" | cut -f1); rm -rf -- "${STAGE:?}/x_$C12"; say "DELLOG $STAGE/x_$C12 (${sz} MB; re-creatable from git)"; }
[ "$(th "cat $R3/rectree_$C12/launch_tree.commit 2>/dev/null")" = "$REC" ] || stop RECTREE "$R3/rectree_$C12/launch_tree.commit is not $REC" 8
[ -n "$(th "test -s $R3/rectree_$C12/launch_tree_$C12.tar && echo y")" ] && { th "rm -f -- $R3/rectree_$C12/launch_tree_$C12.tar" >/dev/null; say "DELLOG Thor $R3/rectree_$C12/launch_tree_$C12.tar (extracted tree kept)"; }
RQM=$(tr -d '\r' < "$RQ" | md5sum | cut -c1-32)
tr -d '\r' < "$RQ" | timeout 90 ssh -o ConnectTimeout=20 tanitad-thor "mkdir -p $R3/kit $R3/logs && cat > $R3/kit/r9_recqueue28.sh" || stop RECQSHIP "ship of r9_recqueue28.sh failed" 8
[ "$(th "md5sum $R3/kit/r9_recqueue28.sh | cut -c1-32")" = "$RQM" ] || stop RECQSHIP "r9_recqueue28.sh md5 differs on Thor" 8
say "r9_recqueue28.sh shipped, md5 $RQM"
DO=$(timeout 900 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $R3 && R9Q28_DRY=1 bash kit/r9_recqueue28.sh $REC K2DC02 < /dev/null > logs/recq28_${C12}.drynohup 2>&1; tail -n 6 logs/recq28_$C12.status" 2>&1 | tr -d '\r')
say "DRY: $(echo "$DO" | tr '\n' '|' | cut -c1-600)"
echo "$DO" | grep -q ZZRECQ28-DRYONLYZZ || stop DRY "queue 28 dry-run did not end ZZRECQ28-DRYONLYZZ" 3
echo "$DO" | grep -q 'D1-DRY-REFUSED' && say "NOTE D1-DRY-REFUSED: E3's records will still be bound, D1's GPU binds are skipped (E3_STUDY_NOTE Q2: the D1 bind needs ckpt_full_30000.pt)"
QO=$(timeout 90 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $R3 && setsid nohup bash kit/r9_recqueue28.sh $REC K2DC02 > logs/recq28_${C12}.nohup 2>&1 < /dev/null & sleep 3; echo started" 2>&1 | tr -d '\r')
echo "$QO" | grep -q started || stop START "queue 28 did not start: $QO" 8
say "ZZPREPQ28-STARTEDZZ queue 28 REAL at $C12 (status $R3/logs/recq28_$C12.status; waits for the Thor GPU lock held by the arm now training; ~5.4 h of GPU ESTIMATED, E3 binds first)"
