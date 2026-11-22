#!/bin/bash
# prep_q27.sh <REC SHA40> -- MM 4a3d05, 2026-11-20 (derived from prep_q27.sh md5 7d852a72). Readies K2DC02cl's launch-gate RECORDS on the idle Thor (no training: the
# retrain itself waits for the PI's go). Steps: Thor-idle + disk guards -> ship rectree_<c12> (r9w_ship.sh tree) -> ship
# r9_recqueue27.sh (LF, md5) -> R9Q27_DRY=1 (all dry-runs, no GPU) -> REAL queue 27 detached (map-banded + box K2DC02cl, then K0 q19).
# Log: scratchpad k2dc/PREP_Q27.txt. Ends ZZPREPQ27-STARTEDZZ or ZZPREPQ27-STOP-<why>ZZ.
export PATH="/usr/bin:/bin:$PATH"
REC=${1:?usage: prep_q27.sh <REC SHA40>}; C12=${REC:0:12}
S=${R9K_LOG:-C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/4a3d05c7-0ead-4eba-a77c-659814aeb9bf/scratchpad/k2dc}
OUT=$S/PREP_Q27.txt; R1=/home/nvidia/r8ns/r9e1
SHIP=C:/Users/Admin/refcv8_pf/refcv9_warm/kit/r9w_ship.sh; RQ="$(cd "$(dirname "$0")" && pwd)/r9_recqueue27.sh"; STAGE=/f/refcv9_warm/ship
say() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$OUT"; }
stop() { say "ZZPREPQ27-STOP-$1ZZ $2"; exit "${3:-8}"; }
th() { timeout 90 ssh -n -o BatchMode=yes -o ConnectTimeout=20 tanitad-thor "$1" 2>&1 | tr -d '\r'; }
[ ${#REC} = 40 ] || stop USAGE "REC must be a 40-char sha" 2
TIP=$(cd /c/Users/Admin/tanitad-push && git ls-remote origin refs/heads/agent/arch-inf-20260803 | cut -c1-40)
[ "$TIP" = "$REC" ] || stop TIP "REC $C12 is not the remote tip ${TIP:0:12}" 2
NB=$(th "ps -eo args | grep -c -E '[r]9e2_roll.sh|[r]efc_v3_train.py|[r]9_recqueue|[r]9gate3_chain.sh'")
[ "$NB" = 0 ] || stop BUSY "$NB trainer / roll / queue / chain processes on Thor" 8
FREE=$(th "df -BG --output=avail / | tail -1 | tr -dc 0-9")
[ "${FREE:-0}" -ge 8 ] 2>/dev/null || stop DISK "Thor / has ${FREE} GB free (< 8)" 8
say "guards OK: tip $C12, Thor idle, / ${FREE} GB free"
[ -n "$(th "test -d $R1/rectree_$C12/launch_tree && echo y")" ] && stop EXISTS "$R1/rectree_$C12/launch_tree already exists (inspect, do not overwrite)" 8
SO=$(R9W_ROOT=$R1/rectree_$C12 bash $SHIP tree $REC 2>&1 < /dev/null); say "rectree ship: $(echo "$SO" | tail -n 2 | cut -c1-200 | tr '\n' '|')"
echo "$SO" | grep -q 'ZZR9WSHIP-TREE-OKZZ' || stop RECTREE "r9w_ship.sh tree did not end ZZR9WSHIP-TREE-OKZZ" 8
for p in "$STAGE/launch_tree_$C12.tar"; do [ -e "$p" ] && { sz=$(du -m "$p" | cut -f1); rm -f -- "$p"; say "DELLOG $p (${sz} MB; re-creatable from git at $C12)"; }; done
[ -d "$STAGE/x_$C12" ] && { sz=$(du -sm "$STAGE/x_$C12" | cut -f1); rm -rf -- "${STAGE:?}/x_$C12"; say "DELLOG $STAGE/x_$C12 (${sz} MB; re-creatable from git)"; }
[ "$(th "cat $R1/rectree_$C12/launch_tree.commit 2>/dev/null")" = "$REC" ] || stop RECTREE "$R1/rectree_$C12/launch_tree.commit is not $REC" 8
[ -n "$(th "test -s $R1/rectree_$C12/launch_tree_$C12.tar && echo y")" ] && { th "rm -f -- $R1/rectree_$C12/launch_tree_$C12.tar" >/dev/null; say "DELLOG Thor $R1/rectree_$C12/launch_tree_$C12.tar (extracted tree kept)"; }
RQM=$(tr -d '\r' < "$RQ" | md5sum | cut -c1-32)
tr -d '\r' < "$RQ" | timeout 90 ssh -o ConnectTimeout=20 tanitad-thor "mkdir -p $R1/kit $R1/logs && cat > $R1/kit/r9_recqueue27.sh" || stop RECQSHIP "ship of r9_recqueue27.sh failed" 8
[ "$(th "md5sum $R1/kit/r9_recqueue27.sh | cut -c1-32")" = "$RQM" ] || stop RECQSHIP "r9_recqueue27.sh md5 differs on Thor" 8
say "r9_recqueue27.sh shipped, md5 $RQM"
DO=$(timeout 900 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $R1 && R9Q27_DRY=1 bash kit/r9_recqueue27.sh $REC K2DC02cl < /dev/null > logs/recq27_${C12}.drynohup 2>&1; tail -n 4 logs/recq27_$C12.status" 2>&1 | tr -d '\r')
say "DRY: $(echo "$DO" | tr '\n' '|' | cut -c1-400)"
echo "$DO" | grep -q ZZRECQ27-DRYONLYZZ || stop DRY "queue 27 dry-run did not end ZZRECQ27-DRYONLYZZ" 3
QO=$(timeout 90 ssh -n -o ConnectTimeout=20 tanitad-thor "cd $R1 && setsid nohup bash kit/r9_recqueue27.sh $REC K2DC02cl > logs/recq27_${C12}.nohup 2>&1 < /dev/null & sleep 3; echo started" 2>&1 | tr -d '\r')
echo "$QO" | grep -q started || stop START "queue 27 did not start: $QO" 8
say "ZZPREPQ27-STARTEDZZ queue 27 REAL at $C12 (status $R1/logs/recq27_$C12.status; ~5.3 h ESTIMATED, then K0 queue 19)"
