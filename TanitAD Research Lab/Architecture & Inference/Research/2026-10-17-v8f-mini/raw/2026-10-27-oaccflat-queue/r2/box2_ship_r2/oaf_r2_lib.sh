#!/usr/bin/env bash
# oaf_r2_lib.sh -- the r2 additions of chain_v8f_dagoacc_r2.sh (the DAG-OACC re-run after the first attempt stopped at the gate asserter's read_o control, 2026-10-27). SOURCED by the r2 chain right after oaf_lib.sh.
# NEW FILE: oaf_lib.sh (which the running FLAT-DAG chain also sources) is not touched. Needs from the chain: B L TAG ART SSH SCP BOOT0 BUSY_TAKEN CUR and oaf_lib.sh's stop / launch / waitrc / clean.
#
# WHAT IT DOES: the first attempt left state behind on box 2 and on the dev box. Nothing of it makes the r2 chain REFUSE except (conditionally) the pinned training argv -- gatedag REFUSES when its pin differs from the argv derived now
# (a tip that moved) -- but a STALE gate record is the worse hazard: a gate that fails to rewrite selfcheck.json would leave the first attempt's file to be pulled and asserted as if fresh. So the first attempt's gate artifacts are
# MOVED ASIDE (renamed to <name>_r1_assert_stop; NEVER deleted, never overwritten) before anything else touches box 2 or E:.
#   dev box (r2_aside_local, FIRST, a rename): $ART (E:/Projects/TanitAD-artifacts/v8f-dagoacc: the first attempt's pulled gate, w_acc record and logs)
#   box 2   (one cmd job, r2_aside_job_text):  <runs>\gate_M2b-dagoacc_s<S> (dir), <runs>\gate_M2b-dagoacc_s<S>.log, <runs>\dagpin\M2b_s<S>_dagoacc.train.argv.txt, for S = 0 and 1
# LEFT IN PLACE ON PURPOSE: <runs>\dagoacc\ (fit windows, w_acc.json, w_acc.txt, waccdag.log): the queue's waccdag stage is "skipped once both exist", C-O3 re-checks the re-pulled record against the offline value, and the record
# is a deterministic CPU measurement that already passed C-O3 (49.6969 == offline). <runs>\dagpin\bank_sha_dagoacc_s<S>.txt is rewritten by the queue on every call (a redirect, no refusal).
# A RELAUNCH OF r2 IS SAFE, AND r2 NEVER MOVES ITS OWN FILES: the aside runs BEFORE r2 writes anything, so "present before the aside" means "the first attempt's".
#   dev box: <ART>_r1_assert_stop exists -> the aside was done by an earlier launch: <ART> (if any) is r2's own progress: KEPT.
#   box 2:   <runs>\r2_aside_done.txt (written by the job's last line) exists -> ASIDE-ALREADY-DONE, nothing is touched. Without the marker, per item: absent -> ASIDE-ABSENT; present -> renamed (ASIDE-MOVED); present AND its aside copy
#            present -> ASIDE-FAIL (an unexpected state: never overwritten, never guessed, the chain stops). A job that fails midway writes no marker, so its retry moves the rest.
R2_SUF=_r1_assert_stop

r2_box_items() {   # the box-2 items, relative to the runs root (cmd syntax); ONE source for the job text and for the count the chain asserts
  printf '%s\n' 'gate_M2b-dagoacc_s0' 'gate_M2b-dagoacc_s0.log' 'gate_M2b-dagoacc_s1' 'gate_M2b-dagoacc_s1.log' 'dagpin\M2b_s0_dagoacc.train.argv.txt' 'dagpin\M2b_s1_dagoacc.train.argv.txt'
}

r2_aside_job_text() {   # r2_aside_job_text <runs root, cmd form>: the box-2 job (CRLF). RENAMES ONLY (plus the one marker line): no del / rmdir / move / copy anywhere in it
  local rs=$1 it
  {
    echo '@echo off'
    echo 'setlocal'
    echo "set \"RS=$rs\""
    echo 'set "RC=0"'
    echo 'echo ===== r2 aside %DATE% %TIME%'
    echo 'if exist "%RS%\r2_aside_done.txt" (echo ASIDE-ALREADY-DONE an earlier launch of r2 moved the first attempt artifacts aside, marker %RS%\r2_aside_done.txt & goto :end)'
    while IFS= read -r it; do
      echo "call :aside \"%RS%\\$it\" || (set \"RC=1\" & goto :end)"
    done < <(r2_box_items)
    echo '>"%RS%\r2_aside_done.txt" echo r2 aside done %DATE% %TIME%'
    echo ':end'
    echo 'echo ===== ZZJOB-END %DATE% %TIME% rc=%RC%'
    echo 'exit /b %RC%'
    cat <<'EOF'
:aside
if not exist "%~1" (echo ASIDE-ABSENT %~1 & exit /b 0)
if exist "%~1_r1_assert_stop" (echo ASIDE-FAIL %~1 and its aside sibling %~nx1_r1_assert_stop both exist: refusing to overwrite or to guess & exit /b 1)
ren "%~1" "%~nx1_r1_assert_stop"
if errorlevel 1 (echo ASIDE-FAIL %~1 could not be renamed & exit /b 1)
echo ASIDE-MOVED %~1 to %~nx1_r1_assert_stop
exit /b 0
EOF
  } | sed 's/$/\r/'
}

r2_aside_local() {   # r2_aside_local <path>: rename <path> to <path>_r1_assert_stop. NEVER deletes, never overwrites; an existing aside copy means <path> is r2's own progress: kept
  local d=$1 n="$1$R2_SUF"
  [ -e "$d" ] || { echo "ASIDE-ABSENT $d"; return 0; }
  [ -e "$n" ] && { echo "ASIDE-KEPT $d : the first attempt's copy is already aside as $n"; return 0; }
  mv "$d" "$n" || stop "r2 aside: cannot rename $d to $n"
  { [ -e "$n" ] && [ ! -e "$d" ]; } || stop "r2 aside: $d -> $n did not happen"
  echo "ASIDE-MOVED $d to $n"
}

r2_aside_all() {   # the chain's FIRST action after BOX2_BUSY is taken and the boot probe answered: the dev box's pull dir (a rename), then the box-2 job (shipped, launched, waited for like the refresh job) and its report checked
  local j=v8fdagoacc${TAG}_r2aside n want
  r2_aside_local "$ART"
  r2_aside_job_text 'D:\tanitad\runs\v8f_mini_full' > $B/$j.cmd || stop "write the r2 aside job"
  $SCP $B/$j.cmd tanitad-box2:D:/tanitad/jobs/ || stop "ship the r2 aside job"
  rm -f $L/$j.log
  launch $j || stop "launch the r2 aside job"
  waitrc $j 40
  clean $j | grep -a '^ASIDE-'
  n=$(clean $j | grep -ac '^ASIDE-')
  if clean $j | grep -aq '^ASIDE-ALREADY-DONE'; then want=1; else want=$(r2_box_items | wc -l); fi
  [ "$n" = "$want" ] || stop "the r2 aside job reported $n ASIDE lines, expected $want"
}
