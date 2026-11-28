#!/bin/bash
# Fs launch gate 1c (MM 2026-11-24): the gate test set on BOTH patched trees (side = 86d174bb + 1c, main = 9b71d987 + 1c), SERIAL, each run after
# >= 5.5 GB free commit and under the shared pytest lock dir. Trees, tmp and logs on C:/Users/Admin/swp/fs1c (own scratch). Prints ZZ1C_TESTS_OK iff every run passes.
set -u
B=C:/Users/Admin/swp/fs1c; PKR="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
CG=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/tools/commit_guard.py
mkdir -p $B/tmp $B/logs; WT="$(cygpath -w $B/tmp)"; OUT=$B/tests.tsv; : > $OUT; bad=0
# the dev box carries T's seam chain (s2 / det leave ~2 GB commit, MEASURED 2026-11-24 08:3xZ) and then the SCORER-FT read: run only after the read has ended
MR=E:/Projects/TanitAD-artifacts/scorerft_read/markers_read1; n=0
until [ -f $MR/read.DONE ] || [ -f $MR/read.FAILED ]; do n=$((n+1)); [ $n -gt 576 ] && { echo "ZZ1C_TESTS_FAIL read never ended in 48 h: NOT RUN"; exit 1; }; sleep 300; done
run() {   # <tree> <file> <args> <token>
  local T="$B/$1" R="$B/$1/$PKR/refe" i=0 g
  [ -f "$R/$2" ] || { printf '%s\t%s\t%s\tMISSING\t-\t-\n' "$1" "$2" "$3" >> $OUT; return; }
  while :; do g=$("$PY" "$CG" | tr -d '\r\n '); case "$g" in ''|*[!0-9]*) g=0;; esac; [ "$g" -ge 6 ] && break; i=$((i+1)); [ $i -gt 1200 ] && { printf '%s\t%s\t%s\tNOMEM\t0\t-\tcommit < 6 GB for 1200 min: NOT RUN\n' "$1" "$2" "$3" >> $OUT; bad=1; return; }; sleep 60; done
  until mkdir "$LK" 2>/dev/null; do sleep 15; done
  local L="$B/logs/${1}_${2%.py}${3// /}.log"; local t0=$(date +%s)
  ( cd "$R" && TMP="$WT" TEMP="$WT" TMPDIR="$WT" PYTHONPATH="$(cygpath -w "$R");$(cygpath -w "$T/stack");$(cygpath -w "$T")" CUDA_VISIBLE_DEVICES=-1 \
      timeout 3600 "$PY" "$2" $3 > "$L" 2>&1 ); local rc=$?
  rmdir "$LK"
  local last; last="$(tail -n 1 "$L" | tr -d '\r' | cut -c1-140)"; local ok=0
  [ $rc = 0 ] && case "$last" in *"$4"*) ok=1;; esac
  [ $ok = 1 ] || bad=1
  printf '%s\t%s\t%s\t%s\t%s\t%s s\t%s\n' "$1" "$2" "$3" "$rc" "$ok" "$(( $(date +%s) - t0 ))" "$last" >> $OUT
}
for t in side main; do
  run $t test_field_p7_profile.py "" ZZFIELDP7PROF_PASS
  run $t test_scorerft_profile.py "" ZZSCORERFTPROF_PASS
  run $t test_launch_gate.py "--static --no-banked" ZZLG_STATIC_PASS
  run $t test_p5c_profile.py --fast ZZP5CPROF_PASS
  run $t test_p5c_live.py "" ZZP5CLIVE_PASS
  run $t test_gate_gpu_probes.py "" ZZTEST_PASS
  run $t test_field_loss.py "" ZZTEST_PASS
done
cat $OUT
[ $bad = 0 ] && echo ZZ1C_TESTS_OK || echo ZZ1C_TESTS_FAIL
