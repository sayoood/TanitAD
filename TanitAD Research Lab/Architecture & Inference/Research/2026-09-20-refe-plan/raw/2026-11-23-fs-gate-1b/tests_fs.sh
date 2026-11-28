#!/bin/bash
# Fs hybrid gate 1b (MM 2026-11-23): Fs's exact G-SUITE (10848403 launch_gate.PROFILES['refe_field_p7_s1']['suite']) on an export of 10848403 + 2 overlays (prep_fs.py).  SERIAL, each after a >= 5.5 GB commit wait and the shared pytest lock dir.
# E: ONLY.  Prints ZZFS_TESTS_OK iff every run passes.
set -u
SHIP=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/ship; O=$SHIP/fs/raw/2026-11-23-fs-gate-1b; X=$SHIP/fs_scratch/y; Q=$X/q
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
cd "$X/refe" || exit 2; mkdir -p "$SHIP/fs_scratch/tmp" "$SHIP/fs_scratch/logs"; WT="$(cygpath -w $SHIP/fs_scratch/tmp)"; PP="$(cygpath -w $X/refe);$(cygpath -w $Q/stack);$(cygpath -w $Q)"
: > "$O/tests.tsv"; bad=0
run() {   # <file> <args> <token>
  "$PY" "$O/wait_commit.py" 5.5 > /dev/null 2>&1
  until mkdir "$LK" 2>/dev/null; do sleep 15; done
  L="$SHIP/fs_scratch/logs/t_${1%.py}${2// /}.log"
  TMP="$WT" TEMP="$WT" TMPDIR="$WT" PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 timeout 3600 "$PY" "$1" $2 > "$L" 2>&1; rc=$?
  rmdir "$LK"
  last="$(tail -n 1 "$L" | tr -d '\r' | cut -c1-140)"
  ok=0; [ $rc = 0 ] && case "$last" in *"$3"*) ok=1;; esac
  [ $ok = 1 ] || bad=1
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$rc" "$ok" "$last" >> "$O/tests.tsv"
}
run test_field_p7_profile.py --fast ZZFIELDP7PROF_PASS
run test_field_p7_profile.py "" ZZFIELDP7PROF_PASS
run test_p2r_chain.py "" ZZTEST_PASS
run test_launch_gate.py "--static --no-banked" ZZLG_STATIC_PASS
run test_p6_levers.py --fast ZZP6_PASS
run test_p6_chain.py "" ZZP6CHAIN_PASS
run test_p7_head_wiring.py --static ZZP7HEAD_PASS
run test_ego_yaw.py --fast ZZEGOYAW_PASS
run test_traj_label_table.py --fast ZZTRAJLABEL_PASS
run test_p5c_profile.py --fast ZZP5CPROF_PASS
run test_p5c_live.py --fast ZZP5CLIVE_PASS
run test_p5c_postgate.py --fast ZZP5CPOST_PASS
run test_p5c_chain.py "" ZZP5CCHAIN_PASS
run test_field_loss.py "" ZZTEST_PASS
cat "$O/tests.tsv"
[ $bad = 0 ] && echo ZZFS_TESTS_OK || echo ZZFS_TESTS_FAIL
