#!/bin/bash
# SFTT G-LIVE L2 amendment (MM 2026-11-23): the SCORER-FT registration test set on the exported tip refe/ + the 3 amended files (prep_l2.py).  13 runs, SERIAL, each preceded by a
# wait for >= 5.5 GB free commit (wait_commit.py, 60 s poll) and the shared pytest lock dir.  E: ONLY (logs, tmp, tests.tsv); nothing on C: or F:.  Prints ZZRANKW_TESTS_OK iff every run passes.
set -u
SHIP=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/ship; O=$SHIP/l2/raw/2026-11-23-sftt-gate-l2; X=$SHIP/l2_scratch/x; Q=$X/q
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
cd "$X/refe" || exit 2; mkdir -p "$SHIP/l2_scratch/tmp" "$SHIP/l2_scratch/logs"; WT="$(cygpath -w $SHIP/l2_scratch/tmp)"; PP="$(cygpath -w $X/refe);$(cygpath -w $Q/stack);$(cygpath -w $Q)"
: > "$O/tests.tsv"; bad=0
run() {   # <file> <args> <token>
  "$PY" "$O/wait_commit.py" 5.5 > /dev/null 2>&1
  until mkdir "$LK" 2>/dev/null; do sleep 15; done
  L="$SHIP/l2_scratch/logs/t_${1%.py}${2// /}.log"
  TMP="$WT" TEMP="$WT" TMPDIR="$WT" PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 timeout 3600 "$PY" "$1" $2 > "$L" 2>&1; rc=$?
  rmdir "$LK"
  last="$(tail -n 1 "$L" | tr -d '\r' | cut -c1-140)"
  ok=0; [ $rc = 0 ] && case "$last" in *"$3"*) ok=1;; esac
  [ $ok = 1 ] || bad=1
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$rc" "$ok" "$last" >> "$O/tests.tsv"
}
run test_scorerft_profile.py --fast ZZSCORERFTPROF_PASS
run test_scorerft_profile.py "" ZZSCORERFTPROF_PASS
run test_rank_loss.py --fast ZZRANK_PASS
for f in test_div_gate.py test_dose_gate.py test_rep_gate.py test_s1zsm_gate.py test_scorer1p6_gate.py test_vdac_gate.py; do run $f "" ZZTEST_PASS; done
run test_field_p7_profile.py --fast ZZFIELDP7PROF_PASS
run test_p5c_profile.py --fast ZZP5CPROF_PASS
run test_p5c_chain.py "" ZZP5CCHAIN_PASS
run test_p7_chain.py "" ZZP7CHAIN_PASS
cat "$O/tests.tsv"
[ $bad = 0 ] && echo ZZRANKW_TESTS_OK || echo ZZRANKW_TESTS_FAIL
