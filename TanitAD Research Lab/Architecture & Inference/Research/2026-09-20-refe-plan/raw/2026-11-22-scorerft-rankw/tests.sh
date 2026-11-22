#!/bin/bash
# MM 2026-11-21: the registration land's tests, on the exported tip refe/ with SCORERFT_RANK_W filled (prep.py). Serial, shared pytest mutex, commit_guard,
# per-run TMP on F:. Writes rankw_land/tests.tsv; prints ZZRANKW_TESTS_OK only if every file passes (rc 0 and its pass token).
set -u
# 11-22: X on E: (F: below floor, C: 0.5 GB) and Q = prep.py's tip export of stack/ + taniteval/ + tools/ (no longer the C: sweep tree swp/q).
S="C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad"; O="$S/rankw_land"; X=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/rankw_x
Q=$X/q; PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe; LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
CG=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/tools/commit_guard.py
cd "$X/refe" || exit 2; mkdir -p "$X/tmp"; WT="$(cygpath -w $X/tmp)"; PP="$(cygpath -w $X/refe);$(cygpath -w $Q/stack);$(cygpath -w $Q)"
: > "$O/tests.tsv"; bad=0
run() {   # <file> <args> <token>
  until "$PY" "$CG" --min 6 > /dev/null 2>&1; do sleep 30; done
  until mkdir "$LK" 2>/dev/null; do sleep 15; done
  TMP="$WT" TEMP="$WT" TMPDIR="$WT" PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 timeout 3600 "$PY" "$1" $2 > "$O/t_${1%.py}${2// /}.log" 2>&1; rc=$?
  rm -rf "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/rankw_x/tmp"; mkdir -p "$X/tmp"; rmdir "$LK"
  last="$(tail -n 1 "$O/t_${1%.py}${2// /}.log" | tr -d '\r' | cut -c1-140)"
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
