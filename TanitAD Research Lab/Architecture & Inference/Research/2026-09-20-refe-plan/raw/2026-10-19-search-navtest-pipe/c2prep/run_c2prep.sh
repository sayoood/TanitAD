#!/bin/bash
# c2prep driver (one detached script, strictly sequential): (a) fixture chain -> (a2) verdict json -> (b) full mutation check. Never parallel (RAM).
export PATH="/usr/bin:/bin:$PATH"
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
export TEMP=C:/Users/Admin/c2prep_run/tmp TMP=C:/Users/Admin/c2prep_run/tmp NPMUT_DIR=C:/Users/Admin/c2prep_run/npmut
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
C2=C:/Users/Admin/par18/c2prep
RUN=C:/Users/Admin/c2prep_run
CODE="$C2/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-search-navtest-pipe/code"
cd "$CODE" || exit 90
echo "ZZC2PREP_START $(date +%FT%T)"
# (a) the fixture chain. The chain is resumable (valid outputs are skipped); a RAM_LOW stop (exit 7, its own guard at 3.0 GiB, left untouched) is re-launched at most 3 times after a 5-minute wait.
attempt=0
while : ; do
  attempt=$((attempt+1))
  echo "ZZC2PREP_FIXTURE_ATTEMPT $attempt $(date +%FT%T)"
  "$PY" -u navpipe_chain.py --mode fixture --work $RUN/fixture >> $C2/logs/fixture_chain.log 2>&1 < /dev/null
  rc=$?
  if [ "$rc" = "7" ] && [ $attempt -lt 4 ] && tail -n 3 $C2/logs/fixture_chain.log | grep -q "RAM_LOW"; then
    echo "ZZC2PREP_RAM_LOW_RETRY $attempt $(date +%FT%T)"; sleep 300; continue
  fi
  break
done
echo "rc=$rc attempts=$attempt $(date +%FT%T)" > $C2/logs/fixture.done
echo "ZZC2PREP_FIXTURE_DONE rc=$rc attempts=$attempt"
# (a2) the verdict json (cheap; written before the mutation run so it exists early)
"$PY" "$C2/fixture_verdict.py" > $C2/logs/fixture_verdict.log 2>&1 < /dev/null
echo "rc=$? $(date +%FT%T)" > $C2/logs/verdict.done
cat $C2/logs/fixture_verdict.log
# (b) the full mutation check: every @GUARD line, scratch copies only
"$PY" -u mutation_check.py --out $C2/raw/mutations_c2prep.json --keep-going > $C2/logs/mutation_check.log 2>&1 < /dev/null
mrc=$?
echo "rc=$mrc $(date +%FT%T)" > $C2/logs/mutation.done
echo "ZZC2PREP_MUTATION_DONE rc=$mrc"
echo "fixture_rc=$rc mutation_rc=$mrc $(date +%FT%T)" > $C2/logs/c2prep.done
echo "ZZC2PREP_ALL_DONE"
