#!/bin/bash
# MM 2026-11-20: full (mutation) modes of the --fast tests whose mutation tables target the files SCORER-FT changed, on the patched tree q.
# One at a time under the shared pytest mutex; per-test TMP/TEMP inside OUT (deleted after each); C: >= 6 GB and commit >= 6 GB first.
S=C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad
OUT="$S/fullmodes"; mkdir -p "$OUT"; : > "$OUT/results.tsv"
LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
CG=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/tools/commit_guard.py
ROOT=/c/Users/Admin/swp/q; PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
for t in test_scorerft_profile.py test_p5c_profile.py test_p5c_live.py test_ego_yaw.py test_traj_label_table.py test_p6_levers.py; do
  until "$PY" "$CG" --min 6 > /dev/null 2>&1; do sleep 60; done
  until [ "$(df -k /c | tail -n 1 | awk '{print int($4/1048576)}')" -ge 6 ]; do sleep 60; done
  until mkdir "$LK" 2>/dev/null; do sleep 15; done
  mkdir -p "${OUT:?}/tmp"; export TMP="$(cygpath -w "${OUT:?}/tmp")" TEMP="$(cygpath -w "${OUT:?}/tmp")"
  cd "$ROOT/$PK/refe" || { rmdir "$LK"; continue; }
  t0=$(date +%s)
  PYTHONPATH="$(cygpath -w $ROOT/stack);$(cygpath -w "$ROOT/$PK/refe");$(cygpath -w $ROOT)" CUDA_VISIBLE_DEVICES=-1 timeout 3000 "$PY" "$t" > "$OUT/$t.log" 2>&1; rc=$?
  rm -rf "${OUT:?}/tmp"; rmdir "$LK"
  printf '%s\tfull\t%s\t%ss\t%s\n' "$t" "$rc" "$(( $(date +%s) - t0 ))" "$(tail -n 1 "$OUT/$t.log" | tr -d '\r' | cut -c1-200)" >> "$OUT/results.tsv"
done
echo FULLMODES_DONE >> "$OUT/results.tsv"
cat "$OUT/results.tsv"
