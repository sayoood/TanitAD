#!/bin/bash
# MM 2026-11-20 (sweep3 = sweep2 + per-test TMP/TEMP inside the sweep dir, deleted after each file, and a C: >= 6 GB free wait: test_p6_levers leaks ~3 GB/run into %TEMP%)
# MM 2026-11-20: launch-gate sweep runner (fixes the agent's sweep.sh: the list lost "&"; commit guard computed but never waited on).
# usage: sweep2.sh <tree-root> <label> <list file of repo-relative paths>. One file at a time under the shared pytest mutex; waits for
# commit >= 6 GB; a listed file that does not exist is a FAIL row (rc 99); rc 4 / rc 5-with-no-script are FAILs, never passes.
S="C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
CG=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/tools/commit_guard.py
ROOT="$1"; LABEL="$2"; LIST="$3"; OUT="$S/sweep2_$LABEL"; mkdir -p "$OUT"; : > "$OUT/results.tsv"
while IFS= read -r f; do
  [ -n "$f" ] || continue
  name=$(printf '%s' "$f" | tr '/ &' '___')
  if [ ! -f "$ROOT/$f" ]; then printf '%s\tmissing\t99\tFILE NOT FOUND\t\n' "$f" >> "$OUT/results.tsv"; continue; fi
  case "$f" in "$PK"*) cwd="$ROOT/$PK/refe"; rel="$(basename "$f")";; *) cwd="$ROOT"; rel="$f";; esac
  until "$PY" "$CG" --min 6 > /dev/null 2>&1; do sleep 60; done
  until [ "$(df -k /c | tail -n 1 | awk '{print int($4/1048576)}')" -ge 6 ]; do sleep 60; done
  until mkdir "$LK" 2>/dev/null; do sleep 15; done
  cd "$cwd" || { rmdir "$LK"; printf '%s\tcd\t98\tcd failed\t\n' "$f" >> "$OUT/results.tsv"; continue; }
  mkdir -p "${OUT:?}/tmp"; export TMP="$(cygpath -w "${OUT:?}/tmp")" TEMP="$(cygpath -w "${OUT:?}/tmp")"
  PP="$(cygpath -w "$ROOT/stack");$(cygpath -w "$ROOT/$PK/refe");$(cygpath -w "$ROOT")"
  PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 timeout 900 "$PY" -m pytest -q -p no:cacheprovider --continue-on-collection-errors --basetemp "$OUT/bt" "$rel" > "$OUT/$name.log" 2>&1
  rc=$?; mode=pytest
  if [ $rc -eq 5 ] && [ "${f#$PK}" != "$f" ]; then
    fl=""; grep -q -- "--fast" "$rel" && fl="--fast"
    PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 timeout 900 "$PY" "$rel" $fl > "$OUT/$name.script.log" 2>&1; rc=$?; mode="script$fl"
  fi
  rm -rf "${OUT:?}/bt" "${OUT:?}/tmp"; rmdir "$LK"
  printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$mode" "$rc" "$(tail -n 1 "$OUT/$name.log" | tr -d '\r' | cut -c1-120)" "$(tail -n 1 "$OUT/$name.script.log" 2>/dev/null | tr -d '\r' | cut -c1-100)" >> "$OUT/results.tsv"
done < "$LIST"
echo SWEEP_DONE >> "$OUT/results.tsv"
