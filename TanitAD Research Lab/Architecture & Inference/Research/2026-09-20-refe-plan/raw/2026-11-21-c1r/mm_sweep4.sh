#!/bin/bash
# MM 2026-11-21 (sweep4 = sweep3 with the per-test TMP / TEMP / TMPDIR and the pytest basetemp on F: (F:/mm_e07efd49/sweeps/<label>, 3 levels deep),
# deleted after each file; waits for F: >= 8 GB and C: >= 4 GB free (C: is at ~6 GB and other sessions' pinned runs need it). Logs + results.tsv stay in
# the scratchpad. usage: sweep4.sh <tree-root> <label> <list file of repo-relative paths>.
S="C:/Users/Admin/AppData/Local/Temp/claude/4ac3a056b343bb35/e07efd49-d158-4484-85f3-d8888196bc46/scratchpad"
PK="TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
LK=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorer_ft/pytest.lockdir
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
CG=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain/tools/commit_guard.py
ROOT="$1"; LABEL="$2"; LIST="$3"; OUT="$S/sweep2_$LABEL"; FT="F:/mm_e07efd49/sweeps/$LABEL"; mkdir -p "$OUT" "$FT"; : > "$OUT/results.tsv"
freegb() { df -k "$1" | tail -n 1 | awk '{print int($4/1048576)}'; }
while IFS= read -r f; do
  [ -n "$f" ] || continue
  name=$(printf '%s' "$f" | tr '/ &' '___')
  if [ ! -f "$ROOT/$f" ]; then printf '%s\tmissing\t99\tFILE NOT FOUND\t\n' "$f" >> "$OUT/results.tsv"; continue; fi
  case "$f" in "$PK"*) cwd="$ROOT/$PK/refe"; rel="$(basename "$f")";; *) cwd="$ROOT"; rel="$f";; esac
  until "$PY" "$CG" --min 6 > /dev/null 2>&1; do sleep 60; done
  until [ "$(freegb /f)" -ge 8 ] && [ "$(freegb /c)" -ge 4 ]; do sleep 60; done
  until mkdir "$LK" 2>/dev/null; do sleep 15; done
  cd "$cwd" || { rmdir "$LK"; printf '%s\tcd\t98\tcd failed\t\n' "$f" >> "$OUT/results.tsv"; continue; }
  mkdir -p "${FT:?}/tmp"; W="$(cygpath -w "${FT:?}/tmp")"; export TMP="$W" TEMP="$W" TMPDIR="$W"
  PP="$(cygpath -w "$ROOT/stack");$(cygpath -w "$ROOT/$PK/refe");$(cygpath -w "$ROOT")"
  PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 timeout 900 "$PY" -m pytest -q -p no:cacheprovider --continue-on-collection-errors --basetemp "$FT/bt" "$rel" > "$OUT/$name.log" 2>&1
  rc=$?; mode=pytest
  if [ $rc -eq 5 ] && [ "${f#$PK}" != "$f" ]; then
    fl=""; grep -q -- "--fast" "$rel" && fl="--fast"
    PYTHONPATH="$PP" CUDA_VISIBLE_DEVICES=-1 timeout 900 "$PY" "$rel" $fl > "$OUT/$name.script.log" 2>&1; rc=$?; mode="script$fl"
  fi
  rm -rf "F:/mm_e07efd49/sweeps/${LABEL:?}/bt" "F:/mm_e07efd49/sweeps/${LABEL:?}/tmp"; rmdir "$LK"
  printf '%s\t%s\t%s\t%s\t%s\n' "$f" "$mode" "$rc" "$(tail -n 1 "$OUT/$name.log" | tr -d '\r' | cut -c1-120)" "$(tail -n 1 "$OUT/$name.script.log" 2>/dev/null | tr -d '\r' | cut -c1-100)" >> "$OUT/results.tsv"
done < "$LIST"
rmdir "F:/mm_e07efd49/sweeps/${LABEL:?}" 2>/dev/null
echo SWEEP_DONE >> "$OUT/results.tsv"
