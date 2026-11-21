#!/usr/bin/env bash
# Rebuild the CLEAN test tree: the lander tip (agent/arch-inf-20260803) stack/{tanitad,scripts,tests} + taniteval,
# then overlay this package's code/fix/ tree on top.  NEVER touches the D: working tree or G:.
#   usage: overlay_test_tree.sh <out_dir>        -> prints the Windows PYTHONPATH to use
set -euo pipefail
OUT="${1:?out dir}"
PK="D:/Projects/TanitAD/FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-x10-pose-timing"
export GIT_DIR=C:/Users/Admin/tanitad-push/.git
if [ ! -f "$OUT/.tip_ok" ]; then
  rm -rf "$OUT"; mkdir -p "$OUT"
  git archive --format=tar -o "$OUT/../tip_stack.tar" agent/arch-inf-20260803 stack/tanitad stack/scripts stack/tests taniteval/taniteval taniteval/tools
  (cd "$OUT" && tar -xf ../tip_stack.tar)
  git show agent/arch-inf-20260803:stack/pyproject.toml > "$OUT/stack/pyproject.toml"
  touch "$OUT/.tip_ok"
fi
# overlay (copy, so a stale overlay never survives a removed fix file: re-run with a fresh OUT to be sure)
(cd "$PK/code/fix" && find . -type f -print0 | while IFS= read -r -d '' f; do mkdir -p "$OUT/$(dirname "$f")"; cp -f "$f" "$OUT/$f"; done)
echo "PYTHONPATH=$OUT/stack;$OUT/stack/scripts;$OUT/taniteval"
