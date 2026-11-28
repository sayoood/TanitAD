#!/bin/bash
# usage: run_suites.sh <refe dir> <log prefix> <tmp dir> <names...>
REFE="$1"; PFX="$2"; export TMP="$3" TEMP="$3"; shift 3
mkdir -p "$TMP"
PY=C:/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$REFE" || exit 9
for n in "$@"; do
  case "$n" in
    lg) (time $PY test_launch_gate.py --static --no-banked) > "${PFX}test_launch_gate_static.log" 2>&1 ;;
    *) (time $PY "test_$n.py") > "${PFX}test_$n.log" 2>&1 ;;
  esac
  echo "$n rc=$?" >> "${PFX}_done.txt"
done
