#!/usr/bin/env bash
# pycap.sh : stands in for the venv python in the dev-box read scripts of this fork (coordination decision D6a).
# A script job or a "-m module" job runs under ops/capmem.py: hard per-process commit limit PYCAP_MB (default 1000),
# one capped job at a time (named mutex), started only with limit + 1500 MB of commit memory free.
# The inline forms "-" (stdin) and "-c" run plain: they are the small json prints of the read scripts.
PYX=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
export PYTHONDONTWRITEBYTECODE=1
case "$1" in
  -|-c) exec "$PYX" "$@" ;;
  -m)   exec "$PYX" F:/TanitAD-scratch/v8proof/ops/capmem.py "${PYCAP_MB:-1000}" "$@" ;;
  -*)   echo "ZZ PYCAP STOP: option $1 is not handled" >&2; exit 94 ;;
  *)    exec "$PYX" F:/TanitAD-scratch/v8proof/ops/capmem.py "${PYCAP_MB:-1000}" "$@" ;;
esac
