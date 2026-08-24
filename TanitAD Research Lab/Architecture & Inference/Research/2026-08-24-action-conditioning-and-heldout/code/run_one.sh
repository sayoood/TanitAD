#!/usr/bin/env bash
# ⛔ SINGLE-INSTANCE LAUNCHER. Four concurrent rangeprobe_rff.py processes ran for
# 28 minutes emitting ZERO rows — I relaunched after each fix and never killed the
# previous run. They competed for BLAS threads AND all defaulted to the same output
# JSON, so they would have overwritten each other's results. Same family as the
# "concurrent torch arms make no progress while looking exactly like a hang" trap.
# ⇒ every relaunch now KILLS its own predecessors first, by command-line match.
set -u
NAME="$1"; shift
PS="/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
for pid in $($PS -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match '$NAME' } | Select-Object -ExpandProperty ProcessId" 2>/dev/null | tr -d '\r'); do
  /c/Windows/System32/taskkill.exe //PID $pid //F >/dev/null 2>&1 && echo "  killed stale $NAME pid $pid"
done
exec /c/Users/Admin/venvs/tanitad/Scripts/python.exe "$NAME" "$@"
