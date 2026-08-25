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
# ⚠️ A NOTE I ADDED AND RETRACTED WITHIN MINUTES, KEPT AS THE CORRECTION.
# I accused this wrapper's `exec` of FORKING because two python.exe appeared
# for one launch. IT DOES NOT. On this box the venv's python.exe is a LAUNCHER
# STUB that spawns the real interpreter, so ONE JOB ALWAYS SHOWS AS TWO:
#     nohup.exe -> python.exe (4 MB stub) -> python.exe (2.3 GB worker)
# Verified by ParentProcessId and WorkingSetSize. See RETRACTION_LOG C155.
# ⛔ A PROCESS COUNT IS NOT A JOB COUNT. Check the relationship, not the tally.
