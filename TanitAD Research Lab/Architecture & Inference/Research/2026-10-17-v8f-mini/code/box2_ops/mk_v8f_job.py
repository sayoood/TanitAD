"""mk_v8f_job.py <jobname> <queue args...>: a box 2 job that runs v8f_mini_box2_queue.cmd under SAFE MODE (PREREG_V8F_MINI Amendment 4).
The whole queue call runs inside one %RUNSAFE% (start /b /wait /belownormal /affinity FF), so every python it spawns inherits the
below-normal priority class and the 0xFF affinity; env.cmd supplies the AVX2 / 4-thread / TEMP settings. Same shape as mk_ra1_job.py.
Every backslash is chr(92): inline escapes have mangled these paths more than once."""
import sys

BS = chr(92)
name, args = sys.argv[1], " ".join(sys.argv[2:])
code = BS.join(["%REPO%", "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "code"])
lines = ["@echo off", "setlocal enabledelayedexpansion", "call " + BS.join(["D:", "tanitad", "env.cmd"]),
         'set "REPO=' + BS.join(["D:", "tanitad", "repo"]) + '"',
         "echo ===== V8FMINI " + args + " %DATE% %TIME%",
         'cd /d "' + code + '"',
         "%RUNSAFE% cmd /c v8f_mini_box2_queue.cmd " + args,
         "set RC=!ERRORLEVEL!",
         "echo ===== ZZJOB-END %DATE% %TIME% rc=!RC!"]
open("C:/Users/Admin/box2_ship/" + name + ".cmd", "wb").write(("\r\n".join(lines) + "\r\n").encode())
print("wrote", name)
