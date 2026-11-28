"""mk_job.py <name> <align|crops>: writes the box-2 .cmd job (SAFE MODE, resume-retry up to 3 attempts, ZZJOB-END rc=).  Every backslash is chr(92)."""
import sys

BS = chr(92)
name, mode = sys.argv[1], sys.argv[2]
J = BS.join(["D:", "tanitad", "jobs", "probe5"])
py = BS.join(["D:", "tanitad", "venv", "Scripts", "python.exe"])
lines = ["@echo off", "setlocal enabledelayedexpansion", "call " + BS.join(["D:", "tanitad", "env.cmd"]),
         "echo ===== P5 " + mode + " %DATE% %TIME%", "set TRY=0", ":again", "set /a TRY+=1",
         "%RUNSAFE% " + py + " -u " + J + BS + "encode_crops.py --mode " + mode,
         "set RC=!ERRORLEVEL!", "echo ===== ATTEMPT !TRY! rc=!RC! %TIME%",
         "if !RC! NEQ 0 if !TRY! LSS 3 (ping -n 31 127.0.0.1 >nul & goto again)",
         "echo ===== ZZJOB-END %DATE% %TIME% rc=!RC!"]
open("E:/Projects/TanitAD-artifacts/dagland/trigger_probe5/" + name + ".cmd", "wb").write(("\r\n".join(lines) + "\r\n").encode())
print("wrote", name)
