"""mk_full_build.py: the box 2 job files for PREREG_RA1_FULL's build (ra1_full_box2.py) and its long telemetry.
The build runs inside ONE %RUNSAFE% (start /b /wait /belownormal /affinity FF), so every builder / encoder / verifier it spawns
inherits the below-normal priority class and the 0xFF affinity; env.cmd supplies AVX2 / threads / TEMP / LABELFLAGS.
Every backslash is chr(92): inline escapes have mangled these paths more than once."""
BS = chr(92)
code = BS.join(["%REPO%", "TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-refav1-hj-mini", "code"])
build = ["@echo off", "setlocal enabledelayedexpansion", "call " + BS.join(["D:", "tanitad", "env.cmd"]),
         'set "REPO=' + BS.join(["D:", "tanitad", "repo"]) + '"',
         "echo ===== RA1FULL build %DATE% %TIME%",
         'cd /d "' + code + '"',
         "%RUNSAFE% %PY% -u ra1_full_box2.py --chunk 400 --workers 4",
         "set RC=!ERRORLEVEL!",
         "echo ===== ZZJOB-END %DATE% %TIME% rc=!RC!"]
tele = ["@echo off", "powershell -NoProfile -ExecutionPolicy Bypass -File " + BS.join(["D:", "tanitad", "telemetry_long.ps1"])]
for name, lines in (("ra1full_build", build), ("telemetry_long", tele)):
    open("C:/Users/Admin/box2_ship/" + name + ".cmd", "wb").write(("\r\n".join(lines) + "\r\n").encode())
    print("wrote", name)
