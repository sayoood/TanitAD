# HANDOFF NAVRB2 read (2026-11-18, read-builder agent)

State: 4 arms BANKED (3 copies verified); read BUILT, 34 unit tests + 19 mutations green; dry run on stand-ins done; the REAL read is NOT launched.

## Next (exact), the MM
1. Launch: `powershell -NoProfile -Command "& 'F:\TanitAD-scratch\strat1\navrb2\read\launch_wmi.ps1'"` (WMI, CreateFlags 0x08000200, stdin NUL, Git\usr\bin\bash.exe).
2. Watch `out/navrb2_read.log` for `ZZ NAVRB2 READ DONE ZZ` / `ZZ NAVRB2 READ FAIL ZZ <why>`; sentinel file `out/NAVRB2_READ.done`. Result `out/RESULT_NAVRB2.json` + `.md`.
3. Then TanitAD_BenchmarkCriteria on the result, register rows in GOALS_AND_CLAIMS in the same turn. Details: BUILD_NAVRB2_READ.md.
