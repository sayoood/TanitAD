@echo off
rem Detached launcher for dagorct1_chain.sh (start it through WMI Win32_Process.Create so it survives the caller; arguments pass through, e.g. --dry-run).
rem   powershell -NoProfile -Command "([wmiclass]'Win32_Process').Create('cmd /c E:\Projects\TanitAD-artifacts\dagorc_t1\dagorct1_chain.cmd') | Select-Object ReturnValue, ProcessId"
rem RESUME = launch it again (finished steps are skipped by their done records).  Output: logs\chain.out (the chain's own log, anchored markers) and logs\dagorct1_chain.launch.out (launcher stdout).
rem Optional environment goes HERE (a WMI-created process does not inherit the caller shell), e.g.:
rem   set DORT1_OUT=F:/TanitAD-scratch/dagorc_t1_out         (if E: is too full: E: had ~2.6 GB free for ~1.7 GB of new data plus the box-2 pulls' 1 GiB floor)
rem   set DORT1_ART=E:/Projects/TanitAD-artifacts/v8f-dag-orc-oacc   (must equal box 2's DAGOACCORC_ART)   set DORT1_MAX_WAIT_H=12
rem
rem ---- THE r2 LAUNCH (A2, the fix commit 8c24823f): the MASTER MIND sets these four, on the lines below, BEFORE launching.  They are NOT set in this file as shipped (an unset DORT1_FIX_PKG = the r1 K-BUILD mode,
rem      which reads UNVERIFIED on the r2 bytes and makes the reader refuse A; an unset DORT1_EXPECT_COMMIT = no pin on the commit the four launch-gate tokens bind).
rem      To use one, delete its leading "rem   " (keep the quotes on DORT1_FIX_PKG: its path holds an ampersand, which cmd would otherwise treat as a command separator):
set "DORT1_FIX_PKG=TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-28-dag-orc-a2/fix"
set DORT1_EXPECT_COMMIT=8c24823f2b1bd24332148ebe434c870bcf3662f8
set DORT1_EVIDENCE_COMMIT=8c24823f2b1bd24332148ebe434c870bcf3662f8
set DORT1_OUT=F:/TanitAD-scratch/dagorc_t1_out
rem      what they do: DORT1_FIX_PKG selects the K-BUILD two-step mode (dagorct1_lib.py kbuild --fix-pkg): the golden-writer bytes come from the r1 compose package (default path, read at e7ea20d1), the re-run suite,
rem      mutation proof and tested-file sha256 from the fix package read at DORT1_EVIDENCE_COMMIT; DORT1_EXPECT_COMMIT makes the chain STOP if any token binds another commit.  Optional: DORT1_COMPOSE_PKG (the r1
rem      package path; the default is right) and DORT1_R1_EVIDENCE_COMMIT (default the r1 commit e7ea20d1).  Proof of the K-BUILD: kbuild_r2_launch_8c24823f.json (every changed hunk PASS) and selftest/SELFTEST.out.
if not exist "E:\Projects\TanitAD-artifacts\dagorc_t1\logs" mkdir "E:\Projects\TanitAD-artifacts\dagorc_t1\logs"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/dagorc_t1/dagorct1_chain.sh %* < /dev/null >> /e/Projects/TanitAD-artifacts/dagorc_t1/logs/dagorct1_chain.launch.out 2>&1" < NUL
