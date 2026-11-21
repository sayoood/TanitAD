@echo off
rem Detached launcher for lit1_chain.sh, the dev-box CPU T1-kin chain of PREREG_V8F_LAUNCH_INTENT (start it through WMI Win32_Process.Create so it survives the caller; arguments pass through, e.g. --dry-run).
rem   powershell -NoProfile -Command "([wmiclass]'Win32_Process').Create('cmd /c E:\Projects\TanitAD-artifacts\launch_intent_t1\lit1_chain.cmd') | Select-Object ReturnValue, ProcessId"
rem RESUME = launch it again (a finished step is skipped by its done record; an incomplete earlier attempt is MOVED to <OUT>/_incomplete_*, never deleted).
rem Output: logs\chain.out (the chain's own log, anchored ZZLIT1-* markers; rotated at every start) and logs\lit1_chain.launch.out (the launcher's stdout).  Wait for it with: bash wake_lit1.sh
rem Optional environment goes HERE (a WMI-created process does not inherit the caller shell).  NONE is set as shipped: every default below is the production value.
rem   set LIT1_B2_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_launchintent.out     (box 2's chain log = the default; the log of a predecessor chain is REFUSED)
rem   set LIT1_ART=E:/Projects/TanitAD-artifacts/v8f-launch-intent                (must equal the directory box 2's chain pulls into)
rem   set LIT1_OUT=F:/TanitAD-scratch/launch_intent_t1_out                         (rollout outputs; the default)
rem   set LIT1_READER=E:/Projects/TanitAD-artifacts/dagland/launch_intent_v2/read/launch_intent_read.py   (the LANDED reader = the default, sha256 5e950dcf... pinned; its sha256 must equal the reader_sha256 every token binds)
rem   set LIT1_PREREG=E:/Projects/TanitAD-artifacts/launch_intent_t1/landed/PREREG_V8F_LAUNCH_INTENT.md   (the LANDED PREREG at 77234276 = the default, sha256 469bac1f... pinned; it must equal the prereg_sha256 every token binds)
rem   set LIT1_EXPECT_COMMIT=b09ac89fc9d6e81baf3977954ae638919b979e71              (every token must bind exactly this commit; unset = no pin)
rem   set LIT1_MAX_WAIT_H=12   set LIT1_RAM_MIN_MB=4096   set LIT1_F_MIN_GB=5   set LIT1_POLL_S=60
if not exist "E:\Projects\TanitAD-artifacts\launch_intent_t1\logs" mkdir "E:\Projects\TanitAD-artifacts\launch_intent_t1\logs"
"C:\Program Files\Git\bin\bash.exe" -lc "bash /e/Projects/TanitAD-artifacts/launch_intent_t1/lit1_chain.sh %* < /dev/null >> /e/Projects/TanitAD-artifacts/launch_intent_t1/logs/lit1_chain.launch.out 2>&1" < NUL
