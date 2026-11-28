@echo off
setlocal enabledelayedexpansion
call D:\tanitad\env.cmd
echo ===== DIAG egoprobe %DATE% %TIME%
%RUNSAFE% "%PY%" D:\tanitad\jobs\diag_egoprobe.py D:\tanitad\jobs\diag_gate_line.txt D:\tanitad\runs\v8f_mini_full\diag_egoprobe_s0 D:\tanitad\jobs\diag_egoprobe_s0.json
set RC=!ERRORLEVEL!
echo ===== ZZJOB-END %DATE% %TIME% rc=!RC!
