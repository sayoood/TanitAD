@echo off
rem SLOWWM planner read: the five basis arms plus the window metadata of R35H s0 / s1 (its planner units are the banked ladder ones).
rem Run once the chain has banked: `cmd /c job_lp_slowwm.cmd < NUL` through WMI (detached), then `report_slowwm.py`.
rem Preconditions: >= 8 GB free RAM, >= 6 GB free on C: (the runner waits on the 6 GB RAM guard, never kills).
call E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\slowwm\read\env_slowwm.cmd
set "PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
set "P=C:/Users/Admin/t2v8f_pulls"
set "BS=C:/Users/Admin/t2v8f_pulls/slowwm_bases"
set "UB=F:/TanitAD-scratch/slowwm/read_cache"
if exist PLAN_SLOWWM.done del PLAN_SLOWWM.done
echo [job] start %DATE% %TIME% >> run_lp_slowwm.log
%PY% -u run_lp_slowwm.py 2 "SLOW_s0|%P%/slowwm_SLOW_s0|%BS%/basis_slow.npz|%UB%/ubasis_slow.npz|all" "PCA-B_s0|%P%/slowwm_PCA-B_s0|%BS%/basis_pca.npz|%UB%/ubasis_pca.npz|all" "SLOW_s1|%P%/slowwm_SLOW_s1|%BS%/basis_slow.npz|%UB%/ubasis_slow.npz|all" "PCA-B_s1|%P%/slowwm_PCA-B_s1|%BS%/basis_pca.npz|%UB%/ubasis_pca.npz|all" "SLOW_rep_s2|%P%/slowwm_SLOW_rep|%BS%/basis_slow.npz|%UB%/ubasis_slow.npz|all" "R35H_s0|%P%/ladder_R35H_s0|||meta" "R35H_s1|%P%/ladder_R35H_s1|||meta" >> run_lp_slowwm.log 2>&1
set "RC=%ERRORLEVEL%"
echo [job] end %DATE% %TIME% rc=%RC% >> run_lp_slowwm.log
(echo rc=%RC% finished=%DATE% %TIME%)> PLAN_SLOWWM.done
