@echo off
set CUDA_VISIBLE_DEVICES=-1
set PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
set CODE=D:\epadm2\w\raw\2026-10-19-ep-admission-r1\code
set RUN=D:\epadm2\run
echo %date% %time% precompute start >> %RUN%\chain.log
%PY% -I %CODE%\ep_admission_sim.py precompute --arrays-npz C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz --landed-result C:/Users/Admin/par18/epreliab/run/results/result.json --workdir %RUN% > %RUN%\precompute.log 2>&1
echo %date% %time% precompute rc=%errorlevel% >> %RUN%\chain.log
if errorlevel 1 goto fail
%PY% -I D:\epadm2\cmp_stats.py %RUN%\stats.npz D:\epadm2\banked\stats.npz %RUN%\stats_vs_banked.json > %RUN%\stats_vs_banked.log 2>&1
echo %date% %time% stats_vs_banked rc=%errorlevel% >> %RUN%\chain.log
if errorlevel 1 goto fail
%PY% -I %CODE%\ep_admission_sim.py repro --workdir %RUN% --rep-offset 0 --against D:\epadm2\banked\replicates.npz --out repro_offset0_vs_banked.json > %RUN%\repro_offset0_vs_banked.log 2>&1
echo %date% %time% repro_offset0_vs_banked rc=%errorlevel% >> %RUN%\chain.log
if errorlevel 1 goto fail
%PY% -I %CODE%\ep_admission_sim.py run --workdir %RUN% --n-reps 1000 --workers 12 --rep-offset 1000000 > %RUN%\run.log 2>&1
echo %date% %time% run rc=%errorlevel% >> %RUN%\chain.log
%PY% -I %CODE%\ep_admission_sim.py repro --workdir %RUN% --rep-offset 1000000 --out repro.json > %RUN%\repro.log 2>&1
echo %date% %time% repro_fresh rc=%errorlevel% >> %RUN%\chain.log
echo CHAIN_DONE > %RUN%\chain.done
goto end
:fail
echo CHAIN_FAILED > %RUN%\chain.done
:end
