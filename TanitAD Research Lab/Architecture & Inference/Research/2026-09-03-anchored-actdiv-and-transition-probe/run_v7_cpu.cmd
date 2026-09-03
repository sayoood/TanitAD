@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set OMP_NUM_THREADS=6
set PY=C:\Users\Admin\venvs\tanitad\Scripts\python.exe
set WT=C:\Users\Admin\tanitad-wt
set AS=C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901
set RAW=%WT%\_gs89_pkg\raw
mkdir "%RAW%" 2>nul
cd /d "%WT%"
echo START %DATE% %TIME% > "%RAW%\run_v7_cpu.status"
"%PY%" taniteval\tools\actdiv_anchored.py --assets "%AS%" --arms postrain30k,k8clip05p30k,k60clip05p30k,rdw8p30k --corpus "%AS%\sp2\cache\physicalai-val-w120-256x640cyl" --device cpu --also-replace-all --out "%RAW%\actdiv_anchored_v7_trainerlift.json" > "%RAW%\actdiv_anchored_v7_trainerlift.log" 2>&1
echo actdiv_trainerlift rc=%ERRORLEVEL% %TIME% >> "%RAW%\run_v7_cpu.status"
"%PY%" taniteval\tools\actdiv_anchored.py --assets "%AS%" --arms postrain30k,k8clip05p30k,k60clip05p30k,rdw8p30k --corpus "%AS%\sp2\cache\physicalai-val-w120-256x640cyl" --device cpu --actdiv-compat --out "%RAW%\actdiv_anchored_v7_actdivcompat.json" > "%RAW%\actdiv_anchored_v7_actdivcompat.log" 2>&1
echo actdiv_compat rc=%ERRORLEVEL% %TIME% >> "%RAW%\run_v7_cpu.status"
"%PY%" taniteval\tools\transition_probe.py --assets "%AS%" --arms postrain30k,k8clip05p30k --corpus "%AS%\sp2\cache\physicalai-val130-heldout" --nclips 129 --device cpu --mlp --out "%RAW%\transition_probe_v7.json" > "%RAW%\transition_probe_v7.log" 2>&1
echo transition rc=%ERRORLEVEL% %TIME% >> "%RAW%\run_v7_cpu.status"
echo DONE %DATE% %TIME% >> "%RAW%\run_v7_cpu.status"
