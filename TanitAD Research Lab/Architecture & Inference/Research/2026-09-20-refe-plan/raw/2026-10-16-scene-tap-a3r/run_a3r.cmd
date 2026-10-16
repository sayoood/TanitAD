@echo off
rem A3R (PREREG_SCENE_TAP_PROBE.md, committed at c4653a6 before any 400-update unit): arm V, seed 0, 5 folds, 400 updates, then the reader.
cd /d "C:\Users\Admin\AppData\Local\Temp\stp\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\raw\2026-10-16-scene-tap-a3r"
echo %date% %time% start >> logs\a3r_chain.log
"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe" -B -I scene_tap_probe.py run --out "C:\Users\Admin\AppData\Local\Temp\stp\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\raw\2026-10-16-scene-tap-a3r" --work D:/Projects/TanitAD-artifacts/scene_tap_probe_a3r --updates 400 --folds-sha256 20dec3431663de2201190020f97c74f7975468a8e9e178f3db4d52fbacd5885f --arms V --seeds 0 --lease-s 600 >> logs\grid.log 2>&1
if errorlevel 1 ( echo grid failed > A3R_FAILED & exit /b 1 )
"C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe" -B -I a2_read.py --out "C:\Users\Admin\AppData\Local\Temp\stp\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\raw\2026-10-16-scene-tap-a3r" --work-a2 D:/Projects/TanitAD-artifacts/scene_tap_probe_a3r --updates-a2 400 >> logs\a3r_read.log 2>&1
if errorlevel 1 ( echo read failed > A3R_FAILED & exit /b 1 )
echo %date% %time% done > A3R_DONE
