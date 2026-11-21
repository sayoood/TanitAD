@echo off
rem T2 re-perception rollout, v8F-mini, STAGE 3 (the 20 guard scenes) on box 2. ONE job, safe mode. Launch: powershell -NoProfile -File D:\tanitad\launch.ps1 -Job D:\t2v8f\pkg\jobs\stage3.cmd -Log D:\t2v8f\logs\stage3.log
rem No labels and no goto on purpose: this file is LF-only, and cmd.exe can mis-read labels in an LF-only batch file.
call D:\tanitad\env.cmd
set "PYTHONPATH="
set "VCV=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCV%" (call "%VCV%" >nul) else (echo ZZNO-VCVARS)
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "TORCH_CUDA_ARCH_LIST=8.9"
set "T2A_KIT=D:/t2a/box2_kit"
if not exist D:\t2v8f\pull3 mkdir D:\t2v8f\pull3
echo ===== STAGE3 START %DATE% %TIME%
%RUNSAFE% %PY% -I -u D:\t2v8f\pkg\code\t2v8f_stage3.py all --pkg D:/t2v8f/pkg --kit D:/t2a/box2_kit --hf D:/tanitad/hf --pull D:/t2v8f/pull3 --scenes3 C:/t2v8f/scenes3 --manifest3 D:/t2v8f/pkg/MANIFEST3.json
echo ===== ZZJOB-END %DATE% %TIME% rc=%ERRORLEVEL%
