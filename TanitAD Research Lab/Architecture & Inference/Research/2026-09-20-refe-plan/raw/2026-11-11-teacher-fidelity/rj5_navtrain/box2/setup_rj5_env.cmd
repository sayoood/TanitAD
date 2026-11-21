@echo off
setlocal enabledelayedexpansion
rem RJ5 navtrain: stand up the teacher env INSIDE D:\tanitad\rj5 (no system install). Idempotent; markers in D:\tanitad\rj5\markers.
set "R=D:\tanitad\rj5"
set "T=D:\tanitad\tmp"
set "PY311=C:\Users\User\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
if not exist "%R%\markers" mkdir "%R%\markers"
if not exist "%R%\dz" (
  mkdir "%R%\dz" & tar -xf "%T%\rj5_dz.tar" -C "%R%\dz"
  if !ERRORLEVEL! NEQ 0 ( echo ZZFAIL untar dz & exit /b 1 )
  del "%T%j5_dz.tar"
)
if not exist "%R%\navsim" (
  mkdir "%R%\navsim" & tar -xf "%T%\rj5_navsim_dk.tar" -C "%R%\navsim"
  if !ERRORLEVEL! NEQ 0 ( echo ZZFAIL untar navsim & exit /b 1 )
  del "%T%j5_navsim_dk.tar"
)
if not exist "%R%\w" (
  mkdir "%R%\w" & tar -xf "%T%\rj5_w.tar" -C "%R%\w"
  if !ERRORLEVEL! NEQ 0 ( echo ZZFAIL untar w & exit /b 1 )
  del "%T%j5_w.tar"
)
if not exist "%R%\maps" (
  mkdir "%R%\maps" & tar -xf "%T%\rj5_maps.tar" -C "%R%\maps"
  if !ERRORLEVEL! NEQ 0 ( echo ZZFAIL untar maps & exit /b 1 )
  del "%T%j5_maps.tar"
)
if not exist "%R%\venv\Scripts\python.exe" (
  "%PY311%" -m venv "%R%\venv"
  if !ERRORLEVEL! NEQ 0 ( echo ZZFAIL venv & exit /b 1 )
)
if not exist "%R%\markers\torch.ok" (
  "%R%\venv\Scripts\python.exe" -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
  if !ERRORLEVEL! NEQ 0 ( echo ZZFAIL torch & exit /b 1 )
  echo ok> "%R%\markers\torch.ok"
)
if not exist "%R%\markers\req.ok" (
  "%R%\venv\Scripts\python.exe" -m pip install --no-cache-dir -r "%T%\rj5_req.txt"
  if !ERRORLEVEL! NEQ 0 ( echo ZZFAIL req & exit /b 1 )
  echo ok> "%R%\markers\req.ok"
)
echo ZZSETUP_DONE
exit /b 0
