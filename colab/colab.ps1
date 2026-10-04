# Run the official Colab CLI (google-colab-cli 0.6.0, venv C:\Users\Admin\venvs\colab) on this Windows dev box.
#   & D:\Projects\TanitAD\colab\colab.ps1 new -s job1 --gpu L4
#   & D:\Projects\TanitAD\colab\colab.ps1 exec -s job1 --timeout 600 -f D:\Projects\TanitAD\colab\gpu_smoke.py
#   & D:\Projects\TanitAD\colab\colab.ps1 stop -s job1
# Why a wrapper (RUNNER.md section 9, all MEASURED 2026-08-19): the CLI needs the termios import shim on Windows
# (COLAB_CLI_MCP.md 3.1) -- and PYTHONPATH, not an in-process stub, because `colab new` spawns its keep-alive daemon
# as a fresh `python -m colab_cli.cli` that inherits only the environment; PYTHONUTF8=1 because `exec -f` reads the
# script with the locale codec (cp1252 here). Call it from PowerShell, never from Git Bash: MSYS rewrites remote
# POSIX paths (`/content` -> `C:/Program Files/Git/content`).
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $PSScriptRoot "win_shims"
& "C:\Users\Admin\venvs\colab\Scripts\colab.exe" @args
exit $LASTEXITCODE
