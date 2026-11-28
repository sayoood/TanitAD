@echo off
rem RECOVER-FT: the box-2 temperature guard (same rule as v8f_guard.sh: >= 4 of the last 20 telemetry samples >= 61 C on the USB HDD, or one >= 63 C => STOP). Exits when the chain ends.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\recft\recft_guard.ps1
