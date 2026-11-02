@echo off
rem RECOVER-FT: the long telemetry (read-only; D:\tanitad\telemetry_long.ps1, one row per 15 s into D:\tanitad\logs\telemetry_full.csv; stops when D:\tanitad\logs\telemetry.stop exists, which the chain writes at its end).
powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\telemetry_long.ps1
