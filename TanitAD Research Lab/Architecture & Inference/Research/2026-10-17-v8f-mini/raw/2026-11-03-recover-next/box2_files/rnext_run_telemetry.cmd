@echo off
rem RECOVER-NEXT: the long telemetry (read-only; D:\tanitad\telemetry_long.ps1, UNCHANGED and reused by path: one row per 15 s into D:\tanitad\logs\telemetry_full.csv; it stops when D:\tanitad\logs\telemetry.stop exists, which the chain writes at its end).
rem A previous chain's telemetry.stop would end the new telemetry at once and leave the guard blind, so it is MOVED ASIDE (never deleted) first.
if exist D:\tanitad\logs\telemetry.stop powershell -NoProfile -Command "Move-Item -LiteralPath 'D:\tanitad\logs\telemetry.stop' -Destination ('D:\tanitad\logs\telemetry.stop.prev_' + (Get-Date -Format yyyyMMdd_HHmmss))"
powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\telemetry_long.ps1
