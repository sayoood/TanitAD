@echo off
rem LEVELHEAD CL: the box-2 temperature guard for the W probe (ends on WPROBE.done / FAILED / STOPPED). Same rule as rnext_guard.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\cl\cl_guard_wprobe.ps1
