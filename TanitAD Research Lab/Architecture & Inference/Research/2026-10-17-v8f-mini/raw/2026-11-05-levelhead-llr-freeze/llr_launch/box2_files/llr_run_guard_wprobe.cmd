@echo off
rem LEVELHEAD LLR: the box-2 temperature guard for the W_lead probe (ends on WLPROBE.done / FAILED / STOPPED). Same rule as rnext_guard.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\llr\llr_guard_wprobe.ps1
