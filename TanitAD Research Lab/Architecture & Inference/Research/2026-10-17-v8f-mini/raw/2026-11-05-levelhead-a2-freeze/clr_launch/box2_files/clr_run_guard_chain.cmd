@echo off
rem LEVELHEAD CLR: the box-2 temperature guard for the chain (ends on CHAIN_CLR.done / STOPPED only: a stale marker of another job cannot end it). Same rule as rnext_guard.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\clr\clr_guard_chain.ps1
