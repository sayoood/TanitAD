@echo off
rem LEVELHEAD LLR: the box-2 temperature guard for the chain (ends on CHAIN_LLR.done / STOPPED only: a stale marker of another job cannot end it). Same rule as rnext_guard.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\llr\llr_guard_chain.ps1
