@echo off
rem LEVELHEAD CL: the box-2 temperature guard for the chain (ends on CHAIN_CL.done / STOPPED only: a stale WPROBE marker cannot end it). Same rule as rnext_guard.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File D:\tanitad\cl\cl_guard_chain.ps1
