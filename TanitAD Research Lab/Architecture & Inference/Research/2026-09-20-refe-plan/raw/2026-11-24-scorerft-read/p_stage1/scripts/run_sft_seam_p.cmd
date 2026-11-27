@echo off
rem MM 2026-11-26: no-argument wrapper for arm P, so a WMI launch needs no quoting.
call E:\Projects\TanitAD-artifacts\scorer_gym_20261107\scorerft_read\run_sft_seam.cmd SFTP scorerft/p scorerft_p
exit /b %ERRORLEVEL%
