@echo off
rem MM 2026-11-25: auto_sft_seam.sh for arm P, started DETACHED (WMI, CreateFlags 0x08000200) so it survives the MM session; stdin from NUL.
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorerft_read/auto_sft_seam.sh" SFTP scorerft/p scorerft_p >> "E:\Projects\TanitAD-artifacts\scorerft_read\auto_seam_SFTP.log" 2>&1 < NUL
echo autoseam exit %ERRORLEVEL% >> "E:\Projects\TanitAD-artifacts\scorerft_read\auto_seam_SFTP.log"
exit /b %ERRORLEVEL%
