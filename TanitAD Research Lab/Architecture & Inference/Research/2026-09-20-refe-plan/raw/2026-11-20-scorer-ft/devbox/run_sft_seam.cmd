@echo off
rem SCORER-FT arm navtest pre-work launcher. Start DETACHED (WMI Win32_Process.Create, CreateFlags 0x08000200), as run_field_seam.cmd.
rem %1 = arm (SFTC, SFTT, SFTP, SFTCs, SFTTs, SFTPs), %2 = HF prefix (scorerft/c ...), %3 = seam name (scorerft_c ...). Git\usr\bin\bash.exe, stdin from NUL.
set "PATH=C:\Program Files\Git\usr\bin;C:\Program Files\Git\bin;%PATH%"
if not exist "E:\Projects\TanitAD-artifacts\scorerft_read" mkdir "E:\Projects\TanitAD-artifacts\scorerft_read"
"C:\Program Files\Git\usr\bin\bash.exe" "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/scorerft_read/sft_seam_chain.sh" %1 %2 %3 > "E:\Projects\TanitAD-artifacts\scorerft_read\chain_stdout_%1.log" 2>&1 < NUL
echo chain exit %ERRORLEVEL% >> "E:\Projects\TanitAD-artifacts\scorerft_read\chain_stdout_%1.log"
exit /b %ERRORLEVEL%
