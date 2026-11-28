@echo off
rem sha256 of every file of the shipped pod eval cache, as 'HASH relpath' lines (for the dev box to compare), then ZZJOB-END
powershell -NoProfile -Command "$r='C:\lorapod_eval\cache_pod\eval\'; Get-ChildItem -LiteralPath $r -Recurse -File | ForEach-Object { $_.FullName.Substring($r.Length).Replace('\','/') + ' ' + (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLower() } | Set-Content -Encoding ascii C:\lorapod_eval\logs\cache_hash.txt"
echo ===== ZZJOB-END %DATE% %TIME% rc=%ERRORLEVEL%
