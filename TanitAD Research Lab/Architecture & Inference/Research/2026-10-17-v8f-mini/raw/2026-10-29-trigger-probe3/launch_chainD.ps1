$cmd = "cmd /c `"`"C:\Program Files\Git\bin\bash.exe`" -c `"bash /e/Projects/TanitAD-artifacts/dagland/trigger_probe3/chainD.sh`" < NUL > E:\Projects\TanitAD-artifacts\dagland\trigger_probe3\raw\chain_stdout_D.log 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
"chain D ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
