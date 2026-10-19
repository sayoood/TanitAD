param([string]$Grid, [string]$Out, [string]$Log)
$py = 'C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe'
$cmd = "cmd /c `"$py -B -W ignore C:\Users\Admin\par18\sh5diag\code\diag_pre.py --grid $Grid --out $Out --max-wait 21600 > $Log 2>&1 < NUL`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory='C:\Users\Admin\par18\sh5diag\code'}
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
