$py = 'C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe'
$w = 'C:\Users\Admin\par18\sh5diag\work'
$c = 'C:\Users\Admin\par18\sh5diag\code\diag_pre5.py'
$gL = "$py -B -W ignore $c --grid $w\grid5L.json --out $w\diag5 --max-wait 21600 > $w\diag5.log 2>&1 < NUL"
$gF = "$py -B -W ignore $c --grid $w\grid5F.json --out $w\diag5f --max-wait 21600 > $w\diag5f.log 2>&1 < NUL"
$cmd = "cmd /c `"$gL & $gF`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory='C:\Users\Admin\par18\sh5diag\code'}
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
