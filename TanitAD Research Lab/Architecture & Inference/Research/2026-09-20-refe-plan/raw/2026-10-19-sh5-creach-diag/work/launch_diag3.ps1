$py = 'C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe'
$w = 'C:\Users\Admin\par18\sh5diag\work'
$c = 'C:\Users\Admin\par18\sh5diag\code\diag_pre3.py'
$g3 = "$py -B -W ignore $c --grid $w\grid3.json --out $w\diag3 --max-wait 21600 > $w\diag3.log 2>&1 < NUL"
$g4 = "$py -B -W ignore $c --grid $w\grid4.json --out $w\diag4 --skip-if-pass $w\diag3\diag_summary.json --best-from $w\diag3\diag_summary.json,$w\grid3.json --max-wait 21600 > $w\diag4.log 2>&1 < NUL"
$cmd = "cmd /c `"$g3 && $g4`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory='C:\Users\Admin\par18\sh5diag\code'}
"ReturnValue=$($r.ReturnValue) PID=$($r.ProcessId)"
