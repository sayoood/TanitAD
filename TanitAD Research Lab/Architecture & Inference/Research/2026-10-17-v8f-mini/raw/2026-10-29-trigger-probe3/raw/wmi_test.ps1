$cmd = 'cmd /c ""C:\Program Files\Git\bin\bash.exe" -c "echo hello from wmi; date; pwd" < NUL > E:\Projects\TanitAD-artifacts\dagland\trigger_probe3\raw\wmi_test.out 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
