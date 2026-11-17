$d = 'F:\TanitAD-scratch\datascale\diag'
New-Item -ItemType Directory -Force "$d\out" | Out-Null
$s = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$s.CreateFlags = 0x08000200
$r = ([wmiclass]"Win32_Process").Create("cmd.exe /c $d\job_diag.cmd < NUL", $d, $s)
"pid=$($r.ProcessId) rv=$($r.ReturnValue)"
