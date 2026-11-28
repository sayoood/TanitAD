param([string]$Job = "job_dscale10k_b1.cmd")
$d = 'F:\TanitAD-scratch\datascale\read'
New-Item -ItemType Directory -Force "$d\out" | Out-Null
$s = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$s.CreateFlags = 0x08000200
$r = ([wmiclass]"Win32_Process").Create("cmd.exe /c $d\$Job < NUL", $d, $s)
"pid=$($r.ProcessId) rv=$($r.ReturnValue)"
