param([string]$Job = "job_dumps_s1.cmd")
$d = 'E:\Projects\TanitAD-artifacts\dagland\motion_c\theory\stopperc\stage1_A\read'
$s = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$s.CreateFlags = 0x08000200
$r = ([wmiclass]"Win32_Process").Create("cmd.exe /c $d\$Job < NUL", $d, $s)
"pid=$($r.ProcessId) rv=$($r.ReturnValue)"
