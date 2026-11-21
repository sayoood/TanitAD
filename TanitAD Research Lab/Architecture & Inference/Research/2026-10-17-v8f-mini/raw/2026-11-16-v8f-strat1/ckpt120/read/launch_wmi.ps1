param([string]$Job = "job_ckpt120_read.cmd")
$d = 'F:\TanitAD-scratch\strat1\read\ckpt120'
New-Item -ItemType Directory -Force "$d\out" | Out-Null
$s = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$s.CreateFlags = 0x08000200
$r = ([wmiclass]"Win32_Process").Create("cmd.exe /c $d\$Job < NUL", $d, $s)
"pid=$($r.ProcessId) rv=$($r.ReturnValue)"
