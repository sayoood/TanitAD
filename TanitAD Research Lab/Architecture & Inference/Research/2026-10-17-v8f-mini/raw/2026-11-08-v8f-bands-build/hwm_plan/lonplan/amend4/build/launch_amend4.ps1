param([string]$Job = "amend4.cmd")
$d = 'F:\TanitAD-scratch\actrsp\lonplan\amend4\build'
$s = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$s.CreateFlags = 0x08000200
$r = ([wmiclass]"Win32_Process").Create("cmd.exe /c $d\$Job < NUL", $d, $s)
"pid=$($r.ProcessId) rv=$($r.ReturnValue)"
