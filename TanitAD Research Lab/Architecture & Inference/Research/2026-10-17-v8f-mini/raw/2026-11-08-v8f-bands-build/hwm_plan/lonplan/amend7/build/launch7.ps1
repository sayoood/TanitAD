param([string]$Mode = "fit")
$d = 'F:\TanitAD-scratch\actrsp\lonplan\amend7\build'
$s = ([wmiclass]"Win32_ProcessStartup").CreateInstance()
$s.CreateFlags = 0x08000200
$r = ([wmiclass]"Win32_Process").Create("cmd.exe /c $d\amend7.cmd $Mode < NUL", $d, $s)
"pid=$($r.ProcessId) rv=$($r.ReturnValue)"
