# Launch a detached job through WMI Win32_Process.Create (survives the calling session). Usage:
#   powershell -NoProfile -File wmi_launch.ps1 -Script <py file> -Log <log file> [-ScriptArgs "<args>"]
param([Parameter(Mandatory = $true)][string]$Script, [Parameter(Mandatory = $true)][string]$Log, [string]$ScriptArgs = "")
$py = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$wd = "E:\Projects\TanitAD-artifacts\dagland\motion_c\t2_scenes"
$cmd = "cmd /c `"`"$py`" -I `"$Script`" $ScriptArgs > `"$Log`" 2>&1 < NUL`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $wd }
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId) launched_utc=$([DateTime]::UtcNow.ToString('o'))"
