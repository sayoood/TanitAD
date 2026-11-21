# dev-box detached launcher (WMI Create survives the calling shell): powershell -NoProfile -File launch_dev.ps1 -Cmd "<cmdline>" -Log <log>
param([Parameter(Mandatory=$true)][string]$Cmd, [Parameter(Mandatory=$true)][string]$Log, [string]$Cwd = "E:\Projects\TanitAD-artifacts\dagland\trigger_probe5")
$line = "cmd.exe /c `"$Cmd >> `"$Log`" 2>&1 < NUL`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $line; CurrentDirectory = $Cwd }
"launched pid=$($r.ProcessId) rc=$($r.ReturnValue)"
