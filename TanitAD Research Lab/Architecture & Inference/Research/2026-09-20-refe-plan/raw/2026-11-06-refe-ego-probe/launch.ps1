# Detached launch of the full ego probe via WMI (survives the agent session; stdin = NUL).  Usage:  powershell -NoProfile -File launch.ps1 <tokens.json> <out_dir> [extra args]
param([string]$Tokens, [string]$Out, [string]$Extra = "")
$Py = "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
$Dir = "E:\Projects\TanitAD-artifacts\refe_ego_probe_20261106"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Log = Join-Path $Out "stdout.log"
$cmd = "cmd.exe /c `"cd /d $Dir && $Py refe_ego_probe.py --out $Out --tokens $Tokens --wait-idle-timeout 3600 $Extra < NUL > $Log 2>&1`""
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $Dir; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) ProcessId(cmd)=$($r.ProcessId)"
