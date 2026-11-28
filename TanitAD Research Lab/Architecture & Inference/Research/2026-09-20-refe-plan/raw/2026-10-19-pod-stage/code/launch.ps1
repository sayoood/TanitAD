# usage: powershell -File launch.ps1 [-Cmd "<cmd.exe command line after /c>"] [-Log <logfile>]
# Detached WMI launch (dev-box memory: a long job launched any other way dies or loses stdin).  The job is the resumable pod-stage chain (frames -> verify -> SH5 plan FULL -> 76 DBs -> verify -> package).
param([string]$Cmd = "C:\Python314\python.exe -B C:\Users\Admin\par18\podstage\code\podstage.py all", [string]$Log = "C:\Users\Admin\par18\podstage\logs\podstage.log", [string]$Cwd = "C:\Users\Admin\par18\podstage\code")
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
$line = 'cmd.exe /c "' + $Cmd + ' < NUL >> ' + $Log + ' 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$line; CurrentDirectory=$Cwd; ProcessStartupInformation=$si}
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
