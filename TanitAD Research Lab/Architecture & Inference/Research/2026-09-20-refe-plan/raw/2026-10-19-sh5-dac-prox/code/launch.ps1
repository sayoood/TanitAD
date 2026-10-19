# usage: powershell -File launch.ps1 -Cmd "<cmd.exe command line after /c>" -Log <logfile> [-Cwd <dir>]
# Detached WMI launch (CLAUDE memory: a long dev-box job launched any other way dies or loses stdin).  The job's own GPU lock is taken INSIDE the job (sh5_gpulock); this script touches no lock.
param([string]$Cmd, [string]$Log, [string]$Cwd = "C:\Users\Admin\par18\sh5\code")
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
$line = 'cmd /c "' + $Cmd + ' < NUL >> ' + $Log + ' 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$line; CurrentDirectory=$Cwd; ProcessStartupInformation=$si}
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
