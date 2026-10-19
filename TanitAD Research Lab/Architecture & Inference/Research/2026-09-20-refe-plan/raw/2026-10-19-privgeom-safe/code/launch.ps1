# usage: powershell -File launch.ps1 -Cmd "<cmd.exe command line after /c>" -Log <logfile> [-Cwd <dir>]
param([string]$Cmd, [string]$Log, [string]$Cwd = "C:\Users\Admin\par18\privgeom")
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
$line = 'cmd /c "' + $Cmd + ' < NUL >> ' + $Log + ' 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$line; CurrentDirectory=$Cwd; ProcessStartupInformation=$si}
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
