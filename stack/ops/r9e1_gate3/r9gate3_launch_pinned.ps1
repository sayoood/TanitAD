param([Parameter(Mandatory)][string]$Sha, [string]$Arms = "C0 C1")
# launches r9gate3_devbox_pinned.sh DETACHED through WMI (CreateFlags 0x08000200: new process group, no window) so it survives the session
$c12 = $Sha.Substring(0, 12)
$cmd = "F:\refcv9_e1\gate\pinned_run_$c12.cmd"
# the driver next to THIS script (was a hard-coded qland/hold_r9_gate3 path)
$here = '/' + $PSScriptRoot.Substring(0, 1).ToLower() + ($PSScriptRoot.Substring(2) -replace '\\', '/')
$line = '"C:\Program Files\Git\bin\bash.exe" -c "bash ' + $here + '/r9gate3_devbox_pinned.sh ' + $Sha + ' ''' + $Arms + '''" < NUL > F:\refcv9_e1\gate\pinned_wmi_' + $c12 + '.log 2>&1'
[System.IO.File]::WriteAllText($cmd, "@echo off`r`n$line`r`n", (New-Object System.Text.UTF8Encoding($false)))
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c $cmd"; CurrentDirectory = 'F:\refcv9_e1\gate'; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { exit 1 }
"pinned driver pid $($r.ProcessId)"
