param([Parameter(Mandatory)][string]$Sha)
# launches r9gate3_devbox_pinned.sh DETACHED through WMI (CreateFlags 0x08000200: new process group, no window) so it survives the session
$c12 = $Sha.Substring(0, 12)
$cmd = "F:\refcv9_e1\gate\pinned_run_$c12.cmd"
$line = '"C:\Program Files\Git\bin\bash.exe" -c "bash /c/Users/Admin/qland/hold_r9_gate3/code/stack/ops/r9e1_gate3/r9gate3_devbox_pinned.sh ' + $Sha + '" < NUL > F:\refcv9_e1\gate\pinned_wmi_' + $c12 + '.log 2>&1'
[System.IO.File]::WriteAllText($cmd, "@echo off`r`n$line`r`n", (New-Object System.Text.UTF8Encoding($false)))
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c $cmd"; CurrentDirectory = 'F:\refcv9_e1\gate'; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { exit 1 }
"pinned driver pid $($r.ProcessId)"
