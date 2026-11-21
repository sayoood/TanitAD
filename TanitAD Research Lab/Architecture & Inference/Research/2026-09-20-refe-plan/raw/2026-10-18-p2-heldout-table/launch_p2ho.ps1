# Launch (or RESUME) the P2 held-out table driver DETACHED, so it survives the launching session.  Same command = resume: done chunks and done score units are skipped.
# Derived from landed/launch_p2full.ps1 (same WMI Win32_Process.Create, CreateFlags 0x08000200 = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW), plus `< NUL` on the cmd line.
#   powershell -NoProfile -File launch_p2ho.ps1                       # the main job (GPU behind the shared dev-box lock, 2 scoring workers)
#   powershell -NoProfile -File launch_p2ho.ps1 -Workers 3            # resume with a different scoring pool
# No -ExecutionPolicy Bypass (never).
param(
    [string]$Tag = "main",
    [int]$Workers = 2,
    [string]$Device = "cuda",
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
$pkg = $PSScriptRoot
$log = if ($Tag -eq "main") { Join-Path $pkg "driver_main.log" } else { Join-Path $pkg ("driver_" + $Tag + ".log") }
$cmdline = 'cmd.exe /c ""' + $py + '" "' + (Join-Path (Join-Path $pkg "code") "p2ho_driver.py") + '" --tag ' + $Tag + ' --workers ' + $Workers + ' --device ' + $Device + ' ' + $Extra + ' < NUL >> "' + $log + '" 2>&1"'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdline; CurrentDirectory = (Join-Path $pkg "code"); ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"launched cmd.exe pid $($r.ProcessId) ; log $log"
