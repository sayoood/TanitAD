# Launch (or RESUME) the P2 full-table driver DETACHED, so it survives the launching session.  Same command = resume: done chunks and done columns are skipped.
#   powershell -NoProfile -File launch_p2full.ps1                      # the main job (tag main, 6 workers, GPU behind the shared lock)
#   powershell -NoProfile -File launch_p2full.ps1 -Workers 6           # resume with a different pool size
#   powershell -NoProfile -File launch_p2full.ps1 -FromStage final     # redo only table + proof + analysis (all columns must exist)
# CreateFlags 0x08000200 = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW: two earlier un-flagged launches of the sibling 509-token job died with a bare ^C in the log.
param(
    [string]$Tag = "main",
    [int]$Workers = 6,
    [string]$FromStage = "all",
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
$pkg = $PSScriptRoot
$log = if ($Tag -eq "main") { Join-Path $pkg "driver_main.log" } else { Join-Path $pkg ("driver_" + $Tag + ".log") }
$cmdline = 'cmd.exe /c ""' + $py + '" "' + (Join-Path $pkg "p2full_driver.py") + '" --tag ' + $Tag + ' --workers ' + $Workers + ' --from-stage ' + $FromStage + ' ' + $Extra + ' >> "' + $log + '" 2>&1"'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdline; CurrentDirectory = $pkg; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"launched cmd.exe pid $($r.ProcessId) ; log $log"
