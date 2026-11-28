# Launch (or RESUME) the Amendment-A2 chain DETACHED, so it survives the launching session.  The same command = resume: finished units are skipped; a unit of another config is refused.
#   powershell -NoProfile -File launch_a2.ps1
# CreateFlags 0x08000200 = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW (copied from the P2 full-table launcher: un-flagged launches died with a bare ^C).
# A second launch while a chain is alive is harmless: chain_a2.py takes chain_a2.lock and the newcomer exits at once.
$ErrorActionPreference = "Stop"
$py = "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
$pkg = $PSScriptRoot
$log = Join-Path $pkg "logs\chain_a2.out"
$cmdline = 'cmd.exe /c ""' + $py + '" -B -I "' + (Join-Path $pkg "chain_a2.py") + '" >> "' + $log + '" 2>&1"'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdline; CurrentDirectory = $pkg; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"launched cmd.exe pid $($r.ProcessId) ; chain log $pkg\chain_a2.log ; stdout/stderr $log"
