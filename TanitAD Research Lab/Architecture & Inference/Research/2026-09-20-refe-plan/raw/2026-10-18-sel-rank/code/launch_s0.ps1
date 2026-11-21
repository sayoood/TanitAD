# Launch (or RESUME) SR-TTC S0 DETACHED, so it survives the launching session.  Same command = resume: banked batches are skipped, staged-but-unbanked batches are re-staged.
# WMI Win32_Process.Create, CreateFlags 0x08000200 = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, stdin from NUL (a job started from a session dies with it otherwise).
# Script execution is disabled on this box and -ExecutionPolicy Bypass is never used: paste the body into a PowerShell prompt instead of running the file.
#   .\launch_s0.ps1 -MinFreeRamMb 4500              # the main job
#   .\launch_s0.ps1 -Post                           # the post-step watcher (manifest + marker for a dead driver)
param(
    [int]$MinFreeRamMb = 4500,
    [switch]$Post
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
$code = "C:\Users\Admin\srttc_s0\code"
$run = "C:\Users\Admin\srttc_s0\run"
if ($Post) {
    $cmdline = 'cmd.exe /c ""' + $py + '" -I "' + $code + '\srttc_s0_post.py" < NUL >> "' + $run + '\post_watcher.log" 2>&1"'
} else {
    $cmdline = 'cmd.exe /c ""' + $py + '" -I "' + $code + '\srttc_s0_driver.py" --tag main --min-free-ram-mb ' + $MinFreeRamMb + ' < NUL >> "' + $run + '\driver_main.log" 2>&1"'
}
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdline; CurrentDirectory = $code; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"launched cmd.exe pid $($r.ProcessId)"
