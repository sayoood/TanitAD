# Launch (or RESUME) the P2-s1 auto-reader DETACHED, so it survives the launching session (Win32_Process.Create: parented by WMI, not by this shell).
# The same command = resume: the checkpoint is re-verified by hash, stage 2 reuses its seam, a previous P2S1READ_FAILED is cleared, a P2S1READ_DONE is left alone (-Extra "--force" re-runs).
#   powershell -NoProfile -File launch_p2s1_autoread.ps1
#   powershell -NoProfile -File launch_p2s1_autoread.ps1 -Extra "--poll-min 5"
# Foreground dry run (no GPU, no stage 2/3, nothing written outside out/dryrun/):
#   C:\Users\Admin\venvs\tanitad\Scripts\python.exe p2s1_autoread.py --dry-run
# CreateFlags 0x08000200 = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW (an un-flagged launch of a sibling dev-box job died with a bare ^C in its log).
# The HF token is NOT passed here: the job reads Keys.txt in place, in its own process. Nothing secret is on this command line or in this file.
param(
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$pkg = $PSScriptRoot
$out = Join-Path $pkg "out"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out "autoread_stdout.log"
$cmdline = 'cmd.exe /c ""' + $py + '" "' + (Join-Path $pkg "p2s1_autoread.py") + '" ' + $Extra + ' >> "' + $log + '" 2>&1"'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdline; CurrentDirectory = $pkg; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"launched cmd.exe pid $($r.ProcessId) ; stdout $log ; progress: $out\autoread.log ; state: $out\autoread_state.json ; markers: $out\P2S1READ_DONE | $out\P2S1READ_FAILED"
