# Launch (or RESUME) the S1c auto-reader DETACHED, so it survives the launching session (Win32_Process.Create: parented by WMI, not by this shell).
# NOT LAUNCHED BY ITS AUTHOR: the Master Mind launches it after review (it polls Sayood/tanitad-refe-s1aux for arms/H1u10k-s0/ every 10 min, up to 48 h, then reads it by itself).
# The same command = resume: completed stages are reused (cached by input fingerprint), a previous S1CREAD_FAILED is cleared, an S1CREAD_DONE is left alone (-Extra "--force" re-runs).
#   powershell -NoProfile -File launch_s1c_autoread.ps1
#   powershell -NoProfile -File launch_s1c_autoread.ps1 -Extra "--poll-min 5"
#   powershell -NoProfile -File launch_s1c_autoread.ps1 -Extra "--forward-device cpu"      # skip the GPU entirely (CPU is the banked-S1 precedent path; ~15-40 min under contention)
# Foreground dry run on S1's banked H1-s0 artifacts (no GPU, no HF poll, nothing written outside out/dryrun/):
#   C:\Users\Admin\venvs\tanitad\Scripts\python.exe -I s1c_autoread.py --dry-run --no-live
# GPU: the forward waits for the shared lock C:\Users\Admin\qland\work\refcv7\devbox_gpu.lock (--gpu-wait-min, default 30) and NEVER breaks it; on a timeout `auto` falls back to CPU.
# CreateFlags 0x08000200 = CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW (an un-flagged launch of a sibling dev-box job died with a bare ^C in its log).
# `-X utf8` makes the redirected stdout UTF-8 (a cp1252 console encoding would choke on the prereg's non-ASCII quotes).
# The HF token is NOT passed here: the job reads Keys.txt in place, in its own process. Nothing secret is on this command line or in this file.
# If D: is absent (the artifacts drive can flip D: -> E:), run `subst D: E:\` first (memory: devbox-drive-letter-and-detached-jobs).
param(
    [string]$Extra = ""
)
$ErrorActionPreference = "Stop"
$py = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$pkg = $PSScriptRoot
$out = Join-Path $pkg "out"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = Join-Path $out "autoread_stdout.log"
$cmdline = 'cmd.exe /c ""' + $py + '" -I -X utf8 "' + (Join-Path $pkg "s1c_autoread.py") + '" ' + $Extra + ' >> "' + $log + '" 2>&1"'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdline; CurrentDirectory = $pkg; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { throw "Win32_Process.Create failed: $($r.ReturnValue)" }
"launched cmd.exe pid $($r.ProcessId) ; stdout $log ; progress: $out\autoread.log ; state: $out\autoread_state.json ; markers: $out\S1CREAD_DONE | $out\S1CREAD_FAILED"
