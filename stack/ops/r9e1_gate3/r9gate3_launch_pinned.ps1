param([Parameter(Mandatory)][string]$Sha, [string]$Arms = "C0 C1", [string]$Study = "e1", [string]$Stage = "F:\refcv9_e1\gate")
# launches r9gate3_devbox_pinned.sh DETACHED through WMI (CreateFlags 0x08000200: new process group, no window) so it survives the session
# -Study / -Stage: handed over by r9gate3_restart.sh from r9gate3_study.sh (R9_STUDY, R9_DEVSTAGE).  A WMI-created process gets the PROVIDER's environment, not this
# one's, so the study cannot travel as an environment variable: it is written into the command line.  The defaults are the E1 study's (what this script always did).
$c12 = $Sha.Substring(0, 12)
if ($Study -notmatch '^[A-Za-z0-9]+$') { Write-Output "REFUSED: -Study must be a study name (letters, digits)"; exit 1 }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
$cmd = "$Stage\pinned_run_$c12.cmd"
# the driver next to THIS script (was a hard-coded qland/hold_r9_gate3 path)
$here = '/' + $PSScriptRoot.Substring(0, 1).ToLower() + ($PSScriptRoot.Substring(2) -replace '\\', '/')
$line = '"C:\Program Files\Git\bin\bash.exe" -c "R9_STUDY=' + $Study + ' bash ' + $here + '/r9gate3_devbox_pinned.sh ' + $Sha + ' ''' + $Arms + '''" < NUL > ' + $Stage + '\pinned_wmi_' + $c12 + '.log 2>&1'
[System.IO.File]::WriteAllText($cmd, "@echo off`r`n$line`r`n", (New-Object System.Text.UTF8Encoding($false)))
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = "cmd.exe /c $cmd"; CurrentDirectory = $Stage; ProcessStartupInformation = $si }
if ($r.ReturnValue -ne 0) { exit 1 }
"pinned driver pid $($r.ProcessId)"
