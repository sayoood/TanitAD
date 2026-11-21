# End-to-end proof of the signed-link route on a CPU VM: provision -> price -> mint per-file signed links on the dev
# box (the token never leaves it) -> upload the job file -> delete it locally -> signed_pull.py verifies bytes + md5
# on the VM -> STOP in a finally -> assert on the SERVER that nothing is assigned.
#   powershell -NoProfile -File D:\Projects\TanitAD\colab\signed_proof.ps1 [-Prefix refe/snapshots] [-Files a,b]
param([string]$Name = "signed-proof", [string]$Prefix = "refe/snapshots", [string[]]$Files = @(), [string]$OutDir = "")
$ErrorActionPreference = "Continue"
$Here = $PSScriptRoot
$Exe = "C:\Users\Admin\venvs\colab\Scripts\colab.exe"
$Py = "C:\Users\Admin\venvs\colab\Scripts\python.exe"
$PyHF = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $Here "win_shims"
if (-not $OutDir) { $OutDir = Join-Path $Here ("raw\" + (Get-Date -Format "yyyy-MM-dd") + "-hf-relay") }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$Job = Join-Path $env:TEMP ("relay_job_" + [guid]::NewGuid().ToString("N") + ".json")   # live links: never in the repo
$Empty = Join-Path $env:TEMP "colab_empty_stdin.txt"
Set-Content -Path $Empty -Value "" -NoNewline

function Invoke-Logged([string]$File, [string[]]$ArgList, [string]$Log, [int]$TimeoutS) {
    $p = Start-Process -FilePath $File -ArgumentList $ArgList -NoNewWindow -PassThru -RedirectStandardInput $Empty `
        -RedirectStandardOutput "$Log.out" -RedirectStandardError "$Log.err"
    $null = $p.Handle
    if (-not $p.WaitForExit($TimeoutS * 1000)) { $p.Kill(); return "TIMEOUT" }
    return $p.ExitCode
}
function Get-Marked([string]$Log, [string]$Tag) {
    $l = Get-Content "$Log.out" -ErrorAction SilentlyContinue | Where-Object { $_ -like "$Tag *" } | Select-Object -Last 1
    if ($l) { return ($l.Substring($Tag.Length + 1) | ConvertFrom-Json) } else { return $null }
}

$L = Join-Path $OutDir $Name
$rec = [ordered]@{ session = $Name; runtime = "CPU"; prefix = $Prefix; t_start_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z" }
try {
    $t = Get-Date
    $rec.new_rc = Invoke-Logged $Exe @("new", "-s", $Name) "$L.new" 420
    $rec.new_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $null = Invoke-Logged $Py @("$Here\ccu_info.py") "$L.ccu_assigned" 90
    $c = Get-Marked "$L.ccu_assigned" "ZZCCU"
    $rec.rate_cu_per_h = $c.ccu.consumptionRateHourly
    $rec.assignments_while_assigned = @($c.assignments).Count
    if ($rec.new_rc -eq 0) {
        $rec.links_rc = Invoke-Logged $PyHF (@("$Here\hf_signed_links.py", "--prefix", $Prefix, "--out", $Job) + $Files) "$L.links" 300
        if ($rec.links_rc -eq 0 -and (Test-Path $Job)) {
            $rec.upload_rc = Invoke-Logged $Exe @("upload", "-s", $Name, $Job, "/content/relay_job.json") "$L.upload" 180
            Remove-Item -Force $Job
            $rec.local_job_deleted = -not (Test-Path $Job)
            $t = Get-Date
            $rec.pull_rc = Invoke-Logged $Exe @("exec", "-s", $Name, "--timeout", "1800", "-f", "$Here\signed_pull.py") "$L.pull" 2400
            $rec.pull_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
            $rec.pull = Get-Marked "$L.pull" "ZZPULL_OK"
            if (-not $rec.pull) { $rec.pull_fail = Get-Marked "$L.pull" "ZZPULL_FAIL" }
        }
    }
} finally {
    if (Test-Path $Job) { Remove-Item -Force $Job }
    $rec.stop_rc = Invoke-Logged $Exe @("stop", "-s", $Name) "$L.stop" 180
    Start-Sleep -Seconds 5
    $null = Invoke-Logged $Py @("$Here\ccu_info.py") "$L.ccu_after_stop" 90
    $c2 = Get-Marked "$L.ccu_after_stop" "ZZCCU"
    $rec.assignments_after_stop = if ($c2) { @($c2.assignments).Count } else { -1 }
    $rec.rate_after_stop = if ($c2) { $c2.ccu.consumptionRateHourly } else { -1 }
    $rec.t_end_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
    $rec | ConvertTo-Json -Depth 8 | Set-Content -Path "$L.json" -Encoding utf8
}
Write-Output ("ZZSIGNEDPROOF " + ($rec | ConvertTo-Json -Depth 8 -Compress))
if ($rec.assignments_after_stop -ne 0) { Write-Output "ZZSIGNEDPROOF_ORPHAN: an assignment is still live -- recover per RUNNER.md 9b" }
