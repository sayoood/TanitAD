# End-to-end proof of the Colab data route on a CPU VM: provision -> price -> is the PI's Colab Secret readable here
# (secret_probe.py, booleans only) -> if yes, pull the relay MANIFEST's files and verify md5 (hf_relay_pull.py) ->
# STOP in a finally -> assert on the SERVER that nothing is assigned. Writes raw/<date>-hf-relay/relay_proof.json.
#   powershell -NoProfile -File D:\Projects\TanitAD\colab\relay_proof.ps1
param([string]$Name = "relay-proof", [string]$OutDir = "")
$ErrorActionPreference = "Continue"
$Here = $PSScriptRoot
$Exe = "C:\Users\Admin\venvs\colab\Scripts\colab.exe"
$Py = "C:\Users\Admin\venvs\colab\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $Here "win_shims"
if (-not $OutDir) { $OutDir = Join-Path $Here ("raw\" + (Get-Date -Format "yyyy-MM-dd") + "-hf-relay") }
New-Item -ItemType Directory -Force $OutDir | Out-Null
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
$rec = [ordered]@{ session = $Name; runtime = "CPU"; t_start_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z" }
try {
    $t = Get-Date
    $rec.new_rc = Invoke-Logged $Exe @("new", "-s", $Name) "$L.new" 420
    $rec.new_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $null = Invoke-Logged $Py @("$Here\ccu_info.py") "$L.ccu_assigned" 90
    $c = Get-Marked "$L.ccu_assigned" "ZZCCU"
    $rec.rate_cu_per_h = $c.ccu.consumptionRateHourly
    $rec.assignments_while_assigned = @($c.assignments).Count
    if ($rec.new_rc -eq 0) {
        $rec.probe_rc = Invoke-Logged $Exe @("exec", "-s", $Name, "--timeout", "90", "-f", "$Here\secret_probe.py") "$L.probe" 180
        $rec.secret = Get-Marked "$L.probe" "ZZSECRET"
        if ($rec.secret -and $rec.secret.payload_looks_like_hf_token) {
            $t = Get-Date
            $rec.pull_rc = Invoke-Logged $Exe @("exec", "-s", $Name, "--timeout", "900", "-f", "$Here\hf_relay_pull.py") "$L.pull" 1200
            $rec.pull_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
            $ok = Get-Content "$L.pull.out" -ErrorAction SilentlyContinue | Where-Object { $_ -like "ZZPULL_*" } | Select-Object -Last 1
            $rec.pull_line = if ($ok) { $ok.Substring(0, [Math]::Min(1500, $ok.Length)) } else { $null }
        }
    }
} finally {
    $rec.stop_rc = Invoke-Logged $Exe @("stop", "-s", $Name) "$L.stop" 180
    Start-Sleep -Seconds 5
    $null = Invoke-Logged $Py @("$Here\ccu_info.py") "$L.ccu_after_stop" 90
    $c2 = Get-Marked "$L.ccu_after_stop" "ZZCCU"
    $rec.assignments_after_stop = if ($c2) { @($c2.assignments).Count } else { -1 }
    $rec.rate_after_stop = if ($c2) { $c2.ccu.consumptionRateHourly } else { -1 }
    $rec.balance_after_stop = if ($c2) { $c2.ccu.currentBalance } else { -1 }
    $rec.t_end_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
    $rec | ConvertTo-Json -Depth 6 | Set-Content -Path "$L.json" -Encoding utf8
}
Write-Output ("ZZPROOF " + ($rec | ConvertTo-Json -Depth 6 -Compress))
if ($rec.assignments_after_stop -ne 0) { Write-Output "ZZPROOF_ORPHAN: an assignment is still live -- recover per RUNNER.md 9b" }
