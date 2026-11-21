# Measure what one compute unit buys on each Colab GPU this account may use: provision -> read the live hourly burn
# from ccu-info (the only measured source of a GPU's price) -> run gpu_smoke.py -> STOP -> assert on the SERVER that
# nothing is still assigned. One GPU at a time; an orphaned assignment halts the survey (it would keep billing).
#   powershell -NoProfile -File D:\Projects\TanitAD\colab\gpu_survey.ps1 -Gpus L4,G4,A100,T4
param([string[]]$Gpus = @("L4", "G4", "A100", "T4"), [string]$OutDir = "")
$ErrorActionPreference = "Continue"
$Here = $PSScriptRoot
$Exe = "C:\Users\Admin\venvs\colab\Scripts\colab.exe"
$Py = "C:\Users\Admin\venvs\colab\Scripts\python.exe"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $Here "win_shims"
if (-not $OutDir) { $OutDir = Join-Path $Here ("raw\" + (Get-Date -Format "yyyy-MM-dd") + "-gpu-survey") }
New-Item -ItemType Directory -Force $OutDir | Out-Null
$Empty = Join-Path $env:TEMP "colab_empty_stdin.txt"
Set-Content -Path $Empty -Value "" -NoNewline

function Invoke-Logged([string]$File, [string[]]$ArgList, [string]$Log, [int]$TimeoutS) {
    $p = Start-Process -FilePath $File -ArgumentList $ArgList -NoNewWindow -PassThru -RedirectStandardInput $Empty `
        -RedirectStandardOutput "$Log.out" -RedirectStandardError "$Log.err"
    $null = $p.Handle                     # without this, ExitCode reads empty after the exit
    if (-not $p.WaitForExit($TimeoutS * 1000)) { $p.Kill(); return "TIMEOUT" }
    return $p.ExitCode
}

function Get-Ccu([string]$Log) {
    $rc = Invoke-Logged $Py @("`"$Here\ccu_info.py`"") $Log 90
    $line = Get-Content "$Log.out" -ErrorAction SilentlyContinue | Where-Object { $_ -like "ZZCCU *" } | Select-Object -Last 1
    if (-not $line) { return $null }
    return ($line.Substring(6) | ConvertFrom-Json)
}

$Summary = @()
foreach ($g in $Gpus) {
    $name = "survey-" + $g.ToLower()
    $L = Join-Path $OutDir $name
    $rec = [ordered]@{ gpu = $g; session = $name; t_start_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z" }
    try {
        $t = Get-Date
        $rec.new_rc = Invoke-Logged $Exe @("new", "-s", $name, "--gpu", $g) "$L.new" 420
        $rec.new_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
        $c = Get-Ccu "$L.ccu_assigned"
        $rec.rate_cu_per_h = $c.ccu.consumptionRateHourly
        $rec.balance_while_assigned = $c.ccu.currentBalance
        $rec.assignments_while_assigned = @($c.assignments).Count
        if ($rec.new_rc -eq 0) {
            $t = Get-Date
            $rec.exec_rc = Invoke-Logged $Exe @("exec", "-s", $name, "--timeout", "600", "-f", "$Here\gpu_smoke.py") "$L.exec" 900
            $rec.exec_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
            $sm = Get-Content "$L.exec.out" -ErrorAction SilentlyContinue | Where-Object { $_ -like "ZZSMOKE *" } | Select-Object -Last 1
            if ($sm) { $rec.smoke = ($sm.Substring(8) | ConvertFrom-Json) }
        }
    } finally {
        $rec.stop_rc = Invoke-Logged $Exe @("stop", "-s", $name) "$L.stop" 180
        Start-Sleep -Seconds 5
        $c2 = Get-Ccu "$L.ccu_after_stop"
        $rec.assignments_after_stop = if ($c2) { @($c2.assignments).Count } else { -1 }
        $rec.rate_after_stop = if ($c2) { $c2.ccu.consumptionRateHourly } else { -1 }
        $rec.balance_after_stop = if ($c2) { $c2.ccu.currentBalance } else { -1 }
        $rec.t_end_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
        $rec | ConvertTo-Json -Depth 6 | Set-Content -Path "$L.json" -Encoding utf8
        $Summary += [pscustomobject]$rec
    }
    Write-Output ("ZZGPU " + ($rec | ConvertTo-Json -Depth 6 -Compress))
    if ($rec.assignments_after_stop -ne 0) {
        Write-Output "ZZSURVEY_ORPHAN: an assignment is still live after stopping $name -- survey halted (RUNNER.md 9b)"
        break
    }
}
$Summary | ConvertTo-Json -Depth 6 | Set-Content -Path (Join-Path $OutDir "survey.json") -Encoding utf8
Write-Output "ZZSURVEY_DONE $OutDir"
