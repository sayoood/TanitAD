# RUN 2 of the CPU smoke (the steps run 1 never reached; see vm_smoke_navplan_run2.py). Same guards as run 1:
# provision (no --gpu) -> read the live CU rate -> refuse if it cannot fit the 0.5 CU cap -> upload the ~18 MB bundle
# -> exec vm_smoke_navplan.py -> download its artifacts -> STOP in finally -> assert on the SERVER that nothing is
# assigned and the burn is 0 (COLAB_PRO.md section 3 rule 2). Own state file + own session name (parallel agents).
#   powershell -NoProfile -File D:\Projects\TanitAD\colab\raw\2026-09-27-navtest-plan\navplan_smoke.ps1 -Bundle <tgz>
param([Parameter(Mandatory = $true)][string]$Bundle, [double]$MaxRate = 0.55, [int]$ExecTimeoutS = 2100)
$ErrorActionPreference = "Continue"
$Here = $PSScriptRoot
$Colab = "D:\Projects\TanitAD\colab"
$Exe = "C:\Users\Admin\venvs\colab\Scripts\colab.exe"
$Py = "C:\Users\Admin\venvs\colab\Scripts\python.exe"
$Cfg = "C:\Users\Admin\.config\colab-cli\sessions_navplan.json"
$Name = "navplan-cpu"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $Colab "win_shims"
$L = Join-Path $Here "smoke_run2"
New-Item -ItemType Directory -Force $L | Out-Null
$Empty = Join-Path $env:TEMP "colab_empty_stdin_navplan.txt"
Set-Content -Path $Empty -Value "" -NoNewline

function Invoke-Logged([string]$File, [string[]]$ArgList, [string]$Log, [int]$TimeoutS) {
    $p = Start-Process -FilePath $File -ArgumentList $ArgList -NoNewWindow -PassThru -RedirectStandardInput $Empty `
        -RedirectStandardOutput "$Log.out" -RedirectStandardError "$Log.err"
    $null = $p.Handle
    if (-not $p.WaitForExit($TimeoutS * 1000)) { $p.Kill(); return "TIMEOUT" }
    return $p.ExitCode
}
function Get-Ccu([string]$Log) {
    $null = Invoke-Logged $Py @("`"$Colab\ccu_info.py`"") $Log 90
    $line = Get-Content "$Log.out" -ErrorAction SilentlyContinue | Where-Object { $_ -like "ZZCCU *" } | Select-Object -Last 1
    if (-not $line) { return $null }
    return ($line.Substring(6) | ConvertFrom-Json)
}

$rec = [ordered]@{ session = $Name; config = $Cfg; bundle = $Bundle; t_start_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z" }
$c0 = Get-Ccu "$L\00_ccu_before"
$rec.balance_before = $c0.ccu.currentBalance
$rec.assignments_before = @($c0.assignments).Count
try {
    $t = Get-Date
    $rec.new_rc = Invoke-Logged $Exe @("--config", $Cfg, "new", "-s", $Name) "$L\01_new" 420
    $rec.new_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $rec.t_assigned_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
    $c = Get-Ccu "$L\02_ccu_assigned"
    $rec.rate_cu_per_h = $c.ccu.consumptionRateHourly
    $rec.balance_while_assigned = $c.ccu.currentBalance
    $rec.assignments_while_assigned = $c.assignments
    if ($rec.new_rc -ne 0) { throw "new failed rc=$($rec.new_rc)" }
    if ($null -eq $rec.rate_cu_per_h -or [double]$rec.rate_cu_per_h -gt $MaxRate) {
        throw "rate $($rec.rate_cu_per_h) CU/h cannot fit the 0.5 CU cap over ~0.5 h (max $MaxRate) -- stopping now"
    }
    $t = Get-Date
    $rec.upload_rc = Invoke-Logged $Exe @("--config", $Cfg, "upload", "-s", $Name, "`"$Bundle`"", "/content/navplan_bundle.tgz") "$L\03_upload" 900
    $rec.upload_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $rec.upload_bytes = (Get-Item $Bundle).Length
    $t = Get-Date
    $rec.exec_rc = Invoke-Logged $Exe @("--config", $Cfg, "exec", "-s", $Name, "--timeout", "$ExecTimeoutS", "-f", "`"$Here\vm_smoke_navplan_run2.py`"") "$L\04_exec" ($ExecTimeoutS + 180)
    $rec.exec_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $rec.dl_results = Invoke-Logged $Exe @("--config", $Cfg, "download", "-s", $Name, "/content/navplan/results_run2.json", "`"$L\vm_results_run2.json`"") "$L\05_dl_results" 300
} catch {
    $rec.error = "$_"
} finally {
    $rec.stop_rc = Invoke-Logged $Exe @("--config", $Cfg, "stop", "-s", $Name) "$L\06_stop" 180
    $rec.t_stopped_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
    Start-Sleep -Seconds 8
    $c2 = Get-Ccu "$L\07_ccu_after_stop"
    $rec.assignments_after_stop = if ($c2) { @($c2.assignments).Count } else { -1 }
    $rec.assignments_after_stop_list = if ($c2) { $c2.assignments } else { $null }
    $rec.rate_after_stop = if ($c2) { $c2.ccu.consumptionRateHourly } else { -1 }
    $rec.balance_after_stop = if ($c2) { $c2.ccu.currentBalance } else { -1 }
    $rec.t_end_utc = (Get-Date).ToUniversalTime().ToString("s") + "Z"
    $rec | ConvertTo-Json -Depth 8 | Set-Content -Path "$L\smoke_record.json" -Encoding utf8
}
Write-Output ("ZZNAVPLAN_SMOKE " + ($rec | ConvertTo-Json -Depth 8 -Compress))
