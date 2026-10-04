# ONE metered Colab session of NAVSIM-on-Colab P0 (colab/NAVTEST_ON_COLAB_PLAN.md section f), start to server-checked stop:
#   ccu before -> new (CPU or L4) -> read the live rate, REFUSE above -MaxRate -> mint signed links (dev box; the token
#   never leaves it) -> upload code bundle + args + job file -> delete the local job file -> launch vm_p0.py DETACHED
#   -> poll every -PollS (refreshing the proxy token before its 3600 s expiry, RUNNER.md 9d) until the runner ends or
#   -CapMin is reached -> download the small results tarball -> STOP in finally -> assert on the SERVER that this
#   session's endpoint is gone and read the burn (COLAB_PRO.md section 3 rule 2). Own state file, own session names.
#   powershell -NoProfile -File D:\Projects\TanitAD\colab\navtest\p0_session.ps1 -Name navtest-p0-cpu `
#       -Stages setup_harness,gh,ge6,gh2 -Out <evidence dir>
#   ... -Name navtest-p0-l4 -Accel L4 -Stages setup_harness,setup_seam,genv,seam,gp,rep,ge6 -E6Workers 10 -MaxRate 1.7
param([Parameter(Mandatory = $true)][string]$Name, [string]$Accel = "", [Parameter(Mandatory = $true)][string]$Stages,
      [int]$E6Workers = 2, [int]$K = 3, [double]$MaxRate = 0.2, [int]$CapMin = 100, [int]$PollS = 60,
      [int]$DeadlineMin = 95, [Parameter(Mandatory = $true)][string]$Out, [string]$CodeTgz = "")
$ErrorActionPreference = "Continue"
$Here = $PSScriptRoot
$Colab = Split-Path $Here -Parent
$Exe = "C:\Users\Admin\venvs\colab\Scripts\colab.exe"
$Py = "C:\Users\Admin\venvs\colab\Scripts\python.exe"
$PyHF = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$Cfg = "C:\Users\Admin\.config\colab-cli\sessions_navtest.json"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = Join-Path $Colab "win_shims"
New-Item -ItemType Directory -Force $Out | Out-Null
$L = Join-Path $Out $Name
New-Item -ItemType Directory -Force $L | Out-Null
$Tmp = Join-Path $env:TEMP ("navtest_" + [guid]::NewGuid().ToString("N"))   # live links: never in the repo
New-Item -ItemType Directory -Force $Tmp | Out-Null
$Empty = Join-Path $Tmp "empty_stdin.txt"
Set-Content -Path $Empty -Value "" -NoNewline
$Seam = $Stages -match "setup_seam"

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
function Get-Ccu([string]$Log) {
    $null = Invoke-Logged $Py @("`"$Colab\ccu_info.py`"") $Log 90
    return (Get-Marked $Log "ZZCCU")
}
function Now-Utc { return (Get-Date).ToUniversalTime().ToString("s") + "Z" }

$rec = [ordered]@{ session = $Name; accel = $(if ($Accel) { $Accel } else { "CPU" }); stages = $Stages; config = $Cfg;
                   t_start_utc = (Now-Utc) }
$c0 = Get-Ccu "$L\00_ccu_before"
$rec.balance_before = $c0.ccu.currentBalance
$rec.assignments_before = @($c0.assignments | ForEach-Object { $_.endpoint })
$tAssigned = $null
try {
    if (-not $CodeTgz) {
        $CodeTgz = Join-Path $Tmp "p0_code.tgz"
        $null = Invoke-Logged $PyHF @("`"$Here\pack_code.py`"", "--out", "`"$CodeTgz`"", "--evidence", "`"$Out`"") "$L\01_pack" 300
        if (-not (Test-Path $CodeTgz)) { throw "pack_code failed" }
    }
    $newArgs = @("--config", $Cfg, "new", "-s", $Name)
    if ($Accel) { $newArgs += @("--gpu", $Accel) }
    $t = Get-Date
    $rec.new_rc = Invoke-Logged $Exe $newArgs "$L\02_new" 600
    $rec.new_s = [math]::Round(((Get-Date) - $t).TotalSeconds, 1)
    $tAssigned = Get-Date
    $rec.t_assigned_utc = (Now-Utc)
    $c = Get-Ccu "$L\03_ccu_assigned"
    $rec.rate_cu_per_h = $c.ccu.consumptionRateHourly
    $rec.assignments_while_assigned = $c.assignments
    $mine = @($c.assignments | Where-Object { $rec.assignments_before -notcontains $_.endpoint })
    $rec.endpoint = if ($mine.Count -ge 1) { $mine[0].endpoint } else { $null }
    if ($rec.new_rc -ne 0) { throw "new failed rc=$($rec.new_rc)" }
    if ($null -eq $rec.rate_cu_per_h -or [double]$rec.rate_cu_per_h -gt $MaxRate) {
        throw "rate $($rec.rate_cu_per_h) CU/h is above -MaxRate $MaxRate -- stopping now"
    }
    # --- signed links, minted NOW (60-min validity), merged into ONE job file for signed_pull.py
    $J1 = Join-Path $Tmp "job1.json"; $J2 = Join-Path $Tmp "job2.json"; $Job = Join-Path $Tmp "relay_job.json"
    $f1 = @("p0_harness_sub200.tar"); if ($Seam) { $f1 += "p0_frames_sub200.tar" }
    $rec.links_rc = Invoke-Logged $PyHF (@("`"$Colab\hf_signed_links.py`"", "--prefix", "refe/navtest_sub200", "--out", "`"$J1`"") + $f1) "$L\04_links1" 300
    if ($Seam) {
        $rec.links2_rc = Invoke-Logged $PyHF @("`"$Colab\hf_signed_links.py`"", "--prefix", "refe/snapshots", "--out", "`"$J2`"", "snap_epoch016.pt") "$L\04_links2" 300
    }
    $null = Invoke-Logged $PyHF @("`"$Here\merge_jobs.py`"", "`"$Job`"", "`"$J1`"", "`"$J2`"") "$L\04_merge" 60
    Remove-Item -Force $J1, $J2 -ErrorAction SilentlyContinue
    if (-not (Test-Path $Job)) { throw "no job file (links failed)" }
    $ArgsJ = Join-Path $Tmp "p0_args.json"
    $a = @("--stages", $Stages, "--e6-workers", "$E6Workers", "--k", "$K", "--deadline-min", "$DeadlineMin")
    if ($Accel) { $a += "--gpu" }
    ConvertTo-Json -InputObject $a -Compress | Set-Content -Path $ArgsJ -Encoding ascii
    $rec.up_code_rc = Invoke-Logged $Exe @("--config", $Cfg, "upload", "-s", $Name, "`"$CodeTgz`"", "/content/p0_code.tgz") "$L\05_up_code" 300
    $rec.up_args_rc = Invoke-Logged $Exe @("--config", $Cfg, "upload", "-s", $Name, "`"$ArgsJ`"", "/content/p0_args.json") "$L\05_up_args" 120
    $rec.up_job_rc = Invoke-Logged $Exe @("--config", $Cfg, "upload", "-s", $Name, "`"$Job`"", "/content/relay_job.json") "$L\05_up_job" 120
    Remove-Item -Force $Job
    $rec.local_job_deleted = -not (Test-Path $Job)
    $rec.launch_rc = Invoke-Logged $Exe @("--config", $Cfg, "exec", "-s", $Name, "--timeout", "240", "-f", "`"$Here\launch_p0.py`"") "$L\06_launch" 400
    $rec.t_launched_utc = (Now-Utc)
    if (-not (Select-String -Path "$L\06_launch.out" -Pattern "ZZLAUNCHED" -Quiet)) { throw "launch did not report ZZLAUNCHED" }
    $tRefresh = Get-Date
    $i = 0
    $rec.polls = 0
    while ($true) {
        Start-Sleep -Seconds $PollS
        $i++
        if (((Get-Date) - $tRefresh).TotalMinutes -gt 40) {
            $null = Invoke-Logged $Py @("`"$Here\refresh_session.py`"", "--config", $Cfg, "--name", $Name) "$L\07_refresh_$i" 120
            $tRefresh = Get-Date
        }
        $rc = Invoke-Logged $Exe @("--config", $Cfg, "exec", "-s", $Name, "--timeout", "120", "-f", "`"$Here\poll_p0.py`"") "$L\08_poll" 240
        $p = Get-Marked "$L\08_poll" "ZZPOLL"
        $rec.polls = $i
        if ($p) {
            Add-Content -Path "$L\polls.jsonl" -Value ((Now-Utc) + " " + ($p | ConvertTo-Json -Depth 6 -Compress)) -Encoding utf8
            $rec.last_poll = $p
            if ($p.ended -or -not $p.alive) { $rec.runner_end = if ($p.ended) { "ENDED" } else { "RUNNER_DEAD" }; break }
        } else {
            Add-Content -Path "$L\polls.jsonl" -Value ((Now-Utc) + " POLL_FAILED rc=$rc") -Encoding utf8
        }
        if (((Get-Date) - $tAssigned).TotalMinutes -gt $CapMin) { $rec.runner_end = "CAP_REACHED"; break }
    }
    # a final pack happens inside vm_p0 after every stage; take what is there
    $rec.dl_results_rc = Invoke-Logged $Exe @("--config", $Cfg, "download", "-s", $Name, "/content/p0/p0_results.tar.gz", "`"$L\p0_results.tar.gz`"") "$L\09_dl_results" 600
    $rec.dl_log_rc = Invoke-Logged $Exe @("--config", $Cfg, "download", "-s", $Name, "/content/p0/runner.log", "`"$L\runner.log`"") "$L\09_dl_log" 120
} catch {
    $rec.error = "$_"
} finally {
    # ⛔ MEASURED 2026-09-27 (session navtest-p0-l4): this block used to delete $Tmp FIRST -- and $Tmp holds the empty
    # stdin file every Invoke-Logged redirects from, so the stop and the ccu readout below could not start, returned
    # $null, and the L4 stayed assigned while the record read "not still assigned". Only the job file (live links)
    # is deleted here; the rest of $Tmp goes after the server check.
    Get-ChildItem $Tmp -Filter "*.json" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    $rec.stop_rc = Invoke-Logged $Exe @("--config", $Cfg, "stop", "-s", $Name) "$L\10_stop" 180
    $tStop = Get-Date
    $rec.t_stopped_utc = (Now-Utc)
    Start-Sleep -Seconds 8
    $c2 = Get-Ccu "$L\11_ccu_after_stop"
    $eps = if ($c2) { @($c2.assignments | ForEach-Object { $_.endpoint }) } else { @("CCU_READ_FAILED") }
    $rec.assignments_after_stop = $eps
    # an UNREADABLE server state is UNKNOWN, never "not assigned" (it was reported False once, while the L4 ran on)
    $rec.my_endpoint_still_assigned = if (-not $c2) { "UNKNOWN_CCU_READ_FAILED" } elseif ($rec.endpoint) { $eps -contains $rec.endpoint } else { "UNKNOWN_NO_ENDPOINT" }
    $rec.rate_after_stop = if ($c2) { $c2.ccu.consumptionRateHourly } else { -1 }
    $rec.balance_after_stop = if ($c2) { $c2.ccu.currentBalance } else { -1 }
    if ($tAssigned) {
        $rec.assigned_s = [math]::Round(($tStop - $tAssigned).TotalSeconds, 0)
        if ($rec.rate_cu_per_h) { $rec.cu_estimate = [math]::Round([double]$rec.rate_cu_per_h * $rec.assigned_s / 3600, 4) }
    }
    $rec.t_end_utc = (Now-Utc)
    $rec | ConvertTo-Json -Depth 8 | Set-Content -Path "$L\session_record.json" -Encoding utf8
    if (Test-Path $Tmp) { Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue }
}
Write-Output ("ZZP0_SESSION " + ($rec | ConvertTo-Json -Depth 8 -Compress))
if ($rec.my_endpoint_still_assigned -ne $false) { Write-Output "ZZP0_ORPHAN_OR_UNKNOWN: $($rec.endpoint) state $($rec.my_endpoint_still_assigned) -- check ccu_info.py and stop it (RUNNER.md 9b)" }
