# LEVELHEAD arm CLR guard (chain), runs ON box 2 (a dev-box guard would die with the launching session). v8F rule, unchanged (rnext_guard.ps1 / v8f_guard_*.sh): the bus-powered USB HDD (D:) holds cache_full;
#   STOP if >= 4 of the last 20 telemetry samples (15 s apart) read >= 61 C, or any one reads >= 63 C. Stop = write GUARD_STOP.txt (the chain halts, never proceeds to the next run) and Stop-Process OUR trainer (cmdline v8f_mini_train) + its children.
# A stopped run resumes from ckpt_last.pt on the same pinned line. Exits when one of its own end markers appears, or after 9 h. Reads only; touches nothing outside D:\tanitad.
# A telemetry csv that has not been written for 10 min is LOGGED as stale (the guard is blind then); it never stops anything.
$rc  = 'D:\tanitad\runs\v8f_mini_full\clr'
$csv = 'D:\tanitad\logs\telemetry_full.csv'
$log = 'D:\tanitad\logs\clr_guard.log'
$ends = @('CHAIN_CLR.done', 'CHAIN_CLR.STOPPED')
$t0 = Get-Date; $hi = 0; $lastStale = $null
Add-Content $log ((Get-Date).ToString('s') + ' guard armed (USB >= 4/20 samples >= 61 C or one >= 63 C)')
while (((Get-Date) - $t0).TotalHours -lt 9) {
  if (@($ends | Where-Object { Test-Path "$rc\$_" }).Count -gt 0) { Add-Content $log ((Get-Date).ToString('s') + " job ended; guard off (max USB $hi C)"); break }
  $fi = Get-Item $csv -ErrorAction SilentlyContinue
  if ((-not $fi) -or (((Get-Date) - $fi.LastWriteTime).TotalMinutes -gt 10)) {
    if ((-not $lastStale) -or (((Get-Date) - $lastStale).TotalMinutes -gt 10)) { Add-Content $log ((Get-Date).ToString('s') + ' WARNING telemetry csv missing or not written for 10 min: the guard is blind'); $lastStale = Get-Date }
  }
  $rows = @(Get-Content $csv -Tail 20 -ErrorAction SilentlyContinue | Where-Object { $_ -match '^20' })
  $u = @($rows | ForEach-Object { $f = $_.Split(','); if ($f.Count -ge 11 -and $f[10] -match '^\d+$') { [int]$f[10] } })
  if ($u.Count -gt 0) {
    $n61 = @($u | Where-Object { $_ -ge 61 }).Count
    $m = ($u | Measure-Object -Maximum).Maximum
    if ($m -gt $hi) { $hi = $m; Add-Content $log ((Get-Date).ToString('s') + " new max USB temp $m C") }
    if ($n61 -ge 4 -or $m -ge 63) {
      New-Item -ItemType Directory -Force $rc | Out-Null
      Set-Content "$rc\GUARD_STOP.txt" ("ZZGUARD-STOP TEMP " + (Get-Date).ToString('s') + " USB HDD: $n61 of the last 20 samples >= 61 C, max $m C")
      $p = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'")
      $ours = @($p | Where-Object { $_.CommandLine -match 'v8f_mini_train' })
      $ids = @($ours | ForEach-Object { $_.ProcessId })
      $kids = @($p | Where-Object { $ids -contains $_.ParentProcessId })
      foreach ($x in @($ours) + @($kids)) { Stop-Process -Id $x.ProcessId -Force -ErrorAction SilentlyContinue }
      Add-Content $log ((Get-Date).ToString('s') + " ZZGUARD-STOP TEMP: $n61 of 20 >= 61 C, max $m C; stopped $(@($ours).Count + @($kids).Count) process(es)")
      break
    }
  }
  Start-Sleep -Seconds 60
}
