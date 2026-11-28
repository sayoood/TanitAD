$log = 'C:\Users\Admin\refcv8_pf\picker\logs\ram_sampler.log'
$peak = @{}
$end = (Get-Date).AddMinutes(90)
while ((Get-Date) -lt $end) {
  $os = Get-CimInstance Win32_OperatingSystem
  $free = [math]::Round($os.FreePhysicalMemory/1MB,2)
  $mine = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*refcv8_pf/picker*' -or $_.CommandLine -like '*refcv8_pf\picker*' }
  $mx = 0
  foreach ($m in $mine) { $p = Get-Process -Id $m.ProcessId -ErrorAction SilentlyContinue; if ($p) { $g = [math]::Round($p.PeakWorkingSet64/1GB,2); if ($g -gt $mx) { $mx = $g }; $k = ($m.CommandLine -split '\s+' | Where-Object { $_ -like '*.py' -or $_ -in @('tf_forward','d8','df','tac','parity','clamp','compliant','full','k9') }) -join ' '; if (-not $peak.ContainsKey($k) -or $peak[$k] -lt $g) { $peak[$k] = $g } } }
  "$(Get-Date -Format 'HH:mm:ss') free_GB=$free n_mine=$($mine.Count) max_peak_GB_now=$mx" | Out-File -Append -Encoding utf8 $log
  ($peak.GetEnumerator() | Sort-Object Value -Descending | ForEach-Object { "  PEAK $($_.Value) $($_.Key)" }) | Out-File -Encoding utf8 'C:\Users\Admin\refcv8_pf\picker\logs\ram_peaks.txt'
  Start-Sleep -Seconds 20
}
