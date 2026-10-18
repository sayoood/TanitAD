# Box 2 telemetry for LONG jobs (PREREG_RA1_FULL build, ~22 h). telemetry.ps1's columns, one line every 15 s, max 40 h.
# Read-only: changes nothing. Stops when D:\tanitad\logs\telemetry.stop exists. Each Add-Content flushes, so a crash loses one line.
$out  = 'D:\tanitad\logs\telemetry_full.csv'
$stop = 'D:\tanitad\logs\telemetry.stop'
if (Test-Path $stop) { Remove-Item $stop -Force }
Add-Content -Path $out -Value 'time,gpu_temp_c,gpu_power_w,gpu_sm_mhz,gpu_util_pct,gpu_mem_mib,gpu_throttle,cpu_perf_pct,cpu_time_pct,nvme_temp_c,usb_hdd_temp_c,py_procs'
$t0 = Get-Date; $i = 0; $nvme = ''; $usb = ''
while (-not (Test-Path $stop) -and ((Get-Date) - $t0).TotalHours -lt 40) {
  $g = (& nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used,clocks_throttle_reasons.active --format=csv,noheader,nounits) -replace ' ', ''
  try {
    $c = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation -Filter "Name='_Total'" -ErrorAction Stop
    $perf = $c.PercentProcessorPerformance; $busy = $c.PercentProcessorTime
  } catch { $perf = ''; $busy = '' }
  if ($i % 4 -eq 0) {
    try {
      foreach ($d in Get-PhysicalDisk) {
        $r = $d | Get-StorageReliabilityCounter -ErrorAction Stop
        if ($d.BusType -eq 'NVMe') { $nvme = $r.Temperature } elseif ($d.BusType -eq 'USB') { $usb = $r.Temperature }
      }
    } catch {}
  }
  $np = @(Get-Process python -ErrorAction SilentlyContinue).Count
  Add-Content -Path $out -Value ((Get-Date).ToString('yyyy-MM-dd HH:mm:ss') + ',' + $g + ',' + $perf + ',' + $busy + ',' + $nvme + ',' + $usb + ',' + $np)
  $i++; Start-Sleep -Seconds 15
}
Add-Content -Path $out -Value ('# stopped ' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
