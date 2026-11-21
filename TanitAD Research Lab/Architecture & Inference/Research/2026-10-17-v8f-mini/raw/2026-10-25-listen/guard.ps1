# memory/disk guard: exit 0 only if commit headroom >= 5 GB and C: free >= 0.7 GB; waits up to 4 min (re-check each minute)
for ($i = 0; $i -lt 5; $i++) {
  $os = Get-CimInstance Win32_OperatingSystem
  $head = $os.FreeVirtualMemory / 1MB
  $cfree = (Get-PSDrive C).Free / 1GB
  $line = "{0} guard: commit_headroom_GB={1:N2} C_free_GB={2:N3}" -f (Get-Date -Format s), $head, $cfree
  Write-Output $line
  Add-Content -Path "$PSScriptRoot\guard.log" -Value $line
  if ($head -ge 5 -and $cfree -ge 0.7) { exit 0 }
  if ($i -lt 4) { Start-Sleep -Seconds 60 }
}
exit 1
