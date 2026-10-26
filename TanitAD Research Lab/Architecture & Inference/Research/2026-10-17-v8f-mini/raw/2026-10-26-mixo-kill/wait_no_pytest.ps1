# Wait (at most 90 minutes) until no OTHER python process is running pytest, and until C: has >= 3 GB free; exit 0 when both hold, 3 on a timeout, 4 if the disk is below 3 GB at the deadline.
$deadline = (Get-Date).AddMinutes(240)
while ((Get-Date) -lt $deadline) {
    $n = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match ' -m pytest' }).Count
    $free = (Get-PSDrive C).Free
    if ($n -eq 0 -and $free -ge 1GB) { exit 0 }
    Start-Sleep -Seconds 20
}
if ((Get-PSDrive C).Free -lt 1GB) { exit 4 }
exit 3
