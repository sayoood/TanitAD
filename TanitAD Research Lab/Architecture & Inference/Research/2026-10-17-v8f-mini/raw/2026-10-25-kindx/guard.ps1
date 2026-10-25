# prints "OK <cfreeGB> <headroomGB>" or "FAIL ..." ; thresholds: C: free >= 1.5 GB, commit headroom >= 4 GB
$os = Get-CimInstance Win32_OperatingSystem
$free = (Get-PSDrive C).Free / 1GB
$head = $os.FreeVirtualMemory / 1MB
if ($free -ge 1.5 -and $head -ge 4.0) { "OK {0:N2} {1:N2}" -f $free, $head; exit 0 } else { "FAIL cfree={0:N2} headroom={1:N2}" -f $free, $head; exit 1 }
