# Generic detached launcher for YF2 jobs (WMI Win32_Process Create, CreateFlags 0x08000200, stdin NUL). A WMI launch WITHOUT these flags died within seconds (MEASURED 2026-11-06, 3 launches).
# usage: powershell -NoProfile -File launch_detached.ps1 -Tag R1_B_s0 -Cmd "<full command line, forward slashes ok>"
# The job writes <Tag>.child.log under markers\ ; wrap GPU commands in run_cache_job.py (lock + DONE/FAILED markers) instead of passing them bare:
#   -Cmd "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe E:\...\YF2\run_cache_job.py --marker-dir E:\...\YF2\markers --tag eval_B_s0 --ok-token ZZ... -- <gpu cmd>"
param([Parameter(Mandatory = $true)][string]$Tag, [Parameter(Mandatory = $true)][string]$Cmd)
$d = "E:\Projects\TanitAD-artifacts\yawrate_wp\YF2"
New-Item -ItemType Directory -Force "$d\markers" | Out-Null
$line = "cmd /c `"$Cmd < NUL > `"$d\markers\$Tag.out`" 2>&1`""
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $line; CurrentDirectory = $d; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId) tag=$Tag out=$d\markers\$Tag.out"
