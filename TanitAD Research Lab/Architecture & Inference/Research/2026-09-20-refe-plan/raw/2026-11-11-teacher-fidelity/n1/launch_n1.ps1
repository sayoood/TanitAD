# TF-N1 detached launch (PREREG_TF_N1.md section 12). Run AFTER the freeze, the ledger entry, n1_prep.sh (ZZN1_PREP PASS)
# and copying the landed ledger to <run>\inputs\ledger_K.json.
#   powershell -NoProfile -ExecutionPolicy Bypass -File launch_n1.ps1 -Sha <frozen sha256 of PREREG_TF_N1.md> [-Run C:\Users\Admin\n1_run]
# The same command resumes a stopped chain (finished stages and tokens are skipped).
param([Parameter(Mandatory = $true)][string]$Sha, [string]$Run = 'C:\Users\Admin\n1_run', [string]$StopAfter = '')
$stop = ''
if ($StopAfter) { $stop = "--stop-after $StopAfter" }
$py = 'C:\Users\Admin\navsim-crun\venv\Scripts\python.exe'
$os = Get-CimInstance Win32_OperatingSystem
"FreeVirtualMemory {0:N0} MB (the chain waits for > 10,240 MB before each child)" -f ($os.FreeVirtualMemory / 1KB)
foreach ($p in @("$Run\n1\tf_n1_chain.py", "$Run\n1\PREREG_TF_N1.md", "$Run\inputs\ledger_K.json")) {
  if (-not (Test-Path $p)) { throw "missing $p" }
}
$cmd = "cmd /c `"`"$py`" `"$Run\n1\tf_n1_chain.py`" --run-dir `"$Run`" --prereg `"$Run\n1\PREREG_TF_N1.md`" --prereg-sha256 $Sha " +
       "--ledger `"$Run\inputs\ledger_K.json`" --procs 2 $stop >> `"$Run\chain.log`" 2>&1 < NUL`""
$si = ([wmiclass]'Win32_ProcessStartup').CreateInstance()
$si.CreateFlags = [uint32]0x08000200          # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
$si.ShowWindow = 0
$r = ([wmiclass]'Win32_Process').Create($cmd, $Run, $si)
"WMI Create rc={0} pid={1}" -f $r.ReturnValue, $r.ProcessId
"check: Get-Content $Run\chain.log -Tail 5 ; dir $Run\markers"
