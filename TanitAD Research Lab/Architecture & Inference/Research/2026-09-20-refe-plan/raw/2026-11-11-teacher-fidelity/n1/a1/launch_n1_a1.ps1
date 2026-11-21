# TF-N1 A1 detached launch (PREREG_TF_N1_A1.md). No -ExecutionPolicy Bypass is needed: paste/dot-run this from the MM's own PowerShell,
# or run `. .\launch_n1_a1.ps1 -AmendmentSha .. -A1PinsSha ..` if scripts are allowed. Win32_Process Create, CreateFlags 0x08000200
# (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP), stdin from NUL, output appended to <Run>\chain_a1.log.
#
# ORDER (the chain REFUSES otherwise): a1_prep.py -> freeze the amendment (sha256) -> append the ledger entry (n = 42) and land it ->
#   copy the landed ledger to <Run>\inputs\ledger_K.json -> `-CheckOnly` (guards, nothing started) -> launch.
# -Print shows the command line and starts nothing. The same command resumes a stopped chain (finished stages are skipped).
param([Parameter(Mandatory = $true)][string]$AmendmentSha, [Parameter(Mandatory = $true)][string]$A1PinsSha,
      [string]$Run = 'C:\Users\Admin\n1_run', [string]$Amendment = '', [string]$StopAfter = '', [switch]$CheckOnly, [switch]$Print,
      [string]$Cmd = '')
$py = 'C:\Users\Admin\navsim-crun\venv\Scripts\python.exe'
if (-not $Amendment) { $Amendment = "$Run\n1\a1\PREREG_TF_N1_A1.md" }
$extra = ''
if ($StopAfter) { $extra += " --stop-after $StopAfter" }
if ($CheckOnly) { $extra += ' --check-only' }
if (-not $Cmd) {
  $Cmd = "cmd /c `"`"$py`" `"$Run\n1\a1\tf_n1_a1_chain.py`" --run-dir `"$Run`" --amendment `"$Amendment`" --amendment-sha256 $AmendmentSha " +
         "--a1-pins-sha256 $A1PinsSha --ledger `"$Run\inputs\ledger_K.json`" --k 42 --procs 2$extra >> `"$Run\chain_a1.log`" 2>&1 < NUL`""
}
$Cmd
if ($Print) { return }
foreach ($p in @("$Run\n1\a1\tf_n1_a1_chain.py", $Amendment, "$Run\inputs\ledger_K.json")) {
  if (-not (Test-Path $p)) { throw "missing $p" }
}
$os = Get-CimInstance Win32_OperatingSystem
"FreeVirtualMemory {0:N0} MB (the chain waits for > 10,240 MB before each child)" -f ($os.FreeVirtualMemory / 1KB)
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200; ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $Cmd; CurrentDirectory = $Run; ProcessStartupInformation = $si }
"Win32_Process Create rc={0} pid={1}" -f $r.ReturnValue, $r.ProcessId
"check: Get-Content $Run\chain_a1.log -Tail 5 ; dir $Run\a1\markers"
