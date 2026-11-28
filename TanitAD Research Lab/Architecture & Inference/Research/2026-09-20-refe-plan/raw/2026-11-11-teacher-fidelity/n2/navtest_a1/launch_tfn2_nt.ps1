# TF-N2 navtest detached launch (PREREG_TF_N2.md). No -ExecutionPolicy Bypass is needed: paste or dot-run this from the MM's own PowerShell.
# Win32_Process Create, CreateFlags 0x08000200 (CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP), stdin from NUL, output appended to <Run>\chain.log.
#
# ORDER (the chain REFUSES otherwise): n2_nt_prep.py -> freeze the card (sha256) -> append the ledger entry (n = 43) and land it ->
#   copy the landed ledger to <Run>\inputs\ledger_K.json -> `-CheckOnly` (guards, nothing started) -> launch.
# -Print shows the command line and starts nothing. The SAME command resumes a stopped chain (finished stages and written rows are skipped).
# -StopAfter <stage> ends the chain after that stage (roll_R0 | ctl_R0 | roll_RJ5 | build | score_T | cidh | cidc | score_N2 | read | families).
param([Parameter(Mandatory = $true)][string]$CardSha, [Parameter(Mandatory = $true)][string]$PinsSha,
      [string]$Run = 'F:\tanitad_scratch\tfn2_nt\run', [string]$Card = '', [string]$StopAfter = '', [switch]$CheckOnly, [switch]$Print,
      [string]$Cmd = '')
$py = 'C:\Users\Admin\navsim-crun\venv\Scripts\python.exe'
if (-not $Card) { $Card = "$Run\n2\PREREG_TF_N2.md" }
$extra = ''
if ($StopAfter) { $extra += " --stop-after $StopAfter" }
if ($CheckOnly) { $extra += ' --check-only' }
if (-not $Cmd) {
  $Cmd = "cmd /c `"`"$py`" `"$Run\n2\tfn2_nt_chain.py`" --run-dir `"$Run`" --card `"$Card`" --card-sha256 $CardSha " +
         "--pins-sha256 $PinsSha --ledger `"$Run\inputs\ledger_K.json`" --k 44 --procs 2$extra >> `"$Run\chain.log`" 2>&1 < NUL`""
}
$Cmd
if ($Print) { return }
foreach ($p in @("$Run\n2\tfn2_nt_chain.py", $Card, "$Run\inputs\ledger_K.json")) {
  if (-not (Test-Path $p)) { throw "missing $p" }
}
$os = Get-CimInstance Win32_OperatingSystem
"FreeVirtualMemory {0:N0} MB (the chain waits for > 10,240 MB before each child)" -f ($os.FreeVirtualMemory / 1KB)
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200; ShowWindow = [uint16]0 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $Cmd; CurrentDirectory = $Run; ProcessStartupInformation = $si }
"Win32_Process Create rc={0} pid={1}" -f $r.ReturnValue, $r.ProcessId
"check: Get-Content $Run\chain.log -Tail 5 ; dir $Run\markers"
