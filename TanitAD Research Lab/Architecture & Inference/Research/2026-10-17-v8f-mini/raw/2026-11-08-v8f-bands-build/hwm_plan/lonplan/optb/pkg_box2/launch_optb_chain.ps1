# launch_optb_chain.ps1 -Commit <40 hex> [-HfRepo owner/name] [-Resume]
# DO NOT RUN WITHOUT THE MM: this starts the Option B chain on the dev box, which then drives box 2. Plain WMI is not enough (the console is handed to Windows Terminal and the chain dies in seconds), so CreateFlags 0x08000200 is passed.
# Order: (1) bash optb_waiter.sh snap  (baseline BEFORE launch)  (2) this script  (3) a waiter: bash optb_waiter.sh wait  (check the heartbeat: chain.log grows within 60 s).
param([Parameter(Mandatory = $true)][string]$Commit, [string]$HfRepo = "", [switch]$Resume)
if ($Commit -notmatch '^[0-9a-f]{40}$') { throw "Commit must be 40 lowercase hex" }
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$bash = 'C:\Program Files\Git\usr\bin\bash.exe'   # NOT Git\bin\bash.exe (its wrapper ^C-kills the batch at exit)
$ssh = (& $bash -lc 'command -v ssh').Trim(); $sftp = (& $bash -lc 'command -v sftp').Trim()
$cmdFile = Join-Path $here 'optb_chain_run.cmd'
$lines = @('@echo off', 'cd /d "' + $here + '"',
  'set "OPTB_COMMIT=' + $Commit + '"', 'set "OPTB_SSH=' + $ssh + '"', 'set "OPTB_SFTP=' + $sftp + '"', 'set "OPTB_HF_REPO=' + $HfRepo + '"', 'set "HF_HUB_DISABLE_XET=1"')
if ($Resume) { $lines += 'set "OPTB_RESUME=1"' }
$lines += '"' + $bash + '" "' + ($here -replace '\', '/') + '/optb_chain.sh" >> "C:\Users\Admin\optb_state\chain_stdout.log" 2>&1'
$lines += 'echo rc=%ERRORLEVEL% finished=%DATE% %TIME%>> "C:\Users\Admin\optb_state\chain_stdout.log"'
Set-Content -Path $cmdFile -Value $lines -Encoding ASCII
New-Item -ItemType Directory -Force 'C:\Users\Admin\optb_state' | Out-Null
$s = ([wmiclass]"Win32_ProcessStartup").CreateInstance(); $s.CreateFlags = 0x08000200
$r = ([wmiclass]"Win32_Process").Create("cmd.exe /c `"$cmdFile`" < NUL", $here, $s)
"pid=$($r.ProcessId) rv=$($r.ReturnValue)"
