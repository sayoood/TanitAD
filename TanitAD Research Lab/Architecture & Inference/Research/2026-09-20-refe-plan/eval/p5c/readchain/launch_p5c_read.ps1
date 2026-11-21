# Detached launch of run_p5c_read.cmd through WMI with CreateFlags 0x08000200 (CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW).
# WMI children do NOT inherit this shell's environment, so every P5C_* / REFE_* / TANITAD_* variable set here is written to <work>\launch_env.cmd first
# and called by run_p5c_read.cmd. Verify by the heartbeat: <P5C_WORK>\CHAIN.log gains a line within a minute.
$hold = 'E:\Projects\TanitAD-artifacts\scorer_gym_20261107\p5c_readchain'
$work = if ($env:P5C_WORK) { $env:P5C_WORK } else { 'E:/Projects/TanitAD-artifacts/p5c_read' }
$workWin = $work -replace '/', '\'
New-Item -ItemType Directory -Force -Path $workWin | Out-Null
$envFile = Join-Path $workWin 'launch_env.cmd'
$lines = @('@echo off')
Get-ChildItem Env: | Where-Object { $_.Name -match '^(P5C_|REFE_|TANITAD_)' } | ForEach-Object { $lines += ('set "{0}={1}"' -f $_.Name, $_.Value) }
$lines += ('set "P5C_WORK={0}"' -f $work)
Set-Content -Path $envFile -Value $lines -Encoding ASCII
$cmd = "cmd.exe /c `"`"$hold\run_p5c_read.cmd`" `"$envFile`"`""
$si = (New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 })
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $hold; ProcessStartupInformation = $si }
"launched: ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId) env=$envFile"
