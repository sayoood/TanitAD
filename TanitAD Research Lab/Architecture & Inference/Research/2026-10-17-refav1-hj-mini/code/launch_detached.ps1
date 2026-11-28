# Launch run_encode.sh DETACHED (Win32_Process.Create), so it survives the Claude session that starts it.
# (memory devbox-drive-letter-and-detached-jobs: nohup / Start-Process children die with the session.)
#   powershell -File launch_detached.ps1 [-Stages all|A|B|C|D] [-DryRun] [-MatchedEval]
# -DryRun runs the preflight only (clip-set assertions, parity gate, source resolution, resume plan): no GPU lock, no model.
# Log: C:\Users\Admin\refav1_hj_cache\logs\run_encode.out   (per-stage logs beside it)
param([string]$Stages = "all", [switch]$DryRun, [switch]$MatchedEval)
$bash = "C:\Program Files\Git\bin\bash.exe"
$code = Split-Path -Parent $MyInvocation.MyCommand.Path
$logs = "C:\Users\Admin\refav1_hj_cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$sh = (Join-Path $code "run_encode.sh") -replace '\\', '/'
$envs = ""
if ($DryRun) { $envs += "set DRY_RUN=1&& " }
if ($MatchedEval) { $envs += "set MATCHED_EVAL=1&& " }
$log = Join-Path $logs $(if ($DryRun) { "run_encode_dry.out" } else { "run_encode.out" })
$cmd = "cmd.exe /c `"$envs`"$bash`" `"$sh`" $Stages > `"$log`" 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $code }
"launched pid=$($r.ProcessId) rc=$($r.ReturnValue) log=$log"
"cmd: $cmd"
