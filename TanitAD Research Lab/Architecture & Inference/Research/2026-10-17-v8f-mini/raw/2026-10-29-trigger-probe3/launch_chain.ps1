param([string]$which = "A")
$stages = if ($which -eq "A") { "linear eval mutate" } else { "mlp evalmlp mlpctl" }
$cmd = "cmd /c `"`"C:\Program Files\Git\bin\bash.exe`" -c `"bash /e/Projects/TanitAD-artifacts/dagland/trigger_probe3/chain.sh $stages`" < NUL > E:\Projects\TanitAD-artifacts\dagland\trigger_probe3\raw\chain_stdout_$which.log 2>&1`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
"chain $which [$stages] ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
