# launch_chain4.ps1 -- start the detached chain via WMI (survives the agent; stdin = NUL).  Safe to re-run: the chain is resumable; check raw/chain4.log first so two copies are not started.
$cmd = 'cmd /c "E:\Projects\TanitAD-artifacts\dagland\trigger_probe4\chain4.cmd < NUL > E:\Projects\TanitAD-artifacts\dagland\trigger_probe4\raw\chain4_stdout.log 2>&1"'
$r = ([wmiclass]'Win32_Process').Create($cmd)
"chain4 ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
