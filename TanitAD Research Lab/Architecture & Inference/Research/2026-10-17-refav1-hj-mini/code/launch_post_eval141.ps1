# S4c: launch the detached post-build runner for the eval-141 matched rebuild (waits for '[build] DONE', then runs
# check_eval141_rebuild.py + the eval dry-run). CPU only. Same CreateFlags as launch_build_eval141.ps1 (0x08000200).
$stage = "C:\Users\Admin\tanitad-data\refav1-eval141-256rb_stage"
$py    = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$scr   = "E:\Projects\TanitAD\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-refav1-hj-mini\code\post_build_eval141.py"
$cmd = "cmd.exe /c `"set CUDA_VISIBLE_DEVICES=-1&& set PYTHONIOENCODING=utf-8&& set PYTHONPATH=E:/Projects/TanitAD/stack&& `"$py`" -u `"$scr`" >> `"$stage\post_runner_stdout.log`" 2>&1`""
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0; CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $stage; ProcessStartupInformation = $si }
"launched pid=$($r.ProcessId) rc=$($r.ReturnValue)"
