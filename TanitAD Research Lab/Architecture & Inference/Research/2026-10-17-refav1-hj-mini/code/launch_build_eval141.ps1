# S4c: launch the CPU-only native-256x640 rebuild of the 141 EVAL v2ep (matched to the train-600 decode path) DETACHED (Win32_Process.Create), so it
# survives the Claude session (memory devbox-drive-letter-and-detached-jobs). Resumable: rerun the same command.
#   powershell -NoProfile -File launch_build_eval141.ps1 [-Workers 8]
# Builder = the banked stack/scripts/pod_build_b1_epcache.py, UNCHANGED (it drives v2_compressed.build_compressed).
# CUDA_VISIBLE_DEVICES=-1 (an EMPTY string does NOT hide the 4060 on this box). No Thor, no GPU.
param([int]$Workers = 6)
$stage = "C:\Users\Admin\tanitad-data\refav1-eval141-256rb_stage"
$out   = "C:\Users\Admin\tanitad-data\refav1-eval141-256rb"
$py    = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$bld   = "E:\Projects\TanitAD\stack\scripts\pod_build_b1_epcache.py"
$log   = Join-Path $stage "build.log"
New-Item -ItemType Directory -Force $out | Out-Null
$envs = "set CUDA_VISIBLE_DEVICES=-1&& set TANITAD_STACK=E:/Projects/TanitAD/stack&& set PYTHONIOENCODING=utf-8&& " +
        "set OMP_NUM_THREADS=1&& set MKL_NUM_THREADS=1&& set V2_TORCH_THREADS=1&& set PAI_DECODE_THREADS=1&& " +
        "set PYTHONPATH=E:/Projects/TanitAD/stack&& "
$cmd = "cmd.exe /c `"$envs`"$py`" -u `"$bld`" --out `"$out`" --root `"$stage`" --workers $Workers --clips `"$stage\clips141.txt`" >> `"$log`" 2>&1`""
# MEASURED 2026-10-17: a plain Win32_Process.Create child that imports torch DIES AT ONCE with
# "forrtl: error (200): program aborting due to window-CLOSE event" (console torn down). CreateFlags
# CREATE_NO_WINDOW|CREATE_NEW_PROCESS_GROUP (0x08000200) SURVIVES; DETACHED_PROCESS (0x8) and NO_WINDOW alone did not.
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ ShowWindow = [uint16]0; CreateFlags = [uint32]0x08000200 }
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $stage; ProcessStartupInformation = $si }
"launched pid=$($r.ProcessId) rc=$($r.ReturnValue) log=$log"
"cmd: $cmd"
