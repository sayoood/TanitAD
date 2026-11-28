# Detached launch of the u400 scene-cache build behind the shared GPU lock (WMI Win32_Process Create, stdin NUL).
$py = "C:\Users\Admin\venvs\driverl-eval\Scripts\python.exe"
$d  = "E:\Projects\TanitAD-artifacts\yawrate_wp\YF2"
$cmd = "cmd /c `"`"$py`" `"$d\run_cache_job.py`" --marker-dir `"$d\markers`" --tag cache_u400_4k_r3 --give-up-h 10 -- `"$py`" `"$d\proxy_cache_yf2.py`" build --manifest `"$d\manifest_yaw_4k.json`" --snapshot `"E:\Projects\TanitAD\data\refe_runs_eval\sft4_peek\model_sft4_B_u400.pt`" --expect-snapshot-sha256 8b4dba48 --out `"$d\cache_u400_4k`" --visual-for sets --amp bf16 --device cuda < NUL > `"$d\markers\cache_u400_4k_r3.wrapper.log`" 2>&1`""
New-Item -ItemType Directory -Force "$d\markers" | Out-Null
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }   # NEW_PROCESS_GROUP | NO_WINDOW: without it the job dies within seconds (MEASURED 2026-11-06, 3 launches)
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $d; ProcessStartupInformation = $si }
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId)"
