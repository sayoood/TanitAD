# Detached launch of the TF-N2 re-roll chain (CIM Win32_Process Create, CreateFlags 0x08000200, stdin from NUL). Re-running it resumes.
$py  = "C:\Users\Admin\navsim-crun\venv\Scripts\python.exe"
$chn = "E:\Projects\TanitAD-artifacts\scorer_gym_20261107\teacher_fidelity\n2\reroll\tfn2_reroll_chain.py"
$run = "F:\tanitad_scratch\tfn2_reroll\run"
$cmd = "cmd /c `"`"$py`" `"$chn`" --run F:/tanitad_scratch/tfn2_reroll/run --w F:/tanitad_scratch/tfn2_reroll/w >> `"$run\logs\chain_stdout.log`" 2>&1 < NUL`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = "F:\tanitad_scratch\tfn2_reroll"; ProcessStartupInformation = (New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }) }
"pid=$($r.ProcessId) rc=$($r.ReturnValue)"
