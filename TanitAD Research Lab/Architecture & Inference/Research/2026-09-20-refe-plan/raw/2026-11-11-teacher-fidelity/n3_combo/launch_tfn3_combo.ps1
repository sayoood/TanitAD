# Detached launch of the TF-N3 combo chain (CIM Win32_Process Create, CreateFlags 0x08000200, stdin from NUL). Re-running it resumes.
# Args: -Run <run dir> (default the full run), -Extra "<more chain args>" (the smoke uses --expect-n 2 --ctl-n 1).
param([string]$Run = "F:\tanitad_scratch\tfn3_combo\run", [string]$Extra = "")
$py  = "C:\Users\Admin\navsim-crun\venv\Scripts\python.exe"
$chn = "E:\Projects\TanitAD-artifacts\scorer_gym_20261107\teacher_fidelity\n3_combo\tfn3_combo_chain.py"
$runf = $Run.Replace('\', '/')
$cmd = "cmd /c `"`"$py`" `"$chn`" --run $runf --w F:/tanitad_scratch/tfn3_combo/w --banked F:/tanitad_scratch/tfn3_combo/run/banked $Extra >> `"$Run\logs\chain_stdout.log`" 2>&1 < NUL`""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = "F:\tanitad_scratch\tfn3_combo"; ProcessStartupInformation = (New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ CreateFlags = [uint32]0x08000200 }) }
"pid=$($r.ProcessId) rc=$($r.ReturnValue)"
