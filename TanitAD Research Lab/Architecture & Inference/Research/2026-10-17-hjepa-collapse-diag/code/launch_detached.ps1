# Launch hj_collapse_diag.py DETACHED (WMI Win32_Process.Create), so it survives the session.
# Usage:  powershell -File launch_detached.ps1 -Work C:\Users\Admin\hjcollapse_work [-Only 0] [-Tag run]
# The job is RESUMABLE per episode (ep\<clip>.json is the commit marker): re-running the same
# command continues where it stopped.  Log: <Work>\<Tag>.log
param(
  [string]$Work = "C:\Users\Admin\hjcollapse_work",
  [string]$Tag = "run",
  [string]$Only = "",
  [string]$EpisodesN = "40"
)
$repo = "E:\Projects\TanitAD"
$pkg  = "$repo\TanitAD Research Lab\Architecture & Inference\Research\2026-10-17-hjepa-collapse-diag"
New-Item -ItemType Directory -Force $Work | Out-Null
$log = "$Work\$Tag.log"
$onlyArg = ""
if ($Only -ne "") { $onlyArg = "--only $Only" }
$py = "C:\Users\Admin\venvs\tanitad\Scripts\python.exe"
$cmd = "cmd /c ""set PYTHONPATH=$repo\stack;$repo\taniteval&& set PYTHONIOENCODING=utf-8&& set OMP_NUM_THREADS=6&& set MKL_NUM_THREADS=6&& ""$py"" -u ""$pkg\code\hj_collapse_diag.py"" --out-dir ""$Work"" --episodes-n $EpisodesN $onlyArg --repo-raw ""$pkg\raw"" >> ""$log"" 2>&1"""
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $repo }
"launched pid=$($r.ProcessId) rc=$($r.ReturnValue) log=$log"
