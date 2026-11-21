"""Print the number of pytest RUNS (a venv launcher + its child count as ONE run): python processes running "-m pytest" whose parent is not one."""
import subprocess

q = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match ' -m pytest' } | "
     "ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }")
out = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True).stdout.split()
pairs = [(int(out[i]), int(out[i + 1])) for i in range(0, len(out) - 1, 2)]
pids = {p for p, _ in pairs}
print(sum(1 for _, pp in pairs if pp not in pids))
