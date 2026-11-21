import pathlib

body = [
    "@echo off",
    "set CUDA_VISIBLE_DEVICES=-1",
    "set PYTHONDONTWRITEBYTECODE=1",
    "set PYTHONIOENCODING=utf-8",
    r"C:\Users\Admin\venvs\tanitad\Scripts\python.exe C:\Users\Admin\dagread\scratch\kb45_chain.py > C:\Users\Admin\dagread\out\chain_console.out 2>&1",
    "",
]
p = pathlib.Path("C:/Users/Admin/dagread/scratch/run_kb45_chain.cmd")
with open(p, "w", encoding="ascii", newline="\r\n") as f:
    f.write("\n".join(body))
print(p.read_bytes())
