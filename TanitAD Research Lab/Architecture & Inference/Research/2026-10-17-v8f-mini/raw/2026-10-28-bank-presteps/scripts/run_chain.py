"""RAM-gated sequential runner (zero model tokens): each step starts only when free RAM >= 3 GB (poll every 5 s, give up after 40 min),
then runs `python -I <script> <args>` with CUDA_VISIBLE_DEVICES=-1. Writes raw/chain_<name>.log per step and raw/CHAIN_DONE_<name>.txt (rc) on exit.
Usage: python run_chain.py <chain_name> "<script.py arg arg>" "<script.py ...>" ...
A step that fails stops the chain (fail loud); nothing here deletes or moves anything.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

PKG = Path("E:/Projects/TanitAD-artifacts/dagland/bank_presteps")
SCRIPTS = PKG / "scripts"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"


def free_gb():
    out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
                         capture_output=True, text=True).stdout.strip().replace(",", ".")
    return float(out) / (1024 * 1024)


def main():
    name, steps = sys.argv[1], sys.argv[2:]
    (PKG / "raw").mkdir(exist_ok=True)
    log = open(PKG / "raw" / f"chain_{name}.log", "a", buffering=1)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    rc_all = 0
    for st in steps:
        parts = st.split()
        t_wait = time.time()
        while True:
            g = free_gb()
            if g >= 3.0:
                break
            if time.time() - t_wait > 2400:
                log.write(f"[{time.strftime('%H:%M:%S')}] GAVE UP waiting for RAM >= 3 GB before: {st} (last {g:.2f} GB)\n")
                (PKG / "raw" / f"CHAIN_DONE_{name}.txt").write_text("RAM_WAIT_TIMEOUT\n")
                return 3
            time.sleep(5)
        log.write(f"[{time.strftime('%H:%M:%S')}] START {st}  (free RAM {g:.2f} GB, waited {time.time() - t_wait:.0f} s)\n")
        p = subprocess.run([PY, "-I", str(SCRIPTS / parts[0])] + parts[1:], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(SCRIPTS))
        log.write(f"[{time.strftime('%H:%M:%S')}] END   {st}  rc={p.returncode}\n")
        if p.returncode != 0:
            rc_all = p.returncode
            break
    (PKG / "raw" / f"CHAIN_DONE_{name}.txt").write_text(f"rc={rc_all}\n")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
