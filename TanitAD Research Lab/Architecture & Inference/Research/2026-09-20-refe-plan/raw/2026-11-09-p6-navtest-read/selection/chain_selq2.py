#!/usr/bin/env python3
"""Resume of chain_selq.py after its score step ended rc 1: 60 of 64 columns stopped on the harness's own RAM guard (W3_RAM_GUARD_ABORT at 1.9 GB free, 5 workers).
Same tool, same seams, same arguments, 3 workers. Before each round the partial output of every column whose log is not PASS is removed (the harness has no handling for a leftover CSV).
Then table and read. Terminal marker: markers/selq2.DONE or markers/selq2.FAILED. CPU only."""
import ctypes
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import chain_p6 as C

W = C.W
MK = C.MK
LOG = W / "CHAIN_SELQ.log"
DATA = W / "selq_data"
DUMP = W / "pass2" / "proposals_p6final.npz"
CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_p6final/refe_p6final.csv"
NAME = "p6final"
SCORE = Path("F:/TanitAD_p6read_work/selq_data/score")
ROUNDS = 12
N_ROWS = 12146


def say(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%F %T')}] selq2: {m}\n")


def free_gb():
    class M(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
    m = M()
    m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2**30


def passed(k):
    lp = SCORE / f"_selq_{NAME}_logs" / f"p{k:02d}.log"
    if not lp.exists():
        return False
    for line in reversed(lp.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("{") and '"status"' in line:
            st = json.loads(line)
            return st.get("status") == "PASS" and st.get("csv_valid_rows") == N_ROWS
    return False


def clear_failed():
    n, size = 0, 0
    for k in range(64):
        if passed(k):
            continue
        d = Path(f"F:/TanitAD_p6read_work/selq_data/score/refe_selq_p6final_p{k:02d}")
        if d.is_dir():
            size += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
            n += 1
    say(f"cleared partial output of {n} columns, {size} B")


def main():
    for m in ("selq2.DONE", "selq2.FAILED"):
        if (MK / m).exists():
            raise SystemExit("marker exists: " + m)
    sys.path.insert(0, C.PKG + "/eval")
    import stage2_sft as S2
    env = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in S2.EC.env_driverl().items()}
    env.update(REFE_NAVTEST_DIR=str(DATA), PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
    py = S2.PY_D if hasattr(S2, "PY_D") else "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
    tool = C.PKG + "/eval/selection_quality.py"

    def run(name, args):
        if shutil.disk_usage("F:/").free < 4 * 2**30:
            raise RuntimeError("F: has under 4 GB free")
        t0 = time.time()
        say("step " + name)
        rc = subprocess.call([py, tool, name, *args], cwd=str(Path(tool).parent), env=env, stdout=open(W / "tactical" / f"{name}.log", "a", encoding="utf-8"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        say(f"step {name} rc {rc} after {round(time.time() - t0)} s; columns PASS {sum(passed(k) for k in range(64))}/64")
        return rc

    for r in range(ROUNDS):
        while free_gb() < 10.0:
            time.sleep(120)
        clear_failed()
        say(f"round {r + 1} of {ROUNDS}, free RAM {free_gb():.1f} GB")
        if run("score", ["--dump", str(DUMP), "--name", NAME, "--workers", "3"]) == 0:
            break
    else:
        raise RuntimeError(f"score not complete after {ROUNDS} rounds")
    table = W / "tactical" / "table_p6final.npz"
    for name, args in (("table", ["--dump", str(DUMP), "--name", NAME, "--arm-csv", CSV, "--out", str(table)]),
                       ("read", ["--table", str(table), "--arm-csv", CSV, "--out", str(W / "tactical" / "selection_quality_p6final.json")])):
        rc = run(name, args)
        if rc != 0:
            raise RuntimeError(f"step {name} rc {rc}")
    (MK / "selq2.DONE").write_text(json.dumps(dict(out=str(W / "tactical" / "selection_quality_p6final.json"))) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        say("stopped: " + tb[-1200:])
        (MK / "selq2.FAILED").write_text(tb[-3000:], encoding="utf-8")
        sys.exit(1)
