#!/usr/bin/env python3
"""What do the DAG-OACC and FLAT-DAG queues READ from the repo?  The DAG2 build's static import closure (E:/Projects/TanitAD-artifacts/dag2q/export_closure.py, imported, unchanged) over the entry points each chain runs on box 2,
computed on the python files of the tip 1d4f4fbb (exported from the mirror with `git cat-file --batch`: no autocrlf), plus the repo DATA paths those programs open (dag2's list; DAG-OACC adds HBLEND's fit_clips.json that `waccdag` copies).

    python export_closure_oaf.py PYTREE_DIR      (PYTREE_DIR is created if absent: the .py files of stack/ and taniteval/ at the tip, ~10 MB, on the C: scratchpad)
writes export_closure_dagoacc.json and export_closure_flatdag.json beside this file: {"files": [...], "data": [...], ...}
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIRROR, TIP = "C:/Users/Admin/tanitad-push", "1d4f4fbb"
spec = importlib.util.spec_from_file_location("ec", "E:/Projects/TanitAD-artifacts/dag2q/export_closure.py")
EC = importlib.util.module_from_spec(spec); spec.loader.exec_module(EC)


def git(*a, inp=None):
    return subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", *a], capture_output=True, check=True, input=inp).stdout


def export_py(dst: Path):
    names = [x for x in git("ls-tree", "-r", "--name-only", TIP, "--", "stack", "taniteval").decode("utf-8").split("\n") if x.endswith(".py") and "/tests/" not in x]
    ids = git("ls-tree", "-r", TIP, "--", "stack", "taniteval").decode("utf-8").split("\n")
    byname = {l.split("\t", 1)[1]: l.split()[2] for l in ids if "\t" in l}
    req = "".join(byname[n] + "\n" for n in names).encode()
    out = git("cat-file", "--batch", inp=req)
    k = 0
    for n in names:
        hdr_end = out.index(b"\n", k)
        size = int(out[k:hdr_end].split()[2])
        body = out[hdr_end + 1:hdr_end + 1 + size]
        k = hdr_end + 1 + size + 1
        p = dst / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    return len(names)


def main():
    root = Path(sys.argv[1])
    if not (root / "stack").is_dir():
        print("exported", export_py(root), "python files to", root)
    for name, extra_entries, extra_data in (("dagoacc", ["stack/scripts/v8f_mini_eval_oacc.py"], [f"{EC.PK}/raw/2026-10-25-hblend/fit_clips.json"]), ("flatdag", [], [])):
        EC.ENTRIES = list(EC.ENTRIES[:4]) + extra_entries
        files, external, relative = EC.closure(root)
        data = [p for p, _, _ in EC.DATA_READS] + extra_data
        size = sum((root / f).stat().st_size for f in files)
        (HERE / f"export_closure_{name}.json").write_text(json.dumps({"entries": EC.ENTRIES, "n_files": len(files), "bytes": size, "files": files, "data": data, "external_top_level_modules": external}, indent=1), encoding="utf-8")
        print(f"{name}: {len(files)} python files {size / 1e6:.2f} MB + {len(data)} data paths; external {external}")


if __name__ == "__main__":
    main()
