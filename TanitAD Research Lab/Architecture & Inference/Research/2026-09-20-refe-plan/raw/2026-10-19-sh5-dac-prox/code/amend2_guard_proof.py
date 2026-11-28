"""Proof that the seal guard ACCEPTS the Amendment-2 package and REFUSES the pre-amendment analysis code (checker; not pinned).  CPU only; no data, no GPU.

    python amend2_guard_proof.py --root C:/Users/Admin/par18/sh5_a2 --old-lib <sh5_lib.py as of Amendment 1> --old-test <test_sh5_lib.py as of Amendment 1> --work D:/.../guard_proof --out raw/amend2_guard_proof.json

Each case builds a COPY of the package (prereg, amendments, pins, code) under --work, applies ONE change, and records (a) what `sh5_guard.require_sealed(copy)` says and (b) what the REAL stage entry points
(`sh5_train_eval.py analyze`, `sh5_train_eval.py fit`, which call `require_sealed()` first) do when started in that copy with dummy paths: a copy that passes the guard fails LATER, on the missing dummy input; a refused copy
fails at the guard with `NotSealed`, before any file is read.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_guard as G  # noqa: E402

PY = os.environ.get("SH5_PY", sys.executable)


def make_copy(root: str, dst: str) -> str:
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(f"{dst}/code")
    for n in os.listdir(root):
        if n.startswith(("PREREG_SH5_DAC_PROX", "CODE_PINS")) and n.endswith((".md", ".sha256", ".json")):
            shutil.copyfile(f"{root}/{n}", f"{dst}/{n}")
    for n in os.listdir(f"{root}/code"):
        if n.endswith((".py", ".ps1")):
            shutil.copyfile(f"{root}/code/{n}", f"{dst}/code/{n}")
    return dst


def guard(dst: str) -> dict:
    try:
        return {"accepted": True, "result": G.require_sealed(dst)}
    except G.NotSealed as e:
        return {"accepted": False, "refused_with": f"NotSealed: {e}"}
    except Exception as e:                                                                    # noqa: BLE001
        return {"accepted": False, "refused_with": f"{type(e).__name__}: {e}"}


def stage(dst: str, args: list) -> dict:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([PY, "-B", "-W", "ignore", "sh5_train_eval.py"] + args, cwd=f"{dst}/code", env=env, capture_output=True, text=True, timeout=240)
    tail = [l for l in (p.stderr or "").strip().splitlines() if l.strip()][-1:] or ["(no stderr)"]
    return {"cmd": "sh5_train_eval.py " + " ".join(args), "returncode": p.returncode, "last_stderr_line": tail[0][:300], "refused_by_guard": "NotSealed" in (p.stderr or "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--old-lib", required=True)
    ap.add_argument("--old-test", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dummy_analyze = ["analyze", "--data", "NO_SUCH_DIR", "--fits", "NO_SUCH_DIR", "--gates", "NO_SUCH.json", "--out-dir", "NO_SUCH_OUT"]
    dummy_fit = ["fit", "--data", "NO_SUCH_DIR", "--ctx", "NO_SUCH.npy", "--ctx-manifest", "NO_SUCH.json", "--seed", "0", "--stage", "A1", "--out-dir", "NO_SUCH_OUT"]
    cases = []

    def case(name, what, expect_accept, mutate=None, with_stage=True):
        d = make_copy(a.root, f"{a.work}/{name}")
        if mutate:
            mutate(d)
        g = guard(d)
        row = {"case": name, "change": what, "expected": "ACCEPT" if expect_accept else "REFUSE", "guard": g, "as_expected": bool(g["accepted"]) == expect_accept}
        if with_stage:
            row["stage_analyze"] = stage(d, dummy_analyze)
            row["stage_fit"] = stage(d, dummy_fit)
            for k in ("stage_analyze", "stage_fit"):
                row["as_expected"] = row["as_expected"] and (row[k]["refused_by_guard"] == (not expect_accept))
        cases.append(row)
        return d

    def put(path, src):
        def f(d):
            shutil.copyfile(src, f"{d}/{path}")
        return f

    def append(path, s):
        def f(d):
            with open(f"{d}/{path}", "ab") as fh:
                fh.write(s)
        return f

    def drop_amend2(d):
        for n in ("PREREG_SH5_DAC_PROX_AMENDMENT2.md", "PREREG_SH5_DAC_PROX_AMENDMENT2.sha256", "CODE_PINS_AMENDMENT2.json"):
            os.remove(f"{d}/{n}")

    def break_chain(d):
        p = f"{d}/CODE_PINS_AMENDMENT2.json"
        j = json.load(open(p, encoding="utf-8"))
        j["from"]["code/sh5_lib.py"] = "0" * 64
        json.dump(j, open(p, "w", encoding="utf-8"), indent=1)

    case("amended_package", "none: the Amendment-2 package as built", True)
    case("old_analysis_code", "code/sh5_lib.py := the pre-amendment (Amendment-1-state) file", False, put("code/sh5_lib.py", a.old_lib))
    case("old_tests", "code/test_sh5_lib.py := the pre-amendment (Amendment-1-state) file", False, put("code/test_sh5_lib.py", a.old_test))
    case("amended_code_without_its_amendment", "Amendment-2 md, sha256 and pins deleted (new code, no written amendment)", False, drop_amend2)
    case("amendment_text_edited", "one byte appended to PREREG_SH5_DAC_PROX_AMENDMENT2.md", False, append("PREREG_SH5_DAC_PROX_AMENDMENT2.md", b" "), with_stage=False)
    case("broken_chain", "CODE_PINS_AMENDMENT2.json `from` for code/sh5_lib.py is not the file's current pin", False, break_chain, with_stage=False)
    case("edit_after_amendment", "one byte appended to code/sh5_lib.py after Amendment 2", False, append("code/sh5_lib.py", b"\n"), with_stage=False)
    case("sealed_prereg_edited", "one byte appended to PREREG_SH5_DAC_PROX.md", False, append("PREREG_SH5_DAC_PROX.md", b" "), with_stage=False)
    out = {"purpose": "the guard accepts the Amendment-2 package and refuses the pre-amendment analysis code, at the function and at the real stage entry points", "root": a.root,
           "old_lib_sha256": G.sha256_file(a.old_lib), "old_test_sha256": G.sha256_file(a.old_test), "new_lib_sha256": G.sha256_file(f"{a.root}/code/sh5_lib.py"), "new_test_sha256": G.sha256_file(f"{a.root}/code/test_sh5_lib.py"),
           "all_as_expected": all(c["as_expected"] for c in cases), "cases": cases}
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    for c in cases:
        print(c["case"], "->", "ACCEPT" if c["guard"]["accepted"] else "REFUSE", "as_expected" if c["as_expected"] else "UNEXPECTED", "|", c["guard"].get("refused_with", "")[:110],
              "|", (c.get("stage_analyze") or {}).get("last_stderr_line", "")[:90])
    print("ALL_AS_EXPECTED", out["all_as_expected"])
    return 0 if out["all_as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
