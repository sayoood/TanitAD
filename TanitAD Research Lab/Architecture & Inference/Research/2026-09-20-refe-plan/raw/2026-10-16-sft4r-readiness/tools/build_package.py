#!/usr/bin/env python3
"""build_package.py [--results-only] -- assemble raw/2026-10-16-sft4r-readiness/ (the deliverable package) from C:/s4r.

Writes to  E:/Projects/TanitAD/<PK>/raw/2026-10-16-sft4r-readiness/   (the repo worktree the Master Mind lands from; a NEW directory, nothing existing is overwritten or staged).
--results-only : just copy the test results (logs/q, SUMMARY.txt, TEST_TABLE.md, queue_done.flag) -- the detached queue calls this when it finishes.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import os
ROOT = Path(os.environ.get("S4R_ROOT", "C:/s4r"))
PKR = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
PKG = Path(os.environ.get("S4R_PKG") or (Path("E:/Projects/TanitAD") / PKR / "raw" / "2026-10-16-sft4r-readiness"))
FILES = ["launch_gate.py", "gate_checks.py", "sft4r_chain.sh", "sft4r_pod.sh", "test_sft4r_gate.py", "test_suite_nesting.py"]
TIP = "6c3d469cd92050f9e31eb6419dfb09a30a35ee05"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copyfile(src, dst)


def results() -> None:
    for name in ("SUMMARY.txt", "TEST_TABLE.md", "queue_done.flag"):
        if (ROOT / name).exists():
            cp(ROOT / name, PKG / "logs" / name)
    if (ROOT / "logs" / "q").exists():
        cp(ROOT / "logs" / "q", PKG / "logs" / "q")
    print("results copied to", PKG / "logs")
    # fill the generated table into READINESS.md between its markers (idempotent: the markers stay)
    tt, rd, flag = ROOT / "TEST_TABLE.md", PKG / "READINESS.md", ROOT / "queue_done.flag"
    if tt.exists() and rd.exists() and flag.exists():
        import time
        pairs, bad = flag.read_text(encoding="utf-8").split()[:2]
        s = open(rd, encoding="utf-8", newline="").read()
        a, b = s.find("<!--TEST_TABLE_BEGIN-->"), s.find("<!--TEST_TABLE_END-->")
        if a != -1 and b != -1:
            block = ("<!--TEST_TABLE_BEGIN-->\n"
                     f"**Generated {time.strftime('%Y-%m-%d %H:%M')} by `tools/run_all.py`: {pairs} (test, form) pairs, {bad} NOT clean.** Raw: `logs/SUMMARY.txt` (one line per run), `logs/q/<test>.<tree>.<form>.txt`.\n"
                     "Verdict words: PASS | PASS-ON-SERIAL-RERUN (failed under 4-way contention, passed alone) | PRE-EXISTING (fails on the unpatched tip tree too) | REGRESSION (passes unpatched) | NEW-TEST-FAILS.\n\n"
                     + tt.read_text(encoding="utf-8") + "\n")
            open(rd, "w", encoding="utf-8", newline="\n").write(s[:a] + block + s[b:])
            print("READINESS.md test table filled:", pairs, "pairs,", bad, "not clean")


def main() -> int:
    if "--results-only" in sys.argv:
        results()
        return 0
    refe_b, refe_p = ROOT / "b" / PKR / "refe", ROOT / "p" / PKR / "refe"
    PKG.mkdir(parents=True, exist_ok=True)
    # 1. the patchers, the apply script
    for f in (ROOT / "pkg" / "patch").glob("*.py"):
        cp(f, PKG / "patch" / f.name)
    cp(ROOT / "pkg" / "apply_all.py", PKG / "apply_all.py")
    # 2. post-images: FINAL (post) and the previous agent's v2 (post_p0) for the files that had no patcher
    for f in FILES:
        cp(refe_p / f, PKG / "post" / "refe" / f)
    for f in ("sft4r_chain.sh", "sft4r_pod.sh", "test_sft4r_gate.py"):
        cp(ROOT / "p0_refe" / f, PKG / "post_p0" / "refe" / f)
    # 3. sha table
    table = {"tip_commit": TIP, "note": "pre = sha256 of the file at the tip (null = the file is new); post = sha256 of the final file. LF bytes.",
             "pre": {f: (sha(refe_b / f) if (refe_b / f).exists() else None) for f in FILES}, "post": {f: sha(refe_p / f) for f in FILES}}
    (PKG / "SHA256.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    # 4. unified diffs tip -> final
    (PKG / "diff").mkdir(exist_ok=True)
    for stale in (PKG / "diff").glob("*.diff"):
        stale.unlink()

    import tempfile

    def udiff(a: Path, b: Path, out: Path) -> None:
        with tempfile.TemporaryDirectory() as td:                    # relative a/<name> b/<name> so the header is clean and the diff text is untouched
            (Path(td) / "a").mkdir()
            (Path(td) / "b").mkdir()
            shutil.copyfile(a, Path(td) / "a" / a.name)
            shutil.copyfile(b, Path(td) / "b" / b.name)
            r = subprocess.run(["git", "diff", "--no-index", "--no-color", "--no-prefix", "--", f"a/{a.name}", f"b/{b.name}"], capture_output=True, cwd=td)
        if r.stdout.strip():
            out.write_bytes(r.stdout.replace(b"\r\n", b"\n"))
    for f in FILES:                                                   # tip -> final (files that exist at the tip)
        if (refe_b / f).exists():
            udiff(refe_b / f, refe_p / f, PKG / "diff" / f"{f}.tip_to_final.diff")
    for f in ("launch_gate.py", "gate_checks.py", "sft4r_chain.sh", "sft4r_pod.sh", "test_sft4r_gate.py"):   # the previous agent's version -> final = exactly what THIS session changed
        udiff(ROOT / "p0_refe" / f, refe_p / f, PKG / "diff" / f"{f}.p0_to_final.diff")
    # 5. tools, evidence
    for f in ("bounded_run.py", "run_all.py", "treediff.py", "build_package.py", "verify_post.py"):
        cp(ROOT / "tools" / f, PKG / "tools" / f)
    ev = Path("C:/Users/Admin/sft4r_ready/raw/2026-10-16-sft4r-readiness/evidence")
    if ev.exists():
        cp(ev, PKG / "evidence")
    # 6. logs of the proofs
    for sub in ("repro", "e2e", "mut", "t1", "manual"):
        if (ROOT / "logs" / sub).exists():
            cp(ROOT / "logs" / sub, PKG / "logs" / sub)
    results()
    print("package at", PKG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
