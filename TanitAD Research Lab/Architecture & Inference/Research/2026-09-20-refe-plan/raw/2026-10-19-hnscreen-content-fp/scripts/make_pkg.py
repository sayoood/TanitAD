"""Assemble C:/Users/Admin/par18/hnfp2/out/ (the deliverable package) from the work tree + the evidence scripts/logs; write paths.json and the sha256 table.
Re-runnable: `python make_pkg.py` after the sibling suites have finished refreshes suite_compare.md and the logs, nothing else moves.  NO git add / commit / push."""
import difflib
import hashlib
import json
import os
import shutil
import sys

ROOT = "C:/Users/Admin/par18/hnfp2"
RP = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
WORK, BASE, OUT = f"{ROOT}/work/{RP}", f"{ROOT}/base/{RP}", f"{ROOT}/out"
TIP = "d2f4a062783f3fd9757ef10cc3266a13b42d8815"
CODE = ["refe/gate_token.py", "refe/launch_gate.py", "refe/ctx_cache.py", "refe/hn_fingerprints.py", "refe/test_launch_gate.py", "refe/test_hn_content_fp.py"]
EV = f"{RP}/raw/2026-10-19-hnscreen-content-fp"

shutil.rmtree(OUT, ignore_errors=True)
paths = []


def put(rel_repo, src_abs, dst_rel):
    dst = f"{OUT}/{dst_rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src_abs, dst)
    paths.append([rel_repo, dst, "raw"])
    return dst


# 1. the code, at its REAL repo paths (byte-exact LF)
sha = {}
for f in CODE:
    d = put(f"{RP}/{f}", f"{WORK}/{f}", f)
    b = open(d, "rb").read()
    assert b"\r" not in b, f
    sha[f] = hashlib.sha256(b).hexdigest()

# 2. the diff vs the tip (the base tree is the tip's bytes, exported with git cat-file --batch)
diff_lines = []
for f in CODE:
    a = open(f"{BASE}/{f}", "rb").read().decode("utf-8").splitlines(keepends=True) if os.path.isfile(f"{BASE}/{f}") else []
    b = open(f"{WORK}/{f}", "rb").read().decode("utf-8").splitlines(keepends=True)
    diff_lines += list(difflib.unified_diff(a, b, f"a/{f}", f"b/{f}", n=2))
open(f"{OUT}/raw/code_vs_tip.diff", "w", encoding="utf-8", newline="\n").writelines(diff_lines) if os.makedirs(f"{OUT}/raw", exist_ok=True) is None else None
paths.append([f"{EV}/code_vs_tip.diff", f"{OUT}/raw/code_vs_tip.diff", "raw"])
open(f"{OUT}/raw/TIP_SHA.txt", "w", newline="\n").write(f"built on tip {TIP} (agent/arch-inf-20260803, git fetch + git cat-file --batch export; the base tree under par18/hnfp2/base is byte-identical to that tip's refe/ eval/ code/ splits/ report/)\n")
paths.append([f"{EV}/TIP_SHA.txt", f"{OUT}/raw/TIP_SHA.txt", "raw"])

# 3. evidence: docs, scripts, logs
EVIDENCE = {
    "RESULT.md": f"{ROOT}/RESULT.md", "HANDOFF.md": f"{ROOT}/HANDOFF.md", "COST.md": f"{ROOT}/COST.md",
    "scripts/suite_runner.py": f"{ROOT}/suite_runner.py", "scripts/bench_cost.py": f"{ROOT}/bench_cost.py", "scripts/bench_cold.py": f"{ROOT}/bench_cold.py",
    "scripts/repro_same_size_flip.py": f"{ROOT}/repro_same_size_flip.py", "scripts/make_pkg.py": f"{ROOT}/make_pkg.py", "scripts/finish_pkg.py": f"{ROOT}/finish_pkg.py",
    "scripts/list_mine.txt": f"{ROOT}/list_mine.txt", "scripts/list_base.txt": f"{ROOT}/list_base.txt", "scripts/export.py": f"{ROOT}/export.py", "scripts/export_extra.py": f"{ROOT}/export_extra.py",
    "scripts/export_root.py": f"{ROOT}/export_root.py",
    "logs/test_hn_content_fp_FINAL.log": f"{ROOT}/logs/test_hn_content_fp_final.log",
    "logs/bench_cost.json": f"{ROOT}/logs/bench_cost.json", "logs/bench_cold_d.txt": f"{ROOT}/logs/bench_cold_d.txt",
    "logs/repro_same_size_flip_base.json": f"{ROOT}/logs/repro_same_size_flip_base.json", "logs/repro_same_size_flip_mine.json": f"{ROOT}/logs/repro_same_size_flip_work.json",
    "logs/suite_compare.md": f"{ROOT}/suite_compare.md", "logs/suite_mine.json": f"{ROOT}/suite_mine.json", "logs/suite_base.json": f"{ROOT}/suite_base.json",
    "logs/suite_extra_mine.json": f"{ROOT}/suite_extra_mine.json", "logs/suite_extra_base.json": f"{ROOT}/suite_extra_base.json",
    "logs/hnscreen_static_harness_mine.log": f"{ROOT}/logs/hnscreen_static_harness_mine.log", "logs/hnscreen_static_harness_base.log": f"{ROOT}/logs/hnscreen_static_harness_base.log",
}
missing = []
for rel, src in EVIDENCE.items():
    if os.path.isfile(src):
        put(f"{EV}/{rel}", src, f"raw/{rel}")
    else:
        missing.append(rel)
json.dump(paths, open(f"{OUT}/paths.json", "w", encoding="utf-8", newline="\n"), indent=1)
print("code sha256:")
for f, h in sha.items():
    print(f"  {h} *{f}")
print("paths.json entries:", len(paths), "| evidence files not (yet) present:", missing)
