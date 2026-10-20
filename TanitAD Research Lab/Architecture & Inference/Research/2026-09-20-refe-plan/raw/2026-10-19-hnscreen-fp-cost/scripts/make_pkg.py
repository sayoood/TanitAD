"""Assemble C:/Users/Admin/par18/hnfp3/out/ (the deliverable package) from the work tree + the evidence scripts/logs; write paths.json.
Re-runnable.  NO git add / commit / push (staging in out/ + paths.json REPLACES git add under this brief).  usage: python make_pkg.py"""
import difflib
import glob
import hashlib
import json
import os
import shutil

ROOT = "C:/Users/Admin/par18/hnfp3"
RP = "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
WORK, BASE, OUT = f"{ROOT}/work/{RP}", f"{ROOT}/base/{RP}", f"{ROOT}/out"
TIP = open(f"{ROOT}/TIP_SHA.txt").read().strip()
CODE = ["refe/gate_token.py", "refe/launch_gate.py", "refe/hnscreen_devbox.py", "eval/hn_screen_read.py", "refe/test_hn_ctx_anchor.py", "refe/test_hnscreen_ctx_chain.py", "refe/test_hnscreen_chain.py",
        "eval/test_hn_screen_read.py", "refe/test_hn_content_fp.py"]
EV = f"{RP}/raw/2026-10-19-hnscreen-fp-cost"

shutil.rmtree(OUT, ignore_errors=True)
paths = []


def put(rel_repo, src_abs, dst_rel):
    dst = f"{OUT}/{dst_rel}"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src_abs, dst)
    paths.append([rel_repo, dst, "raw"])
    return dst


def latest(pattern):
    c = sorted(glob.glob(f"{ROOT}/{pattern}"), key=os.path.getmtime)
    return c[-1] if c else None


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
os.makedirs(f"{OUT}/raw", exist_ok=True)
open(f"{OUT}/raw/code_vs_tip.diff", "w", encoding="utf-8", newline="\n").writelines(diff_lines)
paths.append([f"{EV}/code_vs_tip.diff", f"{OUT}/raw/code_vs_tip.diff", "raw"])
open(f"{OUT}/raw/TIP_SHA.txt", "w", newline="\n").write(f"built on tip {TIP} (agent/arch-inf-20260803; git cat-file --batch export; par18/hnfp3/base is byte-identical to that tip's refe/ eval/ code/ splits/ report/)\n")
paths.append([f"{EV}/TIP_SHA.txt", f"{OUT}/raw/TIP_SHA.txt", "raw"])

# 3. evidence: docs, scripts, logs
EVIDENCE = {
    "RESULT.md": f"{ROOT}/RESULT.md", "CALLMAP.md": f"{ROOT}/CALLMAP.md", "MM_RULING_FPCOST.md": f"{ROOT}/MM_RULING_FPCOST.md", "BRIEF.md": f"{ROOT}/BRIEF.md", "HANDOFF.md": f"{ROOT}/HANDOFF.md",
    "MUTANTS.md": f"{ROOT}/MUTANTS.md",
    "scripts/suite_runner.py": f"{ROOT}/scripts/suite_runner.py", "scripts/make_pkg.py": f"{ROOT}/scripts/make_pkg.py", "scripts/finish_pkg.py": f"{ROOT}/scripts/finish_pkg.py",
    "scripts/mutant_table.py": f"{ROOT}/scripts/mutant_table.py", "scripts/measure_reads.py": f"{ROOT}/scripts/measure_reads.py", "scripts/r7_identity_vs_tip.py": f"{ROOT}/scripts/r7_identity_vs_tip.py",
    "scripts/r7_worker.py": f"{ROOT}/scripts/r7_worker.py", "scripts/r7_control.py": f"{ROOT}/scripts/r7_control.py", "scripts/ramgate.py": f"{ROOT}/scripts/ramgate.py",
    "scripts/list_mine.txt": f"{ROOT}/list_mine.txt", "scripts/list_base.txt": f"{ROOT}/list_base.txt", "scripts/export_both.py": f"{ROOT}/scripts/export_both.py",
    "scripts/export_extra.py": f"{ROOT}/scripts/export_extra.py", "scripts/rls1_queue_race_probe.py": f"{ROOT}/scripts/rls1_queue_race_probe.py", "scripts/patch_fp_table.py": f"{ROOT}/scripts/patch_fp_table.py", "FINISH_NOTES.md": f"{ROOT}/FINISH_NOTES.md", "scripts/run_finish_matrices.sh": f"{ROOT}/scripts/run_finish_matrices.sh", "logs/suite_extra_mine.json": f"{ROOT}/suite_extra_mine.json", "scripts/export_root.py": f"{ROOT}/scripts/export_root.py",
    "logs/reads_per_screen.json": f"{ROOT}/logs/reads_per_screen.json",
    "logs/r7_identity_vs_tip.json": f"{ROOT}/logs/r7_identity_vs_tip.json", "logs/r7_control.json": f"{ROOT}/logs/r7_control.json",
    "logs/suite_compare.md": f"{ROOT}/suite_compare.md", "logs/suite_mine.json": f"{ROOT}/suite_mine.json", "logs/suite_base.json": f"{ROOT}/suite_base.json",
}
for rel, pat in {"logs/test_hn_ctx_anchor_FINAL.log": "logs/test_hn_ctx_anchor_matrix_run*.log", "logs/test_hnscreen_ctx_chain_FINAL.log": "logs/test_hnscreen_ctx_chain_matrix_run*.log",
                 "logs/test_hnscreen_ctx_chain_arms_FINAL.log": "logs/test_hnscreen_ctx_chain_run*.log", "logs/test_hn_screen_read_FINAL.log": "logs/test_hn_screen_read_run*.log",
                 "logs/measure_reads_FINAL.log": "logs/measure_reads_run*.log", "logs/r7_all_FINAL.log": "logs/r7_all_run*.log", "logs/test_hn_content_fp_FINAL.log": "logs/test_hn_content_fp_run*.log"}.items():
    src = latest(pat)
    if src:
        EVIDENCE[rel] = src
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

shutil.copyfile(f"{ROOT}/RESULT.md", f"{OUT}/RESULT.md")          # common.md: out/RESULT.md (not an entry of paths.json: the MM reads it in place)
