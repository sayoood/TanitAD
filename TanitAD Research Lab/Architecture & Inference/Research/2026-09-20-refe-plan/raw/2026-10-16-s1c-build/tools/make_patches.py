"""Build PATCH_n_<file>.diff (against the tip df9871e3) + patched/ copies + apply proof.  usage: make_patches.py <work root> <base root> <out dir>"""
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

W, B, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
FILES = [  # (n, path relative to the plan root, what it is)
    (1, "refe/launch_gate.py", "profile refe_aux_s1c: per-budget arm classes, argv recipe, must-name flags, schedule_reach, the S1c suite"),
    (2, "refe/gate_checks.py", "G-DVB schedule-reach rows (--max-updates must be REACHED by the epoch budget; closes TL-1 for the S1c levers)"),
    (3, "refe/s1aux_checks.py", "run-check at the S1c budgets (S1C_BUDGETS: update counts, eval tags, dumps, lr ends at 0 at update N)"),
    (4, "refe/s1aux_pod.sh", "S1c arms + per-arm gate profile + CR-stripped navtest pin + lock-only RL-S1 dependency (S1c-only) + small dd (S1c-only)"),
    (5, "refe/test_s1c.py", "NEW: 224 checks - tables, real gate on the chain's own argv, G-DVB, run-check, chain static, upstream cases, navtest pin, reachability + RED mutants"),
    (6, "refe/test_s1aux_chain.py", "static checks follow the chain ($PROFILE, ARM_ORDER_S1, arm_profile) + S1c mutants + token_ok per profile"),
    (7, "refe/test_aux_drivable_head.py", "E10 identifies the navtest list by its CR-stripped sha256 (CRLF checkout and LF ship are the same list)"),
    (8, "eval/aux_s1_readout.py", "ARM_RE accepts the S1c head arms; they are never paired with a 400-update N arm"),
    (9, "eval/test_aux_s1_readout.py", "S1c parse/pairing asserts + 1 RED mutation"),
    (10, "eval/aux_navtest_forward.py", "ARM_RE accepts the S1c head arms"),
    (11, "eval/test_aux_navtest_forward.py", "S1c names accepted / near-misses refused (the suite needs the D: checkout to run; the changed function was exercised directly)"),
]
os.makedirs(OUT, exist_ok=True)
manifest = []
for n, rel, what in FILES:
    w = os.path.join(W, rel)
    b = os.path.join(B, rel)
    wt = open(w, encoding="utf-8", newline="").read().splitlines(keepends=True)
    bt = open(b, encoding="utf-8", newline="").read().splitlines(keepends=True) if os.path.exists(b) else None
    if bt is None:
        d = list(difflib.unified_diff([], wt, "/dev/null", f"b/{rel}", n=3))
    else:
        d = list(difflib.unified_diff(bt, wt, f"a/{rel}", f"b/{rel}", n=3))
    d = [x if x.endswith("\n") else x + "\n\\ No newline at end of file\n" for x in d]
    name = f"PATCH_{n}_{os.path.basename(rel)}.diff"
    open(os.path.join(OUT, name), "w", encoding="utf-8", newline="").write("".join(d))
    dst = os.path.join(OUT, "patched", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(w, dst)
    add = sum(1 for x in d if x.startswith("+") and not x.startswith("+++"))
    rem = sum(1 for x in d if x.startswith("-") and not x.startswith("---"))
    manifest.append({"n": n, "file": rel, "diff": name, "what": what, "new_file": bt is None, "lines_added": add, "lines_removed": rem,
                     "sha256_patched": hashlib.sha256(open(w, "rb").read()).hexdigest(), "sha256_base": None if bt is None else hashlib.sha256(open(b, "rb").read()).hexdigest()})
# ---- apply proof: a scratch copy of the BASE files, every diff applied in order with `git apply`, bytes compared with the patched copies
tmp = tempfile.mkdtemp()
for m in manifest:
    if not m["new_file"]:
        dst = os.path.join(tmp, m["file"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(B, m["file"]), dst)
proof = []
for m in manifest:
    p = os.path.join(OUT, m["diff"])
    r = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "apply", "--check", "--whitespace=nowarn", p], cwd=tmp, capture_output=True, text=True)
    r2 = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "apply", "--whitespace=nowarn", p], cwd=tmp, capture_output=True, text=True)
    same = hashlib.sha256(open(os.path.join(tmp, m["file"]), "rb").read()).hexdigest() == m["sha256_patched"]
    proof.append({"diff": m["diff"], "git_apply_check_rc": r.returncode, "git_apply_rc": r2.returncode, "result_equals_patched_copy": same, "stderr": (r.stderr + r2.stderr).strip()[:300]})
    print(m["diff"], "check rc", r.returncode, "apply rc", r2.returncode, "bytes equal:", same, f"(+{m['lines_added']} -{m['lines_removed']})")
json.dump({"base": "origin/agent/arch-inf-20260803 @ df9871e3de11 (refe/ and eval/ unchanged at afbc83508e7f)", "patches": manifest, "apply_proof": proof}, open(os.path.join(OUT, "PATCHES.json"), "w", encoding="utf-8"), indent=1)
shutil.rmtree(tmp, ignore_errors=True)
