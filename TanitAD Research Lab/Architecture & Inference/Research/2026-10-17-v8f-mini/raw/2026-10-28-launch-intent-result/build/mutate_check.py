#!/usr/bin/env python
"""mutate_check.py: prove check_result.py has teeth.

Each mutant is a copy of the finished document with one defect.  check_result.py
must exit non-zero on every mutant and exit 0 on the unmodified document.
Mutants are written under ./mut_tmp/ and removed at the end.
"""
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOC = os.path.join(ROOT, "RESULT_V8F_LAUNCH_INTENT.md")
CHK = os.path.join(ROOT, "check_result.py")
TMP = os.path.join(ROOT, "mut_tmp")
PY = sys.executable

src = open(DOC, "rb").read().decode("utf-8")


def sub(a, b, count=1):
    assert src.count(a) >= 1, "mutation anchor missing: %r" % a
    return src.replace(a, b, count)


MUTANTS = [
    ("table cell digit: ADE6 s0 replicate +0.0431 -> +0.0413", sub("**+0.0431 [+0.0137, +0.0711] W**", "**+0.0413 [+0.0137, +0.0711] W**")),
    ("table flag: ADE6 s0 main B -> W", sub("**-0.0294 [-0.0522, -0.0077] B**", "**-0.0294 [-0.0522, -0.0077] W**")),
    ("table CI bound: L2 s0 replicate hi -0.0048 -> -0.0084", sub("**-0.0215 [-0.0423, -0.0048] B**", "**-0.0215 [-0.0423, -0.0084] B**")),
    ("bar in the floors table: yaw-rate 1.1070 -> 1.1700", sub("**1.1070**", "**1.1700**")),
    ("outcome table: R-ADE s1 main FAILS -> held", sub("FAILS: +0.0483 W", "held: +0.0483 W")),
    ("R-MOVE count: 1 of 4 rows clear -> 2 of 4 rows clear", sub("false: 1 of 4 rows clear", "false: 2 of 4 rows clear")),
    ("prose number: largest slow shift 0.1627 -> 0.1672", sub("is 0.1627 m/s", "is 0.1672 m/s")),
    ("prose count: 2 of the 16 crawl cells -> 3 of the 16", sub("2 of the 16 crawl cells", "3 of the 16 crawl cells")),
    ("prose cell: summary L2 cell digit", sub("L2 curvature in the replicate pair at s0, -0.0215 [-0.0423, -0.0048] B", "L2 curvature in the replicate pair at s0, -0.0251 [-0.0423, -0.0048] B")),
    ("hash prefix: 469bac1f -> 469bac1e", sub("`469bac1f…`", "`469bac1e…`")),
    ("provenance sha256 digit", sub("`e451e1a6ba44e1d30e97561456476dc4223cd916bd3be96163cec4b02844bf23`", "`e451e1a6ba44e1d30e97561456476dc4223cd916bd3be96163cec4b02844bf24`")),
    ("banned word added", sub("Nothing below softens that.", "Nothing below softens that. This is no longer open.")),
    ("a second slot token", sub("(until M-LAUNCH lands this reads NOT BUILT)", "(until M-LAUNCH lands this reads NOT BUILT; @@MLAUNCH@@)")),
    ("slot removed", sub("**OPEN SLOT @@MLAUNCH@@**", "**OPEN SLOT**")),
    ("time: landed 15:24:46 -> 15:42:46", sub("(15:24:46;", "(15:42:46;")),
    ("time: box-2 span 13:54 -> 13:45", sub("First stamped line 13:54", "First stamped line 13:45")),
    ("quote altered", sub("PENDING: train B", "PENDING: train C")),
    ("closed loop without negation", sub("T1-kin is a self-action open loop:", "T1-kin is a closed loop rollout:")),
    ("CRLF line ending", src.replace("\n", "\r\n", 1)),
    ("K-SHARE share digit", sub("0.331451", "0.331415")),
    ("checkpoint prefix", sub("`deb8f28b4ea9`", "`deb8f28b4eaa`")),
    ("family-table n field", sub("2399/141 ; 2399/141 ; 2399/141 ; 2399/141", "2399/141 ; 2399/141 ; 2399/141 ; 2399/142", 1)),
    ("P10 table cell", sub("**+0.0847 [+0.0549, +0.1200] W**", "**+0.0847 [+0.0549, +0.1002] W**")),
    ("derived percentage: 158.1 % -> 185.1 %", sub("158.1 %", "185.1 %")),
    ("prose number not in any source: 0.4321 m", sub("The sizes are small.", "The sizes are small (0.4321 m).")),
]

if os.path.isdir(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP)
rc0 = subprocess.run([PY, "-X", "utf8", CHK, "--doc", DOC], capture_output=True).returncode
print("unmodified document: rc=%d (want 0)" % rc0)
killed = 0
survivors = []
for k, (name, text) in enumerate(MUTANTS):
    path = os.path.join(TMP, "m%02d.md" % k)
    with open(path, "wb") as f:
        f.write(text.encode("utf-8"))
    r = subprocess.run([PY, "-X", "utf8", CHK, "--doc", path], capture_output=True)
    failed = [ln for ln in r.stdout.decode("utf-8", "replace").splitlines() if ln.startswith("  FAILED:")]
    if r.returncode != 0:
        killed += 1
        print("KILLED  m%02d %-70s rc=%d first: %s" % (k, name, r.returncode, (failed[0][:110] if failed else "")))
    else:
        survivors.append(name)
        print("SURVIVED m%02d %s" % (k, name))
shutil.rmtree(TMP)
print("mutants: %d, killed: %d, survived: %d" % (len(MUTANTS), killed, len(survivors)))
sys.exit(0 if (rc0 == 0 and not survivors) else 1)
