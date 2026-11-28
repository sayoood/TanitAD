"""Independent checks of report v46 (the patch's own assertions have already passed to produce out/). python -I checks_v46.py tip [out_dir [checks_path]]"""
import hashlib, os, re, subprocess, sys, shutil
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v46_checks.txt"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
BANNED = tuple(w[::-1] for w in ("diov", "dedesrepus", "detcarter", "ylremrof", "regnol on", "noitcerroc", "nwardhtiw"))
BANRE = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)
L, FAILS = [], []


def chk(name, cond, extra=""):
    L.append(("PASS " if cond else "FAIL ") + name + ((" " + extra) if extra else ""))
    if not cond:
        FAILS.append(name)


def git(args):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout


new = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
html = open(OUT + "v8f_report.html", encoding="utf-8", newline="").read()
bb = open("E:/Projects/TanitAD-artifacts/dagland/report_v45/out/V8F_SCIENTIFIC_REPORT.md", "rb").read()
chk("base is the v45 md (sha256 324c2001...)", hashlib.sha256(bb).hexdigest().startswith("324c200158fe27de"))
base = bb.decode("utf-8")
tpl = git(["show", f"{TIP}:{R}report/2026-10-26/page_template_v30.html"]).decode("utf-8")
chk("html is the page template with the v46 md", html == tpl.replace("%%MD%%", new))
chk("LF only (md, html)", "\r" not in new and "\r" not in html)
chk("status line is v46 and names the tip prefix", "**Status:** research report, v46 of 2026-11-10, results through repository tip `%s`" % TIP[:8] in new)
live = subprocess.run(["git", "-C", M, "-c", "safe.directory=*", "ls-remote", "origin", "refs/heads/agent/arch-inf-20260803"], capture_output=True).stdout.decode().split("\t")[0]
L.append("NOTE live remote tip at check time: %s (%s)" % (live[:8], "equals the build tip" if live == TIP else "differs from the build tip"))
cb, cn = Counter(base.split("\n")), Counter(new.split("\n"))
added = [l for l in (cn - cb).elements() if l.strip()]
removed = [l for l in (cb - cn).elements() if l.strip()]
L.append("added lines %d, removed (edited carried) lines %d" % (len(added), len(removed)))
chk("only the carried lines of edits_v46.md were changed: exactly 5 removed lines (status line, abstract What-is-next line, row 1bz, the Update (v45) blocked bullet, item 23 Not run bullet)", len(removed) == 5, "removed: " + " || ".join(x[:50] for x in removed))
chk("every removed line is one that edits_v46.md rewrites", all(re.search(r"v45 of 2026-11-08|\*\*What is next\.\*\*|^\| 1bz \||\*\*What is blocked:\*\*|\*\*Not run\.\*\* Every item", x) for x in removed))
A = "\n".join(added)
chk("no banned word in the added lines", not BANRE.search(A), str(BANRE.search(A)))
chk("no closed loop wording in the added lines", not re.search(r"closed[- ]loop", A, re.I))
chk("no en or em dash in the added lines", not re.search(r"[\u2013\u2014]", A))
chk("no drives / driving performance wording (one negated phrase in 7.51 (i) allowed)", not re.search(r"\bdrives\b|driving performance", A.lower().replace("nothing here is driving performance", "")))
chk("no unfilled placeholder or directive", "{{" not in new and "@@" not in new and "<<" not in A)
sec = new[new.index("### 7.51 "):new.index("## 8. Discussion")]
chk("sections 7.51 and 7.52 carry no pending wording for a read that landed", not re.search(r"PENDING|has no result and none is quoted", sec))
chk("STRATEGIC is PAUSED and never a pass", "PAUSED" in sec and not re.search(r"STRATEGIC[^.\n]{0,40}\bPASS\b", sec))
chk("tiers named: T0 (7.51) and T1-kin (7.52), no driving result claim", "**Tier T0 throughout" in sec and "**Tier T1-kin throughout" in sec and "never driving results" in sec)
chk("outcome words as the result files write them: NO-GAIN, BAND-MET, BAND-COST", all(x in sec for x in ("NO-GAIN", "BAND-MET", "BAND-COST")))
chk("Nothing in v46 is a driving result. in the status line; No driving result, no DOMINANT closes item 24", "Nothing in v46 is a driving result." in new.split("\n")[2] and "* No driving result, no DOMINANT and no scale signal exist.\n\n---\n\n## 10. Reproducibility appendix" in new)
chk("sections 7.51 and 7.52 are in the built report verbatim (the two section files)", all(open(HERE + f, encoding="utf-8", newline="").read().strip("\n") in new for f in ("sec_751_planner_gates_ladder.md", "sec_752_otband.md")))
chk("section order 7.50.8 < 7.51 < 7.52 < 8", new.index("**7.50.8 ") < new.index("### 7.51 ") < new.index("### 7.52 ") < new.index("## 8. Discussion"))
chk("8.4 rows 1cb to 1ce and the appendix block are present", all(("| 1c%s |" % c) in new for c in "bcde") and "**Reads added in v46 (tip `%s`)" % TIP[:8] in new and "| `2026-11-09-v8f-otband/read/` |" in new)
chk("section 9 item 24, 8.1 item 36 and Update (v46) are present", "\n24. **v46 reads and their limits.**" in new and "\n36. **The planner gates, the band-trained ladder and the band loss**" in new and "\n**Update (v46).**" in new)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
r = subprocess.run([PY, "-I", HERE + "check_numbers_v46.py"], cwd=HERE, capture_output=True, env=env)
last = r.stdout.decode("utf-8", "replace").strip().split("\n")[-1]
chk("check_numbers_v46.py exits 0", r.returncode == 0, last)
tmp = "F:/TanitAD-scratch/report_v46_build/mut/m/"
os.makedirs(tmp, exist_ok=True)
for f in ("check_numbers_v46.py", "sec_751_planner_gates_ladder.md", "sec_752_otband.md", "edits_v46.md"):
    shutil.copy(HERE + f, tmp + f)
a = open(tmp + "sec_752_otband.md", encoding="utf-8", newline="").read()
b = open(tmp + "sec_751_planner_gates_ladder.md", encoding="utf-8", newline="").read()
a2 = a.replace("-0.0168 [-0.0275, -0.0098]", "-0.0169 [-0.0275, -0.0098]", 1)
b2 = b.replace("0.546 / 0.555", "0.547 / 0.555", 1) if "0.546 / 0.555" in b else b.replace("0.546", "0.547", 1)
chk("the mutation changed two numbers (one OTBAND cell, one ladder cell)", a2 != a and b2 != b)
open(tmp + "sec_752_otband.md", "w", encoding="utf-8", newline="\n").write(a2)
open(tmp + "sec_751_planner_gates_ladder.md", "w", encoding="utf-8", newline="\n").write(b2)
r2 = subprocess.run([PY, "-I", tmp + "check_numbers_v46.py"], cwd=tmp, capture_output=True, env=env)
chk("check_numbers_v46.py FAILS on a copy with two numbers changed (mutation arm)", r2.returncode != 0, r2.stdout.decode("utf-8", "replace").strip().split("\n")[-1][:80])
shutil.rmtree("F:/TanitAD-scratch/report_v46_build/mut", ignore_errors=True)
npass = sum(1 for x in L if x.startswith("PASS"))
L += ["", "checks: %d PASS, %d FAIL" % (npass, len(FAILS))]
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L[-3:]))
sys.exit(1 if FAILS else 0)
