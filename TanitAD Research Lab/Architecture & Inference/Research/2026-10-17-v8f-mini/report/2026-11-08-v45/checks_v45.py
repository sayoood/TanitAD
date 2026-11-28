"""Independent checks of report v45 (the patch's own assertions have already passed to produce out/). Writes the checks file.
python -I checks_v45.py tip [out_dir [checks_path]]   (git, python; no network except git ls-remote)
Includes: the base is the landed v45; only the carried lines of edits_v45.md differ from it (the removed lines are counted by site); banned words, dashes and pending wording in the added lines; the status line;
the two section files appear verbatim in the built report; check_numbers_v45.py (every headline number of the new text against its source file, and the tip for PLANTEST and the build facts) and a mutation run
of it on a copy of the folder with two numbers changed (it must FAIL)."""
import hashlib, os, re, subprocess, sys, shutil
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v45_checks.txt"
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
base = git(["show", f"{TIP}:{R}report/2026-11-07-v44/V8F_SCIENTIFIC_REPORT.md"])
chk("base is the landed v44 md (sha256 249a02d9...)", hashlib.sha256(base).hexdigest().startswith("249a02d9e11a17e8"))
base = base.decode("utf-8")
tpl = git(["show", f"{TIP}:{R}report/2026-10-26/page_template_v30.html"]).decode("utf-8")
chk("html is the page template with the v45 md", html == tpl.replace("%%MD%%", new))
chk("LF only (md, html)", "\r" not in new and "\r" not in html)
chk("status line is v45 and names the tip prefix", "**Status:** research report, v45 of 2026-11-08, results through repository tip `%s`" % TIP[:8] in new)
live = subprocess.run(["git", "-C", M, "-c", "safe.directory=*", "ls-remote", "origin", "refs/heads/agent/arch-inf-20260803"], capture_output=True).stdout.decode().split("\t")[0]
chk("the build tip is the tip the Master Mind named (c69bc890...)", TIP == "c69bc890923029d58f67ecd2c4e984c8a823bcaa")
L.append("NOTE live remote tip at check time: %s (%s)" % (live[:8], "equals the build tip" if live == TIP else "differs from the build tip: the remote moved after the named tip; the report is built at the named tip"))
cb, cn = Counter(base.split("\n")), Counter(new.split("\n"))
added = [l for l in (cn - cb).elements() if l.strip()]
removed = [l for l in (cb - cn).elements() if l.strip()]
L.append("added lines %d, removed (edited carried) lines %d" % (len(added), len(removed)))
chk("only the carried lines of edits_v45.md were changed (removed lines at most 8: the status line, the NXT abstract line, the IMPL abstract bullet, the item 34 result line, row 1bu, items 22 and the Update (v44) line)", len(removed) <= 8, "removed: " + " || ".join(x[:50] for x in removed))
chk("every removed line is a status line, an abstract / 8.1 / 8.3 / 8.4 / 9 line that edits_v45.md rewrites", all(re.search(r"v44 of 2026-11-07|has two experiments pre-registered|no IMAG1 or VOE1 number exists|IMAG1 and VOE1 are pre-registered|has IMAG1 and VOE1 pre-registered|imagination plan \(IMAG1, VOE1\) has no result", x) for x in removed))
A = "\n".join(added)
chk("no banned word in the added lines", not BANRE.search(A), str(BANRE.search(A)))
chk("no closed loop wording in the added lines", not re.search(r"closed[- ]loop", A, re.I))
chk("no en or em dash in the added lines", not re.search(r"[\u2013\u2014]", A))
chk("no drives / driving performance wording in the added lines", not re.search(r"\bdrives\b|driving performance", A.lower()))
chk("no unfilled placeholder or directive", "{{" not in new and "@@" not in new and "<<" not in A)
sec = new[new.index("### 7.49 "):new.index("## 8. Discussion")]
chk("sections 7.49 and 7.50 carry no pending wording for a read that landed", not re.search(r"PENDING|running with no result|has no result and none is quoted|is running\b", sec.replace("the register records them as running", "")), "")
chk("STRATEGIC is PAUSED in 7.49 and 7.50 and never a pass", "PAUSED" in sec and not re.search(r"STRATEGIC[^.\n]{0,40}\bPASS\b", sec))
chk("evidence tiers named: T0 and T1-kin; PLANTEST is T0 with an oracle goal", all(x in sec for x in ("T0", "T1-kin")) and "T0 with an oracle" in sec and "never a driving result" in sec)
chk("PLANTEST in 7.50.7 carries tier T0, oracle goal, the NO-GAIN labels, HIER-ADDS NO, the PAUSED strategic family and n = 0 for TACTICAL", all(x in sec for x in ("**7.50.7 PLANTEST", "T0 with an oracle (true-future) goal", "NO-GAIN", "HIER-ADDS is NO in both families", "GOAL-BLIND", "n = 0", "**PAUSED** (PI item 25)")))
chk("7.49 outcome words as the result files write them: NOTICES, INIT-LEVEL, COPIES, UNREADABLE, O-NONE", all(x in sec for x in ("NOTICES", "INIT-LEVEL", "COPIES", "UNREADABLE", "O-NONE")))
chk("7.50.8 names the rulings R-EGO, R-T2O, R-OUT and SPEC_HWM 9.1, and is marked as build facts with no driving claim", all(x in sec for x in ("R-EGO", "R-T2O", "R-OUT", "section 9.1", "build facts, no driving claim")))
chk("no driving result claim: 'Nothing in v45 is a driving result.' in the status line and 'No driving result, no DOMINANT' in item 23", "Nothing in v45 is a driving result." in new.split("\n")[2] and "* No driving result, no DOMINANT and no scale signal exist.\n\n---\n\n## 10. Reproducibility appendix" in new)
chk("sections 7.49 and 7.50 are in the built report verbatim (the two section files, tip prefix filled)", all(open(HERE + f, encoding="utf-8", newline="").read().strip("\n").replace("{{TP}}", TIP[:8]) in new for f in ("sec_749_imagination.md", "sec_750_horizon_bands.md")))
chk("section order 7.48.6 < 7.49 < 7.50 < 7.50.8 < 8", new.index("**7.48.6 ") < new.index("### 7.49 ") < new.index("### 7.50 ") < new.index("**7.50.8 ") < new.index("## 8. Discussion"))
chk("8.4 rows 1bv to 1ca and the 6+2-row appendix block are present", all(("| 1b%s |" % c) in new for c in "vwxyz") and "| 1ca |" in new and "**Reads added in v45 (tip `%s`)" % TIP[:8] in new and "| `2026-11-08-v8f-plantest/` |" in new and "| `2026-11-08-v8f-bands-build/` |" in new)
chk("section 9 item 23 and 8.1 item 35 are present", "\n23. **v45 reads and their limits.**" in new and "\n35. **What the world model imagines, and the horizon bands**" in new)
# the numbers file (every headline number against the tip source AND the built report), then one mutation that must fail it
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
r = subprocess.run([PY, "-I", HERE + "check_numbers_v45.py"], cwd=HERE, capture_output=True, env=env)
last = r.stdout.decode("utf-8", "replace").strip().split("\n")[-1]
chk("check_numbers_v45.py exits 0", r.returncode == 0, last)
tmp = "F:/TanitAD-scratch/v45_mut/m/"
os.makedirs(tmp, exist_ok=True)
for f in ("check_numbers_v45.py", "sec_749_imagination.md", "sec_750_horizon_bands.md", "edits_v45.md"):
    shutil.copy(HERE + f, tmp + f)
t750 = open(tmp + "sec_750_horizon_bands.md", encoding="utf-8", newline="").read()
t749 = open(tmp + "sec_749_imagination.md", encoding="utf-8", newline="").read()
m750 = t750.replace("1.4228 [1.3800, 1.4647]", "1.4229 [1.3800, 1.4647]", 1)
m749 = t749.replace("0.583 [0.538, 0.630]", "0.584 [0.538, 0.630]", 1)
chk("the mutation changed two numbers (one PLANTEST cell, one VOE1B cell)", m750 != t750 and m749 != t749)
open(tmp + "sec_750_horizon_bands.md", "w", encoding="utf-8", newline="\n").write(m750)
open(tmp + "sec_749_imagination.md", "w", encoding="utf-8", newline="\n").write(m749)
r2 = subprocess.run([PY, "-I", tmp + "check_numbers_v45.py"], cwd=tmp, capture_output=True, env=env)
chk("check_numbers_v45.py FAILS on a copy with two numbers changed (mutation arm)", r2.returncode != 0, r2.stdout.decode("utf-8", "replace").strip().split("\n")[-1][:80])
shutil.rmtree("F:/TanitAD-scratch/v45_mut/m", ignore_errors=True)
npass = sum(1 for x in L if x.startswith("PASS"))
L.append("")
L.append("checks: %d PASS, %d FAIL" % (npass, len(FAILS)))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L[-3:]))
sys.exit(1 if FAILS else 0)
