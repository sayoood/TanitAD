"""Independent checks of report v44 (the patch's own assertions have already passed to produce out/). Writes the checks file.
python -I checks_v44.py tip [out_dir [checks_path]]   (git, python; no network except git ls-remote)
Includes: the base is the landed v43; only the stated carried lines differ from it; banned words, dashes and pending wording in the added lines; the status line; check_numbers_v44.py (every headline number
of the new text against its tip source file AND the built report) and a mutation run of it on a copy of the report with one number changed (it must FAIL)."""
import hashlib, os, re, subprocess, sys, shutil
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v44_checks.txt"
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
base = git(["show", f"{TIP}:{R}report/2026-11-07-v43/V8F_SCIENTIFIC_REPORT.md"])
chk("base is the landed v43 md (sha256 ad2df9ba...)", hashlib.sha256(base).hexdigest().startswith("ad2df9ba70655f8b"))
base = base.decode("utf-8")
tpl = git(["show", f"{TIP}:{R}report/2026-10-26/page_template_v30.html"]).decode("utf-8")
chk("html is the page template with the v44 md", html == tpl.replace("%%MD%%", new))
chk("LF only (md, html)", "\r" not in new and "\r" not in html)
chk("status line is v44 and names the tip prefix", "**Status:** research report, v44 of 2026-11-07, results through repository tip `%s`" % TIP[:8] in new)
live = subprocess.run(["git", "-C", M, "-c", "safe.directory=*", "ls-remote", "origin", "refs/heads/agent/arch-inf-20260803"], capture_output=True).stdout.decode().split("\t")[0]
chk("the build tip equals the live remote tip", live == TIP, "live " + live[:8])
cb, cn = Counter(base.split("\n")), Counter(new.split("\n"))
added = [l for l in (cn - cb).elements() if l.strip()]
removed = [l for l in (cb - cn).elements() if l.strip()]
L.append("added lines %d, removed (edited carried) lines %d" % (len(added), len(removed)))
chk("only the carried lines of edits_v44.md were changed (removed lines at most 12)", len(removed) <= 12, "removed: " + " || ".join(x[:50] for x in removed))
chk("every removed line is a status line, an abstract / 8.1 / 7.46 / 8.4 line that edits_v44.md rewrites", all(re.search(r"v43 of 2026-11-07|pre-registered, not run|pre-registered and not run|PRE-REGISTERED|RENC \(M-RENC|then L's own T2 read|committed next lever RENC|Whether the gap closes", x) for x in removed))
A = "\n".join(added)
chk("no banned word in the added lines", not BANRE.search(A), str(BANRE.search(A)))
chk("no closed loop wording in the added lines", not re.search(r"closed[- ]loop", A, re.I))
chk("no en or em dash in the added lines", not re.search(r"[\u2013\u2014]", A))
chk("no drives / driving performance wording in the added lines", not re.search(r"\bdrives\b|driving performance", A.lower()))
chk("no unfilled placeholder or directive", "{{" not in new and "@@" not in new and "<<" not in A)
sec = new[new.index("### 7.47 "):new.index("## 8. Discussion")]
chk("sections 7.47 and 7.48 carry no pending wording for a read that landed", not re.search(r"PENDING|running with no result|has no result and none is quoted|is running\b", sec.replace("the register records them as running", "")), "")
chk("STRATEGIC is PAUSED in 7.47 and 7.48 and never a pass", "PAUSED" in sec and not re.search(r"STRATEGIC[^.\n]{0,40}\bPASS\b", sec))
chk("every evidence tier named: T0, T1-kin, T2", all(x in sec for x in ("T0", "T1-kin", "T2")))
chk("the imagination plan says no result is claimed", "no result is claimed" in sec and "**No IMAG1 or VOE1 result is on the tip, and none is claimed.**" in sec)
chk("outcome words as the result files write them: RE-HARMS and T-NOT", "reads **RE-HARMS**" in sec and "Outcome T-NOT, as the card committed it" in sec)
chk("no driving result claim: 'Nothing in 7.47 is a driving result' and 'nothing in 7.48 is a driving gain'", "Nothing in 7.47 is a driving result." in sec and "nothing in 7.48 is a driving gain" in sec)
chk("the status line says nothing in v44 is a driving result", "Nothing in v44 is a driving result." in new.split("\n")[2])
chk("section order 7.46.11 < 7.47 < 7.48 < 8", new.index("**7.46.11 ") < new.index("### 7.47 ") < new.index("### 7.48 ") < new.index("## 8. Discussion"))
chk("8.4 rows 1bm to 1bu and the 13-row appendix block are present", all(("| 1b%s |" % c) in new for c in "mnopqrstu") and "**Reads added in v44 (tip `%s`)" % TIP[:8] in new)
chk("section 9 item 22 and 8.1 item 34 are present", "\n22. **v44 reads and their limits.**" in new and "\n34. **The world model level by level" in new)
# the numbers file (every headline number against the tip source AND the built report), then one mutation that must fail it
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
r = subprocess.run([PY, "-I", HERE + "check_numbers_v44.py", TIP, OUT + "V8F_SCIENTIFIC_REPORT.md"], cwd=HERE, capture_output=True, env=env)
last = r.stdout.decode("utf-8", "replace").strip().split("\n")[-1]
chk("check_numbers_v44.py exits 0", r.returncode == 0, last)
tmp = "F:/TanitAD-scratch/v44_mut/"
os.makedirs(tmp, exist_ok=True)
mut = new.replace("+0.0031 [−0.0003, +0.0073]", "+0.0032 [−0.0003, +0.0073]", 1).replace("−0.1956 [−0.4391, +0.0895]", "−0.1957 [−0.4391, +0.0895]", 1)
chk("the mutation changed two numbers", mut != new)
open(tmp + "mut.md", "w", encoding="utf-8", newline="\n").write(mut)
r2 = subprocess.run([PY, "-I", HERE + "check_numbers_v44.py", TIP, tmp + "mut.md"], cwd=tmp, capture_output=True, env=env)
chk("check_numbers_v44.py FAILS on a copy with two numbers changed (mutation arm)", r2.returncode != 0, r2.stdout.decode("utf-8", "replace").strip().split("\n")[-1][:80])
shutil.rmtree(tmp, ignore_errors=True)
npass = sum(1 for x in L if x.startswith("PASS"))
L.append("")
L.append("checks: %d PASS, %d FAIL" % (npass, len(FAILS)))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L[-3:]))
sys.exit(1 if FAILS else 0)
