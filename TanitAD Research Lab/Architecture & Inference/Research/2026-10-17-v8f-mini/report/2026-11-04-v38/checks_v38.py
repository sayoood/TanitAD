"""Independent checks of report v38 (the patch's own assertions have already passed to produce out/). Writes the checks file.
python -I checks_v38.py tip [out_dir [checks_path]]   (git, python; no network)"""
import difflib, hashlib, os, re, subprocess, sys, tempfile
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v38_checks.txt"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BANNED = tuple(w[::-1] for w in ("diov", "dedesrepus", "detcarter", "ylremrof", "regnol on", "noitcerroc", "nwardhtiw"))
BANRE = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)
L, FAILS = [], []


def say(x=""):
    L.append(x)


def show(p, b=False):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*", "show", f"{TIP}:{p}"], capture_output=True)
    assert r.returncode == 0, p
    return r.stdout if b else r.stdout.decode("utf-8")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def chk(name, cond, extra=""):
    say(("PASS " if cond else "FAIL ") + name + ((" " + extra) if extra else ""))
    if not cond:
        FAILS.append(name)


norm = lambda t: re.sub(r"\s+", " ", t)
RP37 = R + "report/2026-11-04-v37/"
v37 = show(RP37 + "V8F_SCIENTIFIC_REPORT.md")
s = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
h = open(OUT + "v8f_report.html", encoding="utf-8", newline="").read()
say("# report v38 checks, tip " + TIP)
say()
chk("v37 inputs are the landed deliverables (sha256)", sha(show(RP37 + "V8F_SCIENTIFIC_REPORT.md", True)) == "187744e89d2c9fea857a09d7c424079b680ff47ab04e9e46e8b1a0bc5aa49842"
    and sha(show(RP37 + "v8f_report.html", True)) == "7f11c3ee86b7f34160661115f3dbd3db5d98f3b68504c55bf1fac096c1f4c5cb")
chk("html = page template with the v38 md", h == show(R + "report/2026-10-26/page_template_v30.html").replace("%%MD%%", s))
chk("LF only, no BOM (md and html)", "\r" not in s and "\ufeff" not in s and "\r" not in h and "\ufeff" not in h)
d = list(difflib.ndiff(v37.split("\n"), s.split("\n")))
rem = [l[2:] for l in d if l.startswith("- ") and l[2:].strip()]
add = [l[2:] for l in d if l.startswith("+ ") and l[2:].strip()]
KEYS = ("**Status:** research report, v37 of 2026-11-04",
        "* **A recovery fine-tune moves M1 the right way and does not clear its guards; an a-priori deviation law doubles the closure (§7.36; four arms",
        "* **What is next.** The crawl set is still open",
        "### 7.36 RECOVER-FT and its attribution arms",
        "* **Status of this section (v37) [M].** Four reads",
        "* **Committed next [M].** The combiner selects card section 4 row 3 (RAPRC fails D2",
        "**7.36.5 What the four recovery arms establish",
        "* **Establishes [M].** (1) A recovery fine-tune moves M1 the right way at both seeds in every arm (R, RTL, RP01, RAPR)",
        "* **Cannot [M].** One training run per seed per arm (`H-ESTIM-SEED-1`): at s0 every arm's M1 effect sits inside the replicate's spread (RAPR 0.0237",
        "**7.36.6 Next (Rule Zero).** RAPR-P01 is the committed next arm",
        "**7.36.7 Provenance [M].** Under `R/raw/` at the tip `cd6ce52d`",
        "24. **A recovery fine-tune of O moves the M1 crawl share",
        "| 1as | **RECOVER-NEXT**")
hits = [sum(1 for l in rem if k in l) for k in KEYS]
unexp = [l[:80] for l in rem if not any(k in l for k in KEYS)]
chk("removed lines from v37: exactly the 13 edited lines, each key matched once (status, abstract recovery bullet and next bullet, 7.36 title, 7.36 status bullet, the 7.36.4 Committed-next bullet, "
    "the old 7.36.5 heading, Establishes, Cannot, 7.36.6 Next, 7.36.7 Provenance, 8.1 item 24, row 1as)", not unexp and hits == [1] * len(KEYS) and len(rem) == 13, "(%d removed; unexpected %s; hits %s)" % (len(rem), unexp[:2], hits))
i7354, i736, i7364a, i75, i8 = s.index("**7.35.4 "), s.index("### 7.36 "), s.index("**7.36.4a "), s.index("**7.36.5 "), s.index("## 8. Discussion")
chk("order: 7.35.4 < 7.36 < 7.36.4 < 7.36.4a < 7.36.5 < 7.36.6 < 7.36.7 < section 8, each heading once", i7354 < i736 < s.index("**7.36.4 ") < i7364a < i75 < s.index("**7.36.6 ") < s.index("**7.36.7 ") < i8
    and all(s.count(x) == 1 for x in ("### 7.36 ", "**7.36.4a ", "**7.36.5 ", "**7.36.6 ", "**7.36.7 ")))
sec = s[i7364a:i8]
pr = show(R + "raw/2026-11-04-recover-next-raprp01/RESULTS.md")
dr = show(R + "raw/2026-11-04-recover-rowk-decision/DECISION_ROWK.md")
dj = show(R + "raw/2026-11-04-recover-rowk-decision/d2_closure_need.json")
a3 = show(R + "raw/2026-11-04-recover-next-raprp01/PREREG_RECOVER_NEXT_ADDENDUM3_RAPRP01.md")
reg = show("Project Steering/GOALS_AND_CLAIMS.md")
prl = set(pr.split("\n"))
atl = [l for l in add if l.startswith("|")]
odd = [l for l in atl if l not in prl]
chk("every table line added in 7.36.4a is a line of the RAPR-P01 RESULTS file; the only other added table lines are rows 1at and 1as (the pointer)",
    len(odd) == 2 and sum(l.startswith("| 1at | ") for l in odd) == 1 and sum(l.startswith("| 1as | ") for l in odd) == 1 and len(atl) - len(odd) == 52, "(%d added table lines, 52 from RESULTS, 3 tables)" % len(atl))
corp = norm(" ".join([pr, dr, dj, a3, reg, v37]))
qs = [q.strip('"') for l in add for q in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", l)]
chk("italic quotations of 25+ characters in the added lines verbatim in a source (RESULTS, addendum 3, decision item, register, v37)", qs and all(norm(q) in corp for q in qs), "(%d quotations)" % len(qs))
fams = [len(re.findall(f, sec)) for f in ("LONGITUDINAL|DK-VIOL|DK-HW|along-track|speed MAE|accel MAE", "LATERAL|curvature|yaw-rate|heading|cross-track", "TACTICAL|TAC-LAT|TAC-LON", "STRATEGIC|S-POS")]
chk("each of the four metric families named in 7.36.4a to 7.36.7 at least 3 times (STRATEGIC active for v8F)", all(c >= 3 for c in fams), str(fams))
i76 = s.index("**7.36.6 ")
chk("T1-kin and the estimator stated in 7.36.4a, and T1-kin in 7.36.6", "T1-kin" in s[i7364a:i75] and "B = 2,000" in s[i7364a:i75] and "paired episode-cluster bootstrap" in s[i7364a:i75] and "T1-kin" in s[i76:s.index("**7.36.7 ")])
chk("evidence classes on the new claims: [M] in 7.36.4a, [E] and [H] in 7.36.5 to 7.36.6", all(x in s[i7364a:i75] for x in ("**Setup [M].**", "**Reading [M].**")) and "[E]" in s[i76:i8] and "[H]" in s[i75:i8] and "post hoc" in s[i76:i8])
chk("closure need labelled post hoc and not a registered rule; (b1) recommendation and the PI decision named", "Not a registered rule: it ranks levers, it decides nothing." in s[i76:i8] and "(b), form b1" in s[i76:i8]
    and "neither" in s[i76:i8].lower() and "PI's decision and box-2 go" in s[i76:i8])
cl = lambda t: len(re.findall(r"closed[- ]loop|driving performance|\bdrives\b", t.lower()))
chk("no 'closed loop' / driving wording in the new section 7.36.4a to 7.36.7, and none added anywhere (counts against v37)", cl(sec) == 0 and cl(s) <= cl(v37), "(v37 %d, v38 %d)" % (cl(v37), cl(s)))
bc = lambda t: [len(re.findall(r"\b" + w + r"\b", t, re.I)) for w in BANNED]
chk("no banned word in the added lines; per-word counts of the report unchanged against v37", not any(BANRE.search(l) for l in add) and bc(s) == bc(v37), "(v37 %s, v38 %s)" % (bc(v37), bc(s)))
chk("no dollar sign added; math span counts as in v37", s.count("$") == v37.count("$") and s.count("$$") == v37.count("$$"), "(%d dollar signs)" % s.count("$"))
i83 = s.index("### 8.3 ")
chk("8.3 has Update (v38) before Update (v37) before Update (v36)", s.index("**Update (v38).**", i83) < s.index("**Update (v37).**", i83) < s.index("**Update (v36).**", i83) and s.count("**Update (v38).**") == 1)
i81, i82 = s.index("### 8.1 "), s.index("### 8.2 ")
chk("8.1 ends with item 25, and item 24 points to it", re.findall(r"^(\d+)\. \*\*", s[i81:i82], re.M)[-1] == "25" and "(RAPR-P01, read in item 25)" in s[i81:i82])
i9, i10 = s.index("## 9. "), s.index("## 10. Reproducibility appendix")
chk("section 9 ends with item 16 (v38 reads and their limits)", re.findall(r"^(\d+)\. \*\*", s[i9:i10], re.M)[-1] == "16" and "16. **v38 reads and their limits.**" in s[i9:i10])
chk("8.4 has row 1at once, after row 1as and before row 2; row 1as points to it", s.count("| 1at | ") == 1 and s.index("| 1as | ") < s.index("| 1at | ") < s.index("| 2 | **LoRA on DINOv3**") and "read as 1at) |" in s)
line = s.split("\n")[2]
chk("status line names v38 of 2026-11-04, the tip, 7.36 and RAPR-P01; no date after 2026-11-04 anywhere", "v38 of 2026-11-04" in line and "`%s`" % TIP[:8] in line and "§7.36" in line and "RAPR-P01" in line
    and not [x for x in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if x > "2026-11-04"])
chk("no stale pointer about RAPR-P01 (not run / committed next arm) or four recovery arms in 7.36.5", "four recovery arms establish" not in s and "RAPR-P01 is the committed next arm" not in s and "(RAPR-P01, not run)" not in s
    and "PI box-2 go) |" not in s)
chk("added numbers in the new section agree with the raw JSON of the read and of the closure fit (share closed, frame-freeze, ADE6, floor, slope, r, closure need, extrapolation)",
    all(x in sec for x in ("0.2147", "0.1391", "0.2344", "0.1055", "+0.0371", "+0.0429", "0.0114", "0.0362", "0.0083", "0.0038", "1.097 / 1.146", "0.997 / 0.932", "0.68 (s0) / 0.60 (s1)", "×1.54 / ×1.88", "about 1.04 / 0.95")))
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v38.py", TIP, td], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1"))
    same = r.returncode == 0 and all(open(os.path.join(td, f), "rb").read() == open(OUT + f, "rb").read() for f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html"))
    chk("patch_v38.py re-run reproduces both deliverables byte for byte", same, r.stdout.strip()[-120:] + r.stderr.strip()[-200:])
say()
say("RESULT: %s (%d checks, %d failed)" % ("PASS" if not FAILS else "FAIL", sum(1 for x in L if x[:5] in ("PASS ", "FAIL ")), len(FAILS)))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print(L[-1])
sys.exit(1 if FAILS else 0)
