"""Independent checks of report v39 (the patch's own assertions have already passed to produce out/). Writes the checks file.
The placeholder check FAILS while any `{{` remains in out/ (the Master Mind fills the `{{LLR_...}}` placeholders after the LLR read, then re-runs build_v39.sh).
python -I checks_v39.py tip [out_dir [checks_path]]   (git, python; no network)"""
import hashlib, os, re, subprocess, sys, tempfile
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v39_checks.txt"
M = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BANNED = tuple(w[::-1] for w in ("diov", "dedesrepus", "detcarter", "ylremrof", "regnol on", "noitcerroc", "nwardhtiw"))
BANRE = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)
CLRE = re.compile(r"closed[- ]loop", re.I)
L, FAILS = [], []


def say(x=""):
    L.append(x)


def git(args, binary=True):
    r = subprocess.run(["git", "-C", M, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout if binary else r.stdout.decode("utf-8")


def show(p, b=False):
    r = git(["show", f"{TIP}:{p}"])
    return r if b else r.decode("utf-8")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def chk(name, cond, extra=""):
    say(("PASS " if cond else "FAIL ") + name + ((" " + extra) if extra else ""))
    if not cond:
        FAILS.append(name)


def source_texts():
    """every text blob (md, txt, json, up to 3 MB) under raw/2026-11-05-* at the tip, via one git cat-file --batch"""
    names = [n for n in git(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"], False).split("\n") if "/2026-11-05-" in n and n.endswith((".md", ".txt", ".json"))]
    p = subprocess.Popen(["git", "-C", M, "-c", "safe.directory=*", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = {}
    for n in names:
        p.stdin.write(f"{TIP}:{n}\n".encode()); p.stdin.flush()
        hdr = p.stdout.readline().decode().split()
        size = int(hdr[2])
        data = p.stdout.read(size); p.stdout.read(1)
        if size <= 3_000_000:
            out[n] = data.decode("utf-8", "replace").replace("\r", "")
    p.stdin.close(); p.wait()
    return out


def norm_pipes(line):  # as patch_v39.py: a bare pipe inside a cell is escaped; delimiter = space on both sides, first or last character
    if re.fullmatch(r"\|[\s:\-|]+", line):
        return line
    n, o = len(line), []
    for i, ch in enumerate(line):
        o.append("\\|" if (ch == "|" and 0 < i < n - 1 and line[i - 1] != "\\" and not (line[i - 1] == " " and line[i + 1] == " ")) else ch)
    return "".join(o)


norm = lambda t: re.sub(r"\s+", " ", t)
RP38 = R + "report/2026-11-04-v38/"
v38 = show(RP38 + "V8F_SCIENTIFIC_REPORT.md")
s = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
h = open(OUT + "v8f_report.html", encoding="utf-8", newline="").read()
say("# report v39 checks, tip " + TIP)
say()
chk("v38 inputs are the landed deliverables (sha256)", sha(show(RP38 + "V8F_SCIENTIFIC_REPORT.md", True)) == "7deafaf103a48d8ae22c5146b0adb3f690cd8c487dffaebcbf9c6d0d4155366e"
    and sha(show(RP38 + "v8f_report.html", True)) == "5f6b1b1687bf19a6061c075efa87417379d6d22c3dab7e6109f68fc56b1510dc")
chk("html = page template with the v39 md", h == show(R + "report/2026-10-26/page_template_v30.html").replace("%%MD%%", s))
chk("LF only, no BOM (md and html)", "\r" not in s and "\ufeff" not in s and "\r" not in h and "\ufeff" not in h)
c38, c39 = Counter(v38.split("\n")), Counter(s.split("\n"))
rem = [l for l in (c38 - c39).elements() if l.strip()]
add = [l for l in (c39 - c38).elements() if l.strip()]
KEYS = ("**Status:** research report, v38 of 2026-11-04",
        "* **A recovery fine-tune moves M1 the right way and does not clear its guards; an a-priori deviation law doubles the closure, and its lower dose removes the ADE6 harm and keeps the along-track and distance-keeping harm (§7.36; five arms",
        "* **What is next.** The crawl set is still open",
        "* **Recommendation [E].** (b), form b1.",
        "| 1at | **RAPR-P01**")
hits = [sum(1 for l in rem if k in l) for k in KEYS]
unexp = [l[:80] for l in rem if not any(k in l for k in KEYS)]
chk("removed lines from v38: exactly the 5 edited lines, each key matched once (status line, abstract recovery bullet, abstract next bullet, the 7.36.6 Recommendation bullet, row 1at)",
    not unexp and hits == [1] * len(KEYS) and len(rem) == 5, "(%d removed; unexpected %s; hits %s)" % (len(rem), unexp[:2], hits))
i737, i738, i739, i8 = s.index("### 7.37 "), s.index("### 7.38 "), s.index("### 7.39 "), s.index("## 8. Discussion")
pending = "{{LLR_RESULT_PENDING}}" in s
order = ["**7.36.7 ", "### 7.37 ", "**7.37.1 ", "**7.37.2 ", "**7.37.3 ", "**7.37.4 ", "### 7.38 "] + ["**7.38.%d " % i for i in range(1, 8)] + ["### 7.39 "] + ["**7.39.%d " % i for i in range(1, 6)]
order += ["{{LLR_RESULT_PENDING}}" if pending else "**7.39.6 ", "**7.39.7 ", "**7.39.8 ", "## 8. Discussion"]
pos = [s.index(x) for x in order]
chk("order: 7.36.7 < 7.37 (.1 to .4) < 7.38 (.1 to .7) < 7.39 (.1 to .5) < 7.39.6 (or its placeholder) < 7.39.7 < 7.39.8 < section 8, each heading once",
    pos == sorted(pos) and all(s.count(x) == 1 for x in order if x != "{{LLR_RESULT_PENDING}}"), "(7.39.6 %s)" % ("pending, placeholder in place" if pending else "filled"))
sec = s[i737:i8]
srcs = source_texts()
SL = set()
for t in srcs.values():
    SL |= {norm_pipes(l) for l in t.split("\n") if l.startswith("|")}
tl = [l for l in sec.split("\n") if l.startswith("|")]
odd = [l for l in add if l.startswith("|") and l not in SL]
dirs_total = sum(int(l.split(" | ")[4].split()[0]) for l in open(OUT + "DIRECTIVES.txt", encoding="utf-8").read().strip().split("\n"))
chk("every table line of 7.37 to 7.39 is a line of a 2026-11-05 source file (a bare pipe inside a cell escaped); the only other added table lines are rows 1au to 1ay and the changed row 1at",
    all(l in SL for l in tl) and len(tl) == dirs_total and len(odd) == 6 and sum(l.startswith("| 1at | ") for l in odd) == 1 and sum(bool(re.match(r"\| 1a[u-y] \| ", l)) for l in odd) == 5,
    "(%d table lines in the sections = %d expanded directive lines; %d source files)" % (len(tl), dirs_total, len(srcs)))
cols = lambda l: l.replace("\\|", "").count("|") - 1
chk("every table of 7.37 to 7.39 and of rows 1au to 1ay has one cell count (strict, escaped pipes aside)", all(len({cols(r) for r in blk}) == 1 for blk in
    [[x for x in b.split("\n") if x] for b in re.findall(r"(?:^\|.*\n?)+", sec, flags=re.M)]) and len({cols(l) for l in odd if l.startswith("| 1a")}) == 1)
reg = show("Project Steering/GOALS_AND_CLAIMS.md")
corp = norm(" ".join(list(srcs.values()) + [reg, v38]))
qs = [q.strip('"') for l in add for q in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", l)]
chk("italic quotations of 25+ characters in the added lines verbatim in a source (2026-11-05 files, register, v38)", qs and all(norm(q) in corp for q in qs), "(%d quotations)" % len(qs))
fams = [len(re.findall(f, sec)) for f in ("LONGITUDINAL|DK-VIOL|DK-HW|along-track|speed MAE|accel MAE", "LATERAL|curvature|yaw-rate|heading|cross-track", "TACTICAL|TAC-LAT|TAC-LON", "STRATEGIC|S-POS")]
chk("each of the four metric families named in 7.37 to 7.39 at least 3 times (STRATEGIC active for v8F)", all(c >= 3 for c in fams), str(fams))
s737, s738, s739 = s[i737:i738], s[i738:i739], s[i739:i8]
chk("T1-kin in 7.37, 7.38 and 7.39; the paired episode-cluster bootstrap, B = 2,000 and H-ESTIM-SEED-1 stated in 7.37 and 7.39", all("T1-kin" in x for x in (s737, s738, s739))
    and all("paired episode-cluster bootstrap" in x and "B = 2,000" in x and "H-ESTIM-SEED-1" in x for x in (s737, s739)))
chk("evidence classes: [M] in each of 7.37, 7.38, 7.39; [E] in 7.37 and 7.38; [H] in 7.37, 7.38 and 7.39", all("[M]" in x for x in (s737, s738, s739)) and "[E]" in s737 and "[E]" in s738
    and all("[H]" in x for x in (s737, s738, s739)))
cl = lambda t: len(CLRE.findall(t))
bc = lambda t: [len(re.findall(r"\b" + w + r"\b", t, re.I)) for w in BANNED]
parts = {n: open(HERE + n, encoding="utf-8", newline="").read() for n in ("sec_737_rpt.md", "sec_738_motion_c.md", "sec_739_levelhead.md", "sec_739_llr_result.md", "sec_739_tail.md", "edits_v39.md", "PLACEHOLDERS.md", "HANDOFF_V39.md")
         if os.path.exists(HERE + n)}
chk("the forbidden two-word T2 wording (either separator) is absent from every added line and every part file; the occurrences of the carried v38 text are untouched (v38 %d, v39 %d)" % (cl(v38), cl(s)),
    cl(" ".join(add)) <= cl(" ".join(rem)) and not any(CLRE.search(t) for t in parts.values()) and cl(s) == cl(v38),
    "(occurrences in added lines %d against %d in the 5 edited lines they replace, part files %d)" % (cl(" ".join(add)), cl(" ".join(rem)), sum(cl(t) for t in parts.values())))
chk("banned-word scan: none new in the added lines (net of the 5 edited lines they replace) or in the part files; per-word counts of the whole report unchanged against v38",
    all(a <= r for a, r in zip(bc(" ".join(add)), bc(" ".join(rem)))) and not any(BANRE.search(t) for t in parts.values()) and bc(s) == bc(v38),
    "(v38 %s, v39 %s)" % (bc(v38), bc(s)))
chk("no dollar sign added; math span counts as in v38", s.count("$") == v38.count("$") and s.count("$$") == v38.count("$$"), "(%d dollar signs)" % s.count("$"))
i83 = s.index("### 8.3 ")
ups = [s.index("**Update (v%d).**" % v, i83) for v in (39, 38, 37, 36)]
chk("8.3 has Update (v39) before Update (v38) before Update (v37) before Update (v36), each once", ups == sorted(ups) and all(s.count("**Update (v%d).**" % v) == 1 for v in (39, 38, 37, 36)))
i81, i82 = s.index("### 8.1 "), s.index("### 8.2 ")
nums = [int(x) for x in re.findall(r"^(\d+)\. \*\*", s[i81:i82], re.M)]
n38 = [int(x) for x in re.findall(r"^(\d+)\. \*\*", v38[v38.index("### 8.1 "):v38.index("### 8.2 ")], re.M)]
chk("8.1 ends with item 29: the v38 items (ending at 25) are untouched and items 26 to 29 follow; item 25 is the v38 text", n38[-1] == 25 and nums == n38 + [26, 27, 28, 29] and "25. **The a-priori deviation law at dose 0.1 removes RAPR's ADE6 harm" in s[i81:i82])
i9, i10 = s.index("## 9. "), s.index("## 10. Reproducibility appendix")
chk("section 9 ends with item 17 (v39 reads and their limits), after item 16", re.findall(r"^(\d+)\. \*\*", s[i9:i10], re.M)[-2:] == ["16", "17"] and "17. **v39 reads and their limits.**" in s[i9:i10])
rows = [s.index("| %s | " % r) for r in ("1at", "1au", "1av", "1aw", "1ax", "1ay")] + [s.index("| 2 | **LoRA on DINOv3**")]
chk("8.4 has rows 1au to 1ay once each, in order, after row 1at and before row 2; row 1at points to them", rows == sorted(rows) and all(s.count("| %s | " % r) == 1 for r in ("1au", "1av", "1aw", "1ax", "1ay")) and "(rows 1au to 1ay) |" in s)
line = s.split("\n")[2]
chk("status line names v39 of 2026-11-05, the tip, 7.36, 7.37, 7.39, RPT, CL, CLR and LLR; no date after 2026-11-05 anywhere", "v39 of 2026-11-05" in line and "`%s`" % TIP[:8] in line and all(x in line for x in ("§7.36", "§7.37", "§7.39", "RPT", "CL and CLR", "LLR"))
    and not [x for x in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if x > "2026-11-05"])
chk("no stale pointer: the row 1at 'PI's go |' ending and 'v38 of 2026-11-04, results' are gone; v38 is named as the previous version", "no box-2 job starts without the PI's go |" not in s and "v38 of 2026-11-04, results" not in line and "v38: `ff09d2c1`" in line)
J = __import__("json").loads(show(R + "raw/2026-11-05-recover-next-rpt/RECOVER_READ_RPT.json"))
want = ["%.4f" % J["primary"]["share_closed"][k] for k in ("s0", "s1")] + ["%.4f" % J["frame_freeze"]["share_closed"][k] for k in ("s0", "s1")] + ["%+.4f" % J["guards"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1")]
chk("added numbers agree with the raw RPT JSON and the result files (RPT share closed, frame-freeze, ADE6; CRAWL-LEAD, RI, CL, CLR headline numbers)",
    all(x in s737 for x in want) and all(x in s738 for x in ("0.4639 / 0.4956", "0.2409 / 0.2687", "1.002 / 1.003", "+0.332", "+0.043", "+0.0001 [−0.0005, +0.0007]")) and all(x in s739 for x in
    ("0.233 / 0.130", "−0.0016 [−0.0042, +0.0011]", "−0.0015 [−0.0038, +0.0005]", "0.2027", "3.362", "0.780", "1.898", "−0.00043")), "(JSON label %s, controls_failed %s)" % (J["label"], J["controls_failed"]))
ph = sorted(set(re.findall(r"\{\{[^{}\n]*\}\}", s + h)))
phd = open(HERE + "PLACEHOLDERS.md", encoding="utf-8").read()
chk("every placeholder left in out/ is a documented {{LLR_...}} row of PLACEHOLDERS.md", all(x.startswith("{{LLR_") and "`%s`" % x in phd for x in ph), "(%s)" % ", ".join(ph))
files = [os.path.join(d, f).replace("\\", "/") for d, _, fs in os.walk(HERE) for f in fs if f.endswith((".py", ".md", ".sh", ".js"))]
crs = [os.path.relpath(f, HERE) for f in files if b"\r" in open(f, "rb").read()]
chk("LF only for every .py, .md, .sh and .js file of the package (CR counted in python)", not crs, "(%d files; with CR: %s)" % (len(files), crs[:3]))
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v39.py", TIP, td], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1"))
    same = r.returncode == 0 and all(open(os.path.join(td, f), "rb").read() == open(OUT + f, "rb").read() for f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html"))
    chk("patch_v39.py re-run reproduces both deliverables byte for byte", same, r.stdout.strip()[-100:].replace("\n", " ") + r.stderr.strip()[-200:])
# the placeholder check goes last: it must read FAIL while the Master Mind has not filled the LLR placeholders
nb = len(re.findall(r"\{\{", s)) + len(re.findall(r"\{\{", h))
chk("PLACEHOLDERS: no `{{` remains in out/ (md and html)", nb == 0, "(%d occurrences of `{{` in md + html; distinct: %s)" % (nb, ", ".join(ph) if ph else "none"))
say()
say("RESULT: %s (%d checks, %d failed%s)" % ("PASS" if not FAILS else "FAIL", sum(1 for x in L if x[:5] in ("PASS ", "FAIL ")), len(FAILS), "; failed: " + " | ".join(f[:40] for f in FAILS) if FAILS else ""))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print(L[-1])
sys.exit(1 if FAILS else 0)
