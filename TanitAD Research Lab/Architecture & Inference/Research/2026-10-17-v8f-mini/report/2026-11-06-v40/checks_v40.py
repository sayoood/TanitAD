"""Independent checks of report v40 (the patch's own assertions have already passed to produce out/). Writes the checks file.
python -I checks_v40.py tip [out_dir [checks_path]]   (git, python; no network)"""
import hashlib, json, os, re, subprocess, sys, tempfile
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v40_checks.txt"
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
    """every text blob (md, txt, json, up to 3 MB) under raw/2026-11-05-levelfloor-* and raw/2026-11-06-* at the tip, via one git cat-file --batch"""
    names = [n for n in git(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"], False).split("\n")
             if ("/2026-11-05-levelfloor-" in n or "/2026-11-06-" in n) and n.endswith((".md", ".txt", ".json"))]
    p = subprocess.Popen(["git", "-C", M, "-c", "safe.directory=*", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = {}
    for n in names:
        p.stdin.write(f"{TIP}:{n}\n".encode()); p.stdin.flush()
        size = int(p.stdout.readline().decode().split()[2])
        data = p.stdout.read(size); p.stdout.read(1)
        if size <= 3_000_000:
            out[n] = data.decode("utf-8", "replace").replace("\r", "")
    p.stdin.close(); p.wait()
    return out


def norm_pipes(line):  # as patch_v40.py: a bare pipe inside a cell is escaped; delimiter = space on both sides, first or last character
    if re.fullmatch(r"\|[\s:\-|]+", line):
        return line
    n, o = len(line), []
    for i, ch in enumerate(line):
        o.append("\\|" if (ch == "|" and 0 < i < n - 1 and line[i - 1] != "\\" and not (line[i - 1] == " " and line[i + 1] == " ")) else ch)
    return "".join(o)


norm = lambda t: re.sub(r"\s+", " ", t)
RP39 = R + "report/2026-11-05-v39/"
v39 = show(RP39 + "V8F_SCIENTIFIC_REPORT.md")
s = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
h = open(OUT + "v8f_report.html", encoding="utf-8", newline="").read()
say("# report v40 checks, tip " + TIP)
say()
chk("v39 inputs are the landed deliverables (sha256)", sha(show(RP39 + "V8F_SCIENTIFIC_REPORT.md", True)) == "8ccffcf01f059f10ea0316d214dd40b96c480c3f44f07a5b99e824284326adab"
    and sha(show(RP39 + "v8f_report.html", True)) == "3f2fb6d1280de3ac010465351af7fcbdad943be2a2826bea1b0faf072d81d22e")
chk("html = page template with the v40 md", h == show(R + "report/2026-10-26/page_template_v30.html").replace("%%MD%%", s))
chk("LF only, no BOM (md and html)", "\r" not in s and "\ufeff" not in s and "\r" not in h and "\ufeff" not in h)
c39, c40 = Counter(v39.split("\n")), Counter(s.split("\n"))
rem = [l for l in (c39 - c40).elements() if l.strip()]
add = [l for l in (c40 - c39).elements() if l.strip()]
KEYS = ("**Status:** research report, v39 of 2026-11-05", "* **A recovery fine-tune moves M1 the right way and does not clear its guards;", "* **What is next.** The crawl set is still open",
        "* **Next (Rule Zero) [M].** The committed branch of the A2 card names T2 re-perception and the A40")
hits = [sum(1 for l in rem if l.startswith(k)) for k in KEYS]
unexp = [l[:80] for l in rem if not any(l.startswith(k) for k in KEYS)]
chk("removed lines from v39: exactly the 4 edited lines, each key matched once (status line, abstract recovery bullet, abstract next bullet, the 7.39.7 Next bullet)",
    not unexp and hits == [1] * len(KEYS) and len(rem) == 4, "(%d removed; unexpected %s; hits %s)" % (len(rem), unexp[:2], hits))
order = ["**7.39.8 ", "### 7.40 "] + ["**7.40.%d " % i for i in range(1, 8)] + ["## 8. Discussion"]
pos = [s.index(x) for x in order]
chk("order: 7.39.8 < 7.40 < 7.40.1 to 7.40.7 < section 8, each heading once", pos == sorted(pos) and all(s.count(x) == 1 for x in order))
i740, i8 = s.index("### 7.40 "), s.index("## 8. Discussion")
sec = s[i740:i8]
srcs = source_texts()
SL = set()
for t in srcs.values():
    SL |= {norm_pipes(l) for l in t.split("\n") if l.startswith("|")}
tl = [l for l in sec.split("\n") if l.startswith("|")]
odd = [l for l in add if l.startswith("|") and l not in SL]
dirs_total = sum(int(l.split(" | ")[4].split()[0]) for l in open(OUT + "DIRECTIVES.txt", encoding="utf-8").read().strip().split("\n"))
chk("every table line of 7.40 is a line of a level-floor source file (a bare pipe inside a cell escaped); the only other added table line is row 1az",
    all(l in SL for l in tl) and len(tl) == dirs_total and len(odd) == 1 and odd[0].startswith("| 1az | "), "(%d table lines in 7.40 = %d expanded directive lines; %d source files)" % (len(tl), dirs_total, len(srcs)))
cols = lambda l: l.replace("\\|", "").count("|") - 1
i84 = s.index("### 8.4 ")
_r1ay = [l for l in s[i84:].split("\n") if l.startswith("| 1ay | ")][0]
chk("every table of 7.40 has one cell count; row 1az has the cell count of row 1ay", all(len({cols(r) for r in blk}) == 1 for blk in
    [[x for x in b.split("\n") if x] for b in re.findall(r"(?:^\|.*\n?)+", sec, flags=re.M)]) and cols(odd[0]) == cols(_r1ay) == 4)
reg = show("Project Steering/GOALS_AND_CLAIMS.md")
corp = norm(" ".join(list(srcs.values()) + [reg, v39]))
qs = [q.strip('"') for l in add for q in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", l)]
chk("italic quotations of 25+ characters in the added lines verbatim in a source (level-floor files, register, v39)", qs and all(norm(q) in corp for q in qs), "(%d quotations)" % len(qs))
fams = [len(re.findall(f, sec)) for f in ("LONGITUDINAL|DK-VIOL|DK-HW|along-track|speed MAE|accel MAE", "LATERAL|curvature|yaw-rate|heading|cross-track", "TACTICAL|TAC-LAT|TAC-LON", "STRATEGIC|S-POS")]
chk("each of the four metric families named in 7.40 at least 3 times (STRATEGIC active for v8F)", all(c >= 3 for c in fams), str(fams))
chk("T1-kin, the paired episode-cluster bootstrap, B = 2,000, eval-141 and H-ESTIM-SEED-1 stated in 7.40", all(x in norm(sec) for x in ("T1-kin", "paired episode-cluster bootstrap", "B = 2,000", "eval-141", "H-ESTIM-SEED-1")))
chk("evidence classes [M], [E] and [H] in 7.40", all(x in sec for x in ("[M]", "[E]", "[H]")))
cl = lambda t: len(CLRE.findall(t))
bc = lambda t: [len(re.findall(r"\b" + w + r"\b", t, re.I)) for w in BANNED]
parts = {n: open(HERE + n, encoding="utf-8", newline="").read() for n in ("sec_740_levelfloor.md", "edits_v40.md", "HANDOFF_V40.md", "msg_v40.txt") if os.path.exists(HERE + n)}
chk("the forbidden two-word T2 wording (either separator) is absent from every added line and every part file; the count of the carried v39 text is untouched (v39 %d, v40 %d)" % (cl(v39), cl(s)),
    cl(" ".join(add)) <= cl(" ".join(rem)) and not any(CLRE.search(t) for t in parts.values()) and cl(s) == cl(v39))
chk("banned-word scan: none new in the added lines (net of the 4 edited lines) or in the part files; per-word counts of the whole report unchanged against v39",
    all(a <= r for a, r in zip(bc(" ".join(add)), bc(" ".join(rem)))) and not any(BANRE.search(t) for t in parts.values()) and bc(s) == bc(v39), "(v39 %s, v40 %s)" % (bc(v39), bc(s)))
chk("no dollar sign added; math span counts as in v39", s.count("$") == v39.count("$") and s.count("$$") == v39.count("$$"), "(%d dollar signs)" % s.count("$"))
i83 = s.index("### 8.3 ")
ups = [s.index("**Update (v%d).**" % v, i83) for v in (40, 39, 38, 37)]
chk("8.3 has Update (v40) before Update (v39) before Update (v38) before Update (v37), each once", ups == sorted(ups) and all(s.count("**Update (v%d).**" % v) == 1 for v in (40, 39, 38, 37)))
i81, i82 = s.index("### 8.1 "), s.index("### 8.2 ")
nums = [int(x) for x in re.findall(r"^(\d+)\. \*\*", s[i81:i82], re.M)]
n39 = [int(x) for x in re.findall(r"^(\d+)\. \*\*", v39[v39.index("### 8.1 "):v39.index("### 8.2 ")], re.M)]
chk("8.1 ends with item 30: the v39 items (ending at 29) are untouched and item 30 follows", n39[-1] == 29 and nums == n39 + [30])
i9, i10 = s.index("## 9. "), s.index("## 10. Reproducibility appendix")
chk("section 9 ends with item 18 (v40 reads and their limits), after item 17", re.findall(r"^(\d+)\. \*\*", s[i9:i10], re.M)[-2:] == ["17", "18"] and "18. **v40 reads and their limits.**" in s[i9:i10])
rows = [s.index("| %s | " % r) for r in ("1ay", "1az")] + [s.index("| 2 | **LoRA on DINOv3**")]
chk("8.4 has row 1az once, after row 1ay and before row 2", rows == sorted(rows) and s.count("| 1az | ") == 1)
line = s.split("\n")[2]
chk("status line names v40 of 2026-11-06, the tip, §7.40, LF and A4; no date after 2026-11-06 anywhere", "v40 of 2026-11-06" in line and "`%s`" % TIP[:8] in line and all(x in line for x in ("§7.40", "level floor LF", "addendum A4"))
    and not [x for x in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if x > "2026-11-06"])
chk("no stale pointer: 'Its result is not in v39.' is gone and 7.39.7 points to 7.40; v39 is named as the previous version", "Its result is not in v39." not in s and "Its result is in §7.40." in s and "v39: `c9d0688e`" in line)
J = json.loads(show(R + "raw/2026-11-05-levelfloor-lf-result/lf_read/raw/LF_READ.json"))
want = []
for d in ("050", "075"):
    ru = J["doses"][d]["rules"]
    want.append("%.3f / %.3f" % tuple(ru["P"]["per_seed"][k]["share_closed"] for k in ("s0", "s1")))
    want.append("%.4f" % ru["F"]["per_seed"]["s0"]["abs_R_minus_C0"])
want.append("+%.4f / +%.4f" % tuple(J["doses"]["075"]["rules"]["G"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1")))
want.append("+%.4f" % J["doses"]["050"]["rules"]["G"]["per_seed"]["s0"]["ade6_delta"])
labels = {d: J["doses"][d]["outcome"] for d in ("050", "075")}
chk("added numbers agree with the raw LF JSON (share closed per dose, the F effect, ADE6) and both doses read P-FAILS-M1-MOVED", all(x in norm(sec) for x in want) and all("P-FAILS-M1-MOVED" in json.dumps(v) for v in labels.values()),
    "(%s)" % "; ".join(want))
lfs = open(HERE + "edits_v40.md", encoding="utf-8").read().split("=== LFS ===\n")[1].split("\n")[0]
chk("A4 is disclosed as a ceiling everywhere it is named: 7.40 says CEILING, inadmissible at inference and never a driving result; the shared LF sentence says ceiling diagnostic and never a driving result",
    all(x in norm(sec) for x in ("CEILING", "inadmissible at inference", "never a driving result")) and "ceiling diagnostic" in lfs and "never a driving result" in lfs and s.count(lfs) == 5)
files = [os.path.join(d, f).replace("\\", "/") for d, _, fs in os.walk(HERE) for f in fs if f.endswith((".py", ".md", ".sh", ".js", ".txt")) and "/out/" not in os.path.join(d, f).replace("\\", "/") + "/"]
crs = [os.path.relpath(f, HERE) for f in files if b"\r" in open(f, "rb").read()]
chk("LF only for every .py, .md, .sh, .js and .txt file of the package (CR counted in python)", not crs, "(%d files; with CR: %s)" % (len(files), crs[:3]))
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v40.py", TIP, td], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1"))
    same = r.returncode == 0 and all(open(os.path.join(td, f), "rb").read() == open(OUT + f, "rb").read() for f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html"))
    chk("patch_v40.py re-run reproduces both deliverables byte for byte", same, r.stdout.strip()[-100:].replace("\n", " ") + r.stderr.strip()[-200:])
nb = len(re.findall(r"\{\{", s)) + len(re.findall(r"\{\{", h))
chk("PLACEHOLDERS: no `{{` remains in out/ (md and html)", nb == 0, "(%d occurrences)" % nb)
say()
say("RESULT: %s (%d checks, %d failed%s)" % ("PASS" if not FAILS else "FAIL", sum(1 for x in L if x[:5] in ("PASS ", "FAIL ")), len(FAILS), "; failed: " + " | ".join(f[:40] for f in FAILS) if FAILS else ""))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print(L[-1])
sys.exit(1 if FAILS else 0)
