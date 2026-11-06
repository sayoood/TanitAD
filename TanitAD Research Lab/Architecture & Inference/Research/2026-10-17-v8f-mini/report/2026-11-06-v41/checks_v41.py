"""Independent checks of report v41 (the patch's own assertions have already passed to produce out/). Writes the checks file.
python -I checks_v41.py tip [out_dir [checks_path]]   (git, python; no network)"""
import hashlib, json, os, re, subprocess, sys, tempfile
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v41_checks.txt"
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


def norm_pipes(line):  # as patch_v41.py: a bare pipe inside a cell is escaped; delimiter = space on both sides, first or last character
    if re.fullmatch(r"\|[\s:\-|]+", line):
        return line
    n, o = len(line), []
    for i, ch in enumerate(line):
        o.append("\\|" if (ch == "|" and 0 < i < n - 1 and line[i - 1] != "\\" and not (line[i - 1] == " " and line[i + 1] == " ")) else ch)
    return "".join(o)


norm = lambda t: re.sub(r"\s+", " ", t)
RP40 = R + "report/2026-11-06-v40/"
v40 = show(RP40 + "V8F_SCIENTIFIC_REPORT.md")
s = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
h = open(OUT + "v8f_report.html", encoding="utf-8", newline="").read()
say("# report v41 checks, tip " + TIP)
say()
chk("v40 inputs are the landed deliverables (sha256)", sha(show(RP40 + "V8F_SCIENTIFIC_REPORT.md", True)) == "adb96d9c003990a412fb82c7a1a449f513c18883e9ec8d15b02e4f7c3ed5c285"
    and sha(show(RP40 + "v8f_report.html", True)) == "937cb19baebc258ca85d59658d457e45bf7e0997be2e8ab8f66f29da646a3f7d")
chk("html = page template with the v41 md", h == show(R + "report/2026-10-26/page_template_v30.html").replace("%%MD%%", s))
chk("LF only, no BOM (md and html)", "\r" not in s and "\ufeff" not in s and "\r" not in h and "\ufeff" not in h)
c40, c41 = Counter(v40.split("\n")), Counter(s.split("\n"))
rem = [l for l in (c40 - c41).elements() if l.strip()]
add = [l for l in (c41 - c40).elements() if l.strip()]
KEYS = ("**Status:** research report, v40 of 2026-11-06", "* **A recovery fine-tune moves M1 the right way and does not clear its guards;", "* **What is next.** The crawl set is still open", "  * A4's result is not in v40.")
hits = [sum(1 for l in rem if l.startswith(k)) for k in KEYS]
unexp = [l[:80] for l in rem if not any(l.startswith(k) for k in KEYS)]
chk("removed lines from v40: exactly the 4 edited lines, each key matched once (status line, abstract recovery bullet, abstract next bullet, the 7.40.6 A4 pointer)",
    not unexp and hits == [1] * len(KEYS) and len(rem) == 4, "(%d removed; unexpected %s; hits %s)" % (len(rem), unexp[:2], hits))
order = ["**7.40.7 ", "### 7.41 "] + ["**7.41.%d " % i for i in range(1, 8)] + ["## 8. Discussion"]
pos = [s.index(x) for x in order]
chk("order: 7.40.7 < 7.41 < 7.41.1 to 7.41.7 < section 8, each heading once", pos == sorted(pos) and all(s.count(x) == 1 for x in order))
i741, i8 = s.index("### 7.41 "), s.index("## 8. Discussion")
sec = s[i741:i8]
srcs = source_texts()
SL = set()
for t in srcs.values():
    SL |= {norm_pipes(l) for l in t.split("\n") if l.startswith("|")}
tl = [l for l in sec.split("\n") if l.startswith("|")]
odd = [l for l in add if l.startswith("|") and l not in SL]
dirs_total = sum(int(l.split(" | ")[4].split()[0]) for l in open(OUT + "DIRECTIVES.txt", encoding="utf-8").read().strip().split("\n"))
chk("every table line of 7.41 is a line of a level-floor source file (a bare pipe inside a cell escaped); the only other added table lines are rows 1ba, 1bb and 1bc",
    all(l in SL for l in tl) and len(tl) == dirs_total and len(odd) == 3 and [o.split(" | ")[0] for o in odd] == ["| 1ba", "| 1bb", "| 1bc"],
    "(%d table lines in 7.41 = %d expanded directive lines; %d source files)" % (len(tl), dirs_total, len(srcs)))
cols = lambda l: l.replace("\\|", "").count("|") - 1
i84 = s.index("### 8.4 ")
_r1az = [l for l in s[i84:].split("\n") if l.startswith("| 1az | ")][0]
chk("every table of 7.41 has one cell count; rows 1ba, 1bb and 1bc have the cell count of row 1az", all(len({cols(r) for r in blk}) == 1 for blk in
    [[x for x in b.split("\n") if x] for b in re.findall(r"(?:^\|.*\n?)+", sec, flags=re.M)]) and all(cols(o) == cols(_r1az) == 4 for o in odd))
reg = show("Project Steering/GOALS_AND_CLAIMS.md")
corp = norm(" ".join(list(srcs.values()) + [reg, v40]))
qs = [q.strip('"') for l in add for q in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", l)]
chk("italic quotations of 25+ characters in the added lines verbatim in a source (level-floor files, register, v40)", qs and all(norm(q) in corp for q in qs), "(%d quotations)" % len(qs))
fams = [len(re.findall(f, sec)) for f in ("LONGITUDINAL|DK-VIOL|DK-HW|along-track|speed MAE|accel MAE", "LATERAL|curvature|yaw-rate|heading|cross-track", "TACTICAL|TAC-LAT|TAC-LON", "STRATEGIC|S-POS")]
chk("each of the four metric families named in 7.41 at least 3 times", all(c >= 3 for c in fams), str(fams))
chk("T1-kin, the paired episode-cluster bootstrap, B = 2,000, eval-141 and H-ESTIM-SEED-1 stated in 7.41", all(x in norm(sec) for x in ("T1-kin", "paired episode-cluster bootstrap", "B = 2,000", "eval-141", "H-ESTIM-SEED-1")))
chk("evidence classes [M], [E] and [H] in 7.41", all(x in sec for x in ("[M]", "[E]", "[H]")))
cl = lambda t: len(CLRE.findall(t))
bc = lambda t: [len(re.findall(r"\b" + w + r"\b", t, re.I)) for w in BANNED]
parts = {n: open(HERE + n, encoding="utf-8", newline="").read() for n in ("sec_741_levelfloor.md", "edits_v41.md", "HANDOFF_V41.md", "msg_v41.txt") if os.path.exists(HERE + n)}
chk("the forbidden two-word T2 wording (either separator) is absent from every added line and every part file; the count of the carried v40 text is untouched (v40 %d, v41 %d)" % (cl(v40), cl(s)),
    cl(" ".join(add)) <= cl(" ".join(rem)) and not any(CLRE.search(t) for t in parts.values()) and cl(s) == cl(v40))
chk("banned-word scan: none new in the added lines (net of the 4 edited lines) or in the part files; per-word counts of the whole report unchanged against v40",
    all(a <= r for a, r in zip(bc(" ".join(add)), bc(" ".join(rem)))) and not any(BANRE.search(t) for t in parts.values()) and bc(s) == bc(v40), "(v40 %s, v41 %s)" % (bc(v40), bc(s)))
disp = re.findall(r"^\$\$\n(?:(?!\$\$).)*\n\$\$$", sec, flags=re.M | re.S)
chk("exactly one display equation added (the operator F_B), no other dollar sign; math span counts as in v40 plus that one equation",
    len(disp) == 1 and "\\delta" in disp[0] and "v_O" in disp[0] and "$" not in sec.replace(disp[0], "") and s.count("$$") == v40.count("$$") + 2 and s.count("$") == v40.count("$") + 4,
    "(%d dollar signs)" % s.count("$"))
i83 = s.index("### 8.3 ")
ups = [s.index("**Update (v%d).**" % v, i83) for v in (41, 40, 39, 38, 37)]
chk("8.3 has Update (v41) before Update (v40) before Update (v39) before Update (v38) before Update (v37), each once", ups == sorted(ups) and all(s.count("**Update (v%d).**" % v) == 1 for v in (41, 40, 39, 38, 37)))
i81, i82 = s.index("### 8.1 "), s.index("### 8.2 ")
nums = [int(x) for x in re.findall(r"^(\d+)\. \*\*", s[i81:i82], re.M)]
n40 = [int(x) for x in re.findall(r"^(\d+)\. \*\*", v40[v40.index("### 8.1 "):v40.index("### 8.2 ")], re.M)]
chk("8.1 ends with item 31: the v40 items (ending at 30) are untouched and item 31 follows", n40[-1] == 30 and nums == n40 + [31])
i9, i10 = s.index("## 9. "), s.index("## 10. Reproducibility appendix")
chk("section 9 ends with item 19 (v41 reads and their limits), after item 18", re.findall(r"^(\d+)\. \*\*", s[i9:i10], re.M)[-2:] == ["18", "19"] and "19. **v41 reads and their limits.**" in s[i9:i10])
rows = [s.index("| %s | " % r) for r in ("1az", "1ba", "1bb", "1bc")] + [s.index("| 2 | **LoRA on DINOv3**")]
chk("8.4 has rows 1ba, 1bb and 1bc once each, after row 1az and before row 2", rows == sorted(rows) and all(s.count("| %s | " % r) == 1 for r in ("1ba", "1bb", "1bc")))
line = s.split("\n")[2]
chk("status line names v41 of 2026-11-06, the tip, §7.41, A4 to A6; no date after 2026-11-06 anywhere", "v41 of 2026-11-06" in line and "`%s`" % TIP[:8] in line and all(x in line for x in ("§7.41", "addenda A4 to A6"))
    and not [x for x in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if x > "2026-11-06"])
chk("no stale pointer: 'A4's result is not in v40.' is gone and 7.40.6 points to 7.41.1; v40 is named as the previous version", "A4's result is not in v40." not in s and "  * A4's result is in §7.41.1." in s and "v40: `ffa2637a`" in line)
JO = json.loads(show(R + "raw/2026-11-06-levelfloor-a4-lfo-result/lfo_read/raw/LFO_READ.json"))
J5 = json.loads(show(R + "raw/2026-11-06-levelfloor-a5-lf5-result/lf5_read/raw/LF5_READ.json"))
J6 = json.loads(show(R + "raw/2026-11-06-levelfloor-a6-lf6-result/lf6_read/raw/LF6_READ.json"))
pair = lambda f: "%.3f / %.3f" % (f("s0"), f("s1"))
sg = lambda f, pre: pre + "%.4f / %s%.4f" % (abs(f("s0")), pre, abs(f("s1")))
want = [pair(lambda k: JO["arm"]["rules"]["P_O"]["per_seed"][k]["share_closed"]),
        pair(lambda k: J5["ceiling"]["LFO100"]["rules"]["P_O"]["per_seed"][k]["share_closed"]),
        pair(lambda k: J5["ceiling"]["LFN100x10"]["rules"]["P_O"]["per_seed"][k]["share_closed"]),
        pair(lambda k: J5["ceiling"]["LFN100x20"]["rules"]["P_O"]["per_seed"][k]["share_closed"]),
        pair(lambda k: J5["LF100"]["rules"]["P"]["per_seed"][k]["share_closed"]),
        pair(lambda k: J6["ceiling"]["LFOB05"]["rules"]["P_O"]["per_seed"][k]["share_closed"]),
        pair(lambda k: J6["ceiling"]["LFOB10"]["rules"]["P_O"]["per_seed"][k]["share_closed"]),
        pair(lambda k: J6["LFB"]["LFB05"]["rules"]["P"]["per_seed"][k]["share_closed"]),
        pair(lambda k: J6["LFB"]["LFB10"]["rules"]["P"]["per_seed"][k]["share_closed"]),
        sg(lambda k: J5["LF100"]["rules"]["G"]["per_seed"][k]["ade6_delta"], "+"),
        sg(lambda k: JO["arm"]["rules"]["G_O"]["per_seed"][k]["ade6_delta"], "-"),
        sg(lambda k: J6["ceiling"]["LFOB05"]["rules"]["G_O"]["per_seed"][k]["ade6_delta"], "-"),
        sg(lambda k: J6["ceiling"]["LFOB10"]["rules"]["G_O"]["per_seed"][k]["ade6_delta"], "-")]
labels = [JO["arm"]["outcome"]["label"], J5["joint"]["case"], J5["LF100"]["outcome"]["label"], J6["joint"]["case"]] + [J6["ceiling"][a]["outcome"]["label"] for a in ("LFOB05", "LFOB10")] + [J6["LFB"][a]["outcome"]["label"] for a in ("LFB05", "LFB10")]
secm = norm(sec).replace("\u2212", "-")
chk("added numbers agree with the raw LFO, LF5 and LF6 JSON (shares closed, ADE6 deltas), the labels and joint readings are the JSON's, and no counted DOMINANT or estimator signal in either joint reading",
    all(x in secm for x in want) and all(x in sec for x in labels) and not J5["joint"]["counted_dominant"] and not J6["joint"]["counted_dominant"]
    and not J5["joint"]["estimator_levers_signalled"] and not J6["joint"]["estimator_levers_signalled"], "(%s)" % "; ".join(want[:5]))
v41s = open(HERE + "edits_v41.md", encoding="utf-8").read().split("=== V41S ===\n")[1].split("\n")[0]
chk("the ceiling arms are disclosed as inadmissible everywhere they are named: 7.41 says CEILING, inadmissible at inference and never a driving result; the shared v41 sentence says ceiling diagnostic, inadmissible at inference and never driving results; it occurs 5 times",
    all(x in norm(sec) for x in ("CEILING", "inadmissible at inference", "never a driving result")) and "ceiling diagnostic" in v41s and "inadmissible at inference" in v41s and "never driving results" in v41s and s.count(v41s) == 5)
files = [os.path.join(d, f).replace("\\", "/") for d, _, fs in os.walk(HERE) for f in fs if f.endswith((".py", ".md", ".sh", ".js", ".txt")) and "/out/" not in os.path.join(d, f).replace("\\", "/") + "/"]
crs = [os.path.relpath(f, HERE) for f in files if b"\r" in open(f, "rb").read()]
chk("LF only for every .py, .md, .sh, .js and .txt file of the package (CR counted in python)", not crs, "(%d files; with CR: %s)" % (len(files), crs[:3]))
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v41.py", TIP, td], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1"))
    same = r.returncode == 0 and all(open(os.path.join(td, f), "rb").read() == open(OUT + f, "rb").read() for f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html"))
    chk("patch_v41.py re-run reproduces both deliverables byte for byte", same, r.stdout.strip()[-100:].replace("\n", " ") + r.stderr.strip()[-200:])
# an independent recount of the family lines (the patch computed them from the table lines with its own function; this re-derives them with another parse of the rendered section)
secl = sec.split("\n")
bad, nfl = [], 0
for i, l in enumerate(secl):
    if not l.startswith("* **Families, "):
        continue
    nfl += 1
    j = i - 1
    while j >= 0 and not secl[j].startswith("|"):
        j -= 1
    k = j
    while k >= 0 and secl[k].startswith("|"):
        k -= 1
    tab = secl[k + 1:j + 1]
    cnt = {}
    for r in tab[2:]:
        cells = [c.strip() for c in r.strip("| ").split(" | ")]
        fam = cells[1]
        if fam in ("PRIMARY", "crawl"):
            continue
        for sd, cell in (("s0", cells[2]), ("s1", cells[3])):
            sep = cell.endswith("*")
            val = cell.split()[0]
            worse = (val[0] == "+") != (cells[0] in ("TAC-LAT", "TAC-LON"))
            c = cnt.setdefault((fam, sd), [0, 0, 0])
            c[2] += 1
            if sep:
                c[1 if worse else 0] += 1
    got = {}
    for m in re.finditer(r"(LONGITUDINAL|LATERAL|TACTICAL|STRATEGIC) (\d+) / (\d+) \(s0\) and (\d+) / (\d+) \(s1\) of (\d+)", l):
        got[m.group(1)] = tuple(int(x) for x in m.groups()[1:])
    exp = {f: (cnt[(f, "s0")][0], cnt[(f, "s0")][1], cnt[(f, "s1")][0], cnt[(f, "s1")][1], cnt[(f, "s0")][2]) for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}
    if got != exp or len(tab) != 21:
        bad.append(l[:60])
chk("each family-count line of 7.41 (%d lines) equals an independent recount of the table above it" % nfl, nfl == 9 and not bad, "(mismatch: %s)" % bad[:2])
nb = len(re.findall(r"\{\{", s)) + len(re.findall(r"\{\{", h))
chk("PLACEHOLDERS: no `{{` remains in out/ (md and html)", nb == 0, "(%d occurrences)" % nb)
say()
say("RESULT: %s (%d checks, %d failed%s)" % ("PASS" if not FAILS else "FAIL", sum(1 for x in L if x[:5] in ("PASS ", "FAIL ")), len(FAILS), "; failed: " + " | ".join(f[:40] for f in FAILS) if FAILS else ""))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print(L[-1])
sys.exit(1 if FAILS else 0)
