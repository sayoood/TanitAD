"""Independent checks of report v43 (the patch's own assertions have already passed to produce out/). Writes the checks file.
python -I checks_v43.py tip [out_dir [checks_path]]   (git, python; no network)"""
import hashlib, json, os, re, subprocess, sys, tempfile
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v43_checks.txt"
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


def rawb(p):
    return git(["show", f"{TIP}:{R}raw/2026-11-07-{p}"])  # a file of a 2026-11-07 raw folder, bytes


def rawt(p):
    return rawb(p).decode("utf-8-sig").replace("\r", "")


def rawj(p):
    return json.loads(rawt(p))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def chk(name, cond, extra=""):
    say(("PASS " if cond else "FAIL ") + name + ((" " + extra) if extra else ""))
    if not cond:
        FAILS.append(name)


def source_texts():
    """every text blob (md, txt, json, up to 3 MB) under raw/2026-11-02-* to raw/2026-11-07-* at the tip, via one git cat-file --batch"""
    names = [n for n in git(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"], False).split("\n")
             if re.search(r"/2026-11-0[2-7]-", n) and n.endswith((".md", ".txt", ".json"))]
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


def norm_pipes(line):  # as patch_v43.py: a bare pipe inside a cell is escaped; delimiter = space on both sides, first or last character
    if re.fullmatch(r"\|[\s:\-|]+", line):
        return line
    n, o = len(line), []
    for i, ch in enumerate(line):
        o.append("\\|" if (ch == "|" and 0 < i < n - 1 and line[i - 1] != "\\" and not (line[i - 1] == " " and line[i + 1] == " ")) else ch)
    return "".join(o)


norm = lambda t: re.sub(r"\s+", " ", t)
MN = lambda t: t.replace("\u2212", "-")          # the sources print a Unicode minus in prose and an ASCII hyphen in tables
nt = lambda t: norm(MN(t)).replace("*", "")      # one comparable form: ASCII minus, single spaces, no bold markers
RP42 = R + "report/2026-11-07-v42/"
v42 = show(RP42 + "V8F_SCIENTIFIC_REPORT.md")
s = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
h = open(OUT + "v8f_report.html", encoding="utf-8", newline="").read()
say("# report v43 checks, tip " + TIP)
say()
chk("v42 inputs are the landed deliverables (sha256)", sha(show(RP42 + "V8F_SCIENTIFIC_REPORT.md", True)) == "ada2083063ed8bebcac82ee64444d8760b2766c8bda643c51a8d1af96c637aea"
    and sha(show(RP42 + "v8f_report.html", True)) == "0bf58200cade629f71c58ebba976c70f16c119a35d74a0c558447c8a8dc04508")
chk("html = page template with the v43 md", h == show(R + "report/2026-10-26/page_template_v30.html").replace("%%MD%%", s))
chk("LF only, no BOM (md and html)", "\r" not in s and "\ufeff" not in s and "\r" not in h and "\ufeff" not in h)
c42, c43 = Counter(v42.split("\n")), Counter(s.split("\n"))
rem = [l for l in (c42 - c43).elements() if l.strip()]
add = [l for l in (c43 - c42).elements() if l.strip()]
# the carried-line edits, exactly as edits_v43.md states them (old text --> new text); the removed lines must be exactly the v42 lines that contain an old text
_bl = re.split(r"^=== (\w+) ===\n", open(HERE + "edits_v43.md", encoding="utf-8", newline="").read(), flags=re.M)
EB = {_bl[i]: _bl[i + 1].strip("\n") for i in range(1, len(_bl), 2)}
CARRIED = ["C3A", "C3B", "C42S", "C63", "C64", "C5951", "C6030", "C6036", "C6040", "C6118", "C6247", "C6300"]
pairs = {k: tuple(x.replace("{{TP}}", TIP[:8]) for x in EB[k].split("\n-->\n")) for k in CARRIED}
exp_rem = Counter(l for l in v42.split("\n") if l.strip() and any(a in l for a, _ in pairs.values()))
hits = {k: sum(1 for l in rem if pairs[k][0] in l) for k in CARRIED}
want_hits = {k: (2 if k == "C42S" else 1) for k in CARRIED}


def _apply(line):
    for a, b in pairs.values():
        line = line.replace(a, b)
    return line


v43lines = s.split("\n")
newlines = all(any(x.startswith(_apply(l)) for x in v43lines) for l in rem)  # two edited lines (the status line, the abstract bullet) also get a sentence appended by a later block
chk("removed lines from v42: exactly the v42 lines that contain the old text of the 12 edit blocks (C3A, C3B, C42S, C63, C64, C5951, C6030, C6036, C6040, C6118, C6247, C6300; C42S occurs twice, in the status line and in 8.1 item 32), "
    "each block matched as listed, and each edited line is present in v43 (as a line or as the start of a line that a later block extends)",
    Counter(rem) == exp_rem and hits == want_hits and newlines, "(%d removed lines; hits %s)" % (len(rem), [hits[k] for k in CARRIED]))
order = ["**7.44.9 ", "### 7.45 "] + ["**7.45.%d " % i for i in range(1, 9)] + ["### 7.46 "] + ["**7.46.%d " % i for i in range(1, 12)] + ["## 8. Discussion"]
pos = [s.index(x) for x in order]
chk("order: 7.44.9 < 7.45 < 7.45.1 to 7.45.8 < 7.46 < 7.46.1 to 7.46.11 < section 8, each heading once", pos == sorted(pos) and all(s.count(x) == 1 for x in order))
i742, i745, i746, i8 = s.index("### 7.42 "), s.index("### 7.45 "), s.index("### 7.46 "), s.index("## 8. Discussion")
sec = s[i742:i8]                      # 7.42 to 7.46
sec742 = s[i742:s.index("### 7.43 ")]
sec743 = s[s.index("### 7.43 "):s.index("### 7.44 ")]
sec744 = s[s.index("### 7.44 "):i745]
sec745, sec746 = s[i745:i746], s[i746:i8]
n745, n746 = nt(sec745), nt(sec746)
srcs = source_texts()
SL = set()
for n, t in srcs.items():
    if n.endswith((".md", ".txt")):
        SL |= {norm_pipes(l) for l in t.split("\n") if l.startswith("|")}
blks = [[x for x in b.split("\n") if x] for b in re.findall(r"(?:^\|.*\n?)+", sec, flags=re.M)]
src_blks = [b for b in blks if all(l in SL for l in b)]
hand_blks = [b for b in blks if not all(l in SL for l in b)]
_dl = lambda t: sum(int(l.split(" | ")[4].split()[0]) for l in t.strip().split("\n"))
dirs_total = _dl(show(RP42 + "DIRECTIVES.txt")) + _dl(open(OUT + "DIRECTIVES.txt", encoding="utf-8").read())
app = s[s.index("**Reads added in v43"):s.index("**Figures.**")]
app_blk = [l for l in app.split("\n") if l.startswith("|")]
rows6 = [l for l in s.split("\n") if l.startswith(tuple("| %s | " % r for r in ("1bg", "1bh", "1bi", "1bj", "1bk", "1bl")))]
chk("table lines of 7.42 to 7.46: the 8 directive tables (v42's 4 and v43's 4) are blocks of source lines only (a bare pipe inside a cell escaped) and number as many lines as the directives expanded (v42's DIRECTIVES.txt and v43's); the other 12 tables of the sections are hand-written; "
    "the only added table lines outside the sections are rows 1bg to 1bl (6, not source lines) and the appendix table (23 lines)",
    len(src_blks) == 8 and sum(len(b) for b in src_blks) == dirs_total and len(hand_blks) == 12 and len(rows6) == 6 and not any(r in SL for r in rows6) and len(app_blk) == 23 and app_blk[2] not in SL,
    "(%d source table lines in %d blocks = %d expanded directive lines; %d hand-written tables with %d lines; %d source files)" % (sum(len(b) for b in src_blks), len(src_blks), dirs_total, len(hand_blks), sum(len(b) for b in hand_blks), len(srcs)))
cols = lambda l: l.replace("\\|", "").count("|") - 1
i84 = s.index("### 8.4 ")
_r1az = [l for l in s[i84:].split("\n") if l.startswith("| 1az | ")][0]
chk("every table of 7.42 to 7.46 has one cell count and a header rule (20 tables); rows 1bg to 1bl have the cell count of row 1az (4)",
    len(blks) == 20 and all(len({cols(r) for r in blk}) == 1 and re.fullmatch(r"\|[\s:\-|]+", blk[1]) for blk in blks) and all(cols(o) == cols(_r1az) == 4 for o in rows6), "(%d tables)" % len(blks))
reg = show("Project Steering/GOALS_AND_CLAIMS.md")
brief = open(HERE + "BUILD_BRIEF_V43.md", encoding="utf-8").read()   # the brief records the PI's message of 2026-11-07 verbatim (the one quotation that is in no repository file)
corp = norm(" ".join(list(srcs.values()) + [reg, v42, brief]).replace("**", ""))
qs = [q.strip('"').replace("\u2217", "*") for l in add for q in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", l.replace("\\*", "\u2217"))]
chk("italic quotations of 25+ characters in the added lines verbatim in a source (raw files of 2026-11-02 to 2026-11-07, register, v42, the brief), compared modulo the bold markers `**`, an escaped star counting as a literal star",
    qs and all(norm(q) in corp for q in qs), "(%d quotations)" % len(qs))
fams = [len(re.findall(f, sec742)) for f in ("LONGITUDINAL|along-track|speed MAE|accel MAE", "LATERAL|curvature|yaw-rate|heading|cross-track", "TACTICAL|TAC-LAT|TAC-LON", "STRATEGIC|S-POS")]
chk("each of the four metric families named in 7.42 at least 3 times (STRATEGIC shown as PAUSED by ITEM 25), and in 7.45 (all four named)", all(c >= 3 for c in fams) and "PAUSED" in sec742 and "ITEM 25" in sec742 and all(f in sec745 for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")), str(fams))
chk("tier T2, the paired scene-cluster bootstrap, B = 2,000, H-ESTIM-SEED-1 and the rule R-SCN stated in 7.42; the tier is never pooled with T1-kin and agents replay; 7.45 states the same estimator, B = 2,000, H-ESTIM-SEED-1 and that agents replay",
    all(x in norm(sec742) for x in ("T2", "scene-cluster bootstrap", "B = 2,000", "H-ESTIM-SEED-1", "R-SCN", "never pooled with T1-kin", "replay")) and all(x in norm(sec745) for x in ("scene-cluster", "B = 2,000", "H-ESTIM-SEED-1", "replay")))
chk("evidence classes [M], [E], [H] in 7.42 to 7.46 and [P] in 7.44", all(x in sec for x in ("[M]", "[E]", "[H]")) and "[P]" in sec744)
cl = lambda t: len(CLRE.findall(t))
bc = lambda t: [len(re.findall(r"\b" + w + r"\b", t, re.I)) for w in BANNED]
parts = {n: open(HERE + n, encoding="utf-8", newline="").read() for n in ("sec_745_t2attrib.md", "sec_746_theory.md", "edits_v43.md", "HANDOFF_V43.md") if os.path.exists(HERE + n)}
chk("the forbidden two-word T2 wording (either separator) is absent from every added line and every part file; the count of the carried text falls, never rises (v42 %d lines, v43 %d)" % (cl(v42), cl(s)),
    cl(" ".join(add)) == 0 and not any(CLRE.search(t) for t in parts.values()) and cl(s) <= cl(v42))
chk("banned-word scan: none new in the added lines (net of the removed carried lines) or in the part files; per-word counts of the whole report at most the v42 counts",
    all(a <= r for a, r in zip(bc(" ".join(add)), bc(" ".join(rem)))) and not any(BANRE.search(t) for t in parts.values()) and all(a <= b for a, b in zip(bc(s), bc(v42))), "(v42 %s, v43 %s)" % (bc(v42), bc(s)))
# math: strip code, display equations, then the template's inline-math pattern; no dollar sign may remain in the sections; the display equations are balanced
nc = re.sub(r"`[^`\n]+`", " ", sec)
disp = re.findall(r"\$\$[\s\S]+?\$\$", nc)
rest = re.sub(r"\$\$[\s\S]+?\$\$", " ", nc)
inl = re.findall(r"(?:^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", rest, flags=re.M)
rest2 = re.sub(r"(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", r"\1 ", rest, flags=re.M)
sec42 = v42[v42.index("### 7.42 "):v42.index("## 8. Discussion")]
chk("math spans: 10 display equations in 7.42 to 7.46 (v42's 5 and 5 new; even $$ count), inline math matches the template pattern, no dollar sign remains outside math in the sections, and the span counts outside the sections are those of v42",
    len(disp) == 10 and sec.count("$$") == 20 and "$" not in rest2 and len(inl) > 100 and s.count("$$") == v42.count("$$") + 10 and s.count("$") - sec.count("$") == v42.count("$") - sec42.count("$"),
    "(%d inline spans, %d dollar signs in the sections)" % (len(inl), sec.count("$")))
i83 = s.index("### 8.3 ")
ups = [s.index("**Update (v%d).**" % v, i83) for v in (43, 42, 41, 40, 39, 38, 37)]
chk("8.3 has Update (v43) before Update (v42) before Update (v41) to (v37), each once", ups == sorted(ups) and all(s.count("**Update (v%d).**" % v) == 1 for v in (43, 42, 41, 40, 39, 38, 37)))
i81, i82 = s.index("### 8.1 "), s.index("### 8.2 ")
nums = [int(x) for x in re.findall(r"^(\d+)\. \*\*", s[i81:i82], re.M)]
n42 = [int(x) for x in re.findall(r"^(\d+)\. \*\*", v42[v42.index("### 8.1 "):v42.index("### 8.2 ")], re.M)]
chk("8.1 ends with item 33: the v42 items (ending at 32) are numbered as before and item 33 follows", n42[-1] == 32 and nums == n42 + [33])
i9, i10 = s.index("## 9. "), s.index("## 10. Reproducibility appendix")
chk("section 9 ends with item 21 (v43 reads and their limits), after item 20", re.findall(r"^(\d+)\. \*\*", s[i9:i10], re.M)[-2:] == ["20", "21"] and "21. **v43 reads and their limits.**" in s[i9:i10])
rows = [s.index("| %s | " % r) for r in ("1bf", "1bg", "1bh", "1bi", "1bj", "1bk", "1bl")] + [s.index("| 2 | **LoRA on DINOv3**")]
chk("8.4 has rows 1bg to 1bl once each, after row 1bf and before row 2", rows == sorted(rows) and all(s.count("| %s | " % r) == 1 for r in ("1bg", "1bh", "1bi", "1bj", "1bk", "1bl")))
line = s.split("\n")[2]
chk("status line names v43 of 2026-11-07, the tip, §7.42 to §7.46, v42 and v41 as the previous versions; no date after 2026-11-07 anywhere",
    "v43 of 2026-11-07" in line and "`%s`" % TIP[:8] in line and all(x in line for x in ("§7.42", "§7.43", "§7.44", "§7.45", "§7.46", "v42: `1887e9b8`", "v41: `bd98063d`", "T2 re-perception"))
    and not [x for x in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if x > "2026-11-07"])
chk("no stale T2 statement: the status line's 'no (two-word) (T2) number exists' and the tier line 'T2 (two-word), unprovisioned' are gone; the status wording names 48 scenes and NOT-DOMINANT; 7.41.6 (b) carries the 7.42 pointer",
    ("**no " + "pool-desolc"[::-1] + " (T2) number exists**") not in s and ("T2 " + "pool desolc"[::-1] + ", unprovisioned") not in s and "no reactive-traffic number exists" in line and "48 NuRec scenes" in line and "NOT-DOMINANT" in line
    and "That read exists as of v42 (§7.42" in s)
DJ = json.loads(show(R + "raw/2026-11-06-t2-dominance-read/stage4/score/DOMINANCE_T2.json"))
RJ = json.loads(show(R + "raw/2026-11-06-t2-crawl-read/stage2/READ_STAGE2.json"))
FT = DJ["decision"]["failing_terms"]
secn = norm(sec742).replace("\u2212", "-")
want = ["%+.4f [%+.4f, %+.4f]" % tuple(RJ[x][sd][k] for k in ("delta", "lo", "hi")) for x in ("X1", "X2", "BL") for sd in ("0", "1")]
want += ["%+.4f" % DJ["by_seed"][sd]["rows"]["M1 crawl share"]["delta"] for sd in ("s0", "s1")] + ["%+.4f [%+.4f, %+.4f]" % tuple(DJ["by_seed"][sd]["rows"]["ADE6"][k] for k in ("delta", "lo", "hi")) for sd in ("s0", "s1")]
chk("carried 7.42 numbers still agree with the raw READ_STAGE2 and DOMINANCE_T2 JSON (X1, X2, BL with bounds; M1 crawl share and ADE6 with bounds), the outcome is NOT-DOMINANT at both seeds, and the failing_terms are quoted verbatim",
    all(x.replace("\u2212", "-") in secn for x in want) and DJ["outcome"] == "NOT-DOMINANT" and "NOT-DOMINANT" in sec742 and all(t in sec742 for sd in ("s0", "s1") for t in FT[sd]) and RJ["outcome"] == "CRAWL-PERSISTS" and "CRAWL-PERSISTS" in sec742
    and not any(DJ["decision"]["per_seed"][sd][d] for sd in ("s0", "s1") for d in ("DOMINANT", "D1", "D2", "D3")), "(%d numbers)" % len(want))
chk("7.44 reads as in v42, with the v43 pointers: P1 FAILS as written, P2 HOLDS as written, P3 FAILS as written (all 26 dominating pairs with a CEILING dominator), P1' NOT EVALUABLE as pre-registered at v42 with the pointer to P1'b (FAILS, 7.46.2), P4 not evaluable, "
    "WSP-GAIN built and unread at v42 with the pointer to its read (HARMS in 4 of 4 cells, 7.46.3); the v42 sentence 'No WSP-GAIN result exists' is gone from 7.44",
    all(x in sec744 for x in ("**P1 FAILS as written [M]**", "**P2 HOLDS as written [M]**", "**P3 FAILS as written [M]**", "matched 2 of 4", "all 26", "NOT EVALUABLE as pre-registered, so P1′ is neither held nor failed", "24 windows (floor 30)", "NOT EVALUABLE",
                              "pre-registered, built and unit-tested, not yet read", "No WSP-GAIN result existed at v42", "§7.46.3 (HARMS in 4 of 4 cells)", "P1′b on 60 holdout clips: FAILS, §7.46.2"))
    and "No WSP-GAIN result exists" not in sec744)
chk("7.43 states what is not known (as of v42): no LoRA result, the batch-64 timing not yet measured, the pod handoffs E: files not yet landed, the base is B' and the device alone misses the bar",
    all(x in norm(sec743) for x in ("no LoRA result", "not yet measured", "not yet landed", "B′", "21 to 23 times", "2.11 to 2.25e-2")) and "No LoRA" in sec)
_v43s = open(HERE + "edits_v43.md", encoding="utf-8").read().split("=== V43S ===\n")[1].split("\n")[0]
chk("the shared v43 sentence occurs exactly twice (status line and 8.1 item 33) and names no driving result: no LoRA driving result exists, no DOMINANT, no scale signal, nothing in v43 is a driving result (the v42 sentence before it is kept)",
    s.count(_v43s) == 2 and "nothing in v42 is a driving result" in s and all(x in _v43s for x in ("no LoRA driving result exists", "No DOMINANT exists", "no scale signal", "nothing in v43 is a driving result")),
    "(sentence occurrences %d)" % s.count(_v43s))
chk("the related-work paragraph cites Library key takehi2026weightspace, the banked page sha256 a326bc5b, the 2026-11-02 blog post as not peer-reviewed [P], and does not claim the method generalises; it sits in section 3",
    s.count("takehi2026weightspace") >= 3 and s.index("**Weight-space learning and reading (v42).**") < s.index("## 4. Method") and "we do not claim the method generalises" in s and "a blog post, not peer-reviewed" in s[s.index("**Weight-space learning"):s.index("## 4. Method")])
_a42 = s[s.index("**Reads added in v42"):s.index("**Reads added in v43")]
chk("the appendix block 'Reads added in v43' is in section 10 before the Figures paragraph and names all 21 new raw folders (one table row each); the v42 block 'Reads added in v42' is kept and names its 8",
    s.index("## 10. Reproducibility appendix") < s.index("**Reads added in v43") < s.index("**Figures.**")
    and len(re.findall(r"^\| `2026-11-07-[^`]+/` \|", app, re.M)) == 21 and len(set(re.findall(r"^\| `(2026-11-07-[^`]+/)` \|", app, re.M))) == 21
    and all(("`2026-11-06-%s" % f) in _a42 for f in ("t2-harness-v8f/", "t2-stage1-box2/", "t2-crawl-read/", "t2-stage4-c0c/", "t2-dom-repin/", "t2-dominance-read/", "lora-pod-amend4/", "theory-crawl/")))
# the P1' numbers of 7.44.2 and the 7.44.7 row come from an E: file that has not landed; tie each to that file (the audit's first-match layer could attribute a coincidental tip-file match)
EP = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read/P1P_RESULT.md"
p1p = norm(open(EP, encoding="utf-8").read().replace("\u2212", "-")) if os.path.exists(EP) else ""
p1_nums = ["0.856", "0.491", "0.960", "0.523", "0.365 [0.213, 0.534]", "0.437 [0.240, 0.585]", "0.782 [0.703, 0.853]", "0.750 [0.673, 0.825]", "6.0e-07", "73640c67", "24 windows (floor 30)"]
chk("the P1' numbers of 7.44 (group sizes, median rho by group, the two differences with bounds, the probe AUCs, the refit tolerance, the spec sha prefix) each occur in the E: read P1P_RESULT.md and in 7.44",
    bool(p1p) and all(x in p1p and x in sec744 for x in p1_nums) and "168 windows from 45 episodes" in sec744 and "| 168 |" in p1p and "(45 episodes)" in p1p and "UNSURE group 75" in sec744 and "UNSURE holds 75" in p1p.replace("the UNSURE group holds 75", "UNSURE holds 75"),
    "(E: file %s)" % ("present" if p1p else "ABSENT"))
EQ = "E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/HANDOFF_Q.md"
hq = norm(open(EQ, encoding="utf-8").read()) if os.path.exists(EQ) else ""
chk("the read re-pin of 7.42.6 (fourth pin 7fd65ffd, identical to the read of record on the real pulls, NOT-DOMINANT with 48 scenes and 93 starts, 80 of 80 mutations RED) occurs in the E: file HANDOFF_Q.md and in 7.42; the read of record still cites the third pin d6664718",
    bool(hq) and all(x in hq and x in sec742 for x in ("7fd65ffd", "80 of 80", "NOT-DOMINANT, 48 scenes, 93 starts")) and "identical: true" in hq and "d6664718" in sec742 and "d6664718" in hq, "(E: file %s)" % ("present" if hq else "ABSENT"))
# the two new section files are inserted as written: every non-directive line of each file occurs in the report, in order
ordered = True
for fn in ("sec_745_t2attrib.md", "sec_746_theory.md"):
    p = 0
    for ln in open(HERE + fn, encoding="utf-8").read().strip("\n").split("\n"):
        if ln.startswith("@@T ") or not ln.strip():
            continue
        pat_ln = re.compile("[0-9a-f]{7,8}".join(re.escape(x) for x in re.split(r"\{\{[^{}\n]*\}\}", ln)))
        m = pat_ln.search(s, p)
        if not m:
            ordered = False; print("section line not found:", fn, ln[:80]); break
        p = m.end()
chk("the two section files (sec_745_t2attrib.md, sec_746_theory.md) appear in the report line for line and in order (directive lines replaced by their tables)", ordered)
# privacy: no address, port or pod id in anything the package writes
pat = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|root@|\bssh\s+-|\bport\s+\d{2,5}\b|pod[ _-]?id|:\d{4,5}\b(?!-)", re.I)
leak = [(n, m.group(0)) for n, t in list({"report (added lines)": "\n".join(add), **parts}.items()) for m in pat.finditer(t) if not re.match(r"\d{4}-", m.group(0))]
_sup = rawt("t2-ralign-gap48/lora_L_launch/L_s0/supervisor.log")
_cid = re.search(r"supervisor UP on (\S+)", _sup).group(1)   # the container id the supervisor log carries: it must be in no output
chk("no pod address, port or pod id in the report or the part files (IPv4, root@, ssh -, 'port N', 'pod id', :NNNN), and the container id of the supervisor log occurs in neither the md, the html nor any part file",
    not leak and len(_cid) > 6 and _cid not in s and _cid not in h and not any(_cid in t for t in parts.values()), "(%s)" % leak[:3])
files = [os.path.join(d, f).replace("\\", "/") for d, _, fs in os.walk(HERE) for f in fs if f.endswith((".py", ".md", ".sh", ".js", ".txt")) and "/out/" not in os.path.join(d, f).replace("\\", "/") + "/"
         and "/scratch/" not in os.path.join(d, f).replace("\\", "/") and ".before_" not in f]
crs = [os.path.relpath(f, HERE) for f in files if b"\r" in open(f, "rb").read()]
chk("LF only for every .py, .md, .sh, .js and .txt file of the package (CR counted in python; scratch/ and the moved-aside *.before_* copies are excluded)", not crs, "(%d files; with CR: %s)" % (len(files), crs[:3]))

# ---- v43: each headline number of 7.45 and 7.46 tied to its raw file on the tip -------------------------------------------------------------------------
rawcorp = " ".join(srcs.values())
TB = rawj("t1-blind/RESULT_T1BLIND.json")
tb = ["%+.4f [%+.4f, %+.4f]" % tuple(TB["seeds"][sd]["ADE6_B"][k] for k in ("delta", "lo", "hi")) for sd in ("s0", "s1")]
tbmd = nt(rawt("t1-blind/RESULT_T1BLIND.md"))
chk("VISION-USED (T1-kin BLIND): RESULT_T1BLIND.json outcome VISION-USED; ADE6 B +0.8313 [+0.5576, +1.1530] and +0.8158 [+0.5309, +1.1454] (both separated); F = 0.12074; the result md and 7.45 hold the same intervals and the outcome word",
    TB["outcome"] == "VISION-USED" and tb == ["+0.8313 [+0.5576, +1.1530]", "+0.8158 [+0.5309, +1.1454]"] and all(TB["seeds"][sd]["ADE6_B"]["separated"] for sd in ("s0", "s1")) and "%.5f" % TB["F_m"] == "0.12074"
    and all(x in tbmd and x in n745 for x in tb) and "VISION-USED" in tbmd and "VISION-USED" in n745 and "0.12074" in n745, "(%s)" % tb)
MCJ, BPJ = rawj("t2-mc-read/stage5_MC/score_dom/DOMINANCE_T2.json"), rawj("t2-bpc-read/score_dom/DOMINANCE_T2.json")
_row = lambda J, sd, r: J["by_seed"][sd]["rows"][r]
mcw = ["%+.4f [%+.4f, %+.4f]" % tuple(_row(MCJ, sd, "ADE6")[k] for k in ("delta", "lo", "hi")) for sd in ("s0", "s1")] + ["%+.4f" % _row(MCJ, sd, "M1 crawl share")["delta"] for sd in ("s0", "s1")]
bpw = ["%+.4f" % _row(BPJ, sd, "ADE6")["delta"] for sd in ("s0", "s1")] + ["%+.4f" % _row(BPJ, sd, "M1 crawl share")["delta"] for sd in ("s0", "s1")]
chk("PT/MC and BPC reads: MC alone NOT-DOMINANT (ADE6 +1.4552 / +0.9463 with bounds, M1 crawl share +0.3493 / +0.2919) and BPC NOT-DOMINANT (ADE6 +1.5230 / +1.4097, M1 +0.3741 / +0.3429), each number in its DOMINANCE_T2.json and in 7.45; "
    "C0C's M1 crawl share +0.4073 / +0.4041 (the stage-4 read of record) less MC's gives the 0.06 / 0.11 the text states",
    MCJ["outcome"] == "NOT-DOMINANT" and BPJ["outcome"] == "NOT-DOMINANT" and mcw[2:] == ["+0.3493", "+0.2919"] and [x.split(" ")[0] for x in mcw[:2]] == ["+1.4552", "+0.9463"] and bpw == ["+1.5230", "+1.4097", "+0.3741", "+0.3429"]
    and all(x in n745 for x in mcw + bpw) and ["%+.4f" % DJ["by_seed"][sd]["rows"]["M1 crawl share"]["delta"] for sd in ("s0", "s1")] == ["+0.4073", "+0.4041"]
    and ["%.2f" % (DJ["by_seed"][sd]["rows"]["M1 crawl share"]["delta"] - _row(MCJ, sd, "M1 crawl share")["delta"]) for sd in ("s0", "s1")] == ["0.06", "0.11"] and "0.06 / 0.11" in n745, "(%s)" % (mcw + bpw))
RT = rawj("t2-bpc-read/RT_VERDICT.json")
mg = {r: v["mc_seed_floor"] - abs(v["rt_s1"]["delta"]) for r, v in RT["rows"].items()}
mgs = ["%.4f" % mg["M1 crawl share"], "%.3f" % mg["ADE6"], "%.3f" % mg["along-track MAE"]]
chk("RT-WITHIN-FLOOR: RT_VERDICT.json verdict RT-WITHIN-FLOOR; at s0 no row separated, at s1 all three rows separated and each within MC's seed floor; the s1 margins (floor less |RT_s1|) are 0.0064, 0.045 and 0.038, "
    "from the unrounded JSON values (ADE6 0.04545), and the text states them; the three floors 0.0574, 0.5089, 0.4390 are in the text",
    RT["verdict"] == "RT-WITHIN-FLOOR" and set(RT["rows"]) == {"M1 crawl share", "ADE6", "along-track MAE"} and all(not v["rt_s0"]["separated"] and v["rt_s1"]["separated"] and v["within"] and v["abs_below_floor_both_seeds"] for v in RT["rows"].values())
    and mgs == ["0.0064", "0.045", "0.038"] and "by 0.0064, 0.045 and 0.038" in n745 and "RT-WITHIN-FLOOR" in n745 and all(("%.4f" % v["mc_seed_floor"]) in n745 for v in RT["rows"].values()), "(margins %s)" % mgs)
FP = rawj("lora-fprime/out/FPRIME.json")
chk("F-prime: FPRIME.json F′ 0.12701 m (0.12701453), 2F′ 0.2540 m, F 0.12074 m, 2F 0.2415 m; 2F′ is twice F′; the text states F′ = 0.12701 m, 2F′ = 0.2540 m, F = 0.12074 m and 2F = 0.2415 m, and the PRIMARY bar as 0.2540 m",
    "%.5f" % FP["Fprime_m"] == "0.12701" and "%.4f" % FP["2Fprime_m"] == "0.2540" and abs(2 * FP["Fprime_m"] - FP["2Fprime_m"]) < 1e-12 and "%.5f" % FP["F_M2b_banked_m"] == "0.12074" and "%.4f" % FP["2F_banked_m"] == "0.2415"
    and all(x in n745 for x in ("F′ = 0.12701 m", "2F′ = 0.2540 m", "F = 0.12074 m", "2F = 0.2415 m")) and set(re.findall(r"(?<![\d.])2?F′ = ([0-9.]+) m", s)) == {"0.12701", "0.2540"} and set(re.findall(r"F′ is ([0-9.]+) m", s)) <= {"0.12701"} and "ADE6(L) - ADE6(B′)| > 2F′ = 0.2540 m" in nt(s[s.index("| 1bh | "):s.index("| 1bi | ")]).replace("\\", ""))
MA = rawj("t2-ralign-gap48/ralign/out/M_ALIGN.json")
G48 = nt(rawt("t2-ralign-gap48/gap48/RESULT_GAP48.md"))
chk("M-ALIGN and GAP48: M_ALIGN.json verdict ALIGN-FAILS-MECH, held-out median 0.8401 -> 0.8937 (rise 0.0536 passes, level 0.90 fails, scene-cluster bounds in the file), 47 of 48 scenes improved; the GAP48 result holds 0.8401; "
    "the text states the verdict word, 0.8937 and 0.8401",
    MA["verdict"] == "ALIGN-FAILS-MECH" and "%.4f" % MA["main"]["median_before"] == "0.8401" and "%.4f" % MA["main"]["median_after"] == "0.8937" and "%.4f" % MA["main"]["rise"] == "0.0536" and MA["main"]["pass_rise"] and not MA["main"]["pass_level"]
    and MA["thresholds"]["level_min"] == 0.9 and MA["main"]["scenes_improved"] == 47 and "0.8401" in G48 and all(x in n745 for x in ("ALIGN-FAILS-MECH", "0.8937", "0.8401")))
WD = nt(rawt("wsp-gain-d/RESULT_WSPD.md"))
wdn = ["-0.0175 [-0.0241, -0.0117]", "-0.0172 [-0.0242, -0.0104]", "0.329", "0.289"]
chk("WSP-GAIN-D HARMS: the result file's verdict is HARMS, M1 (D − C0) -0.0175 [-0.0241, -0.0117] and -0.0172 [-0.0242, -0.0104], share_closed 0.329 / 0.289; the same numbers and the word HARMS are in 7.46",
    "Verdict by the card's section 4 table: HARMS" in WD and "READ 2 [M] T1-kin eval-141, E-GAIN-D vs C0 (RIC0) and vs P10: HARMS" in WD and all(x in WD and x in n746 for x in wdn) and "HARMS" in n746)
R2 = rawt("wsp-gain/raw/RESULTS_READ2.md")
r2rows = [l.split("|")[1:-1] for l in R2.split("\n") if re.match(r"\| (0\.5|1\.0) \| s[01] \|", l)]
sc = ["%.3f" % float(r[3]) for r in r2rows]
nsw = [len(r[4].split(",")) for r in r2rows]
chk("WSP-GAIN READ 2: four rows (doses 0.5 and 1.0, seeds 0 and 1), each labelled HARMS; share_closed 0.178 / 0.233 / 0.247 / 0.246 (7.46 states 0.178 to 0.247, the min and max) and 6 to 10 new separated-worse rows (%s); the four M1 cells and the four long share_closed values are in the report as source table lines; 7.46 says all four cells read HARMS" % nsw,
    len(r2rows) == 4 and all(r[5].strip() == "HARMS" for r in r2rows) and sc == ["0.178", "0.233", "0.247", "0.246"] and (min(nsw), max(nsw)) == (6, 10) and ("`share_closed` %s to %s" % (min(sc), max(sc))) in n746
    and "every cell adds 6 to 10 separated-worse rows" in n746 and "All four cells read HARMS" in n746 and all(("| %s |" % r[2].strip()) in s and r[3].strip() in s for r in r2rows))
P5 = rawj("p5-dilution/p5/P5_RESULT.json")
lam = P5["lambda"]
chk("P5 FAILS: P5_RESULT.json VERDICT FAILS, λ 0.9680 [0.9659, 0.9698] (0.96797); the result md says P5 FAILS; 7.46 states λ 0.9680 [0.9659, 0.9698]",
    P5["VERDICT"] == "FAILS" and "%.4f" % lam["point"] == "0.9680" and "[%.4f, %.4f]" % tuple(lam["ci95"]) == "[0.9659, 0.9698]" and "P5 FAILS" in rawt("p5-dilution/p5/P5_RESULT.md") and "λ 0.9680 [0.9659, 0.9698]" in n746)
TL = [json.loads(x) for x in rawt("t2-ralign-gap48/lora_L_launch/L_s0/train_log.jsonl").split("\n") if x.strip()]
st = {r["step"]: r["sec"] for r in TL if "step" in r and "sec" in r}
_tm = re.search(r"median_s_per_step_6_20=([0-9.]+)", rawt("t2-ralign-gap48/lora_L_launch/timing_b64.out"))
fm = lambda x: "{:,.2f}".format(x)
chk("L step time (MEASURED): train_log.jsonl step 50 / 100 / 150 / 200 at 2,257.97 / 4,509.05 / 6,744.30 / 8,994.28 s; 8,994.28 / 200 = 44.97 s per step; the gate timing median is 45.27 (timing_b64.out), so 0.7 % under it; "
    "the supervisor log starts the trainer at 2026-11-07T02:30:42Z; all of these are in 7.45.5 and the 44.97 is in the status line",
    [fm(st[k]) for k in (50, 100, 150, 200)] == ["2,257.97", "4,509.05", "6,744.30", "8,994.28"] and "%.2f" % (st[200] / 200) == "44.97" and _tm and _tm.group(1)[:5] == "45.27" and "%.1f" % (100 * (float(_tm.group(1)) - st[200] / 200) / float(_tm.group(1))) == "0.7"
    and _sup.startswith("[2026-11-07T02:30:42Z]") and all(x in n745 for x in ("2,257.97", "4,509.05", "6,744.30", "8,994.28 / 200 = 44.97 s per step", "45.27", "2026-11-07T02:30:42Z", "0.7 %")) and "44.97 s per step" in line and set(re.findall(r"(\d+\.\d+) s per step", s)) - set(re.findall(r"(\d+\.\d+) s per step", v42)) == {"44.97", "45.27"})
AU = nt(rawt("aucmap/RESULTS_AUCMAP.md"))
chk("AUCMAP outcome (iii): RESULTS_AUCMAP.md states 'Outcome (iii): the corner is not reachable by any gate on this lever family' and reach 0.00 at all 8 AUC levels; 7.46 states outcome (iii)",
    "Outcome (iii): the corner is not reachable by any gate on this lever family" in AU and "reach rate 0.00 at all 8 AUC levels" in AU and "outcome (iii)" in n746)
P1 = nt(rawt("theory-p1pb/P1PB_RESULT.md"))
chk("P1'b FAILS: P1PB_RESULT.md verdict FAILS, the difference of median shrinkage ratios +0.245 [-0.028, +0.407] at s0 (not separated) and +0.312 [+0.068, +0.463] at s1; both are in 7.46, and the 7.46 prediction row reads T1' FAILS with the s0 interval",
    "VERDICT, as the criterion is written: FAILS" in P1 and all(x in P1 and x in n746 for x in ("+0.245 [-0.028, +0.407]", "+0.312 [+0.068, +0.463]")) and "(T1′) | FAILS | Δ +0.245 [-0.028, +0.407]" in n746 and "T1′ falls" in n746)
A4 = nt(rawt("t1-blind/cards/THEORY_CRAWL_AMEND_4.md"))
chk("the stage-2 BLIND M1 intervals +0.011 [-0.070, +0.092] and -0.076 [-0.160, +0.014] are in THEORY_CRAWL_AMEND_4.md (they are quoted from there, not re-read from the stage-2 files) and in 7.46",
    all(x in A4 and x in n746 for x in ("+0.011 [-0.070, +0.092]", "-0.076 [-0.160, +0.014]")))
_dom, _wlr = "t2-dom-lora-freeze/", "wsp-lora-prereg/"
dom_b, wlr_b = git(["show", f"{TIP}:{R}raw/2026-11-07-{_dom}PREREG_T2_DOM_LORA.md"]), git(["show", f"{TIP}:{R}raw/2026-11-07-{_wlr}PREREG_WSP_LORA.md"])
r1bh, r1bl = [l for l in s.split("\n") if l.startswith("| 1bh | ")][0], [l for l in s.split("\n") if l.startswith("| 1bl | ")][0]
chk("the two cards banked after the sections were written: PREREG_T2_DOM_LORA.md (sha256 cad25e83…, same as its .sha256 record, folder raw/2026-11-07-t2-dom-lora-freeze/) is the card row 1bh names (frozen); no added line names DRAFT_PREREG_T2_DOM_LORA; "
    "row 1bl names PREREG_WSP_LORA.md (frozen 0529fcfb…, same as its .sha256 record, folder raw/2026-11-07-wsp-lora-prereg/) as pre-registered and not run; both folders have a row in the section-10 table",
    sha(dom_b) == rawt(_dom + "PREREG_T2_DOM_LORA.sha256").strip() and sha(dom_b)[:8] == "cad25e83" and sha(wlr_b) == rawt(_wlr + "PREREG_WSP_LORA.sha256").strip() and sha(wlr_b)[:8] == "0529fcfb"
    and not any("DRAFT_PREREG_T2_DOM_LORA" in l for l in add) and "`PREREG_T2_DOM_LORA.md` `cad25e83…`" in r1bh and "raw/2026-11-07-" + _dom in r1bh and "frozen" in r1bh
    and "`PREREG_WSP_LORA.md` `0529fcfb…`" in r1bl and "raw/2026-11-07-" + _wlr in r1bl and "frozen" in r1bl and "pre-registered, not run" in r1bl and "`2026-11-07-" + _dom + "`" in app and "`2026-11-07-" + _wlr + "`" in app)
# the section-10 table, re-derived independently of the patch: sha256 prefix of the key file, first commit that added it, and the claim row in the register
rowre = re.compile(r"^\| `(2026-11-07-[^`]+/)` \| `([^`]+)` `([0-9a-f]{8})…` \| `([0-9a-f]{7,8})` \| `([^`]+)` \|$")
bad, ksh = [], set()
for l in app_blk[2:]:
    m = rowre.match(l)
    if not m:
        bad.append(("unparsed", l[:50])); continue
    fold, key, sh8, land, claim = m.groups()
    ksh.add(sh8)
    pth = R + "raw/" + fold + key
    first = git(["log", "--format=%H", "--diff-filter=A", TIP, "--", pth], False).strip().split("\n")[-1]
    if sha(git(["show", f"{TIP}:{pth}"]))[:8] != sh8 or not first.startswith(land) or claim not in reg:
        bad.append((fold, sh8, land, claim))
chk("the 21 rows of the section-10 table, re-derived: each key file's sha256 prefix equals the one printed, each landing commit is the first commit that added the file, each claim row is in the register", not bad and len(app_blk[2:]) == 21, "(%s)" % bad[:3])
_ptk = sorted(set(re.findall(r"`([0-9a-f]{8})…?`", "\n".join(add))))
_unk = [t for t in _ptk if t not in ksh and t not in rawcorp and t not in reg and t not in v42 and subprocess.run(["git", "-C", M, "-c", "safe.directory=*", "cat-file", "-e", t + "^{commit}"], capture_output=True).returncode != 0]
chk("every 8-hex prefix in backticks in the added lines (%d distinct) is the sha256 prefix of a key file of the section-10 table (verified above), a frozen-sha prefix found in the raw corpus, the register or the v42 text, or a commit of the repository" % len(_ptk), not _unk and len(_ptk) > 20, "(unknown: %s)" % _unk[:5])
rl = show("Project Steering/RETRACTION_LOG.md")
_rt = '## 2026-11-07 ~03:40 -- "a0 is the 0.2 s backward difference of measured speed"'
_cm = subprocess.run(["git", "-C", M, "-c", "safe.directory=*", "show", 'a6b854de:Project Steering/RETRACTION_LOG.md'], capture_output=True)
chk("the a0 entry: RETRACTION_LOG.md on the tip holds the entry title 2026-11-07 ~03:40 -- \"a0 is the 0.2 s backward difference of measured speed\"; commit a6b854de (named in 7.46.5) is a commit that holds it; 7.46 names RETRACTION_LOG.md and a6b854de; a0 is written as a 0.1 s difference",
    _rt in rl and _cm.returncode == 0 and _rt in _cm.stdout.decode("utf-8", "replace") and "`a6b854de`" in sec746 and "RETRACTION_LOG.md" in sec746 and "0.1 s" in sec746 and "the 0.1 s backward-difference acceleration" in sec744)
chk("no LoRA driving result is claimed: 7.45, the 8.3 Update, row 1bh, the abstract and section 9 item 21 each state that none exists; the added lines carry no 'LoRA ... DOMINANT' or 'scale signal reached' wording",
    "lora driving result" in n745.lower() and "No LoRA driving result exists" in s[s.index("**Update (v43).**"):s.index("**Update (v42).**")] and "**no LoRA driving result exists**" in r1bh and "no LoRA driving result exists" in s[:s.index("## 2. Introduction")]
    and "No reactive-traffic number, no LoRA driving result and no DOMINANT exist." in s[s.index("21. **v43 reads"):s.index("## 10. Reproducibility appendix")] and not re.search(r"scale signal (is|was) (reached|met)|LoRA[^.\n]{0,40} reads DOMINANT", " ".join(add)))
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v43.py", TIP, td], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1"))
    same = r.returncode == 0 and all(open(os.path.join(td, f), "rb").read() == open(OUT + f, "rb").read() for f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html"))
    chk("patch_v43.py re-run reproduces both deliverables byte for byte", same, r.stdout.strip()[-100:].replace("\n", " ") + r.stderr.strip()[-200:])
nb = len(re.findall(r"\{\{", s)) + len(re.findall(r"\{\{", h))
chk("PLACEHOLDERS: no `{{` remains in out/ (md and html)", nb == 0, "(%d occurrences)" % nb)
say()
say("RESULT: %s (%d checks, %d failed%s)" % ("PASS" if not FAILS else "FAIL", sum(1 for x in L if x[:5] in ("PASS ", "FAIL ")), len(FAILS), "; failed: " + " | ".join(f[:40] for f in FAILS) if FAILS else ""))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print(L[-1])
sys.exit(1 if FAILS else 0)
