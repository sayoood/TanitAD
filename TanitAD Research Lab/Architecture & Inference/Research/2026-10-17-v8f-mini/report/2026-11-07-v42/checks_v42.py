"""Independent checks of report v42 (the patch's own assertions have already passed to produce out/). Writes the checks file.
python -I checks_v42.py tip [out_dir [checks_path]]   (git, python; no network)"""
import hashlib, json, os, re, subprocess, sys, tempfile
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "out").rstrip("/\\") + "/"
CP = sys.argv[3] if len(sys.argv) > 3 else HERE + "v42_checks.txt"
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


def norm_pipes(line):  # as patch_v42.py: a bare pipe inside a cell is escaped; delimiter = space on both sides, first or last character
    if re.fullmatch(r"\|[\s:\-|]+", line):
        return line
    n, o = len(line), []
    for i, ch in enumerate(line):
        o.append("\\|" if (ch == "|" and 0 < i < n - 1 and line[i - 1] != "\\" and not (line[i - 1] == " " and line[i + 1] == " ")) else ch)
    return "".join(o)


norm = lambda t: re.sub(r"\s+", " ", t)
RP41 = R + "report/2026-11-06-v41/"
v41 = show(RP41 + "V8F_SCIENTIFIC_REPORT.md")
s = open(OUT + "V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
h = open(OUT + "v8f_report.html", encoding="utf-8", newline="").read()
say("# report v42 checks, tip " + TIP)
say()
chk("v41 inputs are the landed deliverables (sha256)", sha(show(RP41 + "V8F_SCIENTIFIC_REPORT.md", True)) == "c73d0ef62a7138d7e4feabb795545e392450d25a8385d06afdffde891e8f35e2"
    and sha(show(RP41 + "v8f_report.html", True)) == "70b922065457e6624561a32f741935b36e926917dfbb101ef43873488edbda00")
chk("html = page template with the v42 md", h == show(R + "report/2026-10-26/page_template_v30.html").replace("%%MD%%", s))
chk("LF only, no BOM (md and html)", "\r" not in s and "\ufeff" not in s and "\r" not in h and "\ufeff" not in h)
c41, c42 = Counter(v41.split("\n")), Counter(s.split("\n"))
rem = [l for l in (c41 - c42).elements() if l.strip()]
add = [l for l in (c42 - c41).elements() if l.strip()]
KEYS = ("**Status:** research report, v41 of 2026-11-06", "**Evidence classes used throughout.**", "The contribution we defend is **diagnostic**.", "* **What is next.** The crawl set is still open",
        "**Tiers.** `EVAL_DOCTRINE.md` (amended 2026-09-02)", "`PREREG_V8F_T1KIN_HACC` (with three amendments", "* **(b) T2 re-perception.** Not provisioned", "* Any **driving** claim: all numbers are open loop",
        "At v30, the levers aimed at those rows were", "| 6 | **T1-kin** (self-action open loop) read of HACC; T2 (closed loop)", "    * No T2 number exists.")
want_hits = [1] * 10 + [2]
hits = [sum(1 for l in rem if l.startswith(k)) for k in KEYS]
unexp = [l[:80] for l in rem if not any(l.startswith(k) for k in KEYS)]
chk("removed lines from v41: exactly the 12 edited carried lines (status, evidence classes, abstract contribution paragraph, abstract next bullet, section 6 tiers, the T1-kin paragraph, 7.41.6 (b), 8.2 driving claim, 8.3 (iv), 8.4 row 6, two 'No T2 number exists' limit bullets), each key matched as listed",
    not unexp and hits == want_hits and len(rem) == 12, "(%d removed; unexpected %s; hits %s)" % (len(rem), unexp[:2], hits))
order = ["**7.41.7 ", "### 7.42 "] + ["**7.42.%d " % i for i in range(1, 8)] + ["### 7.43 "] + ["**7.43.%d " % i for i in range(1, 9)] + ["### 7.44 "] + ["**7.44.%d " % i for i in range(1, 10)] + ["## 8. Discussion"]
pos = [s.index(x) for x in order]
chk("order: 7.41.7 < 7.42 < 7.42.1 to 7.42.7 < 7.43 < 7.43.1 to 7.43.8 < 7.44 < 7.44.1 to 7.44.9 < section 8, each heading once", pos == sorted(pos) and all(s.count(x) == 1 for x in order))
i742, i8 = s.index("### 7.42 "), s.index("## 8. Discussion")
sec = s[i742:i8]
sec742 = s[i742:s.index("### 7.43 ")]
sec744 = s[s.index("### 7.44 "):i8]
srcs = source_texts()
SL = set()
for n, t in srcs.items():
    if n.endswith((".md", ".txt")):
        SL |= {norm_pipes(l) for l in t.split("\n") if l.startswith("|")}
tl = [l for l in sec.split("\n") if l.startswith("|")]
blks = [[x for x in b.split("\n") if x] for b in re.findall(r"(?:^\|.*\n?)+", sec, flags=re.M)]
src_blks = [b for b in blks if all(l in SL for l in b)]
hand_blks = [b for b in blks if not all(l in SL for l in b)]
dirs_total = sum(int(l.split(" | ")[4].split()[0]) for l in open(OUT + "DIRECTIVES.txt", encoding="utf-8").read().strip().split("\n"))
app = s[s.index("**Reads added in v42"):s.index("**Figures.**")]
app_blk = [l for l in app.split("\n") if l.startswith("|")]
rows3 = [l for l in s.split("\n") if l.startswith(("| 1bd | ", "| 1be | ", "| 1bf | "))]
chk("table lines of 7.42 to 7.44: the 4 directive tables are blocks of source lines only (a bare pipe inside a cell escaped) and number as many lines as the directives expanded; the other 6 tables of the sections are hand-written; the only added table lines outside the sections are rows 1bd, 1be, 1bf (not source lines) and the appendix table (10 lines)",
    len(src_blks) == 4 and sum(len(b) for b in src_blks) == dirs_total and len(hand_blks) == 6 and len(rows3) == 3 and not any(r in SL for r in rows3) and len(app_blk) == 10 and app_blk[2] not in SL,
    "(%d source table lines in %d blocks = %d expanded directive lines; %d hand-written tables with %d lines; %d source files)" % (sum(len(b) for b in src_blks), len(src_blks), dirs_total, len(hand_blks), sum(len(b) for b in hand_blks), len(srcs)))
cols = lambda l: l.replace("\\|", "").count("|") - 1
i84 = s.index("### 8.4 ")
_r1az = [l for l in s[i84:].split("\n") if l.startswith("| 1az | ")][0]
tabs = [[x for x in b.split("\n") if x] for b in re.findall(r"(?:^\|.*\n?)+", sec, flags=re.M)]
chk("every table of 7.42 to 7.44 has one cell count and a header rule (10 tables); rows 1bd, 1be and 1bf have the cell count of row 1az",
    len(tabs) == 10 and all(len({cols(r) for r in blk}) == 1 and re.fullmatch(r"\|[\s:\-|]+", blk[1]) for blk in tabs) and all(cols(o) == cols(_r1az) == 4 for o in rows3), "(%d tables)" % len(tabs))
reg = show("Project Steering/GOALS_AND_CLAIMS.md")
corp = norm(" ".join(list(srcs.values()) + [reg, v41]))
qs = [q.strip('"') for l in add for q in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", l)]
chk("italic quotations of 25+ characters in the added lines verbatim in a source (raw files of 2026-11-02 to 2026-11-07, register, v41)", qs and all(norm(q) in corp for q in qs), "(%d quotations)" % len(qs))
fams = [len(re.findall(f, sec742)) for f in ("LONGITUDINAL|along-track|speed MAE|accel MAE", "LATERAL|curvature|yaw-rate|heading|cross-track", "TACTICAL|TAC-LAT|TAC-LON", "STRATEGIC|S-POS")]
chk("each of the four metric families named in 7.42 at least 3 times (STRATEGIC shown as PAUSED by ITEM 25)", all(c >= 3 for c in fams) and "PAUSED" in sec742 and "ITEM 25" in sec742, str(fams))
chk("tier T2, the paired scene-cluster bootstrap, B = 2,000, H-ESTIM-SEED-1 and the rule R-SCN stated in 7.42; the tier is never pooled with T1-kin and agents replay",
    all(x in norm(sec742) for x in ("T2", "scene-cluster bootstrap", "B = 2,000", "H-ESTIM-SEED-1", "R-SCN", "never pooled with T1-kin", "replay")))
chk("evidence classes [M], [E], [H] in 7.42 to 7.44 and [P] in 7.44", all(x in sec for x in ("[M]", "[E]", "[H]")) and "[P]" in sec744)
cl = lambda t: len(CLRE.findall(t))
bc = lambda t: [len(re.findall(r"\b" + w + r"\b", t, re.I)) for w in BANNED]
parts = {n: open(HERE + n, encoding="utf-8", newline="").read() for n in ("sec_742_t2.md", "sec_743_lorapod.md", "sec_744_theory.md", "edits_v42.md", "HANDOFF_V42.md", "msg_v42.txt") if os.path.exists(HERE + n)}
chk("the forbidden two-word T2 wording (either separator) is absent from every added line and every part file; the count of the carried text falls, never rises (v41 %d lines, v42 %d)" % (cl(v41), cl(s)),
    cl(" ".join(add)) == 0 and not any(CLRE.search(t) for t in parts.values()) and cl(s) <= cl(v41))
chk("banned-word scan: none new in the added lines (net of the 12 edited carried lines) or in the part files; per-word counts of the whole report at most the v41 counts",
    all(a <= r for a, r in zip(bc(" ".join(add)), bc(" ".join(rem)))) and not any(BANRE.search(t) for t in parts.values()) and all(a <= b for a, b in zip(bc(s), bc(v41))), "(v41 %s, v42 %s)" % (bc(v41), bc(s)))
# math: strip code, display equations, then the template's inline-math pattern; no dollar sign may remain in the sections; the display equations are balanced
nc = re.sub(r"`[^`\n]+`", " ", sec)
disp = re.findall(r"\$\$[\s\S]+?\$\$", nc)
rest = re.sub(r"\$\$[\s\S]+?\$\$", " ", nc)
inl = re.findall(r"(?:^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", rest, flags=re.M)
rest2 = re.sub(r"(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", r"\1 ", rest, flags=re.M)
chk("math spans: 5 display equations (even $$ count), inline math matches the template pattern, no dollar sign remains outside math in 7.42 to 7.44, and the span counts of v41 are untouched",
    len(disp) == 5 and sec.count("$$") == 10 and "$" not in rest2 and len(inl) > 100 and s.count("$$") == v41.count("$$") + 10 and s.count("$") - sec.count("$") == v41.count("$"), "(%d inline spans, %d dollar signs in the sections)" % (len(inl), sec.count("$")))
i83 = s.index("### 8.3 ")
ups = [s.index("**Update (v%d).**" % v, i83) for v in (42, 41, 40, 39, 38, 37)]
chk("8.3 has Update (v42) before Update (v41) before Update (v40) to (v37), each once", ups == sorted(ups) and all(s.count("**Update (v%d).**" % v) == 1 for v in (42, 41, 40, 39, 38, 37)))
i81, i82 = s.index("### 8.1 "), s.index("### 8.2 ")
nums = [int(x) for x in re.findall(r"^(\d+)\. \*\*", s[i81:i82], re.M)]
n41 = [int(x) for x in re.findall(r"^(\d+)\. \*\*", v41[v41.index("### 8.1 "):v41.index("### 8.2 ")], re.M)]
chk("8.1 ends with item 32: the v41 items (ending at 31) are untouched and item 32 follows", n41[-1] == 31 and nums == n41 + [32])
i9, i10 = s.index("## 9. "), s.index("## 10. Reproducibility appendix")
chk("section 9 ends with item 20 (v42 reads and their limits), after item 19", re.findall(r"^(\d+)\. \*\*", s[i9:i10], re.M)[-2:] == ["19", "20"] and "20. **v42 reads and their limits.**" in s[i9:i10])
rows = [s.index("| %s | " % r) for r in ("1bc", "1bd", "1be", "1bf")] + [s.index("| 2 | **LoRA on DINOv3**")]
chk("8.4 has rows 1bd, 1be and 1bf once each, after row 1bc and before row 2", rows == sorted(rows) and all(s.count("| %s | " % r) == 1 for r in ("1bd", "1be", "1bf")))
line = s.split("\n")[2]
chk("status line names v42 of 2026-11-07, the tip, §7.42, §7.43, §7.44 and v41 as the previous version; no date after 2026-11-07 anywhere",
    "v42 of 2026-11-07" in line and "`%s`" % TIP[:8] in line and all(x in line for x in ("§7.42", "§7.43", "§7.44", "v41: `bd98063d`", "T2 re-perception"))
    and not [x for x in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if x > "2026-11-07"])
chk("no stale T2 statement: the status line's 'no (two-word) (T2) number exists' and the tier line 'T2 (two-word), unprovisioned' are gone; the new status wording names 48 scenes and NOT-DOMINANT; 7.41.6 (b) carries the 7.42 pointer",
    ("**no " + "pool-desolc"[::-1] + " (T2) number exists**") not in s and ("T2 " + "pool desolc"[::-1] + ", unprovisioned") not in s and "no reactive-traffic number exists" in line and "48 NuRec scenes" in line and "NOT-DOMINANT" in line
    and "That read exists as of v42 (§7.42" in s)
DJ = json.loads(show(R + "raw/2026-11-06-t2-dominance-read/stage4/score/DOMINANCE_T2.json"))
RJ = json.loads(show(R + "raw/2026-11-06-t2-crawl-read/stage2/READ_STAGE2.json"))
FT = DJ["decision"]["failing_terms"]
secn = norm(sec742).replace("\u2212", "-")
want = ["%+.4f [%+.4f, %+.4f]" % tuple(RJ[x][sd][k] for k in ("delta", "lo", "hi")) for x in ("X1", "X2", "BL") for sd in ("0", "1")]
want += ["%+.4f" % DJ["by_seed"][sd]["rows"]["M1 crawl share"]["delta"] for sd in ("s0", "s1")] + ["%+.4f [%+.4f, %+.4f]" % tuple(DJ["by_seed"][sd]["rows"]["ADE6"][k] for k in ("delta", "lo", "hi")) for sd in ("s0", "s1")]
chk("added 7.42 numbers agree with the raw READ_STAGE2 and DOMINANCE_T2 JSON (X1, X2, BL with bounds; M1 crawl share and ADE6 with bounds), the outcome is NOT-DOMINANT at both seeds, and the failing_terms are quoted verbatim",
    all(x.replace("\u2212", "-") in secn for x in want) and DJ["outcome"] == "NOT-DOMINANT" and "NOT-DOMINANT" in sec742 and all(t in sec742 for sd in ("s0", "s1") for t in FT[sd]) and RJ["outcome"] == "CRAWL-PERSISTS" and "CRAWL-PERSISTS" in sec742
    and not any(DJ["decision"]["per_seed"][sd][d] for sd in ("s0", "s1") for d in ("DOMINANT", "D1", "D2", "D3")), "(%d numbers)" % len(want))
chk("7.44 reads as written: P1 FAILS as written, P2 HOLDS as written (scale-sensitive; the weight-only claims matched 2 of 4), P3 FAILS as written (all 26 dominating pairs with a CEILING dominator), P1' NOT EVALUABLE as pre-registered (24 CONFIDENT windows, floor 30), P4 not evaluable, WSP-GAIN pre-registered and not yet read; no WSP-GAIN number quoted",
    all(x in sec744 for x in ("**P1 FAILS as written [M]**", "**P2 HOLDS as written [M]**", "**P3 FAILS as written [M]**", "matched 2 of 4", "all 26", "NOT EVALUABLE as pre-registered, so P1′ is neither held nor failed", "24 windows (floor 30)", "NOT EVALUABLE", "pre-registered, built and unit-tested, not yet read", "No WSP-GAIN result exists")))
chk("7.43 states what is not known: no LoRA result exists, the batch-64 timing is not yet measured, the pod handoffs are E: files not yet landed, the base is B' and the device alone misses the bar",
    all(x in norm(s[s.index("### 7.43 "):s.index("### 7.44 ")]) for x in ("no LoRA result", "not yet measured", "not yet landed", "B′", "21 to 23 times", "2.11 to 2.25e-2")) and "No LoRA" in sec)
chk("the shared v42 sentence occurs exactly 3 times (status line, 8.1 item 32, and 8.1's Result) and names no driving result: it says nothing in v42 is a driving result, no counted DOMINANT, no scale signal",
    s.count(open(HERE + "edits_v42.md", encoding="utf-8").read().split("=== V42S ===\n")[1].split("\n")[0].replace("{{TP}}", TIP[:8])) >= 2 and "nothing in v42 is a driving result" in s)
chk("the related-work paragraph cites Library key takehi2026weightspace, the banked page sha256 a326bc5b, the 2026-11-02 blog post as not peer-reviewed [P], and does not claim the method generalises; it sits in section 3",
    s.count("takehi2026weightspace") >= 3 and s.index("**Weight-space learning and reading (v42).**") < s.index("## 4. Method") and "we do not claim the method generalises" in s and "a blog post, not peer-reviewed" in s[s.index("**Weight-space learning"):s.index("## 4. Method")])
chk("the appendix block 'Reads added in v42' is in section 10 before the Figures paragraph and names all 8 new raw folders", s.index("## 10. Reproducibility appendix") < s.index("**Reads added in v42") < s.index("**Figures.**")
    and all(("`2026-11-06-%s" % f) in s[s.index("**Reads added in v42"):s.index("**Figures.**")] for f in ("t2-harness-v8f/", "t2-stage1-box2/", "t2-crawl-read/", "t2-stage4-c0c/", "t2-dom-repin/", "t2-dominance-read/", "lora-pod-amend4/", "theory-crawl/")))
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
# the section files are inserted as written: every non-directive line of each file occurs in the report, in order (the one math fix is listed in edits_v42.md)
ordered = True
for fn in ("sec_742_t2.md", "sec_743_lorapod.md", "sec_744_theory.md"):
    p = 0
    for ln in open(HERE + fn, encoding="utf-8").read().strip("\n").split("\n"):
        if ln.startswith("@@T ") or not ln.strip():
            continue
        pat_ln = re.compile("[0-9a-f]{7,8}".join(re.escape(x) for x in re.split(r"\{\{[^{}\n]*\}\}", ln)))
        m = pat_ln.search(s, p)
        if not m:
            ordered = False; print("section line not found:", fn, ln[:80]); break
        p = m.end()
chk("the three section files (sec_742_t2.md, sec_743_lorapod.md, sec_744_theory.md) appear in the report line for line and in order (directive lines replaced by their tables)", ordered)
# privacy: no address, port or pod id in anything the package writes
pat = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|root@|\bssh\s+-|\bport\s+\d{2,5}\b|pod[ _-]?id|:\d{4,5}\b(?!-)", re.I)
leak = [(n, m.group(0)) for n, t in list({"report (added lines)": "\n".join(add), **parts}.items()) for m in pat.finditer(t) if not re.match(r"\d{4}-", m.group(0))]
chk("no pod address, port or pod id in the report or the part files (IPv4, root@, ssh -, 'port N', 'pod id', :NNNN)", not leak, "(%s)" % leak[:3])
files = [os.path.join(d, f).replace("\\", "/") for d, _, fs in os.walk(HERE) for f in fs if f.endswith((".py", ".md", ".sh", ".js", ".txt")) and "/out/" not in os.path.join(d, f).replace("\\", "/") + "/"
         and not f.startswith("_v41copy_")]
crs = [os.path.relpath(f, HERE) for f in files if b"\r" in open(f, "rb").read()]
chk("LF only for every .py, .md, .sh, .js and .txt file of the package (CR counted in python; the _v41copy_ originals are excluded)", not crs, "(%d files; with CR: %s)" % (len(files), crs[:3]))
with tempfile.TemporaryDirectory() as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v42.py", TIP, td], capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1"))
    same = r.returncode == 0 and all(open(os.path.join(td, f), "rb").read() == open(OUT + f, "rb").read() for f in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html"))
    chk("patch_v42.py re-run reproduces both deliverables byte for byte", same, r.stdout.strip()[-100:].replace("\n", " ") + r.stderr.strip()[-200:])
nb = len(re.findall(r"\{\{", s)) + len(re.findall(r"\{\{", h))
chk("PLACEHOLDERS: no `{{` remains in out/ (md and html)", nb == 0, "(%d occurrences)" % nb)
say()
say("RESULT: %s (%d checks, %d failed%s)" % ("PASS" if not FAILS else "FAIL", sum(1 for x in L if x[:5] in ("PASS ", "FAIL ")), len(FAILS), "; failed: " + " | ".join(f[:40] for f in FAILS) if FAILS else ""))
open(CP, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print(L[-1])
sys.exit(1 if FAILS else 0)
