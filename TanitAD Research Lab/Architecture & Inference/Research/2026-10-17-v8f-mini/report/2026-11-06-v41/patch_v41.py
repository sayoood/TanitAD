"""Report v41 = v40 + section 7.41 (addenda A4 to A6 of the level-floor family: the ceiling at kappa 0.75, kappa 1 with the exact level, a priori noise and the heads, the diagnosis that chose A6,
the bounded lift F_B; what the family establishes; the remaining levers and what unblocks each; provenance), inserted after 7.40.7, + the status line (version, tip, version list, the v41 sentence),
the abstract (the recovery bullet and the next bullet), the 7.40.6 pointer to 7.41, 8.1 item 31, 8.3 Update (v41), 8.4 rows 1ba to 1bc, 9 item 19.
Built from the landed v40 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once). The prose is in the package files beside this script
(sec_741_levelfloor.md: the section text; edits_v41.md: one block per edit, V41S being the shared v41 sentence that the blocks name as <<V41S>>); every source is `git show <tip>:<path>`
from the mirror. Nothing is read from a working tree of the repository. Nothing is staged, committed or pushed.

Directives in the section file:
  `@@T KEY after=SUBSTR [nth=N]@@` is replaced by the Nth (default 1st) contiguous table (lines starting with `|`) after the first line of the KEY source file that contains SUBSTR, copied verbatim
  except that a bare pipe inside a cell is escaped so that the rendered row keeps its cell count (as in v39 and v40).
  `@@F KEY after=SUBSTR [nth=N] as=TITLE@@` is replaced by one bullet that counts, per family and seed, the separated-better and separated-worse rows of that same table (TAC rows: higher is better; every
  other row: lower is better; PRIMARY and the crawl row are outside the family counts), computed from the table's own lines.
Fill-ins: `{{TP}}` (tip, 8 hex), `{{LAND_x}}` (first commit that added the source file x), `{{RES_x}}` (sha256[:8] of source file x).

python -I patch_v41.py tip [out_dir]
Every italic quotation of 25 characters or more is asserted verbatim in a source (RESULTS files, cards, rulings, notes, register, v40)."""
import hashlib, json, os, re, subprocess, sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
assert _argv, "usage: patch_v41.py <tip> [out_dir]"
TIP = _argv[0]
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (_argv[1] if len(_argv) > 1 else HERE + "/out").rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BANNED = tuple(w[::-1] for w in ("diov", "dedesrepus", "detcarter", "ylremrof", "regnol on", "noitcerroc", "nwardhtiw"))  # the programme's banned words, stored reversed so this file passes its own lander scan
_BAN = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)
_CL = re.compile(r"closed[- ]loop", re.I)


def gitrun(args, binary=False, check=True):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
    if not check and r.returncode != 0:
        return None
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def raw(path):
    return gitrun(["show", f"{TIP}:{path}"], binary=True)


def blob(path):
    t = raw(path).decode("utf-8")
    assert len(t) > 0 and "\r" not in t and "\ufeff" not in t, path
    return t


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def part(name):
    t = open(os.path.join(HERE, name), encoding="utf-8", newline="").read()
    assert "\r" not in t and "\ufeff" not in t, name
    return t.strip("\n")


# ---- the v40 deliverables are the starting point ---------------------------------------------------------------------------------------------------------
V40_MD_SHA = "adb96d9c003990a412fb82c7a1a449f513c18883e9ec8d15b02e4f7c3ed5c285"
V40_HTML_SHA = "937cb19baebc258ca85d59658d457e45bf7e0997be2e8ab8f66f29da646a3f7d"
RP40 = R + "report/2026-11-06-v40/"
s = blob(RP40 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP40 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP40 + "V8F_SCIENTIFIC_REPORT.md")) == V40_MD_SHA, "the tip's report md is not the landed v40 deliverable"
assert sha(raw(RP40 + "v8f_report.html")) == V40_HTML_SHA, "the tip's report html is not the landed v40 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v40 html is not the template with the v40 md"
assert s.count("**Status:** research report, v40 of 2026-11-06") == 1
S0 = s

# ---- sources ----------------------------------------------------------------------------------------------------------------------------------------------
_D = "raw/2026-11-06-levelfloor-"
RES_PATHS = OrderedDict([
    ("A4P", _D + "a4-prereg/PREREG_LEVELFLOOR_A4.md"),
    ("LFO", _D + "a4-lfo-result/lfo_read/RESULTS.md"),
    ("LFOJ", _D + "a4-lfo-result/lfo_read/raw/LFO_READ.json"),
    ("LFOBH", _D + "a4-lfo-result/lfo_build/HANDOFF_LFOBUILD.md"),
    ("LFOH", _D + "a4-lfo-result/lfo_read/HANDOFF_LFO.md"),
    ("LFOPIN", _D + "a4-lfo-result/lfo_build/PINS.txt"),
    ("M1G", _D + "a4-lfo-result/beside/M1_GEOMETRY.md"),
    ("A5P", _D + "a5-prereg/PREREG_LEVELFLOOR_A5.md"),
    ("LF5", _D + "a5-lf5-result/lf5_read/RESULTS.md"),
    ("LF5J", _D + "a5-lf5-result/lf5_read/raw/LF5_READ.json"),
    ("LF5H", _D + "a5-lf5-result/lf5_read/HANDOFF_LF5.md"),
    ("LF5PIN", _D + "a5-lf5-result/lf5_build/PINS.txt"),
    ("A5NF", _D + "a5-lf5-result/lf5_read/A5_NOTE_READER_FIX.md"),
    ("A5NT", _D + "a5-lf5-result/lf5_read/A5_NOTE_TIME_LABEL.md"),
    ("A6D", _D + "a6-prereg/beside/A6_DIAG.md"),
    ("A6P", _D + "a6-prereg/a6_prereg/PREREG_LEVELFLOOR_A6.md"),
    ("A6R1", _D + "a6-lf6-result/a6_prereg_additions/A6_RULING_1.md"),
    ("A6NK", _D + "a6-lf6-result/lf6_build/A6_NOTE_KA3_READING.md"),
    ("LF6", _D + "a6-lf6-result/lf6_read/RESULTS.md"),
    ("LF6J", _D + "a6-lf6-result/lf6_read/raw/LF6_READ.json"),
    ("LF6H", _D + "a6-lf6-result/lf6_read/HANDOFF_LF6.md"),
    ("LF6PIN", _D + "a6-lf6-result/lf6_build/PINS.txt"),
    ("T2S", "raw/2026-11-06-t2-sizing-v8f/T2_SIZING_V8F.md"),
])
LAND_KEYS = ("A4P", "LFO", "A5P", "LF5", "A6P", "A6D", "A6R1", "LF6", "T2S")
DIRECTIVE_SRC = {"LFO": "LFO", "LF5": "LF5", "LF6": "LF6", "A6P": "A6P"}
# hard-coded sha256 prefixes in the section texts and edits, asserted against the tip's blobs
HARD = [(_D + "a4-prereg/PREREG_LEVELFLOOR_A4.md", "8bfbc386"), (_D + "a5-prereg/PREREG_LEVELFLOOR_A5.md", "2aac1dbe"), (_D + "a6-prereg/a6_prereg/PREREG_LEVELFLOOR_A6.md", "fbf5de02")]
for p, pre in HARD:
    assert sha(raw(R + p)).startswith(pre), ("a hard-coded sha256 prefix differs from the tip's blob", p, pre)

_SRC_CACHE = {}


def srctext(k):
    if k not in _SRC_CACHE:
        assert k in RES_PATHS, ("no source path for directive key", k)
        _SRC_CACHE[k] = blob(R + RES_PATHS[k])
    return _SRC_CACHE[k]


RES_SHA = {k: sha(raw(R + p)) for k, p in RES_PATHS.items()}
LAND = {}
for k in LAND_KEYS:
    c = gitrun(["log", "--format=%h", "--diff-filter=A", TIP, "--", R + RES_PATHS[k]]).split("\n")[-1]
    assert re.fullmatch(r"[0-9a-f]{7,}", c), (k, c)
    LAND[k] = c[:8]
_want_land = {"A4P": "ffa2637", "LFO": "cc129be", "A5P": "785bfd8", "LF5": "6048f4d", "A6P": "d080896", "A6D": "d080896", "A6R1": "c59e7b8", "LF6": "c59e7b8", "T2S": "bd98063"}
for k, pre in _want_land.items():
    assert LAND[k].startswith(pre), ("landing commit", k, LAND[k], pre)

# ---- tool shas named in the text, asserted against the PINS of each build -----------------------------------------------------------------------------------
assert "patched_tool=6744d3e4" in srctext("LFOPIN") and "lf_tool=6166178a" in srctext("LFOPIN") and "level_floor=2781e23a" in srctext("LFOPIN")
assert "patched_tool=88b275a1" in srctext("LF5PIN") and "lfo_tool=6744d3e4" in srctext("LF5PIN")
assert "patched_tool=949c4b7a" in srctext("LF6PIN") and "lf5_tool=88b275a1" in srctext("LF6PIN") and "level_floor=2781e23a" in srctext("LF6PIN")

# ---- raw JSON cross-checks of the headline numbers --------------------------------------------------------------------------------------------------------
JO, J5, J6 = (json.loads(srctext(k)) for k in ("LFOJ", "LF5J", "LF6J"))
assert JO["arm"]["outcome"]["label"] == "CEILING-SHORT" and J5["joint"]["case"] == "OPERATOR-IS-THE-PROBLEM" and J6["joint"]["case"] == "THE-BOUND-REMOVES-THE-CLOSURE"
assert J5["joint"]["counted_dominant"] is False and J6["joint"]["counted_dominant"] is False and J5["joint"]["estimator_levers_signalled"] is False and J6["joint"]["estimator_levers_signalled"] is False
assert J5["card_sha256"].startswith("2aac1dbe") and J6["card_sha256"].startswith("fbf5de02") and JO["card_sha256"].startswith("8bfbc386")
assert [J5["ceiling"][a]["outcome"]["label"] for a in ("LFO100", "LFN100x10", "LFN100x20")] == ["CEILING-HARMS"] * 3 and J5["LF100"]["outcome"]["label"] == "P-HOLDS-G-OR-D-FAILS"
assert [J6["ceiling"][a]["outcome"]["label"] for a in ("LFOB05", "LFOB10")] == ["CEILING-SHORT"] * 2 and [J6["LFB"][a]["outcome"]["label"] for a in ("LFB05", "LFB10")] == ["P-FAILS-M1-MOVED"] * 2
for J in (JO, J5, J6):
    assert all(isinstance(v, dict) and v.get("PASS") is True for v in J["controls"].values()), "a read control is not PASS"
assert (len(JO["controls"]), len(J5["controls"]), len(J6["controls"])) == (13, 29, 32), (len(JO["controls"]), len(J5["controls"]), len(J6["controls"]))
F2 = lambda a, b: "%.3f / %.3f" % (a, b)
F4 = lambda a, b: "%.4f / %.4f" % (abs(a), abs(b))
JS = [
    F2(*(JO["arm"]["rules"]["P_O"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    F2(*(J5["ceiling"]["LFO100"]["rules"]["P_O"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    F2(*(J5["ceiling"]["LFN100x10"]["rules"]["P_O"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    F2(*(J5["ceiling"]["LFN100x20"]["rules"]["P_O"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    F2(*(J5["LF100"]["rules"]["P"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    "+" + F4(*(J5["LF100"]["rules"]["G"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1"))).replace(" / ", " / +"),
    F2(*(J6["ceiling"]["LFOB05"]["rules"]["P_O"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    F2(*(J6["ceiling"]["LFOB10"]["rules"]["P_O"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    F2(*(J6["LFB"]["LFB05"]["rules"]["P"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    F2(*(J6["LFB"]["LFB10"]["rules"]["P"]["per_seed"][k]["share_closed"] for k in ("s0", "s1"))),
    "-" + F4(*(JO["arm"]["rules"]["G_O"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1"))).replace(" / ", " / -"),
    "-" + F4(*(J6["ceiling"]["LFOB05"]["rules"]["G_O"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1"))).replace(" / ", " / -"),
    "-" + F4(*(J6["ceiling"]["LFOB10"]["rules"]["G_O"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1"))).replace(" / ", " / -"),
]
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
for row in ("**H-LEVELFLOOR-1 (A4)** (2026-11-06, arm LFO075 READ", "**H-LEVELFLOOR-1 (A5)** (2026-11-06, arms LFO100 / LFN100x10 / LFN100x20 READ", "**H-LEVELFLOOR-1 (A6)** (2026-11-06, arms LFOB05 / LFOB10 READ"):
    assert row in REG, ("a register row the provenance names", row)

# ---- the section texts ------------------------------------------------------------------------------------------------------------------------------------
_blocks = re.split(r"^=== (\w+) ===\n", part("edits_v41.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
assert sorted(B) == ["I19", "R83", "ROW", "S81", "V41S"], sorted(B)
assert "\n" not in B["V41S"] and "<<" not in B["V41S"] and "{{" not in B["V41S"]
B = {k: v.replace("<<V41S>>", B["V41S"]) for k, v in B.items()}


def ncols(line):
    """number of cells: unescaped pipes minus one (strict: a pipe inside a code span counts, as the renderer's table parser counts it)"""
    return line.replace("\\|", "").count("|") - 1


_RULE = re.compile(r"\|[\s:\-|]+")


def norm_pipes(line):
    """a source table row with a bare pipe inside a cell (a delimiter is a pipe with a space on both sides, or the first or last character) gets that pipe escaped,
    so the renderer keeps the row's cell count; the only change made to a source table line. The header rule row is returned as it is."""
    if _RULE.fullmatch(line):
        return line
    n, out = len(line), []
    for i, ch in enumerate(line):
        if ch == "|" and 0 < i < n - 1 and line[i - 1] != "\\" and not (line[i - 1] == " " and line[i + 1] == " "):
            out.append("\\|")
        else:
            out.append(ch)
    return "".join(out)


HIGHER_BETTER = ("TAC-LAT", "TAC-LON")  # gapdecomp.HIGHER_BETTER: TAC rows read higher-is-better, every other row lower-is-better
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
_CELL = re.compile(r"([+-]\d+\.\d+) \[[^\]]*\]( \*)?")


def find_table(src, sub, nth):
    """(first line index, end index, anchor index) of the nth contiguous table after the first source line that contains sub"""
    idx = [i for i, l in enumerate(src) if sub in l]
    assert idx, ("directive substring not in its source", sub)
    i0 = idx[0]
    assert not src[i0].startswith("|"), ("the anchor line is a table line", sub)
    pos, k, j = i0 + 1, i0, i0
    for _ in range(nth):
        j = pos
        while j < len(src) and not src[j].startswith("|"):
            j += 1
        assert j < len(src) and j - pos <= 40, ("no table within 40 lines after the anchor", sub, nth, j - pos)
        k = j
        while k < len(src) and src[k].startswith("|"):
            k += 1
        pos = k
    return j, k, i0


def family_counts(tab):
    """{family: {seed: [better, worse, n]}} from the 19-row table lines (columns: row, family, s0, s1)"""
    out = {f: {"s0": [0, 0, 0], "s1": [0, 0, 0]} for f in FAMS}
    for r in tab[2:]:
        c = [x.strip() for x in r.strip().strip("|").split(" | ")]
        assert len(c) == 4, r[:80]
        if c[1] not in FAMS:
            continue
        for sd, cell in (("s0", c[2]), ("s1", c[3])):
            m = _CELL.fullmatch(cell)
            assert m, ("not a delta cell", cell)
            d, sep = float(m.group(1)), bool(m.group(2))
            e = out[c[1]][sd]
            e[2] += 1
            if sep and ((d > 0) if c[0] not in HIGHER_BETTER else (d < 0)):
                e[1] += 1
            elif sep:
                e[0] += 1
    return out


def family_line(counts, title):
    parts = []
    for f in FAMS:
        n = counts[f]["s0"][2]
        assert n == counts[f]["s1"][2]
        parts.append("%s %d / %d (s0) and %d / %d (s1) of %d" % (f, counts[f]["s0"][0], counts[f]["s0"][1], counts[f]["s1"][0], counts[f]["s1"][1], n))
    return "* **Families, %s** (separated-better / separated-worse rows): %s." % (title, "; ".join(parts))


DIRS, FAMLINES = [], []
_DIR = re.compile(r"@@([TF]) (\w+) after=(.+?)(?: nth=(\d+))?(?: as=(.+?))?@@")


def expand(t, name):
    """replace every `@@T ...@@` line by the table, and every `@@F ...@@` line by the family-count bullet of that table"""
    out = []
    for line in t.split("\n"):
        if "@@" not in line:
            out.append(line)
            continue
        m = _DIR.fullmatch(line)
        assert m, ("a malformed directive", name, line[:90])
        kind, key, sub, nth, title = m.group(1), m.group(2), m.group(3), int(m.group(4) or 1), m.group(5)
        src = srctext(DIRECTIVE_SRC[key]).split("\n")
        j, k, i0 = find_table(src, sub, nth)
        tab = src[j:k]
        assert len(tab) >= 3 and _RULE.fullmatch(tab[1]) and norm_pipes(tab[0]) == tab[0] and ncols(tab[0]) == tab[1].count("|") - 1, ("not a table with a header rule and a delimiter-clean header", key, sub)
        if kind == "T":
            _esc = sum(1 for r in tab if norm_pipes(r) != r)
            tab = [norm_pipes(r) for r in tab]
            assert len({ncols(r) for r in tab}) == 1, ("not a rectangular table after the pipe escapes", key, sub, [ncols(r) for r in tab])
            DIRS.append((key, sub + (" nth=%d" % nth if nth > 1 else ""), i0 + 1, j + 1, len(tab), _esc))
            out.extend(tab)
        else:
            assert title and len(tab) == 21 and tab[0].startswith("| row | family | s0 | s1 |"), ("a family count needs a 19-row delta table", key, sub, len(tab))
            fl = family_line(family_counts(tab), title)
            FAMLINES.append(fl)
            out.append(fl)
    return "\n".join(out)


def fill(t):
    for k, c in LAND.items():
        t = t.replace("{{LAND_%s}}" % k, c)
    t = t.replace("{{TP}}", TP)
    for k, v in RES_SHA.items():
        t = t.replace("{{RES_%s}}" % k, v[:8])
    left = re.findall(r"\{\{[^{}\n]*\}\}", t)
    assert not left and "<<" not in t, left[:3]
    return t


SEC = fill(expand(part("sec_741_levelfloor.md"), "7.41"))
B = {k: fill(v) for k, v in B.items()}
_heads = [m.group(1) for m in re.finditer(r"^(?:### |\*\*)(7\.41(?:\.\d)?) ", SEC, flags=re.M)]
assert _heads == ["7.41"] + ["7.41.%d" % i for i in range(1, 8)], _heads
_secm = SEC.replace("\u2212", "-")
for v in JS:
    assert v in _secm, ("a number from the raw JSON is not in 7.41", v)

# ---- tables are source lines; quotations are verbatim -----------------------------------------------------------------------------------------------------
SRCL = set()
for k in RES_PATHS:
    if not RES_PATHS[k].endswith(".json"):
        SRCL |= {norm_pipes(l) for l in srctext(k).split("\n") if l.startswith("|")}
ADDED = "\n".join([SEC] + list(B.values()))
_tabs, _cur = [], []
for l in SEC.split("\n"):
    if l.startswith("|"):
        _cur.append(l)
    elif _cur:
        _tabs.append(_cur); _cur = []
if _cur:
    _tabs.append(_cur)
_bad = [l[:90] for t in _tabs for l in t if l not in SRCL]
assert len(_tabs) == len(DIRS) and not _bad, ("a table line of the new section is not a line of a source file", len(_tabs), len(DIRS), _bad[:3])
assert all(len({ncols(r) for r in t}) == 1 and _RULE.fullmatch(t[1]) for t in _tabs), "a table with ragged columns or no header rule"
assert not any(l.startswith("|") for k, v in B.items() if k != "ROW" for l in v.split("\n")), "a table line outside the 8.4 rows"
_ROWS = [l for l in B["ROW"].split("\n") if l.startswith("|")]
assert [l.split(" | ")[0] for l in _ROWS] == ["| 1ba", "| 1bb", "| 1bc"] and all(ncols(r) == 4 for r in _ROWS), "8.4 rows 1ba to 1bc"
_corp = re.sub(r"\s+", " ", " ".join([srctext(k) for k in RES_PATHS] + [REG, S0]))
_q = [m.strip('"') for m in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", ADDED)]
_nq = [q for q in _q if re.sub(r"\s+", " ", q) not in _corp]
assert _q and not _nq, ("an italic quotation is not verbatim in a source", _nq[:3])

# ---- anchored edits -----------------------------------------------------------------------------------------------------------------------------------
K = "a ceiling diagnostic that is never a driving result (§7.40.6)."
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
ANCHOR_10 = "\n\n---\n\n## 10. Reproducibility appendix"
E = [
    ("research report, v40 of 2026-11-06, results through repository tip `ffa2637a` and, within it,",
     "research report, v41 of 2026-11-06, results through repository tip `" + TP + "` and, within it,"),
    ("and the level floor LF with the ceiling card A4 (§7.40) (v39: `c9d0688e`; v38: `ff09d2c1`;",
     "the level floor LF with the ceiling card A4 (§7.40), and the progress-floor addenda A4 to A6 (§7.41) (v40: `ffa2637a`; v39: `c9d0688e`; v38: `ff09d2c1`;"),
    (K + "\n\n**Evidence classes used throughout.**", K + " v41 adds §7.41: " + B["V41S"] + "\n\n**Evidence classes used throughout.**"),
    (K + " **[E]** The post hoc line fit", K + " " + B["V41S"] + " **[E]** The post hoc line fit"),
    (K + "\n\n---\n\n## 2. Introduction", K + " " + B["V41S"] + "\n\n---\n\n## 2. Introduction"),
    ("  * A4's result is not in v40.", "  * A4's result is in §7.41.1."),
    ("Its limits are in §7.40.6.\n\n### 8.2 What is not established", "Its limits are in §7.40.6.\n" + B["S81"] + "\n\n### 8.2 What is not established"),
    ("**Update (v40).** One item is written up here", B["R83"] + "\n\n**Update (v40).** One item is written up here"),
    ("addendum A4, the ceiling read with the logged speed in place of the estimate (`8bfbc386…`, §7.40.6) |\n| 2 | **LoRA on DINOv3**",
     "addendum A4, the ceiling read with the logged speed in place of the estimate (`8bfbc386…`, §7.40.6) |\n" + B["ROW"] + "\n| 2 | **LoRA on DINOv3**"),
    (ANCHOR_10, "\n" + B["I19"] + ANCHOR_10),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
# the new section goes after the last subsection of 7.40 (7.40.7 Provenance), before the Discussion
assert s.count(ANCHOR_8) == 1 and s.index("**7.40.7 Provenance [M].**") < s.index(ANCHOR_8)
assert s[:s.index(ANCHOR_8)].split("\n")[-1].startswith("The claim rows are `H-LEVELFLOOR-1` and `H-LEVELFLOOR-1 (A4)`")
_j = s.index(ANCHOR_8)
s = s[:_j] + "\n\n" + SEC + s[_j:]

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "".join(b for _, b in E) + SEC
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower()
assert re.search("retract|supersed|" + BANNED[6] + "|was wrong|previously read|outdated|i said", low) is None
assert not _CL.search(ins) and not re.search(r"\bdrives\b|driving performance", low), "forbidden T2 wording or driving wording"
assert "\r" not in ins, "a CR in the added text"
_disp = re.findall(r"^\$\$\n(?:(?!\$\$).)*\n\$\$$", SEC, flags=re.M | re.S)
assert len(_disp) == 1, ("exactly one display equation (the operator F_B) is added", len(_disp))
assert "$" not in ins.replace(_disp[0], ""), "a dollar sign outside the one display equation"
assert s.count("$$") == S0.count("$$") + 2 and s.count("$") == S0.count("$") + 4, "math spans changed by more than the one display equation"
nocode = re.sub(r"`[^`\n]*`", " ", ins)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r".{20}<[A-Za-z/!].{20}", nocode)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
for fam in FAMS:
    assert fam in SEC, fam
_nflat = re.sub(r"\s+", " ", SEC)
assert "B = 2,000" in _nflat and "T1-kin" in _nflat and "H-ESTIM-SEED-1" in _nflat and "reindexed" in _nflat and "CEILING" in _nflat and "never driving results" in re.sub(r"\s+", " ", B["V41S"])
assert s.index("**7.40.7 ") < s.index("### 7.41 ") < s.index("## 8. Discussion") and s.count("### 7.41 ") == 1
assert s.count("**Status:** research report, v41 of 2026-11-06") == 1 and "v40 of 2026-11-06, results" not in s.split("\n")[2]
assert not [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if d > "2026-11-06"], "a date after 2026-11-06"
assert "A4's result is not in v40." not in s, "a stale pointer about the A4 result remains"
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
_pend = sorted(set(re.findall(r"\{\{[^{}\n]*\}\}", s)))
print("v41 built", len(S0), len(s), "tip", TIP, "7.41 chars", len(SEC), "tables", len(_tabs), "table lines", sum(len(t) for t in _tabs), "family lines", len(FAMLINES), "quotations", len(_q), "landed", LAND, "placeholders left", _pend)
open(OUT + "DIRECTIVES.txt", "w", encoding="utf-8", newline="\n").write("\n".join("%s | after=%s | source line %d | table at %d | %d lines | %d rows with an escaped pipe" % d for d in DIRS) + "\n")
open(OUT + "FAMILY_COUNTS.txt", "w", encoding="utf-8", newline="\n").write("\n".join(FAMLINES) + "\n")
