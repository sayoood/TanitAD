"""Report v40 = v39 + section 7.40 (addendum A3, the level floor LF: card, rulings, build, the read at both doses, the instruments, what it establishes, the ceiling card A4),
inserted after 7.39.8, + the status line (version, tip, version list, the LF sentence), the abstract (the recovery bullet and the next bullet), the 7.39.7 pointer to 7.40,
8.1 item 30, 8.3 Update (v40), 8.4 row 1az, 9 item 18.
Built from the landed v39 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once). The prose is in the package files beside this script
(sec_740_levelfloor.md: the section text; edits_v40.md: one block per edit, LFS being the shared LF sentence that the blocks name as <<LFS>>); every source is `git show <tip>:<path>`
from the mirror. Nothing is read from a working tree of the repository. Nothing is staged, committed or pushed.

Directives in the section file: a line `@@T KEY after=SUBSTR@@` is replaced by the first contiguous table (lines starting with `|`) after the first line of the KEY source file that contains SUBSTR, copied verbatim
except that a bare pipe inside a cell is escaped so that the rendered row keeps its cell count (as in v39).
Fill-ins: `{{TP}}` (tip, 8 hex), `{{LAND_x}}` (first commit that added the source file x), `{{RES_x}}` (sha256[:8] of source file x).

python -I patch_v40.py tip [out_dir]
Every italic quotation of 25 characters or more is asserted verbatim in a source (RESULTS file, cards, rulings, register, v39)."""
import hashlib, json, os, re, subprocess, sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
assert _argv, "usage: patch_v40.py <tip> [out_dir]"
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


# ---- the v39 deliverables are the starting point ---------------------------------------------------------------------------------------------------------
V39_MD_SHA = "8ccffcf01f059f10ea0316d214dd40b96c480c3f44f07a5b99e824284326adab"
V39_HTML_SHA = "3f2fb6d1280de3ac010465351af7fcbdad943be2a2826bea1b0faf072d81d22e"
RP39 = R + "report/2026-11-05-v39/"
s = blob(RP39 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP39 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP39 + "V8F_SCIENTIFIC_REPORT.md")) == V39_MD_SHA, "the tip's report md is not the landed v39 deliverable"
assert sha(raw(RP39 + "v8f_report.html")) == V39_HTML_SHA, "the tip's report html is not the landed v39 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v39 html is not the template with the v39 md"
assert s.count("**Status:** research report, v39 of 2026-11-05") == 1
S0 = s

# ---- sources ----------------------------------------------------------------------------------------------------------------------------------------------
RES_PATHS = OrderedDict([
    ("A3P", "raw/2026-11-05-levelfloor-a3-prereg/PREREG_LEVELFLOOR_A3.md"),
    ("AR31", "raw/2026-11-05-levelfloor-lf-result/a3_prereg_additions/A3_RULING_1.md"),
    ("AR32", "raw/2026-11-05-levelfloor-lf-result/a3_prereg_additions/A3_RULING_2.md"),
    ("LF", "raw/2026-11-05-levelfloor-lf-result/lf_read/RESULTS.md"),
    ("LFJ", "raw/2026-11-05-levelfloor-lf-result/lf_read/raw/LF_READ.json"),
    ("LFB", "raw/2026-11-05-levelfloor-lf-result/lf_build/HANDOFF_LFBUILD.md"),
    ("A4P", "raw/2026-11-06-levelfloor-a4-prereg/PREREG_LEVELFLOOR_A4.md"),
])
LAND_KEYS = ("A3P", "LF", "A4P")
DIRECTIVE_SRC = {"LF": "LF"}
# hard-coded sha256 prefixes in the section texts and edits, asserted against the tip's blobs
HARD = [("raw/2026-11-05-levelfloor-a3-prereg/PREREG_LEVELFLOOR_A3.md", "977a785e"), ("raw/2026-11-06-levelfloor-a4-prereg/PREREG_LEVELFLOOR_A4.md", "8bfbc386"),
        ("raw/2026-11-05-levelfloor-lf-result/a3_prereg_additions/A3_RULING_1.md", "e2f120d9")]
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
assert LAND["A3P"].startswith("c9d0688") and LAND["LF"].startswith("0f5550a") and LAND["A4P"].startswith("ffa2637"), ("landing commits", LAND)

# ---- raw JSON cross-checks of the headline numbers --------------------------------------------------------------------------------------------------------
J = json.loads(srctext("LFJ"))
assert J["card_sha256"].startswith("977a785e") and sorted(J["doses"]) == ["050", "075"], (J["card_sha256"][:8], sorted(J["doses"]))
assert all(v is True or (isinstance(v, dict) and v.get("PASS", v.get("pass", True)) is True) for v in J["controls"].values()) if isinstance(J["controls"], dict) else True
for k in ("050", "075"):
    assert "P-FAILS-M1-MOVED" in json.dumps(J["doses"][k]["outcome"]), (k, J["doses"][k]["outcome"])
INS = J["instruments"]
JS = {"bind_m1": {r: "%.3f" % INS[r]["bind_rate_in_m1_ticks"]["rate"] for r in ("RILF050_s0", "RILF050_s1", "RILF075_s0", "RILF075_s1")},
      "vf_mae": {r: "%.3f" % INS[r]["vf_vs_logged_overall"]["mae_mps"] for r in ("RILF075_s0", "RILF075_s1")},
      "vf_gt": {r: "%.3f" % INS[r]["vf_vs_logged_overall"]["share_vf_exceeds_logged"] for r in ("RILF075_s0", "RILF075_s1", "RILF050_s0", "RILF050_s1")}}
assert INS["RILF050_s0"]["lift_where_bound_overall"]["n_negative_chord_lift"] == 52 and INS["RILF050_s0"]["lift_where_bound_overall"]["n_bound"] == 543, "the ruling's 52 of 543"
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
assert "**H-LEVELFLOOR-1 (A4)**" in REG and "**H-LEVELFLOOR-1** (2026-11-05, arms LF050 / LF075 READ)" in REG, "the register rows the provenance names"

# ---- the section texts ------------------------------------------------------------------------------------------------------------------------------------
_blocks = re.split(r"^=== (\w+) ===\n", part("edits_v40.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
assert sorted(B) == ["I18", "LFS", "R83", "ROW", "S81"], sorted(B)
assert "\n" not in B["LFS"] and "<<" not in B["LFS"]
B = {k: v.replace("<<LFS>>", B["LFS"]) for k, v in B.items()}


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


DIRS = []


def expand(t, name):
    """replace every `@@T KEY after=SUBSTR@@` line by the first contiguous table after the first source line containing SUBSTR"""
    out = []
    for line in t.split("\n"):
        m = re.fullmatch(r"@@T (\w+) after=(.+?)@@", line)
        if not m:
            assert "@@" not in line, ("a malformed directive", name, line[:80])
            out.append(line)
            continue
        key, sub = m.group(1), m.group(2)
        src = srctext(DIRECTIVE_SRC[key]).split("\n")
        idx = [i for i, l in enumerate(src) if sub in l]
        assert idx, ("directive substring not in its source", key, sub)
        i0 = idx[0]
        assert not src[i0].startswith("|"), ("the anchor line is a table line", key, sub)
        j = i0 + 1
        while j < len(src) and not src[j].startswith("|"):
            j += 1
        assert j < len(src) and j - i0 <= 40, ("no table within 40 lines after the anchor", key, sub, j - i0)
        k = j
        while k < len(src) and src[k].startswith("|"):
            k += 1
        tab = src[j:k]
        assert len(tab) >= 3 and _RULE.fullmatch(tab[1]) and norm_pipes(tab[0]) == tab[0] and ncols(tab[0]) == tab[1].count("|") - 1, ("not a table with a header rule and a delimiter-clean header", key, sub)
        _esc = sum(1 for r in tab if norm_pipes(r) != r)
        tab = [norm_pipes(r) for r in tab]
        assert len({ncols(r) for r in tab}) == 1, ("not a rectangular table after the pipe escapes", key, sub, [ncols(r) for r in tab])
        DIRS.append((key, sub, i0 + 1, j + 1, len(tab), _esc))
        out.extend(tab)
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


SEC = fill(expand(part("sec_740_levelfloor.md"), "7.40"))
B = {k: fill(v) for k, v in B.items()}
_heads = [m.group(1) for m in re.finditer(r"^(?:### |\*\*)(7\.40(?:\.\d)?) ", SEC, flags=re.M)]
assert _heads == ["7.40"] + ["7.40.%d" % i for i in range(1, 8)], _heads
for r, v in JS["bind_m1"].items():
    assert v in SEC, ("an M1 bind rate from the raw JSON is not in 7.40", r, v)
for grp in ("vf_mae", "vf_gt"):
    for r, v in JS[grp].items():
        assert v in SEC, ("a v_f instrument from the raw JSON is not in 7.40", grp, r, v)

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
assert not any(l.startswith("|") for k, v in B.items() if k != "ROW" for l in v.split("\n")), "a table line outside the 8.4 row"
_ROWS = [l for l in B["ROW"].split("\n") if l.startswith("|")]
assert [l.split(" | ")[0] for l in _ROWS] == ["| 1az"] and ncols(_ROWS[0]) == 4, "8.4 row 1az"
_corp = re.sub(r"\s+", " ", " ".join([srctext(k) for k in RES_PATHS] + [REG, S0]))
_q = [m.strip('"') for m in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", ADDED)]
_nq = [q for q in _q if re.sub(r"\s+", " ", q) not in _corp]
assert _q and not _nq, ("an italic quotation is not verbatim in a source", _nq[:3])

# ---- anchored edits -----------------------------------------------------------------------------------------------------------------------------------
K = "as a speed floor (§7.39.6, §7.39.7)."
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
E = [
    ("research report, v39 of 2026-11-05, results through repository tip `c9d0688e` and, within it,",
     "research report, v40 of 2026-11-06, results through repository tip `" + TP + "` and, within it,"),
    ("with the lead head LLR (§7.39) (v38: `ff09d2c1`;",
     "with the lead head LLR (§7.39), and the level floor LF with the ceiling card A4 (§7.40) (v39: `c9d0688e`; v38: `ff09d2c1`;"),
    (K + "\n\n**Evidence classes used throughout.**", K + " v40 adds §7.40: " + B["LFS"] + "\n\n**Evidence classes used throughout.**"),
    (K + " **[E]** The post hoc line fit", K + " " + B["LFS"] + " **[E]** The post hoc line fit"),
    (K + "\n\n---\n\n## 2. Introduction", K + " " + B["LFS"] + "\n\n---\n\n## 2. Introduction"),
    ("Its result is not in v39.", "Its result is in §7.40."),
    (K + " Its limits are in §7.39.7.\n\n### 8.2 What is not established", K + " Its limits are in §7.39.7.\n" + B["S81"] + "\n\n### 8.2 What is not established"),
    ("**Update (v39).** Three items are written up here", B["R83"] + "\n\n**Update (v39).** Three items are written up here"),
    (K + " |\n| 2 | **LoRA on DINOv3**", K + " |\n" + B["ROW"] + "\n| 2 | **LoRA on DINOv3**"),
    ("\n\n---\n\n## 10. Reproducibility appendix", "\n" + B["I18"] + "\n\n---\n\n## 10. Reproducibility appendix"),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
# the new section goes after the last subsection of 7.39 (7.39.8 Provenance), before the Discussion
assert s.count(ANCHOR_8) == 1 and s[:s.index(ANCHOR_8)].split("\n")[-1].startswith("**7.39.8 Provenance [M].**")
_j = s.index(ANCHOR_8)
s = s[:_j] + "\n\n" + SEC + s[_j:]

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "".join(b for _, b in E) + SEC
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower()
assert re.search("retract|supersed|" + BANNED[6] + "|was wrong|previously read|outdated|i said", low) is None
assert not _CL.search(ins) and not re.search(r"\bdrives\b|driving performance", low), "forbidden T2 wording or driving wording"
assert "\r" not in ins and "$" not in ins, "a CR or a dollar sign in the added text (the math count of the report stays as in v39)"
assert s.count("$$") == S0.count("$$") and s.count("$") == S0.count("$"), "math spans changed"
nocode = re.sub(r"`[^`\n]*`", " ", ins)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r".{20}<[A-Za-z/!].{20}", nocode)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
    assert fam in SEC, fam
_nflat = re.sub(r"\s+", " ", SEC)
assert "B = 2,000" in _nflat and "T1-kin" in _nflat and "H-ESTIM-SEED-1" in _nflat and "reindexed" in _nflat and "CEILING" in _nflat
assert s.index("**7.39.8 ") < s.index("### 7.40 ") < s.index("## 8. Discussion") and s.count("### 7.40 ") == 1
assert s.count("**Status:** research report, v40 of 2026-11-06") == 1 and "v39 of 2026-11-05, results" not in s.split("\n")[2]
assert not [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if d > "2026-11-06"], "a date after 2026-11-06"
assert "Its result is not in v39." not in s, "a stale pointer about the A3 result remains"
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
_pend = sorted(set(re.findall(r"\{\{[^{}\n]*\}\}", s)))
print("v40 built", len(S0), len(s), "tip", TIP, "7.40 chars", len(SEC), "tables", len(_tabs), "table lines", sum(len(t) for t in _tabs), "quotations", len(_q), "landed", LAND, "placeholders left", _pend)
open(OUT + "DIRECTIVES.txt", "w", encoding="utf-8", newline="\n").write("\n".join("%s | after=%s | source line %d | table at %d | %d lines | %d rows with an escaped pipe" % d for d in DIRS) + "\n")
