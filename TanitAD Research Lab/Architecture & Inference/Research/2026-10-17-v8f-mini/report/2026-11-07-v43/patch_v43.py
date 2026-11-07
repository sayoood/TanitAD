"""Report v43 = v42 + sections 7.45 (the T2 attribution, B-prime and F-prime, the LoRA launch, the T1-kin BLIND read, GAP48 and RALIGN) and 7.46 (the theory after six reads), inserted after 7.44.9, + the status line,
the abstract (two bullets and the next-bullet sentence), 8.1 item 33, 8.3 Update (v43), 8.4 rows 1bg to 1bl, 9 item 21, the section 10 appendix block (one row per new raw folder), and the carried v42 lines that
said WSP-GAIN was unread or that a0 is a 0.2 s difference (edited in place, each listed in edits_v43.md as old -> new).
Built from the landed v42 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once, or the stated number of times). The prose is in the package files beside this script
(sec_745_t2attrib.md, sec_746_theory.md: the section texts; edits_v43.md: one block per edit, V43S being the shared v43 sentence that the blocks name as <<V43S>>, and the C* blocks being
`old -->` `new` pairs for carried lines); every source is `git show <tip>:<path>` from the mirror. Nothing is read from a working tree of the repository. Nothing is staged, committed or pushed.

Directives in the section files:
  `@@T KEY after=SUBSTR [nth=N]@@` is replaced by the Nth (default 1st) contiguous table (lines starting with `|`) after the first line of the KEY source file that contains SUBSTR, copied verbatim
  except that a bare pipe inside a cell is escaped so that the rendered row keeps its cell count (as in v39 to v42).
Fill-ins: `{{TP}}` (tip, 8 hex), `{{LAND_x}}` (first commit that added the source file x), `{{RES_x}}` (sha256[:8] of source file x).

python -I patch_v43.py tip [out_dir]
Every italic quotation of 25 characters or more is asserted verbatim in a source (every md / txt / json raw file of 2026-11-02 to 2026-11-07, the register, v42)."""
import hashlib, json, os, re, subprocess, sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
assert _argv, "usage: patch_v42.py <tip> [out_dir]"
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


# ---- the v42 deliverables are the starting point ---------------------------------------------------------------------------------------------------------
V42_MD_SHA = "ada2083063ed8bebcac82ee64444d8760b2766c8bda643c51a8d1af96c637aea"
V42_HTML_SHA = "0bf58200cade629f71c58ebba976c70f16c119a35d74a0c558447c8a8dc04508"
RP42 = R + "report/2026-11-07-v42/"  # the landed v42 files sit directly in this folder (no out/)
s = blob(RP42 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP42 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP42 + "V8F_SCIENTIFIC_REPORT.md")) == V42_MD_SHA, "the tip's report md is not the landed v42 deliverable"
assert sha(raw(RP42 + "v8f_report.html")) == V42_HTML_SHA, "the tip's report html is not the landed v42 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v42 html is not the template with the v42 md"
assert s.count("**Status:** research report, v42 of 2026-11-07") == 1
S0 = s

# ---- sources ----------------------------------------------------------------------------------------------------------------------------------------------
_D = "raw/2026-11-07-"
RES_PATHS = OrderedDict([
    ("AT", _D + "t2-attrib-prereg/PREREG_T2_ATTRIB.md"),
    ("MCP", _D + "t2-mc-read/stage5_MC/pair_C0C_MC/PAIR_T2.md"),
    ("BPC", _D + "t2-bpc-read/RT_VERDICT.json"),
    ("FPR", _D + "lora-fprime/out/FPRIME.json"),
    ("PFP", _D + "t2-pair-fprime-prep/MANIFEST.txt"),
    ("LCS", _D + "t2-lc-support/HANDOFF_LC.md"),
    ("PN5", _D + "t2-pin5/README_BANK.md"),
    ("S5D", _D + "t2-stage5-driver/HANDOFF_S5.md"),
    ("T1B", _D + "t1-blind/RESULT_T1BLIND.md"),
    ("GAP", _D + "t2-ralign-gap48/gap48/RESULT_GAP48.md"),
    ("RENC", _D + "t2-renc-prereg/PREREG_T2_RENC.md"),
    ("P1B", _D + "theory-p1pb/P1PB_RESULT.md"),
    ("WG2", _D + "wsp-gain/raw/RESULTS_READ2.md"),
    ("AUC", _D + "aucmap/RESULTS_AUCMAP.md"),
    ("A3", _D + "theory-amend3/THEORY_CRAWL_AMEND_3.md"),
    ("P5R", _D + "p5-dilution/p5/P5_RESULT.md"),
    ("WGD", _D + "wsp-gain-d/RESULT_WSPD.md"),
    ("P4C", _D + "p4-cueclass/PREREG_P4_CUECLASS.md"),
    ("DK", _D + "dk141/README_DK141.md"),
    ("DOM", _D + "t2-dom-lora-freeze/PREREG_T2_DOM_LORA.md"),
    ("WLR", _D + "wsp-lora-prereg/PREREG_WSP_LORA.md"),
    ("A4", _D + "t1-blind/cards/THEORY_CRAWL_AMEND_4.md"),
])
LAND_SRC = OrderedDict((k, k) for k in RES_PATHS if k != "A4")
DIRECTIVE_SRC = {"PAIRMC0": "MCP", "PAIRMC1": "MCP", "WSPREAD2": "WG2", "WSPDREAD2": "WGD"}
_SRC_CACHE = {}


def srctext(k):
    if k not in _SRC_CACHE:
        assert k in RES_PATHS, ("no source path for directive key", k)
        if RES_PATHS[k].endswith(".json"):  # raw JSON reads may carry CRLF (written on Windows); they are only parsed or searched, never copied
            _SRC_CACHE[k] = raw(R + RES_PATHS[k]).decode("utf-8-sig").replace(chr(13), "")
        else:
            _SRC_CACHE[k] = blob(R + RES_PATHS[k])
    return _SRC_CACHE[k]


RES_SHA = {k: sha(raw(R + p)) for k, p in RES_PATHS.items()}
LAND = {}
for k, src in LAND_SRC.items():
    c = gitrun(["log", "--format=%h", "--diff-filter=A", TIP, "--", R + RES_PATHS[src]]).split("\n")[-1]
    assert re.fullmatch(r"[0-9a-f]{7,}", c), (k, c)
    LAND[k] = c[:8]


def all_raw_corpus():
    """every md / txt / json blob (up to 3 MB) under raw/2026-11-02-* to raw/2026-11-07-* at the tip, one git cat-file --batch: the corpus the italic quotations and the frozen sha prefixes are asserted against"""
    names = [n for n in gitrun(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"]).split("\n") if re.search(r"/2026-11-0[2-7]-", n) and n.endswith((".md", ".txt", ".json"))]
    p = subprocess.Popen(["git", "-C", MIRROR, "-c", "safe.directory=*", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    out = []
    for n in names:
        p.stdin.write(f"{TIP}:{n}\n".encode()); p.stdin.flush()
        size = int(p.stdout.readline().decode().split()[2])
        data = p.stdout.read(size); p.stdout.read(1)
        if size <= 3_000_000:
            out.append(data.decode("utf-8", "replace").replace("\r", ""))
    p.stdin.close(); p.wait()
    return out, len(names)


_RAWS, _NRAW = all_raw_corpus()
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
_BRIEF = open(HERE + "/BUILD_BRIEF_V43.md", encoding="utf-8").read()  # the Master Mind's brief records the PI's message of 2026-11-07 verbatim; it is the source of that one quotation
_corp = re.sub(r"\s+", " ", " ".join(_RAWS + [REG, S0, _BRIEF]).replace("**", ""))  # quotations are verbatim modulo the bold markers of the source
# every frozen sha256 prefix the new text names is asserted against the corpus (a card, its frozen-sha record, a log or a result file)
HARD_NEW = ["f7d2cfa1", "9619414a", "a6c84433", "016c2950", "6717a0cf", "425d3496", "dcb82ae9", "92d61dcb", "d6e691a3", "626aa1e5", "7fd65ffd", "56a91667", "15be11b5", "f2cc7efb", "1daf6fbc", "27cab227", "cd6a7ab5", "6e83451c", "9d39ac53", "0427c325", "cad25e83", "0529fcfb"]
_missing = [p for p in HARD_NEW if p not in _corp]
assert not _missing, ("a hard-coded sha256 prefix is in no tip record", _missing)
for row in ("H-T2-ATTRIB-1", "D-T2-ATTRIB-PT-1", "D-T2-ATTRIB-RT-1", "D-V8F-LORA-FPRIME-1", "D-T2-PAIR-1", "D-T2-LC-1", "D-T2-PIN5-1", "D-V8F-T2-STAGE5-1", "D-V8F-T1-BLIND-1", "D-V8F-T2-RALIGN-1", "H-T2-RENC-1",
            "D-V8F-THEORY-P1P-1", "D-V8F-WSPGAIN-1", "D-THEORY-AUCMAP-1", "H-THEORY-T5-1", "D-THEORY-T5-1", "D-V8F-WSPGAIN-D-1", "H-THEORY-P4-1", "D-DK141-1", "H-T2-DOM-LORA-1", "H-WSP-LORA-1"):
    assert row in REG, ("a register row the provenance names", row)

# ---- the section texts ------------------------------------------------------------------------------------------------------------------------------------
_blocks = re.split(r"^=== (\w+) ===\n", part("edits_v43.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
PROSE = ["ABS1", "APP", "I21", "NXT", "R83", "ROW", "S81", "V43S"]
CARRIED = ["C3A", "C3B", "C42S", "C63", "C64", "C5951", "C6030", "C6036", "C6040", "C6118", "C6247", "C6300"]
assert sorted(B) == sorted(PROSE + CARRIED + ["NOTES"]), sorted(B)  # NOTES is documentation of the edits and is never inserted
assert "\n" not in B["V43S"] and "<<" not in B["V43S"] and "{{" not in B["V43S"]
B = {k: v.replace("<<V43S>>", B["V43S"]) for k, v in B.items()}


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


FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")


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


DIRS = []
_DIR = re.compile(r"@@T (\w+) after=(.+?)(?: nth=(\d+))?@@")


def expand(t, name):
    """replace every `@@T ...@@` line by the table it names"""
    out = []
    for line in t.split("\n"):
        if "@@" not in line:
            out.append(line)
            continue
        m = _DIR.fullmatch(line)
        assert m, ("a malformed directive", name, line[:90])
        key, sub, nth = m.group(1), m.group(2), int(m.group(3) or 1)
        src = srctext(DIRECTIVE_SRC[key]).split("\n")
        j, k, i0 = find_table(src, sub, nth)
        tab = src[j:k]
        assert len(tab) >= 3 and _RULE.fullmatch(tab[1]) and norm_pipes(tab[0]) == tab[0] and ncols(tab[0]) == tab[1].count("|") - 1, ("not a table with a header rule and a delimiter-clean header", key, sub)
        _esc = sum(1 for r in tab if norm_pipes(r) != r)
        tab = [norm_pipes(r) for r in tab]
        assert len({ncols(r) for r in tab}) == 1, ("not a rectangular table after the pipe escapes", key, sub, [ncols(r) for r in tab])
        DIRS.append((key, sub + (" nth=%d" % nth if nth > 1 else ""), i0 + 1, j + 1, len(tab), _esc))
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


SECS = [fill(expand(part(f), f)) for f in ("sec_745_t2attrib.md", "sec_746_theory.md")]
SEC = "\n\n".join(SECS)
B = {k: fill(v) for k, v in B.items()}
_heads = [m.group(1) for m in re.finditer(r"^(?:### |\*\*)(7\.4[56](?:\.\d+)?) ", SEC, flags=re.M)]
assert _heads == ["7.45"] + ["7.45.%d" % i for i in range(1, 9)] + ["7.46"] + ["7.46.%d" % i for i in range(1, 12)], _heads

# ---- tables are source lines (directive tables) or hand-written and named; quotations are verbatim -----------------------------------------------------
SRCL = set()
for k in RES_PATHS:
    if not RES_PATHS[k].endswith(".json"):
        SRCL |= {norm_pipes(l) for l in srctext(k).split("\n") if l.startswith("|")}
_tabs, _cur = [], []
for l in SEC.split("\n"):
    if l.startswith("|"):
        _cur.append(l)
    elif _cur:
        _tabs.append(_cur); _cur = []
if _cur:
    _tabs.append(_cur)
_src_tabs = [t for t in _tabs if all(l in SRCL for l in t)]
_hand_tabs = [t for t in _tabs if not all(l in SRCL for l in t)]
assert len(_src_tabs) == len(DIRS) == 4 and sum(len(t) for t in _src_tabs) == sum(d[4] for d in DIRS), ("directive tables are source lines", len(_src_tabs), len(DIRS))
assert len(_hand_tabs) == 6, ("hand-written tables of 7.45 and 7.46 (RT rows, BLIND rows, P1'b set, AUCMAP levers, P5 clauses, predictions)", len(_hand_tabs))
assert all(len({ncols(r) for r in t}) == 1 and _RULE.fullmatch(t[1]) for t in _tabs), "a table with ragged columns or no header rule"
assert not any(l.startswith("|") for k, v in B.items() if k in PROSE and k not in ("ROW", "APP") for l in v.split("\n")), "a table line outside the 8.4 rows and the appendix table"
_ROWS = [l for l in B["ROW"].split("\n") if l.startswith("|")]
assert [l.split(" | ")[0] for l in _ROWS] == ["| 1bg", "| 1bh", "| 1bi", "| 1bj", "| 1bk", "| 1bl"] and all(ncols(r) == 4 for r in _ROWS), "8.4 rows 1bg to 1bl"
_APPT = [l for l in B["APP"].split("\n") if l.startswith("|")]
assert len(_APPT) == 23 and len({ncols(r) for r in _APPT}) == 1 and _RULE.fullmatch(_APPT[1]), "the appendix table (one row per new raw folder: 21)"
ADDED = "\n".join([SEC] + [B[k] for k in PROSE])
_q = [m.strip('"').replace("∗", "*") for m in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", ADDED.replace("\\*", "∗"))]  # an escaped star is a literal star
_nq = [q for q in _q if re.sub(r"\s+", " ", q) not in _corp]
assert _q and not _nq, ("an italic quotation is not verbatim in a source", _nq[:5])

# ---- anchored edits -----------------------------------------------------------------------------------------------------------------------------------
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
ANCHOR_10 = "\n\n---\n\n## 10. Reproducibility appendix"
_pairs = {}
for k in CARRIED:
    a, b = B[k].split("\n-->\n")
    assert a and b and a != b, k
    _pairs[k] = (a, b)
CARRIED_EDITS = [_pairs[k] for k in CARRIED]
E = CARRIED_EDITS + [
    ("nothing in v42 is a driving result.\n\n**Evidence classes used throughout.**", "nothing in v42 is a driving result. " + B["V43S"] + "\n\n**Evidence classes used throughout.**"),
    ("\n* **What is next.** The crawl set is still open", "\n" + B["ABS1"] + "\n* **What is next.** The crawl set is still open"),
    ("then a zero-training frontier point.\n\n---\n\n## 2. Introduction", "then a zero-training frontier point. " + B["NXT"] + "\n\n---\n\n## 2. Introduction"),
    ("Its limits are in §7.42.6, §7.43.7 and §7.44.8.\n\n### 8.2 What is not established", "Its limits are in §7.42.6, §7.43.7 and §7.44.8.\n" + B["S81"] + "\n\n### 8.2 What is not established"),
    ("**Update (v42).** Three items are written up here", B["R83"] + "\n\n**Update (v42).** Three items are written up here"),
    ("\n| 2 | **LoRA on DINOv3**", "\n" + B["ROW"] + "\n| 2 | **LoRA on DINOv3**"),
    ("\n\n**Figures.** `make_figures.py` in this folder", "\n\n" + B["APP"] + "\n\n**Figures.** `make_figures.py` in this folder"),
    (ANCHOR_10, "\n" + B["I21"] + ANCHOR_10),
]
for a, b in E:
    want_n = 2 if a == _pairs["C42S"][0] else 1  # the v42 sentence sits in the status line and in 8.1 item 32
    assert s.count(a) == want_n, (s.count(a), want_n, a[:90])
    s = s.replace(a, b)
# the new sections go after the last subsection of 7.44 (7.44.9 Provenance), before the Discussion
assert s.count(ANCHOR_8) == 1 and s.index("**7.44.9 Provenance") < s.index(ANCHOR_8)
assert s[:s.index(ANCHOR_8)].split("\n")[-1].startswith("**7.44.9 Provenance")
_j = s.index(ANCHOR_8)
s = s[:_j] + "\n\n" + SEC + s[_j:]

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "".join(b for _, b in E) + SEC
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower().replace("retraction_log", "")  # the file name RETRACTION_LOG.md is a path, not prose
assert re.search("retract|supersed|" + BANNED[6] + "|was wrong|previously read|outdated|i said", low) is None
assert not _CL.search(ins) and not re.search(r"\bdrives\b|driving performance", low), "forbidden T2 wording or driving wording"
assert "\r" not in ins, "a CR in the added text"
# math guard: strip fenced and inline code, then the display equations, then the template's inline-math pattern; no dollar sign may remain
_nc = re.sub(r"```[\s\S]*?```", " ", ins)
_nc = re.sub(r"`[^`\n]+`", " ", _nc)
_disp = re.findall(r"\$\$[\s\S]+?\$\$", _nc)
_nd = re.sub(r"\$\$[\s\S]+?\$\$", " ", _nc)
_inl = re.findall(r"(?:^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", _nd, flags=re.M)
_ni = re.sub(r"(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", r"\1 ", _nd, flags=re.M)
assert "$" not in _ni, ("a dollar sign outside the display and inline math", re.search(r".{30}\$.{30}", _ni, flags=re.S))
assert ins.count("$$") % 2 == 0 and s.count("$$") == S0.count("$$") + ins.count("$$"), "display math fences"
assert len(_disp) == 5 and len(_inl) > 20, (len(_disp), len(_inl))
_nm = re.sub(r"\$\$[\s\S]+?\$\$", " ", re.sub(r"`[^`\n]+`", " ", ins))
_nm = re.sub(r"(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", r"\1 ", _nm, flags=re.M)
assert re.search(r"<[A-Za-z/!]", _nm) is None, re.search(r".{20}<[A-Za-z/!].{20}", _nm)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
for fam in FAMS:
    assert fam in SECS[0], fam
_nflat = re.sub(r"\s+", " ", SECS[0])
assert "B = 2,000" in _nflat and "T2" in _nflat and "H-ESTIM-SEED-1" in _nflat and "scene-cluster" in _nflat and "NOT-DOMINANT" in _nflat
assert s.index("**7.44.9 ") < s.index("### 7.45 ") < s.index("### 7.46 ") < s.index("## 8. Discussion")
assert all(s.count("### 7.4%d " % i) == 1 for i in (2, 3, 4, 5, 6))
assert s.count("**Status:** research report, v43 of 2026-11-07") == 1 and "v42 of 2026-11-07, results" not in s.split("\n")[2]
assert not [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if d > "2026-11-07"], "a date after 2026-11-07"
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
_pend = sorted(set(re.findall(r"\{\{[^{}\n]*\}\}", s)))
print("v43 built", len(S0), len(s), "tip", TIP, "7.45-7.46 chars", len(SEC), "directive tables", len(_src_tabs), "hand tables", len(_hand_tabs), "display eq", len(_disp), "inline math", len(_inl), "quotations", len(_q), "raw files", _NRAW, "landed", LAND, "placeholders left", _pend)
open(OUT + "DIRECTIVES.txt", "w", encoding="utf-8", newline="\n").write("\n".join("%s | after=%s | source line %d | table at %d | %d lines | %d rows with an escaped pipe" % d for d in DIRS) + "\n")
