"""Report v39 = v38 + section 7.37 (RPT and the PI's answer), 7.38 (option (c), motion from vision), 7.39 (option (a), the level-from-image head: CL, CLR and the lead head LLR),
inserted after 7.36.7, + the status line, the abstract (the recovery bullet and the next pointer), a pointer in 7.36.6, 8.1 items 26 to 29, 8.3 Update (v39), 8.4 rows 1au to 1ay (and the row 1at pointer), 9 item 17.
Built from the landed v38 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once). The prose is in the package files beside this script
(sec_737_rpt.md, sec_738_motion_c.md, sec_739_levelhead.md + sec_739_llr_result.md + sec_739_tail.md: the section texts; edits_v39.md: one block per edit); every source is `git show <tip>:<path>` from the mirror.
Nothing is read from a working tree of the repository. Nothing is staged, committed or pushed.

Directives in the section files: a line `@@T KEY after=SUBSTR@@` is replaced by the first contiguous table (lines starting with `|`) after the first line of the KEY source file that contains SUBSTR, copied verbatim
except that a bare pipe inside a cell (a pipe without a space on both sides, as in `|change|`) is escaped so that the rendered row keeps its cell count.
Fill-ins: `{{TP}}` (tip, 8 hex), `{{LAND_x}}` (first commit that added the source file x), `{{RES_x}}` (sha256[:8] of source file x). Every `{{LLR_...}}` placeholder is LEFT IN PLACE: the Master Mind fills the LLR
section files (sec_739_llr_result.md, sec_739_tail.md) and edits_v39.md by hand after the LLR read, adds `raw/...llr_read/RESULTS.md` (the LLR key below) and re-runs this script. checks_v39.py FAILS while any `{{` remains.

python -I patch_v39.py tip [out_dir]
Every italic quotation of 25 characters or more is asserted verbatim in a source (RESULTS files, cards, decision item, register, v38)."""
import hashlib, json, os, re, subprocess, sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
assert _argv, "usage: patch_v39.py <tip> [out_dir]"
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


# ---- the v38 deliverables are the starting point ---------------------------------------------------------------------------------------------------------
V38_MD_SHA = "7deafaf103a48d8ae22c5146b0adb3f690cd8c487dffaebcbf9c6d0d4155366e"
V38_HTML_SHA = "5f6b1b1687bf19a6061c075efa87417379d6d22c3dab7e6109f68fc56b1510dc"
RP38 = R + "report/2026-11-04-v38/"
s = blob(RP38 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP38 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP38 + "V8F_SCIENTIFIC_REPORT.md")) == V38_MD_SHA, "the tip's report md is not the landed v38 deliverable"
assert sha(raw(RP38 + "v8f_report.html")) == V38_HTML_SHA, "the tip's report html is not the landed v38 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v38 html is not the template with the v38 md"
assert s.count("**Status:** research report, v38 of 2026-11-04") == 1
S0 = s

# ---- sources ----------------------------------------------------------------------------------------------------------------------------------------------
# key -> path under R (fill-ins {{RES_key}} and {{LAND_key}}); CLR_ is the CL result file, CLRR the CLR result file
RES_PATHS = OrderedDict([
    ("RPT", "raw/2026-11-05-recover-next-rpt/RESULTS.md"),
    ("A4", "raw/2026-11-05-recover-next-rpt/PREREG_RECOVER_NEXT_ADDENDUM4_RPT.md"),
    ("NHW", "raw/2026-11-05-recover-next-rpt/NOTE_HISTORY_WINDOW.md"),
    ("K2", "raw/2026-11-05-recover-rpt-k2-decision/DECISION_K2.md"),
    ("DC", "raw/2026-11-05-motion-c-ri/DESIGN_C.md"),
    ("RI", "raw/2026-11-05-motion-c-ri/ri_read/RESULT_RI_C0.md"),
    ("MP", "raw/2026-11-05-motion-c-probe/RESULT.md"),
    ("CLD", "raw/2026-11-05-motion-c-crawllead/RESULT.md"),
    ("CS", "raw/2026-11-05-motion-c-crawllead/C_STATUS.md"),
    ("CLP", "raw/2026-11-05-levelhead-cl-freeze/cl_prereg/PREREG_LEVELHEAD_CL.md"),
    ("CLR_", "raw/2026-11-05-levelhead-cl-result/cl_read/RESULTS.md"),
    ("A1P", "raw/2026-11-05-levelhead-cl-readout/PREREG_CL_READOUT_A1.md"),
    ("A1", "raw/2026-11-05-levelhead-cl-readout/RESULT.md"),
    ("A2P", "raw/2026-11-05-levelhead-a2-freeze/a2_prereg/PREREG_LEVELHEAD_A2.md"),
    ("CLRR", "raw/2026-11-05-levelhead-clr-result/clr_read/RESULTS.md"),
    ("WIR", "raw/2026-11-05-levelhead-clr-result/clr_read/WIRING.md"),
    ("AR1", "raw/2026-11-05-levelhead-llr-freeze/a2_prereg_additions/A2_RULING_1.md"),
    ("FL", "raw/2026-11-05-levelhead-llr-freeze/a2_prereg_additions/FROZEN_LLR.txt"),
])
# MM: the LLR result key. Verify the path against the tip after the LLR read lands (same layout as CL and CLR); the key is optional until a section file uses it.
LLR_PATH = "raw/2026-11-05-levelhead-llr-result/llr_read/RESULTS.md"
if gitrun(["cat-file", "-e", f"{TIP}:{R}{LLR_PATH}"], check=False) is not None:
    RES_PATHS["LLR"] = LLR_PATH
    RES_PATHS["LLRS"] = "raw/2026-11-05-levelhead-llr-result/llr_read/SENSITIVITY.md"
    RES_PATHS["A3P"] = "raw/2026-11-05-levelfloor-a3-prereg/PREREG_LEVELFLOOR_A3.md"
LAND_KEYS = ("RPT", "K2", "RI", "MP", "CLD", "CLR_", "CLRR") + (("LLR", "A3P") if "LLR" in RES_PATHS else ())
# table directives: KEY -> RES_PATHS key
DIRECTIVE_SRC = {"RPT": "RPT", "K2": "K2", "RI": "RI", "MP": "MP", "CLD": "CLD", "CLRES": "CLR_", "A1": "A1", "CLRR": "CLRR", "WIR": "WIR", "LLR": "LLR", "LLRS": "LLRS"}
# hard-coded sha256 prefixes in the section texts and edits, asserted against the tip's blobs
HARD = [("raw/2026-11-05-levelhead-cl-freeze/cl_prereg/PREREG_LEVELHEAD_CL.md", "875d8c19"), ("raw/2026-11-05-levelhead-cl-readout/PREREG_CL_READOUT_A1.md", "1f96d833"),
        ("raw/2026-11-05-levelhead-a2-freeze/a2_prereg/PREREG_LEVELHEAD_A2.md", "9a117b20"), ("raw/2026-11-05-motion-c-ri/ri_read/PREREG_RI_C0.md", "942b6391"),
        ("raw/2026-11-05-motion-c-crawllead/PREREG_CRAWLLEAD.md", "72f86313"), ("raw/2026-11-03-recover-next/PREREG_RECOVER_NEXT_FROZEN.md", "01046fe6")]
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
assert LAND["RPT"].startswith("6ef84e4") and LAND["K2"].startswith("d9f0941"), "the RPT RESULTS / K2 decision landing commits differ from the ones the handoff names"

# ---- raw JSON cross-checks of the headline numbers --------------------------------------------------------------------------------------------------------
J = json.loads(blob(R + "raw/2026-11-05-recover-next-rpt/RECOVER_READ_RPT.json"))
assert J["label"] == "HARMS" and J["controls_failed"] == [] and len(J["controls"]) == 20 and J["reader_sha256"].startswith("de5cfbf572ba") and J["mode"] == "production"
assert J["composite"]["status"] == "NOT DOMINANT" and J["guards"]["PASS"] is False and J["run_clock"].startswith("2026-11-05T08:40:17")
JS = {"share": {k: "%.4f" % J["primary"]["share_closed"][k] for k in ("s0", "s1")}, "ff": {k: "%.4f" % J["frame_freeze"]["share_closed"][k] for k in ("s0", "s1")},
      "ade": {k: "%+.4f" % J["guards"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1")},
      "floor": {k: ("%.4f" % J["floor"]["per_seed"][k]["abs_R_minus_C0"], "%.4f" % J["floor"]["per_seed"][k]["abs_C0rep_minus_C0"]) for k in ("s0", "s1")}}
assert "P-FAILS-M1-NOT-MOVED" in blob(R + "raw/2026-11-05-levelhead-cl-result/cl_read/raw/CL_READ.json") and "P-FAILS-ATTR-NOT-SEPARATED-BETTER" in blob(R + "raw/2026-11-05-levelhead-clr-result/clr_read/raw/CLR_READ.json")
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")

# ---- the section texts ------------------------------------------------------------------------------------------------------------------------------------
_blocks = re.split(r"^=== (\w+) ===\n", part("edits_v39.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
assert sorted(B) == ["ABS_LAST", "ABS_TITLE", "I17", "NEXT_LAST", "R83", "ROW", "S81", "ST"], sorted(B)


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
    left = [x for x in re.findall(r"\{\{[^{}\n]*\}\}", t) if not x.startswith("{{LLR_")]
    assert not left and "<<" not in t, left[:3]
    return t


SEC737 = fill(expand(part("sec_737_rpt.md"), "7.37"))
SEC738 = fill(expand(part("sec_738_motion_c.md"), "7.38"))
SEC739 = fill(expand("\n\n".join([part("sec_739_levelhead.md"), part("sec_739_llr_result.md"), part("sec_739_tail.md")]), "7.39"))
SEC = SEC737 + "\n\n" + SEC738 + "\n\n" + SEC739
B = {k: fill(v) for k, v in B.items()}
_heads = [m.group(1) for m in re.finditer(r"^(?:### |\*\*)(7\.3[789](?:\.\d)?) ", SEC, flags=re.M)]
_exp = ["7.37", "7.37.1", "7.37.2", "7.37.3", "7.37.4"] + ["7.38"] + ["7.38.%d" % i for i in range(1, 8)] + ["7.39"] + ["7.39.%d" % i for i in range(1, 6)]
_exp += ["7.39.7", "7.39.8"] if "{{LLR_RESULT_PENDING}}" in SEC else ["7.39.6", "7.39.7", "7.39.8"]
assert _heads == _exp, (_heads, _exp)
_nflat = re.sub(r"\s+", " ", SEC)
for k in ("s0", "s1"):
    assert JS["share"][k] in SEC737 and JS["ff"][k] in SEC737 and JS["ade"][k] in SEC737, ("an RPT headline number from the raw JSON is not in 7.37", k)
    assert JS["floor"][k][0] in SEC737 and JS["floor"][k][1] in SEC737, ("floor", k)

# ---- tables are source lines; quotations are verbatim -----------------------------------------------------------------------------------------------------
SRCL = set()
for k in RES_PATHS:
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
assert len(_tabs) == len(DIRS) and not _bad, ("a table line of the new sections is not a line of a source file", len(_tabs), len(DIRS), _bad[:3])
assert all(len({ncols(r) for r in t}) == 1 and _RULE.fullmatch(t[1]) for t in _tabs), "a table with ragged columns or no header rule"
assert not any(l.startswith("|") for k, v in B.items() if k != "ROW" for l in v.split("\n")), "a table line outside the 8.4 rows"
_ROWS = [l for l in B["ROW"].split("\n") if l.startswith("|")]
assert [l.split(" | ")[0] for l in _ROWS] == ["| 1au", "| 1av", "| 1aw", "| 1ax", "| 1ay"] and len({ncols(r) for r in _ROWS}) == 1 and ncols(_ROWS[0]) == 4, "8.4 rows 1au to 1ay"
_corp_src = [srctext(k) for k in RES_PATHS]
for p in ("raw/2026-11-05-motion-c-ri/ri_read/PREREG_RI_C0.md", "raw/2026-11-05-motion-c-probe/PREREG_MOTIONPROBE.md", "raw/2026-11-05-motion-c-crawllead/PREREG_CRAWLLEAD.md",
          "raw/2026-11-05-recover-next-rpt/FROZEN_SHA_ADDENDUM4.txt", "raw/2026-11-05-levelhead-llr-freeze/llr_launch/HANDOFF_LLR.md", "raw/2026-11-05-level-head-code/HANDOFF.md",
          "raw/2026-11-05-lead-head-code/HANDOFF.md", "raw/2026-11-05-recover-pertick-code/HANDOFF_PERTICK.md", "raw/2026-11-05-levelhead-cl-freeze/cl_prereg/FROZEN_W.txt"):
    _corp_src.append(blob(R + p))
_corp = re.sub(r"\s+", " ", " ".join(_corp_src + [REG, S0]))
_q = [m.strip('"') for m in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", ADDED)]
_nq = [q for q in _q if re.sub(r"\s+", " ", q) not in _corp]
assert _q and not _nq, ("an italic quotation is not verbatim in a source", _nq[:3])

# ---- anchored edits -----------------------------------------------------------------------------------------------------------------------------------
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
E = [
    ("research report, v38 of 2026-11-04, results through repository tip `ff09d2c1` and, within it,",
     "research report, v39 of 2026-11-05, results through repository tip `" + TP + "` and, within it,"),
    ("and RAPR-P01) (v37: `cd6ce52d`; v36: `9b79730c`;",
     "and RAPR-P01), the per-tick arm RPT and the PI's answer (§7.37), the option (c) measurements (§7.38) and the option (a) level-head arms CL and CLR with the lead head LLR (§7.39) "
     "(v38: `ff09d2c1`; v37: `cd6ce52d`; v36: `9b79730c`;"),
    ("(§7.36.6; ESTIMATED closure need 0.68 / 0.60 of the excess, post hoc).\n",
     "(§7.36.6; ESTIMATED closure need 0.68 / 0.60 of the excess, post hoc). " + B["ST"] + "\n"),
    ("keeps the along-track and distance-keeping harm (§7.36; five arms trained on box 2 and read on 2026-11-03 and 2026-11-04,", B["ABS_TITLE"]),
    ("Addendum 3 row K closes the one-offset-per-window recipe at the laws and doses measured, and the next arm is a PI decision between a level-from-image auxiliary and a per-tick deviation recipe; "
     "neither starts without the PI's decision and box-2 go. **[E]** A post hoc line fit over the five arms puts the closure that would stop the composite failing D2 on M1 near 0.68 / 0.60 of the excess, "
     "against 0.4447 / 0.3175 measured at most (§7.36.6); the Master Mind recommends the per-tick recipe (form b1).", B["ABS_LAST"]),
    ("It moves M1 the right way in all five arms and clears neither the 0.5 bar nor its guards, §7.35.4 adds that no single per-window speed closes M1, and after §7.36.4a the next arm is a PI decision (§7.36.6).",
     B["NEXT_LAST"]),
    ("This section starts nothing: implementation of (b) begins on the PI's choice, and the box-2 runs need a separate go.",
     "This section starts nothing: implementation of (b) begins on the PI's choice, and the box-2 runs need a separate go. "
     "The per-tick arm was run as RPT (§7.37), and the PI's later answer, (c) then (a), is read in §7.38 and §7.39."),
    ("\n\n### 8.2 What is not established", "\n" + B["S81"] + "\n\n### 8.2 What is not established"),
    ("**Update (v38).** One item is written up here", B["R83"] + "\n\n**Update (v38).** One item is written up here"),
    ("no box-2 job starts without the PI's go |", "no box-2 job starts without the PI's go; the PI answered on 2026-11-05 (rows 1au to 1ay) |"),
    ("| 2 | **LoRA on DINOv3**", B["ROW"] + "\n| 2 | **LoRA on DINOv3**"),
    ("\n\n---\n\n## 10. Reproducibility appendix", "\n" + B["I17"] + "\n\n---\n\n## 10. Reproducibility appendix"),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
# the new sections go after the last subsection of 7.36 (7.36.7 Provenance), before the Discussion
assert s.count(ANCHOR_8) == 1 and s[:s.index(ANCHOR_8)].split("\n")[-1].startswith("**7.36.7 Provenance [M].**")
_j = s.index(ANCHOR_8)
s = s[:_j] + "\n\n" + SEC + s[_j:]

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "".join(b for _, b in E) + SEC
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower()
assert re.search("retract|supersed|" + BANNED[6] + "|was wrong|previously read|outdated|i said", low) is None
assert not _CL.search(ins) and not re.search(r"\bdrives\b|driving performance", low), "forbidden T2 wording or driving wording"
assert "\r" not in ins and "$" not in ins, "a CR or a dollar sign in the added text (the math count of the report stays as in v38)"
assert s.count("$$") == S0.count("$$") and s.count("$") == S0.count("$"), "math spans changed"
nocode = re.sub(r"`[^`\n]*`", " ", ins)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r".{20}<[A-Za-z/!].{20}", nocode)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
    assert fam in SEC737 and fam in SEC739, fam
assert "B = 2,000" in _nflat and "T1-kin" in _nflat and "H-ESTIM-SEED-1" in _nflat and "reindexed" in _nflat and "frame-freeze" in _nflat.lower()
assert s.index("**7.36.7 ") < s.index("### 7.37 ") < s.index("### 7.38 ") < s.index("### 7.39 ") < s.index("## 8. Discussion") and all(s.count(x) == 1 for x in ("### 7.37 ", "### 7.38 ", "### 7.39 "))
assert s.count("**Status:** research report, v39 of 2026-11-05") == 1 and "v38 of 2026-11-04, results" not in s.split("\n")[2]
assert not [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if d > "2026-11-05"], "a date after 2026-11-05"
assert "no box-2 job starts without the PI's go |" not in s, "a stale pointer about the row 1at PI go remains"
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
_pend = sorted(set(re.findall(r"\{\{[^{}\n]*\}\}", s)))
print("v39 built", len(S0), len(s), "tip", TIP, "7.37 to 7.39 chars", len(SEC), "tables", len(_tabs), "table lines", sum(len(t) for t in _tabs), "quotations", len(_q), "landed", LAND, "placeholders left", _pend)
open(OUT + "DIRECTIVES.txt", "w", encoding="utf-8", newline="\n").write("\n".join("%s | after=%s | source line %d | table at %d | %d lines | %d rows with an escaped pipe" % d for d in DIRS) + "\n")
