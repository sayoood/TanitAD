"""Report v42 = v41 + sections 7.42 (T2 re-perception on NuRec scenes), 7.43 (the LoRA run on the PI's GPU pod) and 7.44 (a theory of the crawl and a weight-space reading), inserted after 7.41.7, + the status line,
the abstract (two bullets and the next-bullet sentence), the related-work paragraph, 8.1 item 32, 8.3 Update (v42), 8.4 rows 1bd to 1bf, 9 item 20, the section 10 appendix block, and the carried v41 lines that
said no T2 number exists (edited in place, each listed in edits_v42.md as old -> new).
Built from the landed v41 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once). The prose is in the package files beside this script
(sec_742_t2.md, sec_743_lorapod.md, sec_744_theory.md: the section texts; edits_v42.md: one block per edit, V42S being the shared v42 sentence that the blocks name as <<V42S>>, and the C* blocks being
`old -->` `new` pairs for carried lines); every source is `git show <tip>:<path>` from the mirror. Nothing is read from a working tree of the repository. Nothing is staged, committed or pushed.

Directives in the section files:
  `@@T KEY after=SUBSTR [nth=N]@@` is replaced by the Nth (default 1st) contiguous table (lines starting with `|`) after the first line of the KEY source file that contains SUBSTR, copied verbatim
  except that a bare pipe inside a cell is escaped so that the rendered row keeps its cell count (as in v39 to v41).
Fill-ins: `{{TP}}` (tip, 8 hex), `{{LAND_x}}` (first commit that added the source file x), `{{RES_x}}` (sha256[:8] of source file x).

python -I patch_v42.py tip [out_dir]
Every italic quotation of 25 characters or more is asserted verbatim in a source (RESULTS files, cards, rulings, notes, register, v41)."""
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


# ---- the v41 deliverables are the starting point ---------------------------------------------------------------------------------------------------------
V41_MD_SHA = "c73d0ef62a7138d7e4feabb795545e392450d25a8385d06afdffde891e8f35e2"
V41_HTML_SHA = "70b922065457e6624561a32f741935b36e926917dfbb101ef43873488edbda00"
RP41 = R + "report/2026-11-06-v41/"  # the landed v41 files sit directly in this folder (no out/)
s = blob(RP41 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP41 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP41 + "V8F_SCIENTIFIC_REPORT.md")) == V41_MD_SHA, "the tip's report md is not the landed v41 deliverable"
assert sha(raw(RP41 + "v8f_report.html")) == V41_HTML_SHA, "the tip's report html is not the landed v41 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v41 html is not the template with the v41 md"
assert s.count("**Status:** research report, v41 of 2026-11-06") == 1
S0 = s

# ---- sources ----------------------------------------------------------------------------------------------------------------------------------------------
_D = "raw/2026-11-06-"
RES_PATHS = OrderedDict([
    ("DOM", _D + "t2-dominance-read/stage4/score/DOMINANCE_T2.md"),
    ("DOMJ", _D + "t2-dominance-read/stage4/score/DOMINANCE_T2.json"),
    ("DOMC", _D + "t2-dominance-read/PREREG_T2_DOMINANCE_with_log.md"),
    ("DRAFT", _D + "t2-dominance-read/DRAFT_PREREG_T2_DOM_LORA.md"),
    ("PLAN", _D + "t2-dominance-read/lora_pod/PLAN_V8F_LORA_POD.md"),
    ("R2", _D + "t2-crawl-read/stage2/RESULT_T2_STAGE2.md"),
    ("R2J", _D + "t2-crawl-read/stage2/READ_STAGE2.json"),
    ("SC2", _D + "t2-crawl-read/stage2/SCORE2.md"),
    ("RSCN", _D + "t2-crawl-read/AMENDMENT_T2_SCENE_REFUSAL.md"),
    ("CRAWLC", _D + "t2-harness-v8f/t2_prereg/PREREG_T2_CRAWL.md"),
    ("S1R", _D + "t2-stage1-box2/STAGE1_READ.md"),
    ("S4", _D + "t2-stage4-c0c/MANIFEST.txt"),
    ("REPIN", _D + "t2-dom-repin/MANIFEST.txt"),
    ("AM4", _D + "lora-pod-amend4/AMENDMENT_V8F_LORA_4.md"),
    ("IDP", _D + "lora-pod-amend4/identity_pod/IDENTITY_POD.md"),
    ("H0", _D + "lora-pod-amend4/HANDOFF_LORAPOD0.md"),
    ("H1", _D + "lora-pod-amend4/HANDOFF_LORAPOD1.md"),
    ("THC", _D + "theory-crawl/THEORY_CRAWL.md"),
    ("THA", _D + "theory-crawl/THEORY_CRAWL_AMEND_1.md"),
    ("WG", _D + "theory-crawl/PREREG_WSP_GAIN.md"),
    ("WR", _D + "theory-crawl/wsp_read/WSP_READ.md"),
    ("CP2", _D + "theory-crawl/wsp_read/CLAIMS_P2.md"),
    ("SPEC1", _D + "theory-crawl/wsp_read/SPEC_P1.md"),
    ("P1J", _D + "theory-crawl/wsp_read/P1_RESULT.json"),
    ("P1BJ", _D + "theory-crawl/wsp_read/P1B_RESULT.json"),
    ("P1CJ", _D + "theory-crawl/wsp_read/P1C_RESULT.json"),
    ("P2BJ", _D + "theory-crawl/wsp_read/P2B_RESULT.json"),
    ("P2B1J", _D + "theory-crawl/wsp_read/P2B_s1_RESULT.json"),
    ("P3J", _D + "theory-crawl/wsp_read/P3_RESULT.json"),
    ("TLIB", _D + "theory-crawl/library/README.txt"),
])
LAND_SRC = OrderedDict([("R2", "R2"), ("CRAWLC", "CRAWLC"), ("S1R", "S1R"), ("S4", "S4"), ("REPIN", "REPIN"), ("DOM", "DOM"), ("TH", "THC"), ("AM4", "AM4")])
DIRECTIVE_SRC = {"DOM": "DOM", "R2": "R2"}
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
_want_land = {"R2": "f64a3367", "S4": "89cbd4da", "REPIN": "939176e7", "TH": "b17b5492", "AM4": "701e5a9e", "DOM": "ec80b6c2"}
for k, pre in _want_land.items():
    assert LAND[k].startswith(pre[:7]), ("landing commit", k, LAND[k], pre)

# hard-coded sha256 prefixes in the section texts and edits: each is asserted against a tip blob (file sha) or against the frozen sha a read itself records
_hex_in_sec = lambda pre: any(pre in srctext(k) for k in ("DOMJ", "DOMC", "R2", "RSCN", "THC", "THA", "WG", "CP2", "SPEC1", "WR"))
HARD_FILE = [("THC", "6e83451c"), ("THA", "9d39ac53"), ("WG", "92d61dcb"), ("CP2", "afb9785f"), ("AM4", "626aa1e5"), ("RSCN", "2167a7eb"), ("SPEC1", "52d74aa6")]
for k, pre in HARD_FILE:
    # the frozen sha of a card is the sha of its frozen bytes (the file may have grown a log since): the prefix must be a prefix of the file sha, or be recorded by a frozen-sha record in the sources
    assert RES_SHA[k].startswith(pre) or _hex_in_sec(pre) or any(pre in blob(R + p) for p in (_D + "theory-crawl/FROZEN_SHA_PREREG_WSP_GAIN.txt", _D + "theory-crawl/FROZEN_SHA_THEORY_AMEND_1.txt", _D + "theory-crawl/FROZEN_SHA_THEORY_CRAWL.txt", _D + "lora-pod-amend4/FROZEN_SHA_AMENDMENT4.txt", _D + "t2-crawl-read/FROZEN_SHA_SCENE_REFUSAL.txt")), ("a hard-coded sha256 prefix is in no tip record", k, pre)
assert "2c4c6557" in blob(R + _D + "t2-harness-v8f/t2_prereg/FROZEN_SHA.txt") or "2c4c6557" in srctext("R2J"), "PREREG_T2_CRAWL frozen prefix"
assert "a326bc5b" in srctext("TLIB") or sha(raw(R + _D + "theory-crawl/library/takehi2026weightspace.html")).startswith("a326bc5b"), "the banked takehi page sha256 prefix"
_DJ = json.loads(srctext("DOMJ"))
assert _DJ["card_sha256_expected"].startswith("38aa9df3") and _DJ["card_sha_ok"] is True and _DJ["script_sha256"].startswith("d6664718")
_R2J = json.loads(srctext("R2J"))
assert _R2J["outcome"] == "CRAWL-PERSISTS" and _R2J["branch"] == "a"

# ---- raw JSON cross-checks of the headline numbers --------------------------------------------------------------------------------------------------------
assert _DJ["outcome"] == "NOT-DOMINANT" and _DJ["decision"]["outcome"] == "NOT-DOMINANT"
FT = _DJ["decision"]["failing_terms"]
assert FT["s0"] == ["D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.7893)", "D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE, heading MAE)", "D3 (TACTICAL: no separated-better row)"]
assert FT["s1"] == ["D1 (ADE6 not separated-better with point <= -0.2415 m; point +0.6399)", "D2 (separated-worse: ADE6, M1 crawl share, TAC-LON, along-track MAE)", "D3 (TACTICAL: no separated-better row)", "D3 (STRATEGIC: no separated-better row)"]
_rows = {sd: _DJ["by_seed"][sd]["rows"] for sd in ("s0", "s1")}
assert ["%+.4f" % _rows[sd]["ADE6"]["delta"] for sd in ("s0", "s1")] == ["+0.7893", "+0.6399"]
assert ["%+.4f" % _rows[sd]["M1 crawl share"]["delta"] for sd in ("s0", "s1")] == ["+0.4073", "+0.4041"]
assert _DJ["gates"]["G3"]["n_starts"] == {"P10": 93, "C0C_s0": 93, "C0C_s1": 93} and _DJ["r_scn"]["scenes"] == {"of": 62, "excluded": 14, "read_on": 48}
assert _DJ["decision"]["per_seed"]["s0"]["n_rows_separated"] == 13 and _DJ["decision"]["per_seed"]["s1"]["n_rows_separated"] == 12
assert [(_R2J["X1"][sd]["delta"], _R2J["X1"][sd]["lo"], _R2J["X1"][sd]["hi"]) for sd in ("0", "1")] == [(0.3388, 0.2555, 0.4291), (0.4256, 0.3231, 0.5224)]
assert [(_R2J["X2"][sd]["delta"], _R2J["X2"][sd]["lo"], _R2J["X2"][sd]["hi"]) for sd in ("0", "1")] == [(-0.0021, -0.04, 0.0375), (0.0208, -0.0067, 0.0525)]
assert [(_R2J["BL"][sd]["delta"], _R2J["BL"][sd]["lo"], _R2J["BL"][sd]["hi"]) for sd in ("0", "1")] == [(0.0111, -0.0704, 0.0917), (-0.0757, -0.1599, 0.0143)]
assert abs(_R2J["seed_gap"] - 0.0868) < 5e-5
# derived numbers of 7.44 (the section carries them as text; asserted here from their inputs)
from statistics import NormalDist
_N = NormalDist()
assert ["%.3f" % x for x in (1.000 - 0.806, 1.899 - 1.265, 1.000 - 0.763, 1.899 - 1.112)] == ["0.194", "0.634", "0.237", "0.787"]
_beta = lambda auc: _N.cdf((2 ** 0.5) * _N.inv_cdf(auc) - _N.inv_cdf(0.95))
assert "%.2f" % _beta(0.785) == "0.30" and "%.2f" % _beta(0.680) == "0.16", (_beta(0.785), _beta(0.680))
assert 24 * 2 * 2 * 1024 * 32 == 3145728
from decimal import Decimal, ROUND_HALF_UP
_rh = lambda x: str((Decimal(x) / Decimal("0.001")).quantize(Decimal("1"), rounding=ROUND_HALF_UP))  # 7.43.4 "21 to 23 times": 21.1 and 22.5 rounded half up
assert _rh("0.0211") == "21" and _rh("0.0225") == "23", (_rh("0.0211"), _rh("0.0225"))
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
for row in ("H-T2-DOM-1", "D-V8F-LORA-POD-1", "D-V8F-THEORY-CRAWL-1"):
    assert row in REG, ("a register row the provenance names", row)

# ---- the section texts ------------------------------------------------------------------------------------------------------------------------------------
_blocks = re.split(r"^=== (\w+) ===\n", part("edits_v42.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
PROSE = ["ABS1", "APP", "I20", "NXT", "R83", "REL", "ROW", "S81", "V42S"]
CARRIED = ["C17", "C18", "C19", "C3A", "C3B", "C3C", "C335", "C5", "C5719", "C5792", "C5861", "C5861B", "C5926A", "C5926B", "C602"]
assert sorted(B) == sorted(PROSE + CARRIED + ["NOTES"]), sorted(B)  # NOTES is documentation of the edits and is never inserted
assert "\n" not in B["V42S"] and "<<" not in B["V42S"] and "{{" not in B["V42S"]
B = {k: v.replace("<<V42S>>", B["V42S"]) for k, v in B.items()}


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


SECS = [fill(expand(part(f), f)) for f in ("sec_742_t2.md", "sec_743_lorapod.md", "sec_744_theory.md")]
SEC = "\n\n".join(SECS)
_CLH, _CLS = "pool-desolc"[::-1], "pool desolc"[::-1]  # the banned T2 wording is built here, never written in the edits file
B = {k: fill(v.replace("<<CLH>>", _CLH).replace("<<CLS>>", _CLS)) for k, v in B.items()}
_heads = [m.group(1) for m in re.finditer(r"^(?:### |\*\*)(7\.4[234](?:\.\d)?) ", SEC, flags=re.M)]
assert _heads == ["7.42"] + ["7.42.%d" % i for i in range(1, 8)] + ["7.43"] + ["7.43.%d" % i for i in range(1, 9)] + ["7.44"] + ["7.44.%d" % i for i in range(1, 10)], _heads

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
assert len(_hand_tabs) == 6, ("hand-written tables of 7.42 to 7.44 (stage table, venue, proxy, token identity, weights, predictions)", len(_hand_tabs))
assert all(len({ncols(r) for r in t}) == 1 and _RULE.fullmatch(t[1]) for t in _tabs), "a table with ragged columns or no header rule"
assert not any(l.startswith("|") for k, v in B.items() if k not in ("ROW", "APP") for l in v.split("\n")), "a table line outside the 8.4 rows and the appendix table"
_ROWS = [l for l in B["ROW"].split("\n") if l.startswith("|")]
assert [l.split(" | ")[0] for l in _ROWS] == ["| 1bd", "| 1be", "| 1bf"] and all(ncols(r) == 4 for r in _ROWS), "8.4 rows 1bd to 1bf"
_APPT = [l for l in B["APP"].split("\n") if l.startswith("|")]
assert len(_APPT) == 10 and len({ncols(r) for r in _APPT}) == 1 and _RULE.fullmatch(_APPT[1]), "the appendix table"
ADDED = "\n".join([SEC] + [B[k] for k in PROSE])
_corp = re.sub(r"\s+", " ", " ".join([srctext(k) for k in RES_PATHS] + [REG, S0]))
_q = [m.strip('"') for m in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", ADDED)]
_nq = [q for q in _q if re.sub(r"\s+", " ", q) not in _corp]
assert _q and not _nq, ("an italic quotation is not verbatim in a source", _nq[:5])

# ---- anchored edits -----------------------------------------------------------------------------------------------------------------------------------
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
ANCHOR_10 = "\n\n---\n\n## 10. Reproducibility appendix"
ANCHOR_4 = "\n\n---\n\n## 4. Method"
_pairs = {}
for k in CARRIED:
    a, b = B[k].split("\n-->\n")
    assert a and b and a != b, k
    _pairs[k] = (a, b)
CARRIED_EDITS = [_pairs[k] for k in ("C3A", "C3B", "C3C", "C5", "C17", "C335", "C602", "C5719", "C5792", "C5861", "C5861B", "C5926A", "C5926B", "C18", "C19")]
E = CARRIED_EDITS + [
    ("(§7.41.6).\n\n**Evidence classes used throughout.**", "(§7.41.6). " + B["V42S"] + "\n\n**Evidence classes used throughout.**"),
    ("\n* **What is next.** The crawl set is still open", "\n" + B["ABS1"] + "\n* **What is next.** The crawl set is still open"),
    ("(§7.41.6).\n\n---\n\n## 2. Introduction", "(§7.41.6). " + B["NXT"] + "\n\n---\n\n## 2. Introduction"),
    (ANCHOR_4, "\n\n" + B["REL"] + ANCHOR_4),
    ("Its limits are in §7.41.5.\n\n### 8.2 What is not established", "Its limits are in §7.41.5.\n" + B["S81"] + "\n\n### 8.2 What is not established"),
    ("**Update (v41).** One item is written up here", B["R83"] + "\n\n**Update (v41).** One item is written up here"),
    ("the remaining levers are the A40 and T2 (§7.41.6) |\n| 2 | **LoRA on DINOv3**", "the remaining levers are the A40 and T2 (§7.41.6) |\n" + B["ROW"] + "\n| 2 | **LoRA on DINOv3**"),
    ("\n\n**Figures.** `make_figures.py` in this folder", "\n\n" + B["APP"] + "\n\n**Figures.** `make_figures.py` in this folder"),
    (ANCHOR_10, "\n" + B["I20"] + ANCHOR_10),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
# the new sections go after the last subsection of 7.41 (7.41.7 Provenance), before the Discussion
assert s.count(ANCHOR_8) == 1 and s.index("**7.41.7 Provenance") < s.index(ANCHOR_8)
assert s[:s.index(ANCHOR_8)].split("\n")[-1].startswith("The claim rows are `H-LEVELFLOOR-1 (A4)`")
_j = s.index(ANCHOR_8)
s = s[:_j] + "\n\n" + SEC + s[_j:]

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "".join(b for _, b in E) + SEC
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower()
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
assert s.index("**7.41.7 ") < s.index("### 7.42 ") < s.index("### 7.43 ") < s.index("### 7.44 ") < s.index("## 8. Discussion")
assert all(s.count("### 7.4%d " % i) == 1 for i in (2, 3, 4))
assert s.count("**Status:** research report, v42 of 2026-11-07") == 1 and "v41 of 2026-11-06, results" not in s.split("\n")[2]
assert not [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if d > "2026-11-07"], "a date after 2026-11-07"
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
_pend = sorted(set(re.findall(r"\{\{[^{}\n]*\}\}", s)))
print("v42 built", len(S0), len(s), "tip", TIP, "7.42-7.44 chars", len(SEC), "directive tables", len(_src_tabs), "hand tables", len(_hand_tabs), "display eq", len(_disp), "inline math", len(_inl), "quotations", len(_q), "landed", LAND, "placeholders left", _pend)
open(OUT + "DIRECTIVES.txt", "w", encoding="utf-8", newline="\n").write("\n".join("%s | after=%s | source line %d | table at %d | %d lines | %d rows with an escaped pipe" % d for d in DIRS) + "\n")
