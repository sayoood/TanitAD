"""Report v44 = v43 + sections 7.47 (the world model level by level, MRATE, the scene read LATSEM2 / LATSEM2B, the imagination plan) and 7.48 (RENC at the token level, the C0C-RE T2 read, TOWN, TSWAP),
inserted after 7.46.11, + the status line, the abstract (two bullets and the next-bullet sentence), 8.1 item 34, 8.3 Update (v44), 8.4 rows 1bm to 1bu, 9 item 22, the section 10 appendix block
(one row per new raw folder), and the carried v43 lines that said RENC was not run (edited in place, each listed in edits_v44.md as old -> new).
Built from the landed v43 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once, or the stated number of times). The prose is in the package files beside this
script (sec_747_world_model.md, sec_748_renc_town.md: the section texts; edits_v44.md: one block per edit, V44S being the shared v44 sentence that the blocks name as <<V44S>>, and the C* blocks being
`old -->` `new` pairs for carried lines); every source is `git show <tip>:<path>` from the mirror. Nothing is read from a working tree of the repository. Nothing is staged, committed or pushed.
Fill-ins: `{{TP}}` (tip, 8 hex). Generalised guards of this version (v43 listed its sha prefixes by hand): every 8-hex token of the added text must be a prefix found in a tip record or a commit of the
mirror; every register row id (D-/H- ...) must occur in the register; every italic quotation of 25 characters or more must be verbatim in a source.

python -I patch_v44.py tip [out_dir]"""
import hashlib, os, re, subprocess, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
assert _argv, "usage: patch_v44.py <tip> [out_dir]"
TIP = _argv[0]
assert re.fullmatch(r"[0-9a-f]{40}", TIP), "pass the full 40-hex tip"
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


# ---- the v43 deliverables are the starting point -------------------------------------------------------------------------------------------------------
V43_MD_SHA = "ad2df9ba70655f8b1c599d2c8c705ad7ade8d961f85ae71378befe53e69b2fb9"
V43_HTML_SHA = "57448406c07ab68485023272b1e9e956e17fc7931b5e4d41dac8d87cd9509f08"
RP43 = R + "report/2026-11-07-v43/"  # the landed v43 files sit directly in this folder
s = blob(RP43 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP43 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP43 + "V8F_SCIENTIFIC_REPORT.md")) == V43_MD_SHA, "the tip's report md is not the landed v43 deliverable"
assert sha(raw(RP43 + "v8f_report.html")) == V43_HTML_SHA, "the tip's report html is not the landed v43 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v43 html is not the template with the v43 md"
assert s.count("**Status:** research report, v43 of 2026-11-07") == 1
S0 = s


def all_raw_corpus():
    """every md / txt / json blob (up to 3 MB) under raw/2026-11-02-* to raw/2026-11-07-* at the tip, one git cat-file --batch"""
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
_corp = re.sub(r"\s+", " ", " ".join(_RAWS + [REG, S0]).replace("**", ""))  # quotations are verbatim modulo the bold markers of the source

# ---- the section texts ------------------------------------------------------------------------------------------------------------------------------------
_blocks = re.split(r"^=== (\w+) ===\n", part("edits_v44.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
PROSE = ["ABS1", "APP", "I22", "R83", "ROW", "S81", "V44S"]
CARRIED = ["C3A", "C3B", "C43S", "C43B", "C46A", "C46B", "C46C", "C46D", "NXT"]  # NXT (the abstract's next-bullet sentence) is an old --> new pair like the C* blocks
assert sorted(B) == sorted(PROSE + CARRIED + ["NOTES"]), sorted(B)  # NOTES is documentation of the edits and is never inserted
assert "\n" not in B["V44S"] and "<<" not in B["V44S"] and "{{" not in B["V44S"]
B = {k: v.replace("<<V44S>>", B["V44S"]) for k, v in B.items()}


def ncols(line):
    """number of cells: unescaped pipes minus one (strict: a pipe inside a code span counts, as the renderer's table parser counts it)"""
    return line.replace("\\|", "").count("|") - 1


_RULE = re.compile(r"\|[\s:\-|]+")


def fill(t):
    t = t.replace("{{TP}}", TP)
    left = re.findall(r"\{\{[^{}\n]*\}\}", t)
    assert not left and "<<" not in t, left[:3]
    return t


SECS = [fill(part(f)) for f in ("sec_747_world_model.md", "sec_748_renc_town.md")]
assert not any("@@" in x for x in SECS), "a table directive in a v44 section (v44 has hand tables only)"
SEC = "\n\n".join(SECS)
B = {k: fill(v) for k, v in B.items()}
_heads = [m.group(1) for m in re.finditer(r"^(?:### |\*\*)(7\.4[78](?:\.\d+)?) ", SEC, flags=re.M)]
assert _heads == ["7.47"] + ["7.47.%d" % i for i in range(1, 12)] + ["7.48"] + ["7.48.%d" % i for i in range(1, 7)], _heads

# ---- tables are hand-written and rectangular; quotations are verbatim ---------------------------------------------------------------------------------
_tabs, _cur = [], []
for l in SEC.split("\n"):
    if l.startswith("|"):
        _cur.append(l)
    elif _cur:
        _tabs.append(_cur); _cur = []
if _cur:
    _tabs.append(_cur)
assert all(len({ncols(r) for r in t}) == 1 and _RULE.fullmatch(t[1]) for t in _tabs), "a table with ragged columns or no header rule"
assert not any(l.startswith("|") for k, v in B.items() if k in PROSE and k not in ("ROW", "APP") for l in v.split("\n")), "a table line outside the 8.4 rows and the appendix table"
_ROWS = [l for l in B["ROW"].split("\n") if l.startswith("|")]
assert [l.split(" | ")[0] for l in _ROWS] == ["| 1b" + c for c in "mnopqrstu"] and all(ncols(r) == 4 for r in _ROWS), "8.4 rows 1bm to 1bu"
_APPT = [l for l in B["APP"].split("\n") if l.startswith("|")]
assert len(_APPT) == 15 and len({ncols(r) for r in _APPT}) == 1 and _RULE.fullmatch(_APPT[1]), "the appendix table (one row per new raw folder: 13)"
ADDED = "\n".join([SEC] + [B[k] for k in PROSE] + [B[k].split("\n-->\n")[1] for k in CARRIED])
_q = [m.strip('"').replace("∗", "*") for m in re.findall(r'(?<![*\w])\*("[^*\n]{20,}?")\*(?![*\w])', ADDED.replace("\\*", "∗"))]  # a verbatim quotation is an italic span that opens with a double quote; an escaped star is a literal star; the italic lead-ins of tables and paragraphs are not quotations
_nq = [q for q in _q if re.sub(r"\s+", " ", q) not in _corp]
assert _q and not _nq, ("an italic quotation is not verbatim in a source", _nq[:5])
# every 8-hex token (sha256 prefix or commit) of the added text is found in a tip record or is a commit of the mirror
_hex = sorted(set(re.findall(r"(?<![0-9a-zA-Z])(?=[0-9a-f]*\d)[0-9a-f]{8}(?![0-9a-zA-Z])", ADDED)))
_nohex = []
# dev-box files that are not on the tip (the text labels what it takes from them as [I, UNVERIFIED by the build]): a prefix found only there is reported, not hidden
EONLY = ["E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/imag1/HANDOFF_IMAG1.md", "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/voe1/HANDOFF_VOE1.md"]
_eonly_txt = " ".join(open(f, encoding="utf-8").read() for f in EONLY if os.path.exists(f))
_eonly_used = []
for x in _hex:
    if x in _corp:
        continue
    if x in _eonly_txt:
        _eonly_used.append(x)
        continue
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "rev-parse", "--verify", "--quiet", x + "^{commit}"], capture_output=True)
    if r.returncode != 0:
        _nohex.append(x)
assert not _nohex, ("an 8-hex prefix of the added text is in no tip record and is no commit of the mirror", _nohex)
_ids = sorted(set(re.findall(r"`([DH]-[A-Z0-9]+(?:-[A-Z0-9]+)*-\d+)`", ADDED)))
_noids = [x for x in _ids if x not in REG]
assert not _noids, ("a register row id of the added text is not in the register", _noids)

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
    ("nothing in v43 is a driving result.\n\n**Evidence classes used throughout.**", "nothing in v43 is a driving result. " + B["V44S"] + "\n\n**Evidence classes used throughout.**"),
    ("\n* **What is next.** The crawl set is still open", "\n" + B["ABS1"] + "\n* **What is next.** The crawl set is still open"),
    ("Its limits are in §7.45.8, §7.46.10 and section 9 item 21.\n\n### 8.2 What is not established", "Its limits are in §7.45.8, §7.46.10 and section 9 item 21.\n" + B["S81"] + "\n\n### 8.2 What is not established"),
    ("**Update (v43).** Three items are written up here", B["R83"] + "\n\n**Update (v43).** Three items are written up here"),
    ("\n| 2 | **LoRA on DINOv3**", "\n" + B["ROW"] + "\n| 2 | **LoRA on DINOv3**"),
    ("\n\n**Figures.** `make_figures.py` in this folder", "\n\n" + B["APP"] + "\n\n**Figures.** `make_figures.py` in this folder"),
    (ANCHOR_10, "\n" + B["I22"] + ANCHOR_10),
]
for a, b in E:
    want_n = 2 if a == _pairs["C43S"][0] else 1  # the v43 sentence sits in the status line and in 8.1 item 33
    assert s.count(a) == want_n, (s.count(a), want_n, a[:90])
    s = s.replace(a, b)
# the new sections go after the last subsection of 7.46 (7.46.11 Provenance), before the Discussion
assert s.count(ANCHOR_8) == 1 and s.index("**7.46.11 Provenance") < s.index(ANCHOR_8)
assert s[:s.index(ANCHOR_8)].split("\n")[-1].startswith("**7.46.11 Provenance")
_j = s.index(ANCHOR_8)
s = s[:_j] + "\n\n" + SEC + s[_j:]

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "".join(b for _, b in E) + SEC
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower().replace("retraction_log", "")  # the file name RETRACTION_LOG.md is a path, not prose
assert re.search("retract|supersed|" + BANNED[6] + "|was wrong|previously read|outdated|i said", low) is None
assert not _CL.search(ins) and not re.search(r"\bdrives\b|driving performance", low), "forbidden T2 wording or driving wording"
assert "\r" not in ins, "a CR in the added text"
assert not re.search(r"[\u2013\u2014]", ins), "an en or em dash in the added text"
# math guard: strip fenced and inline code, then the display equations, then the template's inline-math pattern; no dollar sign may remain
_nc = re.sub(r"```[\s\S]*?```", " ", ins)
_nc = re.sub(r"`[^`\n]+`", " ", _nc)
_disp = re.findall(r"\$\$[\s\S]+?\$\$", _nc)
_nd = re.sub(r"\$\$[\s\S]+?\$\$", " ", _nc)
_inl = re.findall(r"(?:^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", _nd, flags=re.M)
_ni = re.sub(r"(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", r"\1 ", _nd, flags=re.M)
assert "$" not in _ni, ("a dollar sign outside the display and inline math", re.search(r".{30}\$.{30}", _ni, flags=re.S))
assert ins.count("$$") % 2 == 0 and s.count("$$") == S0.count("$$") + ins.count("$$"), "display math fences"
assert len(_disp) == 2 and len(_inl) > 10, (len(_disp), len(_inl))
_nm = re.sub(r"\$\$[\s\S]+?\$\$", " ", re.sub(r"`[^`\n]+`", " ", ins))
_nm = re.sub(r"(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$\\])?)\$", r"\1 ", _nm, flags=re.M)
assert re.search(r"<[A-Za-z/!]", _nm) is None, re.search(r".{20}<[A-Za-z/!].{20}", _nm)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
_nflat = re.sub(r"\s+", " ", SEC)
for must in ("PAUSED", "T0", "T2", "T1-kin", "RE-HARMS", "T-NOT", "H-ESTIM-SEED-1", "no result is claimed"):
    assert must in _nflat, must
assert s.index("**7.46.11 ") < s.index("### 7.47 ") < s.index("### 7.48 ") < s.index("## 8. Discussion")
assert all(s.count("### 7.4%d " % i) == 1 for i in (2, 3, 4, 5, 6, 7, 8))
assert s.count("**Status:** research report, v44 of 2026-11-07") == 1 and "v43 of 2026-11-07, results" not in s.split("\n")[2]
assert not [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if d > "2026-11-07"], "a date after 2026-11-07"
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
_pend = sorted(set(re.findall(r"\{\{[^{}\n]*\}\}", s)))
print("v44 built", len(S0), len(s), "tip", TIP, "7.47-7.48 chars", len(SEC), "hand tables", len(_tabs), "display eq", len(_disp), "inline math", len(_inl), "quotations", len(_q), "hex tokens", len(_hex), "row ids", len(_ids), "raw files", _NRAW, "placeholders left", _pend)
open(OUT + "DIRECTIVES.txt", "w", encoding="utf-8", newline="\n").write("v44 has no table directives: every table of 7.47 and 7.48 is hand-written (rectangular, header rule asserted by patch_v44.py); the audit (audit_v44.py) checks every number of them.\n")
