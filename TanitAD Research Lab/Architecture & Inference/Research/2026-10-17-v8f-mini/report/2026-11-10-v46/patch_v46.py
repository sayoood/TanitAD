"""Report v46 = the v45 report md (report_v45/out, sha256 pinned below) + sections 7.51 (planner gates, readout ceiling, band-trained ladder; T0) and 7.52 (band loss on the 6 s plan and the levers behind
DAG-OACC; T1-kin) inserted after 7.50, + the status line, abstract (two bullets, What-is-next sentence), 8.1 item 36, 8.3 Update (v46), 8.4 rows 1cb to 1ce, 9 item 24, the appendix block, and the carried v45 lines
that said the ladder was untrained / the PLANTEST2 result absent (edits_v46.md: old --> new). Anchored edits, each anchor asserted to occur exactly once. Template and register read with git show (read only).
python -I patch_v46.py tip [out_dir]"""
import hashlib, os, re, subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
assert _argv, "usage: patch_v46.py <tip> [out_dir]"
TIP = _argv[0]
assert re.fullmatch(r"[0-9a-f]{40}", TIP), "pass the full 40-hex tip"
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
OUT = (_argv[1] if len(_argv) > 1 else HERE + "/out").rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BASE_MD = "E:/Projects/TanitAD-artifacts/dagland/report_v45/out/V8F_SCIENTIFIC_REPORT.md"
BASE_HTML = "E:/Projects/TanitAD-artifacts/dagland/report_v45/out/v8f_report.html"
BASE_SHA = "324c200158fe27decacccb4aac1fbf2e0b6e1711fb34fb8b213ab32e556218b0"
BANNED = tuple(w[::-1] for w in ("diov", "dedesrepus", "detcarter", "ylremrof", "regnol on", "noitcerroc", "nwardhtiw"))
_BAN = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)
_CL = re.compile(r"closed[- ]loop", re.I)


def gitrun(args, binary=False):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout if binary else r.stdout.decode("utf-8").strip()


def blob(path):
    t = gitrun(["show", f"{TIP}:{path}"], binary=True).decode("utf-8")
    assert len(t) > 0 and "\ufeff" not in t, path
    return t.replace("\r", "")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def part(name):
    t = open(os.path.join(HERE, name), encoding="utf-8", newline="").read()
    assert "\r" not in t and "\ufeff" not in t, name
    return t.strip("\n")


raw0 = open(BASE_MD, "rb").read()
assert sha(raw0) == BASE_SHA, "the v45 md is not the pinned base"
s = raw0.decode("utf-8")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert tpl.count("%%MD%%") == 1
assert open(BASE_HTML, "rb").read().decode("utf-8") == tpl.replace("%%MD%%", s), "v45 html is not the template with the v45 md"
assert s.count("**Status:** research report, v45 of 2026-11-08") == 1
S0 = s
# corpus for the hex-token guard: tip raw files of 2026-11-09 and 2026-11-10 + register + base
names = [n for n in gitrun(["ls-tree", "-r", "--name-only", TIP, "--", R + "raw"]).split("\n") if re.search(r"/2026-11-(09|10)-", n) and n.endswith((".md", ".txt", ".json"))]
p = subprocess.Popen(["git", "-C", MIRROR, "-c", "safe.directory=*", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
_RAWS = []
for n in names:
    p.stdin.write(f"{TIP}:{n}\n".encode()); p.stdin.flush()
    size = int(p.stdout.readline().decode().split()[2]); data = p.stdout.read(size); p.stdout.read(1)
    if size <= 3_000_000:
        _RAWS.append(data.decode("utf-8", "replace").replace("\r", ""))
p.stdin.close(); p.wait()
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
_corp = re.sub(r"\s+", " ", " ".join(_RAWS + [REG, S0]).replace("**", ""))

_blocks = re.split(r"^=== (\w+) ===\n", part("edits_v46.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
PROSE = ["ABS3", "APP", "I24", "R85", "ROW", "S82", "V46S"]
CARRIED = ["C3A", "C3B", "NXT2", "ROW1BZ", "BLK", "NOTRUN"]
assert sorted(B) == sorted(PROSE + CARRIED + ["NOTES"]), sorted(B)
assert "\n" not in B["V46S"] and "<<" not in B["V46S"] and "{{" not in B["V46S"]
B = {k: v.replace("<<V46S>>", B["V46S"]) for k, v in B.items()}


def ncols(line):
    return line.replace("\\|", "").count("|") - 1


_RULE = re.compile(r"\|[\s:\-|]+")


def fill(t):
    t = t.replace("{{TP}}", TP)
    left = re.findall(r"\{\{[^{}\n]*\}\}", t)
    assert not left and "<<" not in t, left[:3]
    return t


SECS = [fill(part(f)) for f in ("sec_751_planner_gates_ladder.md", "sec_752_otband.md")]
SEC = "\n\n".join(SECS)
B = {k: fill(v) for k, v in B.items()}
_heads = re.findall(r"^### (7\.\d+) ", SEC, flags=re.M)
assert _heads == ["7.51", "7.52"], _heads
_tabs, _cur = [], []
for l in SEC.split("\n"):
    if l.startswith("|"):
        _cur.append(l)
    elif _cur:
        _tabs.append(_cur); _cur = []
if _cur:
    _tabs.append(_cur)
assert all(len({ncols(r) for r in t}) == 1 and _RULE.fullmatch(t[1]) for t in _tabs), "ragged table in a section"
_ROWS = [l for l in B["ROW"].split("\n") if l.startswith("|")]
assert [l.split(" | ")[0] for l in _ROWS] == ["| 1c" + c for c in "bcde"] and all(ncols(r) == 4 for r in _ROWS), "8.4 rows 1cb to 1ce"
_APPT = [l for l in B["APP"].split("\n") if l.startswith("|")]
assert len(_APPT) == 10 and len({ncols(r) for r in _APPT}) == 1 and _RULE.fullmatch(_APPT[1]), "appendix table"
ADDED = "\n".join([SEC] + [B[k] for k in PROSE] + [B[k].split("\n-->\n")[1] for k in CARRIED])
_hex = sorted(set(re.findall(r"(?<![0-9a-zA-Z])(?=[0-9a-f]*\d)[0-9a-f]{8}(?![0-9a-zA-Z])", ADDED)))
_nohex = []
for x in _hex:
    if x in _corp:
        continue
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "rev-parse", "--verify", "--quiet", x + "^{commit}"], capture_output=True)
    if r.returncode != 0:
        _nohex.append(x)
assert not _nohex, ("8-hex prefix in no tip record and no commit", _nohex)
_ids = sorted(set(re.findall(r"`([DH]-[A-Z0-9]+(?:-[A-Z0-9]+)*)`", ADDED)))
_noids = [x for x in _ids if x not in REG]
assert not _noids, ("register row id not in the register", _noids)

ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
ANCHOR_10 = "\n\n---\n\n## 10. Reproducibility appendix"
_pairs = {}
for k in CARRIED:
    a, b = B[k].split("\n-->\n")
    assert a and b and a != b, k
    _pairs[k] = (a, b)
E = [_pairs[k] for k in CARRIED] + [
    ("Nothing in v45 is a driving result.\n\n**Evidence classes used throughout.**", "Nothing in v45 is a driving result. " + B["V46S"] + "\n\n**Evidence classes used throughout.**"),
    ("\n* **What is next.** The crawl set is still open", "\n" + B["ABS3"] + "\n* **What is next.** The crawl set is still open"),
    ("Its limits are in §7.49.8, §7.50.5 and section 9 item 23.\n\n### 8.2 What is not established", "Its limits are in §7.49.8, §7.50.5 and section 9 item 23.\n" + B["S82"] + "\n\n### 8.2 What is not established"),
    ("**Update (v45).** Two items are written up here", B["R85"] + "\n\n**Update (v45).** Two items are written up here"),
    ("\n| 2 | **LoRA on DINOv3**", "\n" + B["ROW"] + "\n| 2 | **LoRA on DINOv3**"),
    ("\n\n**Figures.** `make_figures.py` in this folder", "\n\n" + B["APP"] + "\n\n**Figures.** `make_figures.py` in this folder"),
    (ANCHOR_10, "\n" + B["I24"] + ANCHOR_10),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
assert s.count(ANCHOR_8) == 1 and s.index("**7.50.8 ") < s.index(ANCHOR_8)
_j = s.index(ANCHOR_8)
s = s[:_j] + "\n\n" + SEC + s[_j:]

ins = "".join(b for _, b in E) + SEC
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower().replace("retraction_log", "")
assert re.search("retract|supersed|" + BANNED[6] + "|was wrong|previously read|outdated|i said", low) is None
assert not _CL.search(ins), "forbidden T2 wording"
low2 = low.replace("nothing here is driving performance", "")  # the one negated use, in section 7.51 (i)
assert not re.search(r"\bdrives\b|driving performance", low2), "drives / driving performance wording"
assert low.count("nothing here is driving performance") <= 1
assert "\r" not in ins and not re.search(r"[\u2013\u2014]", ins), "CR or en/em dash"
assert "$" not in re.sub(r"`[^`\n]+`", " ", ins), "dollar sign in added text"
assert re.search(r"<[A-Za-z/!]", re.sub(r"`[^`\n]+`", " ", ins)) is None, "html-like text in added text"
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
_nflat = re.sub(r"\s+", " ", SEC)
for must in ("PAUSED", "T0", "T1-kin", "H-ESTIM-SEED-1", "NO-GAIN", "BAND-MET", "BAND-COST", "never driving results"):
    assert must in _nflat, must
assert s.index("**7.50.8 ") < s.index("### 7.51 ") < s.index("### 7.52 ") < s.index("## 8. Discussion")
assert s.count("**Status:** research report, v46 of 2026-11-10") == 1
assert not [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if d > "2026-11-10"], "a date after 2026-11-10"
h = tpl.replace("%%MD%%", s)
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
open(OUT + "DIRECTIVES.txt", "w", encoding="utf-8", newline="\n").write("v46 has no table directives: every table of 7.51 and 7.52 is hand-written (rectangular, header rule asserted by patch_v46.py); check_numbers_v46.py and audit_v46.py check the numbers.\n")
print("v46 built", len(S0), len(s), "tip", TIP, "7.51-7.52 chars", len(SEC), "hand tables", len(_tabs), "hex tokens", len(_hex), "row ids", len(_ids), "placeholders left", sorted(set(re.findall(r"\{\{[^{}\n]*\}\}", s))))
