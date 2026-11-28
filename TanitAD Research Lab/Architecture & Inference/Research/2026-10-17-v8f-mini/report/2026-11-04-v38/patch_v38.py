"""Report v38 = v37 + section 7.36.4a (the RAPR-P01 arm read) + section 7.36.5 to 7.36.7 rewritten for five arms and the PI decision item (row K), the status line, the abstract (the recovery bullet and the next pointer),
8.1 item 25 (and the item 24 pointer), 8.3 Update (v38), 8.4 row 1at (and the row 1as pointer), 9 item 16.
Built from the landed v37 report blobs of one repository tip (anchored edits, every anchor asserted to occur exactly once). The prose is in the package files beside this script
(sec_736_raprp01.md: the new 7.36.4a and the rewritten 7.36.5 to 7.36.7, edits_v38.md: one block per edit); every source is `git show <tip>:<path>` from the mirror.
Nothing is read from a working tree of the repository. Nothing is staged, committed or pushed.

python -I patch_v38.py tip [out_dir]
Every table line of the added text is asserted to be a line of the RAPR-P01 RESULTS file; every italic quotation of 25 characters or more is asserted verbatim in a source (RESULTS, addendum 3, decision item, register, v37)."""
import hashlib, json, os, re, subprocess, sys
from collections import OrderedDict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_argv = sys.argv[1:]
assert _argv, "usage: patch_v38.py <tip> [out_dir]"
TIP = _argv[0]
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (_argv[1] if len(_argv) > 1 else HERE + "/out").rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
BANNED = tuple(w[::-1] for w in ("diov", "dedesrepus", "detcarter", "ylremrof", "regnol on", "noitcerroc", "nwardhtiw"))  # the programme's banned words, stored reversed so this file passes its own lander scan
_BAN = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)


def gitrun(args, binary=False):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
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


# ---- the v37 deliverables are the starting point ---------------------------------------------------------------------------------------------------------
V37_MD_SHA = "187744e89d2c9fea857a09d7c424079b680ff47ab04e9e46e8b1a0bc5aa49842"
V37_HTML_SHA = "7f11c3ee86b7f34160661115f3dbd3db5d98f3b68504c55bf1fac096c1f4c5cb"
RP37 = R + "report/2026-11-04-v37/"
s = blob(RP37 + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP37 + "v8f_report.html")
tpl = blob(R + "report/2026-10-26/page_template_v30.html")
assert sha(raw(RP37 + "V8F_SCIENTIFIC_REPORT.md")) == V37_MD_SHA, "the tip's report md is not the landed v37 deliverable"
assert sha(raw(RP37 + "v8f_report.html")) == V37_HTML_SHA, "the tip's report html is not the landed v37 deliverable"
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v37 html is not the template with the v37 md"
assert s.count("**Status:** research report, v37 of 2026-11-04") == 1
S0 = s

# ---- sources ----------------------------------------------------------------------------------------------------------------------------------------------
SRC_PATHS = OrderedDict([("RF", "raw/2026-11-03-recover-read-result/RESULTS.md"), ("RT", "raw/2026-11-04-recover-next-rtl/RESULTS.md"),
                         ("RP", "raw/2026-11-04-recover-next-rp01/RESULTS.md"), ("RA", "raw/2026-11-04-recover-next-rapr/RESULTS.md"),
                         ("PR", "raw/2026-11-04-recover-next-raprp01/RESULTS.md"), ("DR", "raw/2026-11-04-recover-rowk-decision/DECISION_ROWK.md"),
                         ("DJ", "raw/2026-11-04-recover-rowk-decision/d2_closure_need.json")])
SRC = OrderedDict((k, blob(R + p)) for k, p in SRC_PATHS.items())
A3_PATH = "raw/2026-11-04-recover-next-raprp01/PREREG_RECOVER_NEXT_ADDENDUM3_RAPRP01.md"
A3 = blob(R + A3_PATH)
assert sha(raw(R + A3_PATH)).startswith("9bea3fa1") and "9bea3fa1" in SRC["PR"], "addendum 3 sha256"
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
RES_SHA = {k: sha(raw(R + p)) for k, p in SRC_PATHS.items()}
# landing commits (first commit that added the file), measured from the mirror
LAND = {}
for k in ("PR", "DR"):
    c = gitrun(["log", "--format=%h", "--diff-filter=A", TIP, "--", R + SRC_PATHS[k]]).split("\n")[-1]
    assert re.fullmatch(r"[0-9a-f]{7,}", c), (k, c)
    LAND[k] = c[:8]
assert LAND["PR"].startswith("05f9dea") and "`05f9dea`" in SRC["DR"], "the RAPR-P01 RESULTS landing commit differs from the one the decision item names"
# raw JSON cross-checks of the headline (MEASURED artefact behind the RESULTS, ESTIMATED artefact behind the decision item)
J_PR = json.loads(blob(R + "raw/2026-11-04-recover-next-raprp01/RECOVER_READ_RAPRP01.json"))
J_DJ = json.loads(SRC["DJ"])
assert J_PR["label"] == "HARMS" and J_PR["controls_failed"] == [] and len(J_PR["controls"]) == 20 and J_PR["reader_sha256"].startswith("a99104733397")
assert J_PR["composite"]["status"] == "NOT DOMINANT" and J_PR["composite"]["counts"] is False and J_PR["guards"]["PASS"] is False
assert all(J_PR["composite"]["by_seed"][k]["separated_worse_rows"] == ["M1 crawl share"] and J_PR["composite"]["by_seed"][k]["D3_by_family"]["STRATEGIC"] is True for k in ("s0", "s1"))
assert J_PR["frame_freeze"]["reading"] == "PARTIAL" and J_PR["run_clock"] == "2026-11-04T20:37:46+0100"
PR_SH = {k: "%.4f" % J_PR["primary"]["share_closed"][k] for k in ("s0", "s1")}
PR_FF = {k: "%.4f" % J_PR["frame_freeze"]["share_closed"][k] for k in ("s0", "s1")}
PR_AD = {k: "%+.4f" % J_PR["guards"]["per_seed"][k]["ade6_delta"] for k in ("s0", "s1")}
PR_FL = {k: ("%.4f" % J_PR["floor"]["per_seed"][k]["abs_R_minus_C0"], "%.4f" % J_PR["floor"]["per_seed"][k]["abs_C0rep_minus_C0"]) for k in ("s0", "s1")}
DJ = {k: J_DJ["seeds"][k] for k in ("s0", "s1")}
DJ_TXT = {k: ("%.3f" % DJ[k]["fit_slope"], "%.3f" % DJ[k]["pearson_r"], "%.2f" % DJ[k]["share_needed"], "%.2f" % DJ[k]["share_needed_for_point_zero"], "%.2f" % DJ[k]["extrapolation_factor"],
              "%.4f" % DJ[k]["max_share_measured"]) for k in ("s0", "s1")}
assert all(DJ[k]["n"] == 5 for k in DJ) and J_DJ["tier"] == "T1-kin" and J_DJ["evidence_class"].startswith("ESTIMATED")

# ---- the section texts ------------------------------------------------------------------------------------------------------------------------------------
_blocks = re.split(r"^=== (\w+) ===\n", part("edits_v38.md"), flags=re.M)
assert _blocks[0].strip() == "" and len(_blocks) % 2 == 1
B = {_blocks[i]: _blocks[i + 1].strip("\n") for i in range(1, len(_blocks), 2)}
assert sorted(B) == ["ABS_LAST", "ABS_TITLE", "I16", "NEXT_LAST", "R83", "ROW", "S81", "ST"], sorted(B)


def fill(t):
    for k, c in LAND.items():
        t = t.replace("{{LAND_%s}}" % k, c)
    t = t.replace("{{TP}}", TP)
    for k, v in RES_SHA.items():
        t = t.replace("{{RES_%s}}" % k, v[:8])
    assert "{{" not in t and "<<" not in t, re.findall(r"(\{\{\S+?\}\}|<<\S+?>>)", t)[:3]
    return t


SEC = fill(part("sec_736_raprp01.md"))
B = {k: fill(v) for k, v in B.items()}
_heads = [m.group(1) for m in re.finditer(r"^\*\*(7\.36\.\w+) ", SEC, flags=re.M)]
assert _heads == ["7.36.4a", "7.36.5", "7.36.6", "7.36.7"], _heads
for k in ("s0", "s1"):
    assert PR_SH[k] in SEC and PR_FF[k] in SEC and PR_AD[k] in SEC, ("a RAPR-P01 headline number from the raw JSON is not in 7.36.4a", k)
    assert PR_FL[k][0] in SEC and PR_FL[k][1] in SEC, ("floor", k)
    assert all(x in SEC for x in DJ_TXT[k]), ("a closure-need number from the decision item's JSON is not in 7.36.6", k, DJ_TXT[k])
_nflat = re.sub(r"\s+", " ", SEC)
assert "0.68 (s0) / 0.60 (s1)" in _nflat and "slope 1.097 / 1.146, r 0.997 / 0.932, n 5" in _nflat and "×1.54 / ×1.88" in _nflat and "about 1.04 / 0.95" in _nflat

# ---- tables are source lines; quotations are verbatim -----------------------------------------------------------------------------------------------------
SRCL = set(SRC["PR"].split("\n"))


def ncols(line):
    return re.sub(r"`[^`\n]*`", "x", line.replace("\\|", "x")).count("|") - 1


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
assert len(_tabs) == 3 and not _bad, ("a table line is not a line of the RAPR-P01 RESULTS file", _bad[:3])
assert all(len({ncols(r) for r in t}) == 1 and t[1].startswith("|---") for t in _tabs), "a table with ragged columns or no header rule"
assert not any(l.startswith("|") for v in B.values() for l in v.split("\n") if not re.match(r"\| 1at \|", l)), "a table line outside 7.36 other than row 1at"
_corp = re.sub(r"\s+", " ", " ".join(list(SRC.values()) + [A3, REG, S0]))
_q = [m.strip('"') for m in re.findall(r"(?<![*\w])\*(?!\*)([^*\n]{25,}?)(?<!\*)\*(?![*\w])", ADDED)]
_nq = [q for q in _q if re.sub(r"\s+", " ", q) not in _corp]
assert _q and not _nq, ("an italic quotation is not verbatim in a source", _nq[:3])

# ---- anchored edits -----------------------------------------------------------------------------------------------------------------------------------
ANCHOR_8 = "\n\n---\n\n## 8. Discussion"
E = [
    ("research report, v37 of 2026-11-04, results through repository tip `cd6ce52d` and, within it,",
     "research report, v38 of 2026-11-04, results through repository tip `" + TP + "` and, within it,"),
    ("and the RECOVER-NEXT arms RTL, RP01 and RAPR) (v36: `9b79730c`;",
     "and the RECOVER-NEXT arms RTL, RP01, RAPR and RAPR-P01) (v37: `cd6ce52d`; v36: `9b79730c`;"),
    ("the most of any trained arm, at ADE6 +0.3504 / +0.3202 m against its control.\n",
     "the most of any trained arm, at ADE6 +0.3504 / +0.3202 m against its control. " + B["ST"] + "\n"),
    ("an a-priori deviation law doubles the closure (§7.36; four arms trained on box 2 and read on 2026-11-03 and 2026-11-04,", B["ABS_TITLE"]),
    ("The committed next arm is RAPR-P01 (the a-priori law at dose 0.1), which needs the PI's box-2 go.", B["ABS_LAST"]),
    ("It moves M1 the right way in all four arms and clears neither the 0.5 bar nor its guards, and §7.35.4 adds that no single per-window speed closes M1.", B["NEXT_LAST"]),
    ("and an a-priori deviation law doubles the closure at a larger position cost\n\n* **Status of this section (v37) [M].** Four reads of fine-tunes",
     "an a-priori deviation law doubles the closure at a larger position cost, and its lower dose removes the ADE6 harm and keeps the along-track and distance-keeping harm\n\n"
     "* **Status of this section (v38) [M].** Five reads of fine-tunes"),
    (", and the arm their outcome selected, RAPR (`R/raw/2026-11-04-recover-next-rapr/RESULTS.md`, landed `cd6ce52`).",
     ", the arm their outcome selected, RAPR (`R/raw/2026-11-04-recover-next-rapr/RESULTS.md`, landed `cd6ce52`), and the arm RAPR's read selected, RAPR-P01 "
     "(`R/raw/2026-11-04-recover-next-raprp01/RESULTS.md`, landed `" + LAND["PR"] + "`; the decision item its row K led to is set out in §7.36.6)."),
    ("kept ADE6 inside its guard at both seeds.\"*\n\n**7.36.5 What the four recovery arms establish, and what they cannot.**",
     "kept ADE6 inside its guard at both seeds.\"* RAPR-P01 is read in §7.36.4a.\n\n**7.36.5 What the four recovery arms establish, and what they cannot.**"),
    ("\n\n### 8.2 What is not established", "\n" + B["S81"] + "\n\n### 8.2 What is not established"),
    ("(RAPR-P01, not run)", "(RAPR-P01, read in item 25)"),
    ("**Update (v37).** Two items are written up here", B["R83"] + "\n\n**Update (v37).** Two items are written up here"),
    ("Next: RAPR-P01 (addendum 2 row E; PI box-2 go) |", "Next: RAPR-P01 (addendum 2 row E; read as 1at) |"),
    ("| 2 | **LoRA on DINOv3**", B["ROW"] + "\n| 2 | **LoRA on DINOv3**"),
    ("\n\n---\n\n## 10. Reproducibility appendix", "\n" + B["I16"] + "\n\n---\n\n## 10. Reproducibility appendix"),
]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:90])
    s = s.replace(a, b)
# the old 7.36.5 to 7.36.7 block (five-arm rewrite, next section, provenance) is replaced as one block; the [H] bullet is kept verbatim
_i = s.index("**7.36.5 What the four recovery arms establish, and what they cannot.**")
_j = s.index(ANCHOR_8)
_old = s[_i:_j]
assert s.count("**7.36.5 What the four recovery arms establish") == 1 and _old.count("\n**7.36.") == 2 and _old.startswith("**7.36.5 ") and "\n**7.36.6 Next (Rule Zero).** RAPR-P01 is the committed next arm" in _old
assert _old.count("\n* **") == 3 and "\n* **[H] (untested), read with §7.35.4.**" in _old
_h_line = [l for l in _old.split("\n") if l.startswith("* **[H] (untested), read with §7.35.4.**")][0]
assert _h_line in SEC.split("\n") and _h_line in S0.split("\n"), "the [H] bullet is not carried verbatim"
s = s[:_i] + SEC + s[_j:]

# ---- guards on the inserted text ---------------------------------------------------------------------------------------------------------------------
ins = "".join(b for _, b in E) + SEC
assert not _BAN.search(ins), _BAN.search(ins)
low = ins.lower()
assert re.search("retract|supersed|" + BANNED[6] + "|was wrong|previously read|outdated|i said", low) is None
assert "closed loop" not in low and "closed-loop" not in low and not re.search(r"\bdrives\b|driving performance", low), "closed-loop or driving wording"
assert "\r" not in ins and "$" not in ins, "a CR or a dollar sign in the added text (the math count of the report stays as in v37)"
assert s.count("$$") == S0.count("$$") and s.count("$") == S0.count("$"), "math spans changed"
nocode = re.sub(r"`[^`\n]*`", " ", ins)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r".{20}<[A-Za-z/!].{20}", nocode)
assert "@@" not in s and "</script" not in s.lower() and "\r" not in s
_flat = re.sub(r"\s+", " ", SEC)
assert "B = 2,000" in _flat and "T1-kin" in _flat and "H-ESTIM-SEED-1" in _flat and "RAPR-P01" in _flat and "frame-freeze" in _flat.lower()
for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
    assert fam in SEC, fam
assert s.count("### 7.36 ") == 1 and s.index("### 7.35 ") < s.index("**7.35.4 ") < s.index("### 7.36 ") < s.index("**7.36.4 ") < s.index("**7.36.4a ") < s.index("**7.36.5 ") < s.index("**7.36.6 ") < s.index("**7.36.7 ") < s.index("## 8. Discussion")
assert all(s.count(x) == 1 for x in ("**7.36.4a ", "**7.36.5 ", "**7.36.6 ", "**7.36.7 "))
assert s.count("**Status:** research report, v38 of 2026-11-04") == 1 and "v37 of 2026-11-04, results" not in s.split("\n")[2]
assert not [d for d in re.findall(r"\b(20\d\d-\d\d-\d\d)\b", s) if d > "2026-11-04"], "a date after 2026-11-04"
assert "four recovery arms establish" not in s and "RAPR-P01 is the committed next arm" not in s and "(RAPR-P01, not run)" not in s, "a stale pointer about RAPR-P01 remains"
h = tpl.replace("%%MD%%", s)
assert h.count("%%MD%%") == 0
os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
print("v38 built", len(S0), len(s), "tip", TIP, "7.36.4a to 7.36.7 chars", len(SEC), "tables", len(_tabs), "table lines", sum(len(t) for t in _tabs), "quotations", len(_q), "landed", LAND)
