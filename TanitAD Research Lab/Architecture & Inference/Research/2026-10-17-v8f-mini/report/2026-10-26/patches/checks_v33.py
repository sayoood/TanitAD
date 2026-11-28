"""Runs the report-v33 checks and writes v33_checks.txt (the patch's own assertions have already passed to produce out/; these are the independent ones).
python checks_v33.py [tip [out_dir [checks_path [audit_path]]]]   (needs: python, git, node; no network; the superset and demotion checkers are exported from the tip).
out_dir: the folder holding V8F_SCIENTIFIC_REPORT.md and v8f_report.html (default out/ beside this script, else final/, else the folder above it).
The deliverable may be the PRE-DRAFT (marked slots @@LI_RESULT_<NAME>@@) or a FINAL version (no slot); the checks read the mode from the file and test each mode's own rules.
The final mode also re-runs the slot gate (refusals and a complete test fill) on throw-away folders, rebuilds the deliverables byte for byte from FILL.json, and regenerates FILL.json from mk_fill_v33.py."""
import contextlib
import datetime
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1] if len(sys.argv) > 1 else "8c1e9636388dbb105f6b1853bd95dfdccea9eb93"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"


def _out_dir():
    if len(sys.argv) > 2:
        return sys.argv[2]
    for d in (HERE + "out", HERE + "final"):
        if os.path.exists(d + "/V8F_SCIENTIFIC_REPORT.md"):
            return d
    up = os.path.dirname(HERE.rstrip("/"))
    return up if os.path.exists(up + "/V8F_SCIENTIFIC_REPORT.md") else HERE + "out"


OUT = os.path.abspath(_out_dir()).replace("\\", "/").rstrip("/") + "/"
CHECKS_PATH = sys.argv[3] if len(sys.argv) > 3 else HERE + "v33_checks.txt"
AUDIT_FILE = sys.argv[4] if len(sys.argv) > 4 else HERE + "v33_NUMBERS_AUDIT.md"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RP = R + "report/2026-10-26/"
ENV = dict(os.environ, GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*", PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="-1")
lines = []


def say(s=""):
    s = s.replace(chr(13), "")  # tool output from Windows subprocesses carries CRLF; the checks file is LF-only
    lines.append(s)
    print(s)


def git(args):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*"] + args, capture_output=True)
    assert r.returncode == 0, (args, r.stderr[:200])
    return r.stdout


def blob(path):
    return git(["show", f"{TIP}:{path}"]).decode("utf-8")


md_p, html_p = OUT + "V8F_SCIENTIFIC_REPORT.md", OUT + "v8f_report.html"
md = open(md_p, encoding="utf-8", newline="").read()
html = open(html_p, encoding="utf-8", newline="").read()
v32 = blob(RP + "V8F_SCIENTIFIC_REPORT.md")
v32h = blob(RP + "v8f_report.html")
tpl = blob(RP + "page_template_v30.html")
full_tip = git(["rev-parse", TIP]).decode().strip()
SLOT = re.compile(r"@@LI_RESULT_([A-Z0-9]+)@@")
PREDRAFT = bool(SLOT.search(md))

say(f"checks.txt for report v33 (source tip {TIP[:8]} = {full_tip}; v32 blobs read with git show; run {datetime.datetime.now().astimezone().isoformat(timespec='seconds')})")
say("checked deliverables (sha256): V8F_SCIENTIFIC_REPORT.md " + hashlib.sha256(open(md_p, "rb").read()).hexdigest() + " | v8f_report.html " + hashlib.sha256(open(html_p, "rb").read()).hexdigest())
say("mode of the checked deliverable: %s" % ("PRE-DRAFT (marked LAUNCH-INTENT outcome slots present)" if PREDRAFT else "FINAL (no slot marker)"))
say()
_stack = contextlib.ExitStack()  # the scratch folders below are removed when the script ends
tools = _stack.enter_context(tempfile.TemporaryDirectory(prefix="v33tools_"))
for name, path in (("superset_check.py", "tools/superset_check.py"),
                   ("demote_check.py", "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-demotion-guard/code/demote_check.py")):
    open(tools + "/" + name, "wb").write(git(["show", f"{TIP}:{path}"]))
for label, rel, newp in (("V8F_SCIENTIFIC_REPORT.md", RP + "V8F_SCIENTIFIC_REPORT.md", md_p), ("v8f_report.html", RP + "v8f_report.html", html_p)):
    for tool, args in (("superset_check.py", [f"{TIP[:8]}:{rel}", newp, "none", MIRROR + "/.git"]), ("demote_check.py", [f"{TIP[:8]}:{rel}", newp, "none"])):
        say(f"### {tool} {TIP[:8]}:<repo>/{label}  <v33 {label}>  none")
        r = subprocess.run([sys.executable, tools + "/" + tool] + args, capture_output=True, env=ENV, cwd=tools)
        say((r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")).strip())
        assert r.returncode == 0, (tool, label)
    say()

say("== bytes / line endings / encoding")
FILES = [("V8F_SCIENTIFIC_REPORT.md", md_p), ("v8f_report.html", html_p), ("patch_v33.py", HERE + "patch_v33.py"), ("audit_v33.py", HERE + "audit_v33.py"),
         ("audit_reviewed_v33.py", HERE + "audit_reviewed_v33.py"), ("checks_v33.py", HERE + "checks_v33.py"), ("render_check_v33.js", HERE + "render_check_v33.js"), ("v33_NUMBERS_AUDIT.md", AUDIT_FILE)]
for extra in ("LI_FILL_TEMPLATE.json", "LI_FILL_HELP.txt"):
    if os.path.exists(OUT + extra):
        FILES.append((extra, OUT + extra))
for extra in ("FILL.json", "mk_fill_v33.py", "mk_hold_report_v33.py"):
    if os.path.exists(HERE + extra):
        FILES.append((extra, HERE + extra))
for label, p in FILES:
    b = open(p, "rb").read()
    b.decode("utf-8")
    say(f"{label}: {len(b)} bytes, CR bytes={b.count(bytes([13]))}, LF bytes={b.count(bytes([10]))}, BOM={b.startswith(bytes([239, 187, 191]))}, utf-8 decode OK")
    assert b.count(bytes([13])) == 0

say(f"== html == page_template_v30.html with the md placeholder replaced by the v33 md: {html == tpl.replace('%%MD%%', md)}")
assert html == tpl.replace("%%MD%%", md)
say("== v32 html == the same template with the v32 md (the starting point is consistent): %s" % (v32h == tpl.replace("%%MD%%", v32)))
assert v32h == tpl.replace("%%MD%%", v32)
secs = ("### 7.24 SPEEDREAD", "### 7.25 DAgger iteration 2", "### 7.26 DAG-ORC", "### 7.27 SPEEDSUB", "### 7.28 SPEEDSUB-T1", "### 7.29 LAUNCH-INTENT")
say("== html and md contain the sections 7.24 to 7.29: %s" % all(s in html and s in md for s in secs))
assert all(s in html and s in md for s in secs)
say("== section order 7.26 < 7.27 < 7.28 < 7.29 < 8. Discussion: %s" % (md.index("### 7.26 ") < md.index("### 7.27 ") < md.index("### 7.28 ") < md.index("### 7.29 ") < md.index("## 8. Discussion")))
assert md.index("### 7.26 ") < md.index("### 7.27 ") < md.index("### 7.28 ") < md.index("### 7.29 ") < md.index("## 8. Discussion")

a, b = v32.split("\n"), md.split("\n")
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
added, removed = [], []
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op in ("replace", "insert"):
        added += b[j1:j2]
    if op in ("replace", "delete"):
        removed += a[i1:i2]
say("== diff v32 -> v33 md: %d added/changed lines, %d v32 lines replaced or removed" % (len(added), len(removed)))
for l in removed:
    say("   REPLACED v32 line: " + l[:150] + (" ..." if len(l) > 150 else ""))
add_text = "\n".join(added).lower()
say("== banned words in added/changed lines:")
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    n = len(re.findall(r"\b" + w + r"\b", add_text))
    say(f"   {w}: {n}")
    assert n == 0
extra_m = len(re.findall(r"retract|supersed|withdrawn|was wrong|not the floor|previously read|outdated|i said", add_text))
say(f"   demotion-guard retraction-context markers (RETRACT, SUPERSED, WITHDRAWN, was wrong, not the floor, previously read, outdated, I said): {extra_m}")
assert extra_m == 0
CARRIED = "**no closed-loop (t2) number exists**"  # the status line's own negation, carried from v30 (the line is re-counted as changed because the version string moved)
hits = [w for w in ("closed loop", "closed-loop", "drives") if w in add_text.replace(CARRIED, "")]
say("== 'closed loop' / 'closed-loop' / 'drives' in added lines, after setting aside the status line's carried negation (%s): %s" % (CARRIED, hits))
assert not hits

i27, i28, i29 = md.index("### 7.27 "), md.index("### 7.28 "), md.index("### 7.29 ")
iend = md.index("\n---\n\n## 8. Discussion")
S = {"7.27": md[i27:i28], "7.28": md[i28:i29], "7.29": md[i29:iend]}
for k, t in S.items():
    stamps = [l for l in t.split("\n") if re.search(r"T1-kin|T1-class|\bT0\b", l)]
    fams = {f: (f in t) for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}
    flat = re.sub(r"\s+", " ", t)
    say(f"== {k}: lines carrying a tier stamp (T1-kin / T1-class / T0): {len(stamps)} | families present: {fams} | 'B = 2,000': {'B = 2,000' in flat} | '141 episodes': {'141 episodes' in flat}")
    assert all(fams.values()) and "B = 2,000" in flat and "141 episodes" in flat
flat27, flat28, flat29 = (re.sub(r"\s+", " ", S[k]) for k in ("7.27", "7.28", "7.29"))
assert "STRATEGIC is reported, not paused, for v8F" in flat27 and "STRATEGIC is reported, not paused, for v8F" in flat28 and "STRATEGIC is not paused for v8F" in flat29
say("== STRATEGIC is stated as reported, not paused, for v8F in 7.27: True | 7.28: True | 7.29: True ('is not paused for v8F')")
say("== tier discipline: T1-kin is never called closed loop; the only T2 mentions are negations ('not T2'): %s" % [l.strip()[:90] for l in added if "T2" in l])
tier27 = "T1-class open loop" in flat27 and "no replicate arm" in flat27 and "screening result" in flat27 and "(`H-ESTIM-SEED-1`)" in flat27
say("== 7.27 states: tier T1-class open loop, no replicate arm, screening result, H-ESTIM-SEED-1: %s" % tier27)
assert tier27
decl27 = "D-SPEEDBAND-EGO-ECHO-1" in flat27 and "declared" in flat27 and "that is why SPEEDSUB-T1" in flat27
say("== 7.27 states plainly that the PASS rests on the declared open channel `limit` (D-SPEEDBAND-EGO-ECHO-1) and why SPEEDSUB-T1 followed: %s" % decl27)
assert decl27
n28 = "NULL-ONE" in flat28 and "REP s0" in flat28 and "does not bind" in re.sub(r"\s+", " ", md[md.index("## 1. Abstract"):md.index("## 2. Introduction")]) + flat28
say("== 7.28 states NULL, NULL-ONE, the one hit at the slow-T draw REP s0, and that the speed-cell channel does not bind on healthy DAG-OACC draws: %s" % n28)
assert n28
red = "The bar and arms were changed by the Master Mind before promotion" in flat29 and "1.3 to 9.1 times" in flat29 and "2 × max(F, RF0)" in flat29 and "524,288" in flat29
say("== 7.29 states the Master Mind's redesign before any data (bar 2 x max(F, RF0), the old bar 1.3 to 9.1 times the gap, two same-seed pairs per seed) and the 524,288-cell table: %s" % red)
assert red
if PREDRAFT:
    li_not = [k for k in ("NOT BUILT", "NOT COMPUTABLE") if k in flat29] + [k for k in ("anti-echo block", "M-LAUNCH", "UNDECIDED") if k in flat29]
    say("== 7.29 states per family what the reader does not build (M-LAUNCH, anti-echo, M2 on the launch subset, T0 rows) and that next_launch prints UNDECIDED: %s" % li_not)
    assert "NOT COMPUTABLE" in flat29 and "anti-echo block" in flat29 and "M-LAUNCH" in flat29 and "UNDECIDED" in flat29 and "**Not built in the reader, per family**" in flat29
else:
    t0rows = {f: [l for l in S["7.29"].split("\n") if l.startswith("| %s |" % f)] for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}
    t0ok = all(len(v) == 1 and "NOT RUN" in v[0] for v in t0rows.values())
    say("== 7.29 states, per family, that the T0 four families are NOT RUN (one table row each, with its reason): %s" % t0ok)
    assert t0ok
    nb = all(k in flat29 for k in ("NOT BUILT", "NOT COMPUTABLE", "NOT RUN", "anti-echo block", "M-LAUNCH", "The JSON field `next_launch` is null"))
    say("== 7.29 names what the reader of record does not build or compute (M-LAUNCH, anti-echo block, M2 / L2 launch subset: NOT BUILT; T-head F1: NOT COMPUTABLE; T0 families: NOT RUN) and that next_launch is null: %s" % nb)
    assert nb
    say("== 7.29 states that the families are reported per seed and per pair, never pooled: %s" % ("never pooled" in flat29))
    assert "never pooled" in flat29
say()

# ---- slots and the final-mode gate ----
slots_md = SLOT.findall(md)
slots_html = SLOT.findall(html)
TEMPLATE = OUT + "LI_FILL_TEMPLATE.json"
if PREDRAFT:
    names = ["LINE", "ABSTRACT", "STATUS729", "OUTCOME", "CONTROLS", "RUNS", "FLOORS", "CRAWL", "GUARDS", "FAMILIES", "MLAUNCH", "NUMBERS", "NEXT", "PROVENANCE", "R81", "R83", "R84"]
    say("== PRE-DRAFT: the md carries %d slot markers, each name once: %s | the html carries the same: %s" % (len(slots_md), sorted(slots_md) == sorted(names), slots_html == slots_md))
    assert sorted(slots_md) == sorted(names) and slots_html == slots_md
    tj = json.load(open(TEMPLATE, encoding="utf-8"))
    say("== LI_FILL_TEMPLATE.json holds the 17 marker names and TITLE729 (the heading slot), all empty: %s" % (sorted(tj) == sorted(names + ["TITLE729"]) and all(v == "" for v in tj.values())))
    assert sorted(tj) == sorted(names + ["TITLE729"]) and all(v == "" for v in tj.values())
else:
    say("== FINAL: no slot marker in md (%d) or html (%d); '@@' count md %d, html %d (the template's own: %d)" % (len(slots_md), len(slots_html), md.count("@@"), html.count("@@"), tpl.count("@@")))
    assert not slots_md and not slots_html and "@@" not in md and html.count("@@") == tpl.count("@@")  # the page template's own @@CODE / @@MATH placeholders are not slots
    for stale in ("not read in this version", "Not read in this version", "pre-draft", "not started in this version", "PRE-REGISTERED, RUNNING"):
        assert stale not in S["7.29"], stale
    say("== FINAL: the pre-draft wording is gone from 7.29: True")
    FILLJ = json.load(open(HERE + "FILL.json", encoding="utf-8"))
    NAMES18 = ["LINE", "ABSTRACT", "TITLE729", "STATUS729", "OUTCOME", "CONTROLS", "RUNS", "FLOORS", "CRAWL", "GUARDS", "FAMILIES", "MLAUNCH", "NUMBERS", "NEXT", "PROVENANCE", "R81", "R83", "R84"]
    okf = sorted(FILLJ) == sorted(NAMES18) and all(isinstance(v, str) and v.strip() for v in FILLJ.values()) and not any("@@" in v for v in FILLJ.values())
    say("== FILL.json holds the 18 slot names, each a non-empty text with no slot marker (%d characters in all): %s" % (sum(len(v) for v in FILLJ.values()), okf))
    assert okf
    if os.path.exists(TEMPLATE):
        tjt = json.load(open(TEMPLATE, encoding="utf-8"))
        say("== FILL.json has exactly the keys of LI_FILL_TEMPLATE.json (the template is the pre-draft aid kept beside the deliverables): %s" % (sorted(tjt) == sorted(FILLJ)))
        assert sorted(tjt) == sorted(FILLJ)
    tj = {k: "" for k in FILLJ}
# the final-mode gate and the final path, on throw-away folders
with tempfile.TemporaryDirectory(prefix="v33final_") as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v33.py", TIP, td + "/refused"], capture_output=True, env=ENV)
    refused = r.returncode != 0 and not os.path.exists(td + "/refused")
    say("== final mode without --fill REFUSES (non-zero exit, nothing written): %s | message: %s" % (refused, (r.stderr.decode("utf-8", "replace").strip() or r.stdout.decode("utf-8", "replace").strip())[:140]))
    assert refused
    part = {k: "x" for k in tj}
    part["CRAWL"] = ""
    json.dump(part, open(td + "/partial.json", "w", encoding="utf-8"))
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v33.py", TIP, td + "/partial", "--fill", td + "/partial.json"], capture_output=True, env=ENV)
    say("== final mode with one empty slot REFUSES and writes nothing: %s | %s" % (r.returncode != 0 and not os.path.exists(td + "/partial"), (r.stderr.decode("utf-8", "replace").strip())[:140]))
    assert r.returncode != 0 and not os.path.exists(td + "/partial")
    marked = {k: "Test text." for k in tj}
    marked["ABSTRACT"] = "* **Test.** Text.\n* **What is next.** Text."
    marked["TITLE729"] = "TEST title"
    marked["R84"] = "**TEST** status"
    marked["FAMILIES"] = "* **Families.** LONGITUDINAL, LATERAL, TACTICAL, STRATEGIC."
    marked["OUTCOME"] = "* **Outcome.** @@LI_RESULT_OUTCOME@@ left in by mistake"
    json.dump(marked, open(td + "/marked.json", "w", encoding="utf-8"))
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v33.py", TIP, td + "/marked", "--fill", td + "/marked.json"], capture_output=True, env=ENV)
    say("== final mode with a slot marker left inside a fill text REFUSES: %s" % (r.returncode != 0 and not os.path.exists(td + "/marked")))
    assert r.returncode != 0 and not os.path.exists(td + "/marked")
    marked["OUTCOME"] = "* **Outcome.** Test outcome."
    marked["STATUS729"] = "Test status."
    marked["R81"] = "**Test.** Item."
    marked["R83"] = "**Test.** Bullet."
    marked["LINE"] = "the test clause."
    json.dump(marked, open(td + "/ok.json", "w", encoding="utf-8"))
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v33.py", TIP, td + "/ok", "--fill", td + "/ok.json"], capture_output=True, env=ENV)
    assert r.returncode == 0, r.stderr[-400:]
    fm = open(td + "/ok/V8F_SCIENTIFIC_REPORT.md", encoding="utf-8", newline="").read()
    fh = open(td + "/ok/v8f_report.html", encoding="utf-8", newline="").read()
    say("== final mode with a complete test fill builds: no '@@' in md or html: %s | html == template + md: %s | 'PRE-REGISTERED, RUNNING' gone: %s | 'not read in this version' gone: %s" % (
        "@@" not in fm and fh.count("@@") == tpl.count("@@"), fh == tpl.replace("%%MD%%", fm), "RUNNING" not in fm.split("### 7.29 ")[1].split("\n")[0], "not read in this version" not in fm.split("### 7.29 ")[1]))
    assert "@@" not in fm and fh.count("@@") == tpl.count("@@") and fh == tpl.replace("%%MD%%", fm) and "RUNNING" not in fm.split("### 7.29 ")[1].split("\n")[0]
    assert "not read in this version" not in fm.split("### 7.29 ")[1]
    r2 = subprocess.run(["node", HERE + "render_check_v33.js", td + "/ok/v8f_report.html"], capture_output=True)
    say("   offline render of that test build: " + r2.stdout.decode().strip())
    assert r2.returncode == 0 and '"slots_rendered":0' in r2.stdout.decode()
say()

tok = "no SPEEDSUB number exists in this version"
say(f"== the v32 sentence '{tok}' is gone from md and html: {tok not in md and tok not in html}")
assert tok not in md and tok not in html
stale24 = [l.strip()[:120] for l in md.split("\n") if "the PREREG is a draft" in l or "its PREREG is a draft" in l]
say("== lines that still call the LAUNCH-INTENT PREREG a draft (a dated statement in 7.26, left in place on purpose, with the Update (v33) bullet above it): %d" % len(stale24))
for l in stale24:
    say("   " + l)
upd = [l for l in md.split("\n") if "Update (v33)" in l]
say("== lines carrying 'Update (v33)' (the dated additions that name the changed routing next to the v32 statements they qualify): %d" % len(upd))
for l in upd:
    say("   " + l[:130])
sites = {"status line": md.split("\n")[2], "abstract": md[md.index("## 1. Abstract"):md.index("## 2. Introduction")], "8.1": md[md.index("### 8.1 "):md.index("### 8.2 ")],
         "8.3 Update (v33)": md[md.index("**Update (v33).**"):md.index("**Update (v32).**")], "8.4 rows 1z / 1aa / 1ab": "\n".join(l for l in md.split("\n") if l.startswith(("| 1z |", "| 1aa |", "| 1ab |")))}
for name, text in sites.items():
    ok = ("SPEEDSUB" in text) and ("SPEEDSUB-T1" in text or name == "8.4 rows 1z / 1aa / 1ab") and ("LAUNCH-INTENT" in text)
    say(f"== SPEEDSUB, SPEEDSUB-T1 and LAUNCH-INTENT are all named in {name}: {ok}")
    assert ok, name
for name in ("abstract", "8.1", "8.3 Update (v33)"):
    t = sites[name]
    ok = ("PASS" in t and "NULL" in t)
    say(f"== the words PASS (SPEEDSUB) and NULL (SPEEDSUB-T1) are stated in {name}: {ok}")
    assert ok, name
say("== SPEEDSUB's PASS is written as a screening result and as resting on the declared channel in the abstract, 8.1 and 8.3: %s" % all("screening result" in sites[k] for k in ("abstract", "8.1", "8.3 Update (v33)")))
assert all("screening result" in sites[k] for k in ("abstract", "8.1", "8.3 Update (v33)"))
chan = all("D-SPEEDBAND-EGO-ECHO-1" in sites[k] or "declared" in sites[k] for k in ("abstract", "8.1", "8.3 Update (v33)"))
say("== the declared open channel (D-SPEEDBAND-EGO-ECHO-1) is named in the abstract, 8.1 and 8.3: %s" % chan)
assert chan
s84 = md[md.index("### 8.4 "):]
order84 = [l.split("|")[1].strip() for l in s84.split("\n") if l.startswith(("| 1y |", "| 1z |", "| 1aa |", "| 1ab |", "| 2 |"))]
say("== 8.4 has the rows 1z, 1aa, 1ab in order after 1y and before row 2: %s" % order84)
assert order84[:5] == ["1y", "1z", "1aa", "1ab", "2"]  # a later table of 8.x may start a row with '| 2 |' as well
say()
if not PREDRAFT:
    FIRST = "d8c9f7bfc91798b6fafb5a13dd793771e2f79f6a"
    same_blobs = all(git(["rev-parse", f"{FIRST}:{RP}{n}"]) == git(["rev-parse", f"{TIP}:{RP}{n}"]) for n in ("V8F_SCIENTIFIC_REPORT.md", "v8f_report.html", "page_template_v30.html"))
    say("== the three v32 report blobs (md, html, template) are the same at d8c9f7bf and at the tip %s (the v32 text the patch edits is the one the first draft was built on): %s" % (TIP[:8], same_blobs))
    assert same_blobs
    need29 = [
        ("the title names the word and the routing", "LAUNCH-INTENT: HARMS (arm A, W 1.58926, not adopted; M-LAUNCH NOT MOVED)" in md),
        ("HARMS for arm A, not adopted", "outcome **HARMS** (arm A, W 1.58926, not adopted)" in flat29 and "**Outcome: HARMS, as registered [M].** Arm A (W 1.58926) is not adopted." in flat29),
        ("all 206 reader controls PASS", "all **206 reader controls PASS**" in flat29),
        ("M-LAUNCH NOT MOVED routes section 11 to LAUNCH-INTENT-HI", "M-LAUNCH reads NOT MOVED" in flat29 and "The route is **LAUNCH-INTENT-HI**" in flat29),
        ("the committed next, exactly: B after HARMS with its own PI go, HI the section 11 branch with its own PI go, the PI is asked which, if either, box 2 runs",
         "**Both routes need a PI go: B, registered after HARMS, and HI, the branch M-LAUNCH selects. The PI is being asked which, if either, box 2 runs.**" in flat29),
        ("arm B is not trained and not read", "**Arm B (W 0.726175) was not trained and is not read.**" in flat29 and "B and HI are not read" in flat29),
        ("the pin-fill stamp is quoted only with its disclosure (filled about 13:50, before d8c9f7bf at 13:51:52)",
         "~14:10" in flat29 and "They were filled at ~13:50 (**[I]**, the Master Mind's brief), before commit `d8c9f7bf` (13:51:52)" in flat29),
        ("the launch facts are tagged MEASURED from the banked box-2 chain log", "(**[M]**, the banked box-2 chain log `chain_v8f_launchintent.out`" in flat29 and "**174 min or 2.9 h**" in flat29),
        ("the REP provenance: gate tokens MEASURED, the timing from the incident note INHERITED", "bind commit `8c24823f` (**[M]**" in flat29 and "`R2C_NOTE.md` (**[I]**" in flat29),
        ("Amendment 1 is disclosed (K-BUILD data-module clause, before any row was read)", "**Amendment 1 (`A1-2026-10-28`) was appended before any row was read" in flat29),
        ("T1-kin is a self-action open loop and the estimator is the paired episode-cluster bootstrap", "T1-kin, a self-action open loop" in flat29 and "paired episode-cluster bootstrap" in flat29),
        ("the replicate pair carries the training-run question (H-ESTIM-SEED-1)", "`H-ESTIM-SEED-1`" in flat29 and "another TRAINING RUN" in flat29),
    ]
    for name, ok in need29:
        say("== 7.29 final text: %s: %s" % (name, ok))
        assert ok, name
    needle = "MM" + "_STATE"
    in_added = [l[:90] for l in added if needle in l]
    patch_hits = [l.strip()[:90] for l in open(HERE + "patch_v33.py", encoding="utf-8").read().split("\n") if needle in l]
    leaks = [n for n in ("audit_v33.py", "mk_fill_v33.py", "audit_reviewed_v33.py", "FILL.json") if needle in open(HERE + n, encoding="utf-8").read()]
    carried = md.count(needle)
    say("== no launch fact of 7.29 is attributed to a Master Mind state note outside the repository (%s): in 7.29 %d, in the added or changed lines %d, in audit / fill scripts %s, in patch_v33.py only its own guard (%d line, an assert that the fill never carries it)" % (
        needle, S["7.29"].count(needle), len(in_added), leaks, len(patch_hits)))
    say("   the md still holds %d occurrence(s) of %s, all in v32 text of section 7.26 that is carried unchanged (the DAG-ORC launch time 03:50, tagged INHERITED there; not a LAUNCH-INTENT fact)" % (carried, needle))
    assert S["7.29"].count(needle) == 0 and not in_added and not leaks and len(patch_hits) == 1 and patch_hits[0].startswith("assert") and carried == v32.count(needle)
    say()
with tempfile.TemporaryDirectory(prefix="v33rebuild_") as td:
    r = subprocess.run([sys.executable, "-I", HERE + "patch_v33.py", TIP, td] + (["--predraft"] if PREDRAFT else []) + ([] if PREDRAFT else ["--fill", HERE + "FILL.json"]), capture_output=True, env=ENV)
    assert r.returncode == 0, r.stderr[-400:]
    same = open(td + "/V8F_SCIENTIFIC_REPORT.md", "rb").read() == open(md_p, "rb").read() and open(td + "/v8f_report.html", "rb").read() == open(html_p, "rb").read()
say("== patch_v33.py re-run on the v32 blobs%s reproduces both deliverables byte for byte: %s | %s" % ("" if PREDRAFT else " and FILL.json", same, r.stdout.decode().strip()))
assert same
if not PREDRAFT:
    with tempfile.TemporaryDirectory(prefix="v33fill_") as td2:
        r = subprocess.run([sys.executable, "-I", HERE + "mk_fill_v33.py", TIP, td2 + "/FILL.json"], capture_output=True, env=ENV)
        assert r.returncode == 0, r.stderr[-400:]
        same_fill = open(td2 + "/FILL.json", "rb").read() == open(HERE + "FILL.json", "rb").read()
    say("== mk_fill_v33.py re-run on the tip regenerates FILL.json byte for byte (the fill is generated from the repository blobs, not typed): %s" % same_fill)
    assert same_fill

say()
say("== offline render (the template's own pre-processing + local marked 16.4.2; v32 baseline first, no network)")
v32tmp = _stack.enter_context(tempfile.TemporaryDirectory(prefix="v33v32_")).replace("\\", "/") + "/v32.md"
open(v32tmp, "w", encoding="utf-8", newline="\n").write(v32)
rr = {}
for p in (v32tmp, md_p, html_p):
    r = subprocess.run(["node", HERE + "render_check_v33.js", p], capture_output=True)
    say("   " + r.stdout.decode().strip() + (r.stderr.decode()[-200:] if r.returncode else ""))
    assert r.returncode == 0
    rr[p] = json.loads(r.stdout.decode())
v32r, v33r, v33h = rr[v32tmp], rr[md_p], rr[html_p]
assert v33r["tables_with_ragged_rows"] == 0 and v33r["pre_blocks"] == 0 and v33r["stray_placeholders"] == 0 and v33h["tables"] == v33r["tables"] and v33h["h3"] == v33r["h3"] and v33h["math"] == v33r["math"]
assert PREDRAFT or v33r["slots_rendered"] == 0
say("Not run: a browser render (the page loads marked and MathJax from CDNs). The check above parses the md with the template's own pre-processing and a local marked;"
    " v32 had %d tables, %d h3, %d math spans; v33 has %d tables (+%d, copied line for line from the SPEEDSUB, SPEEDSUB-T1 and LAUNCH-INTENT results and the LAUNCH-INTENT PREREG), %d h3, %d math spans,"
    " no ragged table, no code block, no stray placeholder%s." % (v32r["tables"], v32r["h3"], v32r["math"], v33r["tables"], v33r["tables"] - v32r["tables"], v33r["h3"], v33r["math"],
                                                          ". The pre-draft's slot markers render as plain text" if PREDRAFT else " and no slot marker"))
open(CHECKS_PATH, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
_stack.close()
print("checks written")
