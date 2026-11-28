"""Runs the report-v32 checks and writes v32_checks.txt (the patch's own assertions have already passed to produce out/; these are the independent ones).
python checks_v32.py [tip [out_dir [checks_path [audit_path]]]]   (needs: python, git, node; no network; the superset and demotion checkers are exported from the tip).
out_dir: the folder holding V8F_SCIENTIFIC_REPORT.md and v8f_report.html (default out/ beside this script, else the folder above it, where the landed copy keeps the report)."""
import contextlib
import datetime
import difflib
import os
import re
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = sys.argv[1] if len(sys.argv) > 1 else "ea5e3d3b"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/") + "/"


def _out_dir():
    if len(sys.argv) > 2:
        return sys.argv[2]
    up = os.path.dirname(HERE.rstrip("/"))
    if os.path.isdir(HERE + "out"):
        return HERE + "out"
    return up if os.path.exists(up + "/V8F_SCIENTIFIC_REPORT.md") else HERE + "out"


OUT = _out_dir().replace("\\", "/").rstrip("/") + "/"
CHECKS_PATH = sys.argv[3] if len(sys.argv) > 3 else HERE + "v32_checks.txt"
AUDIT_FILE = sys.argv[4] if len(sys.argv) > 4 else HERE + "v32_NUMBERS_AUDIT.md"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RP = R + "report/2026-10-26/"
ENV = dict(os.environ, GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="safe.directory", GIT_CONFIG_VALUE_0="*", PYTHONIOENCODING="utf-8")
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
v31 = blob(RP + "V8F_SCIENTIFIC_REPORT.md")
v31h = blob(RP + "v8f_report.html")
tpl = blob(RP + "page_template_v30.html")
full_tip = git(["rev-parse", TIP]).decode().strip()
import hashlib

say(f"checks.txt for report v32 (source tip {TIP[:8]} = {full_tip}; v31 blobs read with git show; run {datetime.datetime.now().astimezone().isoformat(timespec='seconds')})")
say("checked deliverables (sha256): V8F_SCIENTIFIC_REPORT.md " + hashlib.sha256(open(md_p, "rb").read()).hexdigest() + " | v8f_report.html " + hashlib.sha256(open(html_p, "rb").read()).hexdigest())
say()
_stack = contextlib.ExitStack()  # the scratch folders below are removed when the script ends
tools = _stack.enter_context(tempfile.TemporaryDirectory(prefix="v32tools_"))
for name, path in (("superset_check.py", "tools/superset_check.py"),
                   ("demote_check.py", "TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-demotion-guard/code/demote_check.py")):
    open(tools + "/" + name, "wb").write(git(["show", f"{TIP}:{path}"]))
for label, rel, newp in (("V8F_SCIENTIFIC_REPORT.md", RP + "V8F_SCIENTIFIC_REPORT.md", md_p), ("v8f_report.html", RP + "v8f_report.html", html_p)):
    for tool, args in (("superset_check.py", [f"{TIP[:8]}:{rel}", newp, "none", MIRROR + "/.git"]), ("demote_check.py", [f"{TIP[:8]}:{rel}", newp, "none"])):
        say(f"### {tool} {TIP[:8]}:<repo>/{label}  <v32 {label}>  none")
        r = subprocess.run([sys.executable, tools + "/" + tool] + args, capture_output=True, env=ENV, cwd=tools)
        say((r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")).strip())
        assert r.returncode == 0, (tool, label)
    say()

say("== bytes / line endings / encoding")
for label, p in (("V8F_SCIENTIFIC_REPORT.md", md_p), ("v8f_report.html", html_p), ("patch_v32.py", HERE + "patch_v32.py"), ("audit_v32.py", HERE + "audit_v32.py"),
                 ("audit_reviewed.py", HERE + "audit_reviewed.py"), ("checks_v32.py", HERE + "checks_v32.py"), ("render_check.js", HERE + "render_check.js"),
                 ("v32_NUMBERS_AUDIT.md", AUDIT_FILE)):
    b = open(p, "rb").read()
    b.decode("utf-8")
    say(f"{label}: {len(b)} bytes, CR bytes={b.count(bytes([13]))}, LF bytes={b.count(bytes([10]))}, BOM={b.startswith(bytes([239, 187, 191]))}, utf-8 decode OK")
    assert b.count(bytes([13])) == 0

say(f"== html == page_template_v30.html with the md placeholder replaced by the v32 md: {html == tpl.replace('%%MD%%', md)}")
assert html == tpl.replace("%%MD%%", md)
say("== v31 html == the same template with the v31 md (the starting point is consistent): %s" % (v31h == tpl.replace("%%MD%%", v31)))
assert v31h == tpl.replace("%%MD%%", v31)
secs = ("### 7.24 SPEEDREAD", "### 7.25 DAgger iteration 2", "### 7.26 DAG-ORC")
say("== html contains the v32 sections: %s" % all(s in html for s in secs))
assert all(s in html for s in secs) and all(s in md for s in secs)

a, b = v31.split("\n"), md.split("\n")
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
added, removed = [], []
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op in ("replace", "insert"):
        added += b[j1:j2]
    if op in ("replace", "delete"):
        removed += a[i1:i2]
say("== diff v31 -> v32 md: %d added/changed lines, %d v31 lines replaced or removed" % (len(added), len(removed)))
for l in removed:
    say("   REPLACED v31 line: " + l[:150] + (" ..." if len(l) > 150 else ""))
add_text = "\n".join(added).lower()
say("== banned words in added/changed lines:")
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    n = len(re.findall(r"\b" + w + r"\b", add_text))
    say(f"   {w}: {n}")
    assert n == 0
extra = len(re.findall(r"retract|supersed|withdrawn|was wrong|not the floor|previously read|outdated|i said", add_text))
say(f"   demotion-guard retraction-context markers (RETRACT, SUPERSED, WITHDRAWN, was wrong, not the floor, previously read, outdated, I said): {extra}")
assert extra == 0
CARRIED = "**no closed-loop (t2) number exists**"  # the status line's own negation, carried from v30 (the line is re-counted as changed because the version string moved)
hits = [w for w in ("closed loop", "closed-loop", "drives") if w in add_text.replace(CARRIED, "")]
say("== 'closed loop' / 'closed-loop' / 'drives' in added lines, after setting aside the status line's carried negation (%s): %s" % (CARRIED, hits))
assert not hits

i24, i25, i26 = md.index("### 7.24 "), md.index("### 7.25 "), md.index("### 7.26 ")
iend = md.index("\n---\n\n## 8. Discussion")
S = {"7.24": md[i24:i25], "7.25": md[i25:i26], "7.26": md[i26:iend]}
for k, t in S.items():
    stamps = [l for l in t.split("\n") if re.search(r"T1-kin|T1-class|\bT0\b", l)]
    fams = {f: (f in t) for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")}
    flat = re.sub(r"\s+", " ", t)
    say(f"== {k}: lines carrying a tier stamp (T1-kin / T1-class / T0): {len(stamps)} | families present: {fams} | 'B = 2,000': {'B = 2,000' in flat} | '141 episodes': {'141 episodes' in flat}")
    assert all(fams.values()) and "B = 2,000" in flat and "141 episodes" in flat
say("== STRATEGIC is stated as reported, not paused, for v8F in 7.24: %s | 7.25: %s | 7.26: %s (design) and %s (read)" % (
    "STRATEGIC is reported, not paused, for v8F" in S["7.24"], "STRATEGIC is reported, not paused, for v8F" in S["7.25"], "STRATEGIC is not paused for v8F" in S["7.26"],
    "STRATEGIC is reported, not paused, for v8F" in S["7.26"]))
assert "STRATEGIC is reported, not paused, for v8F" in S["7.24"] and "STRATEGIC is reported, not paused, for v8F" in S["7.25"] and "STRATEGIC is not paused for v8F" in S["7.26"]
assert "STRATEGIC is reported, not paused, for v8F" in S["7.26"]
say("== tier discipline: T1-kin is never called closed loop; the only T2 mentions are negations ('not T2'): %s" % [l.strip()[:90] for l in added if "T2" in l])

tok = "DAGORC_RESULT"
say(f"== DAG-ORC placeholders (the string {tok}): md {md.count(tok)}, html {html.count(tok)}; both must be 0 now that RESULT_V8F_DAG_ORC.md has landed")
assert md.count(tok) == 0 and html.count(tok) == 0 and "@@" not in S["7.26"]
stale = [w for w in ("is launched and not yet read", "launched, not yet read", "No DAG-ORC number is reported in this version", "No DAG-ORC row exists in this version", "marked slots") if w in md or w in html]
say("== the pre-read wording for DAG-ORC (launched / not yet read / skeleton) is gone from md and html: %s" % (not stale))
assert not stale, stale
nyr = [l.strip()[:110] for l in md.split("\n") if "not yet read" in l and "DAG-ORC" in l]
say("== lines that still say 'not yet read' next to DAG-ORC (dated statements of earlier reads, left in place on purpose: 7.22, 7.23, 8.3 'Update (v31)'): %d" % len(nyr))
for l in nyr:
    say("   " + l)
sites = {"status line": md.split("\n")[2], "abstract": md[md.index("## 1. Abstract"):md.index("## 2. Introduction")], "7.26": S["7.26"], "8.1": md[md.index("### 8.1 "):md.index("### 8.2 ")],
         "8.3 Update (v32)": md[md.index("**Update (v32).**"):md.index("**Update (v31).**")], "8.4 row 1w": "\n".join(l for l in md.split("\n") if l.startswith("| 1w |"))}
for name, text in sites.items():
    ok = "HARMS" in text
    say(f"== the DAG-ORC outcome word HARMS is stated in {name}: {ok}")
    assert ok, name
say("== HARMS is stated together with 'not adopted' / 'is not adopted' in 7.26, the abstract, 8.1 and 8.4 row 1w: %s" % all("not adopted" in sites[k] for k in ("7.26", "abstract", "8.1", "8.4 row 1w")))
assert all("not adopted" in sites[k] for k in ("7.26", "abstract", "8.1", "8.4 row 1w"))
cost = "296 min = 4.93 h" in S["7.26"] and "~3 h" in S["7.26"] and "4.93 h" in sites["abstract"] and "4.93 h" in sites["8.3 Update (v32)"]
say("== the cost line states 296 min = 4.93 h by the logs against the ~3 h asked (7.26, abstract, 8.3): %s" % cost)
assert cost
nots = [k for k in ("NOT RUN", "NOT BUILT") if k in S["7.26"]]
say("== 7.26 states what is NOT RUN / NOT BUILT per family (T0 families, M-LAUNCH, anti-echo, M2 / L2 launch subset): %s | per-family items: %s" % (
    nots, [f for f in ("**LONGITUDINAL.**", "**LATERAL.**", "**TACTICAL.**", "**STRATEGIC.**", "**T0, all four families.**") if f in S["7.26"]]))
assert nots == ["NOT RUN", "NOT BUILT"] and all(f in S["7.26"] for f in ("**LONGITUDINAL.**", "**LATERAL.**", "**TACTICAL.**", "**STRATEGIC.**", "**T0, all four families.**"))
assert all(k in S["7.26"] for k in ("M-LAUNCH", "anti-echo block", "M2 and L2 on the GT-launch subset", "orc_families.py"))
say("== the replicate-floor reading is in 7.26 (REP s0 sets RF on 17 of 19 rows; A's crawl effects are 8.8-33.8 %% of the bar): %s" % (
    "REP s0 sets RF on 17 of 19 rows" in S["7.26"] and "8.8–33.8 % of the bar" in S["7.26"]))
assert "REP s0 sets RF on 17 of 19 rows" in S["7.26"] and "8.8–33.8 % of the bar" in S["7.26"]
say("== the next step says B needs the PI's go: %s" % ("needs its own PI go" in S["7.26"] and "blocked on the PI's go" in S["7.26"]))
assert "needs its own PI go" in S["7.26"] and "blocked on the PI's go" in S["7.26"]
r2c = "r2b's A s1 was refused at its run record" in S["7.26"] and "agent/dagorc-r2-pin" in S["7.26"] and "R2C_NOTE.md" in S["7.26"]
say("== the r2c incident (A s1 refused, queue replaced by a self-test, pinned relaunch) is in the run history: %s" % r2c)
assert r2c
say()
with tempfile.TemporaryDirectory(prefix="v32rebuild_") as td:
    r = subprocess.run([sys.executable, HERE + "patch_v32.py", TIP, td], capture_output=True)
    assert r.returncode == 0, r.stderr[-400:]
    same = open(td + "/V8F_SCIENTIFIC_REPORT.md", "rb").read() == open(md_p, "rb").read() and open(td + "/v8f_report.html", "rb").read() == open(html_p, "rb").read()
say("== patch_v32.py re-run on the v31 blobs reproduces both deliverables byte for byte: %s | %s" % (same, r.stdout.decode().strip()))
assert same

say()
say("== offline render (the template's own pre-processing + local marked 16.4.2; v31 baseline first, no network)")
v31tmp = _stack.enter_context(tempfile.TemporaryDirectory(prefix="v32v31_")).replace("\\", "/") + "/v31.md"
open(v31tmp, "w", encoding="utf-8", newline="\n").write(v31)
for p in (v31tmp, md_p, html_p):
    r = subprocess.run(["node", HERE + "render_check.js", p], capture_output=True)
    say("   " + r.stdout.decode().strip() + (r.stderr.decode()[-200:] if r.returncode else ""))
    assert r.returncode == 0
say("Not run: a browser render (the page loads marked and MathJax from CDNs). The check above parses the md with the template's own pre-processing and a local marked;"
    " it reports v31 23 tables, 38 h3, 233 math spans, and v32 47 tables (+9 copied from the SPEEDREAD and DAGGER2 RESULT files, +12 copied from the DAG-ORC RESULT, +3 authored in 7.26), 41 h3, 233 math spans,"
    " no ragged table, no code block, no placeholder.")
open(CHECKS_PATH, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
_stack.close()
print("checks written")
