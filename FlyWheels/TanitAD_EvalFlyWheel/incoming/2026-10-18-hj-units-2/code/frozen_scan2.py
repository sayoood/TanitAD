"""HJ-UNITS-2 frozen-caller scans over the mirror tip (read via `git show`, never a worktree).

(A) CLI: invocations of taniteval/tools/refav1_arm.py (rollout mode) that do not state --action-units.
(B) API: calls to RefAV1.plan / imagined_goal / _imagine_tactical_goal / refa_v1_plan.unicycle_paths that do
    not state the unit, in files outside the live roots.
ASCII-only stdout.
"""
import ast
import collections
import json
import re
import subprocess
import sys

TIP = "5fb509a"
REPO = "C:/Users/Admin/tanitad-push"
OUT = "C:/Users/Admin/hjunits2/out/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-hj-units-2/raw"
LIVE = ("stack/tanitad/", "stack/scripts/", "stack/tests/", "taniteval/taniteval/", "taniteval/tools/",
        "taniteval/tests/", "tools/", "Project Steering/BACKLOG.md", "CLAUDE.md")


def git(*a):
    return subprocess.run(["git", "-C", REPO, "-c", "core.autocrlf=false", *a], capture_output=True).stdout


def show(path):
    raw = git("show", f"{TIP}:{path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def is_live(path):
    return path.startswith(LIVE)


# ----------------------------------------------------------------------------- (A) CLI
files = [l.split(":", 1)[1] for l in git("grep", "-l", "-I", "refav1_arm", TIP, "--", ".").decode("utf-8", "replace").split("\n") if l]
SKIP_EXT = (".json", ".jsonl", ".log", ".html", ".csv", ".npz", ".txt", ".diff", ".patch")
interp = re.compile(r"(?:^|[\s/\\\"'=(])(?:python[0-9.]*(?:\.exe)?|py|\$PY|\$\{PY\}|\"\$PY\"|\$PYTHON|\"\$PYTHON\"|\$\{PYTHON\})\s+(?:-\S+\s+)*\S*refav1_arm\.py(?:\s|$|\"|')")
cli_rows = []


def logical_cmds(text):
    cur, start = [], 0
    for i, ln in enumerate(text.split("\n"), 1):
        if not cur:
            start = i
        st = ln.rstrip()
        if st.endswith("\\") or st.endswith("`"):
            cur.append(st[:-1])
            continue
        cur.append(st)
        yield start, " ".join(c.strip() for c in cur)
        cur = []
    if cur:
        yield start, " ".join(c.strip() for c in cur)


for f in files:
    if f.lower().endswith(SKIP_EXT):
        continue
    text = show(f)
    if f.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        # string-literal lists / tuples naming refav1_arm.py = an argv for subprocess
        for n in ast.walk(tree):
            if isinstance(n, (ast.List, ast.Tuple)):
                strs = []
                for e in ast.walk(n):
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        strs.append(e.value)
                if any(s.endswith("refav1_arm.py") for s in strs) and any(s.startswith("--") for s in strs):
                    mode = "analyze" if "--analyze-only" in strs else ("help" if "--help" in strs else "rollout")
                    cli_rows.append(dict(path=f, line=n.lineno, kind="py-argv", mode=mode,
                                         has_flag="--action-units" in strs,
                                         file_has_flag="action-units" in text, live=is_live(f)))
        # the same, built by string concatenation / f-strings on ONE line, or a 'cmd' string
        for i, ln in enumerate(text.split("\n"), 1):
            if interp.search(ln):
                joined = ln
                cli_rows.append(dict(path=f, line=i, kind="py-cmdstring", mode="analyze" if "--analyze-only" in joined else "rollout",
                                     has_flag="--action-units" in joined, file_has_flag="action-units" in text, live=is_live(f)))
        continue
    for ln, cmd in logical_cmds(text):
        if "refav1_arm.py" in cmd and interp.search(cmd):
            mode = "analyze" if "--analyze-only" in cmd else ("help" if "--help" in cmd else "rollout")
            cli_rows.append(dict(path=f, line=ln, kind="shell/doc", mode=mode, has_flag="--action-units" in cmd,
                                 file_has_flag="action-units" in text, live=is_live(f)))

roll = [r for r in cli_rows if r["mode"] == "rollout"]
bad = [r for r in roll if not r["has_flag"]]
bad_certain = [r for r in bad if not r["file_has_flag"]]
bad_maybe = [r for r in bad if r["file_has_flag"]]
print("CLI: invocations found:", len(cli_rows), "| rollout:", len(roll), "| analyze/help:", len(cli_rows) - len(roll))
print("CLI: rollout invocations WITHOUT --action-units in the command:", len(bad), "in", len({r['path'] for r in bad}), "files")
print("  of which the file never mentions action-units (will exit 2):", len(bad_certain), "in", len({r['path'] for r in bad_certain}), "files")
print("  of which the file mentions it elsewhere (flag may ride in a variable; UNVERIFIED):", len(bad_maybe), "in", len({r['path'] for r in bad_maybe}), "files")
print("CLI: LIVE rows without the flag:", [(r['path'], r['line']) for r in bad if r['live']])

# ----------------------------------------------------------------------------- (B) API
CAL = {"plan": "model_action_units", "imagined_goal": "model_action_units", "_imagine_tactical_goal": "units",
       "unicycle_paths": "action_units"}
pat = r"\.plan\(|imagined_goal|_imagine_tactical_goal|unicycle_paths"
pyfiles = sorted({l.split(":", 1)[1] for l in git("grep", "-l", "-I", "-E", pat, TIP, "--", "*.py").decode("utf-8", "replace").split("\n") if l})
api_rows = []
for f in pyfiles:
    src = show(f)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        api_rows.append(dict(path=f, kind="UNPARSEABLE", err=str(e)))
        continue
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    refs_v1 = bool(re.search(r"refa_v1|RefAV1|refav1_arm", src))
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
        if name not in CAL:
            continue
        kws = {k.arg for k in n.keywords}
        if name == "plan" and "v0" not in kws:
            continue                                  # another API's plan()
        omits = (None in kws) or (CAL[name] not in kws)
        api_rows.append(dict(path=f, line=n.lineno, callee=name, omits=omits, own_copy=name in defined,
                             refs_v1=refs_v1, live=is_live(f)))
calls = [r for r in api_rows if r.get("callee")]
omit = [r for r in calls if r["omits"]]
fz = [r for r in omit if not r["live"]]
fz_raise = [r for r in fz if not r["own_copy"] and r["refs_v1"]]
fz_own = [r for r in fz if r["own_copy"]]
fz_other = [r for r in fz if not r["own_copy"] and not r["refs_v1"]]
print("API: unit-bearing calls scanned:", len(calls), "| omit the unit:", len(omit))
print("API: FROZEN call sites that omit the unit:", len(fz), "in", len({r['path'] for r in fz}), "files")
print("  will RAISE against the live module (no own copy, file names refa_v1/RefAV1/refav1_arm):", len(fz_raise), "in", len({r['path'] for r in fz_raise}), "files")
print("  define their OWN copy of the callee (unaffected):", len(fz_own), "in", len({r['path'] for r in fz_own}), "files")
print("  other (no reference to the live module found; UNVERIFIED):", len(fz_other), "in", len({r['path'] for r in fz_other}), "files")
print("API: by callee (frozen, raising):", dict(collections.Counter(r['callee'] for r in fz_raise)))
live_omit = [r for r in omit if r["live"]]
print("API: LIVE call sites that omit the unit (tests included):", [(r['path'], r['line'], r['callee']) for r in live_omit])

json.dump(dict(tip=TIP, cli_rows=cli_rows, api_rows=api_rows), open("C:/Users/Admin/hjunits2/scan/frozen_scan2.json", "w"), indent=1)
import os
os.makedirs(OUT, exist_ok=True)
json.dump(dict(tip=TIP, cli=cli_rows, api=api_rows), open(OUT + "/frozen_callers_scan2.json", "w"), indent=1)
print("wrote", OUT + "/frozen_callers_scan2.json")
print("--- frozen CLI files (certain):")
for p in sorted({r['path'] for r in bad_certain}):
    print("  ", p)
print("--- frozen CLI files (maybe):")
for p in sorted({r['path'] for r in bad_maybe}):
    print("  ", p)
print("--- frozen API raising files:")
for p in sorted({r['path'] for r in fz_raise}):
    print("  ", p, sorted({(r['callee']) for r in fz_raise if r['path'] == p}))
