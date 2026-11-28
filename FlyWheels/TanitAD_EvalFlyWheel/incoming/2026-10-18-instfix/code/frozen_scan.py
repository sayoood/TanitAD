r"""INSTFIX frozen/live CLI-caller scan over the mirror tip (read via `git show`, never a worktree).

usage: frozen_scan.py <tip> <script-basename> [--scope REGEX] [--list] [--json OUT]

Finds invocations of `<script>` (run as a CLI) that do not state `--action-units`:
  * shell / ps1 / cmd / md : logical commands (\ ` ^ continuations joined) that run the script under a python interpreter
  * .py, direct            : a list/tuple of string literals naming the script plus a `--flag`
  * .py, via a variable    : `NAME = ... "<script>" ...` then a list/tuple containing NAME plus a `--flag`
`--scope REGEX` limits to commands matching it (e.g. "--family[ =]refav1" for actdiv_anchored, where v7 has no unit).
An invocation is EXEMPT when it carries --analyze-only / --arm-json / --help / --selftest (never reads the unit).
It is a LOWER BOUND: a command built from an unresolvable variable is not counted.  ASCII-only stdout.
"""
import ast
import json
import re
import subprocess
import sys
import warnings

warnings.simplefilter("ignore")          # banked files with bad escapes raise SyntaxWarning in ast.parse

TIP = sys.argv[1]
script = sys.argv[2]
scope = sys.argv[sys.argv.index("--scope") + 1] if "--scope" in sys.argv else None
REPO = "C:/Users/Admin/tanitad-push"
LIVE = ("stack/tanitad/", "stack/scripts/", "stack/tests/", "taniteval/taniteval/", "taniteval/tools/",
        "taniteval/tests/", "tools/", "Project Steering/BACKLOG.md", "CLAUDE.md")
EXEMPT = ("--analyze-only", "--arm-json", "--help", "--selftest")
FLAG = "--action-units"


def git(*a):
    return subprocess.run(["git", "-C", REPO, "-c", "core.autocrlf=false", *a], capture_output=True).stdout


def show(path):
    raw = git("show", f"{TIP}:{path}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def is_live(p):
    return p.startswith(LIVE)


files = [l.split(":", 1)[1] for l in git("grep", "-l", "-I", script, TIP, "--", ".").decode("utf-8", "replace").split("\n") if l]
SKIP_EXT = (".json", ".jsonl", ".log", ".html", ".csv", ".npz", ".txt", ".diff", ".patch", ".xml", ".tsv")
interp = re.compile(r"(?:^|[\s/\\\"'=(&])(?:python[0-9.]*(?:\.exe)?|py|\$PY|\$\{PY\}|\"\$PY\"|%PY%|\"%PY%\"|\$py|\$PYTHON|\$\{PYTHON\}|\"\$PYTHON\")\s+(?:-\S+\s+)*[\"']?\S*"
                    + re.escape(script) + r"[\"']?(?:\s|$|\"|')")
amp = re.compile(r"&\s*\"?[^\"\n]*" + re.escape(script))


def logical(text):
    cur, start = [], 0
    for i, ln in enumerate(text.split("\n"), 1):
        if not cur:
            start = i
        st = ln.rstrip()
        if st.endswith("\\") or st.endswith("`") or st.endswith("^"):
            cur.append(st[:-1])
            continue
        cur.append(st)
        yield start, " ".join(c.strip() for c in cur)
        cur = []
    if cur:
        yield start, " ".join(c.strip() for c in cur)


rows = []


def add(path, line, kind, text_cmd, file_text):
    if scope and not re.search(scope, text_cmd):
        return
    exempt = any(x in text_cmd for x in EXEMPT)
    rows.append(dict(path=path, line=line, kind=kind, exempt=exempt, has_flag=FLAG in text_cmd,
                     file_has_flag=FLAG in file_text, live=is_live(path)))


for f in files:
    if f.lower().endswith(SKIP_EXT):
        continue
    text = show(f)
    if f.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                if any(isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value.endswith(script) for e in ast.walk(n.value)):
                    names.add(n.targets[0].id)
        for n in ast.walk(tree):
            if isinstance(n, (ast.List, ast.Tuple)):
                strs = [e.value for e in ast.walk(n) if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                nm = {e.id for e in ast.walk(n) if isinstance(e, ast.Name)}
                direct = any(s.endswith(script) for s in strs)
                via = bool(nm & names)
                if (direct or via) and any(s.startswith("--") for s in strs):
                    add(f, n.lineno, "py-argv" if direct else "py-argv-via-var", " ".join(strs), text)
        for i, ln in enumerate(text.split("\n"), 1):
            if interp.search(ln):
                add(f, i, "py-cmdstring", ln, text)
        continue
    for ln, cmd in logical(text):
        if script in cmd and (interp.search(cmd) or amp.search(cmd)):
            add(f, ln, "shell/doc", cmd, text)

run = [r for r in rows if not r["exempt"]]
bad = [r for r in run if not r["has_flag"]]
print(f"{script} (scope={scope}) @ {TIP}: invocations {len(rows)} | unit-reading {len(run)} | WITHOUT --action-units {len(bad)} "
      f"in {len({r['path'] for r in bad})} files")
print("  of which the file never mentions the flag (will exit 2):", len([r for r in bad if not r["file_has_flag"]]),
      "| mentions it elsewhere (UNVERIFIED):", len([r for r in bad if r["file_has_flag"]]))
print("  LIVE unit-reading rows WITHOUT the flag:", [(r["path"], r["line"]) for r in bad if r["live"]])
if "--list" in sys.argv:
    for r in sorted(bad, key=lambda r: (r["path"], r["line"])):
        print("  OMIT", r["path"], r["line"], r["kind"], "(flag elsewhere in file)" if r["file_has_flag"] else "")
if "--json" in sys.argv:
    json.dump(dict(tip=TIP, script=script, scope=scope, rows=rows), open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
