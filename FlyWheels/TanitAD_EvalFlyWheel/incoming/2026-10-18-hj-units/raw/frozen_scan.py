import ast, subprocess, json, sys, re, collections
TIP = "ac07b1a"
REPO = "C:/Users/Admin/tanitad-push"
CALLEES = {"paths_from_controls": {"action_units"}, "ha0_ext_paths": {"action_units"},
           "const_control_paths": {"action_units", "steer_units"}, "integrate_select": {"action_units"}}
LIVE_PREFIXES = ("stack/tanitad/", "stack/scripts/", "stack/tests/", "taniteval/taniteval/", "taniteval/tools/", "taniteval/tests/", "tools/")
def git(*a, text=True):
    return subprocess.run(["git", "-C", REPO, "-c", "core.autocrlf=false", *a], capture_output=True, text=text, encoding="utf-8" if text else None, errors="replace").stdout
pat = "paths_from_controls|ha0_ext_paths|const_control_paths|integrate_select"
files = sorted({l.split(":",2)[1] for l in git("grep", "-l", "-E", pat, TIP, "--", "*.py").splitlines() if l})
files = [f.split(":",1)[-1] if f.startswith(TIP+":") else f for f in files]
# git grep -l output with a tree-ish is "ac07b1a:path"
res = []
for line in git("grep", "-l", "-E", pat, TIP, "--", "*.py").splitlines():
    path = line.split(":", 1)[1]
    res.append(path)
res = sorted(set(res))
rows = []
for path in res:
    src = git("show", f"{TIP}:{path}")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        rows.append(dict(path=path, kind="UNPARSEABLE", err=str(e))); continue
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    refs_live = bool(re.search(r"taniteval[/\\\\\"']+tools|refav1_arm|floors_kin|refcv3_arm", src))
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if name in CALLEES:
                kws = {k.arg for k in n.keywords}
                bad = (None in kws) or not (kws & CALLEES[name])
                rows.append(dict(path=path, line=n.lineno, callee=name, omits_unit=bad,
                                 defines_own_copy=name in defined, refs_live_module=refs_live,
                                 live=path.startswith(LIVE_PREFIXES)))
json.dump(rows, open("C:/Users/Admin/hjunits/out/raw/frozen_callers_scan.json", "w"), indent=1)
frozen = [r for r in rows if r.get("callee") and not r["live"]]
omit = [r for r in frozen if r["omits_unit"]]
def summarize(rs, label):
    byc = collections.Counter(r["callee"] for r in rs)
    files = {r["path"] for r in rs}
    print(f"{label}: {len(rs)} call sites in {len(files)} files; by callee {dict(byc)}")
summarize(frozen, "frozen files, all calls to the four units-bearing callees")
summarize(omit, "frozen files, calls that OMIT the unit")
own = [r for r in omit if r["defines_own_copy"]]
summarize(own, "  of which the file defines its OWN copy of the callee (unaffected by the live change)")
rest = [r for r in omit if not r["defines_own_copy"]]
summarize(rest, "  of which call something it does not define (resolve to live module if imported)")
rest_live = [r for r in rest if r["refs_live_module"]]
summarize(rest_live, "    and the file names refav1_arm/floors_kin/refcv3_arm/taniteval/tools (will RAISE if run against the live module)")
print("---- list (omit, not own copy):")
for r in sorted(rest, key=lambda r:(r["path"], r["line"])):
    print(f'{r["callee"]:22s} {r["path"]}:{r["line"]}  refs_live={r["refs_live_module"]}')
live_omit = [r for r in rows if r.get("callee") and r["live"] and r["omits_unit"]]
print("LIVE call sites that omit the unit (excluding tests):", [(r["path"], r["line"], r["callee"]) for r in live_omit if "/tests/" not in r["path"]])
print("LIVE TESTS that omit the unit:", [(r["path"], r["line"], r["callee"]) for r in live_omit if "/tests/" in r["path"]])
