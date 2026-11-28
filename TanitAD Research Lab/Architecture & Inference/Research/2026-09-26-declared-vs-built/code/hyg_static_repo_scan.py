"""G-HYG static census, repo-wide: every `X.<...>.<name> = ...` whose chain looks like a config
(rooted at a cfg-like name, or passing through `.core/.encoder/.decoder/.anchors`) and whose
`<name>` is NOT a declared field of ANY of the 12 strict classes (read by AST from refc.py and
refc_v3.py -- no import, no torch). Each hit is then classified by hand in RESULT.md: a hit on
one of the 12 strict classes would RAISE at runtime after G-HYG.

    python hyg_static_repo_scan.py <tree> > raw/hyg_static_repo_scan.txt
"""
import ast
import pathlib
import sys

tree = pathlib.Path(sys.argv[1])
src = [tree / "stack/tanitad/refs/refc.py", tree / "stack/tanitad/refs/refc_v3.py"]
fields = {}
for p in src:
    t = ast.parse(p.read_text(encoding="utf-8"))
    for node in t.body:
        if isinstance(node, ast.ClassDef) and any(getattr(d, "id", "") == "strict_fields"
                                                  for d in node.decorator_list):
            fields[node.name] = {s.target.id for s in node.body
                                 if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
allf = set().union(*fields.values())
print(f"strict classes ({len(fields)}): {sorted(fields)}; declared names: {len(allf)}")
CFG_ROOTS = ("cfg", "c", "cfg_h", "cfg_f", "core", "enc", "dcfg", "config")
SUB = ("core", "encoder", "decoder", "anchors", "trajectory", "lan", "measurement", "strategic",
       "imagination", "law")
hits: dict = {}
n_files = 0
for root in ("stack", "taniteval"):
    for p in sorted((tree / root).rglob("*.py")):
        try:
            t = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        n_files += 1
        for node in ast.walk(t):
            tg = (node.targets if isinstance(node, ast.Assign) else
                  [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
            for x in tg:
                if not isinstance(x, ast.Attribute):
                    continue
                chain, cur = [], x
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                base = cur.id if isinstance(cur, ast.Name) else "?"
                if base == "self":
                    continue
                chain.reverse()
                mid, name = chain[:-1], chain[-1]
                cfgish = ((base in CFG_ROOTS and (not mid or mid[-1] in SUB))
                          or (mid and mid[-1] in ("core", "encoder", "decoder", "anchors")))
                if cfgish and name not in allf:
                    rel = p.relative_to(tree).as_posix()
                    hits.setdefault(name, []).append(f"{rel}:{node.lineno} {base}.{'.'.join(chain)}")
print(f"files parsed: {n_files}; candidate names: {len(hits)}")
for k, v in sorted(hits.items()):
    print(f"{k}: {len(v)}")
    for s in v:
        print(f"    {s}")
