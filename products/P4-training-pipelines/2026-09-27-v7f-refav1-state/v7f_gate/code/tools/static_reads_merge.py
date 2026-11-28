"""Per-dest args read sites WITH the enclosing function (trainer + helper functions)."""
import ast, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:/Users/Admin/v7f_gate/tree_m/stack")
TRAINER = ROOT / "scripts" / "train_v6_staged.py"
HELPERS = [(ROOT / "scripts" / "train_v58f_unicycle_head.py", {"build_train_episodes": "a"}),
           (ROOT / "scripts" / "train_flagship_v4.py", {"resolve_v2_frames": "a"}),
           (ROOT / "scripts" / "eval_flagship_v4.py", {"resolve_eval_frames": "a"}),
           (ROOT / "tanitad" / "geometry.py", {"apply_geometry_args": "args", "frame_from_args": "args"})]
dests = sorted({r["dest"] for r in json.load(open(sys.argv[1]))})
DS = set(dests)
sites = defaultdict(list)


def fn_map(tree):
    """line -> innermost enclosing top-level-or-nested function name."""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append((node.lineno, node.end_lineno, node.name))
    def enclosing(line):
        best = None
        for s, e, n in spans:
            if s <= line <= e and (best is None or s >= best[0]):
                best = (s, e, n)
        return best[2] if best else "<module>"
    return enclosing


def scan(node, argnames, where, enclosing):
    for sub in ast.walk(node):
        hit = None
        if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load) \
                and isinstance(sub.value, ast.Name) and sub.value.id in argnames and sub.attr in DS:
            hit = sub.attr
        elif isinstance(sub, ast.Call):
            f = sub.func
            if isinstance(f, ast.Name) and f.id in ("getattr", "hasattr") and len(sub.args) >= 2 \
                    and isinstance(sub.args[0], ast.Name) and sub.args[0].id in argnames \
                    and isinstance(sub.args[1], ast.Constant) and sub.args[1].value in DS:
                hit = sub.args[1].value
            elif sub.args and isinstance(sub.args[0], ast.Name) and sub.args[0].id in argnames:
                for x in sub.args[1:]:
                    if isinstance(x, ast.Constant) and isinstance(x.value, str) and x.value in DS:
                        sites[x.value].append((where, sub.lineno, enclosing(sub.lineno) + f"->{ast.unparse(f)}"))
        if hit:
            sites[hit].append((where, sub.lineno, enclosing(sub.lineno)))


t = ast.parse(TRAINER.read_text(encoding="utf-8"))
scan(t, {"a", "args", "_a", "ns"}, "T", fn_map(t))
for path, fns in HELPERS:
    tt = ast.parse(path.read_text(encoding="utf-8"))
    enc = fn_map(tt)
    for node in ast.walk(tt):
        if isinstance(node, ast.FunctionDef) and node.name in fns:
            scan(node, {fns[node.name]}, path.stem, enc)
out = {}
for d in dests:
    fns = defaultdict(int)
    for w, ln, fn in sites.get(d, []):
        fns[f"{w}:{fn}"] += 1
    out[d] = {"n_sites": len(sites.get(d, [])), "by_function": dict(sorted(fns.items())),
              "lines": [f"{w}:{ln}" for w, ln, _ in sites.get(d, [])][:12]}
Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
ECHO = {"T:_launch_line", "T:_run_config"}
PRE = {"T:preflight"}
for d in dests:
    f = set(out[d]["by_function"])
    core = {x.split("->")[0] for x in f}
    if core <= (ECHO | PRE) or not core:
        print(f"{d:34s} {sorted(f)}")
