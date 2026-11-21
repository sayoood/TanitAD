"""Batch 3 (b) -- a static census BEFORE decorating the six config classes the launch gate's
G-HYG probe found non-strict (AgentSeamConfig, DiffusionFlags, EgoHistoryConfig, MaxSpeedConfig,
Refcv7HeadConfig, TacticalDecoderConfig). Once decorated, any assignment of an UNDECLARED name
on an instance RAISES, so every such assignment in the repo must be found first.

Two routes to an instance, both scanned by AST (no import, no torch):
  1. a chain through the holder field: `<...>.agents.X`, `.refcv6.X`, `.ego_history.X`,
     `.max_speed_cfg.X`, `.tac_decoder_cfg.X`, `.refcv7_head_cfg.X`;
  2. a local name bound to an instance -- from the class constructor (`Cls(...)`, `mod.Cls(...)`),
     from a holder attribute (`x = cfg.core.agents`) or from `dataclasses.replace(<that>)` --
     then `name.X = ...` / `setattr(name, "X", ...)`.
A hit is `X` not a DECLARED field of the class the route implies (fields read by AST from the
class body). Output: one line per hit; `0 hits` is only admissible with the parse count beside it.

    python hyg_scan_b3.py <tree>
"""
import ast
import pathlib
import sys

TREE = pathlib.Path(sys.argv[1])
CLASSES = {"AgentSeamConfig": "stack/tanitad/refs/refc_agents.py",
           "DiffusionFlags": "stack/tanitad/models/refcv6_diffusion.py",
           "EgoHistoryConfig": "stack/tanitad/models/ego_history.py",
           "MaxSpeedConfig": "stack/tanitad/refs/max_speed_input.py",
           "Refcv7HeadConfig": "stack/tanitad/refs/refcv7_heads.py",
           "TacticalDecoderConfig": "stack/tanitad/refs/refcv6_tactical.py"}
HOLDER = {"agents": "AgentSeamConfig", "refcv6": "DiffusionFlags", "ego_history": "EgoHistoryConfig",
          "max_speed_cfg": "MaxSpeedConfig", "refcv7_head_cfg": "Refcv7HeadConfig",
          "tac_decoder_cfg": "TacticalDecoderConfig"}

FIELDS: dict = {}
for cls, rel in CLASSES.items():
    t = ast.parse((TREE / rel).read_text(encoding="utf-8"))
    for node in t.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            FIELDS[cls] = {s.target.id for s in node.body
                           if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
            FIELDS[cls] |= {s.name for s in node.body if isinstance(s, ast.FunctionDef)
                            and any(ast.unparse(d) == "property" for d in s.decorator_list)}
assert set(FIELDS) == set(CLASSES), FIELDS.keys()


def chain_of(node):
    out = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    base = node.id if isinstance(node, ast.Name) else None
    return base, list(reversed(out))


def cls_of_value(v, env):
    """The config class an expression evaluates to, or None."""
    if isinstance(v, ast.Call):
        f = v.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if name in CLASSES:
            return name
        if name == "replace" and v.args:
            return cls_of_value(v.args[0], env)
        return None
    base, ch = chain_of(v)
    if ch and ch[-1] in HOLDER:
        return HOLDER[ch[-1]]
    if base in env and not ch:
        return env[base]
    return None


hits, n_files, n_assign_seen = [], 0, 0
for root in ("stack", "taniteval", "tools"):
    for p in sorted((TREE / root).rglob("*.py")):
        try:
            t = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        n_files += 1
        rel = p.relative_to(TREE).as_posix()
        for fn in [t] + [n for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            env: dict = {}
            body_nodes = list(ast.walk(fn)) if fn is not t else [n for n in t.body]
            # pass 1: local names bound to an instance (per function scope, order-insensitive)
            for n in body_nodes:
                if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                    c = cls_of_value(n.value, env)
                    if c:
                        env[n.targets[0].id] = c
            # pass 2: assignments
            for n in body_nodes:
                tg = (n.targets if isinstance(n, ast.Assign) else
                      [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else [])
                for x in tg:
                    if not isinstance(x, ast.Attribute):
                        continue
                    n_assign_seen += 1
                    base, ch = chain_of(x)
                    name = ch[-1]
                    owner = None
                    if len(ch) >= 2 and ch[-2] in HOLDER:
                        owner = HOLDER[ch[-2]]
                    elif len(ch) == 1 and base in env:
                        owner = env[base]
                    if owner and name not in FIELDS[owner]:
                        hits.append(f"{rel}:{n.lineno} {owner}: {base}.{'.'.join(ch)}")
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "setattr" \
                        and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant):
                    owner = cls_of_value(n.args[0], env)
                    if owner is None:
                        base, ch = chain_of(n.args[0])
                        owner = HOLDER.get(ch[-1]) if ch else env.get(base)
                    if owner and n.args[1].value not in FIELDS[owner]:
                        hits.append(f"{rel}:{n.lineno} {owner}: setattr(..., {n.args[1].value!r})")

print(f"classes: {sorted(FIELDS)}")
print(f"files parsed: {n_files}; attribute assignments seen: {n_assign_seen}")
print(f"hits (an assignment of an UNDECLARED name on one of the six): {len(sorted(set(hits)))}")
for h in sorted(set(hits)):
    print("  " + h)
