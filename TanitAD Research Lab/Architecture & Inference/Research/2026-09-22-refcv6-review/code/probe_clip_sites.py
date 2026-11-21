#!/usr/bin/env python
"""Q3 — where does the gradient-clip max-norm actually live, and what is it?

SPEC_REFCV6_V2.md §10.7: "RL gradient clipping: **100, not 1.0**" —
MEASURED there that 1.0 binds on 600/600 steps.

⛔ THE PROGRAMME RULE THIS EXISTS FOR: a constant living at FIVE sites makes a
one-site change look applied while the chain stays blocked. So this does not
ask "is 100 present anywhere"; it ENUMERATES every site that supplies a
max-norm to ``clip_grad_norm_`` and reports the value at each, by AST — a
literal argument, an argparse default, or a dataclass field default.

Reported per site so a reader can see which consumer gets which number.
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib

ROOT = pathlib.Path(r"D:/Projects/TanitAD/stack")

#: the LITERAL from the SPEC, never an expression over the code
SPEC_RL_CLIP = 100.0


def _unparse(n) -> str:
    try:
        return ast.unparse(n)
    except Exception:                                    # pragma: no cover
        return "<unparseable>"


def scan(path: pathlib.Path) -> dict:
    """Every clip_grad_norm_ call-site, argparse default and dataclass field."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"calls": [], "argparse": [], "fields": [], "consts": [],
                "parse_error": True}
    calls, argp, fields, consts = [], [], [], []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = _unparse(n.func)
            if f.endswith("clip_grad_norm_"):
                arg = _unparse(n.args[1]) if len(n.args) > 1 else next(
                    (_unparse(k.value) for k in n.keywords
                     if k.arg == "max_norm"), "<none>")
                calls.append({"line": n.lineno, "params": (
                    _unparse(n.args[0]) if n.args else "?"), "max_norm": arg})
            if f.endswith("add_argument") and n.args:
                name = _unparse(n.args[0]).strip("'\"")
                if "clip" in name:
                    d = next((_unparse(k.value) for k in n.keywords
                              if k.arg == "default"), "<no default>")
                    argp.append({"line": n.lineno, "flag": name, "default": d})
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            if "clip" in n.target.id.lower() and n.value is not None:
                fields.append({"line": n.lineno, "name": n.target.id,
                               "default": _unparse(n.value)})
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and "CLIP" in t.id:
                    consts.append({"line": n.lineno, "name": t.id,
                                   "value": _unparse(n.value)})
    return {"calls": calls, "argparse": argp, "fields": fields,
            "consts": consts, "parse_error": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    out, n_files, unreadable = {}, 0, []
    for p in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        n_files += 1
        try:
            r = scan(p)
        except OSError as ex:                            # mount flap -> INCONCLUSIVE
            unreadable.append({"path": str(p), "err": repr(ex)})
            continue
        if any(r[k] for k in ("calls", "argparse", "fields", "consts")):
            out[str(p.relative_to(ROOT)).replace("\\", "/")] = r

    # ⚠️ SAME-BREATH CONTROL: a scan that read nothing must not look like
    # "no sites exist". A known site must be present or the run is INVALID.
    control = "scripts/refc_v3_train.py"
    if control not in out or not out[control]["calls"]:
        raise SystemExit(f"INVALID: control site {control} produced no "
                         f"clip_grad_norm_ call — the scan did not read. "
                         f"(files walked: {n_files}, unreadable: "
                         f"{len(unreadable)})")

    print(f"files walked: {n_files}   unreadable: {len(unreadable)}")
    print(f"SPEC §10.7 literal for the RL clip: {SPEC_RL_CLIP}\n")
    n_call = n_def = 0
    for f, r in sorted(out.items()):
        lines = []
        for c in r["calls"]:
            n_call += 1
            lines.append(f"    CALL      :{c['line']:<6} max_norm={c['max_norm']}"
                         f"   over {c['params']}")
        for c in r["argparse"]:
            n_def += 1
            lines.append(f"    ARGPARSE  :{c['line']:<6} {c['flag']} "
                         f"default={c['default']}")
        for c in r["fields"]:
            n_def += 1
            lines.append(f"    FIELD     :{c['line']:<6} {c['name']} = {c['default']}")
        for c in r["consts"]:
            n_def += 1
            lines.append(f"    CONST     :{c['line']:<6} {c['name']} = {c['value']}")
        if lines:
            print(f"  {f}")
            print("\n".join(lines))
    print(f"\n  call sites: {n_call}   value-supplying definitions: {n_def}")

    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(
            {"spec_rl_clip": SPEC_RL_CLIP, "files_walked": n_files,
             "unreadable": unreadable, "sites": out}, indent=2),
            encoding="utf-8")
        print(f"\n[artifact] {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
