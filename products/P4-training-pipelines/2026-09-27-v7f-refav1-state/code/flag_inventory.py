"""G-DVB flag inventory for a trainer: every argparse flag, its declared default, a PROPOSED lever class,
and whether its value is ever READ anywhere in the stack.

Why: the binding launch gate (CLAUDE.md "NO TRAINING LAUNCH WITHOUT A LAUNCH-GATE PASS"; SPEC_REFCV7 §2)
refuses any trainer flag without a G-DVB entry, and today it covers only refc_v3_train.py. This produces the
raw material for refa_v1_train.py and train_v6_staged.py. It does NOT decide anything: the class is a
name-pattern PROPOSAL for the gate owner, and "no read found" is a CANDIDATE for declared-but-inert, never proof
(a flag can be consumed through vars(args) or a config built from it).

Method (static, no import, so no heavy dependencies): AST-parse the trainer, collect every `add_argument(...)`
call (option strings, dest, default, type, choices, action, first help line); then count, for each dest,
  * attribute reads `<anything>.<dest>` and `getattr(<x>, "<dest>")` in the trainer, and
  * the same attribute reads in every other .py under stack/ (flags are often forwarded into configs by name).
⛔ Deliberate-regression arm (`--self-test`): a synthetic flag `--zz-inert-probe` is appended to a temp copy of the
trainer and must come back NO_READ_FOUND, while a known-consumed flag must come back READ. If either fails, the
instrument is inert and its output is void.

Usage: python flag_inventory.py --stack <stack dir> --trainer <path> --out <json> --md <md> [--self-test]
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CLASS_RULES = [  # first match wins; a PROPOSAL, never a ruling
    ("LOSS", r"^(w_|loss|lambda|lam_|weight_|.*_weight$|.*_w$)"),
    ("OPTIM", r"(^lr|_lr$|lr_|warmup|weight_decay|^wd$|clip|sched|^ema|beta|accum|^bs$|batch|^steps$|epochs?$|grad_)"),
    ("DATA", r"(cache|data|label|episode|window|stride|split|exclude|root|manifest|corpus|lru|nav_labels|s2_labels)"),
    ("RUNTIME", r"(^out|log|device|worker|save|ckpt|resume|dry|print|profile|verbose|tag|note|every$|_dir$|^allow_|^refuse_|precision|tf32|compile|seed$)"),
    ("MODEL", r"(size|depth|width|head|dim|d_model|trunk|encoder|arm$|readout|horizon|^n_|pos|anchor|init|stage|cond|predictor|target|vocab|grid|patch|in_channels|image|frame|k$|_k$)"),
]


def lever_class(dest: str) -> str:
    for cls, pat in CLASS_RULES:
        if re.search(pat, dest):
            return cls
    return "UNCLASSIFIED"


def _lit(node):
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except Exception:
        return "<expr> " + ast.unparse(node)[:80]


def collect_flags(src: str) -> list[dict]:
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        opts = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
        dynamic = not opts and bool(node.args)
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        longs = [o for o in opts if o.startswith("--")]
        dest = _lit(kw.get("dest")) if "dest" in kw else (
            longs[0][2:].replace("-", "_") if longs else (opts[0].lstrip("-").replace("-", "_") if opts else None))
        help_txt = _lit(kw.get("help"))
        out.append({
            "options": opts or ["<dynamic> " + ast.unparse(node.args[0])[:60]] if node.args else opts,
            "dest": dest if isinstance(dest, str) else None,
            "dynamic": dynamic,
            "default": _lit(kw.get("default")),
            "type": ast.unparse(kw["type"]) if "type" in kw else None,
            "choices": _lit(kw.get("choices")),
            "action": _lit(kw.get("action")),
            "help": (help_txt.splitlines()[0][:140] if isinstance(help_txt, str) and help_txt else None),
            "lineno": node.lineno,
        })
    return out


def attr_reads(src: str) -> Counter:
    """Count `<x>.<name>` loads and getattr(<x>, "<name>") across one source file."""
    c = Counter()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return c
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            c[node.attr] += 1
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
              and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
              and isinstance(node.args[1].value, str)):
            c[node.args[1].value] += 1
        elif isinstance(node, ast.Call):
            # INDIRECT reads: a flag name passed as a positional string to a helper that does the getattr
            # itself -- e.g. `resolve_gc(a, "enc_grad_checkpoint")` (train_v6_staged.py:5273). MEASURED
            # 2026-09-27: without this branch the first run reported 2 such flags as NO_READ_FOUND.
            fname = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else "")
            if fname != "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.isidentifier():
                        c[arg.value] += 1
    return c


def inventory(stack: Path, trainer: Path) -> dict:
    src = trainer.read_text(encoding="utf-8")
    flags = collect_flags(src)
    in_trainer = attr_reads(src)
    elsewhere = Counter()
    n_files = 0
    for f in stack.rglob("*.py"):
        if f.resolve() == trainer.resolve() or "__pycache__" in f.parts:
            continue
        elsewhere.update(attr_reads(f.read_text(encoding="utf-8", errors="replace")))
        n_files += 1
    rows = []
    for fl in flags:
        d = fl["dest"]
        r_t = in_trainer.get(d, 0) if d else 0
        r_e = elsewhere.get(d, 0) if d else 0
        status = ("DYNAMIC" if fl["dynamic"] or not d else
                  "READ" if (r_t + r_e) > 0 else "NO_READ_FOUND")
        rows.append({**fl, "reads_in_trainer": r_t, "reads_elsewhere_in_stack": r_e,
                     "status": status, "proposed_class": lever_class(d) if d else "UNCLASSIFIED"})
    return {"trainer": str(trainer), "stack": str(stack), "n_stack_files_scanned": n_files,
            "n_flags": len(rows), "flags": rows}


def self_test(stack: Path, trainer: Path, known_read: str) -> bool:
    src = trainer.read_text(encoding="utf-8")
    probe = ('\n\ndef _zz_probe_parser():\n    import argparse as _a\n    _p = _a.ArgumentParser()\n'
             '    _p.add_argument("--zz-inert-probe", type=float, default=1.0, help="never read")\n'
             '    return _p\n')
    with tempfile.TemporaryDirectory() as td:
        t = Path(td) / trainer.name
        t.write_text(src + probe, encoding="utf-8")
        inv = inventory(stack, t)
    by = {r["dest"]: r for r in inv["flags"] if r["dest"]}
    ok_probe = by.get("zz_inert_probe", {}).get("status") == "NO_READ_FOUND"
    ok_known = by.get(known_read, {}).get("status") == "READ"
    print("SELF-TEST: planted unread flag -> %s (want NO_READ_FOUND) | known flag %r -> %s (want READ)"
          % (by.get("zz_inert_probe", {}).get("status"), known_read, by.get(known_read, {}).get("status")))
    return ok_probe and ok_known


def to_md(inv: dict) -> str:
    rows = inv["flags"]
    cls = Counter(r["proposed_class"] for r in rows)
    st = Counter(r["status"] for r in rows)
    lines = ["# G-DVB flag inventory — `%s`" % Path(inv["trainer"]).name, "",
             "Static AST inventory (no import). **Class = name-pattern PROPOSAL; NO_READ_FOUND = candidate, not "
             "proof.** Stack files scanned for forwarded reads: %d." % inv["n_stack_files_scanned"], "",
             "**%d flags** · status %s · proposed class %s" % (
                 inv["n_flags"], dict(st), dict(cls)), "",
             "## ⛔ NO_READ_FOUND — parsed, never read anywhere in the stack (declared-but-inert CANDIDATES)", ""]
    nr = [r for r in rows if r["status"] == "NO_READ_FOUND"]
    lines += ["| flag | default | proposed class | line |", "|---|---|---|---|"]
    lines += ["| `%s` | `%s` | %s | %d |" % ("/".join(r["options"]), r["default"], r["proposed_class"], r["lineno"])
              for r in nr] or ["| (none) | | | |"]
    lines += ["", "## Every flag", "", "| flag | dest | default | class | reads (trainer / stack) | status |",
              "|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["proposed_class"], r["dest"] or "")):
        lines.append("| `%s` | `%s` | `%s` | %s | %d / %d | %s |" % (
            "/".join(r["options"])[:48], r["dest"], str(r["default"])[:40], r["proposed_class"],
            r["reads_in_trainer"], r["reads_elsewhere_in_stack"], r["status"]))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--known-read", default="lr", help="a flag the trainer certainly reads (self-test control)")
    a = ap.parse_args(argv)
    stack, trainer = Path(a.stack), Path(a.trainer)
    if a.self_test and not self_test(stack, trainer, a.known_read):
        print("SELF-TEST FAILED -- the instrument is inert; no inventory written")
        return 2
    inv = inventory(stack, trainer)
    Path(a.out).write_text(json.dumps(inv, indent=1, default=str), encoding="utf-8")
    Path(a.md).write_text(to_md(inv), encoding="utf-8")
    st = Counter(r["status"] for r in inv["flags"])
    print("%s: %d flags  %s" % (trainer.name, inv["n_flags"], dict(st)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
