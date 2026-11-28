#!/usr/bin/env python3
"""UNBOUNDED inventory of every ``_jack_*`` / ``heldout`` / ``overlapping_holdout_se``
site in the repo, classified DECIDES / REPORTS / DEAD.

⚠️ C69: an absence claim was made from ``find -maxdepth 4`` when the files sat at
depth 6. This walks the whole tree from the repo root with no depth limit and
records the max depth actually reached, so the coverage is in the artifact.

Classification, per FILE (a file is the unit a reader promotes a number from):

  DECIDES  the AST gate guard (``taniteval.gate_guard``) finds a verdict-carrying
           key whose value is data-dependent on a banned estimator call.
  REPORTS  the banned estimator is CALLED, but no verdict reads it.
  DEFINES  the banned estimator is DEFINED here and never called in this file.
  DEAD     the name appears only in a string/comment/docstring — i.e. an AST walk
           sees no call at all. (This is the class a regex guard would have
           reported as a violation: it matches its own documentation.)
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "stack"))
sys.path.insert(0, str(ROOT / "taniteval"))

from taniteval import gate_guard as gg          # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "raw"
SKIP_DIRS = ("/.git/", "/__pycache__/", "/.claude/worktrees/", "/node_modules/")
TEXT_RE = re.compile(r"_jack_|\bheldout\b|overlapping_holdout_se|jackknife")
# The enforced scope of the CI guard (mirrors tests/test_no_jack_in_gates.py).
ENFORCED = ("/taniteval/taniteval/", "/taniteval/tools/", "/stack/tanitad/",
            "/stack/scripts/")


def calls_in(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            b = gg.is_banned_callable(n)
            if b:
                out.append((n.lineno, b))
    return out


def defs_in(tree):
    return [(n.lineno, n.name) for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (n.name in gg.BANNED_CALLS or gg.BANNED_CALL_RE.match(n.name))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, depths = [], []
    n_scanned = 0
    for p in ROOT.rglob("*"):
        pp = p.as_posix()
        if not p.is_file() or any(s in pp for s in SKIP_DIRS):
            continue
        if p.suffix.lower() not in (".py", ".md", ".json", ".txt", ".ipynb",
                                    ".env", ".sh", ".yaml", ".yml"):
            continue
        n_scanned += 1
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = [(i + 1, ln.strip()[:180])
                for i, ln in enumerate(src.splitlines()) if TEXT_RE.search(ln)]
        if not hits:
            continue
        rel = p.relative_to(ROOT).as_posix()
        depths.append(len(p.relative_to(ROOT).parts))
        row = {"path": rel, "suffix": p.suffix.lower(), "n_text_hits": len(hits),
               "depth": len(p.relative_to(ROOT).parts),
               "in_ci_enforced_scope": any(e in "/" + rel for e in ENFORCED)}
        if p.suffix.lower() == ".py":
            try:
                tree = ast.parse(src, filename=rel)
            except SyntaxError:
                row["klass"] = "UNPARSEABLE"
                rows.append(row)
                continue
            calls = calls_in(tree)
            defs = defs_in(tree)
            viol = gg.scan_source(src, rel)
            row["banned_calls"] = [{"line": ln, "fn": fn} for ln, fn in calls]
            row["banned_defs"] = [{"line": ln, "fn": fn} for ln, fn in defs]
            row["violations"] = [{"line": v[1], "key": v[2], "why": v[3]}
                                 for v in viol]
            row["klass"] = ("DECIDES" if viol else
                            "REPORTS" if calls else
                            "DEFINES" if defs else "DEAD")
            if row["klass"] == "DEAD":
                # ⚠️ Not all DEAD files are about the estimator at all: `heldout`
                # is an ordinary English word (`stack/tanitad/train/heldout_gate.py`
                # is a TRAINING held-out gate). Split them, or the headline count
                # overstates the blast radius.
                toks = set()
                if re.search(r"_jack_|overlapping_holdout_se", src):
                    toks.add("estimator_named_in_text_only")
                if re.search(r"\bjackknife\b", src):
                    toks.add("jackknife_word_only")
                if re.search(r"\bheldout\b", src) and not toks:
                    toks.add("heldout_word_unrelated")
                row["dead_reason"] = sorted(toks) or ["heldout_word_unrelated"]
        else:
            row["klass"] = "PROSE_OR_ARTIFACT"
            row["sample"] = hits[:3]
        rows.append(row)

    py = [r for r in rows if r["suffix"] == ".py"]
    res = {
        "_what": "unbounded repo-wide inventory of the banned-estimator family",
        "_root": str(ROOT),
        "_coverage": {"files_scanned": n_scanned,
                      "files_with_a_textual_hit": len(rows),
                      "max_path_depth_reached": max(depths) if depths else 0,
                      "note": "rglob from the repo root; NO maxdepth (C69)"},
        "_classes": Counter(r["klass"] for r in rows),
        "_python_classes": Counter(r["klass"] for r in py),
        "_totals": {
            "banned_call_sites": sum(len(r.get("banned_calls", [])) for r in py),
            "banned_definition_sites": sum(len(r.get("banned_defs", [])) for r in py),
            "deciding_violations": sum(len(r.get("violations", [])) for r in py),
        },
        "DECIDES": [r for r in py if r["klass"] == "DECIDES"],
        "REPORTS": sorted([r["path"] for r in py if r["klass"] == "REPORTS"]),
        "DEFINES": sorted([r["path"] for r in py if r["klass"] == "DEFINES"]),
        "DEAD": sorted([r["path"] for r in py if r["klass"] == "DEAD"]),
        "_dead_reasons": dict(Counter(
            t for r in py if r["klass"] == "DEAD" for t in r["dead_reason"])),
        "_dead_estimator_named_in_text_only": sorted(
            r["path"] for r in py if r["klass"] == "DEAD"
            and "estimator_named_in_text_only" in r["dead_reason"]),
        "PROSE_OR_ARTIFACT": sorted([r["path"] for r in rows
                                     if r["klass"] == "PROSE_OR_ARTIFACT"]),
        "rows": rows,
    }
    res["_classes"] = dict(res["_classes"])
    res["_python_classes"] = dict(res["_python_classes"])
    p = OUT / "jack_site_inventory.json"
    p.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("_coverage", "_classes", "_python_classes", "_totals")},
                     indent=2))
    print("DECIDES:", [(r["path"], r["violations"]) for r in res["DECIDES"]])
    print("REPORTS (n=%d):" % len(res["REPORTS"]))
    for x in res["REPORTS"]:
        print("   ", x)
    print(f"[jack-in-gates] wrote {p}")


if __name__ == "__main__":
    main()
