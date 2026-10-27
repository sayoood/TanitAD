#!/usr/bin/env python3
"""What does box 2's DAGGER2 queue READ from the repo?  A static import closure of the four python entry points the queue runs on box 2, plus the repo DATA paths those programs open by default.

Entry points (the queue's TRAIN / EVAL / DAGTOOL, and the generator the tool spawns): stack/scripts/v8f_mini_train.py (gatedag, traindag), v8f_mini_eval.py (evaldag), v8f_dagger_queue_tool.py + v8f_dagger_bank.py (bankdag2).
The closure follows `import x` / `from x import y` (module level AND inside functions) through the roots the queue puts on PYTHONPATH (%REPO%\\stack;%REPO%\\taniteval) and the script directory, adds every package `__init__.py`
on the way, and follows `importlib.import_module("literal")`. Anything it cannot resolve to a repo file is reported as EXTERNAL (torch, numpy, ...) -- never silently dropped.

    python export_closure.py TREE_ROOT [--json out.json]        TREE_ROOT = an extraction of the repo tip containing at least stack/ and taniteval/ python files

The DATA reads are NOT discovered by the closure (a file open is not an import): they are listed below BY HAND from a grep of the closure for repo-relative paths, each with the file:line that opens it, and the selftest of the chain
checks that the export list names every one of them. A path added here without that grep would be a claim, so each entry carries its source line.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ENTRIES = ["stack/scripts/v8f_mini_train.py", "stack/scripts/v8f_mini_eval.py", "stack/scripts/v8f_dagger_queue_tool.py", "stack/scripts/v8f_dagger_bank.py"]
ROOTS = ["stack", "taniteval", "stack/scripts"]
PK = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini"
HJ = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-refav1-hj-mini"

# (repo path, who opens it, why). Source lines are from the tip 8e121ea8's files.
DATA_READS = [
    (f"{PK}/code/v8f_mini_box2_queue.cmd", "the box-2 job files call the queue at the repo path", "the queue itself (mk_v8f_job.py writes `call <REPO>\\...\\v8f_mini_box2_queue.cmd <stage>` into each job)"),
    (f"{PK}/raw/eval_grid_windows.json", "v8f_mini_train.py:51 DAGGER_EVAL_GRID (trainer K-B3); the queue's :prepdag copies it; v8f_mini_eval.py:610 --eval-grid default", "eval-141 grid"),
    (f"{PK}/raw/eval141_kin.npz", "v8f_mini_eval.py:611 --eval-kin default (the `run` eval line does not override it)", "eval-141 kinematics"),
    (f"{PK}/raw/join_eval141.npz", "v8f_mini_eval.py:612 --eval-join default (the `run` eval line does not override it)", "eval-141 label join"),
    (f"{PK}/raw/2026-10-26-dagger-draft/probe/THRESHOLDS_FIXED.txt", "v8f_dagger_bank.py:284 (thresholds_file_sha256 in every shard header)", "the bank header records its sha256; absent => None, a different bank header than iteration 1's"),
    (f"{HJ}/raw/p10_yaw4512/p10_prior.pt", "the queue's :prep `copy /y %HJRAW%\\p10_prior.pt` (every stage runs :prep)", "P10 prior"),
    (f"{HJ}/raw/p10_yaw4512/p10_eval_pred6.npy", "the queue's :prep `copy /y %HJRAW%\\p10_eval_pred6.npy`", "P10 eval predictions"),
]
B1 = "TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-02-b1-eval-lead-block/raw/b1_eval_lead_block.npz"     # NOT in DATA_READS: no program of the closure names it (grep below); see NOTES.md


def module_files(root: Path, mod: str):
    """Repo .py files that `import mod` can load from any root: mod.py, mod/__init__.py, and every parent package's __init__.py."""
    parts = mod.split(".")
    out = []
    for r in ROOTS:
        base = root / r
        for k in range(1, len(parts) + 1):
            pk = base.joinpath(*parts[:k])
            if (pk / "__init__.py").is_file():
                out.append((pk / "__init__.py").relative_to(root).as_posix())
        leaf = base.joinpath(*parts)
        if leaf.with_suffix(".py").is_file():
            out.append(leaf.with_suffix(".py").relative_to(root).as_posix())
            for k in range(1, len(parts)):
                ip = base.joinpath(*parts[:k]) / "__init__.py"
                if ip.is_file():
                    out.append(ip.relative_to(root).as_posix())
    return sorted(set(out))


def imports_of(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                mods.add(a.name)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module)
            for a in n.names:
                mods.add(n.module + "." + a.name)                # `from pkg import submodule`
        elif isinstance(n, ast.ImportFrom) and n.level > 0:
            mods.add(("." * n.level) + (n.module or ""))
        elif isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name)) and getattr(n.func, "attr", getattr(n.func, "id", "")) in ("import_module", "__import__"):
            if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str):
                mods.add(n.args[0].value)
    return mods


def closure(root: Path):
    seen, todo, external, relative = set(), list(ENTRIES), set(), {}
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        p = root / f
        if not p.is_file():
            raise SystemExit(f"closure: {f} is not in {root}")
        for m in sorted(imports_of(p)):
            if m.startswith("."):
                relative.setdefault(f, []).append(m)
                # package-relative: resolve against the file's own package
                pkg = Path(f).parent
                up = len(m) - len(m.lstrip("."))
                for _ in range(up - 1):
                    pkg = pkg.parent
                name = m.lstrip(".")
                for cand in ([pkg / (name.replace(".", "/") + ".py"), pkg / name.replace(".", "/") / "__init__.py"] if name else [pkg / "__init__.py"]):
                    if (root / cand).is_file():
                        todo.append(cand.as_posix())
                continue
            fs = module_files(root, m)
            if fs:
                todo.extend(fs)
            else:
                external.add(m.split(".")[0])
    return sorted(seen), sorted(external), relative


def main():
    root = Path(sys.argv[1])
    files, external, relative = closure(root)
    top = {}
    for f in files:
        top.setdefault("/".join(f.split("/")[:3]) if f.startswith("stack/tanitad") or f.startswith("taniteval/") else "/".join(f.split("/")[:2]), []).append(f)
    size = sum((root / f).stat().st_size for f in files)
    out = {"entries": ENTRIES, "n_files": len(files), "bytes": size, "files": files, "external_top_level_modules": external, "package_relative_imports_resolved": relative,
           "by_dir": {k: len(v) for k, v in sorted(top.items())}, "data_reads": [{"path": p, "opened_by": w, "why": y} for p, w, y in DATA_READS]}
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"closure: {len(files)} python files, {size / 1e6:.2f} MB; external top-level modules: {external}")
    for k, v in out["by_dir"].items():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
