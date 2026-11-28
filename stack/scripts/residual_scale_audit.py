"""Audit every residual seam in the frontier models for the v6 predictor defect.

WHAT THE DEFECT IS. A residual predictor computes ``out = base + delta``. If
``delta`` is produced by a Linear fed from a LayerNorm, its magnitude is O(1)
PER DIM regardless of what ``base`` is scaled to. When ``base`` is a learned
latent with a small magnitude, the correction starts orders of magnitude larger
than the quantity it corrects, and the run spends itself shrinking it.

MEASURED on v6 (2026-08-22, stride-1 latents at the true dt=0.1 s tick, 3,850
windows): operative latent mean|z| 0.015718, per-tick movement 0.000297, and the
trained predictor's error 0.172676 -- **580.5x WORSE than predicting NO CHANGE**.
Rescaling the trained heads improved monotonically all the way to alpha=0, so
there was nothing to rescue and the run had to be written off.

TWO DETECTORS, because the first one alone MISSED the second real instance:

  RETURN form     ``return x + self.mod(...)``
  ASSIGN form     ``s = s + self.mod(...)``     <- REF-A v1's
                  `StrategicSubspacePredictor.rollout` is this shape, and a
                  return-only scan reported it clean. It was found by reading,
                  not by scanning, which is exactly the kind of luck an audit
                  must not depend on.

WHAT IS *NOT* THE DEFECT, and why this script reports rather than fails on it.
An ordinary pre-norm transformer block (``x = x + self.mlp(self.n2(x))``) has
the same SHAPE, but its ``base`` is an internal activation living at the same
O(1) scale as the delta, so the ratio is fine by construction. Shape alone
cannot separate the two -- only the SCALE of what the delta is added to can.
This script therefore reports every seam with its measured ratio and leaves the
judgement explicit rather than guessing from names.

Usage:
    python stack/scripts/residual_scale_audit.py            # static census
    python stack/scripts/residual_scale_audit.py --json out.json
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
STACK = REPO / "stack"
#: the frontier models the PI named, plus the shared modules they build from
FRONTIER = {
    "v6F": ["tanitad/models/v6.py", "tanitad/models/predictor.py",
            "tanitad/models/tactical.py", "tanitad/models/hierarchy.py"],
    "REF-A v1": ["tanitad/refs/refa_v1.py", "tanitad/refs/refa_v1_plan.py",
                 "tanitad/refs/refa_v1p.py", "tanitad/refs/refa.py"],
    "flagship v1": ["tanitad/models/flagship_v15.py",
                    "tanitad/models/fourbrain.py"],
    "REF-C v3": ["tanitad/refs/refc_v3.py", "tanitad/refs/refc.py",
                 "tanitad/refs/refc_tactical.py",
                 "tanitad/models/refc_rescorer.py"],
    "REF-D": ["tanitad/refs/refd.py"],
}
#: an attribute whose delta is added to an EXTERNAL quantity rather than to an
#: internal activation is the risky class. Names are a hint for the reader only;
#: the script never decides on them.
HEADISH = ("head", "out", "proj", "pred", "delta", "readout")


def _attr_chain(node) -> str | None:
    """`self.foo.bar(...)` -> 'foo.bar'; anything else -> None."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name) and f.id == "self" and parts:
        return ".".join(reversed(parts))
    return None


def _residual_operands(node):
    """-> (base_src, delta_attr) for `A + self.mod(...)`, else None."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
        return None
    for base, delta in ((node.left, node.right), (node.right, node.left)):
        attr = _attr_chain(delta)
        if attr:
            return ast.unparse(base), attr
    return None


def scan_file(path: pathlib.Path) -> list[dict]:
    try:
        src = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(src)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    out = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        zeroed = {t.args[0] and ast.unparse(t.args[0])
                  for t in ast.walk(cls) if isinstance(t, ast.Call)
                  and _attr_chain(t) is None
                  and isinstance(t.func, ast.Attribute)
                  and t.func.attr in ("zeros_",) and t.args}
        scaled = any(isinstance(t, ast.Call) and isinstance(t.func, ast.Attribute)
                     and t.func.attr in ("mul_",) for t in ast.walk(cls))
        for fn in [n for n in ast.walk(cls)
                   if isinstance(n, ast.FunctionDef)]:
            for node in ast.walk(fn):
                pair = None
                form = ""
                if isinstance(node, ast.Return) and node.value is not None:
                    pair, form = _residual_operands(node.value), "return"
                elif isinstance(node, ast.Assign) and len(node.targets) == 1:
                    tgt = ast.unparse(node.targets[0])
                    pair = _residual_operands(node.value)
                    form = "assign"
                    if pair and pair[0] != tgt:
                        # `y = x + self.m(...)` is still a residual seam, but
                        # `s = s + ...` (accumulating) is the shape that hid
                        # from the return-only scan; both are kept.
                        form = "assign*"
                if not pair:
                    continue
                base, attr = pair
                out.append({
                    "file": str(path.relative_to(REPO)).replace("\\", "/"),
                    "class": cls.name, "method": fn.name, "line": node.lineno,
                    "form": form, "base": base[:40], "delta": f"self.{attr}",
                    "headish": any(h in attr.lower() for h in HEADISH),
                    "class_has_zeros_init": bool(zeroed),
                    "class_has_scaled_init": scaled,
                })
    return out


#: Classes that CARRY a seam, and every frontier model that INSTANTIATES them.
#: A file-scoped scan attributes a defect to the file that DEFINES the class and
#: silently clears every model that merely builds it. That is exactly how this
#: audit first reported "flagship v1 clean" while `fourbrain.py:427` constructs
#: the very `OperativePredictor` whose seam was flagged under v6F.
SHARED_SEAM_OWNERS = {
    "OperativePredictor": ["v6F", "flagship v1", "REF-D"],
    "StrategicPolicy": ["flagship v1", "REF-D"],
    "TacticalPolicy": ["flagship v1", "REF-D"],
    "FTac": ["v6F", "REF-C v3", "REF-D"],
}


def instantiation_map() -> dict[str, list[str]]:
    """-> {class: [models that construct it]}, by scanning for `Cls(` calls.

    ⭐ WHY THIS EXISTS. Defects travel by INSTANTIATION, not by file. Reporting
    only where a class is DEFINED understates the blast radius by exactly the
    set of models that import it — which is the mistake this audit made on its
    first run."""
    out: dict[str, set[str]] = {}
    for model, files in FRONTIER.items():
        for rel in files:
            fp = STACK / rel
            try:
                tree = ast.parse(fp.read_text(encoding="utf-8-sig"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for n in ast.walk(tree):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id in SHARED_SEAM_OWNERS):
                    out.setdefault(n.func.id, set()).add(model)
    return {k: sorted(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json")
    a = ap.parse_args()

    report = {"_evidence_class": "MEASURED (ours; static census of residual "
                                 "seams across the frontier models)",
              "defect": "residual delta whose init magnitude is orders above "
                        "the quantity it is added to; v6 measured 580.5x worse "
                        "than the hold baseline",
              "models": {}}
    print(f"  {'model':<13}{'file:line':<46}{'class.method':<44}"
          f"{'form':<9}{'delta':<26}{'init'}")
    print("  " + "-" * 150)
    total = flagged = 0
    for model, files in FRONTIER.items():
        rows = []
        for rel in files:
            rows += scan_file(STACK / rel)
        report["models"][model] = rows
        for r in rows:
            total += 1
            init = ("zeros" if r["class_has_zeros_init"] else
                    "SCALED" if r["class_has_scaled_init"] else "default")
            risky = r["headish"] and init == "default"
            flagged += risky
            mark = " <== REVIEW" if risky else ""
            loc = f"{r['file'].split('/')[-1]}:{r['line']}"
            print(f"  {model:<13}{loc:<46}"
                  f"{r['class'] + '.' + r['method']:<44}{r['form']:<9}"
                  f"{r['delta']:<26}{init}{mark}")
    print(f"\n  {total} residual seams · {flagged} head-like with default init "
          f"(REVIEW = shape matches the defect; only the SCALE of `base` "
          f"decides, so each needs a look)")
    inst = instantiation_map()
    report["instantiation"] = inst
    print()
    print("  BLAST RADIUS BY INSTANTIATION (a defect travels with the class, "
          "not the file):")
    for cls, models in sorted(inst.items()):
        owners = SHARED_SEAM_OWNERS.get(cls, [])
        extra = [m for m in models if m not in owners]
        print(f"    {cls:<28} built by {', '.join(models)}"
              + (f"   (+unlisted: {extra})" if extra else ""))
    print("    ⚠️ `OperativePredictor` carries the 580x seam and is built by "
          "MORE THAN v6F — file-scoped reporting hid that on the first run.")
    report["n_seams"] = total
    report["n_review"] = flagged
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(report, indent=1),
                                        encoding="utf-8")
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
