#!/usr/bin/env python3
"""registry_numbers — every ``<!-- src: … -->``-annotated NUMBER in MODEL_REGISTRY.md, checked.

Why this exists
---------------
`tools/registry_paths.py` made the registry's **paths** checkable. Its numbers were
not, and they fail in exactly the same way — quietly, and while looking sourced.
Two were found by hand on 2026-08-03:

* two different **"8.4 %"** figures were being merged. One is a *relative change*
  in the flagship's fan under unfreezing; the other is a *fraction of the oracle
  gap* closed by REF-C's re-scorer. Unrelated quantities, same digits.
* **`0.1640` / `45.4 %`** are REF-C-**XL**'s and were being quoted for **base**,
  whose true values are `0.1914` / `41.09 %`.

Neither is a dead path, and neither would be caught by a path sweep. Both are the
same root class: **a quantity printed without the qualifier that identifies it.**

What this checks
----------------
§6's leaderboard already carries a provenance convention that nothing verified::

    | 5 | **REF-B v2** | `refb-v2-30k` | … | **0.5913** [0.4766, 0.7131]
        <!-- src: taniteval/results/driving_refb-v2-30k.json#headline.ade_0_2s.mean --> | …

This resolves each annotation and compares it to the number printed in the SAME
cell — the point estimate, and the ``[lo, hi]`` interval when the artifact carries
one. A mismatch is reported at the precision the document prints, so a genuine
rounding difference is never called a defect and a genuinely wrong digit always is.

⚠️ It deliberately does NOT check un-annotated numbers. Guessing which artifact an
unannotated figure came from is how a wrong number becomes a sourced number. The
answer to an unannotated number is to **annotate it**, not to infer it.

Usage
-----
    python tools/registry_numbers.py
    python tools/registry_numbers.py --json out.json

Exit ``1`` if any annotated number disagrees with its artifact, or if an
annotation points at a file or key that does not exist.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "Project Steering" / "MODEL_REGISTRY.md"

MATCH = "MATCH"
MISMATCH = "MISMATCH"
SRC_MISSING = "SRC_MISSING"
KEY_MISSING = "KEY_MISSING"
NO_NUMBER = "NO_NUMBER"

SRC_RE = re.compile(r"<!--\s*src:\s*([^#\s>]+)#([^\s>]+?)\s*-->")
#: a number as the registry prints one: 0.5913 / -0.0278 / 1.0 / 29 999 is NOT
#: wanted here (the annotated cells are all decimals), so require a decimal point
NUM_RE = re.compile(r"-?\d+\.\d+")


def dig(obj, dotted: str):
    """Resolve ``a.b.c`` against nested dicts. Returns (found, value)."""
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def cell_before(text: str, end: int) -> str:
    """The markdown table cell that the annotation at ``end`` sits in.

    Bounded by the previous ``|`` on the same line — an annotation belongs to the
    number beside it, not to a number three columns away.
    """
    line_start = text.rfind("\n", 0, end) + 1
    bar = text.rfind("|", line_start, end)
    return text[(bar + 1 if bar >= line_start else line_start):end]


def printed_numbers(cell: str) -> list[str]:
    return NUM_RE.findall(cell.replace("−", "-"))


def _agrees(printed: str, value) -> bool:
    """True when ``value`` rounds to exactly what the document printed.

    Comparing at the document's own precision is the whole point: it cannot
    manufacture a defect out of rounding, and it cannot miss a wrong digit.
    """
    if not isinstance(value, (int, float)):
        return False
    nd = len(printed.split(".")[1]) if "." in printed else 0
    return f"{round(float(value), nd):.{nd}f}" == f"{float(printed):.{nd}f}"


def check(registry: Path | None = None, repo_root: Path | None = None) -> dict:
    reg = registry or REGISTRY
    root = repo_root or REPO_ROOT
    text = reg.read_text(encoding="utf-8", errors="replace")
    out: list[dict] = []
    for m in SRC_RE.finditer(text):
        rel, key = m.group(1), m.group(2)
        rec = {"src": rel, "key": key}
        p = root / rel
        if not p.exists():
            rec["status"] = SRC_MISSING
            out.append(rec)
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            rec["status"] = SRC_MISSING
            rec["note"] = f"unreadable: {e}"
            out.append(rec)
            continue
        found, val = dig(data, key)
        if not found:
            rec["status"] = KEY_MISSING
            rec["top_level"] = sorted(data)[:12] if isinstance(data, dict) else None
            out.append(rec)
            continue
        rec["artifact_value"] = val
        nums = printed_numbers(cell_before(text, m.start()))
        if not nums:
            rec["status"] = NO_NUMBER
            out.append(rec)
            continue
        rec["printed"] = nums[0]
        rec["status"] = MATCH if _agrees(nums[0], val) else MISMATCH

        # the interval, when the artifact publishes one and the cell prints one
        sib = key.rsplit(".", 1)[0]
        for i, bound in enumerate(("lo", "hi"), start=1):
            ok, bval = dig(data, f"{sib}.{bound}")
            if ok and len(nums) > i:
                rec[f"printed_{bound}"] = nums[i]
                rec[f"artifact_{bound}"] = bval
                if not _agrees(nums[i], bval):
                    rec["status"] = MISMATCH
                    rec.setdefault("why", []).append(bound)
        out.append(rec)
    counts: dict[str, int] = {}
    for r in out:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"registry": str(reg.relative_to(root)).replace("\\", "/"),
            "n_annotations": len(out), "counts": counts, "annotations": out}


def exit_code(result: dict) -> int:
    bad = sum(result["counts"].get(k, 0)
              for k in (MISMATCH, SRC_MISSING, KEY_MISSING))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args(argv)
    res = check()
    print(f"{res['n_annotations']} src-annotated numbers in {res['registry']}")
    for k in (MATCH, MISMATCH, SRC_MISSING, KEY_MISSING, NO_NUMBER):
        if res["counts"].get(k):
            print(f"  {k:12s} {res['counts'][k]}")
    print()
    for r in res["annotations"]:
        if r["status"] == MATCH:
            continue
        print(f"{r['status']:12s} {r['src']}#{r['key']}")
        if "printed" in r:
            print(f"{'':12s}   doc {r['printed']}  vs artifact {r.get('artifact_value')}")
        if r.get("why"):
            print(f"{'':12s}   interval bound(s) disagree: {r['why']}")
        if r.get("top_level"):
            print(f"{'':12s}   top-level keys: {r['top_level']}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json}")
    return exit_code(res)


if __name__ == "__main__":
    raise SystemExit(main())
