"""Independent read-back of the RENDERED leaderboard page.

⭐ Why this exists, and why it does not import the renderer. A generator that checks its own output by
re-running its own derivation measures **determinism, not correctness** — the programme's
"a check that shares the defect it checks for is green forever" rule. So this module:

* parses the markdown tables of ``Benchmarks & Eval/LEADERBOARD.md`` with its own small parser
  (``_tables``), sharing **no code** with ``render.py`` — it never imports it;
* looks every published number up **directly in the source JSON** (``published_results.json``) rather
  than through ``sources.py``'s loaders;
* compares the page's printed cell against the artifact's value, after parsing the cell back to a float.

⛔ It is only a check if it can fail: ``--mutate`` swaps two rows' values in the model *before*
rendering and requires the read-back to report errors.

    python -m taniteval.leaderboard readback            # exit 0 = every printed value traced to an artifact
    python -m taniteval.leaderboard readback --mutate   # deliberate-regression arm; exit 0 = it went RED
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAGE = REPO / "Benchmarks & Eval" / "LEADERBOARD.md"
PUB = REPO / "products" / "P7-TanitEval" / "benchmarks" / "published_results.json"
NUM = re.compile(r"^[−-]?\d+(?:\.\d+)?$")


def _tables(text: str) -> list:
    """Every markdown table as (heading, header cells, [row cells]) — an independent parser."""
    out, head, cur, hdr = [], "", None, None
    for line in text.splitlines():
        if line.startswith("#"):
            head = line.strip("# ").strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue                                  # the ---|--- separator
            if cur is None:
                hdr, cur = cells, []
                continue
            cur.append(cells)
        elif cur is not None:
            out.append((head, hdr, cur))
            cur, hdr = None, None
    if cur is not None:
        out.append((head, hdr, cur))
    return out


def _cellnum(s: str):
    """The number a cell prints, or None. Reads the FIRST bare numeric token so a cell carrying a
    trailing note still yields its value; the minus sign is the page's U+2212."""
    for t in re.split(r"[\s|]+", s.replace("−", "-")):
        t = t.strip("*`()[],;")
        if NUM.match(t):
            return float(t)
    return None


def check(page_text: str | None = None) -> dict:
    pub = json.loads(PUB.read_text(encoding="utf-8"))
    text = page_text if page_text is not None else PAGE.read_text(encoding="utf-8")
    # index the ARTIFACT by (system, protocol-ish decimals) — the page prints system names verbatim
    by_system: dict[str, list] = {}
    for r in pub["results"]:
        by_system.setdefault(r["system"], []).append(r)
    errors, checked, unmatched = [], 0, []
    for head, hdr, rows in _tables(text):
        if not hdr or hdr[0] != "system":
            continue
        # the value column is the second one in every external table this page renders
        for cells in rows:
            name = cells[0].strip("* ")
            cands = by_system.get(name)
            if not cands:
                unmatched.append((head, name))
                continue
            got = _cellnum(cells[1])
            if got is None:
                continue
            want = {c.get("value") for c in cands if c.get("value") is not None}
            want |= {v for c in cands for v in (c.get("values") or {}).values() if isinstance(v, (int, float))}
            checked += 1
            # ⚠️ the page PRINTS at the row's decimals, so the admissible tolerance is half a printed
            # unit — a tighter one flags DISPLAY ROUNDING as a discrepancy (measured: 0.38 vs 0.3844).
            dp = max(len(cells[1].split(".")[1].rstrip("*`)] ")) if "." in cells[1].split()[0] else 0, 1)
            tol = 0.5 * 10 ** -dp + 1e-9
            if not any(abs(got - w) <= tol for w in want):
                errors.append({"heading": head, "system": name, "page_prints": got,
                               "artifact_has": sorted(want)[:8]})
    return {"checked_cells": checked, "errors": errors, "unmatched_systems": unmatched[:20],
            "n_unmatched": len(unmatched)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m taniteval.leaderboard readback")
    ap.add_argument("--mutate", action="store_true", help="swap two published values before rendering; the read-back MUST report errors")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.mutate:
        from .build import load_model, render_section                  # imported only for the mutation arm
        from . import sources as S
        cfg = S.load_config(REPO)
        real = S.load_published

        def swapped(root, c):
            d = real(root, c)
            rs = [x for x in d["results"] if x["protocol"] == "PDMS_v1_navtest" and x.get("value") is not None][:2]
            rs[0]["value"], rs[1]["value"] = rs[1]["value"], rs[0]["value"]
            return d

        S.load_published = swapped
        try:
            sec = render_section(REPO, cfg)
        finally:
            S.load_published = real
        res = check(sec)
        print(json.dumps({"arm": "MUTATION (two v1 values swapped)", "errors_found": len(res["errors"]),
                          "verdict": "RED (the read-back caught it)" if res["errors"] else "⛔ GREEN — the read-back is BLIND",
                          "examples": res["errors"][:3]}, ensure_ascii=False, indent=1))
        return 0 if res["errors"] else 1
    res = check()
    print(json.dumps({"checked_cells": res["checked_cells"], "errors": len(res["errors"]),
                      "unmatched_systems": res["n_unmatched"],
                      "verdict": "PASS" if not res["errors"] else "FAIL", "detail": res["errors"][:5]},
                     ensure_ascii=False, indent=1))
    return 0 if not res["errors"] else 1


if __name__ == "__main__":                                              # pragma: no cover
    raise SystemExit(main())
