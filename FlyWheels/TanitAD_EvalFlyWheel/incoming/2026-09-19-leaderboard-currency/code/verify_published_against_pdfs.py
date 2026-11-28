"""Verify products/P7-TanitEval/benchmarks/published_results.json against the BANKED PDFs.

For every row with a library key:
  1. the PDF's sha256 must equal library.json's (a banked file whose bytes changed is worse than a
     missing one — CLAUDE.md research-banking rule);
  2. every string in ``page_tokens`` must occur in the PyMuPDF text of ``source.page`` (1-based).

Two deliberate-regression arms prove the check can fail (a check that cannot go RED is not a check):
  M1  every token perturbed by one unit in its last printed decimal  -> expected RED on (nearly) every row
  M2  every row pointed at the NEXT page                               -> expected RED on most rows
A perturbed token that happens to exist on the page is counted and reported, never hidden.

Usage (repo root):  python "FlyWheels/.../code/verify_published_against_pdfs.py" [--json out.json]
Exit 0 only if the true arm passes on every verifiable row AND both mutation arms go RED on
>= 90 % (M1) / >= 50 % (M2) of the rows they touch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF

REPO = Path(__file__).resolve().parents[5]
PUB = REPO / "products" / "P7-TanitEval" / "benchmarks" / "published_results.json"
LIB = REPO / "TanitAD Research Lab" / "Library" / "library.json"


def _perturb(tok: str) -> str:
    """One unit in the last printed decimal (e.g. 56.6 -> 56.7, 0.41 -> 0.42, 18.05 -> 18.06)."""
    if "." in tok:
        d = len(tok.split(".")[1])
        v = round(float(tok) + 10 ** (-d), d)
        return f"{v:.{d}f}"
    return str(int(tok) + 1)


def _scope(txt: str, win: dict | None) -> str | None:
    """The text a row's tokens must sit in: the whole page, or the row window
    [anchor, end) found after ``after``. None = the window's anchors are not on this page."""
    if not win:
        return txt
    start = 0
    if win.get("after"):
        start = txt.find(win["after"])
        if start < 0:
            return None
    a = txt.find(win["anchor"], start)
    if a < 0:
        return None
    e = txt.find(win["end"], a + len(win["anchor"]))
    return txt[a:e] if e >= 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pub = json.loads(PUB.read_text(encoding="utf-8"))
    lib = json.loads(LIB.read_text(encoding="utf-8"))["entries"]
    lib = lib if isinstance(lib, dict) else {e["key"]: e for e in lib}

    page_cache: dict[tuple[str, int], str] = {}
    sha_ok: dict[str, bool] = {}

    def page_text(key: str, page: int) -> str | None:
        ent = lib.get(key)
        if ent is None:
            return None
        if key not in sha_ok:
            p = REPO / ent["path"]
            sha_ok[key] = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == ent["sha256"]
        if (key, page) not in page_cache:
            doc = fitz.open(REPO / ent["path"])
            page_cache[(key, page)] = doc[page - 1].get_text() if 1 <= page <= doc.page_count else ""
        return page_cache[(key, page)]

    rows, true_fail, m1_red, m1_touch, m2_red, m2_touch = [], 0, 0, 0, 0, 0
    coincidences = []
    for r in pub["results"]:
        src = r["source"]
        key, page, toks = src.get("library_key"), src.get("page"), r.get("page_tokens", [])
        if not key or not page or not toks:
            rows.append({"id": r["id"], "status": "NOT VERIFIABLE (no banked primary)", "evidence_class": r.get("evidence_class", "PUBLISHED")})
            continue
        txt = page_text(key, int(page))
        if txt is None:
            rows.append({"id": r["id"], "status": "FAIL: library key missing", "key": key})
            true_fail += 1
            continue
        win = r.get("row_window")
        scope = _scope(txt, win)
        missing = toks if scope is None else [t for t in toks if t not in scope]
        ok = sha_ok[key] and not missing
        true_fail += 0 if ok else 1
        # M1 — perturbed tokens, same scope
        m1_touch += 1
        pert = [_perturb(t) for t in toks]
        present = [] if scope is None else [p for p in pert if p in scope]
        if len(present) < len(pert):
            m1_red += 1
        else:
            coincidences.append({"id": r["id"], "perturbed_tokens_all_present": present})
        # M2 — wrong page (a row window that cannot be found there also counts as RED)
        m2_touch += 1
        scope2 = _scope(page_text(key, int(page) + 1) or "", win)
        if scope2 is None or any(t not in scope2 for t in toks):
            m2_red += 1
        # a PARAMETER COUNT is a published number like any other: it is re-read from the page the row
        # cites for it (usually NOT the page the score comes from), or the row may not carry one
        pm = r.get("params_source")
        p_missing = None
        if pm and r.get("params_tokens"):
            ptxt = page_text(pm.get("library_key", key), int(pm["page"])) or ""
            pscope = _scope(ptxt, pm.get("row_window"))
            p_missing = r["params_tokens"] if pscope is None else [t for t in r["params_tokens"] if t not in pscope]
            if p_missing:
                ok = False
                true_fail += 1
        rows.append({"id": r["id"], "key": key, "page": page, "sha_ok": sha_ok[key],
                     "row_window": bool(win), "tokens": toks, "missing": missing,
                     "params_missing": p_missing,
                     "status": "PASS" if ok else "FAIL"})

    n_ver = sum(1 for x in rows if x.get("status") in ("PASS", "FAIL"))
    out = {
        "published_results": str(PUB.relative_to(REPO)),
        "n_rows": len(pub["results"]), "n_verifiable": n_ver, "true_arm_failures": true_fail,
        "m1_perturbed_red": f"{m1_red}/{m1_touch}", "m2_wrong_page_red": f"{m2_red}/{m2_touch}",
        "m1_coincidences": coincidences, "pdf_sha256_ok": sha_ok, "rows": rows,
    }
    verdict = (true_fail == 0 and m1_touch and m1_red >= 0.9 * m1_touch and m2_red >= 0.5 * m2_touch)
    out["verdict"] = "PASS" if verdict else "FAIL"
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rows {len(pub['results'])} · verifiable {n_ver} · true-arm failures {true_fail} · "
          f"M1 perturbed RED {m1_red}/{m1_touch} · M2 wrong-page RED {m2_red}/{m2_touch} · "
          f"sha256 ok {sum(sha_ok.values())}/{len(sha_ok)} · VERDICT {out['verdict']}")
    for x in rows:
        if x.get("status") not in ("PASS",):
            print("  ", x["id"], "->", x.get("status"), x.get("missing", ""))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
