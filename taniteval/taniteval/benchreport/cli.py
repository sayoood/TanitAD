"""``python -m taniteval.benchreport <run_dir>`` — render, then PROVE the page prints summary.json.

Exit codes (assert on the artifact, not the status — but make the status honest too):
  0  report written and ``report/verify.json`` says PASS (and the gallery, if requested, rendered)
  1  the parse-back verification FAILED (a number / label / refusal disagrees with summary.json)
  2  the run directory violates the contract badly enough that no report could be written
  3  report verified, but the requested failure gallery could not be rendered (banner in the page)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PUBLISHED = REPO / "products" / "P7-TanitEval" / "benchmarks" / "published_results.json"
DEFAULT_NAVSIM_PY = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
DEFAULT_DEVKIT = "C:/Users/Admin/navsim-crun/devkit"


def build(run_dir, *, published=None, gallery_n: int = 8, no_gallery: bool = False, navsim_python=None,
          devkit=None, max_per_log: int = 2, quiet: bool = False) -> dict:
    from .contract import ContractError
    from .render import render_report
    from .verify import verify_report
    run_dir = Path(run_dir)
    pub = Path(published) if published else DEFAULT_PUBLISHED
    extra = []
    if not pub.is_file():
        extra.append(("serious", f"published-results file not found ({pub}) — no published rows drawn"))
        pub = None
    gal = None
    gal_status = "SKIPPED"
    if not no_gallery:
        from .gallery import build_gallery
        gal = build_gallery(run_dir, n=gallery_n, max_per_log=max_per_log,
                            navsim_python=navsim_python or DEFAULT_NAVSIM_PY, devkit=devkit or DEFAULT_DEVKIT,
                            quiet=quiet)
        gal_status = gal.get("status", "FAILED")
        if gal_status != "RENDERED":
            extra.append(("critical", f"failure gallery NOT rendered: {gal.get('reason')}"))
    try:
        res = render_report(run_dir, published_path=pub, gallery=gal, extra_banners=extra)
    except ContractError as e:
        return {"exit": 2, "error": str(e)}
    idx = Path(res["index"])
    v1 = verify_report(idx, run_dir / "summary.json", pub)
    note = (f"{v1['status']} — {v1['n_numbers_checked']} numbers and {v1['n_refusals_checked']} refusals re-read from this "
            f"HTML and checked against summary.json by an independent parser; {v1['n_errors']} errors, "
            f"{len(v1['coverage_missing'])} coverage gaps")
    res = render_report(run_dir, published_path=pub, gallery=gal, extra_banners=extra, verify_note=note)
    v2 = verify_report(Path(res["index"]), run_dir / "summary.json", pub)
    if v2["n_numbers_checked"] != v1["n_numbers_checked"]:
        v2["errors"].append(f"pass-2 checked {v2['n_numbers_checked']} numbers, pass-1 {v1['n_numbers_checked']}")
        v2["status"] = "FAIL"
    v2["published_file"] = str(pub) if pub else None
    v2["gallery"] = {k: v for k, v in (gal or {}).items() if k in ("status", "reason", "n_rendered", "selection_rule")}
    (idx.parent / "verify.json").write_text(json.dumps(v2, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    code = 0 if v2["status"] == "PASS" else 1
    if code == 0 and not no_gallery and gal_status != "RENDERED":
        code = 3
    if not quiet:
        print(f"[benchreport] {idx}")
        print(f"[benchreport] verify: {v2['status']} ({v2['n_numbers_checked']} numbers, {v2['n_errors']} errors, "
              f"{len(v2['coverage_missing'])} coverage gaps) · gallery {gal_status}")
        for e in v2["errors"][:10]:
            print("   ERR", e)
        for m in v2["coverage_missing"][:10]:
            print("   MISSING", m)
    return {"exit": code, "index": str(idx), "verify": v2, "figs": res["figs"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m taniteval.benchreport",
                                 description="Render and verify one bench run's report (W5).")
    ap.add_argument("run_dir")
    ap.add_argument("--published", default=None, help=f"published results JSON (default: {DEFAULT_PUBLISHED})")
    ap.add_argument("--gallery", type=int, default=8, help="number of worst scenes in the failure gallery")
    ap.add_argument("--max-per-log", type=int, default=2, help="gallery diversity cap per log")
    ap.add_argument("--no-gallery", action="store_true", help="skip the NavSim-venv failure gallery")
    ap.add_argument("--navsim-python", default=None, help=f"NavSim venv python (default {DEFAULT_NAVSIM_PY})")
    ap.add_argument("--devkit", default=None, help=f"NavSim devkit root (default {DEFAULT_DEVKIT})")
    a = ap.parse_args(argv)
    r = build(a.run_dir, published=a.published, gallery_n=a.gallery, no_gallery=a.no_gallery,
              navsim_python=a.navsim_python, devkit=a.devkit, max_per_log=a.max_per_log)
    if r.get("error"):
        print(f"[benchreport] CONTRACT: {r['error']}", file=sys.stderr)
    return int(r["exit"])


if __name__ == "__main__":
    sys.exit(main())
