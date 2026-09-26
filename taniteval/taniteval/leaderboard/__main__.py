"""CLI: ``python -m taniteval.leaderboard build [--check] [--no-html] [--root PATH]``.

Run from ``<repo>/taniteval`` (the same working directory as ``python -m taniteval.runner``).
``--check`` writes nothing and exits 1 if the page or the HTML is out of date with its inputs.
"""
from __future__ import annotations

import argparse
import json
import sys

from .build import build
from .sources import SourceError


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m taniteval.leaderboard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="regenerate the BENCH section of LEADERBOARD.md and leaderboard.html")
    b.add_argument("--check", action="store_true", help="write nothing; exit 1 if out of date")
    b.add_argument("--no-html", action="store_true")
    b.add_argument("--root", default=None, help="repo root (default: derived from this package's location)")
    fx = sub.add_parser("fixtures", help="write W1-schema summary.json fixtures converted from the banked legacy outputs")
    fx.add_argument("--out", required=True, help="fixture root; files land at <out>/<benchmark>/<split>/<run_id>/summary.json")
    rb = sub.add_parser("readback", help="INDEPENDENT read-back: re-parse the rendered page and trace every printed value to its artifact")
    rb.add_argument("--mutate", action="store_true", help="deliberate-regression arm: swap two values before rendering; the read-back must go RED")
    a, rest = ap.parse_known_args(argv)
    try:
        if a.cmd == "readback":
            from .readback import main as readback_main
            return readback_main(["--mutate"] if a.mutate else [])
        if a.cmd == "fixtures":
            paths = write_fixtures(a.out)
            print(json.dumps({"written": paths}, indent=1))
            return 0
        out = build(root=a.root, check=a.check, write_html=not a.no_html)
    except SourceError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1))
    if a.check:
        return 0 if out["md_up_to_date"] and out["html_up_to_date"] else 1
    return 0


def fixture_text(summary: dict) -> str:
    return json.dumps(summary, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def write_fixtures(out_root) -> list:
    """Each legacy summary is validated by W1's schema_check BEFORE it is written."""
    from pathlib import Path
    from . import sources
    root = sources.REPO
    conv, _ = sources.legacy_summaries(root, sources.load_config(root))
    written = []
    for spec, s in conv:
        sources.w1_validate(s, spec["path"])
        p = Path(out_root) / s["benchmark"] / s["split"] / s["run_id"] / "summary.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(fixture_text(s).encode("utf-8"))
        written.append(str(p))
    return written


if __name__ == "__main__":
    sys.exit(main())
