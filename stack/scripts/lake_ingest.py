"""Ingest a permissive source into the TanitAD Data Lake (Phase A driver).

Dev-box only; NEVER touches pods/training/caches. Two comma2k19 intake modes:

    # (a) no-decode: ingest already-built ep_*.pt caches (fast; spec Phase A)
    python -m scripts.lake_ingest --lake-root <LAKE> --mode cached \\
        --train-cache <dir-of-ep_*.pt> --val-cache <dir-of-ep_*.pt>

    # (b) raw video: wrap comma2k19.build_episode on real segments (PyAV decode)
    python -m scripts.lake_ingest --lake-root <LAKE> --mode segments \\
        --root <comma2k19 extracted root> --size 256 --max-units 188

The lake lands under <LAKE>/{shards,catalog}/ + MANIFEST.json + NOTICE.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

_STACK = Path(__file__).resolve().parents[1]
if str(_STACK) not in sys.path:
    sys.path.insert(0, str(_STACK))

from tanitad.lake.catalog import catalog_summary            # noqa: E402
from tanitad.lake.ingest import (Comma2k19Ingestor,          # noqa: E402
                                 CachedEpisodeIngestor, ingest_source)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake-root", required=True)
    ap.add_argument("--source", default="comma2k19")
    ap.add_argument("--mode", choices=["cached", "segments"], default="cached")
    ap.add_argument("--root", help="comma2k19 extracted root (mode=segments)")
    ap.add_argument("--train-cache", help="dir of ep_*.pt (mode=cached)")
    ap.add_argument("--val-cache", help="dir of ep_*.pt (mode=cached)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--run-id", default="0")
    args = ap.parse_args()

    lake = Path(args.lake_root)
    lake.mkdir(parents=True, exist_ok=True)

    if args.mode == "cached":
        assert args.train_cache or args.val_cache, \
            "mode=cached needs --train-cache and/or --val-cache"
        root = {}
        if args.train_cache:
            root["train"] = sorted(glob.glob(str(Path(args.train_cache) / "ep_*.pt")))
        if args.val_cache:
            root["val"] = sorted(glob.glob(str(Path(args.val_cache) / "ep_*.pt")))
        ing = CachedEpisodeIngestor(source=args.source, size=args.size)
    else:
        assert args.root, "mode=segments needs --root"
        root = args.root
        ing = Comma2k19Ingestor(source=args.source, size=args.size)

    summary = ingest_source(ing, root, lake, run_id=args.run_id,
                            max_units=args.max_units)
    print("\n=== INGEST SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "skipped"},
                     indent=2, default=str))
    if summary["skipped"]:
        print(f"skipped {len(summary['skipped'])} units "
              f"(first: {summary['skipped'][0]})")
    print("\n=== CATALOG ===")
    print(json.dumps(catalog_summary(lake), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
