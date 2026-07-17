"""ACCEPTANCE GATE — prove LakeWindowDataset == EpisodeWindowDataset (byte-for-byte).

Builds the reference the trainer would build (``EpisodeWindowDataset`` over the
source ``ep_*.pt``) and the lake's ``LakeWindowDataset`` over the same episodes,
then asserts every window is bit-identical (frames/actions/future_*/poses) and
that the catalog preserves the CORPUS_META / I7 identity.

    python -m scripts.lake_byteproof --lake-root <LAKE> --source comma2k19 \\
        --ref-cache <dir-of-ingested-ep_*.pt> [--ref-cache <another>] \\
        --window 8 --max-horizon 16
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

import pyarrow.dataset as pads                               # noqa: E402

from tanitad.data._contract import EpisodeWindowDataset      # noqa: E402
from tanitad.data.mixing import load_episode                 # noqa: E402
from tanitad.lake.catalog import resolve_view                # noqa: E402
from tanitad.lake.proof import (assert_corpus_meta_identity,  # noqa: E402
                                assert_datasets_bit_identical)
from tanitad.lake.view import LakeView, LakeWindowDataset     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake-root", required=True)
    ap.add_argument("--source", default="comma2k19")
    ap.add_argument("--ref-cache", action="append", required=True,
                    help="dir(s) of the SAME ep_*.pt that were ingested")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--max-horizon", type=int, default=16)
    ap.add_argument("--split", default=None,
                    help="restrict the lake view to this split (match --ref-cache)")
    ap.add_argument("--cache-dir", default=None,
                    help="hydrate ep_*.pt here (use fast local SSD, not the Drive)")
    args = ap.parse_args()

    # --- reference: exactly what the trainer builds today ---
    ref_files = []
    for d in args.ref_cache:
        ref_files += sorted(glob.glob(str(Path(d) / "ep_*.pt")))
    assert ref_files, f"no ep_*.pt under {args.ref_cache}"
    ref_eps = [load_episode(p, mmap=True) for p in ref_files]
    ref_eps.sort(key=lambda e: int(e.episode_id))            # episode_id order
    ds_ref = EpisodeWindowDataset(ref_eps, window=args.window,
                                  max_horizon=args.max_horizon)

    # --- lake: same episodes, resolved through the catalog + hydrated shards ---
    expr = (pads.field("source") == args.source)
    if args.split:
        expr = expr & (pads.field("split") == args.split)
    view = LakeView(args.lake_root, name=f"proof-{args.source}", filter_expr=expr)
    ds_lake = LakeWindowDataset(view, window=args.window,
                                max_horizon=args.max_horizon, split=args.split,
                                cache_dir=args.cache_dir)

    print(f"[byteproof] ref windows={len(ds_ref)}  lake windows={len(ds_lake)}")
    result = assert_datasets_bit_identical(ds_ref, ds_lake)

    # --- CORPUS_META / I7 identity from the catalog ---
    from tanitad.data.comma2k19 import CORPUS_META
    rows = resolve_view(args.lake_root,
                        filter_expr=(pads.field("source") == args.source),
                        columns=["channels", "image_size", "f_eff_px", "hz"]
                        ).to_pylist()
    meta_checks = assert_corpus_meta_identity(rows, CORPUS_META)

    print("\n=== BYTE-EQUIVALENCE PROOF: PASS ===")
    print(json.dumps({**result, "corpus_meta_identity": meta_checks,
                      "n_episodes": len(ref_eps)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
