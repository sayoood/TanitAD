"""PH0 clip sampler — 50 clips stratified by road class, seed 0, deterministic.

Per `PREREG_PH0_VLM.md`: "50 clips sampled from the augmentation set
stratified by road class (the card's coverage table), seed 0, clip_ids
listed in `ph0_clips.json` committed WITH the sampler line that produced
them. Same clips for all arms."

Determinism contract (unit-tested): clip ids are sorted before any draw, the
RNG is `numpy.random.default_rng(seed)` consumed in a fixed stratum order
(sorted road classes), and allocation is largest-remainder proportional with
a floor of 1 clip per non-empty stratum. Running the sampler twice on the
same parquet yields byte-identical clip lists.

`--mini 8` produces the first mini-pilot subset (same machinery, n=8).

The emitted ph0_clips.json embeds the sampler line (argv), the sha256 of this
script, the seed, and the per-stratum counts — the prereg's provenance
requirement. It is directly consumable by `ph0_pilot.py --clips`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# candidate road-class column names, probed in order (rule: absence found at
# one name is not absence — we probe the plausible names, then fail LOUDLY
# with the real column list rather than guessing).
ROAD_COL_CANDIDATES = ("road_class", "road_type", "strata_road_class",
                       "strata_cell", "road")


def find_road_col(columns, override: str | None = None) -> str:
    if override:
        if override not in columns:
            raise SystemExit(f"--road-col {override!r} not in parquet columns:"
                             f" {sorted(columns)}")
        return override
    for c in ROAD_COL_CANDIDATES:
        if c in columns:
            return c
    raise SystemExit(
        "no road-class column found (tried "
        f"{ROAD_COL_CANDIDATES}); parquet columns are {sorted(columns)}. "
        "Pass --road-col explicitly.")


def stratified_sample(clip_to_class: dict[str, str], n: int,
                      seed: int = 0) -> list[dict]:
    """Deterministic stratified sample: proportional largest-remainder
    allocation over road classes (floor 1 per non-empty stratum), then a
    seeded permutation draw inside each stratum over SORTED clip ids."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    strata: dict[str, list[str]] = {}
    for cid in sorted(clip_to_class):
        strata.setdefault(str(clip_to_class[cid]), []).append(cid)
    classes = sorted(strata)
    total = sum(len(v) for v in strata.values())
    n = min(n, total)

    # proportional allocation, largest remainder, floor 1 where it fits
    quota = {c: n * len(strata[c]) / total for c in classes}
    alloc = {c: int(quota[c]) for c in classes}
    if len(classes) <= n:
        for c in classes:
            alloc[c] = max(1, alloc[c])
    while sum(alloc.values()) > n:                 # floors overshot: trim the
        c = max(classes, key=lambda c_: (alloc[c_] - quota[c_], c_))
        alloc[c] -= 1                              # most over-allocated
    rem = sorted(classes, key=lambda c: (-(quota[c] - int(quota[c])), c))
    i = 0
    while sum(alloc.values()) < n:
        c = rem[i % len(rem)]
        if alloc[c] < len(strata[c]):
            alloc[c] += 1
        i += 1
    for c in classes:                              # never over-draw a stratum
        alloc[c] = min(alloc[c], len(strata[c]))

    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for c in classes:                              # fixed consumption order
        ids = strata[c]
        pick = rng.permutation(len(ids))[: alloc[c]]
        out += [{"clip_id": ids[int(k)], "road_class": c}
                for k in sorted(pick)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True,
                    help="augmentation records.parquet (road-class column)")
    ap.add_argument("--out", default="ph0_clips.json")
    ap.add_argument("--n", type=int, default=50,
                    help="pilot size (prereg: 50)")
    ap.add_argument("--mini", type=int, default=None,
                    help="mini-pilot mode: sample this many clips (e.g. 8) "
                         "instead of --n")
    ap.add_argument("--seed", type=int, default=0,
                    help="prereg seed (0); changing it changes the pilot set")
    ap.add_argument("--road-col", default=None,
                    help="road-class column override")
    args = ap.parse_args(argv)

    import pandas as pd
    df = pd.read_parquet(args.records)
    if "clip_id" not in df.columns:
        raise SystemExit(f"records has no clip_id column: {sorted(df.columns)}")
    road_col = find_road_col(df.columns, args.road_col)
    per_clip = (df[["clip_id", road_col]].dropna()
                .drop_duplicates("clip_id"))
    clip_to_class = {str(r.clip_id): str(getattr(r, road_col))
                     for r in per_clip.itertuples()}
    n = args.mini if args.mini is not None else args.n
    clips = stratified_sample(clip_to_class, n, seed=args.seed)

    counts: dict[str, int] = {}
    for c in clips:
        counts[c["road_class"]] = counts.get(c["road_class"], 0) + 1
    payload = {
        "sampler_line": "python " + " ".join(
            [Path(sys.argv[0]).name] + (argv if argv is not None
                                        else sys.argv[1:])),
        "sampler_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "n": len(clips),
        "mini": args.mini is not None,
        "records": str(args.records),
        "road_col": road_col,
        "strata_counts": dict(sorted(counts.items())),
        "clips": clips,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"[ph0-sample] {len(clips)} clips ({counts}) -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
