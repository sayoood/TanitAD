#!/usr/bin/env python3
"""RL-S1 priority 2: the ONE admissible change to the frozen partition -- an availability filter on the POOL, with `rl_partition.py` itself untouched.

`PREREG_RL_S1.md` ("OPEN-BASE DECIDED") allows exactly one change before any RL number exists: re-running the SAME tool and seed with an availability filter,
if the dumper cannot reach some scenes.  `rl_partition.py` has no such flag and must not get one (its logic is frozen), but its INPUT can be filtered:

    1. on the pod:   python refe/rl_fan_dump.py --plan-only ... --keys rl_train_scenes.tsv rl_val_scenes.tsv --out <d>   (exit 3 + <d>/unavailable_scenes_*.tsv)
    2.               python refe/rl_pool_available.py --pool raw/rl_pool/navtrain_token_table.tsv.gz --unavailable <d>/unavailable_scenes_*.tsv --out pool_available.tsv.gz
    3.               python refe/rl_partition.py --pool pool_available.tsv.gz <the SAME --navtest / --heldout-logs / --n-train / --n-val / --min-val-logs> --seed 20261012 --out <new dir>
    4. log the OLD and NEW partition_id in PREREG_RL_S1.md before the first RL number exists (never after).

Why filtering the pool is "the same tool and seed": `rl_partition` orders LOGS by sha256(seed|log) and scenes inside a log by sha256(seed|token), both independent of
the input, so removing a scene from the pool removes it from the order and changes nothing else (`test_rl_pool_available.py` proves it on the real
`partition()`: removing a scene the partition did not use leaves train and val byte-identical; removing a train scene replaces it by the next scene in order).

The output keeps the pool's header, every column and the row order; only rows whose token is listed as unavailable are dropped.  An unavailable token that
is not in the pool is an ERROR (the lists do not belong together), never skipped.  Exit: 0 ok - 4 refused.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys

TOOL = "rl_pool_available/1"


def _lines(path):
    if str(path).endswith(".gz"):
        with io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="") as f:
            return f.read().split("\n")
    with open(path, encoding="utf-8", newline="") as f:
        return f.read().split("\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_unavailable(paths):
    """TSVs with a `token` column (rl_fan_dump.py's unavailable_scenes_*.tsv) -> set of tokens."""
    out = set()
    for p in paths:
        ln = _lines(p)
        head = ln[0].rstrip("\r").split("\t")
        if "token" not in head:
            raise ValueError(f"{p}: no 'token' column in {head!r}")
        it = head.index("token")
        for row in ln[1:]:
            row = row.rstrip("\r")
            if row:
                out.add(row.split("\t")[it])
    return out


def filter_pool(lines, unavailable, allow_unknown=False):
    """lines: the pool file split on '\\n' (header first, a trailing '' if the file ends with a newline).  -> (kept lines, removed tokens, unknown tokens)."""
    head = lines[0].rstrip("\r").split("\t")
    if "token" not in head or "log" not in head:
        raise ValueError(f"the pool needs 'token' and 'log' columns, got {head!r}")
    it = head.index("token")
    kept, removed = [lines[0]], []
    for row in lines[1:]:
        if not row.rstrip("\r"):
            kept.append(row)
            continue
        tok = row.rstrip("\r").split("\t")[it]
        if tok in unavailable:
            removed.append(tok)
        else:
            kept.append(row)
    unknown = sorted(unavailable - set(removed))
    if unknown and not allow_unknown:
        raise ValueError(f"{len(unknown)} unavailable token(s) are not in the pool, e.g. {unknown[:3]} -- these lists do not belong together")
    return kept, removed, unknown


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--unavailable", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-unknown", action="store_true")
    a = ap.parse_args(argv)
    try:
        un = read_unavailable(a.unavailable)
        lines = _lines(a.pool)
        kept, removed, unknown = filter_pool(lines, un, a.allow_unknown)
    except (ValueError, OSError) as e:
        print(f"REFUSED: {e}")
        return 4
    text = "\n".join(kept)
    data = text.encode("utf-8")
    if a.out.endswith(".gz"):
        with open(a.out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as g:
            g.write(data)
    else:
        with open(a.out, "wb") as f:
            f.write(data)
    side = {"tool": TOOL, "pool": os.path.basename(a.pool), "pool_text_sha256": sha256_text("\n".join(lines)), "unavailable_tokens": len(un), "removed": len(removed),
            "unknown": unknown, "out": os.path.basename(a.out), "out_text_sha256": sha256_text(text), "rows_in": sum(1 for l in lines[1:] if l.strip()),
            "rows_out": sum(1 for l in kept[1:] if l.strip())}
    with open(a.out + ".available.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(side, f, indent=1)
    print(f"ZZPOOL_AVAILABLE removed {len(removed)} of {side['rows_in']} -> {side['rows_out']} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
