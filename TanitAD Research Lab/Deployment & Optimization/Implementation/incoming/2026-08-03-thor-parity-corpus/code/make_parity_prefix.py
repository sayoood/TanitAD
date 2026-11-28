#!/usr/bin/env python3
"""Assemble a CONTIGUOUS, COMPLETE sorted PREFIX of the parity train corpus while
the full transfer is still in flight.

WHY A PREFIX, AND WHY IT IS NOT A RE-SELECTION
----------------------------------------------
``parity.check_uids(mode="subset")`` admits exactly one non-full episode set: the
sorted PREFIX of the manifest's uid list. That is the shape ``--episodes N``
produces, and it is refused the moment a single foreign, renumbered or out-of-order
episode appears. So this is the ONLY subset that keeps the guard meaningful — a
deliberate, self-labelling truncation of the canonical corpus, never a re-selection
of it (which CLAUDE.md §Invariants requires us to refuse).

⛔ IT IS NOT STRICT PARITY. Every number off a prefix run is a THROUGHPUT or
WIRING fact. It is not cross-arm comparable with the full-corpus arms, and the
guard says so itself: ``check_uids`` prints a loud ``SUBSET`` line naming the
shortfall.

⚠️ Files are HARDLINKED (0 bytes) but SIZE-CHECKED first, against the HF
expectation table. A partially-downloaded episode has the RIGHT NAME — which is
exactly what the parity name-check cannot see, and exactly the failure that put a
92,299,264 B ``ep_00028.pt`` in a directory everything reported as healthy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() /
                    "epcache/epcache-256px-phase0/physicalai-train-e438721ae894"))
    ap.add_argument("--dst-root", default=str(Path.home() / "epcache_prefix"))
    ap.add_argument("--expected", default=str(Path.home() /
                    "parity_verify/hf_expected_train.json"))
    ap.add_argument("--val-src", default=str(Path.home() /
                    "valdata/physicalai-val-0c5f7dac3b11"))
    ap.add_argument("--max", type=int, default=0, help="0 = as many as are complete")
    a = ap.parse_args(argv)

    src = Path(a.src)
    exp = json.loads(Path(a.expected).read_text())["files"]
    d = Path(a.dst_root) / "physicalai-train-e438721ae894"
    if d.exists():
        for f in d.glob("ep_*.pt"):
            f.unlink()
    d.mkdir(parents=True, exist_ok=True)

    n, stop = 0, None
    while True:
        if a.max and n >= a.max:
            stop = f"--max {a.max} reached"
            break
        name = f"ep_{n:05d}.pt"
        p = src / name
        if not p.exists():
            stop = f"{name} not downloaded yet"
            break
        got, want = p.stat().st_size, int(exp[name]["size"])
        if got != want:
            stop = (f"{name} INCOMPLETE on disk: {got} B vs {want} B expected "
                    f"(short by {want - got})")
            break
        os.link(p, d / name)
        n += 1

    v = Path(a.dst_root) / "physicalai-val-0c5f7dac3b11"
    if not v.exists():
        os.symlink(a.val_src, v)

    print(json.dumps({
        "prefix_episodes": n, "stopped_because": stop, "dir": str(d),
        "bytes": sum(f.stat().st_size for f in d.glob("ep_*.pt")),
        "val_link": str(v),
        "note": "SUBSET of the canonical corpus — not strict parity, not "
                "cross-arm comparable"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
