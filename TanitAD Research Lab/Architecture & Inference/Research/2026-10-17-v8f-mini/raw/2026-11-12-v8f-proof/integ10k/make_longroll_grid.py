"""make_longroll_grid.py --grid IN.json --preroll-steps P --out OUT.json
Restricts the pinned evaluation grid ({"order": [clip ids], "windows": {clip: [t0, ...]}, "n_windows": N, "source": ..., ...}; the pinned tool reads only "order" and "windows") to the
windows whose pre-roll can start inside the clip: t0 - P >= 3. Clips left without a window are dropped from "order" and "windows". Every other field is kept as it is, "n_windows" is
set to the kept count and a block "longroll" {preroll_steps, source_sha256, kept, dropped, min_pre_start} is added. Refuses to overwrite OUT, and refuses when no window is kept.
Standard library only."""
import argparse
import hashlib
import json
import os
import sys

MIN_PRE_START = 3


def restrict(grid, P, lo=MIN_PRE_START):
    """-> (new grid dict without the sha, kept, dropped)."""
    if int(P) < 0:
        raise ValueError("preroll steps must be 0 or more")
    order, wins, kept, dropped = [], {}, 0, 0
    for c in grid["order"]:
        ts = [int(t) for t in grid["windows"][c]]
        k = [t for t in ts if t - int(P) >= lo]
        kept += len(k)
        dropped += len(ts) - len(k)
        if k:
            order.append(c)
            wins[c] = k
    if kept == 0:
        raise ValueError("no window is left with --preroll-steps %d (every t0 - P is below %d)" % (int(P), lo))
    out = dict(grid)
    out["order"], out["windows"], out["n_windows"] = order, wins, kept
    out["longroll"] = {"preroll_steps": int(P), "kept": kept, "dropped": dropped, "min_pre_start": lo}
    return out, kept, dropped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--preroll-steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        sys.exit("REFUSED: %s exists" % a.out)
    raw = open(a.grid, "rb").read()
    g = json.loads(raw.decode("utf-8"))
    try:
        out, kept, dropped = restrict(g, a.preroll_steps)
    except ValueError as e:
        sys.exit("REFUSED: %s" % e)
    out["longroll"]["source_sha256"] = hashlib.sha256(raw).hexdigest()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh)
    print("wrote %s kept=%d dropped=%d clips=%d" % (a.out, kept, dropped, len(out["order"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
