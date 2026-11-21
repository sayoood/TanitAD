"""PREREG_V8F_GRID, the cache check: does EVERY clip of box 2's train and eval caches carry the 8x20 grid?

  python check_grid_cache.py --train <train/manifest.jsonl> --eval <eval/manifest.jsonl> --eval-grid <eval_grid_windows.json> [--n-train 4572] [--out cache_check.json]

The cache builder (pull_encode_pooled.py) appends one manifest line per finished clip: `clip`, `n_frames`, `grids` (the list it encoded, e.g. ["4x10", "8x20"]) and `shapes`
({array name: shape} of the payload it wrote to clips/<clip>.pt). A clip's EFFECTIVE row is its LAST line (the builder's own `load_manifest` and v8f_mini_data.read_manifest
both let the last line win). A trainer at `--grid pool8x20` reads `pool8x20` out of the clip file lazily, so a clip without it crashes the run the first time the sampler
reaches it -- hours in. This checker refuses BEFORE any training:
  * a missing / unreadable manifest, or an unparseable non-empty line;
  * the train manifest holding a number of distinct clips other than --n-train (the join covers exactly the 4,572 train clips);
  * the eval manifest missing any clip of the eval-141 grid (`order` of eval_grid_windows.json);
  * ANY effective row (train, or eval for the clips the eval reads) whose `grids` does not list "8x20" and "4x10", or whose `shapes` does not carry pool8x20 = [n_frames, 160, 1024]
    (and pool4x10 = [n_frames, 40, 1024], the grid the 4x10 base was trained on).
Superseded (non-last) lines are COUNTED, never blocking: they describe a file that was overwritten. Exit 0 = OK, 2 = refused."""
from __future__ import annotations

import argparse
import json
import sys

D_FEAT = 1024
WANT = {"pool4x10": ("4x10", 40), "pool8x20": ("8x20", 160)}


def refuse(msg):
    print(f"GRID CACHE REFUSED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_rows(path, split):
    rows, n_lines = {}, 0
    try:
        fh = open(path, encoding="utf-8")
    except OSError as e:
        refuse(f"the {split} manifest {path} cannot be opened: {e}")
    with fh:
        for k, ln in enumerate(fh, 1):
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
                rows.setdefault(d["clip"], []).append(d)
            except (ValueError, KeyError, TypeError) as e:
                refuse(f"the {split} manifest line {k} is not a manifest row ({type(e).__name__}: {e})")
            n_lines += 1
    if not rows:
        refuse(f"the {split} manifest {path} holds no rows")
    return rows, n_lines


def row_problems(d):
    out = []
    grids = d.get("grids")
    shapes = d.get("shapes")
    if not isinstance(grids, list):
        return ["the row has no `grids` list"]
    n = d.get("n_frames")
    for arr, (g, n_tok) in WANT.items():
        if g not in grids:
            out.append(f"grids {grids} does not list {g}")
        if not isinstance(shapes, dict) or arr not in shapes:
            out.append(f"shapes carries no {arr}")
        elif list(shapes[arr]) != [n, n_tok, D_FEAT]:
            out.append(f"shapes.{arr} = {shapes[arr]}, expected {[n, n_tok, D_FEAT]}")
    return out


def check_split(split, rows, required=None):
    """-> {n_clips, n_checked, n_superseded_lines_without_8x20, bad: [(clip, problems)]} over the EFFECTIVE (last) rows of the clips in `required` (default: all)."""
    names = sorted(rows) if required is None else list(required)
    bad = []
    for c in names:
        if c not in rows:
            bad.append((c, ["the clip has no manifest row"]))
            continue
        p = row_problems(rows[c][-1])
        if p:
            bad.append((c, p))
    sup = sum(1 for c in names if c in rows for d in rows[c][:-1] if "8x20" not in (d.get("grids") or []))
    return {"split": split, "n_clips": len(rows), "n_checked": len(names), "n_superseded_lines_without_8x20": sup, "bad": bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--eval-grid", required=True, help="eval_grid_windows.json: the 141 clips the eval reads (key `order`)")
    ap.add_argument("--n-train", type=int, default=4572)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        tr, tr_lines = read_rows(a.train, "train")
        ev, ev_lines = read_rows(a.eval, "eval")
        try:
            order = json.load(open(a.eval_grid, encoding="utf-8"))["order"]
        except (OSError, ValueError, KeyError) as e:
            refuse(f"the eval grid {a.eval_grid} cannot be read ({type(e).__name__}: {e})")
        if not order:
            refuse("the eval grid lists no clips")
        res_t, res_e = check_split("train", tr), check_split("eval", ev, order)
        problems = []
        if len(tr) != a.n_train:
            problems.append(f"the train manifest holds {len(tr)} distinct clips, not {a.n_train}")
        for r in (res_t, res_e):
            for c, p in r["bad"][:5]:
                problems.append(f"{r['split']} clip {c}: " + "; ".join(p))
            if len(r["bad"]) > 5:
                problems.append(f"... and {len(r['bad']) - 5} more {r['split']} clips")
        out = {"train": {k: v for k, v in res_t.items() if k != "bad"} | {"n_bad": len(res_t["bad"]), "n_lines": tr_lines},
               "eval": {k: v for k, v in res_e.items() if k != "bad"} | {"n_bad": len(res_e["bad"]), "n_lines": ev_lines, "n_grid_clips": len(order)}, "ok": not problems}
        if a.out:
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(out, f, indent=1)
        if problems:
            refuse("; ".join(problems))
        print(f"GRID CACHE OK: {len(tr)} train clips and the {len(order)} eval-grid clips all list 8x20 (+ 4x10) with the expected shapes "
              f"({res_t['n_superseded_lines_without_8x20'] + res_e['n_superseded_lines_without_8x20']} superseded lines without 8x20, not blocking)")
        return 0
    except SystemExit as e:
        return int(e.code or 2)


if __name__ == "__main__":
    sys.exit(main())
