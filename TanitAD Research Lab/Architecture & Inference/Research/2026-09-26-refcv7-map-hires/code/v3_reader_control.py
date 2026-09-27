#!/usr/bin/env python
"""SPEC_REFCV7 §12 item 3 control, run through the NEW-2 READER on a real `/3` export.

For every `<sha12>.sam3mapgt.npz` under --v3-root whose clip is in the --episodes cache and
has a `/2` twin under --v2-root:

* the `/3` file opens at the declared A7 extent (100 m x +-30 m) through every guard;
* the same file is REFUSED under the `/2` extent (60 m x +-16 m) -- never cropped;
* inside the anchored old window, its `fine_codes` AND its `cart_frac` equal the `/2`
  file's byte for byte (`semantic_map_gt_fine.compare_v2_window`), on every frame.

Rows carry sha12 only; a raw clip id is never printed or written. The output JSON is the
evidence (asserted on its counts, never on the exit code).

Usage: v3_reader_control.py --v3-root <dir> --v2-root <dir> --episodes <v2 cache dir>
                            --out <json> [--expect-files N]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tanitad.data import semantic_map_gt as G
from tanitad.data import semantic_map_gt_fine as F
from tanitad.data.perception_targets import MapGTStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-root", required=True, type=Path)
    ap.add_argument("--v2-root", required=True, type=Path)
    ap.add_argument("--episodes", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expect-files", type=int, default=None,
                    help="refuse unless exactly this many /3 files are present")
    a = ap.parse_args(argv)
    by12 = {G.sha12(p.name[:-len(".v2ep.pt")]): p.name[:-len(".v2ep.pt")]
            for p in a.episodes.glob("*.v2ep.pt")}
    files = sorted(a.v3_root.glob("*.sam3mapgt.npz"))
    if a.expect_files is not None and len(files) != int(a.expect_files):
        raise SystemExit(f"{len(files)} /3 files present, --expect-files {a.expect_files}: "
                         f"the export is incomplete or has grown; refusing to report a subset")
    v2store = MapGTStore(a.v2_root)
    t0 = time.time()
    rows = []
    for p3 in files:
        s12 = p3.name[:12]
        if s12 not in by12:
            rows.append({"sha12": s12, "status": "no v2ep clip id"})
            continue
        cid = by12[s12]
        p2 = v2store.resolve(cid)
        if p2 is None:
            rows.append({"sha12": s12, "status": "no /2 twin"})
            continue
        v3 = F.open_path_fine(p3, cid, F.EXTENT_REFCV7)
        v2 = F.open_path_fine(p2, cid)
        idx = np.arange(v3.n_frames)
        with np.load(p3, allow_pickle=False) as z:
            c3 = z["cart_frac"]
        r = F.compare_v2_window(v3.read(idx).codes, F.EXTENT_REFCV7, v2.read(idx).codes,
                                c3, v2.base.cart_u8())
        refused = False
        try:
            F.open_path_fine(p3, cid, F.EXTENT_V2)
        except F.FineSpecMismatch:
            refused = True
        rows.append({"sha12": s12, "frames": int(v3.n_frames),
                     "n_cells": int(r["n_cells"]), "n_differ": int(r["n_differ"]),
                     "n_cart_cells": int(r["n_cart_cells"]),
                     "n_cart_differ": int(r["n_cart_differ"]),
                     "v2_extent_refused": bool(refused)})
    ok = [r for r in rows if "n_cells" in r]
    rec = {"what": ("SPEC_REFCV7 §12 item 3 control, run by the NEW-2 READER on the real "
                    f"/3 export ({a.v3_root.as_posix()})"),
           "script": "code/v3_reader_control.py",
           "n_files_present": len(rows), "n_files_compared": len(ok),
           "n_not_compared": len(rows) - len(ok),
           "frames": sum(r["frames"] for r in ok),
           "fine_cells": sum(r["n_cells"] for r in ok),
           "fine_differ": sum(r["n_differ"] for r in ok),
           "cart_cells": sum(r["n_cart_cells"] for r in ok),
           "cart_differ": sum(r["n_cart_differ"] for r in ok),
           "v2_extent_refused_all": all(r["v2_extent_refused"] for r in ok),
           "elapsed_s": round(time.time() - t0, 1), "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_bytes(json.dumps(rec, indent=1, ensure_ascii=False).encode("utf-8"))
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
