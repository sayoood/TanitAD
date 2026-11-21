"""refcv7 NEW-2 -- the fine reader's analytic control, as a banked record.

On every eval-kit GT file whose clip id is known from a v2ep file name (never printed):
open through ``semantic_map_gt_fine.open_path_fine`` (every guard), read every 50th frame,
and compare ``block_fraction_u8(fine_codes)`` with ``cart_frac`` on all 9 channels. Also the
discriminating arm: the same comparison with the 10 cm map mirrored left-right must differ.
Writes ``--out`` (JSON); sha12 only.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tanitad.data import semantic_map_gt as G
from tanitad.data import semantic_map_gt_fine as F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--v2-dir", type=Path, required=True)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    by12 = {G.sha12(p.name[:-len(".v2ep.pt")]): p.name[:-len(".v2ep.pt")]
            for p in a.v2_dir.glob("*.v2ep.pt")}
    files = sorted(a.gt_dir.glob(f"*{G.GT_SUFFIX}"))
    n_files = n_frames = n_cells = n_bad = 0
    mirror_min, codes_seen = None, set()
    skipped = []
    for p in files:
        s12 = p.name[:-len(G.GT_SUFFIX)]
        if s12 not in by12:
            skipped.append(s12)
            continue
        c = F.open_path_fine(p, by12[s12])
        idx = np.arange(0, c.n_frames, a.every)
        fine = c.read(idx).codes
        cart = c.base.cart_u8()[idx]
        n_bad += int((F.block_fraction_u8(fine) != cart).sum())
        n_cells += int(cart.size)
        mb = int((F.block_fraction_u8(fine[:, :, ::-1]) != cart).sum())
        mirror_min = mb if mirror_min is None else min(mirror_min, mb)
        codes_seen |= set(np.unique(fine).tolist())
        n_files += 1
        n_frames += int(idx.size)
    rec = {"what": "cart_frac == 5x5 block fraction of fine_codes (rint(count*255/25)), "
                   "all 9 channels, through the fine reader's guards",
           "n_files": n_files, "n_files_skipped_no_clip_id": len(skipped),
           "every_nth_frame": a.every, "n_frames": n_frames,
           "n_cells_compared": n_cells, "n_cells_differing": n_bad,
           "mirrored_min_differing_cells_per_file": mirror_min,
           "codes_seen": sorted(codes_seen), "elapsed_s": round(time.time() - t0, 1)}
    a.out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
