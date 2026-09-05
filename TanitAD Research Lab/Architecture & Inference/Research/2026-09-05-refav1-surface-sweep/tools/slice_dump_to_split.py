#!/usr/bin/env python3
"""Slice a full-grid refav1 dump down to one half of the pre-registered split.

Stage B runs on the SCORE episodes only (the weights were chosen on SELECT), so its dump has
71 episodes while every banked arm has 141. `paired_delta_refav1.py` REFUSES to pair dumps whose
`ws` / `v0` / `g` / `clip_index` are not bit-exact — correctly — so a banked arm must be sliced
before it can be paired against a Stage B arm.

What this does, and what it deliberately does NOT do:

* selects the episodes named in the split, in the SAME sorted-name order the Stage B arm's own
  loader uses (`episode_names()` sorts the cache stems), and renames them `ep000...`;
* **renumbers `clip_index` to the new position**, because the Stage B dump numbers 0..70 while
  the banked dump carries the original 0..140. This is the one field that is rewritten, and it
  is stamped in the sliced manifest as `sliced_from` so it can never be mistaken for original.
* ⛔ **touches no trajectory, no floor and no decision array.** The real join guarantee is
  `ws` / `v0` / `g` bit-exact over the sliced windows, which the paired tool still asserts.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np


def main():
    src, split_p, half, dst = sys.argv[1:5]
    S = json.load(open(split_p, encoding="utf-8"))
    names = set(S[half + "_names"])
    man = json.load(open(os.path.join(src, "manifest.json"), encoding="utf-8"))
    keep = sorted((e["name"], f"ep{e['file_index']:03d}") for e in man["episodes"]
                  if e["name"] in names)
    if len(keep) != len(names):
        raise SystemExit(f"REFUSED: split names {len(names)} but matched {len(keep)}")
    os.makedirs(os.path.join(dst, "decisions"), exist_ok=True)
    out_eps = []
    for new_i, (nm, old) in enumerate(keep):
        with np.load(os.path.join(src, old + ".npz")) as z:
            d = {k: z[k] for k in z.files}
        m = len(d["ws"])
        d["clip_index"] = np.full_like(np.asarray(d["clip_index"]).ravel()[:1], new_i)
        np.savez(os.path.join(dst, f"ep{new_i:03d}.npz"), **d)
        dsrc = os.path.join(src, "decisions", old + ".npz")
        if os.path.exists(dsrc):
            shutil.copyfile(dsrc, os.path.join(dst, "decisions", f"ep{new_i:03d}.npz"))
        out_eps.append({"file_index": new_i, "episode_index": new_i, "name": nm,
                        "clip_id": nm, "n_windows": int(m), "source_file": old})
    man["episodes"] = out_eps
    man["grid"] = dict(man.get("grid", {}))
    man["grid"]["n_episodes"] = len(out_eps)
    man["grid"]["n_windows"] = int(sum(e["n_windows"] for e in out_eps))
    man["sliced_from"] = {
        "source_dump": src, "split_file": split_p, "half": half,
        "n_episodes": len(out_eps),
        "clip_index_rewritten": ("YES — renumbered 0..n-1 to match a Stage B dump built from a "
                                 "71-episode cache. Nothing else is touched; ws/v0/g and every "
                                 "arm array are the originals."),
    }
    json.dump(man, open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8"), indent=1)
    print(f"sliced {src} -> {dst}: {len(out_eps)} episodes / {man['grid']['n_windows']} windows")


if __name__ == "__main__":
    main()
