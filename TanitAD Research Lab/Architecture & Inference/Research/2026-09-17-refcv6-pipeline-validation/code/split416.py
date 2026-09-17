"""Build two DISJOINT halves of the 416x1024 eval cache, for the 139-clip coverage pass.

⛔ WHY A SPLIT AT ALL. `refc_v3_train` REFUSES a run whose train and eval caches share
episodes ("a held-out split that is not held out measures memorisation"). That refusal is
correct and is not worked around: the coverage pass gets genuinely disjoint halves.

⛔ WHY COPIES AND NOT LINKS. D: is **exFAT** -- `os.link` AND `os.symlink` both fail there
with WinError 1 ("Incorrect function"), which is a filesystem limit, not a permission one.
MEASURED 2026-09-17.

The v2 loader globs `*.v2ep.pt` and builds its own `_v2manifest.pt`, so a split directory
needs no `manifest.json` and no other machinery.

The split is sorted-order ALTERNATING, not first-half/second-half, so neither half is
biased toward one end of an id-sorted corpus.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import time

import numpy as np
import torch

SRC = pathlib.Path("D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl")
files = sorted(SRC.glob("*.v2ep.pt"))
assert len(files) == 139, f"expected 139 clips, got {len(files)}"
halves = {"A": files[0::2], "B": files[1::2]}

man = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))
nf = {c["clip_sha12"]: c["n_frames"] for c in man["clips"]}
sha = lambda p: hashlib.sha256(p.name.split(".")[0].encode()).hexdigest()[:12]

t0 = time.time()
for tag, fs in halves.items():
    d = SRC.parent / f"v2ep-eval139-416x1024cyl-split{tag}"
    d.mkdir(exist_ok=True)
    done = nb = 0
    for f in fs:
        t = d / f.name
        if t.exists() and t.stat().st_size == f.stat().st_size:
            done += 1; nb += t.stat().st_size; continue
        shutil.copy2(f, t)
        done += 1; nb += t.stat().st_size
        if done % 20 == 0:
            print(f"  split{tag}: {done}/{len(fs)}  {nb/1e9:.2f} GB  "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"split{tag}: {done}/{len(fs)} clips, {nb/1e9:.2f} GB -> {d.name}", flush=True)

    # ⭐ The eval subset is `randperm(len(e_ds), seed 12345)[:nb*batch]` -- DETERMINISTIC,
    # so which clips it touches is a COMPUTABLE FACT, not a probability. Find the
    # smallest --eval-batches that covers EVERY clip of this half.
    sizes = [nf[sha(f)] - 30 for f in fs]        # windows/clip = n_frames - 30
    tot = sum(sizes)
    owner = torch.tensor([i for i, s in enumerate(sizes) for _ in range(s)])
    perm = torch.randperm(tot, generator=torch.Generator().manual_seed(12345))
    lo = None
    for cand in range(50, 4000, 5):
        if cand * 2 > tot:
            break
        if torch.unique(owner[perm[:cand * 2]]).numel() == len(sizes):
            lo = cand; break
    print(f"  -> {len(sizes)} clips, {tot} windows; smallest --eval-batches (batch 2) "
          f"covering ALL of split{tag}: {lo} ({None if lo is None else lo*2} windows)",
          flush=True)

# the two halves must be disjoint and must reconstitute the whole corpus
A = {p.name for p in (SRC.parent / "v2ep-eval139-416x1024cyl-splitA").glob("*.v2ep.pt")}
B = {p.name for p in (SRC.parent / "v2ep-eval139-416x1024cyl-splitB").glob("*.v2ep.pt")}
whole = {p.name for p in files}
print(f"\nDISJOINT: {not (A & B)}   UNION == 139: {A | B == whole}   |A|={len(A)} |B|={len(B)}")
assert not (A & B) and (A | B) == whole
print(f"total {time.time()-t0:.0f}s")
