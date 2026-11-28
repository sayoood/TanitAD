"""D3 scoping probe 2: what is in the candidate pool, and how many gate hits should a D3 draw expect?  (read-only, CPU, seconds)

Pool = PhysicalAI-AV clips (local clip_index.parquet, 306,152 rows) MINUS the 4,719 B1 clips whose front-camera mp4 is on this box.
Digest oracles are the repo's own (`stack/tanitad/data/*_clip_digests.json`, sha256(clip_id) hex, ids never printed):
  deployed val40  -> `parity.guard_corpus_build(role="train")` REFUSES (the published open-loop statistic's episodes)
  parity train    -> `physicalai-train-e438721ae894` (2,400 digests; informational for a supervision corpus, a hazard for a held-out one)
  v7.2 eval (147) -> the eval-141 episodes live here (all inside B1 by construction)
Also: how many chunk zips a random draw touches (each costs one 256 KiB central-directory range read).
Run:  python -I probe_pool_gates.py > probe_pool_gates.out
"""
import hashlib
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BS = chr(92)
PRE = BS + BS + "?" + BS


def lp(path):
    p = os.path.normpath(os.path.abspath(path))
    return PRE + p if os.name == "nt" and not p.startswith(PRE) else p


EXPORT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))          # .../hist
DIG = os.path.join(EXPORT, "stack", "tanitad", "data")
CLIP_INDEX = "C:/Users/Admin/tanitad-data/physicalai/clip_index.parquet"
CAM = "C:/Users/Admin/tanitad-data/physicalai/camera/camera_front_wide_120fov"


def digs(name):
    return set(json.load(open(lp(os.path.join(DIG, name)), encoding="utf-8"))["clip_id_digests"])


val40, par, ev147 = digs("deployed_val40_clip_digests.json"), digs("parity_train_clip_digests.json"), digs("v72_eval_clip_digests.json")
print(f"digest oracles: val40 {len(val40)}, parity-train {len(par)}, v7.2-eval {len(ev147)}")

idx = pd.read_parquet(CLIP_INDEX)
b1 = set(f[:-4] for f in os.listdir(CAM) if f.endswith(".mp4"))
dg = {c: hashlib.sha256(c.encode("utf-8")).hexdigest() for c in idx.index}
print(f"clip_index {len(idx)}; B1 mp4s on disk {len(b1)}; B1 in clip_index {len(b1 & set(idx.index))}")


def census(ids, label):
    d = [dg[c] for c in ids]
    print(f"{label:34s} n={len(ids):>7}  in val40 {sum(x in val40 for x in d):>3}  in parity-train {sum(x in par for x in d):>5}"
          f"  in v7.2-eval {sum(x in ev147 for x in d):>3}")


census(sorted(b1), "B1 (4,719)")
pool = idx.drop(index=list(b1 & set(idx.index)))
census(list(pool.index), "pool (306,152 - B1)")
for sp in ("train", "val", "test"):
    census(list(pool.index[pool["split"] == sp]), f"pool, NVIDIA split={sp}")
print("NVIDIA split of the 4,719 B1 clips:", idx.loc[sorted(b1 & set(idx.index)), "split"].value_counts().to_dict(),
      "-> B1 itself ignores NVIDIA's split; the programme's eval-141 is a B1 subset")

n_val_in_pool = sum(dg[c] in val40 for c in pool.index)
for n in (4512, 40000):
    print(f"expected deployed-val40 hits in a uniform draw of {n}: {n * n_val_in_pool / len(pool):.2f}  "
          f"(the gate must run; it REFUSES, mode='refuse', or drops, mode='exclude')")

g = pool.groupby("chunk").size()
print(f"pool: {pool['chunk'].nunique()} chunks, clips/chunk mean {g.mean():.1f} (min {g.min()}, max {g.max()})")
for n in (4512, 40000):
    s = pool.sample(n, random_state=20261025)
    print(f"uniform draw of {n}: touches {s['chunk'].nunique()} distinct chunks (each = 1 central-directory range read, 256 KiB)")
print("alternative (whole chunks): 4,512 clips = about %d chunks; 40,000 clips = about %d chunks" % (round(4512 / g.mean()), round(40000 / g.mean())))
