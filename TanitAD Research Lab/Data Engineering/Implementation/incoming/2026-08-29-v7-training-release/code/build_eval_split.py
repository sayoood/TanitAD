"""A first-class, shipped EVAL SPLIT for v7.1 — stratified, fixed, leak-checked.

PI: a val round every 100 steps. That is worthless without a defined held-out
split shipped WITH the corpus rather than derived by each trainer.

⛔ THE THREE HARD CONSTRAINTS
 1. EPISODE-DISJOINT. Each clip is one episode here, so a clip appears in exactly
    one side. Windows within an episode are not independent — that is why the
    programme's estimator is an episode-cluster bootstrap — and sharing an
    episode across train/val is a leak that reads as skill.
 2. DISJOINT FROM THE DEPLOYED VAL40, by DIGEST (`deployed_val40_clip_digests.json`),
    never by provenance. 6 B1 clips are in val40; an eval split overlapping it
    would make the in-training number and THE published open-loop number
    non-independent (C6 family).
 3. STRATIFIED ON THE SHIPPED CELLS, with FLOORS on the rare competences. A
    uniform 5 % draw would take ~11.8 % stop-launch by chance and could contain
    almost none of the competence we are most worried about.

⭐ THE DESIGN TRADE, DECIDED BY MEASUREMENT, NOT PREFERENCE. The alternative was
drawing the split from the ~301,000 parent clips we do not train on — costing no
training data and giving a genuinely held-out distribution. **Refused: 301,423 of
those clips are UNLABELLED** (only 4,729 carry Alpamayo labels). A val round that
scores tactical/strategic decisions needs labels; an unlabelled split could only
score trajectory/reconstruction — the exact axis DINO-WM measured as NOT tracking
control competence (92x data: +4 % prediction, 11.5x control). So the split is
carved from the labelled corpus, and it costs training data. That cost is real
and is stated rather than hidden.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
BUNDLE = REL / "tanitad-v7-training-corpus"
VAL40 = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/tanitad/data"
             "/deployed_val40_clip_digests.json")
OUT = REL / "v71"
SEED = 0
TARGET = 240              # small on purpose: it runs every 100 steps
FLOOR_STOP_LAUNCH = 40    # >= this many clips containing a full stop->launch
FLOOR_HIGHWAY = 30
FLOOR_INTERSECTION = 30

rows = [json.loads(x) for x in gzip.open(OUT / "s2_labels_v7.1.jsonl.gz",
                                         "rt", encoding="utf-8") if x.strip()]
df = pd.DataFrame([{"clip_id": r["clip_id"], **(r.get("strata") or {})} for r in rows])
print(f"v7.1 rows {len(df)}")

# --- constraint 2: drop the deployed val40 overlap, by digest ---------------
v40 = set(json.loads(VAL40.read_text(encoding="utf-8"))["clip_id_digests"])
df["digest"] = df.clip_id.map(lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest())
overlap = df[df.digest.isin(v40)]
df = df[~df.digest.isin(v40)].copy()
print(f"excluded deployed-val40 overlap: {len(overlap)} -> eligible {len(df)}")

rng = np.random.default_rng(SEED)
picked: set[str] = set()


def take(pool: pd.DataFrame, n: int, why: str):
    pool = pool[~pool.clip_id.isin(picked)]
    if len(pool) == 0 or n <= 0:
        return
    take_n = min(n, len(pool))
    sel = pool.sample(n=take_n, random_state=SEED)
    picked.update(sel.clip_id)
    print(f"  +{take_n:4d} {why} (pool {len(pool)})")


# --- constraint 3: floors FIRST, so the rare competences cannot be crowded out
take(df[df.has_stop_launch_20s == True], FLOOR_STOP_LAUNCH, "stop->launch floor")
take(df[df.road_class == "highway"], FLOOR_HIGHWAY, "highway floor")
take(df[df.road_class == "intersection"], FLOOR_INTERSECTION, "intersection floor")

# --- then proportional over the occupied cells, round-robin so rare cells get in
cells = df.strata_cell.value_counts()
order = list(cells.index)
rng.shuffle(order)
while len(picked) < TARGET:
    added = False
    for cell in order:
        if len(picked) >= TARGET:
            break
        pool = df[(df.strata_cell == cell) & (~df.clip_id.isin(picked))]
        if len(pool):
            picked.add(pool.sample(n=1, random_state=SEED + len(picked)).clip_id.iloc[0])
            added = True
    if not added:
        break

split = df[df.clip_id.isin(picked)].copy()
train = df[~df.clip_id.isin(picked)].copy()
print(f"\nEVAL SPLIT {len(split)} | TRAIN {len(train)} | eligible {len(df)}")

# --- the split states its own composition -----------------------------------
def comp(d, col):
    return (d[col].value_counts(normalize=True) * 100).round(1).to_dict()


print("\n=== composition: EVAL vs remaining TRAIN ===")
for col in ("road_class", "daynight_clock"):
    print(f"  {col:16s} eval {comp(split,col)}")
    print(f"  {'':16s} train {comp(train,col)}")
sl_e = float(split.has_stop_launch_20s.mean()) * 100
sl_t = float(train.has_stop_launch_20s.mean()) * 100
print(f"  stop->launch     eval {sl_e:.1f}%  train {sl_t:.1f}%  (corpus 11.8%)")
print(f"  cells occupied   eval {split.strata_cell.nunique()} / train {train.strata_cell.nunique()}")

# --- leak assertions ---------------------------------------------------------
assert not (set(split.clip_id) & set(train.clip_id)), "EPISODE LEAK: clip in both sides"
assert not (set(split.digest) & v40), "VAL40 OVERLAP in the eval split"
assert not (set(train.digest) & v40), "VAL40 OVERLAP in train"
assert split.has_stop_launch_20s.sum() >= FLOOR_STOP_LAUNCH, "stop-launch floor missed"
print("\n✅ assertions: episode-disjoint · val40-disjoint (both sides) · floors met")

cell_counts = split.strata_cell.value_counts().to_dict()
payload = {
    "schema": "tanitad_eval_split/1", "version": "v7.1-eval-split-1", "seed": SEED,
    "based_on_labels_md5": hashlib.md5((OUT / "s2_labels_v7.1.jsonl.gz").read_bytes()).hexdigest(),
    "n_eval": len(split), "n_train": len(train),
    "excluded_deployed_val40": len(overlap),
    "constraints": ["episode-disjoint (clip = episode)",
                    "disjoint from deployed val40 by sha256 digest",
                    "stratified on strata_cell with competence floors",
                    "FIXED — the same episodes every val round, so the curve is "
                    "comparable across checkpoints rather than re-sampled noise"],
    "floors": {"stop_launch": FLOOR_STOP_LAUNCH, "highway": FLOOR_HIGHWAY,
               "intersection": FLOOR_INTERSECTION},
    "composition": {"eval": {"road_class": comp(split, "road_class"),
                             "daynight_clock": comp(split, "daynight_clock"),
                             "stop_launch_pct": round(sl_e, 1),
                             "cells_occupied": int(split.strata_cell.nunique())},
                    "train": {"road_class": comp(train, "road_class"),
                              "daynight_clock": comp(train, "daynight_clock"),
                              "stop_launch_pct": round(sl_t, 1),
                              "cells_occupied": int(train.strata_cell.nunique())}},
    # ⚠️ THE INTERPRETATION RULE, shipped WITH the split so it cannot be read off.
    "interpretation": {
        "kind": "STRESS / DIAGNOSTIC split — NOT representative of the corpus",
        "why": ("Cell-uniform stratification plus competence floors deliberately "
                "over-weights the rare strata: eval is 46.7% intersection / 32.1% "
                "highway / 21.2% urban against a corpus that is 67.2% urban, and "
                "31.2% stop-launch against 10.8%. With 240 clips over 143 occupied "
                "cells the split is close to one-clip-per-cell, which IS uniform-"
                "over-cells and is what makes rare competences visible."),
        "admissible_use": ("tracking whether a competence is IMPROVING across "
                           "checkpoints — the split is FIXED, so the curve is "
                           "comparable to itself over time."),
        "inadmissible_use": ("quoting its absolute value as corpus or deployment "
                             "performance. It is harder than the corpus BY "
                             "CONSTRUCTION and will read worse than reality. It is "
                             "also NOT the deployed val40 and must never be "
                             "substituted for THE published open-loop statistic."),
    },
    "per_cell_counts": cell_counts,
    "eval_clip_ids": sorted(split.clip_id), "train_clip_ids": sorted(train.clip_id),
}
p = OUT / "eval_split_v1.json"
json.dump(payload, open(p, "w"), indent=1)
print(f"wrote {p.name} ({p.stat().st_size/1e6:.2f} MB) "
      f"sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
