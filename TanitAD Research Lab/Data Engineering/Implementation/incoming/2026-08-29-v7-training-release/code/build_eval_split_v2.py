"""eval_split_v2 — the REPRESENTATIVE split, shipped alongside the v1 stress split.

They answer different questions and one file cannot honestly do both:
  * v1 (STRESS, cell-uniform + competence floors) -> "is the competence MOVING".
    Sensitive instrument, monitoring cadence (every 100 steps).
  * v2 (REPRESENTATIVE, proportional to the corpus) -> "WHERE ARE WE". A
    checkpoint-SELECTION question, and selection wants the corpus distribution or
    we would pick the checkpoint best at rare competences rather than best
    overall. Save-interval cadence.

⛔ CONSTRAINTS, all ASSERTED rather than arranged:
  1. episode-disjoint (clip = episode)
  2. disjoint from the deployed val40, by DIGEST, on both sides
  3. ⭐ DISJOINT FROM v1 — if a clip sat in both, the two curves would not be
     independent and a reader would compare them as though they were
  4. fixed and seeded, so each curve is comparable to itself across checkpoints
  5. proportional to the CORPUS distribution, not to the post-v1 remainder (v1
     deliberately drained the rare cells, so sampling the remainder would
     under-represent them relative to the corpus we actually train on)
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
OUT = REL / "v71"
VAL40 = Path("G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack/tanitad/data"
             "/deployed_val40_clip_digests.json")
SEED = 1
TARGET = 200          # lean on purpose: held-out clips are training clips we lose

rows = [json.loads(x) for x in gzip.open(OUT / "s2_labels_v7.1.jsonl.gz",
                                         "rt", encoding="utf-8") if x.strip()]
df = pd.DataFrame([{"clip_id": r["clip_id"], **(r.get("strata") or {})} for r in rows])
v40 = set(json.loads(VAL40.read_text(encoding="utf-8"))["clip_id_digests"])
df["digest"] = df.clip_id.map(lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest())
eligible = df[~df.digest.isin(v40)].copy()

v1 = json.load(open(OUT / "eval_split_v1.json"))
v1_ids = set(v1["eval_clip_ids"])
pool = eligible[~eligible.clip_id.isin(v1_ids)].copy()
print(f"eligible {len(eligible)} | v1 holds {len(v1_ids)} | v2 pool {len(pool)}")

# --- proportional to the CORPUS, over road_class x daynight -----------------
# (strata_cell has 146 occupied cells for 200 clips — too fine to be proportional
#  at this n, so the practical axes are used and the cell coverage is REPORTED.)
corpus_p = (eligible.groupby(["road_class", "daynight_clock"]).size()
            / len(eligible))
picked: list[str] = []
rng = np.random.default_rng(SEED)
for (rc, dn), p in corpus_p.items():
    want = int(round(p * TARGET))
    sub = pool[(pool.road_class == rc) & (pool.daynight_clock == dn)]
    take = min(want, len(sub))
    if take:
        picked += list(sub.sample(n=take, random_state=SEED).clip_id)
# top up / trim to exactly TARGET without disturbing proportions much
rest = pool[~pool.clip_id.isin(picked)]
while len(picked) < TARGET and len(rest):
    picked.append(rest.sample(n=1, random_state=SEED + len(picked)).clip_id.iloc[0])
    rest = pool[~pool.clip_id.isin(picked)]
picked = picked[:TARGET]

v2 = eligible[eligible.clip_id.isin(picked)].copy()
train = eligible[~eligible.clip_id.isin(set(picked) | v1_ids)].copy()
print(f"\nv2 {len(v2)} | v1 {len(v1_ids)} | TRAIN {len(train)} of {len(eligible)} eligible")


def comp(d, col):
    return (d[col].value_counts(normalize=True) * 100).round(1).to_dict()


print("\n=== v2 vs CORPUS (should track closely) ===")
for col in ("road_class", "daynight_clock"):
    print(f"  {col:16s} v2     {comp(v2, col)}")
    print(f"  {'':16s} corpus {comp(eligible, col)}")
sl2 = float(v2.has_stop_launch_20s.mean()) * 100
slc = float(eligible.has_stop_launch_20s.mean()) * 100
print(f"  stop->launch     v2 {sl2:.1f}%  corpus {slc:.1f}%")
print(f"  cells occupied   v2 {v2.strata_cell.nunique()} of {eligible.strata_cell.nunique()}")

# --- ASSERTIONS --------------------------------------------------------------
assert not (set(v2.clip_id) & v1_ids), "⛔ v2 OVERLAPS v1 — the curves would not be independent"
assert not (set(v2.clip_id) & set(train.clip_id)), "episode leak: clip in v2 and train"
assert not (set(v2.digest) & v40), "v2 overlaps the deployed val40"
assert not (set(train.digest) & v40), "train overlaps the deployed val40"
assert len(v2) == TARGET, (len(v2), TARGET)
print("\n✅ assertions: disjoint from v1 · episode-disjoint · val40-disjoint (both sides) · exact n")

payload = {
    "schema": "tanitad_eval_split/1", "version": "v7.1-eval-split-2-representative",
    "seed": SEED,
    "based_on_labels_md5": hashlib.md5((OUT / "s2_labels_v7.1.jsonl.gz").read_bytes()).hexdigest(),
    "n_eval": len(v2), "n_train_remaining": len(train),
    "disjoint_from": {"eval_split_v1": True, "deployed_val40": True},
    "constraints": ["episode-disjoint (clip = episode)",
                    "disjoint from deployed val40 by sha256 digest, both sides",
                    "disjoint from eval_split_v1 (asserted, not arranged)",
                    "proportional to the CORPUS distribution, not the post-v1 remainder",
                    "FIXED and seeded — comparable to itself across checkpoints"],
    "composition": {"v2": {"road_class": comp(v2, "road_class"),
                           "daynight_clock": comp(v2, "daynight_clock"),
                           "stop_launch_pct": round(sl2, 1),
                           "cells_occupied": int(v2.strata_cell.nunique())},
                    "corpus": {"road_class": comp(eligible, "road_class"),
                               "daynight_clock": comp(eligible, "daynight_clock"),
                               "stop_launch_pct": round(slc, 1),
                               "cells_occupied": int(eligible.strata_cell.nunique())}},
    "interpretation": {
        "kind": "REPRESENTATIVE split — tracks the corpus distribution",
        "admissible_use": ("CHECKPOINT SELECTION and 'where are we' — it is "
                           "proportional to the corpus, so the best checkpoint on "
                           "it is the best checkpoint overall rather than the best "
                           "at rare competences."),
        "inadmissible_use": ("substituting for THE deployed val40 published "
                             "open-loop statistic — this is a different split with "
                             "a different n; and reading it as sensitive to rare "
                             "competences, which is what eval_split_v1 is for. "
                             "⚠️ At n=200 the rare strata carry FEW clips (see "
                             "cells_occupied) so per-stratum numbers from this "
                             "split are underpowered — use v1 for those."),
        "cadence": "save-interval, NOT every 100 steps (v1 is the monitoring split)",
    },
    "training_data_cost": {
        "held_out_total": len(v1_ids) + len(v2),
        "pct_of_eligible": round((len(v1_ids) + len(v2)) / len(eligible) * 100, 2),
        "note": ("Held-out clips are training clips we lose, and data is the "
                 "scarcest resource in the efficiency thesis. Both splits are "
                 "deliberately lean for that reason."),
    },
    "per_cell_counts": v2.strata_cell.value_counts().to_dict(),
    "eval_clip_ids": sorted(v2.clip_id), "train_clip_ids": sorted(train.clip_id),
}
p = OUT / "eval_split_v2.json"
json.dump(payload, open(p, "w"), indent=1)
print(f"\nwrote {p.name} ({p.stat().st_size/1e6:.2f} MB) "
      f"sha256 {hashlib.sha256(p.read_bytes()).hexdigest()[:16]}")
print(f"TOTAL HELD OUT: {len(v1_ids)+len(v2)} = "
      f"{(len(v1_ids)+len(v2))/len(eligible)*100:.2f}% of eligible")
