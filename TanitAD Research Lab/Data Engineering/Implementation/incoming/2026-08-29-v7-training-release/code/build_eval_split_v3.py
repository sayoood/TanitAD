"""eval_split_v3 — ONE split at the PI's 3 % budget. Supersedes v1 and v2.

PI: *"hopefully small because we need max training data, i think 3% is
sufficient"*. v1 (240) + v2 (200) = 440 = **9.34 %**, three times over.

⭐ THE BUDGET CHANGES THE DESIGN, NOT JUST THE SIZE. 3 % of 4,713 eligible is
**141 clips**. Splitting that into two disjoint splits gives ~85 + ~56, and BOTH
are underpowered — 56 clips is very thin for checkpoint selection. **ONE split of
141 is strictly more powerful than two of 85 and 56**, and because it is a single
split there is no independence problem in evaluating it at two cadences: the
monitoring curve and the selection curve are then the SAME curve read at
different intervals, which is simpler and cheaper as well as stronger.

⇒ v3 is PURELY PROPORTIONAL, with NO competence floors. Floors at n=141 would
distort the aggregate — the number selection depends on — and would STILL not buy
enough per-competence power to be worth it (a floor of 25 on stop-launch turns
11.8 % into 17.7 % and still leaves 25 clips). Better an honest representative
141 than a distorted one that is underpowered anyway.

⚠️ THE CONSEQUENCE, STATED RATHER THAN DISCOVERED: at 141 clips a representative
split holds ~17 stop-launch and ~23 highway clips. **PER-COMPETENCE numbers from
this split are not admissible.** Competence questions need a separate targeted
eval RUN — which costs GPU time, not training data, and so does not compete with
the 3 % budget.

Constraints, all ASSERTED: episode-disjoint · val40-disjoint by digest on both
sides · fixed and seeded.
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path("C:/Users/Admin/tanitad-wt/_s2build/release")
OUT = REL / "v71"
# local cache: G: flaps, and a val40 read failing mid-build would be worse
# than a stale one — the digests are immutable, so caching is safe here.
VAL40 = Path("C:/Users/Admin/tanitad-wt/_s2build/release/val40_digests.json")
SEED = 7
PCT = 3.0

rows = [json.loads(x) for x in gzip.open(OUT / "s2_labels_v7.1.jsonl.gz",
                                         "rt", encoding="utf-8") if x.strip()]
df = pd.DataFrame([{"clip_id": r["clip_id"], **(r.get("strata") or {})} for r in rows])
v40 = set(json.loads(VAL40.read_text(encoding="utf-8"))["clip_id_digests"])
df["digest"] = df.clip_id.map(lambda c: hashlib.sha256(c.encode("utf-8")).hexdigest())
elig = df[~df.digest.isin(v40)].copy()
TARGET = int(round(len(elig) * PCT / 100))
print(f"eligible {len(elig)} | {PCT}% = {TARGET} clips")

# --- proportional over road_class x daynight, deterministic -------------------
picked: list[str] = []
p = (elig.groupby(["road_class", "daynight_clock"]).size() / len(elig))
for (rc, dn), frac in p.items():
    sub = elig[(elig.road_class == rc) & (elig.daynight_clock == dn)]
    take = min(int(round(frac * TARGET)), len(sub))
    if take:
        picked += list(sub.sample(n=take, random_state=SEED).clip_id)
rest = elig[~elig.clip_id.isin(picked)]
while len(picked) < TARGET and len(rest):
    picked.append(rest.sample(n=1, random_state=SEED + len(picked)).clip_id.iloc[0])
    rest = elig[~elig.clip_id.isin(picked)]
picked = picked[:TARGET]

ev = elig[elig.clip_id.isin(picked)].copy()
tr = elig[~elig.clip_id.isin(picked)].copy()
print(f"EVAL {len(ev)} | TRAIN {len(tr)} | held out {len(ev)/len(elig)*100:.2f}%")


def comp(d, c):
    return (d[c].value_counts(normalize=True) * 100).round(1).to_dict()


print("\n=== v3 vs CORPUS ===")
for c in ("road_class", "daynight_clock"):
    print(f"  {c:15s} v3 {comp(ev,c)}")
    print(f"  {'':15s} co {comp(elig,c)}")
sl = float(ev.has_stop_launch_20s.mean()) * 100
print(f"  stop->launch    v3 {sl:.1f}%  corpus {elig.has_stop_launch_20s.mean()*100:.1f}%")
print(f"  ⚠️ absolute counts: stop-launch {int(ev.has_stop_launch_20s.sum())} | "
      f"highway {int((ev.road_class=='highway').sum())} | "
      f"intersection {int((ev.road_class=='intersection').sum())}")
print(f"  cells occupied  {ev.strata_cell.nunique()} of {elig.strata_cell.nunique()}")

assert not (set(ev.clip_id) & set(tr.clip_id)), "episode leak"
assert not (set(ev.digest) & v40), "val40 overlap in eval"
assert not (set(tr.digest) & v40), "val40 overlap in train"
assert len(ev) == TARGET
print("\n✅ episode-disjoint · val40-disjoint (both sides) · exact n")

payload = {
    "schema": "tanitad_eval_split/1", "version": "v7.1-eval-split-3",
    "supersedes": ["v7.1-eval-split-1", "v7.1-eval-split-2-representative"],
    "seed": SEED,
    "based_on_labels_md5": hashlib.md5((OUT / "s2_labels_v7.1.jsonl.gz").read_bytes()).hexdigest(),
    "n_eval": len(ev), "n_train": len(tr),
    "held_out_pct_of_eligible": round(len(ev) / len(elig) * 100, 2),
    "constraints": ["episode-disjoint (clip = episode)",
                    "disjoint from deployed val40 by sha256 digest, both sides",
                    "proportional to the corpus — NO competence floors",
                    "FIXED and seeded — comparable to itself across checkpoints"],
    "composition": {"v3": {"road_class": comp(ev, "road_class"),
                           "daynight_clock": comp(ev, "daynight_clock"),
                           "stop_launch_pct": round(sl, 1),
                           "stop_launch_clips": int(ev.has_stop_launch_20s.sum()),
                           "highway_clips": int((ev.road_class == "highway").sum()),
                           "intersection_clips": int((ev.road_class == "intersection").sum()),
                           "cells_occupied": int(ev.strata_cell.nunique())},
                    "corpus": {"road_class": comp(elig, "road_class"),
                               "daynight_clock": comp(elig, "daynight_clock"),
                               "stop_launch_pct": round(
                                   float(elig.has_stop_launch_20s.mean()) * 100, 1)}},
    "interpretation": {
        "kind": "REPRESENTATIVE — one split, both cadences",
        "admissible_use": ("the AGGREGATE val curve, at any cadence. Because it is "
                           "ONE split, the every-100-steps monitoring curve and the "
                           "save-interval selection curve are the SAME curve read at "
                           "different intervals — no independence question arises."),
        "inadmissible_use": ("⛔ PER-COMPETENCE numbers. At n=141 the split holds "
                             "~17 stop-launch and ~23 highway clips; a per-stratum "
                             "figure from this split is underpowered and must not be "
                             "quoted. Competence questions need a separate targeted "
                             "eval RUN, which costs GPU time rather than training "
                             "data. Also NOT a substitute for the deployed val40."),
        "cadence": "every 100 steps for monitoring AND at save-interval for selection",
    },
    "why_one_split_not_two": (
        "At the PI's 3 % budget, two disjoint splits would be ~85 and ~56 clips and "
        "BOTH underpowered. One split of 141 is strictly stronger, and a single split "
        "read at two cadences has no independence problem."),
    "per_cell_counts": ev.strata_cell.value_counts().to_dict(),
    "eval_clip_ids": sorted(ev.clip_id), "train_clip_ids": sorted(tr.clip_id),
}
q = OUT / "eval_split_v3.json"
json.dump(payload, open(q, "w"), indent=1)
print(f"wrote {q.name} ({q.stat().st_size/1e6:.2f} MB) "
      f"sha256 {hashlib.sha256(q.read_bytes()).hexdigest()[:16]}")
