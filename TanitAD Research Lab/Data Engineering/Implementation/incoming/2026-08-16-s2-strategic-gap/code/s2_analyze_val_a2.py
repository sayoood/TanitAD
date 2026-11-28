"""S2 gap review — fused_w120val (600) + Alpamayo-2 parquet (23,644 rows)."""
import glob
import json
import os
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_pull")
OUT = {}

# ---- fused_w120val --------------------------------------------------------- #
files = sorted(glob.glob(os.path.join(ROOT, "fused_w120val", "*.json")))
recs = []
for f in files:
    if os.path.basename(f).startswith("_"):
        continue
    recs.append(json.load(open(f, encoding="utf-8")))
OUT["val_n_files"] = len(files)
OUT["val_n_records"] = len(recs)
OUT["val_summary"] = json.load(open(
    os.path.join(ROOT, "fused_w120val", "_summary.json"), encoding="utf-8"))

OUT["val_g_str_dist"] = dict(Counter(
    r["vocab"]["g_str"]["token"] for r in recs).most_common())
OUT["val_g_str_src"] = dict(Counter(
    r["vocab"]["g_str"].get("src") for r in recs))
OUT["val_corrob_by_route"] = dict(Counter(
    str(r["vocab"]["g_str"].get("corroborated_by_route")) for r in recs))

verb_clip = Counter()
for r in recs:
    acts = ((r.get("semantics") or {}).get("symbols") or {}).get("actions") or []
    for v in {a.get("verb") for a in acts}:
        verb_clip[v] += 1
OUT["val_a_str_verbs_clips"] = dict(verb_clip.most_common())

# the 4 silent-empty perception records: fused with s3 = {} -> tracks == []
# AND per_concept_hits None AND no 'absent' marker
empty = []
zero_track_with_hits = 0
for r in recs:
    p = r.get("perception") or {}
    if p.get("absent"):
        continue
    if not p.get("tracks") and p.get("per_concept_hits") is None:
        empty.append(r["clip_id"])
    elif not p.get("tracks"):
        zero_track_with_hits += 1
OUT["val_silent_empty_perception"] = empty
OUT["val_zero_tracks_but_hits_present"] = zero_track_with_hits
emg = []
for r in recs:
    if r["clip_id"] in empty:
        emg.append({
            "clip": r["clip_id"][:20],
            "g_str": r["vocab"]["g_str"]["token"],
            "goal_evidence": (r.get("corroboration") or {}).get("goal_evidence"),
            "census_vs_scene": (r.get("corroboration") or {})
            .get("census_vs_scene"),
            "road_type": ((r.get("semantics") or {}).get("scene") or {})
            .get("road_type")})
OUT["val_silent_empty_details"] = emg

# ROUTE_TO evidence sign kinds on val
rt_kind = Counter()
for r in recs:
    if r["vocab"]["g_str"]["token"] != "ROUTE_TO":
        continue
    sym = (r.get("semantics") or {}).get("symbols") or {}
    ev = sym.get("goal_evidence_sign")
    signs = ((r.get("semantics") or {}).get("signs") or {}).get("signs") or []
    kind = signs[ev].get("kind") if isinstance(ev, int) and ev < len(signs) \
        else None
    rt_kind[kind] += 1
OUT["val_route_to_sign_kinds"] = dict(rt_kind)

# schema field census on 30-record sample (deterministic)
sample = recs[:30]
OUT["val_sample_vocab_keys"] = sorted(
    {k for r in sample for k in r["vocab"]["g_str"]})

# ---- Alpamayo-2 parquet ---------------------------------------------------- #
import pandas as pd

df = pd.read_parquet(os.path.join(ROOT, "records.parquet"))
OUT["a2_rows"] = int(len(df))
OUT["a2_clips"] = int(df["clip_id"].nunique())
OUT["a2_columns"] = list(df.columns)
OUT["a2_tasks"] = {k: int(v) for k, v in
                   df["task"].value_counts().to_dict().items()}
OUT["a2_error_nonnull"] = int(df["error"].notna().sum()) \
    if "error" in df.columns else None

ma = df[df["task"] == "meta_action"] if "meta_action" in set(df["task"]) \
    else df[df["task"].str.contains("meta", case=False, na=False)]
OUT["a2_meta_task_name"] = sorted(set(ma["task"]))
OUT["a2_meta_rows"] = int(len(ma))
texts = []
for _, row in ma.head(2000).iterrows():
    rj = row.get("raw_json")
    texts.append(rj if isinstance(rj, str) else json.dumps(rj))
tok = Counter()
for t in texts:
    tl = (t or "").lower()
    for w in ("turn left", "turn right", "lane change", "left lane",
              "right lane", "slow", "stop", "accelerate", "decelerate",
              "exit", "merge", "yield", "straight", "speed"):
        if w in tl:
            tok[w] += 1
OUT["a2_meta_phrase_census_first2000"] = dict(tok.most_common())
OUT["a2_meta_samples"] = [t[:300] for t in texts[:8]]

json.dump(OUT, open(os.path.join(ROOT, "..", "val_a2_analysis.json"), "w"),
          indent=1)
print(json.dumps({k: v for k, v in OUT.items()
                  if k not in ("a2_meta_samples", "val_silent_empty_details")},
                 indent=1)[:6000])
print("EMPTY_DETAILS", json.dumps(OUT["val_silent_empty_details"], indent=1))
print("A2_SAMPLES", json.dumps(OUT["a2_meta_samples"][:4], indent=1)[:2500])
