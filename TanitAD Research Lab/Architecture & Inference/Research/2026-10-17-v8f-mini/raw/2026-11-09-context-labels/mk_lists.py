import json, sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strat_common as C
t = pd.read_parquet("splits/clips_table.parquet")
tr = t[(t.split == "train4512") & t.video_local].copy()
tr["h"] = [C.stable_hash(c, "smoke") for c in tr.clip_id]
sm = []
for _, g in tr.sort_values("h").groupby("country"):
    sm.append(g.clip_id.iloc[0])
sm = sm[:20]
json.dump({"clips": sm}, open("smoke/smoke20.json", "w"))
for sp in ("train4512", "eval"):
    ids = sorted(t[(t.split == sp) & t.video_local].clip_id)
    json.dump({"clips": ids}, open(f"splits/clips_{sp}.json", "w"))
    print(sp, len(ids))
print(len(sm), t[t.clip_id.isin(sm)].country.nunique())
