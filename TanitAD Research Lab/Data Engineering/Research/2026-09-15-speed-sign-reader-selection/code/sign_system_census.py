"""Country / sign-system census of our clip sets (0 GPU). A reader must cover every sign system its clips carry."""
import gzip, json, re, sys
import pandas as pd
META = "C:/Users/Admin/tanitad-data/physicalai/metadata/data_collection.parquet"
LR = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-04-v72-label-release/raw/"
CAND = "D:/Projects/TanitAD/TanitAD Research Lab/Data Engineering/Research/2026-09-13-posted-speed-limit-supplier/raw/"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
out = sys.argv[1]
d = pd.read_parquet(META)
d.index = d.index.str.lower()
MUTCD = {"United States"}          # rectangular "SPEED LIMIT NN" (mph); every other country here is a Vienna-Convention signatory (red circle, km/h)
def system(c): return "US_MUTCD_mph" if c in MUTCD else "Vienna_circle_kmh"
res = {"meta": META, "corpus_clips": int(len(d)),
       "corpus_by_system": d["country"].map(system).value_counts().to_dict(),
       "corpus_by_country": d["country"].value_counts().to_dict(), "sets": {}}
for name, p in [("v7.2_train", LR + "s2_labels_v7.2_train.jsonl.gz"), ("v7.2_eval", LR + "s2_labels_v7.2_eval.jsonl.gz")]:
    ids, n = set(), 0
    for line in gzip.open(p, "rt", encoding="utf-8"):
        n += 1
        m = UUID.findall(line.lower())
        ids.update(m)
    hit = [i for i in ids if i in d.index]
    sub = d.loc[hit]
    res["sets"][name] = {"records": n, "unique_clip_ids": len(ids), "matched_in_meta": len(hit),
                         "by_system": sub["country"].map(system).value_counts().to_dict(),
                         "by_country_top8": sub["country"].value_counts().head(8).to_dict(),
                         "by_hour_night_frac(hour<6 or >=20)": round(float(((sub.hour_of_day < 6) | (sub.hour_of_day >= 20)).mean()), 4)}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res["corpus_by_system"]), json.dumps({k: (v["unique_clip_ids"], v["matched_in_meta"], v["by_system"]) for k, v in res["sets"].items()}))
