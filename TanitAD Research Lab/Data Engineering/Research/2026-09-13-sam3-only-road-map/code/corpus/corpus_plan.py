import re, json, gzip, hashlib, glob, os, collections
import truststore; truststore.inject_into_ssl()
import pandas as pd
from huggingface_hub import hf_hub_download
tok = re.search(r"hf_[A-Za-z0-9]+", open(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/Keys.txt", encoding="utf-8", errors="ignore").read()).group(0)
# 1) which v8 labels are newest: HF labels/s2_labels_v8_train.jsonl.gz vs git s2_labels_v8.0_train.jsonl.gz
p = hf_hub_download("Sayood/tanitad-v7-training-corpus", "labels/s2_labels_v8_train.jsonl.gz", repo_type="dataset", token=tok, local_dir="corpus/hf")
hf = {json.loads(l)["clip_id"]: json.loads(l) for l in gzip.open(p, "rt", encoding="utf-8")}
gt = {json.loads(l)["clip_id"]: json.loads(l) for l in gzip.open("v80_train.jsonl.gz", "rt", encoding="utf-8")}
print("HF v8 train:", len(hf), "release field:", collections.Counter(r.get("release") for r in hf.values()).most_common(3), "keys:", len(next(iter(hf.values()))))
print("git v8.0 train:", len(gt), "release field:", collections.Counter(r.get("release") for r in gt.values()).most_common(3), "keys:", len(next(iter(gt.values()))))
print("keys only in HF:", sorted(set(next(iter(hf.values()))) - set(next(iter(gt.values())))), "| only in git:", sorted(set(next(iter(gt.values()))) - set(next(iter(hf.values())))))
common = sorted(set(hf) & set(gt))
diff = collections.Counter()
for c in common:
    for k in set(hf[c]) | set(gt[c]):
        if json.dumps(hf[c].get(k), sort_keys=True) != json.dumps(gt[c].get(k), sort_keys=True):
            diff[k] += 1
print("clips in both:", len(common), "| fields differing (clips):", dict(diff.most_common(12)))
c0 = next(c for c in common if json.dumps(hf[c].get("nav_command"), sort_keys=True) != json.dumps(gt[c].get("nav_command"), sort_keys=True)) if diff.get("nav_command") else None
if c0:
    print("example nav_command HF:", json.dumps(hf[c0]["nav_command"])[:300]); print("example nav_command git:", json.dumps(gt[c0]["nav_command"])[:300])
# 2) production order: BEV-head clips first
idx = pd.read_parquet(r"C:/Users/Admin/tanitad-data/physicalai/clip_index.parquet") if False else None
b1idx = pd.read_parquet(hf_hub_download("Sayood/tanitad-v7-training-corpus", "index/clip_to_chunk.parquet", repo_type="dataset", token=tok, local_dir="corpus/hf"))
cids = list(b1idx.index.astype(str))
sha = {hashlib.sha256(c.encode()).hexdigest()[:12]: c for c in cids}
bev = []
for d in (r"D:/Projects/TanitAD-artifacts/bev-lidar-gt-b1eval-20260913", r"D:/Projects/TanitAD-artifacts/bev-lidar-gt-b1train200-20260913"):
    for f in sorted(glob.glob(os.path.join(d, "*.bevgt.npz"))):
        s = os.path.basename(f).split(".")[0]
        if s in sha:
            bev.append(sha[s])
bev = list(dict.fromkeys(bev))
rest = [c for c in cids if c not in set(bev)]
print("corpus clips:", len(cids), "| valid:", int(b1idx["clip_is_valid"].sum()), "| BEV-head clips found:", len(bev), "| splits:", b1idx["split"].value_counts().to_dict())
order = bev + rest
open("corpus/production_order.txt", "w").write("\n".join(order) + "\n")
print("production order written:", len(order))
