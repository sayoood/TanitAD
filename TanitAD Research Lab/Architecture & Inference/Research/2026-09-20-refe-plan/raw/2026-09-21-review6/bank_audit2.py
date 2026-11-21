import json, os, collections
BASE = r"D:/Projects/TanitAD/data/refe_targets_4cam"
r0 = [json.loads(l) for l in open(BASE+"/targets_rank0.jsonl", encoding="utf-8")]
r1 = [json.loads(l) for l in open(BASE+"/targets_rank1.jsonl", encoding="utf-8")]
k0 = set((r["token"], r["step"]) for r in r0); k1 = set((r["token"], r["step"]) for r in r1)
print(f"(token,step) distinct: rank0={len(k0)} rank1={len(k1)} inter={len(k0&k1)} union={len(k0|k1)}")
print("logs rank0:", sorted(set(r['log_name'] for r in r0)))
print("logs rank1:", sorted(set(r['log_name'] for r in r1)))
# do identical (token,step) rows have identical content?
m0 = {(r["token"], r["step"]): r for r in r0}
diffs = collections.Counter(); same = 0
for r in r1:
    a = m0.get((r["token"], r["step"]))
    if a is None: continue
    b = dict(r); a2 = dict(a); a2.pop("rank"); b.pop("rank")
    if a2 == b: same += 1
    else:
        for k in a2:
            if a2[k] != b.get(k): diffs[k] += 1
print(f"rows identical apart from 'rank': {same} / {len(r1)}; differing fields: {dict(diffs)}")
# sample image paths and existence
print("\nsample image paths:"); 
for p in r0[0]["image"]: print("   ", p)
# existence test
roots = [r"D:/Projects/TanitAD/data/nuplan-camera", r"D:/Projects/TanitAD/data/nuplan"]
exists_abs = collections.Counter(); basename_found = collections.Counter()
# build index of all jpgs on disk once (basename -> path) limited
idx = set()
for root in roots:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.lower().endswith((".jpg",".jpeg")): idx.add(f)
print(f"\nJPEG basenames on disk under {roots}: {len(idx)}")
tot = collections.Counter(); found = collections.Counter()
for r in r0[:200]:
    for c, p in zip(r["cameras"], r["image"]):
        tot[c]+=1
        bn = os.path.basename(p.replace("\\","/"))
        if bn in idx: found[c]+=1
        if os.path.exists(p): exists_abs[c]+=1
print("first 200 rows: per-channel path basenames resolvable on disk:", dict(found), "of", dict(tot))
print("absolute path exists:", dict(exists_abs))
