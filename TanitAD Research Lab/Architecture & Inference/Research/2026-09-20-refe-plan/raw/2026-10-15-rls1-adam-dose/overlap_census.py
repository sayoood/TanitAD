import json, glob, os
CACHE = "E:/Projects/TanitAD/data/refe_trunk_cache/d7c59f4f"
ck = {}
for sp in ("navtrain_train", "navtrain_val", "navtest_sub200"):
    m = json.load(open(f"{CACHE}/{sp}/manifest.json"))
    ck[sp] = set(m["keys"])
    print(sp, m["n"], len(m["keys"]), {k: m[k] for k in m if k not in ("keys","ckpt_meta","checkpoint")}.keys())
def keys_of(pat):
    out = set(); 
    for fp in sorted(glob.glob(pat)):
        for line in open(fp, encoding="utf-8"):
            d = json.loads(line)
            if "key" in d:
                k = d["key"]; out.add("|".join(str(x) for x in k))
    return out
tr = keys_of("C:/Users/Admin/rls1b_dl/data/x/pdm_train/pdm_train/*.jsonl")
va = keys_of("C:/Users/Admin/rls1b_dl/data/x/pdm_val/pdm_val/*.jsonl")
print("RL-train sets", len(tr), "RL-val sets", len(va))
allc = set().union(*ck.values())
ft = lambda s: {"|".join(k.split("|")[:2]) for k in s}
print("example cache key", next(iter(allc)), "example rl key", next(iter(tr)))
for nm, ks in (("RL-train", tr), ("RL-val", va)):
    for sp, cs in ck.items():
        print(nm, "x", sp, "exact", len(ks & cs), "frame(log|token)", len(ft(ks) & ft(cs)))
