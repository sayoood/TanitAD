import json, os, glob, collections
tb = r"D:/Projects/TanitAD/data/refe_targets_4cam"
rows = []
for f in ("targets_rank0.jsonl","targets_rank1.jsonl"):
    rows += [json.loads(l) for l in open(os.path.join(tb,f), encoding="utf-8")]
print(f"target tuples: {len(rows)}")
for sd in ("refe_scorer_targets_full","refe_scorer_targets_r1","refe_scorer_targets"):
    p = os.path.join(r"D:/Projects/TanitAD/data", sd)
    fs = sorted(glob.glob(p+"/*.jsonl"))
    if not fs: print(f"\n{sd}: NO .jsonl (dir exists: {os.path.isdir(p)})"); continue
    sr = []
    for f in fs: sr += [json.loads(l) for l in open(f, encoding="utf-8")]
    ranks = collections.Counter(r.get("rank","MISSING") for r in sr)
    keyf = [k for k in ("log_name","token","step","rank") if k in sr[0]]
    keys = set((r.get("log_name"), r.get("token"), r.get("step"), r.get("rank")) for r in sr)
    cov = sum(1 for r in rows if (r["log_name"], r["token"], r["step"], r["rank"]) in keys)
    print(f"\n{sd}: files {[os.path.basename(x) for x in fs]}")
    print(f"  rows {len(sr):,}  distinct frame-keys {len(keys):,}  rank histogram {dict(ranks)}")
    print(f"  JOIN onto the 2,182 target tuples: {cov} covered = {100*cov/len(rows):.1f} %")
    # rank-blind join (what coverage would be if rank were ignored)
    kb = set((r.get("log_name"), r.get("token"), r.get("step")) for r in sr)
    cb = sum(1 for r in rows if (r["log_name"], r["token"], r["step"]) in kb)
    print(f"  rank-BLIND join: {cb} = {100*cb/len(rows):.1f} %   (the difference is what `rank` costs)")
