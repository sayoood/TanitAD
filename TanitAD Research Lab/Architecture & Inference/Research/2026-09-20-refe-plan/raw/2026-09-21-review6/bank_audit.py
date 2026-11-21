import json, collections, os, sys
BASE = r"D:/Projects/TanitAD/data/refe_targets_4cam"
rows = {}
for rk in ("targets_rank0.jsonl", "targets_rank1.jsonl"):
    p = os.path.join(BASE, rk)
    rr = [json.loads(l) for l in open(p, encoding="utf-8")]
    rows[rk] = rr
    print(f"{rk}: {len(rr)} rows")
r0, r1 = rows["targets_rank0.jsonl"], rows["targets_rank1.jsonl"]
print("keys:", sorted(r0[0].keys()))
# identity fields
for k in sorted(r0[0].keys()):
    v = r0[0][k]
    print(f"  {k}: type={type(v).__name__}", (f"len={len(v)}" if hasattr(v,'__len__') and not isinstance(v,str) else f"val={str(v)[:90]}"))
print()
# overlap by token
def key(r):
    for cand in ("token","sample_token","lidar_pc_token","ts","timestamp"):
        if cand in r: return (cand, r[cand])
    return ("image0", r["image"][0] if isinstance(r.get("image"), list) else r.get("image"))
k0 = set(key(r)[1] for r in r0); k1 = set(key(r)[1] for r in r1)
print("key field:", key(r0[0])[0])
print(f"distinct keys rank0={len(k0)} rank1={len(k1)} intersection={len(k0&k1)} union={len(k0|k1)}")
# channel paths per row
nch = collections.Counter()
distinct_paths = collections.Counter()
cams_field = collections.Counter()
ego_w = collections.Counter(); goal_w = collections.Counter()
chan_tokens = collections.Counter()
for r in r0 + r1:
    im = r.get("image")
    if isinstance(im, list):
        nch[len(im)] += 1
        distinct_paths[len(set(im))] += 1
        for p in im:
            for c in ("CAM_F0","CAM_L0","CAM_R0","CAM_B0"):
                if c in p: chan_tokens[c]+=1
    else:
        nch["str"] += 1
    cams_field[tuple(r.get("cameras")) if r.get("cameras") else None] += 1
    ego_w[len(r.get("ego", []))] += 1
    goal_w[len(r.get("goal", []))] += 1
print("images-per-row:", dict(nch))
print("DISTINCT paths per row:", dict(distinct_paths))
print("cameras field:", {str(k): v for k, v in cams_field.items()})
print("ego widths:", dict(ego_w)); print("goal widths:", dict(goal_w))
print("channel token counts across all paths:", dict(chan_tokens))
# stats files
for f in ("targets_rank0_stats.json","targets_rank1_stats.json"):
    d = json.load(open(os.path.join(BASE,f), encoding="utf-8"))
    print(f"\n--- {f} ---")
    print(json.dumps(d, indent=1)[:2500])
