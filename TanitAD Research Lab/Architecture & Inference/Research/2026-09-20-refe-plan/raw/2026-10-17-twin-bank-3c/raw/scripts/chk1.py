import json, hashlib, glob, collections
P = "/workspace/data/refe_navtrain/train_grow_cmd/targets_rank1.jsonl"
mini = {}
for l in open("/workspace/ops/reroll_probe/bank_in/targets_rank1.jsonl"):
    r = json.loads(l); mini[(r["log_name"], r["token"], int(r["aug"]["value"]))] = l
n = same = 0
tokens = set((k[0], k[1]) for k in mini)
extra_goal = 0; kinds = collections.Counter()
for l in open(P):
    r = json.loads(l); a = r.get("aug")
    kind = a.get("kind") if isinstance(a, dict) else None
    kinds[kind] += 1
    if kind == "lane_rank":
        k = (r["log_name"], r["token"], int(a["value"]))
        if k in mini:
            n += 1; same += (mini[k] == l)
    elif (r["log_name"], r["token"]) in tokens:
        extra_goal += 1
print("kinds", dict(kinds)); print("mini twins found in prod", n, "byte-identical line", same, "goal rows on mini frames", extra_goal)
# sample row keys
r = json.loads(next(iter(mini.values()))); print(sorted(r.keys())); print(r["aug"])
rr = glob.glob("/workspace/ops/reroll_probe/rr_h6/s*/targets_reroll.jsonl"); print(rr, sum(1 for p in rr for _ in open(p)))
