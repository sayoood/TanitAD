"""compare the model weights, the optimiser state and the per-step log of two trainer runs (bit for bit)."""
import json
import sys

import torch

a, b = sys.argv[1], sys.argv[2]
ca, cb = (torch.load(f"{x}/ckpt_last.pt", map_location="cpu", weights_only=False) for x in (a, b))
n_model = sum(1 for n in ca["model"] if torch.equal(ca["model"][n], cb["model"][n]))
print("model tensors equal", n_model, "of", len(ca["model"]), "| key sets equal", sorted(ca) == sorted(cb), "| extra keys in B:", sorted(set(cb) - set(ca)))
oa, ob = ca["opt"]["state"], cb["opt"]["state"]
same_opt = all(torch.equal(oa[k]["exp_avg"], ob[k]["exp_avg"]) and torch.equal(oa[k]["exp_avg_sq"], ob[k]["exp_avg_sq"]) for k in oa)
print("optimiser moments equal", same_opt)
la = [json.loads(x) for x in open(f"{a}/train_log.jsonl")]
lb = [json.loads(x) for x in open(f"{b}/train_log.jsonl")]
strip = lambda L: [{k: v for k, v in r.items() if k != "sec"} for r in L]
print("train_log (minus wall seconds) equal", strip(la) == strip(lb), "steps", len(la), "final total", la[-1]["total"], lb[-1]["total"])
sa, sb = json.load(open(f"{a}/summary.json")), json.load(open(f"{b}/summary.json"))
print("summary keys only in B:", sorted(set(sb) - set(sa)), "| only in A:", sorted(set(sa) - set(sb)))
common = [k for k in sa if k in sb and k not in ("seconds", "levers", "argv_sha")]
print("summary values equal on the shared keys:", all(sa[k] == sb[k] for k in common))
