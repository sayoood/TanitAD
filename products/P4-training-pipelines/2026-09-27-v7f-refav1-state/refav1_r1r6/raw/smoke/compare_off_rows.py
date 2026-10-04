"""The OFF real-data smoke, run through the UNMODIFIED c36b6ddd trainer and through the R1-R6
trainer: every train_log row (minus wall-clock) and the final checkpoint must be bit-identical."""
import json, sys
import torch
a_dir, b_dir = sys.argv[1], sys.argv[2]
rows = lambda d: [json.loads(x) for x in open(f"{d}/train_log.jsonl", encoding="utf-8") if x.strip()]
ra, rb = rows(a_dir), rows(b_dir)
for r in ra + rb:
    r.pop("elapsed_s", None)
ca = torch.load(f"{a_dir}/ckpt.pt", map_location="cpu", weights_only=False)
cb = torch.load(f"{b_dir}/ckpt.pt", map_location="cpu", weights_only=False)
same_model = (list(ca["model"]) == list(cb["model"])
              and all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"]))
sa, sb = ca["opt"]["state"], cb["opt"]["state"]
same_opt = (set(sa) == set(sb) and all(torch.equal(sa[i][k], sb[i][k]) if torch.is_tensor(sa[i][k])
                                       else sa[i][k] == sb[i][k] for i in sa for k in sa[i]))
print(json.dumps({"n_rows": [len(ra), len(rb)], "rows_identical": ra == rb,
                  "ckpt_model_identical": same_model, "ckpt_opt_identical": same_opt,
                  "n_model_tensors": len(ca["model"])}))
