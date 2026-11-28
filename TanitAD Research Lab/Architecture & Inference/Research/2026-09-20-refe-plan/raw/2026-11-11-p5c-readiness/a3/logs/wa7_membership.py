import gzip, json, sys, hashlib, struct
def sha(traj):
    h = hashlib.sha256()
    for p in traj:
        for v in p:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()
wa7 = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
keys = {(r["log_name"], r["token"]): sha(r["traj"]) for r in wa7}
logs = {k[0] for k in keys}
jn = {}
bank_logs = set()
with gzip.open(sys.argv[2], "rt", encoding="utf-8") as f:
    for ln in f:
        d = json.loads(ln)
        bank_logs.add(d["log_name"])
        k = (d["log_name"], d["token"])
        if k in keys:
            jn[k] = (d["status"], d["bank_traj_sha256"], d.get("source"))
n_in = len(jn); n_rj5 = sum(1 for v in jn.values() if v[0] == "rj5"); n_sha = sum(1 for k, v in jn.items() if v[1] == keys[k])
print(json.dumps({"wa7_rows": len(wa7), "wa7_logs": len(logs), "wa7_logs_in_bank": len(logs & bank_logs), "wa7_keys_in_p7_bank_rank0": n_in,
                  "with_rj5_traj": n_rj5, "bank_sha_equals_wa7_traj_sha": n_sha}))
from collections import Counter
print("sources", Counter(v[2] for v in jn.values()))
print("logs", sorted(logs))
