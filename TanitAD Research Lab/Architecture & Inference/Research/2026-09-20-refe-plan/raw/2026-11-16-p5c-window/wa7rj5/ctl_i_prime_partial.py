"""SUPPLEMENT (not the pinned control (i')): what of (i') can be checked WITHOUT the pod's rekeyed records (labels_rekeyed.jsonl, pod-only).
Compares this run's no-A7 P5-R labels to the C1 table labels on the 1,018 rows (table key log|token|0|sha256(bank traj)).
Reports R-TOL-1-style per-row max abs diff (median <= 1e-3, p99 <= 5e-3) and exact-equality count.  sim8 / TB-6 criteria need the records: NOT evaluated.
Also: sim8 agreement on the rows present in the dev-box local RJ5 records (label_local), reported as context only.
"""
import gzip, json, hashlib, sys
import numpy as np
run = sys.argv[1]
table_path = sys.argv[2]
local = sys.argv[3]
out = sys.argv[4]


def tsha(t):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(t, np.float64)).tobytes()).hexdigest()


rows = {}
for ln in open(f"{run}/bank_real/targets_rank0.jsonl", encoding="utf-8"):
    d = json.loads(ln)
    rows[(d["log_name"], d["token"])] = np.asarray(d["traj"], np.float64)
labs = {}
for ln in open(f"{run}/labels/labels_s0of1.jsonl", encoding="utf-8"):
    r = json.loads(ln)
    labs[(r["log"], r["token"])] = r
table = json.load(gzip.open(table_path, "rt", encoding="utf-8"))["table"]
diffs, exact, miss, errs, tc_diff = [], 0, 0, 0, []
for k, bt in rows.items():
    L = labs[k]
    if "p5r_error" in L or "p5r" not in L:
        errs += 1
        continue
    t = table.get(f"{k[0]}|{k[1]}|0|{tsha(bt)}")
    if t is None:
        miss += 1
        continue
    a, b = np.asarray(L["p5r"], np.float64), np.asarray(t, np.float64)
    diffs.append(float(np.abs(a - b).max()))
    exact += int(np.array_equal(a, b))
d = np.array(diffs)
res = {"n_rows": len(rows), "n_compared": len(d), "n_table_key_missing": miss, "n_p5r_error_this_run": errs, "n_bit_equal": exact,
       "max_abs_diff": {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max())},
       "R_TOL_1_style_pass(median<=1e-3,p99<=5e-3)": bool(np.median(d) <= 1e-3 and np.percentile(d, 99) <= 5e-3)}
# sim8 on local-record rows (context)
loc = {}
for fp in (f"{local}/labels_s0of2.jsonl", f"{local}/labels_s1of2.jsonl"):
    for ln in open(fp, encoding="utf-8"):
        r = json.loads(ln)
        loc[(r["log"], r["token"], r.get("traj_sha256"))] = r
n = agree = 0
for k, L in labs.items():
    fed = L.get("traj_sha256")
    r = loc.get((k[0], k[1], fed))
    if r is not None and "p5r_sim8" in r and "p5r_sim8" in L:
        n += 1
        agree += int(bool(r["p5r_sim8"]) == bool(L["p5r_sim8"]))
res["local_records_sim8_context"] = {"n_rows_in_local_records": n, "n_sim8_equal": agree}
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
