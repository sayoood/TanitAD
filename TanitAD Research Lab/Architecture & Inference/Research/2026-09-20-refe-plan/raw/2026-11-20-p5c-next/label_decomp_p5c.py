#!/usr/bin/env python3
"""Where do C1's label roughness and speed come from? Rank-0 rows of the P5-C C1 label table (p5r-rj5, sha256 3b198314...), navtrain.

Three trajectories per rank-0 row (92,896 rows), all [20,3] at 0.2 s, ego frame:
  bank  = the bank `traj` (C0's WTA label)              F:/tanitad_scratch/rj5nt/bank_p7_rank0_slim.jsonl.gz
  rj5   = the RJ5 optimised-driver rollout (pre-inversion) F:/tanitad_scratch/rj5nt/join_c/rj5_bank_join.jsonl.gz
  label = the P5-R tracker-inverted RJ5 label (C1's label) F:/tanitad_scratch/rj5nt/label_a33/P5C_label_table_p5r-rj5_rank0.json.gz
Split of label - bank into CONTENT (rj5 - bank: what the optimised driver changes) and INVERSION (label - rj5: what the tracker
inversion adds). Roughness = p5c_table_gates.learn() (per-row max|acc|, max|heading rate|, origin prepended, DT 0.2); speed = path / 4 s.
CONTROL (exit 1 if it fails): label/bank ratio of medians must reproduce the banked TB-6 rank-0 value 1.323 (acc) / 1.705 (hr)
(LABEL_SELECTION.md, gates/main/gates.json) to 5e-4, and the sha of the table file must be 3b198314...
Point values, navtrain rank-0 rows, no tracker. Usage: label_decomp_p5c.py OUT.json"""
import gzip, hashlib, json, sys
import numpy as np

TAB = "F:/tanitad_scratch/rj5nt/label_a33/P5C_label_table_p5r-rj5_rank0.json.gz"
BANK = "F:/tanitad_scratch/rj5nt/bank_p7_rank0_slim.jsonl.gz"
JOIN = "F:/tanitad_scratch/rj5nt/join_c/rj5_bank_join.jsonl.gz"
DT = 0.2


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kin(q):
    q = np.asarray(q, np.float64)
    Q = np.concatenate([np.zeros((q.shape[0], 1, 3)), q], 1)
    acc = np.linalg.norm(np.diff(Q[:, :, :2], n=2, axis=1), axis=2) / DT ** 2
    hr = np.degrees(np.abs(np.diff(np.unwrap(Q[:, :, 2], axis=1), axis=1)) / DT)
    jerk = np.linalg.norm(np.diff(Q[:, :, :2], n=3, axis=1), axis=2) / DT ** 3
    path = np.linalg.norm(np.diff(Q[:, :, :2], axis=1), axis=2).sum(1)
    v1 = np.linalg.norm(Q[:, 1, :2], axis=1) / DT
    return dict(acc=acc.max(1), hr=hr.max(1), jerk=jerk.max(1), vmean=path / (DT * 20), v1=v1, x_end=Q[:, -1, 0])


tsha = sha(TAB)
T = json.load(gzip.open(TAB, "rt"))["table"]
B = {}
for ln in gzip.open(BANK, "rt"):
    r = json.loads(ln)
    B[(r["log_name"], r["token"])] = r["traj"]
keys, bank, rj5, lab = [], [], [], []
for ln in gzip.open(JOIN, "rt"):
    r = json.loads(ln)
    if r.get("status") != "rj5" or r["rank"] != 0:
        continue
    k = r["bank_key"]
    bt = B.get((r["log_name"], r["token"]))
    if bt is None or k not in T:
        continue
    bb = np.asarray(bt, np.float64)
    if hashlib.sha256(bb.astype("<f8").tobytes()).hexdigest() != r["bank_traj_sha256"]:
        continue
    keys.append(k); bank.append(bb); rj5.append(r["rj5_traj"]); lab.append(T[k])
bank, rj5, lab = (np.asarray(x, np.float64) for x in (bank, rj5, lab))
K = {n: kin(x) for n, x in (("bank", bank), ("rj5", rj5), ("label", lab))}
med = lambda x: float(np.median(x))
rom = lambda a, b, m: med(K[a][m]) / med(K[b][m])
res = {"n_rank0_rows_joined": len(keys), "table_sha256": tsha,
       "medians": {n: {m: med(v) for m, v in K[n].items()} for n in K},
       "ratio_of_medians": {f"{a}/{b}": {m: rom(a, b, m) for m in ("acc", "hr", "jerk")} for a, b in (("label", "bank"), ("rj5", "bank"), ("label", "rj5"))},
       "speed_mps_mean_diff": {"rj5-bank (content)": float((K["rj5"]["vmean"] - K["bank"]["vmean"]).mean()),
                               "label-rj5 (inversion)": float((K["label"]["vmean"] - K["rj5"]["vmean"]).mean()),
                               "label-bank (total)": float((K["label"]["vmean"] - K["bank"]["vmean"]).mean())},
       "x_end_m_mean_diff": {"rj5-bank": float((K["rj5"]["x_end"] - K["bank"]["x_end"]).mean()),
                             "label-rj5": float((K["label"]["x_end"] - K["rj5"]["x_end"]).mean())},
       "v1_mps_mean_diff": {"rj5-bank": float((K["rj5"]["v1"] - K["bank"]["v1"]).mean()), "label-rj5": float((K["label"]["v1"] - K["rj5"]["v1"]).mean())},
       "share_rows_rj5_faster_than_bank_pct": float((K["rj5"]["vmean"] > K["bank"]["vmean"] + 0.05).mean() * 100),
       "share_rows_rj5_slower_than_bank_pct": float((K["rj5"]["vmean"] < K["bank"]["vmean"] - 0.05).mean() * 100)}
lb = res["ratio_of_medians"]["label/bank"]
ok = tsha.startswith("3b198314721fc7e7") and abs(lb["acc"] - 1.323) < 5e-4 and abs(lb["hr"] - 1.705) < 5e-4 and len(keys) == 92896
res["control_reproduces_banked_TB6_rank0"] = bool(ok)
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
