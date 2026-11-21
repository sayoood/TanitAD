"""builds the navtrain smoke inputs for the TF-N6 build CLI from the 7 real navtrain DAC-0 instances of the TF-N5 smoke (DEV30): RJ5-style rolls rows, N2-R-style plans
(the base label R of each t20), a stand-in N5-A plan file (= the base label: all 7 are 'residual'), its seam, a stand-in harness csv (the TF-N5 smoke's builder base rows),
the expected residual list and the replay expectation (L1 certified every one of them in the TF-N5 smoke: N5-R 7/7)."""
import glob, gzip, json, os, sys
os.environ.setdefault("N4_RUN", "C:/Users/Admin/n6nt/run")
sys.path.insert(0, "C:/Users/Admin/n6nt/run/n5"); sys.path.insert(0, "C:/Users/Admin/n6nt/run/n6")
import n5_env  # noqa
import numpy as np
import n4_geom as G
T = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n5_l1/"
rows5 = [json.loads(l) for l in open(T + "chain/smoke/derive.jsonl", encoding="utf-8") if l.strip()]
toks = {r["token"]: (r["log"], r["base"]) for r in rows5}
dev = {}
for l in gzip.open(T + "raw/dev30.jsonl.gz", "rt", encoding="utf-8"):
    r = json.loads(l)
    if r["token"] in toks:
        dev[r["token"]] = r
print("tokens", len(toks), "with t20", len(dev), flush=True)
assert set(dev) == set(toks), set(toks) - set(dev)
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
by = {}
for t, (lg, _b) in toks.items():
    by.setdefault(lg, []).append(t)
order, plans, logs = [], [], []
for lg, tk, cx in G.contexts(by):
    t20 = np.asarray(dev[tk]["t20"], float)
    order.append(tk); logs.append(lg); plans.append(cx.labels(t20)["R"])
plans = np.stack(plans)
np.savez(out + "/n2r_plans.npz", tokens=np.array(order), logs=np.array(logs), plans20=plans)
np.savez(out + "/n5a_plans.npz", tokens=np.array(order), logs=np.array(logs), plans20=plans, arm=np.array("N5A"))
with open(out + "/rolls.jsonl", "w", encoding="utf-8") as f:
    for t in order:
        f.write(json.dumps({"token": t, "traj20": dev[t]["t20"]}) + "\n")
p8 = np.stack([G.ND.to_navsim(p).astype(np.float32) for p in plans])
np.savez(out + "/n5a_seam.npz", token=np.array(order), fingerprint=np.array([f"fp{i}" for i in range(len(order))]), poses=p8, sampling=np.array([8.0, 0.5]),
         arm=np.array("TEACHER_smoke_refe_tfn5_n5a"))
import csv
with open(out + "/n5a_harness.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"])
    for i, t in enumerate(order):
        b = toks[t][1]
        w.writerow([i, t, "True", b["nc"], b["dac"], b["ep"], b["ttc"], b["c"], b["ddc"], b["pdms"]])
json.dump({"tokens": sorted(order)}, open(out + "/resid.json", "w"))
json.dump({"d_returns_plan": None, "l1_returns_plan": sorted(order), "why": "TF-N5 smoke: N5-R (L1, full certificate) certified 7 of 7 of these navtrain instances; REPAIR-D has no banked smoke result (unchecked)"}, open(out + "/replay_expect.json", "w"))
print("wrote", out, len(order))
