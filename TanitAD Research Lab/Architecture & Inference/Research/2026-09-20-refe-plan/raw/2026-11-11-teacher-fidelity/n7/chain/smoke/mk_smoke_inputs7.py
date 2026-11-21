"""builds the navtrain smoke inputs for the TF-N7 build CLI: the 17 navtrain residual tokens + 3 of the 73 already repaired by TF-N6's tier (T0). RJ5-style rolls rows (nt_rolls.jsonl),
N2-R-style and stand-in N5-A plans = the banked P5-R label (nt_plans.npz plans20; all 20 are builder DAC 0, so all are residual), a stand-in N5-A seam, a stand-in harness csv (the
development's builder base rows), the expected residual list and the replay expectation (no banked repair result for navtrain: unchecked, None)."""
import csv, json, sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/n7chain/run/n5")
import os
os.environ.setdefault("N4_RUN", "C:/Users/Admin/n7chain/run")
import n5_env  # noqa
import n4_geom as G
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
D = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/"
t17 = json.load(open(D + "rj5_navtrain/repair_nt/navtrain_residual17.json"))["tokens"]
s73 = sorted(json.load(open(D + "n7_gen/raw/sel73_floor05.json"))["per_token"])[:3]
toks = sorted(set(t17) | set(s73))
refs = {}
import glob
for f in glob.glob("C:/Users/Admin/n7chain/dev/base17_s*.jsonl") + glob.glob("C:/Users/Admin/n7chain/dev/chp73_s*.jsonl"):
    for ln in open(f):
        r = json.loads(ln)
        if r["token"] in toks:
            refs[r["token"]] = r["ref"]
assert set(refs) == set(toks), set(toks) - set(refs)
z = np.load("F:/tanitad_scratch/rj5nt/repair_nt/nt_plans.npz", allow_pickle=False)
ix = {str(t): i for i, t in enumerate(z["tokens"])}
rolls = {}
for ln in open("F:/tanitad_scratch/rj5nt/repair_nt/nt_rolls.jsonl", encoding="utf-8"):
    if ln.strip():
        r = json.loads(ln)
        if r["token"] in set(toks):
            rolls[r["token"]] = r
assert set(rolls) == set(toks)
order = toks
logs = [str(z["logs"][ix[t]]) for t in order]
plans = np.stack([np.asarray(z["plans20"][ix[t]], float) for t in order])
np.savez(out + "/n2r_plans.npz", tokens=np.array(order), logs=np.array(logs), plans20=plans)
np.savez(out + "/n5a_plans.npz", tokens=np.array(order), logs=np.array(logs), plans20=plans, arm=np.array("N5A"))
with open(out + "/rolls.jsonl", "w", encoding="utf-8") as f:
    for t in order:
        f.write(json.dumps({"token": t, "traj20": rolls[t]["traj20"]}) + "\n")
p8 = np.stack([G.ND.to_navsim(p).astype(np.float32) for p in plans])
np.savez(out + "/n5a_seam.npz", token=np.array(order), fingerprint=np.array([f"fp{i}" for i in range(len(order))]), poses=p8, sampling=np.array([8.0, 0.5]), arm=np.array("TEACHER_smoke_refe_tfn5_n5a"))
with open(out + "/n5a_harness.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["", "token", "valid", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"])
    for i, t in enumerate(order):
        b = refs[t]
        w.writerow([i, t, "True", b["nc"], b["dac"], b["ep"], b["ttc"], b["c"], b["ddc"], b["pdms"]])
json.dump({"tokens": sorted(order)}, open(out + "/resid.json", "w"))
json.dump({"d_returns_plan": None, "l1_returns_plan": None, "why": "navtrain smoke: no banked repair result for these tokens (unchecked)"}, open(out + "/replay_expect.json", "w"))
print("wrote", out, len(order), "the 3 from the 73:", s73)
