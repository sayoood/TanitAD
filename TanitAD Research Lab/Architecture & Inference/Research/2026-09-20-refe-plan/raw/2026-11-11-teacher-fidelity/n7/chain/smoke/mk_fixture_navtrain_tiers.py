import json, glob, hashlib, sys
import numpy as np
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n7_gen")
import n7_analyze as A, n7_select as S
files = sorted(glob.glob("C:/Users/Admin/n7chain/dev/*.jsonl"))
rows = A.derive_caps(A.load([f for f in files if "chp73" not in f]))
want = {"T0_N6": "69b54b1998f05c13", "T1_L2": "faf94859c4095b79", "T2_STR": "fc61cadc28715436", "T3_OFF": "24bcf45bcc6f5dfb", "T4_DEEP": "8f0f005a5f6b58e6"}
z = np.load("F:/tanitad_scratch/rj5nt/repair_nt/nt_plans.npz", allow_pickle=False)
ix = {str(t): i for i, t in enumerate(z["tokens"])}
rolls = {}
need = set(want.values())
for ln in open("F:/tanitad_scratch/rj5nt/repair_nt/nt_rolls.jsonl", encoding="utf-8"):
    if ln.strip():
        r = json.loads(ln)
        if r["token"] in need:
            rolls[r["token"]] = r
out = {"what": "one real NAVTRAIN token per cascade tier (the 17 residual tokens of rj5_navtrain/repair_nt/navtrain_residual17.json); expect = the tier / lever / k / param the n7_gen development selected "
               "(n7_gen/raw/sel17_floor05.json) and the sha256 of the chosen plan (float64 bytes); t20 = RJ5 target row (nt_rolls.jsonl), label = banked P5-R label (nt_plans.npz plans20)",
       "instances": {}}
for tier, tok in want.items():
    row = rows[tok]
    c = S.cascade(row, floor=0.5)
    assert c["tier"] == tier, (tok, c["tier"], tier)
    plan = [x for x in row["levers"][c["lever"]] if x["k"] == c["k"]][0]["plan"]
    out["instances"][tier] = {"token": tok, "log": str(z["logs"][ix[tok]]), "t20": rolls[tok]["traj20"], "label": np.asarray(z["plans20"][ix[tok]], float).tolist(),
                              "expect": {"tier": c["tier"], "lever": c["lever"], "k": c["k"], "param": c["param"]},
                              "plan_sha256": hashlib.sha256(np.ascontiguousarray(plan, float).tobytes()).hexdigest()}
    assert rolls[tok]["log"] == out["instances"][tier]["log"]
json.dump(out, open("E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n7_chain/fixture_navtrain_tiers.json", "w", encoding="utf-8"), indent=1)
print({t: (v["token"], v["expect"]) for t, v in out["instances"].items()})
