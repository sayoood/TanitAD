import sys, json, time
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/adh_decomp/scripts")
from common import *
t0 = time.time()
exp, tokens, cmd, logs, poses, z, order = setup()
P = z["proposals"][order]; pk = z["pick"][order]; logit = z["logits"][order]
print("N", len(tokens), "turn", int(((cmd==0)|(cmd==2)).sum()), flush=True)
# classes of all 64 proposals
cls64, _ = CA.plan_classes(P.reshape(-1, 8, 3)); cls64 = cls64.reshape(len(tokens), 64)
np.save(SCR + "/tmp/cls64.npy", cls64)
plan = CA.plan_classes(poses)[0]
at_pick = P[np.arange(len(tokens)), pk]
print("dump[pick]==seam poses bitwise:", bool(np.array_equal(at_pick, poses)), flush=True)
cls_pick = cls64[np.arange(len(tokens)), pk]
print("class of dump[pick] == class of seam:", bool((cls_pick == plan).all()))
ta = (cmd == 0) | (cmd == 2)
res = {}
_,_,_,_,cl = CA._cells(cmd, plan, logs, CA.N_BOOT, CA.SEED, "pick")
res["pick"] = cl
F = CA.proposal_fractions(P)
fl = {}
for sub, m in (("all_tokens", None), ("turn_active", ta)):
    ul, Cf = CA.grouped(cmd, F, logs, m); ul2, Ca = CA.grouped(cmd, plan, logs, m)
    fl[sub] = {"floor": CA.cell(Cf, CA.N_BOOT, CA.SEED), "pick_minus_floor": CA.paired(Ca, Cf)}
res["floor"] = fl
old = json.load(open("E:/Projects/TanitAD-artifacts/p7_full_read/command_adherence_p7_full.json"))
res["old_json_keys"] = list(old["arms"].keys()) if "arms" in old else None
def g(d, *k):
    for x in k: d = d[x]
    return d
try:
    arm = old["arms"][old["arm"]["name"] if isinstance(old["arm"], dict) else old["arm"]]
    res["old_pick_turn"] = g(arm, "executed_plan_A7", "turn_active", "accuracy")
    res["old_floor_turn"] = old["random_proposal_floor"]["turn_active"]["floor"]["accuracy"]
except Exception as e:
    res["old_err"] = repr(e)
json.dump(res, open(SCR + "/tmp/s1_res.json", "w"), default=str, indent=1)
print("pick turn", res["pick"]["turn_active"]["accuracy"], "all", res["pick"]["all_tokens"]["accuracy"])
print("floor turn", fl["turn_active"]["floor"]["accuracy"], "all", fl["all_tokens"]["floor"]["accuracy"])
print("old", res.get("old_pick_turn"), res.get("old_floor_turn"), res.get("old_err"))
print("sec", time.time() - t0)
