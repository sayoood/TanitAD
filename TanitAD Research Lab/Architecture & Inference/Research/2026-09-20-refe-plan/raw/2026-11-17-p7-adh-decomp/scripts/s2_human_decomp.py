import sys, json
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/adh_decomp/scripts")
from common import *
exp, tokens, cmd, logs, poses, z, order = setup()
N = len(tokens); ta = (cmd == 0) | (cmd == 2)
cls64 = np.load(SCR + "/tmp/cls64.npy"); pk = z["pick"][order]
plan = CA.plan_classes(poses)[0]
gt = np.asarray([exp[t]["human_future_poses"] for t in tokens], dtype=np.float32)
print("gt shape", gt.shape)
hc, hprov = CA.plan_classes(gt)
res = {"human": {}, "decomp": {}}
for sub, m in (("all_tokens", None), ("turn_active", ta)):
    ul, C = CA.grouped(cmd, hc, logs, m)
    res["human"][sub] = CA.cell(C, CA.N_BOOT, CA.SEED, "human logged future")
    ul2, Cp = CA.grouped(cmd, plan, logs, m)
    res["human"][sub]["pick_minus_human"] = CA.paired(Cp, C)
res["human"]["prov"] = hprov
# decomposition
adh = (cls64 == cmd[:, None])                  # [N,64]; UNKNOWN never equals (cmd 3 absent)
share = adh.mean(1); any_ad = adh.any(1)
pick_ad = adh[np.arange(N), pk]
names = CA.CMD_NAMES
def block(m):
    n = int(m.sum())
    sel = int((any_ad & ~pick_ad & m).sum()); gen = int((~any_ad & m).sum()); ok = int((pick_ad & m).sum())
    return {"n": n, "pick_adheres": ok, "pick_acc": ok / n, "best_of_64_adherent_exists": int((any_ad & m).sum()), "oracle_acc": float((any_ad & m).sum() / n),
            "SELECTION_miss(adherent exists, pick not)": sel, "GENERATION_miss(no adherent proposal)": gen,
            "mean_share_adherent": float(share[m].mean()), "median_share": float(np.median(share[m])),
            "share_among_selection_miss": float(share[m & any_ad & ~pick_ad].mean()) if sel else None,
            "pick_acc_when_share_ge_0.5": float(pick_ad[m & (share >= .5)].mean()) if (m & (share >= .5)).any() else None,
            "pick_acc_when_share_lt_0.5": float(pick_ad[m & (share < .5)].mean()) if (m & (share < .5)).any() else None}
res["decomp"]["turn_active"] = block(ta); res["decomp"]["all_tokens"] = block(np.ones(N, bool))
for c in (0, 1, 2): res["decomp"][names[c]] = block(cmd == c)
# per-class pick/proposal class distribution
res["decomp"]["pick_class_by_cmd"] = {names[c]: [int(((cmd == c) & (plan == k)).sum()) for k in range(3)] for c in range(3)}
# paired: is the pick better than a random proposal? mean (pick_ad - share) by log-cluster
json.dump(res, open(HOLD + "/adh_q2_q3.json", "w"), default=str, indent=1)
h = res["human"]
for sub in h:
    if sub == "prov": continue
    print("HUMAN", sub, h[sub]["accuracy"], "kappa", h[sub]["kappa"]["point"], h[sub]["confusion_command_rows_plan_cols"], h[sub]["per_command_recall"])
    print("  pick-human", h[sub]["pick_minus_human"]["accuracy_delta"])
for k, v in res["decomp"].items():
    print(k, v if k != "pick_class_by_cmd" else v)
