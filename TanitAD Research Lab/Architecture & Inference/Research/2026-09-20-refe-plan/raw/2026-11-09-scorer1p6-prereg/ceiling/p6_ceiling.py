import sys, json, numpy as np
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read")
import selc6_analyze as SA
Z = np.load("F:/TanitAD_p6read_work/run/dump/dump_p6.npz", allow_pickle=False)
T = np.load("F:/TanitAD_p6read_work/run/score/table_p6.npz", allow_pickle=False)
dtok = [str(t) for t in Z["token"]]
assert [str(t) for t in T["token"]] == dtok
keys = ("logits", "pick", "traj", "props8", "ego", "gt_valid", "gt_pose20", "gt_t20", "human8", "frame_ctrl_m", "fingerprint")
arrs = {k: Z[k] for k in keys}; arrs.update(pdms=T["pdms"], sub=T["sub"], valid=T["valid"])
D, A = SA.align_to_c3(arrs, dtok)
zero = np.asarray(D["stratum"]) == "zero"; cru = np.asarray(D["cruise"]).astype(bool)
print("n", len(zero), "zero", int(zero.sum()), "random", int((~zero).sum()), "cruise", int(cru.sum()), "valid all", bool(A["valid"].all()))
w = np.where(zero, SA.W_ZERO / zero.sum(), SA.W_RANDOM / (~zero).sum())
L, P, S, V = A["logits"], A["pdms"], A["sub"], A["valid"]
p = 1 / (1 + np.exp(-L.astype(np.float64)))
names = ["NC", "DAC", "EP", "TTC", "C", "DDC"]
def agg(x):  # NC*DAC*(5EP+5TTC+2C)/12
    return x[..., 0] * x[..., 1] * (5 * x[..., 2] + 5 * x[..., 3] + 2 * x[..., 4]) / 12
def pick(x):
    a = agg(x).copy(); a[~V] = -1e9
    return a.argmax(1)
ix = np.arange(len(w))
def rd(k):
    return dict(pdms=float(100 * (w * P[ix, k]).sum()), **{n: float(100 * (w * S[ix, k, i]).sum()) for i, n in enumerate(names)})
base = pick(p)
print("own pick equals banked pick on", int((base == A["pick"]).sum()), "of", len(w))
R = {"weighted_estimate_of_navtest": True}
R["P6_pick"] = rd(A["pick"])
PP = P.copy(); PP[~V] = -1
R["oracle_best_of_64"] = rd(PP.argmax(1))
R["random_proposal_mean"] = float(100 * (w * ((P * V).sum(1) / V.sum(1))).sum())
for i in range(5):
    x = p.copy(); x[..., i] = S[..., i]
    R["true_" + names[i]] = rd(pick(x))
x = p.copy()
for i in (0, 1, 3): x[..., i] = S[..., i]
R["true_NC_DAC_TTC"] = rd(pick(x))
x = p.copy()
for i in (0, 1, 2, 3): x[..., i] = S[..., i]
R["true_NC_DAC_EP_TTC"] = rd(pick(x))
# random and zero stratum levels
R["P6_pick_random_stratum_unweighted"] = float(100 * P[ix, A["pick"]][~zero].mean())
R["oracle_random_stratum_unweighted"] = float(100 * PP.max(1)[~zero].mean())
for k, v in R.items():
    print(k, {a: round(b, 2) for a, b in v.items()} if isinstance(v, dict) else v)
json.dump(R, open("C:/Users/Admin/par18/p6_ceiling.json", "w"), indent=1)

# exact Shapley over {NC, DAC, EP, TTC}: value = weighted PDMS of the pick when the heads in the coalition are true
import itertools, math
H = [0, 1, 2, 3]
def val(sub):
    x = p.copy()
    for i in sub: x[..., i] = S[..., i]
    return float(100 * (w * P[ix, pick(x)]).sum())
vals = {frozenset(c): val(c) for r in range(5) for c in itertools.combinations(H, r)}
sh = {}
for i in H:
    tot = 0.0
    others = [j for j in H if j != i]
    for r in range(4):
        for c in itertools.combinations(others, r):
            wt = math.factorial(r) * math.factorial(3 - r) / math.factorial(4)
            tot += wt * (vals[frozenset(c) | {i}] - vals[frozenset(c)])
    sh[names[i]] = tot
print("v(empty)", round(vals[frozenset()], 2), "v(all)", round(vals[frozenset(H)], 2), "Shapley", {k: round(v, 2) for k, v in sh.items()}, "sum", round(sum(sh.values()), 2))
# does the EP label carry the violation terms?
viol = (S[..., 0] == 0) | (S[..., 1] == 0)
print("proposals with NC or DAC 0:", round(float(viol[V].mean()), 4))
print("EP label on violating proposals: share exactly 0 =", round(float((S[..., 2][viol & V] == 0).mean()), 4), "mean", round(float(S[..., 2][viol & V].mean()), 4))
print("EP label on clean proposals: share exactly 0 =", round(float((S[..., 2][(~viol) & V] == 0).mean()), 4), "mean", round(float(S[..., 2][(~viol) & V].mean()), 4))
ttc0 = (S[..., 3] == 0) & ~viol & V
print("EP label on TTC-fail clean proposals: share exactly 0 =", round(float((S[..., 2][ttc0] == 0).mean()), 4), "n", int(ttc0.sum()))
R["shapley_over_NC_DAC_EP_TTC_pdms_points"] = sh
json.dump(R, open("C:/Users/Admin/par18/p6_ceiling.json", "w"), indent=1)

# how much of the EP head's value is pure progress, and how much is the violation content of the EP label?
clean = ~viol
x = p.copy(); x[..., 2] = np.where(clean, S[..., 2], x[..., 2]); a = val_ep_clean = float(100 * (w * P[ix, pick(x)]).sum())
x = p.copy(); x[..., 2] = np.where(viol, S[..., 2], x[..., 2]); b = val_ep_viol = float(100 * (w * P[ix, pick(x)]).sum())
print("P6", round(vals[frozenset()], 2), "| true EP on clean proposals only", round(a, 2), "| true EP on violating proposals only", round(b, 2), "| true EP everywhere", round(vals[frozenset({2})], 2))
# same on top of perfect safety heads (the conditional value of pure progress)
x = p.copy()
for i in (0, 1, 3): x[..., i] = S[..., i]
base_s = float(100 * (w * P[ix, pick(x)]).sum())
print("perfect NC,DAC,TTC", round(base_s, 2), "| plus true EP on clean", round(vals[frozenset(H)], 2))
R.update(true_EP_clean_only=a, true_EP_violating_only=b, perfect_safety=base_s)
json.dump(R, open("C:/Users/Admin/par18/p6_ceiling.json", "w"), indent=1)
