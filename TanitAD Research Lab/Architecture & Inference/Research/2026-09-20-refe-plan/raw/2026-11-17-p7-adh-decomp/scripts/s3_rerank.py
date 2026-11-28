"""Q4: command-consistent re-rank. Reads only: the navigation command (input), the planner's own 64 proposals and its own scorer logits.
The human logged future / ego labels appear ONLY in clearly marked DESCRIPTIVE blocks after the selection vector is fixed."""
import sys, json, hashlib, csv
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/adh_decomp/scripts")
from common import *
import torch
exp, tokens, cmd, logs, poses, z, order = setup()
N = len(tokens); ta = (cmd == 0) | (cmd == 2)
cls64 = np.load(SCR + "/tmp/cls64.npy")
P = z["proposals"][order]; pk = z["pick"][order]; L = z["logits"][order]
print("logits dtype", L.dtype, "rule", str(z["rule"]), "select", str(z["select"]), "repair", bool(z["repair_last_heading"]))


# ---- control: replicate pick EXACTLY (planner.py aggregate rule navsim_v1, V1_W=(5,5,2); argmax first-max)
def aggregate(score):
    p = score.sigmoid()
    w1 = torch.tensor((5.0, 5.0, 2.0), dtype=p.dtype)
    return p[..., 0] * p[..., 1] * (p[..., 2] * w1[0] + p[..., 3] * w1[1] + p[..., 4] * w1[2]) / w1.sum()


A = np.empty((N, 64), dtype=np.float32)
rep_pick = np.empty(N, dtype=np.int64)
for i in range(N):
    a = aggregate(torch.from_numpy(L[i:i + 1]).float())     # [1,64], exactly as _pick sees it
    A[i] = a[0].numpy()
    rep_pick[i] = int(a[0].argmax())
print("replicated pick == dump pick:", int((rep_pick == pk).sum()), "/", N)
assert (rep_pick == pk).all(), "PICK NOT REPLICATED"
ties = int(((A == A.max(1, keepdims=True)).sum(1) > 1).sum())
print("tokens with tied max score:", ties)
rep_done = "dump proposals built by planner.executed(traj, j) for every j (refe_navtest_seam.py:294-295), i.e. post-A7 at the 20-pose level BEFORE to_navsim; verified: dump[pick] == seam poses bitwise (s1)"
print(rep_done)


def rerank(cmd_in):
    adh = (cls64 == cmd_in[:, None])
    Am = np.where(adh, A, -np.inf)
    k = Am.argmax(1)
    has = adh.any(1)
    return np.where(has, k, pk), has


def evalsel(sel, label):
    cl = cls64[np.arange(N), sel]
    out = {}
    for sub, m in (("all_tokens", None), ("turn_active", ta)):
        _, C = CA.grouped(cmd, cl, logs, m)
        _, C0 = CA.grouped(cmd, cls64[np.arange(N), pk], logs, m)
        out[sub] = {"cell": CA.cell(C, CA.N_BOOT, CA.SEED, label), "vs_pick": CA.paired(C, C0)}
    return out, cl


sel, has = rerank(cmd)
res = {"control_pick_replicated": {"equal": int((rep_pick == pk).sum()), "n": N, "ties": ties,
                                   "rule": "navsim_v1 V1_W=(5,5,2) sigmoid aggregate, argmax first-max, per-token [1,64,6] torch float32"},
       "proposals_already_repaired": rep_done}
ev, cl = evalsel(sel, "rerank")
res["rerank"] = ev
chg = sel != pk
res["n_changed"] = {"all": int(chg.sum()), "turn_active": int((chg & ta).sum()), "by_cmd": {CA.CMD_NAMES[c]: int((chg & (cmd == c)).sum()) for c in range(3)}}
pcl = cls64[np.arange(N), pk]
res["rerank_per_class"] = {CA.CMD_NAMES[c]: {"pick": float((pcl[cmd == c] == c).mean()), "rerank": float((cl[cmd == c] == c).mean())} for c in range(3)}
# ---- PDMS of the changed tokens' CURRENT picks (descriptive)
rows = {r["token"]: r for r in csv.DictReader(open(CSV)) if r["token"] != "average"}
assert len(rows) == N
sc = np.array([float(rows[t]["score"]) for t in tokens])
res["pdms_all_tokens_mean"] = float(sc.mean())


def pd(m):
    return {"n": int(m.sum()), "mean_pdms_of_current_pick": float(sc[m].mean()) if m.any() else None,
            "median": float(np.median(sc[m])) if m.any() else None, "frac_zero": float((sc[m] == 0).mean()) if m.any() else None,
            "frac_ge_0.9": float((sc[m] >= .9).mean()) if m.any() else None}


res["pdms_of_changed_current_picks"] = {"changed_all": pd(chg), "changed_turn_active": pd(chg & ta), "unchanged_all": pd(~chg), "all_tokens": pd(np.ones(N, bool))}
a_old, a_new = A[np.arange(N), pk], A[np.arange(N), sel]
res["scorer_aggregate_on_changed"] = {"mean_old_pick": float(a_old[chg].mean()), "mean_new_pick": float(a_new[chg].mean()), "n": int(chg.sum())}
# ---- permuted-command control
rng = np.random.default_rng(20261117)
perm = rng.permutation(N)
selp, _ = rerank(cmd[perm])
clp = cls64[np.arange(N), selp]
ctl = {}
for sub, m in (("all_tokens", None), ("turn_active", ta)):
    _, C = CA.grouped(cmd, clp, logs, m)
    _, C0 = CA.grouped(cmd, pcl, logs, m)
    ctl[sub] = {"adherence_vs_TRUE_command": CA.cell(C, CA.N_BOOT, CA.SEED), "vs_pick": CA.paired(C, C0)}
res["control_permuted_command"] = {"seed": 20261117, "n_changed": int((selp != pk).sum()), **ctl,
                                   "expect": "adherence to the TRUE command must NOT exceed the pick"}
# ---- DESCRIPTIVE ONLY (after selection is fixed; the human future is not an input of the re-rank)
gt = np.asarray([exp[t]["human_future_poses"] for t in tokens], dtype=np.float32)
hc = CA.plan_classes(gt)[0]
hum_ok = hc == cmd
st = {}
for nm, m in (("human_adheres", ta & hum_ok), ("human_does_not_adhere", ta & ~hum_ok)):
    st[nm] = {"n": int(m.sum()), "pick": float((pcl[m] == cmd[m]).mean()), "rerank": float((cl[m] == cmd[m]).mean()),
              "changed": int((chg & m).sum()), "mean_pdms_changed_current": float(sc[chg & m].mean()) if (chg & m).any() else None}
res["DESCRIPTIVE_strata_by_human_adherence_turn_active"] = st
res["DESCRIPTIVE_plan_class_equals_human_class_turn_active"] = {"pick": float((pcl == hc)[ta].mean()), "rerank": float((cl == hc)[ta].mean())}
# ---- seam: SAME schema as stage2_p7_full.npz. NO extra A7 repair: the A7 repair is a 20-pose-grid operation already inside every dump proposal (it is NOT h[-1]==h[-2] on the 8-pose grid); re-applying it on 8 poses would be wrong.
newposes = P[np.arange(N), sel].copy()
res["repair_changed_values"] = "not applied: proposals are already executed/post-A7 forms"
zs = np.load(SEAM, allow_pickle=False)
out = {k: zs[k] for k in zs.files}
tok_pos = {t: i for i, t in enumerate(tokens)}
new_full = zs["poses"].copy()
for j, t in enumerate(zs["token"]):
    new_full[j] = newposes[tok_pos[str(t)]]
out["poses"] = new_full.astype(zs["poses"].dtype)
out["arm"] = np.array("REFe_p7_rerank_cmd")
np.savez(HOLD + "/rerank_p7_seam.npz", **out)
unch = np.array([not chg[tok_pos[str(t)]] for t in zs["token"]])
res["seam_unchanged_tokens_bit_equal"] = bool(np.array_equal(out["poses"][unch], zs["poses"][unch]))
res["seam_keys"] = list(out.keys())
res["seam_n_changed_rows"] = int((~unch).sum())
np.savez(HOLD + "/rerank_selection_idx.npz", token=np.array(tokens), pick=pk, rerank=sel)
h = hashlib.sha256(open(HOLD + "/rerank_p7_seam.npz", "rb").read()).hexdigest()
open(HOLD + "/rerank_p7_seam.npz.sha256", "w").write(h + "  rerank_p7_seam.npz\n")
res["seam_sha256"] = h
json.dump(res, open(HOLD + "/adh_q4.json", "w"), default=str, indent=1)
r = res["rerank"]
for sub in r:
    print("RERANK", sub, r[sub]["cell"]["accuracy"], "vs pick", r[sub]["vs_pick"]["accuracy_delta"], "kappa d", r[sub]["vs_pick"]["kappa_delta"]["point"])
for k in ("n_changed", "rerank_per_class", "pdms_all_tokens_mean", "pdms_of_changed_current_picks", "scorer_aggregate_on_changed",
          "DESCRIPTIVE_strata_by_human_adherence_turn_active", "DESCRIPTIVE_plan_class_equals_human_class_turn_active", "repair_changed_values",
          "seam_unchanged_tokens_bit_equal", "seam_n_changed_rows", "seam_sha256"):
    print(k, res[k])
c = res["control_permuted_command"]
print("PERM n_changed", c["n_changed"], {s: (c[s]["adherence_vs_TRUE_command"]["accuracy"]["point"], c[s]["vs_pick"]["accuracy_delta"]) for s in ("all_tokens", "turn_active")})
