"""Zero-GPU: how much of v8F-mini's error is training variance?

Average the 6 s plans of independently trained M2b-family checkpoints window by window (all scored on eval-141's one grid,
`--ages cycle`, kin_mode yaw) and score the average with the evaluator's own `evaluate` (no re-implementation).
EXPLORATORY: decides nothing; it ranks the next lever (Rule Zero item 5).   python ens_facts.py <export of the tip>"""
import itertools, json, os, sys
import numpy as np
X_ROOT = sys.argv[1]
sys.path[:0] = [os.path.join(X_ROOT, "stack"), os.path.join(X_ROOT, "stack", "scripts"), os.path.join(X_ROOT, "taniteval")]
import v8f_mini_eval as EV                                                    # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PAIR            # noqa: E402

L = "C:/Users/Admin/box2_ship/logs/"
A = {"B0": "v8f_mini/eval/eval_M2b_s0", "B1": "v8f_mini/eval/eval_M2b_s1", "G0": "v8f_grid/eval/eval_M2b-8x20_s0",
     "H0": "v8f_hist/eval/eval_M2b-h10_s0", "P0": "v8f_pgt/eval/eval_M2b-pgt_s0", "W0": "v8f_wc/eval/eval_M2b-wc_s0"}
Z = {k: np.load(L + v + ".npz") for k, v in A.items()}
for k, v in A.items():
    j = json.load(open(L + v + ".json", encoding="utf-8"))
    assert (j["kin_mode"], j["ages"], j["nav_shuffle"]) == ("yaw", "cycle", False), k
    assert (Z[k]["eid"] == Z["B0"]["eid"]).all(), k


class _A:
    pass


a = _A()
a.eval_grid, a.eval_kin, a.eval_join = (os.path.join(EV.RAW, f) for f in ("eval_grid_windows.json", "eval141_kin.npz", "join_eval141.npz"))
assert all(os.path.exists(p) for p in (a.eval_grid, a.eval_kin, a.eval_join))
grid, J, X = EV.ctx_from_args(a, "yaw")
eid = X["eid"]
assert (eid == Z["B0"]["eid"]).all()
gt = np.asarray(X["gt"], np.float64)
ade6 = lambda p: np.linalg.norm(np.asarray(p, np.float64) - gt, axis=-1).mean(1)
out = {"control": {}, "single": {}, "ens": {}}
# control: our GT reproduces every arm's stored per-window ADE6
for k in Z:
    err = float(np.abs(ade6(Z[k]["pred"]) - Z[k]["ade6"]).max())
    out["control"][k] = err
    assert err < 1e-4, (k, err)
    out["single"][k] = float(Z[k]["ade6"].mean())
print("control max |ade6 - stored| per arm:", {k: f"{v:.1e}" for k, v in out["control"].items()})
print("single ADE6:", {k: round(v, 4) for k, v in out["single"].items()})


def pair(p, q):
    r = PAIR(p, q, eid, n_boot=2000, seed=0)
    d, lo, hi = r.get("delta", r.get("mean")), r["lo"], r["hi"]
    return {"delta": round(float(d), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4), "separated": bool(hi < 0 or lo > 0)}


# the two M2b seeds: the only pair that differs in NOTHING but the training seed
e01 = ade6((Z["B0"]["pred"] + Z["B1"]["pred"]) / 2)
out["ens"]["B0+B1"] = {"ade6": float(e01.mean()), "vs_B0": pair(e01, Z["B0"]["ade6"]), "vs_B1": pair(e01, Z["B1"]["ade6"]),
                       "mean_single": (out["single"]["B0"] + out["single"]["B1"]) / 2}
# every 2- and 3-member subset, and all six (the other four differ by one lever each from M2b: an ensemble of near-replicates)
keys = list(Z)
for n in (2, 3, 6):
    rows = []
    for c in itertools.combinations(keys, n):
        e = ade6(sum(Z[k]["pred"] for k in c) / n)
        rows.append((float(e.mean()), float(np.mean([out["single"][k] for k in c])), c))
    rows.sort()
    out["ens"][f"n{n}"] = {"best": rows[0][0], "worst": rows[-1][0], "mean": float(np.mean([r[0] for r in rows])),
                           "mean_single": float(np.mean([r[1] for r in rows])), "n_subsets": len(rows), "best_members": rows[0][2]}
e6 = ade6(sum(Z[k]["pred"] for k in keys) / 6)
out["ens"]["all6"] = {"ade6": float(e6.mean()), "vs_B0": pair(e6, Z["B0"]["ade6"]), "vs_B1": pair(e6, Z["B1"]["ade6"])}
# the four families on the B0+B1 average, from the evaluator itself (deciding nothing)
res, _ = EV.evaluate((Z["B0"]["pred"] + Z["B1"]["pred"]).astype(np.float64) / 2, X, None, n_boot=2000, with_families=True)
fl = res.get("families", {}).get("longitudinal", {})
out["ens"]["B0+B1"]["longitudinal"] = {k: fl.get(k) for k in ("speed_mae_mps", "along_mae_m", "accel_mae_mps2")}
fa = res.get("families", {}).get("lateral", {})
out["ens"]["B0+B1"]["lateral"] = {k: v for k, v in fa.items() if isinstance(v, (int, float))}
print(json.dumps(out["ens"], indent=1, default=str))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ens_facts.json"), "w"), indent=1, default=str)
