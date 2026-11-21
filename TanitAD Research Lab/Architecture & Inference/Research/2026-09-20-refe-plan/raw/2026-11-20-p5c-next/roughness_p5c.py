#!/usr/bin/env python3
"""P5-C roughness + speed-bias diagnosis from the plans themselves (no tracker, no scorer in any number here except where named).

Q: is C1's transparent-comfort deficit (-15.2) and its speed bias in the C1 LABELS or in the student's EXECUTION of them?
Instruments (all CPU, numpy):
  K1  TB-6 learnability on pick20 (0.2 s grid, origin prepended), EXACTLY p5c_table_gates.learn(): per-row max|acc|, max|heading rate|;
      ratio of medians C1/C0 -- read against the SAME statistic on the labels, P5-R label / traj (RESULT_P5R.md: rank 0 acc 1.158, hr 1.564).
  K2  the same statistic on the 8-pose 0.5 s grid for C1 / C0 / base picks AND the logged human (pass1 shared.human8), so the human
      enters on one grid; plus max |jerk| (third difference) and start-speed mismatch (|p1|/0.5 s - v0).
  K3  speed: mean plan speed over 4 s (path length / 4 s) minus the human's, on the PLAN and on the tracked EXECUTION (pass1 exec_tracked8).
  K4  generation vs selection: K2 roughness for all 64 proposals (median over proposals, min over proposals) against the pick.
  K5  association with pass1's transparent comfort c (the pick, dt 0.2 transparent builder): C1 c-fail rate by start-mismatch quartile.
Controls: proposals[pick] must equal the seam poses bit-for-bit (else exit 1); learn() on the human8 vs itself ratio 1.0 (identity);
  K1 on C0 vs C0 = 1.0. Estimator: point values, medians over 12,146 tokens; bootstrap CIs for the three headline ratios over the
  136 logs (log-cluster, 2,000 draws, seed 20261120) -- answers another draw of LOGS only; one training seed per arm.
Usage: roughness_p5c.py OUT.json"""
import glob, json, sys
import numpy as np

SEAMS = "E:/Projects/TanitAD/data/refe_navtest/seams/"
ARMS = {"C1": "stage2_p5c_c1", "C0": "stage2_p5c_c0", "base": "stage2_p7_full"}
ROWS = sorted(glob.glob("E:/Projects/TanitAD-artifacts/p5c_read/out/pass1/rows_s*of2.jsonl"))


def learn(q, dt):
    q = np.asarray(q, np.float64)
    sh = q.shape
    q = q.reshape(-1, sh[-2], 3)
    Q = np.concatenate([np.zeros((q.shape[0], 1, 3)), q], 1)
    acc = np.linalg.norm(np.diff(Q[:, :, :2], n=2, axis=1), axis=2) / dt ** 2
    hr = np.abs(np.diff(np.unwrap(Q[:, :, 2], axis=1), axis=1)) / dt
    jerk = np.linalg.norm(np.diff(Q[:, :, :2], n=3, axis=1), axis=2) / dt ** 3
    v1 = np.linalg.norm(Q[:, 1, :2], axis=1) / dt
    path = np.linalg.norm(np.diff(Q[:, :, :2], axis=1), axis=2).sum(1)
    out = dict(acc=acc.max(1), hr=np.degrees(hr.max(1)), jerk=jerk.max(1), v1=v1, vmean=path / (dt * (Q.shape[1] - 1)))
    return {k: v.reshape(sh[:-2]) for k, v in out.items()}


rows = {}
for f in ROWS:
    for ln in open(f, encoding="utf-8"):
        r = json.loads(ln)
        if "arms" in r:
            rows[r["token"]] = r
D = {}
for a, n in ARMS.items():
    z = np.load(SEAMS + n + ".proposals.npz")
    s = np.load(SEAMS + n + ".npz")
    tok = [str(t) for t in z["token"]]
    assert [str(t) for t in s["token"]] == tok
    P, pick = z["proposals"], z["pick"]
    same = np.array_equal(P[np.arange(len(tok)), pick], s["poses"])
    D[a] = dict(tok=tok, P=P, pick=pick, p20=z["pick20"], seam_pick_identical=bool(same))
toks = D["C1"]["tok"]
for a in D:
    assert D[a]["tok"] == toks
assert set(toks) == set(rows), (len(toks), len(rows))
R = [rows[t] for t in toks]
logs = np.array([r["log"] for r in R])
v0 = np.array([r["shared"]["v0"] for r in R])
H8 = np.array([r["shared"]["human8"] for r in R])
EX = {a: np.array([r["arms"][a]["fam"]["exec_tracked8"] for r in R]) for a in ("C1", "C0", "base", "human")}
TC = {a: np.array([r["arms"][a]["transparent"]["c"] for r in R]) for a in ("C1", "C0", "base", "human")}
N = len(toks)
res = {"n_tokens": N, "n_logs": int(len(set(logs))), "controls": {a: {"proposals_at_pick_equal_seam_poses": D[a]["seam_pick_identical"]} for a in D}}
k1 = {a: learn(D[a]["p20"], 0.2) for a in D}
k2 = {a: learn(D[a]["P"][np.arange(N), D[a]["pick"]], 0.5) for a in D}
k2["human"] = learn(H8, 0.5)
kx = {a: learn(EX[a], 0.5) for a in EX}
med = lambda x: float(np.median(x))
res["K1_pick20_dt0.2"] = {a: {k: med(v) for k, v in k1[a].items()} for a in k1}
res["K1_ratio_of_medians"] = {f"{x}/{y}": {"acc": med(k1[x]["acc"]) / med(k1[y]["acc"]), "hr": med(k1[x]["hr"]) / med(k1[y]["hr"])}
                              for x, y in (("C1", "C0"), ("C1", "base"), ("C0", "base"), ("C0", "C0"))}
res["K1_label_reference_INHERITED"] = {"source": "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5r/RESULT_P5R.md (EVAL split, navtrain)",
                                       "P5R_label_over_traj_rank0": {"acc": 1.158, "hr": 1.564}, "P5R_label_over_traj_rank1": {"acc": 1.148, "hr": 1.344},
                                       "Tc20s_label_over_traj_rank0_same_rows": {"acc": 1.736, "hr": 1.950}}
res["K2_plan8_dt0.5"] = {a: {k: med(v) for k, v in k2[a].items()} for a in k2}
res["K2_ratio_vs_human"] = {a: {k: med(k2[a][k]) / med(k2["human"][k]) for k in ("acc", "hr", "jerk")} for a in k2}
res["K2_start_speed_mismatch_mps"] = {a: {"mean_signed": float((k2[a]["v1"] - v0).mean()), "median_abs": med(np.abs(k2[a]["v1"] - v0)),
                                          "p90_abs": float(np.percentile(np.abs(k2[a]["v1"] - v0), 90))} for a in k2}
res["K3_speed_minus_human_mps"] = {"plan": {a: float((k2[a]["vmean"] - k2["human"]["vmean"]).mean()) for a in ("C1", "C0", "base")},
                                   "tracked_exec": {a: float((kx[a]["vmean"] - kx["human"]["vmean"]).mean()) for a in ("C1", "C0", "base")},
                                   "plan_minus_own_exec": {a: float((k2[a]["vmean"] - kx[a]["vmean"]).mean()) for a in ("C1", "C0", "base", "human")}}
res["K3_exec_roughness_ratio_vs_human_exec"] = {a: {k: med(kx[a][k]) / med(kx["human"][k]) for k in ("acc", "hr", "jerk")} for a in ("C1", "C0", "base")}
k4 = {}
for a in D:
    L = learn(D[a]["P"], 0.5)  # [N, 64]
    pk = D[a]["pick"]
    k4[a] = {k: {"pick_median": med(L[k][np.arange(N), pk]), "median_over_props_median": med(np.median(L[k], 1)),
                 "min_over_props_median": med(L[k].min(1)), "pick_rank_pct_median": med((L[k] < L[k][np.arange(N), pk][:, None]).mean(1) * 100)}
             for k in ("acc", "hr", "jerk")}
    k4[a]["share_tokens_with_prop_jerk_le_human"] = float((L["jerk"] <= k2["human"]["jerk"][:, None]).any(1).mean() * 100)
res["K4_generation_vs_selection_dt0.5"] = k4
mis = np.abs(k2["C1"]["v1"] - v0)
qs = np.quantile(mis, [0.25, 0.5, 0.75])
bins = np.digitize(mis, qs)
res["K5_C1_transparent_c_fail_by_start_mismatch_quartile"] = {"edges_mps": qs.tolist(),
                                                               "fail_pct": [float((TC["C1"][bins == b] == 0).mean() * 100) for b in range(4)]}
res["K5_transparent_c_fail_pct"] = {a: float((TC[a] == 0).mean() * 100) for a in TC}
both = (TC["C1"] == 0) & (TC["C0"] == 1)
res["K5_C1fail_C0pass_tokens"] = {"n": int(both.sum()), "C1_jerk_median": med(k2["C1"]["jerk"][both]), "C0_jerk_median": med(k2["C0"]["jerk"][both]),
                                   "C1_acc_median": med(k2["C1"]["acc"][both]), "C0_acc_median": med(k2["C0"]["acc"][both]),
                                   "C1_start_mismatch_median": med(mis[both]), "C0_start_mismatch_median": med(np.abs(k2["C0"]["v1"] - v0)[both])}
# log-cluster bootstrap for the headline ratios
ul = sorted(set(logs))
idx = {l: np.where(logs == l)[0] for l in ul}
rng = np.random.default_rng(20261120)
bs = {"K1_C1/C0_acc": [], "K1_C1/C0_hr": [], "K3_plan_speed_C1-C0": [], "K3_exec_speed_C1-C0": []}
for _ in range(2000):
    s = np.concatenate([idx[ul[i]] for i in rng.integers(0, len(ul), len(ul))])
    bs["K1_C1/C0_acc"].append(np.median(k1["C1"]["acc"][s]) / np.median(k1["C0"]["acc"][s]))
    bs["K1_C1/C0_hr"].append(np.median(k1["C1"]["hr"][s]) / np.median(k1["C0"]["hr"][s]))
    bs["K3_plan_speed_C1-C0"].append((k2["C1"]["vmean"][s] - k2["C0"]["vmean"][s]).mean())
    bs["K3_exec_speed_C1-C0"].append((kx["C1"]["vmean"][s] - kx["C0"]["vmean"][s]).mean())
pt = {"K1_C1/C0_acc": res["K1_ratio_of_medians"]["C1/C0"]["acc"], "K1_C1/C0_hr": res["K1_ratio_of_medians"]["C1/C0"]["hr"],
      "K3_plan_speed_C1-C0": float((k2["C1"]["vmean"] - k2["C0"]["vmean"]).mean()), "K3_exec_speed_C1-C0": float((kx["C1"]["vmean"] - kx["C0"]["vmean"]).mean())}
res["bootstrap_log_cluster_2000"] = {k: {"point": pt[k], "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5))} for k, v in bs.items()}
ok = all(D[a]["seam_pick_identical"] for a in D) and abs(res["K1_ratio_of_medians"]["C0/C0"]["acc"] - 1) < 1e-12
res["controls"]["pass"] = bool(ok)
json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print(json.dumps(res, indent=1))
sys.exit(0 if ok else 1)
