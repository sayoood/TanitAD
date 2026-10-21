"""POST-HOC DIAGNOSTIC (not pre-registered, never deciding): does the RETIME stage cost PDMS on navtrain?  Reads the scorer CSVs."""
import csv, json, os, sys
import numpy as np
root = sys.argv[1]; sample = json.load(open(sys.argv[2]))
toks = sorted(sample["tokens"]); log_of = sample["token_log"]
def read(path):
    with open(path, newline="") as fh:
        return {r["token"]: r for r in csv.DictReader(fh) if r["token"] != "average"}
S = lambda lab: os.path.join(root, "scores", lab, lab + ".csv")
G = lambda lab: os.path.join(root, "diag", "scores", lab, lab + ".csv")
arms = {"RT_40k_s0": S("r8sel_ema40000_s0"), "RT_40k_s1": S("r8sel_ema40000_s1"), "RT_48852_s0": S("r8sel_ema48852_s0"), "RT_48852_s1": S("r8sel_ema48852_s1"),
        "LEGAL_40k_s0": G("r8sel_diag_ema40000_LEGAL_s0"), "LEGAL_40k_s1": G("r8sel_diag_ema40000_LEGAL_s1"),
        "LEGAL_48852_s0": G("r8sel_diag_ema48852_LEGAL_s0"), "LEGAL_48852_s1": G("r8sel_diag_ema48852_LEGAL_s1"),
        "PRIOR_ha0p": S("r8sel_ref_PRIOR_ha0p"), "STOP": S("r8sel_ref_STOP"), "CV": S("r8sel_ref_CV")}
cols = {"PDMS": "score", "NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress", "TTC": "time_to_collision_within_bound",
        "C": "comfort", "DDC": "driving_direction_compliance"}
T = {k: read(p) for k, p in arms.items()}
for k, t in T.items():
    assert set(t) == set(toks) and all(t[x]["valid"] == "True" for x in toks), k
V = {k: {m: np.array([float(T[k][x][c]) for x in toks]) for m, c in cols.items()} for k in T}
logs = sorted({log_of[x] for x in toks}); li = {l: i for i, l in enumerate(logs)}; cid = np.array([li[log_of[x]] for x in toks])
def boot(d, seed=7, B=2000):
    rng = np.random.default_rng(seed); sums = np.bincount(cid, weights=d, minlength=len(logs)); cnt = np.bincount(cid, minlength=len(logs))
    bs = [sums[p].sum() / cnt[p].sum() for p in (rng.integers(0, len(logs), len(logs)) for _ in range(B))]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return [round(100 * float(lo), 2), round(100 * float(hi), 2)]
out = {"label": "POST-HOC DIAGNOSTIC, not pre-registered, never deciding; NavSim v1 open-loop PDMS on the navtrain selection sample; Thor CUDA rolls, Thor CPU scorer",
       "n_tokens": len(toks), "n_log_clusters": len(logs), "estimator": "independent log-cluster bootstrap (own RNG, B 2000), paired, 95 %",
       "means_x100": {k: {m: round(100 * float(v.mean()), 2) for m, v in V[k].items()} for k in V}, "paired_x100": {}}
pairs = [("RT_40k_s0", "LEGAL_40k_s0"), ("RT_40k_s1", "LEGAL_40k_s1"), ("RT_48852_s0", "LEGAL_48852_s0"), ("RT_48852_s1", "LEGAL_48852_s1"),
         ("RT_48852_s0", "PRIOR_ha0p"), ("LEGAL_48852_s0", "PRIOR_ha0p"), ("RT_48852_s0", "STOP"), ("LEGAL_48852_s0", "STOP"),
         ("LEGAL_40k_s0", "LEGAL_48852_s0"), ("LEGAL_40k_s1", "LEGAL_48852_s1")]
for a, b in pairs:
    row = {}
    for m in ("PDMS", "NC", "DAC", "EP", "TTC"):
        d = V[a][m] - V[b][m]; row[m] = {"delta": round(100 * float(d.mean()), 2), "ci95": boot(d)}
    out["paired_x100"][f"{a} - {b}"] = row
print(json.dumps(out, indent=1))
