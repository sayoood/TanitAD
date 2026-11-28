"""Independent cross-check of DECISION.json: re-reads the scorer CSVs with the csv module (not the decision script's reader), re-derives
per-arm means, paired deltas, the floor and margin, and an independently written log-cluster bootstrap (own RNG; intervals approximate)."""
import csv, json, sys, os
import numpy as np
root = sys.argv[1]; sample = json.load(open(sys.argv[2])); dec = json.load(open(os.path.join(root, "DECISION.json")))
toks = sorted(sample["tokens"]); log_of = sample["token_log"]
def read(label):
    d = {}
    with open(os.path.join(root, "scores", label, label + ".csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            if r["token"] == "average":
                continue
            d[r["token"]] = r
    return d
labs = {"40k_s0": "r8sel_ema40000_s0", "40k_s1": "r8sel_ema40000_s1", "48852_s0": "r8sel_ema48852_s0", "48852_s1": "r8sel_ema48852_s1",
        "STOP": "r8sel_ref_STOP", "CV": "r8sel_ref_CV", "PRIOR_ha0p": "r8sel_ref_PRIOR_ha0p"}
tab = {k: read(v) for k, v in labs.items()}
out = {"n_tokens": len(toks)}
for k, t in tab.items():
    assert set(t) == set(toks), (k, len(set(t) ^ set(toks)))
    assert all(t[x]["valid"] == "True" for x in toks), k
    out[f"mean_pdms_x100_{k}"] = round(100 * float(np.mean([float(t[x]["score"]) for x in toks])), 4)
logs = sorted({log_of[x] for x in toks}); li = {l: i for i, l in enumerate(logs)}
cid = np.array([li[log_of[x]] for x in toks])
rng = np.random.default_rng(12345)
for s in (0, 1):
    a = np.array([float(tab[f"40k_s{s}"][x]["score"]) for x in toks]); b = np.array([float(tab[f"48852_s{s}"][x]["score"]) for x in toks])
    d = a - b
    sums = np.bincount(cid, weights=d, minlength=len(logs)); cnt = np.bincount(cid, minlength=len(logs))
    boots = []
    for _ in range(4000):
        pick = rng.integers(0, len(logs), len(logs))
        boots.append(sums[pick].sum() / cnt[pick].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out[f"delta_x100_s{s}"] = round(100 * float(d.mean()), 4)
    out[f"indep_boot_ci_x100_s{s}"] = [round(100 * lo, 2), round(100 * hi, 2)]
    out[f"n_tokens_40k_better_s{s}"] = int((d > 0).sum()); out[f"n_tokens_48852_better_s{s}"] = int((d < 0).sum()); out[f"n_tie_s{s}"] = int((d == 0).sum())
floor = abs(out["delta_x100_s0"] - out["delta_x100_s1"])
out["floor_x100"] = round(floor, 4); out["margin_min_x100"] = min(out["delta_x100_s0"], out["delta_x100_s1"]); out["needed_gt_x100"] = round(2 * floor, 4)
ps = dec["per_seed"]
out["agree_with_DECISION"] = {
    "delta_s0": abs(out["delta_x100_s0"] - ps["0"]["delta_x100"]) < 1e-3, "delta_s1": abs(out["delta_x100_s1"] - ps["1"]["delta_x100"]) < 1e-3,
    "pdms_40k_s0": abs(out["mean_pdms_x100_40k_s0"] - ps["0"]["pdms_x100"]["ema40000"]) < 1e-3,
    "pdms_48852_s1": abs(out["mean_pdms_x100_48852_s1"] - ps["1"]["pdms_x100"]["ema48852"]) < 1e-3,
    "STOP": abs(out["mean_pdms_x100_STOP"] - dec["beside"]["references"]["STOP"]["PDMS"]["mean_x100"]) < 0.01,
    "PRIOR": abs(out["mean_pdms_x100_PRIOR_ha0p"] - dec["beside"]["references"]["PRIOR_ha0p"]["PDMS"]["mean_x100"]) < 0.01,
    "verdict_rule_reapplied": ("CHOOSE_EMA_40000" if (all(out[f"indep_boot_ci_x100_s{s}"][0] > 0 and out[f"delta_x100_s{s}"] > 0 for s in (0, 1))
                                                     and out["margin_min_x100"] > 2 * floor) else "DEFAULT_EMA_48852") == dec["verdict"]}
print(json.dumps(out, indent=1))
