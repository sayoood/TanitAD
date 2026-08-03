"""Regenerate every measured artifact in this package. ~40 s, CPU only, $0.

  python make_results.py --src ../2026-07-24-v2-corpus-50h-balanced
"""
import argparse, itertools, json, sys, time
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, ".")
from pool_columns import validate_pool, to_rates
from clean_val_select import (ALL_AXES, balanced_select, stratified_select, standardised_diffs,
                              balance_summary, feasibility, census_fraction, cell_labels,
                              speed_edges, build_manifest, manifest_sha256, load)

ap = argparse.ArgumentParser(); ap.add_argument("--src", default="../2026-07-24-v2-corpus-50h-balanced")
ap.add_argument("--parity", default="C:/Users/Admin/tanitad-data/physicalai/r0/phase0_selection.parquet")
a = ap.parse_args(); t0 = time.time()
POOL, SEL = f"{a.src}/v2_pool_scored.parquet", f"{a.src}/r0_selection_v2.parquet"

raw = pd.read_parquet(POOL)
rep = validate_pool(raw)
train, rem = load(POOL, SEL)
edges = speed_edges(train); tcell = cell_labels(train, edges).value_counts(normalize=True)
CONT = ["mean_v", "cum_head", "net_head", "stop_frac", "tl_rate", "tr_rate", "ac_rate", "bs_rate", "hw"]

def score(v):
    d = standardised_diffs(train, v); s = balance_summary(d)
    vp = cell_labels(v, edges).value_counts(normalize=True)
    return {"n": len(v), **{k: s[k] for k in ("max_abs_d", "median_abs_d", "n_axes_over_bar", "worst_axis")},
            "per_family_max_abs_d": s["per_family_max_abs_d"],
            "max_ks": float(max(stats.ks_2samp(train[c], v[c]).statistic for c in CONT)),
            "cell_l1": float((tcell - vp.reindex(tcell.index).fillna(0)).abs().sum()),
            "max_census_fraction": float(max(r["census_fraction"] for r in census_fraction(v, rem, edges)))}

rng = np.random.default_rng(0)
arms = {"balanced": balanced_select(train, rem, 600, seed=0),
        "hybrid": balanced_select(train, rem, 600, seed=0, cell_quota=True),
        "quota_random": stratified_select(train, rem, 600, seed=0),
        "uniform": rem.iloc[rng.choice(len(rem), 600, replace=False)]}

# within-cell residual skew: train vs remainder INSIDE each matched cell
tr_c = train.assign(cell=cell_labels(train, edges)); rm_c = rem.assign(cell=cell_labels(rem, edges))
skew = []
for c in sorted(tr_c.cell.unique()):
    A, B = tr_c[tr_c.cell == c], rm_c[rm_c.cell == c]
    if len(A) < 30 or len(B) < 30: continue
    row = {"cell": c, "n_train": len(A), "n_remainder": len(B)}
    for k in CONT:
        x, y = A[k].to_numpy(), B[k].to_numpy()
        sd = np.sqrt(((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1)) / (len(x)+len(y)-2))
        row[k] = float((x.mean() - y.mean()) / sd) if sd > 0 else 0.0
    skew.append(row)
sk = np.array([[r[k] for k in CONT] for r in skew])

out = {
  "generated": "2026-08-02", "wall_clock_s": None, "hardware": "dev box CPU (no GPU)",
  "inputs": {"pool": POOL, "selection": SEL, "pool_rows": int(len(raw)),
             "pool_unique_clips": int(raw.clip_id.nunique()),
             "duplicate_clip_ids": sorted(raw.clip_id[raw.clip_id.duplicated()].tolist()),
             "train_n": len(train), "remainder_n": len(rem)},
  "semantics_validation": rep.as_dict(),
  "arms": {k: score(v) for k, v in arms.items()},
  "size_sweep": [{"mode": m, **score(balanced_select(train, rem, n, seed=0, cell_quota=q)),
                  "requested_n": n,
                  "quota_binding_headroom": feasibility(train, rem, n).binding_headroom}
                 for m, q in (("hybrid", True), ("balanced", False))
                 for n in (300, 400, 500, 600, 800, 1000)],
  "axis_set_sweep": [feasibility(train, rem, 600, cell_axes=ax).as_dict()
                     for ax in (("junction",), ("junction", "has_turn"), ("junction", "speed"),
                                ("junction", "has_turn", "speed"),
                                ("junction", "has_turn", "speed", "has_brake"))],
  "within_cell_skew": {"per_cell": skew, "features": CONT,
                       "median_abs_d": float(np.median(np.abs(sk))),
                       "p90_abs_d": float(np.percentile(np.abs(sk), 90)),
                       "max_abs_d": float(np.abs(sk).max()),
                       "per_feature_median_abs_d": {k: float(np.median(np.abs(sk[:, i])))
                                                    for i, k in enumerate(CONT)}},
}
seeds = {}
for s in range(5):
    v = balanced_select(train, rem, 600, seed=s, cell_quota=True)
    seeds[s] = {"sha256": manifest_sha256(v.clip_id),
                "max_abs_d": balance_summary(standardised_diffs(train, v))["max_abs_d"],
                "ids": set(v.clip_id)}
ov = [len(seeds[i]["ids"] & seeds[j]["ids"]) / 600 for i, j in itertools.combinations(range(5), 2)]
out["seed_stability"] = {"per_seed": {k: {kk: vv for kk, vv in d.items() if kk != "ids"}
                                      for k, d in seeds.items()},
                         "pairwise_overlap": {"min": float(min(ov)), "max": float(max(ov)),
                                              "mean": float(np.mean(ov))}}
# ---- v1 parity contamination of a v2-only-clean draw (C64 in mirror image) ----
import torch
par = pd.read_parquet(a.parity); pids = sorted(par["clip_id"].astype(str).tolist())
g = torch.Generator().manual_seed(0)                       # the canonical program split
perm = torch.randperm(len(pids), generator=g).tolist()
v1_val = {pids[i] for i in perm[:max(1, int(len(pids) * 0.2))]}
v1_train = set(pids) - v1_val
v600 = set(arms["hybrid"].clip_id.astype(str))
out["v1_parity_contamination"] = {
    "parity_selection": a.parity, "parity_n": len(pids),
    "v1_train_n": len(v1_train), "v1_val_n": len(v1_val),
    "v2only_val600_in_v1_train": len(v600 & v1_train),
    "v2only_val600_in_v1_val": len(v600 & v1_val),
    "remainder_in_parity": int(rem.clip_id.astype(str).isin(set(pids)).sum()),
    "parity_free_remainder": int((~rem.clip_id.astype(str).isin(set(pids))).sum()),
    "v2_train_in_parity": int(train.clip_id.astype(str).isin(set(pids)).sum()),
    "v2_train_in_v1_val": int(train.clip_id.astype(str).isin(v1_val).sum()),
    "note": ("C64 at CLIP granularity, from the selection parquets alone (no pod). NOT the same "
             "statistic as C64's 21/40 EVAL EPISODES — compare within a granularity, never across."),
}

# ---- the shipped manifests: FULLY clean (v2 train AND the whole v1 parity corpus excluded) ----
train_f, rem_f = load(POOL, SEL, [a.parity])
out["fully_clean"] = {"remainder_n": len(rem_f),
                      "feasibility_600": feasibility(train_f, rem_f, 600).as_dict(),
                      "sizes": []}
for n, name in ((400, "v2_clean_val_manifest.json"), (300, "v2_clean_val_manifest_n300.json")):
    v = balanced_select(train_f, rem_f, n, seed=0, cell_quota=True)
    man = build_manifest(train_f, rem_f, v, n_val=n, seed=0)
    man["selector"] = "balanced_select(cell_quota=True)  # hybrid: cell quotas + mean balancing"
    man["excluded"] = {"v2_train_selection": SEL, "v1_parity_selection": a.parity,
                       "why": ("a val that must host BOTH the v1 and v2 arms has to be disjoint "
                               "from both corpora; 62 of a v2-only-clean 600-draw sat in v1 TRAIN")}
    man["feasibility"] = feasibility(train_f, rem_f, n).as_dict()
    json.dump(man, open(name, "w", encoding="utf-8"), indent=2)
    edges_f = speed_edges(train_f)
    out["fully_clean"]["sizes"].append(
        {"n": len(v), "file": name, "sha256": man["sha256"],
         "max_abs_d": man["balance"]["max_abs_d"],
         "max_census_fraction": float(max(r["census_fraction"]
                                          for r in census_fraction(v, rem_f, edges_f)))})
    print(f"[write] {name} n={len(v)} sha256={man['sha256'][:16]} "
          f"max|d|={man['balance']['max_abs_d']:.4f} disjoint={man['disjointness']['disjoint']}")

out["wall_clock_s"] = round(time.time() - t0, 1)
json.dump(out, open("selector_comparison.json", "w", encoding="utf-8"), indent=2, default=str)
print(f"[write] selector_comparison.json  ({out['wall_clock_s']}s)")
