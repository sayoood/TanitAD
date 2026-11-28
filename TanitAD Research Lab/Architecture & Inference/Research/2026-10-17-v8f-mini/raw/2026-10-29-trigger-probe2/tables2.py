#!/usr/bin/env python3
"""tables2.py -- prints the numbers RESULTS.md quotes, straight from raw/*.json (so no figure is typed by hand).  python -I tables2.py > raw/tables2.txt"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
R = f"{HERE}/raw"
J = lambda f: json.load(open(f"{R}/{f}")) if os.path.exists(f"{R}/{f}") else None
a = lambda d: f"{d['auc']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"
dl = lambda d: f"{d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"

print("### ARM 1 -- nav descriptive (FULL populations)")
d = J("results_nav2_desc.json")
for pop, P in d["pops"].items():
    print(pop, "n", P["n"], "GO", f"{P['go_rate']['rate']:.3f}", "| GO|turn ahead (args_valid)", f"{P['by_args_valid']['valid']['rate']:.3f} (share {P['by_args_valid']['valid']['share_of_population']:.2f})",
          "GO|no turn ahead", f"{P['by_args_valid']['not_valid']['rate']:.3f}", "| by token", {k: round(v["rate"], 3) for k, v in P["by_token"].items()}, "| by d_next", {k: round(v["rate"], 3) for k, v in P["by_d_next_bin"].items()},
          "| P(valid|GO) %.3f P(valid|notGO) %.3f" % (P["share_of_GO_windows_with_args_valid"], P["share_of_notGO_windows_with_args_valid"]))
print("\n### ARM 1 -- AUCs (fold-mean [CI]; pooled)")
for pop in ("pstop", "pmove"):
    r = J(f"results_nav2_{pop}.json")
    for fam in ("linear", "mlp"):
        print(pop, fam, {k: f"{v['fold']['auc']:.3f}/{v['pooled']['auc']:.3f}" for k, v in r["arms"][fam].items() if not k.startswith("C")})
        print("   paired fold:", {k: dl(v["fold"]) for k, v in r["paired"][fam].items() if k in ("N5+E_minus_E", "NL+E_minus_E", "L+E_minus_E", "Z+E_minus_N5+E", "Z+E_minus_NL+E")})

print("\n### ARM 2 -- Z_nn / Z_nl (TRAIN)")
for pop in ("pstop", "pmove"):
    for src in ("Znn", "Znl"):
        r = J(f"results2_{src}_{pop}.json")
        if not r:
            continue
        for fam in ("linear", "mlp"):
            if fam not in r["arms"]:
                continue
            arms = {k: f"{v['fold']['auc']:.3f} [{v['fold']['lo']:.3f},{v['fold']['hi']:.3f}] / pooled {v['pooled']['auc']:.3f}" for k, v in r["arms"][fam].items() if not k.startswith("C")}
            print(pop, src, fam, arms)
            print("   paired fold:", {k: dl(v["fold"]) for k, v in r["paired"][fam].items()}, "| pooled:", {k: f"{v['pooled']['delta']:+.3f}" for k, v in r["paired"][fam].items()})

print("\n### ARM 3 -- V4 / Vdiff (TRAIN)")
for pop in ("pstop", "pmove"):
    for src in ("V4", "Vdiff"):
        r = J(f"results2_{src}_{pop}.json")
        if not r:
            continue
        for fam in ("linear", "mlp"):
            if fam not in r["arms"]:
                continue
            arms = {k: f"{v['fold']['auc']:.3f} [{v['fold']['lo']:.3f},{v['fold']['hi']:.3f}] / pooled {v['pooled']['auc']:.3f}" for k, v in r["arms"][fam].items() if not k.startswith("C")}
            print(pop, src, fam, arms)
            print("   paired fold:", {k: dl(v["fold"]) for k, v in r["paired"][fam].items()}, "| pooled:", {k: f"{v['pooled']['delta']:+.3f}" for k, v in r["paired"][fam].items()})

print("\n### eval-141 (no refit)")
e = J("results_eval2.json")
if e:
    for pop, P in e["pops"].items():
        print(pop, "n", P["n"], "GO", P["n_go"], "episodes", P["n_episodes"])
        for fam, F in P["families"].items():
            print("  ", fam, {k: a(v) for k, v in F["arms"].items()})
            print("    paired:", {k: dl(v) for k, v in F["paired"].items()})
            print("    EV0 refit-vs-banked |diff|:", {k: v["abs_diff"] for k, v in F["ev0_refit_vs_banked"].items()})
            if F.get("subsets"):
                print("    launch-vs-stay:", {k: (round(v["auc_launch_vs_stay"]["auc"], 3), round(v["mean_percentile_launch"], 2), round(v["mean_percentile_stay"], 2)) for k, v in F["subsets"].items()})

print("\n### strata (TRAIN, linear fold-mean AUC)")
s = J("results_strata2.json")
if s:
    for pop, P in s["pops"].items():
        for st, r in P["strata"].items():
            if isinstance(r["auc"], dict):
                gr = r.get("go_rate_full")
                print(pop, st, "n_sample", r["n_sample"], "GO(full)", None if not gr else round(gr["rate"], 3), {k: round(v["fold"]["auc"], 3) for k, v in r["auc"].items()},
                      "| d:", {k: f"{v['fold']['delta']:+.3f}" for k, v in r["paired"].items() if k in ("V+E_minus_E", "Znn+E_minus_E", "Znl+E_minus_E", "V4+E_minus_V+E", "Vdiff+E_minus_V+E", "Z+E_minus_Znn+E")})
            else:
                print(pop, st, "n_sample", r["n_sample"], "too small")
