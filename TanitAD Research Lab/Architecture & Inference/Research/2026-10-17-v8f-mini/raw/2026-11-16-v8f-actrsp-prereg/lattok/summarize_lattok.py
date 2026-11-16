"""Print the LAT-TOK verdict table from lattok_result.json (pulled from the pod).  Usage: python summarize_lattok.py lattok_result.json"""
import json, sys
R = json.load(open(sys.argv[1]))
print("verdict:", R["verdict"], "| prereg", R["prereg_sha256"][:12], "| estimator:", R["estimator"])
for cell, v in R["rule_cells"].items():
    print(cell, "PASS-both" if v["pass_both"] else "no", {t: (round(x["dR2_lo"], 4), round(x["s_swap_lo"], 4), x["pass"]) for t, x in v["per_chassis"].items()})
for tag, res in R["results"].items():
    print("==", tag, "n_pairs", res["_ctrl"]["n_pairs"], "(kstep", res["_ctrl"]["kstep_n_pairs"], ")")
    for k, x in res.items():
        if k != "_ctrl" and (".yaw." in k or ".acc." in k or ".shuf" in k):
            print(f"  {k:28s} {x['point']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]")
