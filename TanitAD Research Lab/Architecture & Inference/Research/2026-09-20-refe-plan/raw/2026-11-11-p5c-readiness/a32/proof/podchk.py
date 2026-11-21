import json
W="/workspace/data/p5c_a32_check"
old=json.load(open("/workspace/data/p5c_label/gates/control/gates.json"))
for R in ("0","all"):
    g=json.load(open(f"{W}/r{R}/gates.json"))
    t=g["gates"]["TB-0"]; d=t["detail"]["tc20s_warm_start_vs_dev_build"]
    print("scope",R,"tc20s_ranks",g["tc20s_ranks"],"TB-0",t["status"],"TB-1",g["gates"]["TB-1"]["status"],"failing",g["failing"])
    print("  ra", json.dumps({k:v for k,v in d.items()}))
    if R=="all":
        o=old["gates"]["TB-0"]
        print("  bit-identical gate TB-0 vs run1:", json.dumps(t,sort_keys=True)==json.dumps(o,sort_keys=True))
        print("  all gates identical vs run1:", all(json.dumps(g["gates"][k],sort_keys=True)==json.dumps(old["gates"][k],sort_keys=True) for k in g["gates"]), "failing equal", g["failing"]==old["failing"], "value repr", repr(d["value"]))
    else:
        print("  other gates identical to run1:", all(json.dumps(g["gates"][k],sort_keys=True)==json.dumps(old["gates"][k],sort_keys=True) for k in g["gates"] if k!="TB-0"))
        a=t["detail"]["p5r_vs_eval_labels"]; b=old["gates"]["TB-0"]["detail"]["p5r_vs_eval_labels"]
        print("  R-B part identical to run1:", json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True))
