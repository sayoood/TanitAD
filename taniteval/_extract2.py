import json
R = "/root/taniteval/results/"


def L(p):
    try:
        return json.load(open(R + p))
    except Exception as e:
        return {"ERR": str(e)}


def m(blk, k):
    v = blk.get(k, {})
    return v.get("mean") if isinstance(v, dict) else v


print("############ CORE (heldout: model vs CV baseline) ############")
for key, f in [("30k", "refb-v2-30k.json"), ("20k", "refb-v2-20k.json"),
               ("v1", "refb.json")]:
    d = L(f)
    if "ERR" in d:
        print(key, "MISSING", d["ERR"]); continue
    hm = d["heldout"]["model"]; hc = d["heldout"].get("cv", {})
    print("\n== %s step=%s n=%s beats_cv=%s ==" %
          (key, d.get("ckpt_step"), d.get("n_windows"), d.get("beats_cv_ade_0_2s")))
    print("  model ade@ 0.5/1/1.5/2s:", [m(hm, "ade@%s" % h) for h in ("0.5s", "1s", "1.5s", "2s")],
          "| fde@2s", m(hm, "fde@2s"), "| miss@2m", m(hm, "miss_rate@2m"),
          "| tms", m(hm, "tms_openloop"))
    print("  CV    ade_0_2s:", m(hc, "ade_0_2s"), "| fde@2s", m(hc, "fde@2s"),
          "| miss@2m", m(hc, "miss_rate@2m"))
    for panel in ("by_speed", "by_curvature"):
        p = d.get(panel, {})
        row = {}
        for k, v in p.items():
            if isinstance(v, dict) and "model" in v:
                row[k] = (m(v["model"], "ade_0_2s"), m(v.get("cv", {}), "ade_0_2s"), v.get("n"))
        print("  %s (model, cv, n):" % panel, row)

print("\n############ GENERALIZATION (planner direct) ############")
for c in ["", "_comma", "_cosmos"]:
    d = L("gen_refb-v2-30k%s.json" % c)
    if "ERR" in d:
        print(c or "physicalai", "MISSING"); continue
    hm = d["heldout"]["model"]; hc = d["heldout"].get("cv", {})
    cor = d.get("corpus", {}).get("key", c or "physicalai")
    print("  %-11s n=%s ade@2s=%s ci=%s fde=%s miss@2m=%s | CV ade_0_2s=%s" %
          (cor, d.get("n_windows"), m(hm, "ade@2s"),
           hm.get("ade@2s", {}).get("ci95"),
           m(hm, "fde@2s"), m(hm, "miss_rate@2m"), m(hc, "ade_0_2s")))

print("\n############ A/B by_curvature (+delta => B better) ############")
for f in ["ab_refb-v2-30k_vs_refb-v2-20k.json", "ab_refb-v2-30k_vs_refb.json",
          "ab_refb-v2-30k_vs_flagship-30k.json"]:
    d = L(f)
    if "ERR" in d:
        print(f, "MISSING"); continue
    print("\n  %s: ade_a=%s ade_b=%s win_rate_b=%s dCI=%s verdict=%s" %
          (f.replace("ab_", "").replace(".json", ""), d.get("ade_a"), d.get("ade_b"),
           d.get("win_rate_b"), d.get("delta_ci95"), d.get("verdict")))
    print("   by_curvature:", d.get("by_curvature"))
