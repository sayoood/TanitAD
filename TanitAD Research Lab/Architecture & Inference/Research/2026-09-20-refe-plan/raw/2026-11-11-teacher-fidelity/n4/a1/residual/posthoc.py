"""POST-HOC descriptive join (written AFTER the pre-written rule's table; labelled post-hoc): oscillation of the return vector, executed-path follow ratio, what-if by cause."""
import glob, json, os
H = os.path.dirname(os.path.abspath(__file__))
an = json.load(open(H + "/anatomy_25.json", encoding="utf-8")); wi = json.load(open(H + "/whatif_25.json", encoding="utf-8"))
ex = {}
for fp in glob.glob("F:/tanitad_scratch/n4resid/exec_s*.jsonl"):
    for ln in open(fp, encoding="utf-8"):
        r = json.loads(ln); ex[r["token"]] = r
out = {"label": "POST-HOC, descriptive, seen-on-navtest. Not pre-registered; the pre-written rule's classes are in anatomy_25.json.", "tokens": {}}
for t, e in an["tokens"].items():
    uy = [i["rv"]["u_ego"][1] for i in e["iterations"] if i["rv"]]
    sg = [1 if v > 0 else -1 for v in uy]
    flips = sum(1 for a, b in zip(sg, sg[1:]) if a != b)
    x = ex[t]
    out["tokens"][t] = {"pre_rule_cause": e["cause"]["cause"], "n_rv": len(uy), "lateral_sign_flips": flips, "oscillating": flips >= 2,
                        "exec_over_req_it0": x["exec_over_req"], "stage1_label_over_target": x["stage1_label_over_target"],
                        "whatif": {k: wi["tokens"][t][k]["first_full_cert_it"] for k in wi["tokens"][t]}}
nb = [t for t, v in out["tokens"].items() if v["pre_rule_cause"] != "b"]
out["non_b"] = len(nb)
out["non_b_oscillating"] = sum(out["tokens"][t]["oscillating"] for t in nb)
out["non_b_follow_ge_0.85"] = sum(out["tokens"][t]["exec_over_req_it0"] >= 0.85 for t in nb)
out["non_b_osc_and_follow"] = sum(out["tokens"][t]["oscillating"] and out["tokens"][t]["exec_over_req_it0"] >= 0.85 for t in nb)
out["non_b_not_osc"] = [t for t in nb if not out["tokens"][t]["oscillating"]]
out["rule_c_follow_split"] = {"c_total": sum(out["tokens"][t]["pre_rule_cause"] == "c" for t in nb),
                              "c_exec_follows_ge_0.85": sum(out["tokens"][t]["pre_rule_cause"] == "c" and out["tokens"][t]["exec_over_req_it0"] >= 0.85 for t in nb)}
dm_dac5 = set(t for t, e in wi["tokens"].items() if e["m0.05"]["first_dac1_it"] is not None and e["m0.05"]["first_dac1_it"] <= 5)
out["control_dac_only_cap5_m005_equals_n4dm_only_set"] = dm_dac5 == set(an["dm_only_tokens"])
cert = {}
for k in ("m0.05_cap10", "m0.05_cap20", "m0.10_cap20", "m0.20_cap5", "m0.20_cap10", "m0.20_cap20"):
    ts = wi["counts"][k]["full_cert_tokens"]
    cert[k] = {"n": len(ts), "by_cause": {c: sum(1 for t in ts if an["tokens"][t]["cause"]["cause"] == c) for c in "abcde"}}
out["whatif_full_cert_by_cause"] = cert
json.dump(out, open(H + "/posthoc_25.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "tokens"}, indent=1))
