"""The DSCALE committed read's numbers, printed for RESULT_V8F_DSCALE.md (reads dscale_read.json; computes nothing new)."""
import json
d = json.load(open("C:/Users/Admin/box2_ship/logs/v8f_dscale/dscale_read.json", encoding="utf-8"))
r = d["rows"]["s0"]; b = r["beside"]
print("tier", d["tier"], "| estimator", d["estimator"][:120], "| n", d["n_windows"], d["n_episodes"], "| F", d["F"], "2F", d["two_F"])
print("DT", {k: r["dt"][k] for k in ("r_percent", "r_window", "outcome", "definitions_agree")})
for k in ("mse_delta", "mse"):
    if k in r["dt"]: print("DT", k, {q: r["dt"][k].get(q) for q in ("delta", "lo", "hi", "separated")})
p = r["primary"]; print("PRIMARY", {k: p.get(k) for k in ("b", "a", "delta", "lo", "hi", "separated")})
for k in ("mleak", "mdk"):
    x = r[k]; x = x.get("delta", x) if isinstance(x, dict) else x
    print(k, json.dumps(x)[:260])
print("ade6", json.dumps(b["ade6"])[:300]); print("oracle", json.dumps(b["oracle_ade6"])[:300])
print("present_slot", json.dumps(b["present_slot"])[:400]); print("groups", json.dumps(b["groups"])[:500])
print("train_constraint_loss", json.dumps(b["train_constraint_loss"])[:600])
print("generalisation_signature", json.dumps(b["generalisation_signature"])[:600])
f = b["families"]; print("families keys", list(f))
for fam in ("longitudinal", "lateral"):
    for k, x in f[fam].items():
        if isinstance(x, dict) and "delta" in x:
            print(f"  {fam[:4]} {k:22s} {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}] {'SEP' if x['separated'] else 'ns'}")
for fam in ("tactical", "strategic"):
    print(fam, json.dumps(f.get(fam))[:500])
print("strategic_family", json.dumps(d["strategic_family"])[:400])
dk = b["distance_keeping_k10"]; print("dk10", json.dumps(dk)[:400])
print("data", json.dumps(b["data"])[:300]); print("train_log", json.dumps(d["train_log"])[:300])
