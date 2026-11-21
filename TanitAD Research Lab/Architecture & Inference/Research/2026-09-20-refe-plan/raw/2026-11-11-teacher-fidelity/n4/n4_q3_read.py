"""N4-DAC Q3 read: repair counts, what each repair changes, and the paired PDMS contrasts of the repair pipelines (navtrain, 469 tokens / 13 logs)."""
import json
import sys
import numpy as np
sys.path.insert(0, ".")
import n4_common as N

S = N.load_scores()
common = sorted(set.intersection(*[set(v) for v in S.values()]))
logs = [S["RJ5"][t]["log"] for t in common]
Q = json.load(open("Q3_repairs.json", encoding="utf-8"))["instances"]
out = {"n_instances": len(Q), "n_tokens": len({q["token"] for q in Q})}

# 1. repaired counts, per method: primary arm RJ5.R and pooled form-R instances
cnt = {}
for m in "BHAD":
    cnt[m] = {"RJ5.R_repaired_of": [sum(1 for q in Q if q["run"] == "RJ5" and q["methods"][m].get("repaired")), sum(1 for q in Q if q["run"] == "RJ5")],
              "pooled_instances_repaired_of": [sum(1 for q in Q if q["methods"][m].get("repaired")), len(Q)],
              "errors": sum(1 for q in Q if "error" in q["methods"][m])}
out["repaired_counts"] = cnt

# 2. what a repair changes (certified repairs only; base DAC = 0 so base PDMS = 0 on every instance by construction)
chg = {}
for m in "BHAD":
    rows = [q for q in Q if q["methods"][m].get("repaired")]
    if not rows:
        continue
    chg[m] = {"n": len(rows),
              "repaired_pdms_mean": float(np.mean([q["methods"][m]["metrics"]["pdms"] for q in rows])),
              "repaired_ep_mean": float(np.mean([q["methods"][m]["metrics"]["ep"] for q in rows])),
              "d_nc_worse": sum(q["methods"][m]["metrics"]["nc"] < q["base"]["nc"] for q in rows),
              "d_ttc_worse": sum(q["methods"][m]["metrics"]["ttc"] < q["base"]["ttc"] - 1e-12 for q in rows),
              "d_c_worse": sum(q["methods"][m]["metrics"]["c"] < q["base"]["c"] - 1e-12 for q in rows),
              "d_ddc_worse": sum(q["methods"][m]["metrics"]["ddc"] < q["base"]["ddc"] - 1e-12 for q in rows),
              "label_shift_vs_base_label_xy_max_m": {"median": float(np.median([q["methods"][m]["label_shift_xy_max_m"] for q in rows])),
                                                    "max": float(np.max([q["methods"][m]["label_shift_xy_max_m"] for q in rows]))},
              "label_vs_teacher_xy_max_m": {"median": float(np.median([q["methods"][m]["label_shift_vs_teacher_xy_max_m"] for q in rows])),
                                            "max": float(np.max([q["methods"][m]["label_shift_vs_teacher_xy_max_m"] for q in rows]))}}
out["repair_effects_on_repaired_instances"] = chg

# 3. the DAC-fail tokens of RJ5.R, one line per method
tab = {}
for q in Q:
    if q["run"] != "RJ5":
        continue
    tab[q["token"]] = {m: ({"ok": True, "pdms": round(v["metrics"]["pdms"], 4), "ep": round(v["metrics"]["ep"], 4), "ttc": round(v["metrics"]["ttc"], 4),
                            "c": v["metrics"]["c"], "shift_vs_teacher_m": v["label_shift_vs_teacher_xy_max_m"], "param": v["param"]}
                           if v.get("repaired") else {"ok": False}) for m, v in q["methods"].items()}
    tab[q["token"]]["base_RJ5.R"] = {k: round(v, 4) for k, v in q["base"].items()}
out["RJ5R_dac_fail_tokens"] = tab


# 4. pipelines (paired vs RJ5.R and vs CERT-A), 469 tokens
def base_vec(k):
    return np.array([S["RJ5"][t]["arms"]["R"]["tracked"][k] for t in common])


def pipeline(method, with_cert):
    ORDER = ["RJ5.R", "RJ5.C", "R0.C", "RJ3.R", "RJ3.C", "R0.R", "R5HZ.R", "RJ5H5.R", "RG8.R", "R5HZ.C", "RJ5H5.C", "RG8.C", "R0.T", "RJ3.T", "RJ5.T"]
    rep = {(q["run"], q["token"]): q["methods"][method] for q in Q}
    V = {k: [] for k in N.SUB}
    n_rep = 0
    for t in common:
        sel = None
        for nm in (ORDER if with_cert else ["RJ5.R"]):
            r, f = nm.split(".")
            c = S[r][t]["arms"][f]["tracked"]
            if c["dac"] == 1.0:
                sel = c
                break
        if sel is None:                              # none-pass (or, without CERT, an RJ5.R failure): repair RJ5.R's label
            m = rep.get(("RJ5", t), {})
            if m.get("repaired"):
                sel = m["metrics"]
                n_rep += 1
            else:
                sel = S["RJ5"][t]["arms"]["R"]["tracked"]
        if not with_cert and sel is None:
            pass
        for k in N.SUB:
            V[k].append(sel[k])
    return {k: np.array(v) for k, v in V.items()}, n_rep


res = {}
cA = json.load(open("Q2_cert_wider.json", encoding="utf-8"))["rules"]["CERT-A"]
for name, method, with_cert in [("REPAIR-D on RJ5.R (no selection)", "D", False), ("REPAIR-A on RJ5.R (no selection)", "A", False),
                                ("REPAIR-H on RJ5.R (no selection)", "H", False), ("REPAIR-B on RJ5.R (no selection)", "B", False),
                                ("CERT-A then REPAIR-D on none-pass", "D", True)]:
    V, n_rep = pipeline(method, with_cert)
    res[name] = {"tokens_repaired_by_repair_step": n_rep, "levels_x100": {k: 100 * float(V[k].mean()) for k in N.SUB},
                 "zero_dac_tokens": int((V["dac"] == 0).sum()),
                 "paired_vs_RJ5.R_x100": {k: [100 * x for x in N.boot(V[k], logs, base_vec(k))] for k in ("pdms", "dac", "ep", "nc", "ttc", "c", "ddc")},
                 "worse_flips_vs_RJ5.R": {k: int((V[k] < base_vec(k) - 1e-9).sum()) for k in ("nc", "dac", "c", "ddc", "ttc")},
                 "better_flips_vs_RJ5.R": {k: int((V[k] > base_vec(k) + 1e-9).sum()) for k in ("nc", "dac", "c", "ddc", "ttc")}}
    if with_cert:
        # versus CERT-A (paired)
        ORDER = json.load(open("Q2_cert_wider.json", encoding="utf-8"))["order"]
        cv = {}
        for k in ("pdms", "dac"):
            sel = []
            for t in common:
                got = None
                for nm in ORDER:
                    r, f = nm.split(".")
                    c = S[r][t]["arms"][f]["tracked"]
                    if c["dac"] == 1.0:
                        got = c
                        break
                sel.append((got or S["RJ5"][t]["arms"]["R"]["tracked"])[k])
            cv[k] = [100 * x for x in N.boot(V[k], logs, np.array(sel))]
        res[name]["paired_vs_CERT-A_x100"] = cv
out["pipelines"] = res
json.dump(out, open("Q3_read.json", "w", encoding="utf-8"), indent=1)
print(json.dumps(out["repaired_counts"]))
print(json.dumps(chg, indent=0)[:1800])
for k, v in res.items():
    print(k, "rep", v["tokens_repaired_by_repair_step"], "zeroDAC", v["zero_dac_tokens"], "PDMS %.2f DAC %.2f" % (v["levels_x100"]["pdms"], v["levels_x100"]["dac"]),
          "dPDMS", [round(x, 2) for x in v["paired_vs_RJ5.R_x100"]["pdms"]], "dDAC", [round(x, 2) for x in v["paired_vs_RJ5.R_x100"]["dac"]], "worse", v["worse_flips_vs_RJ5.R"],
          "vsCERTA", v.get("paired_vs_CERT-A_x100"))
