"""Reads the primary traces (scratch jsonl) + the build report, applies anatomy_rule.classify, writes anatomy_25.json and the table rows. Plain python (no numpy)."""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anatomy_rule as AR  # noqa: E402

SCR = "F:/tanitad_scratch/n4resid"
REP = json.load(open("F:/tanitad_scratch/n4nt/run_a1/n4run/build/out/n4_build_report.json", encoding="utf-8"))
rows = {}
for fp in sorted(glob.glob(SCR + "/primary_s*.jsonl")):
    for ln in open(fp, encoding="utf-8"):
        r = json.loads(ln)
        rows[r["token"]] = r
un_d, un_dm = set(REP["unrepaired_dac0"]["N4D"]), set(REP["unrepaired_dac0"]["N4DM"])
out = {"rule_sha256": hashlib.sha256(open(HERE + "/anatomy_rule.py", "rb").read()).hexdigest(), "n_tokens": len(un_d), "tokens": {}}
for t in sorted(un_d):
    r = rows[t]
    d, dm = r["runs"]["N4D"], r["runs"]["N4DM"]
    assert not d["applied"], t                                    # control: applied false reproduces
    c = AR.classify(d["iterations"])
    e = {"cause": c, "n4d_applied": d["applied"], "n4dm_applied": dm["applied"], "base": r["base"], "n_candidates_scored": len(d["iterations"]),
         "ext_seq_m": [round(i["ext_max_m"], 4) for i in d["iterations"]], "dac_seq": [i["metrics"]["dac"] for i in d["iterations"]],
         "t_worst_s_seq": [i["t_worst_s"] for i in d["iterations"]], "failed_seq": [i["failed"] for i in d["iterations"]],
         "cum_shift_final_ego_m": d["iterations"][-1]["cum_shift_ego_m"], "iterations": d["iterations"]}
    if t not in un_dm:                                            # repaired by N4-DM, not by N4-D
        assert dm["applied"], t
        e["dm_only"] = True
        e["dm_param"] = dm["applied_param"]
        e["dm_certified_candidate"] = dm["iterations"][-1]["metrics"]
        e["n4d_blocking_clauses"] = c.get("clauses")
    out["tokens"][t] = e
out["counts"] = {}
for e in out["tokens"].values():
    out["counts"][e["cause"]["cause"]] = out["counts"].get(e["cause"]["cause"], 0) + 1
out["dm_only_tokens"] = sorted(t for t, e in out["tokens"].items() if e.get("dm_only"))
out["dm_only_clauses"] = {t: out["tokens"][t]["n4d_blocking_clauses"] for t in out["dm_only_tokens"]}
out["cert_refused_clause_counts"] = {}
for e in out["tokens"].values():
    if e["cause"]["cause"] == "b":
        k = "+".join(e["cause"]["clauses"]) or "none"
        out["cert_refused_clause_counts"][k] = out["cert_refused_clause_counts"].get(k, 0) + 1
json.dump(out, open(HERE + "/anatomy_25.json", "w", encoding="utf-8"), indent=1)
print(json.dumps({k: out[k] for k in ("counts", "dm_only_tokens", "dm_only_clauses", "cert_refused_clause_counts")}, indent=1))
for t, e in out["tokens"].items():
    print(t, e["cause"]["cause"], e["cause"].get("clauses", ""), "dm" if e["n4dm_applied"] else "--", e["ext_seq_m"], e["dac_seq"])
