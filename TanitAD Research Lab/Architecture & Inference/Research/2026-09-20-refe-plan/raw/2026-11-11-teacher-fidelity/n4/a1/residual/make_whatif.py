"""Summarises the what-if traces (iters=20 per margin; scratch whatif_s*.jsonl) -> whatif_25.json. DESCRIPTIVE, SEEN-ON-NAVTEST: the 25 tokens were read on navtest; never an effect."""
import glob, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
rows = {}
for fp in sorted(glob.glob("F:/tanitad_scratch/n4resid/whatif_s*.jsonl")):
    for ln in open(fp, encoding="utf-8"):
        r = json.loads(ln); rows[r["token"]] = r
an = json.load(open(HERE + "/anatomy_25.json", encoding="utf-8"))
out = {"label": "DESCRIPTIVE ONLY - seen on navtest (these 25 tokens were read there); NOT a result, NOT quotable as an effect. Which lever to develop on navtrain.",
       "n": len(rows), "caps": [5, 10, 20], "margins_m": [0.05, 0.10, 0.20], "tokens": {}, "counts": {}}
for t, r in sorted(rows.items()):
    e = {}
    for mk, run in r["runs"].items():
        its = run["iterations"]
        full = next((i["it"] for i in its if not i["failed"]), None)
        dac = next((i["it"] for i in its if i["metrics"]["dac"] == 1.0), None)
        e[mk] = {"first_full_cert_it": full, "first_dac1_it": dac, "n_candidates": len(its), "last_ext_m": its[-1]["ext_max_m"],
                 "stopped_early_no_outside": bool(its[-1]["rv"] is None), "dac1_failed_clauses": next((i["failed"] for i in its if i["metrics"]["dac"] == 1.0 and i["failed"]), None)}
    out["tokens"][t] = e
for mk in ("m0.05", "m0.10", "m0.20"):
    for c in (5, 10, 20):
        full = [t for t, e in out["tokens"].items() if e[mk]["first_full_cert_it"] is not None and e[mk]["first_full_cert_it"] <= c]
        dac = [t for t, e in out["tokens"].items() if e[mk]["first_dac1_it"] is not None and e[mk]["first_dac1_it"] <= c]
        out["counts"][f"{mk}_cap{c}"] = {"full_cert": len(full), "dac_only_cert": len(dac), "full_cert_tokens": full}
        if mk == "m0.05" and c == 5:
            out["control_m0.05_cap5_full_must_be_0"] = len(full) == 0
json.dump(out, open(HERE + "/whatif_25.json", "w", encoding="utf-8"), indent=1)
print(out["n"], {k: (v["full_cert"], v["dac_only_cert"]) for k, v in out["counts"].items()}, out.get("control_m0.05_cap5_full_must_be_0"))
