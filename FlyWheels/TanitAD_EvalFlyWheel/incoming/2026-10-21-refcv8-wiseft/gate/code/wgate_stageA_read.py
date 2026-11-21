"""R8-WISEFT-GATE stage A (registered f0ae643): gate X admissible iff GW_X - theta_0.5 > 0 separated at log_name at BOTH seeds and
min(d_s0, d_s1) > 2 x |d_s0 - d_s1|; X* = admissible gate with the higher mean PDMS.  Reads r8_gate_eval.py's output, where the
'R8_RT' plan arm IS theta_0.5 R8_LEGAL (--rt-csv).  Controls: the tool's own (never-fires == plan, always-fires == prior, G-G == 0)."""
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
assert r["controls"]["all_pass"], r["controls"]
assert r["arms"]["R8_RT_s0"]["PDMS"] == 76.3342 and r["arms"]["R8_RT_s1"]["PDMS"] == 76.4399, "plan arm is not theta_0.5"
out = {"id": "R8-WISEFT-GATE stage A", "registered": "f0ae643 (2026-10-21 15:31 Berlin)", "gates": {}}
for g in ("G_CMD", "G_ARC"):
    p = {s: r["pairs"][f"{g}_{s}__minus__R8_RT"] for s in ("s0", "s1")}
    d = {s: p[s]["delta_x100"] for s in p}
    sep = {s: bool(p[s]["interval"]["separated_log_name"]) and p[s]["interval"]["lo"] > 0 for s in p}
    fl = abs(d["s0"] - d["s1"])
    adm = all(v > 0 for v in d.values()) and all(sep.values()) and min(d.values()) > 2 * fl
    out["gates"][g] = {"minus_theta05": {s: {k: p[s]["interval"][k] for k in ("delta", "lo", "hi")} for s in p}, "floor": round(fl, 4),
                       "admissible": adm, "PDMS": {s: r["arms"][f"{g}_{s}"]["PDMS"] for s in ("s0", "s1")},
                       "fallback_frac": {s: r["arms"][f"{g}_{s}"]["fallback_frac"] for s in ("s0", "s1")},
                       "minus_PRIOR": {s: r["pairs"][f"{g}_{s}__minus__PRIOR"]["delta_x100"] for s in ("s0", "s1")}}
adm = [g for g in out["gates"] if out["gates"][g]["admissible"]]
out["X_star"] = max(adm, key=lambda g: sum(out["gates"][g]["PDMS"].values())) if adm else None
out["verdict"] = f"X* = {out['X_star']}" if adm else "NONE ADMISSIBLE -- no navtest gate read for theta_0.5"
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("QQWGA " + json.dumps(out) + " QQ")
