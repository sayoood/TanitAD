"""Apply SPEC Amendment A3's committed bar verbatim to probe_te_result.json:
PRIMARY BAR-A3: P9 - P10 ADE < 0 with the CI excluding 0 at 6 s, AND P9 - P10 ADE not separated-worse at 2 s.
Validity: P11 - P10 NOT separated-better at either horizon (else VOID: capacity leak).
Outcomes: POSITIVE | NEGATIVE | VOID."""
import json, sys

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_probe/out_te/probe_te_result.json"))
p2, p6 = r["pairs_2s_ade"], r["pairs_6s"]


def f(v):
    return f"{v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{' SEP' if v['separated'] else ''}"


a2, a6, f6 = p2["P9_trunk_sp_mlp-P10_kin_mlp"], p6["P9_trunk_sp_mlp-P10_kin_mlp"]["ade"], p6["P9_trunk_sp_mlp-P10_kin_mlp"]["fde"]
s2, s6 = p2["P11_shuf_sp_mlp-P10_kin_mlp"], p6["P11_shuf_sp_mlp-P10_kin_mlp"]["ade"]
k2, k6 = p2["P10_kin_mlp-P2_kin"], p6["P10_kin_mlp-P2_kin"]["ade"]
print("means 2s ADE:", {k: round(v, 4) for k, v in r["means_2s_ade"].items()})
print("means 6s ADE:", {k: round(v, 4) for k, v in r["means_6s_ade"].items()})
print(f"P9 - P10  2s ADE {f(a2)} | 6s ADE {f(a6)} | 6s FDE {f(f6)}")
print(f"P11 - P10 2s ADE {f(s2)} | 6s ADE {f(s6)}   (validity: must NOT be separated-better)")
print(f"P10 - P2  2s ADE {f(k2)} | 6s ADE {f(k6)}   (nonlinear vs linear kinematic readout)")
for k, v in r.get("strata", {}).items():
    if "P9_trunk_sp_mlp-P10_kin_mlp" in v:
        a = v["P9_trunk_sp_mlp-P10_kin_mlp"]["ade"]
        print(f"  stratum {k}: n={v['n_windows']} cl={v['n_clusters']} P9-P10 ADE {f(a)}")
valid = not (s2["separated"] and s2["delta"] < 0) and not (s6["separated"] and s6["delta"] < 0)
bar = (a6["separated"] and a6["delta"] < 0) and not (a2["separated"] and a2["delta"] > 0)
print(f"VALIDITY {'OK' if valid else 'FAILED'} | BAR-A3 {'MET' if bar else 'NOT MET'} | VERDICT: "
      f"{'VOID' if not valid else ('POSITIVE' if bar else 'NEGATIVE')}")
