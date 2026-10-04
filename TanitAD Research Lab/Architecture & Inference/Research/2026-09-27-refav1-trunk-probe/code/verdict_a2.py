"""Apply SPEC Amendment A2's committed bars to probe_te_result.json, verbatim:
PRIMARY BAR-A2: P6 - P2 ADE@2s < 0 with the CI excluding 0.
Validity: P0 exact (P0 - P0 = 0 and zero-width); P5 - P2 and P8 - P2 NOT separated-better.
Outcomes: POSITIVE (BAR-A2 met) | NEGATIVE (not met, linear AND the nonlinear P9 arm also not better than P2) |
VOID (a validity control fails). P9 - P2 is reported beside it (A1's committed nonlinear arm)."""
import json, sys

r = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refav1_probe/out_te/probe_te_result.json"))
p2 = r["pairs_2s_ade"]


def fmt(k):
    v = p2[k]
    return f"{v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]{' SEP' if v['separated'] else ''}"


ctl0 = p2["P0_kdx-P0_kdx"]
valid = (ctl0["delta"] == 0 and ctl0["lo"] == 0 and ctl0["hi"] == 0
         and not (p2["P5_shuf-P2_kin"]["separated"] and p2["P5_shuf-P2_kin"]["delta"] < 0)
         and not (p2["P8_shuf_sp-P2_kin"]["separated"] and p2["P8_shuf_sp-P2_kin"]["delta"] < 0))
bar = p2["P6_trunk_sp-P2_kin"]["separated"] and p2["P6_trunk_sp-P2_kin"]["delta"] < 0
mlp = p2["P9_trunk_sp_mlp-P2_kin"]["separated"] and p2["P9_trunk_sp_mlp-P2_kin"]["delta"] < 0
verdict = "VOID" if not valid else ("POSITIVE" if bar else ("NEGATIVE" if not mlp else "NEGATIVE-LINEAR / MLP-POSITIVE"))
print("meta:", json.dumps(r["meta"])[:400])
print("means 2s ADE:", {k: round(v, 4) for k, v in r["means_2s_ade"].items()})
print("means 6s ADE:", {k: round(v, 4) for k, v in r["means_6s_ade"].items()})
for k in p2:
    print(f"  2s ADE {k}: {fmt(k)}")
for k, v in r["pairs_6s"].items():
    a, f = v["ade"], v["fde"]
    print(f"  6s {k}: ADE {a['delta']:+.4f} [{a['lo']:+.4f}, {a['hi']:+.4f}]{' SEP' if a['separated'] else ''} | "
          f"FDE {f['delta']:+.4f} [{f['lo']:+.4f}, {f['hi']:+.4f}]{' SEP' if f['separated'] else ''}")
for k, v in r.get("strata", {}).items():
    if v.get("status") == "UNDERPOWERED":
        print(f"  stratum {k}: n={v['n_windows']} clusters={v['n_clusters']} UNDERPOWERED")
        continue
    row = []
    for pk in ("P6_trunk_sp-P2_kin", "P9_trunk_sp_mlp-P2_kin", "P2_kin-P0_kdx"):
        a = v[pk]["ade"]
        row.append(f"{pk.split('-')[0]}-{pk.split('-')[1][:6]} {a['delta']:+.4f}{'*' if a['separated'] else ''}")
    print(f"  stratum {k}: n={v['n_windows']} cl={v['n_clusters']} | " + " | ".join(row))
print(f"VALIDITY {'OK' if valid else 'FAILED'} | BAR-A2 (P6-P2 < 0 sep) {'MET' if bar else 'NOT MET'} | "
      f"P9 MLP-P2 better-sep: {mlp} | VERDICT: {verdict}")
