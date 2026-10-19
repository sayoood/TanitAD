#!/usr/bin/env python3
"""render raw/design_calc.json into the markdown power table the draft prereg quotes (numbers come from the JSON, never retyped).  python power_table.py --in ../raw/design_calc.json --out ../raw/design_calc_table.md"""
import argparse, json
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
d = json.load(open(a.inp)); C = d["conditions"]
rows = []
order = ["B_REST", "carveB_replica", "U_s1b_actual", "U_nominal_123x4", "navtest"]
rows.append("| structure (tokens / logs) | scheme | true effect (pts) | P(O1 verbatim) | P(2-term: point >= 0.30 and lo > 0) | P(sign: lo > 0) | P(NULL: CI contains 0) | P(NEGATIVE: hi < 0) | mean CI half-width |")
rows.append("|---|---|---|---|---|---|---|---|---|")
for s in order:
    st = d["structures"].get(s)
    if not st: continue
    for sc in ("R", "S", "E"):
        for mu in (0.327, 0.16, 0.0):
            k = f"{s}|{sc}|mu={mu}"
            if k not in C: continue
            p = C[k]["prob"]
            rows.append(f"| {s} ({st['n_tokens']} / {st['n_logs']}) | {sc} | {mu} | {p['O1_verbatim']:.3f} | {p['O1_2term']:.3f} | {p['sign_pos']:.3f} | {p['null']:.3f} | {p['negative']:.3f} | {C[k]['ci_half_width_mean']:.3f} |")
open(a.out, "w", encoding="utf-8", newline="\n").write("\n".join(rows) + "\n")
print("ZZTABLE_OK", len(rows))
