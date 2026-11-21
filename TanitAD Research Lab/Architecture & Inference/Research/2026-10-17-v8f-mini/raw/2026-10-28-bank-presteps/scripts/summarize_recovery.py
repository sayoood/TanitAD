import json, sys
from pathlib import Path
RAW = Path("E:/Projects/TanitAD-artifacts/dagland/bank_presteps/raw")
for tag, pat in (("V1 (registered)", "recovery_{}.json"), ("V2 (post-hoc)", "recoveryV2_{}.json")):
    for sd in ("s0", "s1"):
        o = json.load(open(RAW / pat.format(sd)))
        f = o["feasibility"]
        t1, t2 = o["kept"]["tick1"], o["kept"]["tick2"]
        dv = o["deviation_m"]
        print(f"{tag} {sd}: tick2 kept {100*t2['baseline_kept_fraction']:.2f}% -> {100*t2['kept_fraction_with_recovery']:.2f}% (ceiling {100*t2['ceiling_if_all_offroute_recovered']:.2f}%), recovered {t2['recovered_emitted']}; "
              f"tick1 kept {100*t1['baseline_kept_fraction']:.2f}% -> {100*t1['kept_fraction_with_recovery']:.2f}% (ceiling {100*t1['ceiling_if_all_offroute_recovered']:.2f}%)")
        print(f"    emitted {f['n_emitted']} violations {f['n_emitted_with_checker_violation']}; refusals {f['refusals_by_status']}")
        print(f"    dev mean_over_k p50 {dv['mean_over_k']['p50']:.2f} p95 {dv['mean_over_k']['p95']:.2f} max {dv['mean_over_k']['max']:.1f}; end6s p50 {dv['end_6s']['p50']:.2f} p95 {dv['end_6s']['p95']:.2f}; human-baseline viol rate {o['human_label_baseline_beside']['rate']:.4f}")
