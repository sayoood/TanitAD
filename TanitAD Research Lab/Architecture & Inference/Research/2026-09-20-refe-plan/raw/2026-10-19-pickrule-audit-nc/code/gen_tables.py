"""Render run/results/result.json as Markdown tables (stdout).  python gen_tables.py [result.json]"""
from __future__ import annotations

import json
import sys

DEFAULT = "C:/Users/Admin/par18/pickrule/run/results/result.json"


def ci(d: dict, nd: int = 4, pct: bool = False) -> str:
    k = 100.0 if pct else 1.0
    return f"{d['point'] * k:.{nd}f} [{d['lo'] * k:.{nd}f}, {d['hi'] * k:.{nd}f}]"


def share(d: dict) -> str:
    return f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def render(R: dict) -> str:
    out = []
    A = R["analysis"]
    out.append(f"### Outcome\n\n* **Final outcome: {R['final_outcome']}** (PRIMARY source OOF, both seeds must agree). Source-sensitive (OOF vs BANKED outcome differ): **{R['source_sensitive']}**.")
    for src, v in A["by_source"].items():
        out.append(f"* {src}: seed classes {v['classes']} -> **{v['outcome']}**; VETO-RECOVERS **{v['veto_recovers']}** (recovering tau: {v['recovering_taus'] or 'none'}); committed next arm: {v['next_arm']}")
    out.append("")
    out.append("### Gate and controls\n\n| check | result | detail |\n|---|---|---|")
    for k, v in R["gate"]["checks"].items():
        det = json.dumps(v["detail"], sort_keys=True)
        out.append(f"| {k} | {'PASS' if v['pass'] else 'FAIL'} | {det[:420]} |")
    out.append("")
    out.append("### Q1 / Q2: overridden share (paired log-cluster bootstrap, B = 10,000, seed 20260927; CI = percentile 2.5 / 97.5)\n")
    out.append("| source / seed | n sets A | overridden | **Q1** | class | Q2 (P2 own NC-unsafe picks) | n | Q2 matched | Q1 - Q2 | median NC sigmoid at A picks |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for key, c in A["composites"].items():
        q = c["q"]
        med = q["sigmoid_nc_at_A_picks_quantiles_min_q25_med_q75_max"]
        out.append(f"| {key} | {q['n_A']} | {q['n_over']} | **{share(q['q1'])}** | **{q['class']}** | {share(q['q2'])} | {q['n_B']} | {share(q['q2_matched'])} (n {q['n_B_matched']}) | {ci(q['q1_minus_q2'], 3)} | {med[2]:.3f} (q25 {med[1]:.3f}, q75 {med[3]:.3f}) |")
    out.append("")
    out.append("Descriptive sensitivities (no authority): unsafe := NC label < 1, and the DAC analogue (DAC label 0, DAC sigmoid < 0.5).\n")
    out.append("| source / seed | Q1 (NC < 1) | n A | Q2 (NC < 1) | DAC Q1 | n A (DAC) | DAC Q2 | EP-led picks changed vs P2 |")
    out.append("|---|---|---|---|---|---|---|---|")
    for key, c in A["composites"].items():
        l1, d = c["q_nc_lt1_sensitivity"], c["dac_analogue"]
        out.append(f"| {key} | {share(l1['q1'])} | {l1['n_A']} | {share(l1['q2'])} | {share(d['q1'])} | {d['n_A']} | {share(d['q2'])} | {c['n_picks_changed_vs_p2']} |")
    out.append("")
    out.append("### Q3: hard NC veto sweep on the EP-led composite (rows unchanged; x100 for PDMS; rates over all 15,808 sets; d = composite_tau minus P2 UNVETOED, paired)\n")
    p2 = A["p2"]["P2_unvetoed"]
    out.append(f"P2 unvetoed: NC-unsafe pick rate {ci(p2['nc_unsafe_rate'])}, DAC-unsafe pick rate {ci(p2['dac_unsafe_rate'])}, pick-level PDMS x100 {ci(p2['pdms_x100'], 3)}; P2's NC-unsafe 97.5th percentile (P2_hi) = {A['p2_nc_unsafe_rate_hi']:.5f}.\n")
    hdr = "| source / seed | tau | NC-unsafe rate | DAC-unsafe rate | PDMS x100 | d NC-unsafe | d DAC-unsafe | **d PDMS x100** | fallback sets | picks changed by the veto |\n|---|---|---|---|---|---|---|---|---|---|"
    out.append(hdr)
    for key, c in A["composites"].items():
        for t, s in c["sweep"].items():
            out.append(f"| {key} | {t} | {ci(s['nc_unsafe_rate'])} | {ci(s['dac_unsafe_rate'])} | {ci(s['pdms_x100'], 3)} | {ci(s['d_nc_unsafe_rate'])} | {ci(s['d_dac_unsafe_rate'])} | **{ci(s['d_pdms_x100'], 3)}** | {s['n_fallback']} | {s['n_changed_vs_unvetoed']} |")
    out.append("")
    out.append("Descriptive reference: the same veto applied to P2's own rows (no EP change).\n")
    out.append("| tau | NC-unsafe rate | DAC-unsafe rate | PDMS x100 | d PDMS x100 vs P2 unvetoed | fallback | changed |\n|---|---|---|---|---|---|---|")
    for t, s in A["p2"]["P2_with_veto"].items():
        out.append(f"| {t} | {ci(s['nc_unsafe_rate'])} | {ci(s['dac_unsafe_rate'])} | {ci(s['pdms_x100'], 3)} | {ci(s['d_pdms_x100'], 3)} | {s['n_fallback']} | {s['n_changed_vs_unvetoed']} |")
    out.append("")
    out.append("### VETO-RECOVERS, every tau (both seeds must satisfy both conditions)\n")
    out.append("| source | tau | (i) NC-unsafe rate <= P2_hi, seeds 0 / 1 | (ii) d PDMS lower CI >= 0, seeds 0 / 1 | rate points | d PDMS lower CI x100 | recovers |\n|---|---|---|---|---|---|---|")
    for src, v in A["by_source"].items():
        for t, f in v["veto_flags"].items():
            out.append(f"| {src} | {t} | {f['cond_i_rate_le_p2_hi']} | {f['cond_ii_dpdms_lo_ge_0']} | {[round(x, 5) for x in f['rate_points']]} | {[round(x, 3) for x in f['dpdms_lo_x100']]} | **{f['recovers']}** |")
    out.append("")
    out.append("### C-NULL (labels permuted within each set; a replicate FIRES iff its Q1 lower CI > 0.5; VOID iff >= 2 of 5 fire for any source / seed)\n")
    out.append(f"VOID: **{R['null']['void']}**; fire rate over all replicates: {R['null']['fire_rate']:.3f}.\n")
    out.append("| source / seed | fires | Q1 per replicate (point [lo, hi]; n A) |\n|---|---|---|")
    for key, v in R["null"]["by_key"].items():
        reps = "; ".join(f"{share(r['q1'])} n{r['n_A']}" for r in v["replicates"])
        out.append(f"| {key} | {v['fires']} of {v['of']} | {reps} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    txt = render(json.load(open(p, "r", encoding="utf-8")))
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
            f.write(txt)
    else:
        sys.stdout.write(txt)
