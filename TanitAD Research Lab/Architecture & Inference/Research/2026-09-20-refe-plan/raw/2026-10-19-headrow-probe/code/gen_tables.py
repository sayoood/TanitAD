"""Render the markdown tables of RESULT.md from the run's JSON (probe_result.json, cid.json, nullrate.json, noise.json, ares.json, cpos/planted_info.json).  Pure formatting: no number is computed here
that is not in the JSON.  usage: python gen_tables.py <run_dir> <out.md>"""
import json
import os
import sys

run, outp = sys.argv[1], sys.argv[2]
J = lambda n: json.load(open(os.path.join(run, n), encoding="utf-8"))     # noqa: E731
R = J("probe_result.json")
ROWS = ("nc", "dac", "ep", "ttc")
NAME = {"nc": "NC", "dac": "DAC", "ep": "EP", "ttc": "TTC"}


def ci(d, f="{:.4f}", scale=1.0):
    if d is None:
        return "n/a"
    return f"{f.format(d['point'] * scale)} [{f.format(d['lo'] * scale)}, {f.format(d['hi'] * scale)}]"


L = []
w = L.append
w("### T1. Decision table (concordance over all 64 proposals, OOF; M - L; paired log-cluster bootstrap 10,000 / seed 20260927)\n")
w("| row | seed | P2 ref (in-sample) | L | L - P2 | M | M - L [95 %] | CAPACITY clause (lo > +0.01) | FEATURE clause (hi < +0.01) |")
w("|---|---|---|---|---|---|---|---|---|")
for r in ROWS:
    row = R["rows"][r]
    for sd in ("0", "1"):
        s = row["seeds"][sd]
        p2 = row["p2_reference"]["conc64"]["point"]
        l, m = s["L"]["conc64"]["point"], s["M"]["conc64"]["point"]
        ml = s["M_minus_L"]["conc64"]
        w(f"| {NAME[r]} | {sd} | {p2:.4f} | {l:.4f} | {l - p2:+.4f} | {m:.4f} | {ci(ml)} | {'yes' if ml['lo'] > 0.01 else 'no'} | {'yes' if ml['hi'] < 0.01 else 'no'} |")
w("\n| row | CAPACITY (both seeds) | FEATURE M-L clause | L within 0.01 of P2 | C2 row RED | C-POS GREEN | base call | **final call** | reasons |")
w("|---|---|---|---|---|---|---|---|---|")
for r in ROWS:
    row = R["rows"][r]
    d = row["decision"]
    w(f"| {NAME[r]} | {d['cap']} | {d['feat_ml']} | {d['l_within_p2']} | {row['c2']['row_red']} | {R['c_pos']['green']} | {d['base_call']} | **{d['call']}** | {'; '.join(d['reasons']) or '-'} |")

w("\n### T2. Per row x arm x seed (OOF; SELECTION tier; intervals = another draw of LOGS only; P2 reference is IN-SAMPLE for P2)\n")
w("| row | seed | arm | conc all-64 | conc top-8 | Brier | dPDMS x100 vs P2 | d unsafe-pick rate vs P2 | changed picks | train BCE |")
w("|---|---|---|---|---|---|---|---|---|---|")
for r in ROWS:
    row = R["rows"][r]
    p = row["p2_reference"]
    w(f"| {NAME[r]} | - | P2 ref | {ci(p['conc64'])} | {ci(p['conc8'])} | {ci(p['brier'], '{:.5f}')} | 0 | 0 (rate {p['unsafe_rate_p2']:.5f}) | 0 | - |")
    for sd in ("0", "1"):
        for arm in ("L", "M"):
            c = row["seeds"][sd][arm]
            w(f"| {NAME[r]} | {sd} | {arm} | {ci(c['conc64'])} | {ci(c['conc8'])} | {ci(c['brier'], '{:.5f}')} | {ci(c['d_pdms_x100_vs_p2'], '{:+.3f}')} | {ci(c['d_unsafe_vs_p2'], '{:+.4f}')} | {c['changed_pick_fraction']:.3f} | {c['train_bce_mean_over_folds']:.5f} |")

w("\n### T3. M - L on every metric (paired)\n")
w("| row | seed | conc all-64 | conc top-8 | Brier | dPDMS x100 (M-L) | d unsafe-pick (M-L) |")
w("|---|---|---|---|---|---|---|")
for r in ROWS:
    for sd in ("0", "1"):
        m = R["rows"][r]["seeds"][sd]["M_minus_L"]
        w(f"| {NAME[r]} | {sd} | {ci(m['conc64'])} | {ci(m['conc8'])} | {ci(m['brier'], '{:+.5f}')} | {ci(m['d_pdms_x100'], '{:+.3f}')} | {ci(m['d_unsafe'], '{:+.4f}')} |")

w("\n### T4. Controls\n")
cid = J("cid.json")
w(f"**C-ID** {'GREEN' if cid['ok'] else 'RED'}: all 8 (row, arm) cells: z == z_stored under torch.equal on all {R['n_sets']} sets, skip == P2's row, M output layer exactly 0, picks / concordance / Brier sums equal P2's exactly (max |z - z_stored| = {max(c['max_abs_diff'] for c in cid['cells'].values())}).\n")
w("**C2** (cold start, labels permuted within set, 5 replicates; RED iff any replicate has lower CI of (conc - 0.5) > 0):\n")
w("| row | arm | RED | lower CI of (conc - 0.5): min over reps | max over reps | point (conc - 0.5) range |")
w("|---|---|---|---|---|---|")
for r in ROWS:
    for arm in ("L", "M"):
        c = R["rows"][r]["c2"]["cells"].get(arm, {})
        if "replicates" not in c:
            w(f"| {NAME[r]} | {arm} | MISSING | | | |")
            continue
        pts = [x["point"] - 0.5 for x in c["replicates"]]
        w(f"| {NAME[r]} | {arm} | {c['red']} | {c['min_lo_minus_half']:+.4f} | {c['max_lo_minus_half']:+.4f} | {min(pts):+.4f} .. {max(pts):+.4f} |")
w("\n**C2-DEC** (shuffled M - L, lower CI > +0.01 on any replicate): " + ", ".join(f"{NAME[r]} {R['rows'][r]['c2'].get('dec_red')}" for r in ROWS) + "\n")
w("| row | C2-DEC replicate M - L [95 %] (5 replicates) |")
w("|---|---|")
for r in ROWS:
    reps = R["rows"][r]["c2"].get("dec_replicates_M_minus_L", [])
    w(f"| {NAME[r]} | " + "; ".join(ci(x, '{:+.4f}') for x in reps) + " |")
nr = R.get("null_rate")
if nr:
    w("\n**The C2 rule's false-RED rate under its null** (iid scores independent of the labels, same per-log concordance, same bootstrap, same rule; 200 trials x 5 replicates per row):\n")
    w("| row | trials with >= 1 RED | rate per cell [Wilson 95 %] | replicates RED | rate per replicate [Wilson 95 %] |")
    w("|---|---|---|---|---|")
    for r in ROWS:
        x = nr["rows"][r]
        w(f"| {NAME[r]} | {x['trials_with_red']}/{nr['trials']} | {x['rate_per_cell']:.3f} [{x['wilson95_cell'][0]:.3f}, {x['wilson95_cell'][1]:.3f}] | {x['replicates_red']}/{nr['trials'] * nr['reps']} | {x['rate_per_replicate']:.4f} [{x['wilson95_replicate'][0]:.4f}, {x['wilson95_replicate'][1]:.4f}] |")
    w(f"\nMean per-cell false-RED rate {nr['mean_rate_per_cell']:.3f}; family rate over the 8 (row, arm) cells 1-(1-p)^8 = {nr['family_rate_8_cells']:.3f}.")
cp = R["c_pos"]
w("\n**C-POS** (planted nonlinear signal on the real features, cold start; GREEN iff M - L lower CI > +0.01 on both seeds): " + str(cp["green"]) + "\n")
w("| seed | L conc | M conc | M - L [95 %] |")
w("|---|---|---|---|")
for sd, x in cp["seeds"].items():
    w(f"| {sd} | {ci(x['L'])} | {ci(x['M'])} | {ci(x['M_minus_L'])} |")

w("\n### T5. C-LEAK (NC and DAC first) with the matched noise-row comparator\n")
w("| row | seed | arm | d unsafe-pick vs P2 [95 %] | changed picks | matched noise sigma | noise d unsafe (2 noise seeds) | noise changed picks |")
w("|---|---|---|---|---|---|---|---|")
for r in ("nc", "dac"):
    row = R["rows"][r]
    for sd in ("0", "1"):
        mn = row["c_leak"].get("matched_noise", {}).get(sd)
        for arm in ("M", "L"):
            c = row["seeds"][sd][arm]
            if arm == "M" and mn:
                nz = "; ".join(f"{x['d_unsafe']['point']:+.4f}" for x in mn["cells"])
                w(f"| {NAME[r]} | {sd} | M | {ci(c['d_unsafe_vs_p2'], '{:+.4f}')} | {c['changed_pick_fraction']:.3f} | {mn['sigma']} | {nz} | {mn['mean_changed_pick_fraction']:.3f} |")
            else:
                w(f"| {NAME[r]} | {sd} | {arm} | {ci(c['d_unsafe_vs_p2'], '{:+.4f}')} | {c['changed_pick_fraction']:.3f} | | | |")
w("\nFlags: " + ", ".join(f"{NAME[r]} {R['rows'][r]['c_leak']['flag']}" for r in ("nc", "dac")))
try:
    nz = J("noise.json")
    w("\n**C-NOISE grid** (P2's stored logit of the row + iid Gaussian noise; no fit; all 15,808 sets):\n")
    w("| row | sigma | noise seed | changed picks | d unsafe [95 %] | dPDMS x100 |")
    w("|---|---|---|---|---|---|")
    for c in nz["cells"]:
        w(f"| {NAME[c['row']]} | {c['sigma']} | {c['noise_seed']} | {c['changed_pick_fraction']:.3f} | {ci(c['d_unsafe'], '{:+.4f}')} | {ci(c['d_pdms_x100'], '{:+.3f}')} |")
except FileNotFoundError:
    pass

try:
    A = J("ares.json")
    w("\n### T6. A-RES (zero training; EP-R1 f10 rows applied; P2 stored NC/DAC logits)\n")
    w("| seed | n V | n S | n D | frac(V, m >= 0) | median m (V) | AUC_V (safe > unsafe, top-8) | AUC all | share V new = best pNC*pDAC | frac(D, m >= 0) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for sd, x in A["per_seed"].items():
        w(f"| {sd} | {x['n_V']} | {x['n_S']} | {x['n_D']} | {ci(x['frac_V_m_ge0'])} | {x['median_m_V']} | {ci(x['auc_V'])} | {ci(x['auc_all'])} | {ci(x['share_V_new_is_best_pnc_pdac'])} | {ci(x['frac_D_m_ge0'])} |")
    w(f"\n**Flag: {A['flag']['flag']}** ({json.dumps({k: v for k, v in A['flag'].items() if k != 'flag'})}).  Controls: " + json.dumps({k: (v.get('ok', v)) for k, v in A["controls"].items()})[:600])
except FileNotFoundError:
    pass


for nm, title in (("c2x.json", "T8a. C2x (Amendment 2; cross-set label permutation; POST-HOC, descriptive)"), ("c2y.json", "T8b. C2y (Amendment 3; entry-level label permutation; POST-HOC, descriptive)")):
    try:
        X = J(nm)
    except FileNotFoundError:
        continue
    w("\n### " + title + "\n")
    w("| cell | concordance - 0.5 | lower bound | RED |")
    w("|---|---|---|---|")
    for k, v in X["cells"].items():
        w(f"| {k} | {v['conc_minus_half']:+.4f} | {v['lo_minus_half']:+.4f} | {v['red']} |")
    w("\nM - L per row: " + "; ".join(f"{NAME[r]} {ci(d, '{:+.4f}')}" for r, d in X["rows_M_minus_L"].items()) + f".  RED cells: {X['n_red']}/8; all points within +-0.01: {X['all_points_within_0.01']}.")
try:
    K = J("index_diag.json")
    w("\n### T9. Proposal-index diagnostic (Amendment 3; z = proposal index vs the real labels; no training)\n")
    w("| row | all-64 concordance [95 %] |")
    w("|---|---|")
    for r in ROWS:
        w(f"| {NAME[r]} | {ci(K['rows'][r])} |")
except FileNotFoundError:
    pass
try:
    Z = J("c2z.json")
    w("\n### T10. C2z (Amendment 4; L arm, noise-trained probes, entry-level permutation, 8 replicates per row; POST-HOC, descriptive)\n")
    w("| row | RED replicates (lower bound > 0) | significantly BELOW 0.5 (upper bound < 0) | rate RED [Wilson 95 %] | (concordance - 0.5): min .. max |")
    w("|---|---|---|---|---|")
    for r in ROWS:
        x = Z["rows"][r]
        w(f"| {NAME[r]} | {x['n_red']}/8 | {x['n_significantly_below']}/8 | {x['rate_red']:.3f} [{x['wilson95_red'][0]:.3f}, {x['wilson95_red'][1]:.3f}] | {x['min']:+.4f} .. {x['max']:+.4f} |")
    w(f"\nMax per-row RED rate {Z['max_rate_red']:.3f}; above 10 %: {Z['c2_rule_miscalibrated_rate_gt_10pct']}.")
except FileNotFoundError:
    pass

try:
    SP = J("slot_prior.json")
    w("\n### T11. Slot-prior diagnostic (Amendment 5; the leave-fold-out mean label of each proposal SLOT, no scene information, no training; POST-HOC, descriptive)\n")
    w("| row | SLOT PRIOR all-64 concordance [95 %] | P2 ref | P2 - slot prior | L seed 0 - slot prior | L seed 1 - slot prior | slot-marginal range (fold 0) |")
    w("|---|---|---|---|---|---|---|")
    for r in ROWS:
        x = SP["rows"][r]
        sp0 = x["prior_spread_by_fold"].get("0", x["prior_spread_by_fold"].get(0))
        w(f"| {NAME[r]} | {ci(x['slot_prior'])} | {x['p2_reference']['point']:.4f} | {ci(x['p2_minus_slot'], '{:+.4f}')} | {ci(x['L_s0_minus_slot'], '{:+.4f}')} | {ci(x['L_s1_minus_slot'], '{:+.4f}')} | {sp0['min']:.3f} .. {sp0['max']:.3f} |")
    cr = SP["committed_reading"]
    w(f"\nCommitted reading (Amendment 5): slot prior >= 0.80 on NC ({cr['nc']:.4f}) and TTC ({cr['ttc']:.4f}) => C2 / C2x REDs explained by the slot prior: **{cr['explained']}**.")
except FileNotFoundError:
    pass

oc = R["outcome"]
w("\n### T7. Outcome row\n")
w(f"Label: **{oc['label']}**.  Next arms: " + "; ".join(f"[{a['row']}] {a['arm']}" for a in oc["next_arms"]) + (".  Notes: " + "; ".join(oc["notes"]) if oc["notes"] else ""))
open(outp, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote", outp)
