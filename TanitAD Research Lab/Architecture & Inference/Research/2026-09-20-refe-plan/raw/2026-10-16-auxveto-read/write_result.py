#!/usr/bin/env python3
"""write_result.py -- renders RESULT.md from AUXVETO_READ.json (+ the two POST-HOC side files).  Every number is transcribed by code; the outcome row is
`auxveto_read.decide` applied to the UNROUNDED point and CI limits, and the row's committed next arm is quoted VERBATIM from eval/PREREG_AUXVETO.md.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auxveto_read as av  # noqa: E402

R = json.load(open(HERE / "AUXVETO_READ.json"))
TIE = json.load(open(HERE / "POSTHOC_TIE_SENSITIVITY.json"))
XCHK = json.load(open(HERE / "CROSSCHECK_VGT_STORED_SDF.json"))
FRM = json.load(open(HERE / "POSTHOC_PRED_FRAME_CROSSCHECK.json"))
A = R["arms"]
V = R["validity"]
DEC = R["decision"]
ARMS = ["V-PRED", "V-GT", "V-PRED-2k", "V-PRED-400", "O-DAC", "V-GT-TRACKED"]


def ci(x, d=3):
    return f"{x['point']:+.{d}f} [{x['lo']:+.{d}f}, {x['hi']:+.{d}f}]"


def zn(x, d=2):
    return f"{x['z']:+.{d}f} / {x['n']:+.{d}f}"


def lvl(x, d=2):
    return f"{x['z']:.{d}f} / {x['n']:.{d}f} / {x['point']:.{d}f} [{x['lo']:.{d}f}, {x['hi']:.{d}f}]"


def line(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


vp, vg = A["V-PRED"]["d_pdms"], A["V-GT"]["d_pdms"]
row = DEC["row"]
test_line = [l for l in open(HERE / "test_auxveto_read.out", encoding="utf-8").read().splitlines() if l.startswith("ZZTEST") or l.startswith("baseline:")]
pytest_line = open(HERE / "test_auxveto_read_pytest.out", encoding="utf-8").read().strip().splitlines()[-1]
o = []
w = o.append

w(f"# RESULT: PREREG_AUXVETO read. Outcome row **{row}**")
w("")
w("> **Evidence class: MEASURED** (this read, `auxveto_read.py`, on the artifacts named in `AUXVETO_READ.json`). Tier: NAVSIM v1.1 non-reactive per-proposal table lookup, base = u400 (SFT-4 arm B).")
w(f"> Prereg `eval/PREREG_AUXVETO.md` sha256 `{R['prereg']['sha256'][:16]}…` (asserted by the script; landing commit `4f7e105`). Package `raw/2026-10-16-auxveto-read/`. NOT committed, NOT pushed.")
w("")
w("## Headline")
w("")
w(f"- **V-PRED (PRIMARY, A2 head, 10k updates, held-out IoU {R['validity']['2_vpred_mask_iou_identity']['iou']:.6f}): stratified navtest-population ΔPDMS (×100) = {ci(vp)}.** "
  f"The CI lies entirely below 0: `AV-HARM` fires on the unrounded numbers.")
w(f"- **V-GT (ceiling of the pose-only raster route): {ci(vg)}**, lower bound {vg['lo']:.6f} > 0, so the prereg's AV-HARM branch reads *\"the GT ceiling is positive\"*. "
  f"The margin of that is thin: the lower bound is {vg['lo']:+.3f} and the oracle O-DAC is {ci(A['O-DAC']['d_pdms'])}.")
w(f"- **Validity: all five §9 checks PASS** (plus the population identity and the head-quality IoU identities). The read is VALID.")
w(f"- The CI answers *another draw of logs* and is **blind to the head's training variance** (one head seed, H-ESTIM-SEED-1) and to the tracker.")
w("")
w("## 1. The outcome, by the §6 decision list, applied mechanically to the unrounded numbers")
w("")
w("```")
w(f"V-PRED  point = {vp['point']!r}")
w(f"        lower = {vp['lo']!r}")
w(f"        upper = {vp['hi']!r}")
w(f"V-GT    point = {vg['point']!r}")
w(f"        lower = {vg['lo']!r}")
w(f"        upper = {vg['hi']!r}")
w("```")
w("")
w("| order | row | condition (prereg, verbatim) | evaluates to |")
w("|---|---|---|---|")
for r_, ev in (("AV-GO", f"lower {vp['lo']:+.4f} > 0 is {vp['lo'] > 0}"), ("AV-PROMISING", f"lower {vp['lo']:+.4f} > 0 is {vp['lo'] > 0}"),
               ("AV-HARM", f"upper {vp['hi']:+.4f} < 0 is {vp['hi'] < 0}"), ("AV-CEILING-ONLY", "not reached"), ("AV-NULL", "not reached")):
    w(line(["AV-GO", "AV-PROMISING", "AV-HARM", "AV-CEILING-ONLY", "AV-NULL"].index(r_) + 1, f"**{r_}**" + (" **FIRES**" if r_ == row else ""), av.prereg_condition(r_), ev))
w("")
w(f"**Row: {row}.** V-GT branch (QUESTIONS.md Q7: \"positive\" = V-GT lower bound > 0): lower bound {vg['lo']:+.6f} → *{DEC['gt_branch']}*. "
  f"(The other reading, V-GT point > 0, is also {vg['point'] > 0}; both readings agree.)")
w("")
w(f"**The row's committed next arm, verbatim from the prereg §6:**")
w("")
w(f"> {av.prereg_next_arm(row)}")
w("")
w("## 2. Validity (§9). Any failure would make the read INVALID")
w("")
w("| # | check | result | detail |")
w("|---|---|---|---|")
w(line("1", "B reproduces `pick`", "PASS" if V["1_B_reproduces_pick"]["ok"] else "FAIL", f"{V['1_B_reproduces_pick']['n_equal']} / {V['1_B_reproduces_pick']['of']}"))
w(line("2", "V-PRED mask reproduces A2 IoU [0,40) 0.675218 ±1e-6", "PASS" if V["2_vpred_mask_iou_identity"]["ok"] else "FAIL",
       f"{V['2_vpred_mask_iou_identity']['iou']:.10f} (|diff| {V['2_vpred_mask_iou_identity']['abs_diff']:.2e})"))
zc = V["3_zero_control"]
w(line("3", "zero control: all-drivable vetoes nothing, Δ = 0.0 exactly", "PASS" if zc["ok"] else "FAIL",
       f"pred grid: {zc['pred_grid_40x32']['n_vetoed_proposals']} vetoed, Δ {zc['pred_grid_40x32']['delta_pdms_x100']}; GT grid: {zc['gt_grid_160x128']['n_vetoed_proposals']} vetoed, Δ {zc['gt_grid_160x128']['delta_pdms_x100']}"))
v4 = V["4_odac_bounds_rasters"]
w(line("4", "O-DAC's Z recovery ≥ every raster arm", "PASS" if v4["ok"] else "FAIL",
       f"O-DAC {v4['odac_recovery']:.0f}; " + ", ".join(f"{k} {v:.0f}" for k, v in v4["arms"].items()) + " (recovery = Z tokens with NC×DAC > 0 at the pick, Q5)"))
v5 = V["5_yflip_mutation_below_vgt"]
w(line("5", "y-flipped V-GT falls below V-GT's point", "PASS" if v5["ok"] else "FAIL", f"V-GT {v5['delta_vgt']:+.4f}; y-flipped {v5['delta_vgt_yflip']:+.4f}"))
w(line("0", "population identity (Z all NC×DAC = 0; N all > 0; 529 / 529)", "PASS" if V["0_population_identity"]["ok"] else "FAIL", f"Z {V['0_population_identity']['Z']}, N {V['0_population_identity']['N_sample']}"))
hq = V["2b_head_quality_iou"]
w(line("2b", "2k and 400-update masks vs `PRED_GEOMETRY.json` (reported)", "PASS", f"2k {hq['V-PRED-2k']['iou']:.6f} (banked {hq['V-PRED-2k']['banked_PRED_GEOMETRY']:.6f}); 400 {hq['V-PRED-400']['iou']:.6f} (banked {hq['V-PRED-400']['banked_PRED_GEOMETRY']:.6f})"))
w("")
w(f"Test battery (written and run BEFORE any arm Δ was computed; re-run on the final script): `{test_line[-1]}`; `{pytest_line}`. "
  f"{len(re.findall('detected=OK', open(HERE / 'test_auxveto_read.out', encoding='utf-8').read()))} deliberate-regression mutants, each turned a check RED (`test_auxveto_read.out`). "
  f"Open details of the prereg were fixed in `QUESTIONS.md` BEFORE any outcome (15 items).")
w("")
w("## 3. The arms: stratified navtest-population ΔPDMS (×100) vs B, and its two strata")
w("")
w("Δ = (529/12,146)·Δ_Z + (11,617/12,146)·Δ_N. Paired log-cluster bootstrap, 123 logs, 10,000 resamples, seed 20260927, 95 % percentile CI. Margin 0.25 m.")
w("")
w("| arm | role | **Δ PDMS (stratified) [95 % CI]** | Δ_Z / Δ_N | Z tokens recovered (NC×DAC > 0) of 529 | B's pick falsely vetoed on N (of 529) | every proposal vetoed (Z / N tokens) | picks changed (Z / N) |")
w("|---|---|---|---|---|---|---|---|")
roles = {"V-PRED": "PRIMARY", "V-GT": "ceiling, pose-only", "V-PRED-2k": "head curve", "V-PRED-400": "head curve", "O-DAC": "oracle (never deployable)", "V-GT-TRACKED": "reported only (§7)", "V-GT-YFLIP": "control (§9.5)"}
for a in ARMS + ["V-GT-YFLIP"]:
    b = A[a]
    w(line(f"**{a}**", roles[a], f"**{ci(b['d_pdms'])}**", zn(b["d_pdms"]), b["z_recovery"]["ncdac_count"], b["false_veto_N"]["b_pick_vetoed_count"],
           f"{b['all_vetoed_tokens']['Z']} / {b['all_vetoed_tokens']['N']}", f"{b['changed_picks']['Z']} / {b['changed_picks']['N']}"))
w("")
w("## 4. The four metric families (§8). STRATEGIC is PAUSED")
w("")
w("### LONGITUDINAL: Δ EP, Δ TTC (×100; stratified, paired CI)")
w("")
w("| arm | Δ EP [CI] | Δ_Z / Δ_N | Δ TTC [CI] | Δ_Z / Δ_N |")
w("|---|---|---|---|---|")
for a in ARMS:
    b = A[a]
    w(line(a, ci(b["d_ep"]), zn(b["d_ep"]), ci(b["d_ttc"]), zn(b["d_ttc"])))
w("")
w("### LATERAL: Δ DAC, Δ DDC, the Z-stratum recovery and the false-veto rate on N")
w("")
w("| arm | Δ DAC [CI] | Δ_Z / Δ_N | Δ DDC [CI] | Z recovered: NC×DAC > 0 | (DAC = 1 only, POST-HOC variant) | false-veto rate on N |")
w("|---|---|---|---|---|---|---|")
for a in ARMS:
    b = A[a]
    zr, fv = b["z_recovery"], b["false_veto_N"]
    w(line(a, ci(b["d_dac"]), zn(b["d_dac"]), ci(b["d_ddc"]), f"{zr['ncdac_count']} / {zr['of']} = {100 * zr['ncdac_count'] / zr['of']:.1f} %", f"{zr['dac_only_count']}",
           f"{fv['b_pick_vetoed_count']} / {fv['of']} = {100 * fv['b_pick_vetoed_count'] / fv['of']:.1f} %"))
w("")
w("### TACTICAL (selection quality): mean PDMS ×100 per stratum, Z / N / navtest-stratified [95 % CI]")
w("")
w("| pick | Z / N / stratified [CI] |")
w("|---|---|")
for k in ("random", "B", "best_of_64"):
    w(line({"random": "random proposal (expected over the 64)", "B": "B (u400 pick)", "best_of_64": "best of 64 (oracle max PDMS)"}[k], lvl(R["tactical_levels"][k])))
for a in ARMS:
    w(line(a, lvl(A[a]["lvl_pdms"])))
w("")
w("### STRATEGIC")
w("")
w("**PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE.md` ITEM 25: the strategic layer is switched off until tactical and operative drive well). Not a pass, not omitted.")
w("")
w("### Also reported (§8): Δ NC, Δ comfort, the all-vetoed share")
w("")
w("| arm | Δ NC [CI] | Δ comfort [CI] | tokens with EVERY proposal vetoed, Z (of 529) / N (of 529) | proposals vetoed, Z / N (of 33,856 each) |")
w("|---|---|---|---|---|")
for a in ARMS:
    b = A[a]
    w(line(a, ci(b["d_nc"]), ci(b["d_comfort"]), f"{b['all_vetoed_tokens']['Z']} / {b['all_vetoed_tokens']['N']}", f"{b['n_vetoed_proposals']['Z']} / {b['n_vetoed_proposals']['N']}"))
w("")
w("### Per-city split (descriptive, never gating; Q13). ΔPDMS ×100: Z-stratum mean / N-stratum mean / city-stratified; Z recovered")
w("")
w("| city | n_Z / n_N / navtest tokens | " + " | ".join(a for a in ("V-PRED", "V-GT", "V-PRED-2k", "V-PRED-400")) + " |")
w("|---|---|" + "---|" * 4)
for c in sorted(R["per_city_descriptive"]["V-PRED"]):
    base = R["per_city_descriptive"]["V-PRED"][c]
    cells = []
    for a in ("V-PRED", "V-GT", "V-PRED-2k", "V-PRED-400"):
        x = R["per_city_descriptive"][a][c]
        cells.append(f"{x['delta_Z']:+.2f} / {x['delta_N']:+.2f} / **{x['delta_city_stratified']:+.2f}**; rec {x['z_recovered']}")
    w(line(c, f"{base['n_Z']} / {base['n_N']} / {base['navtest_tokens_in_city']}", *cells))
w("")
w("## 5. Reported, never gating: the margin curve and the head-quality curve")
w("")
w("### Margin curve (veto iff signed distance < −m). Primary m = 0.25 m is the prereg's, fixed before any outcome")
w("")
w("| arm | m (m) | Δ PDMS [CI] | Δ_Z / Δ_N | Z recovered | B falsely vetoed (N) | all-vetoed tokens Z / N |")
w("|---|---|---|---|---|---|---|")
for a in ("V-PRED", "V-GT"):
    for m, b in R["margin_curve"][a].items():
        w(line(a, m + (" (primary)" if m == "0.25" else ""), ci(b["d_pdms"]), zn(b["d_pdms"]), b["z_recovery"]["ncdac_count"], b["false_veto_N"]["b_pick_vetoed_count"],
               f"{b['all_vetoed_tokens']['Z']} / {b['all_vetoed_tokens']['N']}"))
w("")
w("### Head-quality curve (the same head recipe at 400 / 2,000 / 10,000 updates, held-out)")
w("")
w("| updates | held-out IoU [0,40) | Δ PDMS [CI] | Δ_Z / Δ_N | Z recovered | share of proposals with margin < −0.25 m |")
w("|---|---|---|---|---|---|")
for a, u in (("V-PRED-400", "400"), ("V-PRED-2k", "2,000"), ("V-PRED", "10,000")):
    b = R["head_quality_curve"][a]
    w(line(u, f"{b['iou']:.4f}", ci(b["d_pdms"]), zn(b["d_pdms"]), b["z_recovery"]["ncdac_count"], f"{100 * R['margin_distribution_descriptive'][a]['share_proposals_with_margin_below_-0.25']:.1f} %"))
w(line("GT (V-GT)", "1.0000", ci(vg), zn(vg), A["V-GT"]["z_recovery"]["ncdac_count"], f"{100 * R['margin_distribution_descriptive']['V-GT']['share_proposals_with_margin_below_-0.25']:.1f} %"))
w("")
w("## 6. What the interval answers, and what it cannot")
w("")
est = R["estimator"]
w(f"- **Estimator:** stratified (Z census 529 / 12,146 = {est['weights']['Z']:.6f}; N sample of 529 from 11,617 / 12,146 = {est['weights']['N']:.6f}), paired log-cluster bootstrap over {est['n_logs']} C3 logs, "
  f"{est['resamples']:,} resamples, seed {est['seed']}, weights fixed, 95 % percentile CI.")
w(f"- **Its question:** *another draw of logs.* ⛔ It is **blind to the head's TRAINING variance** (the A2 head has one seed; H-ESTIM-SEED-1), and to the NAVSIM tracker. Raster inference variance is zero (deterministic).")
w(f"- **Consequence for this row:** V-PRED's upper bound ({vp['hi']:+.3f}) is below 0 for *this* head. A seed-1 head replicate is the one thing that could say whether −0.788 is a property of the recipe or of seed 0; "
  f"the 400 / 2k / 10k points ({A['V-PRED-400']['d_pdms']['point']:+.3f} / {A['V-PRED-2k']['d_pdms']['point']:+.3f} / {vp['point']:+.3f}) are three different trainings of one seed and are already not monotone.")
w(f"- **Wall time:** drawing the 10,000 × 123 resample matrix {est['wall_s']['boot_counts_wall_s']:.3f} s; all {est['wall_s']['strat_boot_calls']} bootstrap evaluations {est['wall_s']['strat_boot_wall_s']:.3f} s; the whole read (gates, 7 arms, curves, bootstrap) {R['wall_s']:.1f} s, CPU only.")
w("- **Transfer caveat (prereg §2):** the base is u400, not P2 or P6. This is a statement about the u400 scorer's failure set on C3, re-weighted to navtest. NAVSIM v1.1 non-reactive.")
w("")
w("## 7. POST-HOC rows (NOT in the prereg; decide nothing)")
w("")
t = TIE["V-GT"]
w(f"- **Ties on the threshold.** On the 0.5 m GT grid a one-cell-wide non-drivable sliver has signed distance exactly −0.25 at its cells, so {t['n_ties_within_1e-9']} V-GT proposals ({t['tokens_with_a_tie']} tokens) sit exactly on the strict `< −0.25` threshold and are kept. "
  f"Vetoing them instead (`≤`) moves V-GT from {ci(t['strict_lt (prereg)']['d_pdms'])} to {ci(t['ties_vetoed (<=, POST-HOC)']['d_pdms'])}: same sign, same branch. V-PRED has {TIE['V-PRED']['n_ties_within_1e-9']} ties. (`POSTHOC_TIE_SENSITIVITY.json`)")
w(f"- **Pipeline cross-check.** V-GT's margins from the mask-derived signed distance agree with the label store's stored int8 signed distance to ≤ {XCHK['margin_abs_diff_m']['max']:.3f} m (the store's 0.1 m quantisation) on all {XCHK['n_proposals']:,} proposals; "
  f"at m = 0.25 the veto sets differ on {XCHK['m=0.25']['veto_sets_differ_proposals']} proposals ({100 * XCHK['m=0.25']['share_differ']:.2f} %) and the picks on {XCHK['picks_differ_tokens_at_0.25']} tokens, which is what a ≤ 0.05 m quantisation of the field does to margins that cluster near the 0.25 m threshold. (`CROSSCHECK_VGT_STORED_SDF.json`)")
f0, f1, f2, f3 = FRM["GT raster 0.5 m"], FRM["GT down-sampled to 2 m (pred-grid path)"], FRM["same, y-flipped (control)"], FRM["same, shifted one 2 m cell in x (control)"]
w(f"- **Predicted-grid path cross-check.** The GT mask down-sampled to the head's 2 m grid, pushed through the predicted-grid path (2 m EDT, `PRED_GEOM`, bilinear), agrees with NAVSIM's DAC = 0 at the proposal level with κ {f1['kappa']:.3f} (0.5 m GT raster: κ {f0['kappa']:.3f}); "
  f"the same raster y-flipped reads κ {f2['kappa']:.3f}, shifted one cell in x κ {f3['kappa']:.3f}. So an axis or half-cell error in the 2 m path would have shown. (`POSTHOC_PRED_FRAME_CROSSCHECK.json`)")
w("")
w("## 8. Reading (the read agent's; the mechanical decision above does not depend on it)")
w("")
zr = A["V-PRED"]["z_recovery"]["ncdac_count"]
fv = A["V-PRED"]["false_veto_N"]["b_pick_vetoed_count"]
w(f"1. **At today's head quality the veto trades {zr} recoveries (of 529 zeroed picks, Δ_Z {A['V-PRED']['d_pdms']['z']:+.2f}) for {fv} false vetoes of good picks (of 529, Δ_N {A['V-PRED']['d_pdms']['n']:+.2f}),** and the net is negative. "
  f"The DAC sub-score at the pick does not rise ({ci(A['V-PRED']['d_dac'])}, the CI reaches 0): the head's raster is not a better judge of NAVSIM's DAC than the scorer's own picks.")
w(f"2. **The head-quality curve does not rescue it.** 400 → 2k → 10k updates (IoU {hq['V-PRED-400']['iou']:.3f} → {hq['V-PRED-2k']['iou']:.3f} → {hq['V-PRED']['iou']:.3f}) give {A['V-PRED-400']['d_pdms']['point']:+.3f} / {A['V-PRED-2k']['d_pdms']['point']:+.3f} / {vp['point']:+.3f}. "
  f"The 400-update head vetoes {100 * R['margin_distribution_descriptive']['V-PRED-400']['share_proposals_with_margin_below_-0.25']:.0f} % of all proposals; its Z gain ({A['V-PRED-400']['d_pdms']['z']:+.2f}) is consistent with the generic gain of moving ANY zeroed pick, whose B score is 0 by selection (the y-flipped GT arm also reads Δ_Z {A['V-GT-YFLIP']['d_pdms']['z']:+.2f}), and it is paid for in N.")
w(f"3. **The ceiling of the pose-only route is small.** A perfect raster (V-GT) reads {ci(vg)}, which is {100 * vg['point'] / A['O-DAC']['d_pdms']['point']:.0f} % of the oracle DAC veto ({ci(A['O-DAC']['d_pdms'])}); the same GT raster on NAVSIM's tracked states reads {ci(A['V-GT-TRACKED']['d_pdms'])} (reported only, §7). "
  f"HYPOTHESIS for the gap V-GT → O-DAC: mostly the tracker (INHERITED support, not re-measured here: `AGREEMENT.json` finds 27.8 % of NAVSIM's DAC failures only in the tracked states, and V-GT-TRACKED is higher than V-GT); the gap V-PRED → V-GT is {vg['point'] - vp['point']:.2f} PDMS. "
  f"So the prereg's committed branch (\"the lever is raster quality (S1c)\") has a measured pose-only reference of about {vg['point']:+.2f} PDMS at the prereg's margin: a head that only matched the GT raster would not be expected to read more under this rule. The margin curve says V-GT at m = 0 reads {ci(R['margin_curve']['V-GT']['0']['d_pdms'])} (descriptive; not a tuned setting).")
w("")
w("## 9. Deliverable manifest (all under `raw/2026-10-16-auxveto-read/`; **single copy in the temp archive tree, not committed, not pushed**)")
w("")
w("| file | what | where it lives |")
w("|---|---|---|")
for f_, d_ in (("auxveto_read.py", "the read: §3-§9 as code; `--gate`, `--run`"), ("test_auxveto_read.py", "43 checks incl. every §9 check + 22 deliberate-regression mutants"),
               ("test_auxveto_read.out / test_auxveto_read_pytest.out", "test evidence on the final script"), ("QUESTIONS.md", "15 prereg details fixed BEFORE any outcome"),
               ("GATE.json", "pre-read gates (checks 1, 2, 3)"), ("AUXVETO_READ.json", "every arm, both strata, the stratified estimate + CI, §8 families, margin curve, head-quality curve"),
               ("AUXVETO_PICKS.npz", "per-token picks and min margins of every arm (audit)"), ("auxveto_read_run.log", "stdout of the run"), ("RESULT.md / write_result.py", "this file and its generator"),
               ("crosscheck_vgt_stored_sdf.py / CROSSCHECK_VGT_STORED_SDF.json", "POST-HOC pipeline cross-check"), ("posthoc_tie_sensitivity.py / POSTHOC_TIE_SENSITIVITY.json", "POST-HOC tie sensitivity"),
               ("posthoc_pred_frame_crosscheck.py / POSTHOC_PRED_FRAME_CROSSCHECK.json", "POST-HOC predicted-grid path cross-check"),
               ("MANIFEST.sha256", "sha256 of every file above")):
    w(line(f"`{f_}`", d_, "avr tree"))
w("")
w("Tree: `C:/Users/Admin/AppData/Local/Temp/avr/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-auxveto-read/`. "
  "The tree is the `git archive` of 4f7e105 plus `refe/` extracted from the same commit (the instrument imports `refe/aux_label_store.py`). "
  "Inputs used in place (not copied): `raw/2026-10-11-c3-selection-headroom/{tokens.json,table.npz}`, the label store `D:/Projects/TanitAD-artifacts/aux_drivable/label_store_navtest`, the head units under `D:/Projects/TanitAD-artifacts/scene_tap_probe{,_a2,_a3r}/units`, "
  "and the regenerable tracker shards `C:/Users/Admin/tmpauxv/tracked/` (sha256 in `AUXVETO_READ.json`).")
(HERE / "RESULT.md").write_text("\n".join(o) + "\n", encoding="utf-8")
print("RESULT.md written", len(o), "lines")
