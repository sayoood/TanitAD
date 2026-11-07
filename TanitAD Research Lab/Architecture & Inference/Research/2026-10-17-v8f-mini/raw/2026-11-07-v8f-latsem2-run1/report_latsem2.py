"""Generates RESULT_LATSEM2.md from RESULT_LATSEM2.json and PICTURES_LATSEM2.json: every number in the md's tables comes from the json. Run from theory/latsem2."""
import json
import numpy as np

R = json.load(open("RESULT_LATSEM2.json", encoding="utf-8"))
P = json.load(open("PICTURES_LATSEM2.json", encoding="utf-8"))
TAGS = ("M2b_s0", "M2b_s1")
KMAX = {1: 10, 2: 3, 3: 3}
KS = {1: (1, 5, 10), 2: (1, 3), 3: (1, 3)}
DEC = lambda l: "shrunk" if l == 1 else "raw"
NAMES = {"lead_present": "lead present (BA - majority)", "lead_gap": "lead gap (R2)", "lead_closing": "lead closing speed (R2)", "agents30": "agents within 30 m (R2)",
         "nearest": "nearest-agent distance (R2)", "lon7": "7-class longitudinal (BA - majority)", "turn": "turn ahead (BA - majority)",
         "lead_closing_uncapped": "beside: closing speed, no 20 m/s cap (R2)", "agents30_view": "beside: agents within 30 m inside the frustum (R2)"}
Q7 = ("lead_present", "lead_gap", "lead_closing", "agents30", "nearest", "lon7", "turn")
QB = ("lead_closing_uncapped", "agents30_view")
c = R["controls"]


def f(x, d=3):
    return f"{x['point']:+.{d}f} [{x['lo']:+.{d}f}, {x['hi']:+.{d}f}]" + ("*" if x["separated"] else "")


def pos_sep(x):
    return x["separated"] and x["point"] > 0


def both(fn):
    return all(fn(t) for t in TAGS)


valid = R["controls_pass"]
tab = R["outcomes_if_controls_ignored_NOT_DECISIONS"]
L = []
w = L.append
w("# RESULT_LATSEM2: does each world-model level encode and imagine the scene, not only ego motion?\n")
w(f"MEASURED 2026-11-07, dev-box CPU, `run_latsem2.py` run 1 ({R['wall_s']:.0f} s for the numbers incl. beside draws). **Tier T0: a world-model diagnostic, never a driving claim. The planner reads none of these predictions.**\n")
if valid:
    w("**Controls: all pass.**\n")
else:
    w(f"**READ INVALID: control(s) {', '.join(R['controls_failed'])} failed.** Per the card the read is INVALID and no outcome below is a decision. The numbers are reported as measured, flagged `uncontrolled`, with the evidence for the Master Mind to amend a control that the data shows mis-specified. Nothing was re-drawn to pass.\n")
w("* Card `PREREG_V8F_LATSEM2.md`: frozen prefix 3,769 bytes, `1325f21c`, frozen 12:57:27. `IMPL_CHOICES_LATSEM2.md` (14,840 bytes, `b17688dc`) froze 13:08:14, before any statistic; the run asserts both hashes.")
w(f"* Data: M2b s0 / s1, eval-141 restricted to the 139 episodes with a frame file (LATSEM's bank): {R['n_bank_frames']:,} frames carry the probes, {R['n_windows']:,} windows with 16 real frames carry S2, {R['n_windows_S3']} of them ({R['n_episodes_S3']} episodes) have a near lead at frame 0 and carry S3.")
w("* Estimator: episode-cluster bootstrap over the 139 episodes, B 2,000, seed 0. It answers \"another draw of EPISODES\". It does not answer \"another PROBE FIT\": the control failures below are exactly that blind spot (`H-ESTIM-SEED-1` family). The two training seeds stand in for another training run. `*` marks a separated interval.")
w("* Probes: ridge lambda 10 (continuous), logistic C 0.1 (binary, 7-class), cross-fitted by the same two episode folds; features standardised on the fit fold. 72 logistic fits, all converged. Code `run_latsem2.py`, `labels_latsem2.py`; raw numbers `RESULT_LATSEM2.json`; log `run_latsem2.run1.log`.\n")

w("## Targets and n (labels only; frozen in IMPL_CHOICES)\n")
w("| target | n frames | note |\n|---|---|---|")
notes = {"lead_present": "trainer's `lead_near` (lead within 40 m) on `lead_eval139.npz`; 28 % positive", "lead_gap": "m, clipped [0, 40]; 56 episodes", "lead_closing": "m/s over 0.4 s, abs value above 20 m/s set NaN (lead-track switches)",
         "agents30": "obstacle join, centre distance <= 30 m, all 11 classes the join carries, any azimuth", "nearest": "m, frames with at least one cuboid", "lon7": "v9 `lon_cls`, 7 classes (HOLD 95 ... KEEP 2,891)",
         "turn": "abs heading change over 4 s above 0.3 rad; 15 % positive", "lead_closing_uncapped": "beside", "agents30_view": "beside"}
for q in Q7 + QB:
    w(f"| {NAMES[q]} | {R['n'][q]:,} | {notes[q]} |")
w("\nAll seven targets are buildable from the data: the obstacle join (`b1eval_agents_3d.jsonl.xz`, 139 clips) and the lead file (`lead_eval139.npz`) both exist for eval-141's 139 episodes. The card says 10 dynamic classes; the join carries 11 class strings (adds `protruding_object`) and none was dropped.\n")

w("## Controls\n")
w("| control | committed value | observed | status |\n|---|---|---|---|")
w(f"| C1 start-state query SKILL | exactly 0.0 on every S2 row | max abs point {c['C1_start_query_skill_max_abs']}, max abs interval end {c['C1_start_query_interval_max_abs']} ({sum(1 for k in R['S2'] if k.endswith('/start'))} rows) | PASS |")
w(f"| C2 two identical action sequences, S3 | exactly 0.0 | max abs {c['C2_identical_sequences_S3_max_abs']} (6 rows, intervals [0, 0]) | PASS |")
st3 = "PASS" if not c["C3_permuted_probe_any_separated_above_0"] else "**FAIL**"
w(f"| C3 permuted-label probes (S1, {len(R['S1'])} rows) | none separated above 0 | rows separated above 0: {', '.join(c['C3_rows_separated_above_0']) or 'none'} | {st3} |")
st4 = "PASS" if not c["C4_shuffled_S3_any_separated_below_0"] else "**FAIL**"
rows4 = [k for k, v in R["S3_shuffled"]["primary"].items() if v["hi"] < 0]
w(f"| C4 label-shuffled S3 | not separated below 0 | rows separated below 0: {', '.join(rows4) or 'none'} | {st4} |")
w(f"| LATSEM's copy / level checks, re-run | z1 copies <= 1e-5, actions exactly equal, z2/z3 equal `encode` to 1e-5 | {c['copy_maxdiff_z1']:.2e}, {c['copy_maxdiff_actions']}, {max(c['level_maxdiff/M2b_s0'], c['level_maxdiff/M2b_s1']):.2e} | PASS |")
w(f"| metric self-test (synthetic) | perfect predictor BA 1.0, baseline 1/3, paired self-difference exactly 0 | {R['selftest']['perfect_bal_model']}, {R['selftest']['perfect_bal_base']:.4f}, {R['selftest']['same_as_base_point']} | PASS |")
w("\n**What the two failures show (evidence, not a decision).**")
s = R["S3_shuffled"]["extra_summary"]
w(f"* **C4.** Shuffled gap labels give a separated S3 of EITHER sign. Primary draw: {', '.join(f'{k} {f(v)}' for k, v in R['S3_shuffled']['primary'].items())}. Over 20 further draws ({s['rows']} rows) {s['separated_below_0']} are separated below 0 and {s['separated_above_0']} above 0. The reason is structural: accelerate and brake move the imagined latent along a nearly fixed direction, so ANY probe direction reads a consistent nonzero difference, and the bootstrap (episodes resampled, probe held fixed) does not see the probe's own variance. A separated interval is necessary, not sufficient. Permutation reference (beside): the real S3 point against the 20 shuffled draws, per seed and level, in the S3 table below.")
if "C3_extra_draws_beside" in R:
    ex = R["C3_extra_draws_beside"]
    kinds = {"classification (lead_present, lon7, turn)": [k for k in ex if k.split("/")[2] in ("lead_present", "lon7", "turn")], "continuous (the other 6)": [k for k in ex if k.split("/")[2] not in ("lead_present", "lon7", "turn")]}
    kc = kinds["classification (lead_present, lon7, turn)"]
    rate = sum(ex[k]["above_0"] for k in kc) / sum(ex[k]["draws"] for k in kc)
    w("* **C3.** Twenty further permutation draws per row (beside; rows x draws separated above 0): " + "; ".join(f"{nm}: {sum(ex[k]['above_0'] for k in ks)} of {sum(ex[k]['draws'] for k in ks)}" for nm, ks in kinds.items()) + f". A classification row is separated above 0 at a rate of {100 * rate:.1f} % per draw, so one draw over the {len(kc)} classification rows fails with probability {100 * (1 - (1 - rate) ** len(kc)):.0f} %: the control as committed cannot pass reliably. The primary failure is row `M2b_s0/L2/turn`, balanced accuracy minus majority " + f(R["S1"]["M2b_s0/L2/turn"]["permuted"]) + ". Balanced accuracy minus a majority baseline has a null centred on 0 (a junk classifier predicts the minority class at random), so with 18 classification rows a single separated draw is expected at some rate; R2 rows have a strongly negative junk value and never fail.")
w("")

w("## S1: what a level ENCODES (R2 for continuous, balanced accuracy minus the fit-fold majority's for binary and 7-class)\n")
w("Cell = s0 / s1, point [95 % interval]; `*` separated. ENCODES = point > 0 and separated at BOTH seeds (**bold**). Permuted-probe rows beside in the json.\n")
w("| target | L1 | L2 | L3 |\n|---|---|---|---|")
for q in Q7 + QB:
    cells = []
    for l in (1, 2, 3):
        a = " / ".join(f(R["S1"][f"{t}/L{l}/{q}"]["main"], 2) for t in TAGS)
        cells.append(f"**{a}**" if tab["ENCODES"][f"L{l}/{q}"] else a)
    w(f"| {NAMES[q]} | " + " | ".join(cells) + " |")
w("\n**SCENE-AWARE at level l** (card: ENCODES lead gap OR agent count at both seeds):\n")
w("| level | by lead gap | by agents within 30 m | verdict (uncontrolled) |\n|---|---|---|---|")
for l in (1, 2, 3):
    v = tab["SCENE-AWARE"][f"L{l}"]
    w(f"| L{l} | {'yes' if v['by_lead_gap'] else 'no'} | {'yes' if v['by_agents30'] else 'no'} | {'SCENE-AWARE' if v['verdict'] else 'not SCENE-AWARE'} |")
lp_all = all(tab["ENCODES"][f"L{l}/lead_present"] for l in (1, 2, 3))
w(f"\nBeside the card's definition: **lead present** ENCODES at {'all three levels' if lp_all else 'some levels'} at both seeds, the **7-class label** at {', '.join('L%d' % l for l in (1, 2, 3) if tab['ENCODES'][f'L{l}/lon7'])} and **turn ahead** at {', '.join('L%d' % l for l in (1, 2, 3) if tab['ENCODES'][f'L{l}/turn'])}. The card's SCENE-AWARE counts only the gap and the agent count, which are continuous targets read by ridge lambda 10; none of them is separated positive at both seeds at any level (the only separated cells are agents within 30 m at L1 s0, +0.31, and lead closing speed, which is separated NEGATIVE).\n")

w("## S2: what a level IMAGINES (SKILL of the rollout against the start state through the same probe)\n")
w("Decision query: L1 shrunk (TRAIN c_k), L2 and L3 raw. IMAGINES = SKILL > 0 separated at both seeds. **Lead gap, every level and horizon** (s0 / s1; `perm` = the permuted-label probe through the same query, beside):\n")
w("| level, k | n (s0) | decision SKILL s0 | decision SKILL s1 | IMAGINES | oracle s0 / s1 | perm s0 / s1 |\n|---|---|---|---|---|---|---|")
for l in (1, 2, 3):
    for k in KS[l]:
        nm = DEC(l)
        d = [R["S2"][f"{t}/L{l}/k{k}/lead_gap/{nm}"] for t in TAGS]
        o = [R["S2"][f"{t}/L{l}/k{k}/lead_gap/oracle"] for t in TAGS]
        p = [R["S2_permuted"][f"{t}/L{l}/k{k}/lead_gap/{nm}"] for t in TAGS]
        mark = " (longest)" if k == KMAX[l] else ""
        w(f"| L{l} k{k}{mark} | {d[0]['n']} | {f(d[0], 2)} | {f(d[1], 2)} | {'**yes**' if tab['IMAGINES'][f'L{l}/k{k}/lead_gap'] else 'no'} | {f(o[0], 2)} / {f(o[1], 2)} | {f(p[0], 2)} / {f(p[1], 2)} |")
w("\n**All targets at each level's longest horizon** (decision query; s0 / s1 point, `*` separated, `P` = a permuted-label probe is separated above 0 at that cell at either seed; **bold** = IMAGINES):\n")
w("| target | L1 k10 | L2 k3 | L3 k3 |\n|---|---|---|---|")
for q in Q7 + QB:
    cells = []
    for l in (1, 2, 3):
        k = KMAX[l]; nm = DEC(l)
        a = " / ".join(f"{R['S2'][f'{t}/L{l}/k{k}/{q}/{nm}']['point']:+.3f}{'*' if R['S2'][f'{t}/L{l}/k{k}/{q}/{nm}']['separated'] else ''}" for t in TAGS)
        pf = any(R["S2_permuted"][f"{t}/L{l}/k{k}/{q}/{nm}"]["lo"] > 0 for t in TAGS)
        a += " P" if pf else ""
        cells.append(f"**{a}**" if tab["IMAGINES"][f"L{l}/k{k}/{q}"] else a)
    w(f"| {NAMES[q]} | " + " | ".join(cells) + " |")
im = [k for k, v in tab["IMAGINES"].items() if v]
imc = [k for k in im if k.split("/")[2] in Q7]
imb = [k for k in im if k.split("/")[2] in QB]
pflag = lambda cell: any(R["S2_permuted"][f"{t}/{cell}/{DEC(int(cell[1]))}"]["lo"] > 0 for t in TAGS)       # cell = "L2/k3/lead_closing"
nperm = sum(1 for k, v in R["S2_permuted"].items() if v["lo"] > 0)
w(f"\nIMAGINES holds at both seeds for {len(imc)} of the card's {sum(len(KS[l]) for l in KS) * len(Q7)} (level, k, target) cells: {', '.join(imc) or 'none'}; {sum(pflag(k) for k in imc)} of them also read a separated permuted-label probe at one seed. Beside targets: {', '.join(imb) or 'none'}; {sum(pflag(k) for k in imb)} of {len(imb)} carry the same flag. "
  f"**SCENE-IMAGINING (lead gap at the longest horizon): " + ", ".join(f"L{l} {'yes' if tab['SCENE-IMAGINING'][f'L{l}'] else 'no'}" for l in (1, 2, 3)) + ".** "
  f"The permuted-label probe is itself separated above 0 on {nperm} of {len(R['S2_permuted'])} S2 cells: a probe fitted on junk reads a positive SKILL when the rollout state sits nearer the label mean than the start state does (a variance effect), so a positive SKILL on a target the probe cannot encode (closing speed has S1 R2 below 0 at every level) is not foresight. The oracle query (the true future latent) is separated above 0 on the lead gap at no cell, so this instrument has no headroom on the gap.\n")

w("## S3: does the imagined lead gap follow the action? (windows with a near lead at frame 0)\n")
w("Imagined gap under accelerate (+1.5) minus brake (-3.0) at the longest horizon, m. The card's ACTION-SEMANTIC-LEAD needs this **below 0** (accelerating closes the gap). Beside: the 20 shuffled-label draws.\n")
w("| seed, level | accel - brake (decision) | identical sequences | decoded mean gap accel / brake | shuffled draws (20): point min .. max | shuffled primary | ACTION-SEMANTIC-LEAD |\n|---|---|---|---|---|---|---|")
for t in TAGS:
    for l in (1, 2, 3):
        v = R["S3"][f"{t}/L{l}"]
        pts = [R["S3_shuffled"]["extra_draws_beside"][str(d)][f"{t}/L{l}"]["point"] for d in range(1, 21)]
        w(f"| {t[-2:]} L{l} | {f(v['decision'], 2)} | {v['identical_sequences']['point']:+.1f} | {v['decoded_mean']['accel']:.1f} / {v['decoded_mean']['brake']:.1f} | {min(pts):+.2f} .. {max(pts):+.2f} | {f(R['S3_shuffled']['primary'][f'{t}/L{l}'], 2)} | {'yes' if tab['ACTION-SEMANTIC-LEAD'][f'L{l}'] else 'no'} |")
w("\nThe decoded gap is LARGER under accelerate than under brake, at every level and both seeds, by 2.4 to 7.6 m: the wrong direction for the card, and larger than the largest shuffled draw (|point| at most 1.6 m) in all six rows. Read with LATSEM (CF-LON: accelerate raises the imagined SPEED) and the label relation below, the lead-gap probe most likely reads a speed correlate, not the physical constraint (HYPOTHESIS: not tested by an intervention here). L1 raw (unshrunk) is beside in the json and has the same sign (+5.2 / +6.4 m).\n")
G = json.load(open("BESIDE_GAP_SPEED.json", encoding="utf-8"))
w(f"Beside (labels only, `beside_gap_speed.py`): over the {G['n']:,} bank frames with a near lead, the lead gap and the ego speed correlate at r = {G['pearson_gap_speed']:+.2f} (folds {G['by_fold_pearson'][0]:+.2f} / {G['by_fold_pearson'][1]:+.2f}), slope {G['ols_slope_m_per_ms']:.2f} m per m/s; the median gap is {G['gap_median_by_speed_tercile_m'][0]:.1f} / {G['gap_median_by_speed_tercile_m'][1]:.1f} / {G['gap_median_by_speed_tercile_m'][2]:.1f} m in the slow / middle / fast speed tercile. A latent that encodes speed (S1 and LATSEM R1) gives a gap probe the sign of this correlation, which is the sign S3 reads.\n")

w("## Outcomes per level (as the card defines them; uncontrolled, NOT decisions while controls 3 and 4 stand failed)\n")
w("| level | SCENE-AWARE | SCENE-IMAGINING | ACTION-SEMANTIC-LEAD |\n|---|---|---|---|")
for l in (1, 2, 3):
    w(f"| L{l} | {'yes' if tab['SCENE-AWARE'][f'L{l}']['verdict'] else 'no'} | {'yes' if tab['SCENE-IMAGINING'][f'L{l}'] else 'no'} | {'yes' if tab['ACTION-SEMANTIC-LEAD'][f'L{l}'] else 'no (wrong sign)'} |")
w(f"\nReported outcome of the read: **{R['outcomes']['all']}**. The table is the card's definitions applied to uncontrolled numbers." if not valid else "")
w("\n## What this does not say\n")
w("* Not a driving claim (T0). The planner reads none of these latents.\n* Not that the scene is absent from the levels: the card's probe is a 128-d linear map with ridge lambda 10 fitted on 56 episodes' lead frames; the continuous gap and count targets are read at R2 near or below 0, while the binary and class targets are read with positive balanced accuracy.\n* Each level's state is a pointwise function of ONE frame (w = 1), so closing speed and any rate are out of reach by construction (all six closing-speed rows are separated NEGATIVE, worse than the mean).\n* A rollout that moves toward the label mean scores positive SKILL with a junk probe (the S2 permuted rows), so SKILL > 0 is not foresight by itself.\n* MRATE's W0 / W1 do not exist yet (`theory/mrate/` is an empty tree): no beside rows on them.\n")
w("## Next lever (the card's rule, then the evidence)\n")
if valid:
    w(f"* {R['next_lever_per_card']}")
else:
    w("* The read is INVALID, so the card's rule does not fire. If the Master Mind amends controls 3 and 4 (a permutation reference on the point estimate instead of a not-separated interval, and R draws with a rate bar instead of one draw), the uncontrolled tables above give: no level SCENE-AWARE by the card's definition, so the card's next lever is MRATE's W1 and then a scene-target auxiliary (own card); S2 would need a reference that removes the variance effect before any SCENE-IMAGINING reading; S3 reads the wrong sign at all levels. An amendment written after these numbers were seen is post hoc and should say so; run 2 would recompute everything from the same inputs.")
w("* The positive finding to carry: presence of a lead, the manoeuvre class and a turn ahead are linearly decodable from every level (balanced accuracy 0.08 to 0.24 above the majority baseline) while their magnitudes and rates are not. A scene-target auxiliary should therefore ask for the gap and the agent count, which the frozen levels do not hold.\n")
w("## Pictures (decide nothing; seed s0)\n")
w("| file | sha256 | content |\n|---|---|---|")
for fn, sh in P["files"].items():
    w(f"| `fig/{fn}` | `{sh}` | " + ("per-level PCA maps coloured by lead gap and by agents within 30 m; no camera frame, safe to land" if fn.startswith("pca") else f"retrieval strip, window {P['pictures'][fn[6:-4]]['window']} (lead gap {P['pictures'][fn[6:-4]]['lead_gap_frame0_m']:.1f} m at frame 0); embeds downscaled camera frames: local only, not for the public repo") + " |")
w("\nStrips show, per level, the frame now, the real future frame, and the nearest real frame (from ANOTHER episode) to the imagined state under accelerate and under brake, each titled with that frame's own lead gap and agent count. CLOSE is the S3 window with the smallest frame-0 gap: 0.0 m, a clipped negative gap with no lead visible in the frame (3 of the 2,489 gap labels are 0.0, 5 are at most 1 m: a label-quality flag at the extreme, not a finding); FAR is the one nearest the median gap.\n")
open("RESULT_LATSEM2.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote RESULT_LATSEM2.md", len(L), "lines")
