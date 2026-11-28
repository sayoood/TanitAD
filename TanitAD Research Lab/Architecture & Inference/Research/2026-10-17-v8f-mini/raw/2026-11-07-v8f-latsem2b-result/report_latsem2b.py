"""Generates RESULT_LATSEM2B.md from RESULT_LATSEM2B.json (every number in a table comes from the json) and the run 1 json for the labelled 'beside' rows.
Run from theory/latsem2b:  python report_latsem2b.py"""
import json, sys
sys.dont_write_bytecode = True

R = json.load(open("RESULT_LATSEM2B.json", encoding="utf-8"))
R1 = json.load(open("../latsem2/RESULT_LATSEM2.json", encoding="utf-8"))
TAGS = ("M2b_s0", "M2b_s1")
KMAX = {1: 10, 2: 3, 3: 3}
KS = {1: (1, 5, 10), 2: (1, 3), 3: (1, 3)}
HS = {(1, 1): "0.2 s", (1, 5): "1.0 s", (1, 10): "2.0 s", (2, 1): "0.4 s", (2, 3): "1.2 s", (3, 1): "0.8 s", (3, 3): "2.4 s"}
DEC = lambda l: "shrunk" if l == 1 else "raw"
ALLQ = ("lead_present", "lead_gap", "lead_closing", "agents30", "nearest", "lon7", "turn", "lead_closing_uncapped", "agents30_view")
NAMES = {"lead_present": "lead present (BA - majority)", "lead_gap": "lead gap (R2)", "lead_closing": "lead closing speed (R2)", "agents30": "agents within 30 m (R2)",
         "nearest": "nearest-agent distance (R2)", "lon7": "7-class longitudinal (BA - majority)", "turn": "turn ahead (BA - majority)",
         "lead_closing_uncapped": "beside: closing speed, no 20 m/s cap (R2)", "agents30_view": "beside: agents within 30 m in the frustum (R2)"}
D = R["datasets"]
TR, EV = D["train"], D["eval141"]
valid = R["controls_pass"]
L = []
w = L.append


def fm(x, d=3):
    return f"{x['point']:+.{d}f} [{x['lo']:+.{d}f}, {x['hi']:+.{d}f}]"


def flag(boot, two):
    return "†" if two else ("*" if boot else "")


def cell1(a, d=3):
    return f"{fm(a['main'], d)}{flag(a['separated_bootstrap_only'], a['separated_two_part'])} thr {a['ref']['p97.5']:+.{d}f}, p {a['rank_p']:.3f}"


def cell2(a, d=3):
    return f"{fm(a['row'], d)}{flag(a['separated_bootstrap_only'], a['separated_two_part'])} thr {a['ref']['p97.5']:+.{d}f}, p {a['rank_p']:.3f}"


def yn(b):
    return "yes" if b else "no"


w("# RESULT_LATSEM2B: does each world-model level encode and imagine the scene, read with a permutation-calibrated instrument?\n")
w(f"MEASURED 2026-11-07, dev-box CPU, `run_latsem2b.py` ({R['wall_s'] / 60:.0f} min wall, 4 threads). **Tier T0: a world-model diagnostic, never a driving claim. The planner reads none of these predictions.**\n")
if valid:
    w("**Controls: all pass on both datasets. The outcomes below are decisions of the card.**\n")
else:
    w(f"**READ INVALID: control(s) {', '.join(R['controls_failed'])} failed.** Per the card the read is INVALID and no outcome below is a decision. The numbers are reported as measured and flagged `uncontrolled`. Nothing was re-drawn or tuned to pass.\n")
w("* Card `PREREG_V8F_LATSEM2B.md` (frozen prefix 4,611 bytes, `bf6b3717`), `AMENDMENT_LATSEM2B_1.md` (prefix 2,778 bytes, `120072ff`), `IMPL_CHOICES_LATSEM2B.md` (16,005 bytes, `a90cd9ca`, frozen 13:44:00 before any statistic). The run asserts all three hashes.")
w(f"* **Deciding set: TRAIN** (WMACT's 141 TRAIN clips, M2b's own training clips, so optimistic for S2): {TR['n_bank_frames']:,} bank frames, {TR['n_episodes']} episodes ({TR['n_episodes_by_fold'][0]} / {TR['n_episodes_by_fold'][1]} by fold), {TR['n_windows']:,} windows with 16 real frames, {TR['n_windows_S3']} S3 windows ({TR['n_episodes_S3']} episodes). "
  f"**Beside, post hoc: eval-141** (LATSEM2's bank): {EV['n_bank_frames']:,} frames, {EV['n_episodes']} episodes, {EV['n_windows']:,} windows, {EV['n_windows_S3']} S3 windows ({EV['n_episodes_S3']} episodes).")
w(f"* Estimators: the episode-cluster bootstrap (B 2,000, seed 0) answers \"another draw of EPISODES\"; the two model seeds stand in for \"another training run\"; the permutation reference (R = {TR['analysis']['R']} draws per cell, plus {TR['analysis']['n_cal']} calibration draws) answers \"would this appear with no relation between latent and label?\". A row is separated (`†`) only if its interval excludes 0 on the claimed side AND its point is beyond the cell's 97.5th (S3-PARTIAL: 2.5th) reference percentile. `*` marks a row whose interval alone excludes 0 on the claimed side (the first part only). `thr` is the cell's 97.5th percentile, `p` the reference rank of the point.")
w("* Probes: ridge lambda 10 (continuous), logistic C 0.1 (binary, 7-class), cross-fitted over two episode folds, features standardised on the fit fold: run 1's, unchanged. Code `run_latsem2b.py`; raw numbers `RESULT_LATSEM2B.json`; reference draws `draws_train.jsonl`, `draws_eval141.jsonl`; log `run_latsem2b.run1.log`.\n")

# ------------------------------------------------------------------------------------------------ controls
w("## Controls (first; a failing control makes the read INVALID)\n")
w("| control | committed value | TRAIN observed | eval-141 observed | status |\n|---|---|---|---|---|")
ct, ce = TR["controls_raw"], EV["controls_raw"]
cs = R["controls"]
st = lambda k: "PASS" if (cs["train"][k] and cs["eval141"][k]) else "**FAIL** (" + ", ".join(d for d in ("train", "eval141") if not cs[d][k]) + ")"
w(f"| copy and per-level checks (not controls) | copies agree to 1e-5; per-frame z2 / z3 equal `encode` to 1e-5 | z1 {ct['copy_maxdiff_z1']:.1e}, actions {ct['copy_maxdiff_actions']}, levels {max(ct['level_maxdiff'].values()):.1e} | z1 {ce['copy_maxdiff_z1']:.1e}, actions {ce['copy_maxdiff_actions']}, levels {max(ce['level_maxdiff'].values()):.1e} | {st('copy_levels')} |")
w(f"| C1 start-state query SKILL | exactly 0.0 on every S2 row | max abs {ct['C1_start_query_skill_max_abs']}, interval {ct['C1_start_query_interval_max_abs']} ({ct['C1_start_rows']} rows) | max abs {ce['C1_start_query_skill_max_abs']}, interval {ce['C1_start_query_interval_max_abs']} ({ce['C1_start_rows']} rows) | {st('C1')} |")
w(f"| C2 identical action sequences, S3 and S3-PARTIAL | exactly 0.0, interval [0, 0] | max abs {ct['C2_identical_max_abs']} | max abs {ce['C2_identical_max_abs']} | {st('C2')} |")
w(f"| C4 speed removed | abs corr(residual gap, ego speed) <= 1e-6 on every fit fold | max abs {ct['C4_speed_corr_max_abs']:.1e} (permuted draws: {TR['analysis']['speed_control_permuted_max_abs_corr']:.1e}) | max abs {ce['C4_speed_corr_max_abs']:.1e} (permuted draws: {EV['analysis']['speed_control_permuted_max_abs_corr']:.1e}) | {st('C4')} |")
w(f"| C3 calibration | share declared separated <= 5 % in each of four families | see next table | see next table | {st('C3')} |\n")
w("**C3 in detail** (20 calibration draws, seeds 1001 to 1020, each run through the full two-part rule as if real; rows pool both model seeds, three levels and all 9 targets):\n")
w("| family | dataset | rows | declared by the two-part rule | share | share by the interval alone (first part only) | <= 5 % |\n|---|---|---|---|---|---|---|")
for fam in ("S1_classification", "S1_continuous", "S2", "S3_PARTIAL"):
    for dsn, d in (("TRAIN", TR), ("eval-141", EV)):
        c = d["analysis"]["calibration"][fam]
        w(f"| {fam} | {dsn} | {c['rows']:,} | {c['two_part']} | {100 * c['share_two_part']:.2f} % | {100 * c['share_bootstrap_only']:.2f} % ({c['bootstrap_only']}) | {yn(c['pass'])} |")
w("")
w(f"Instrument notes: logistic fits in the real rows TRAIN {TR['logistic_real']['fits']} (not converged {TR['logistic_real']['not_converged']}), eval-141 {EV['logistic_real']['fits']} (not converged {EV['logistic_real']['not_converged']}); in the draws TRAIN {TR['analysis']['logistic_fits_in_draws']['fits']:,} (not converged {TR['analysis']['logistic_fits_in_draws']['not_converged']}), eval-141 {EV['analysis']['logistic_fits_in_draws']['fits']:,} (not converged {EV['analysis']['logistic_fits_in_draws']['not_converged']}). "
  f"Distinct permuted label vectors per cell over the {TR['analysis']['distinct_label_assignments']['draws']} draws (model seed s0): minimum {TR['analysis']['distinct_label_assignments']['min_distinct']} on TRAIN and {EV['analysis']['distinct_label_assignments']['min_distinct']} on eval-141 (a diagnostic; targets with few labelled episodes can repeat a vector). "
  f"Reference points that were NaN and dropped: TRAIN {TR['analysis']['nan_points_dropped_from_reference']}, eval-141 {EV['analysis']['nan_points_dropped_from_reference']}. Balanced-accuracy self-test: perfect {TR['selftest']['perfect_bal_model']}, baseline {TR['selftest']['perfect_bal_base']:.4f}, same-as-base {TR['selftest']['same_as_base_point']}.")
rp = EV["reproduction_vs_run1_max_abs_diff"]
_KIND = {"lead_present": "bin", "lead_gap": "cont", "lead_closing": "cont", "agents30": "cont", "nearest": "cont", "lon7": "cls", "turn": "bin", "lead_closing_uncapped": "cont", "agents30_view": "cont"}
_bk = {}
for _k, _v in EV["real"]["S1"].items():
    _kd = ("S1", _KIND[_k.split("/")[2]])
    _bk[_kd] = max(_bk.get(_kd, 0.0), abs(_v["point"] - R1["S1"][_k]["main"]["point"]))
for _k, _v in EV["real"]["S2"].items():
    _t, _l, _kk, _q, _nm = _k.split("/")
    if _nm == DEC(int(_l[1:])):
        _kd = ("S2", _KIND[_q])
        _bk[_kd] = max(_bk.get(_kd, 0.0), abs(_v["point"] - R1["S2"][_k]["point"]))
w(f"Reproduction check (not a control): the real eval-141 rows of this run against run 1's `RESULT_LATSEM2.json` (same probes, same labels). Max abs difference: S1 points {rp['S1_point']:.1e}, S1 interval ends {rp['S1_ends']:.1e}, S2 decision points {rp['S2_decision_point']:.1e}, raw S3 points {rp['S3_raw_point']:.1e}. "
  f"By probe kind: S1 binary {_bk[('S1', 'bin')]:.1e}, continuous {_bk[('S1', 'cont')]:.1e}, 7-class {_bk[('S1', 'cls')]:.1e}; S2 binary {_bk[('S2', 'bin')]:.1e}, continuous {_bk[('S2', 'cont')]:.1e}, 7-class {_bk[('S2', 'cls')]:.1e}. "
  "Binary and continuous rows reproduce; the 7-class multinomial rows differ at the 1e-4 to 1e-3 level, most likely the lbfgs stopping point (tol 1e-4, max_iter 5000, no non-convergence) moving with the BLAS thread layout (not tested here). "
  "The 7-class rows are not outcome targets of the card (SCENE-AWARE reads lead gap and agents within 30 m; SCENE-IMAGINING reads lead gap).\n")

# ------------------------------------------------------------------------------------------------ outcomes
T = R["outcomes_if_controls_ignored_NOT_DECISIONS"]
w("## Outcomes (the card's definitions, per level; deciding set first, eval-141 beside)\n")
if not valid:
    w("*The read is INVALID, so the table below is `uncontrolled` and decides nothing.*\n")
w("| level | SCENE-AWARE | by lead gap: TRAIN ENCODES both seeds / eval-141 points above 0 | by agents within 30 m: TRAIN / eval-141 | SCENE-IMAGINING (lead gap, longest horizon): TRAIN IMAGINES / eval-141 above 0 | ACTION-SEMANTIC-LEAD: TRAIN S3-PARTIAL separated below 0 / eval-141 below 0 |\n|---|---|---|---|---|---|")
for l in (1, 2, 3):
    sa, si, sl = T["SCENE-AWARE"][f"L{l}"], T["SCENE-IMAGINING"][f"L{l}"], T["ACTION-SEMANTIC-LEAD"][f"L{l}"]
    g, a = sa["by"]["lead_gap"], sa["by"]["agents30"]
    w(f"| L{l} | **{yn(sa['verdict'])}** | {yn(g['encodes_train'])} / {yn(g['eval141_points_above_0'])} | {yn(a['encodes_train'])} / {yn(a['eval141_points_above_0'])} | **{yn(si['verdict'])}**: {yn(si['imagines_train'])} / {yn(si['eval141_points_above_0'])} | **{yn(sl['verdict'])}**: {yn(sl['separated_train'])} / {yn(sl['eval141_points_below_0'])} |")
w("")
w(f"* **Next lever (card):** {R['next_lever_per_card']}.")
enc_tr = sorted(k for k, v in T["ENCODES"]["train"].items() if v)
imag_tr = sorted(k for k, v in T["IMAGINES"]["train"].items() if v)
w(f"* ENCODES on TRAIN at both seeds (two-part rule), all targets: {', '.join(enc_tr) if enc_tr else 'none'}.")
w(f"* IMAGINES on TRAIN at both seeds (two-part rule), all targets and horizons: {', '.join(imag_tr) if imag_tr else 'none'}.")
enc_ev = sorted(k for k, v in T["ENCODES"]["eval141"].items() if v)
imag_ev = sorted(k for k, v in T["IMAGINES"]["eval141"].items() if v)
w(f"* Beside, on eval-141 (post hoc, its own reference): ENCODES at both seeds: {', '.join(enc_ev) if enc_ev else 'none'}; IMAGINES at both seeds: {', '.join(imag_ev) if imag_ev else 'none'}.")
for l in (1, 2, 3):
    sa = T["SCENE-AWARE"][f"L{l}"]
    for q, v in sa["by"].items():
        if v["holds"]:
            parts = []
            for t in TAGS:
                a, b = TR["analysis"]["S1"][f"{t}/L{l}/{q}"], EV["analysis"]["S1"][f"{t}/L{l}/{q}"]
                parts.append(f"{t[-2:]}: TRAIN {fm(a['main'])}{flag(a['separated_bootstrap_only'], a['separated_two_part'])}, eval-141 {fm(b['main'])}{flag(b['separated_bootstrap_only'], b['separated_two_part'])}")
            w(f"* SCENE-AWARE at L{l} holds through `{q}`: " + "; ".join(parts) + ". The card asks for the same sign on eval-141 (point above 0 at both seeds); eval-141's own two-part separation is shown here and decides nothing.")
_thr = [a["ref"]["p97.5"] for d in (TR, EV) for k, a in d["analysis"]["S1"].items() if _KIND[k.split("/")[2]] == "cont"]
w(f"* Instrument note: for the continuous S1 targets (R2) the 97.5th reference percentile is negative (range {min(_thr):+.2f} to {max(_thr):+.2f} over {len(_thr)} cells), because a probe fitted on permuted labels scores a negative R2 on the other fold. The reference therefore adds little to the interval there, and calibration reads 0.00 % for that family with or without it; the 7-class, binary and S2 rows are where the reference binds (S2: the interval alone declares {100 * TR['analysis']['calibration']['S2']['share_bootstrap_only']:.0f} % of calibration rows, the two-part rule {100 * TR['analysis']['calibration']['S2']['share_two_part']:.1f} % on TRAIN).\n")

# ------------------------------------------------------------------------------------------------ S1
w("## S1: what each level ENCODES (R2 for continuous targets, balanced accuracy minus the majority baseline for classes)\n")
for dsn, d, title in (("train", TR, "TRAIN (deciding)"), ("eval141", EV, "eval-141 (beside, post hoc)")):
    w(f"### {title}\n")
    w("| target | n frames | level | seed s0: point [95 % interval] | seed s1: point [95 % interval] |\n|---|---|---|---|---|")
    for q in ALLQ:
        for l in (1, 2, 3):
            cells = [cell1(d["analysis"]["S1"][f"{t}/L{l}/{q}"]) for t in TAGS]
            w(f"| {NAMES[q]} | {d['n'][q]:,} | L{l} | {cells[0]} | {cells[1]} |")
    w("")

# ------------------------------------------------------------------------------------------------ S2
w("## S2: what each level IMAGINES (SKILL = 1 - error of the rolled-out state / error of the start state, true actions; L1 shrunk, L2 and L3 raw)\n")
w("### Lead gap at every horizon\n")
w("| dataset | level, horizon | seed s0: SKILL [95 % interval] | seed s1: SKILL [95 % interval] |\n|---|---|---|---|")
for dsn, d, nm in (("train", TR, "TRAIN"), ("eval141", EV, "eval-141")):
    for l in (1, 2, 3):
        for k in KS[l]:
            cells = [cell2(d["analysis"]["S2"][f"{t}/L{l}/k{k}/lead_gap/{DEC(l)}"]) for t in TAGS]
            w(f"| {nm} | L{l} k{k} ({HS[(l, k)]}) | {cells[0]} | {cells[1]} |")
w("")
w("### Every target at the longest horizon of each level\n")
for dsn, d, title in (("train", TR, "TRAIN (deciding)"), ("eval141", EV, "eval-141 (beside, post hoc)")):
    w(f"#### {title}\n")
    w("| target | level, horizon | seed s0 | seed s1 |\n|---|---|---|---|")
    for q in ALLQ:
        for l in (1, 2, 3):
            k = KMAX[l]
            cells = [cell2(d["analysis"]["S2"][f"{t}/L{l}/k{k}/{q}/{DEC(l)}"]) for t in TAGS]
            w(f"| {NAMES[q]} | L{l} k{k} | {cells[0]} | {cells[1]} |")
    w("")

# ------------------------------------------------------------------------------------------------ S3
w("## S3-PARTIAL and raw S3 (does the imagined lead gap follow the action; accelerate minus brake, decoded, windows with a lead)\n")
w("S3-PARTIAL decodes the lead gap after a linear regression on ego speed; claimed side is below 0 (accelerating should shrink the gap). Raw S3 is run 1's quantity (gap probe on the unresidualised gap), reported beside with the same frame-permuted reference.\n")
w("| dataset | level | seed | S3-PARTIAL [95 % interval] | 2.5th pct of reference, rank p | separated | raw S3 [95 % interval] | 2.5th pct of reference, rank p | identical sequences (partial / raw) |\n|---|---|---|---|---|---|---|---|---|")
for dsn, d, nm in (("train", TR, "TRAIN"), ("eval141", EV, "eval-141")):
    for l in (1, 2, 3):
        for t in TAGS:
            a = d["analysis"]["S3"][f"{t}/L{l}"]; rr = d["real"]["S3"][f"{t}/L{l}"]
            w(f"| {nm} | L{l} | {t[-2:]} | {fm(a['partial'], 2)} m | {a['ref_partial']['p2.5']:+.2f}, {a['rank_p_partial']:.3f} | {'†' if a['partial_separated_two_part'] else ('*' if a['partial_separated_bootstrap_only'] else 'no')} | {fm(a['raw'], 2)} m | {a['ref_raw']['p2.5']:+.2f}, {a['rank_p_raw']:.3f} | {rr['partial_identical']['point']} / {rr['raw_identical']['point']} |")
w("")
w("Decoded mean gap under accelerate / brake (m, S3 windows; raw probe | residual probe):\n")
w("| dataset | level | seed | raw probe: accelerate / brake | residual probe: accelerate / brake |\n|---|---|---|---|---|")
for dsn, d, nm in (("train", TR, "TRAIN"), ("eval141", EV, "eval-141")):
    for l in (1, 2, 3):
        for t in TAGS:
            dm = d["real"]["S3"][f"{t}/L{l}"]["decoded_mean"]
            w(f"| {nm} | L{l} | {t[-2:]} | {dm['raw']['accel']:+.2f} / {dm['raw']['brake']:+.2f} | {dm['partial']['accel']:+.2f} / {dm['partial']['brake']:+.2f} |")
w("")

# ------------------------------------------------------------------------------------------------ run 1 beside
w("## Run 1 beside (LATSEM2 `RESULT_LATSEM2.json`: eval-141, bootstrap interval alone, INVALID read; shown for comparison, decides nothing)\n")
w("| quantity | seed | run 1 (eval-141, one-part rule) | this run, eval-141 (reproduced point, reference thr 97.5 / rank p) |\n|---|---|---|---|")
for q in ("lead_gap", "agents30"):
    for l in (1, 2, 3):
        for t in TAGS:
            a = EV["analysis"]["S1"][f"{t}/L{l}/{q}"]; r1 = R1["S1"][f"{t}/L{l}/{q}"]["main"]
            w(f"| S1 {q} L{l} R2 | {t[-2:]} | {fm(r1)}{'*' if r1['separated'] else ''} | {fm(a['main'])}, thr {a['ref']['p97.5']:+.3f}, p {a['rank_p']:.3f} |")
for l in (1, 2, 3):
    for t in TAGS:
        a = EV["analysis"]["S2"][f"{t}/L{l}/k{KMAX[l]}/lead_gap/{DEC(l)}"]; r1 = R1["S2"][f"{t}/L{l}/k{KMAX[l]}/lead_gap/{DEC(l)}"]
        w(f"| S2 lead gap L{l} k{KMAX[l]} SKILL | {t[-2:]} | {fm(r1)}{'*' if r1['separated'] else ''} | {fm(a['row'])}, thr {a['ref']['p97.5']:+.3f}, p {a['rank_p']:.3f} |")
for l in (1, 2, 3):
    for t in TAGS:
        a = EV["analysis"]["S3"][f"{t}/L{l}"]; r1 = R1["S3"][f"{t}/L{l}"]["decision"]
        w(f"| S3 raw, accelerate minus brake L{l} (m) | {t[-2:]} | {fm(r1, 2)}{'*' if r1['separated'] else ''} | {fm(a['raw'], 2)}, 2.5th pct {a['ref_raw']['p2.5']:+.2f}, p {a['rank_p_raw']:.3f}; S3-PARTIAL {fm(a['partial'], 2)} |")
w("")

# ------------------------------------------------------------------------------------------------ reading
w("## What the read does not say\n")
w("* **It is a probe of the latent, not of the planner.** T0, a world-model diagnostic: nothing here shows what the driving stack uses. The Master Mind's next lever follows the card, not this paragraph.")
w("* **The deciding set is M2b's own training clips.** TRAIN skill and encoding are optimistic for what the model learned on those clips; the eval-141 rows are the held-out view, post hoc, and the card asks for the same sign there. The L1 shrink factor was fitted on these TRAIN windows (in-sample on the deciding set).")
w("* **A permutation reference removes the probe-direction effect, not every confound.** It tests the relation between latent and label across episodes; it cannot say which scene variable a probe reads when two correlate (lead gap and ego speed correlate at r +0.45 on TRAIN, +0.52 on eval-141; S3-PARTIAL removes the linear speed term, nothing nonlinear).")
w("* **The two model seeds are two training runs of one recipe.** Agreement at both seeds is the committed bar; it is not a population estimate over recipes.")
w("* **Label coverage differs by target.** Lead-gap rows rest on 59 TRAIN episodes (29 / 30 in the folds); the permutation reference is bounded by that, and 7 TRAIN clips carry no label but `turn`. n is in each table.")
w("* **The label alignment on TRAIN peaked at a one-raw-frame offset** (corridor distance against lead gap, r 0.682 at -1 against 0.652 at 0). Nothing was shifted; `agents30` is insensitive to a 0.1 s offset, the lead gap is not exempt.")
w("* **An INVALID read decides nothing**, and a PASS decides only the card's outcomes; neither says the model drives well or badly.\n")

# ------------------------------------------------------------------------------------------------ pictures and files
w("## Pictures (no camera frame inside; decide nothing)\n")
w("Per-level PCA maps of the bank frames (model seed s0), coloured by lead gap (grey: no lead within 40 m or unlabelled).\n")
w("| file | sha256 |\n|---|---|")
for p, h in R["pictures"].items():
    w(f"| `{p}` | `{h}` |")
w("")
w("## Files (`theory/latsem2b/`, unstaged on E:)\n")
w(f"`IMPL_CHOICES_LATSEM2B.md` (+ `.sha256`), `run_latsem2b.py` (sha256 `{R['run_script_sha256'][:12]}`), `report_latsem2b.py`, `test_synth_latsem2b.py` (synthetic smoke test of the code paths), `RESULT_LATSEM2B.json`, `RESULT_LATSEM2B.md`, `draws_train.jsonl`, `draws_eval141.jsonl`, `run_latsem2b.run1.log`, `fig/`, labels `labels_latsem2b.py`, `labels_latsem2b_train.*`.")
open("RESULT_LATSEM2B.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote RESULT_LATSEM2B.md", len("\n".join(L)))
