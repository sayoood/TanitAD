"""Writes PAIRFAM_MIXO.md from the banked run's json, so that every number in the write-up is read from `pairfam_mixo_s0.json` and nothing is retyped.

    python make_pairfam_mixo_md.py <repo or export root> <pairfam_mixo_s0.json> <mutate_pairfam.out> <out.md>

The prose that interprets the numbers is fixed text; where it states a count, the count is computed here. Section 1 asserts the point differences against the unpaired ones that
RESULT_V8F_MIXO.md section 5.3 / 5.4 print (typed below, from that file), so a disagreement stops the write-up instead of being smoothed over.
"""
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT, JS, MUT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
spec = importlib.util.spec_from_file_location("pf", ROOT / "stack" / "scripts" / "v8f_paired_families.py")
sys.path[:0] = [str(ROOT / "stack"), str(ROOT / "taniteval")]
pf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pf)
R = json.loads(JS.read_text(encoding="utf-8"))
T, S = R["tactical"], R["strategic"]
cells = {p: c for p, c in pf.iter_cells({"tactical": T, "strategic": S})}

# RESULT_V8F_MIXO.md section 5.3 / 5.4: the unpaired point differences (X - B) the paired read must agree with, to 1.5e-4 (the paired delta rounds the unrounded difference)
RESULT_POINT_DIFF = {
    "tactical.plan_derived.lateral_decision.kappa": +0.0033, "tactical.plan_derived.lateral_decision.accuracy": -0.0017,
    "tactical.plan_derived.longitudinal_decision.kappa": -0.0168, "tactical.plan_derived.longitudinal_decision.accuracy": -0.0129,
    "tactical.plan_derived.maneuver_5way_collapsed.kappa": -0.0057, "tactical.goal_setting.goal_point_error_m": +0.1496,
    "tactical.goal_setting.goal_bearing_mae_deg": +0.0133, "tactical.heads.lateral.macro_f1": +0.0112, "tactical.heads.longitudinal.macro_f1": +0.0024,
    "tactical.heads.goal.macro_f1": +0.0095, "strategic.position_error_m.30": -0.0090, "strategic.position_error_m.50": +0.0246, "strategic.position_error_m.80": +0.1623,
    "strategic.heading_error_deg.30": -0.0108, "strategic.heading_error_deg.50": +0.0816, "strategic.heading_error_deg.80": -0.0685}
for k, v in RESULT_POINT_DIFF.items():
    assert abs(cells[k]["a_point"] - cells[k]["b_point"] - v) < 1.5e-4, (k, cells[k]["a_point"] - cells[k]["b_point"], v)

ok = [(p, c) for p, c in cells.items() if c["status"] == "OK"]
po = [(p, c) for p, c in cells.items() if c["status"] == "POINT_ONLY"]
un = [(p, c) for p, c in cells.items() if c["status"] == "UNAVAILABLE"]
sep = [(p, c) for p, c in ok if c["separated"]]
fam_n = {f: sum(1 for p, c in ok if p.startswith(f)) for f in ("tactical", "strategic")}
ctl = {c["name"]: c for c in R["controls"]}
ident = R["controls"][0]
repA, repB = R["controls"][1], R["controls"][2]
mut = MUT.read_text(encoding="utf-8")
m_last = re.search(r"(\d+) of (\d+) mutants RED \((\d+) on an assertion\)", mut)
m_base = re.search(r"BASELINE \(unmutated\): (\w+)\s+(.*)", mut)
m_after = re.search(r"AFTER \(unmutated, bytes identical: (\w+)\): (\w+)\s+(.*)", mut)
alig = R["alignment"]
hr = S["heading_rebuild"]["rebuild_vs_stored_ck_pos_err_max_dev_m"]
ea = json.loads((Path(R["a"]["eval_json"])).read_text(encoding="utf-8"))["raw"]["traj"]["ade6_m"]["mean"]
eb = json.loads((Path(R["b"]["eval_json"])).read_text(encoding="utf-8"))["raw"]["traj"]["ade6_m"]["mean"]

reasons = {}
for p, c in cells.items():
    if c["status"] == "OK" and "n_reason" in c and "grid windows (no mask)" not in c["n_reason"] and ".per_token." not in p:
        reasons.setdefault(c["n_reason"], []).append(p)

L = []
w = L.append
w("# PAIRFAM_MIXO - paired X - B intervals for the TACTICAL and STRATEGIC families, MIXO seed 0")
w("")
w("**Status: descriptive. No outcome table exists for these cells, so nothing here decides a bar or a next step.** It closes the gap `RESULT_V8F_MIXO.md` section 5.3 / 5.4 lists as a work item "
  "(tactical and strategic cells had only each arm's own unpaired interval), using the new tool `stack/scripts/v8f_paired_families.py`.")
w("")
w("**Tier:** " + R["tier"] + ".  **Estimator:** " + R["estimator"] + ". X - B on the same resampled episodes in every draw; every non-additive statistic (kappa, macro-F1, the goal head's 22-token macro-F1) is re-computed on each resample.")
w("")
w("**Which variance this answers.** \"Separated\" means the 95 % interval excludes 0, and it answers *another draw of episodes* only. X (`M2b-mixo_s0`) and B (`M2b_s0`) are each ONE training seed, so a separated cell would be necessary and not sufficient for a lever effect (`H-ESTIM-SEED-1`). "
  "STRATEGIC is active for v8F-mini (PI-V8F-2026-10-17; R72/R73) and is reported in full below.")
w("")
w("## 1. Result in one paragraph")
w("")
w(f"**No tactical or strategic cell is separated.** {len(ok)} cells carry a paired interval ({fam_n['tactical']} tactical, {fam_n['strategic']} strategic); {len(sep)} of them exclude 0. "
  f"A further {len(po)} goal-token F1 cells carry a point difference only (the token's F1 is undefined on some resamples) and {len(un)} are UNAVAILABLE with their reason (section 5). "
  f"The {len(RESULT_POINT_DIFF)} point differences that `RESULT_V8F_MIXO.md` section 5.3 / 5.4 print equal the paired read's (asserted by the generator of this file, to 1.5e-4); what is new is the interval. "
  f"With {len(ok)} intervals at 95 %, about {len(ok) * 0.05:.1f} would be expected to exclude 0 by chance alone; none did. That is not equivalence: the intervals are wide "
  "(for example the lateral decision kappa difference is " + f"{cells['tactical.plan_derived.lateral_decision.kappa']['delta']:+.4f} [{cells['tactical.plan_derived.lateral_decision.kappa']['lo']:+.4f}, {cells['tactical.plan_derived.lateral_decision.kappa']['hi']:+.4f}]"
  ", and the 80 m checkpoint position error difference is " + f"{cells['strategic.position_error_m.80']['delta']:+.4f} [{cells['strategic.position_error_m.80']['lo']:+.4f}, {cells['strategic.position_error_m.80']['hi']:+.4f}] m). "
  "The data do not rule out a tactical or strategic difference of that size in either direction; they show none.")
w("")
w("## 2. Inputs and controls")
w("")
w(f"* **X** = `{Path(R['a']['npz']).name}` (sha256 `{R['a']['sha256']}`), **B** = `{Path(R['b']['npz']).name}` (sha256 `{R['b']['sha256']}`), from `C:/Users/Admin/box2_ship/logs/v8f_mixo/eval/` (read only). "
  f"B's own-code ADE6 is {eb:.4f} m (the required 2.1753); X's is {ea:.4f} m (`RESULT_V8F_MIXO.md`: +0.022 m).")
w(f"* Grid and labels: `{', '.join(R['inputs_sha256'])}` from `raw/` of this package (sha256 in the json, `inputs_sha256`), the eval's own `eval_context` ({alig['n_windows']:,} windows, {alig['n_episodes']} episodes, kin mode `{R['kin_mode']}`, ages `{R['ages']}`).")
w("")
w("| control | condition | result |")
w("|---|---|---|")
w(f"| same windows | identical `eid` sequence, and equal to the grid's | **{alig['eid_identical']}** ({alig['n_windows']:,} windows) |")
w(f"| window order inside episodes (the thing `eid` cannot see) | each dump's stored per-window `ade6` equals ADE6 recomputed from its `pred` and the grid GT | max deviation **{alig['ade6_recompute_max_dev']['A']} m** (X), **{alig['ade6_recompute_max_dev']['B']} m** (B) |")
w(f"| checkpoint-label mask | each dump's `ck_pos_err` NaN mask equals the grid's `rc_ok` | X **{alig['ck_nan_mask_equals_grid_label_mask']['A']}**, B **{alig['ck_nan_mask_equals_grid_label_mask']['B']}** |")
w(f"| identity | the whole pipeline on X against X reads exactly 0.0 [0.0, 0.0], not separated, on every cell | **{ident['ok']}**: {ident['n_cells']} cells checked, {ident['n_not_zero']} not zero |")
w(f"| reproduction, X | each point value, and each per-arm interval where the eval JSON has one, equals `eval_M2b-mixo_s0.json` to 1e-6 | **{repA['ok']}**: {repA['n_checked']} checks, {repA['n_failed']} failed, max abs difference {repA['max_abs_diff']:.3g} |")
w(f"| reproduction, B | the same against `eval_M2b_s0.json` | **{repB['ok']}**: {repB['n_checked']} checks, {repB['n_failed']} failed, max abs difference {repB['max_abs_diff']:.3g} |")
w(f"| heading rebuild | the held-age rebuild of S's NOW-frame checkpoints reproduces the stored `ck_pos_err` on every window (the gate for the heading cells; the dump does not store the heading error) | max deviation X **{hr['A']} m**, B **{hr['B']} m** |")
w(f"| refusal on misaligned windows | different episode ids, different length, another grid, a within-episode reordering: rc 2, nothing written | pinned in `stack/tests/test_v8f_paired_families.py` (`test_misaligned_windows_are_refused`, `test_the_cli_refuses_rc2_on_misaligned_dumps_and_writes_nothing`) |")
if m_last and m_base and m_after:
    w(f"| mutation proof | every mutant turns the suite RED; the unmutated tool is GREEN before and after, bytes identical | **{m_last.group(1)} of {m_last.group(2)}** mutants RED ({m_last.group(3)} on an assertion); before: {m_base.group(1)} ({m_base.group(2).strip()}); after: {m_after.group(2)} ({m_after.group(3).strip()}), bytes identical {m_after.group(1)} (`mutate_pairfam.out`) |")
w("")
w("Reproduction compares at the eval JSON's own precision (its intervals are rounded to 4 dp); the per-arm intervals match because the tool uses the eval's seed 0 and B = 2,000.")
w("")
w("## 3. TACTICAL")
w("")
w("Two reads, kept apart. **Plan-derived**: the manoeuvre read off the arm's PLAN by the programme's own labeller (`maneuver_kinematics` -> `factor_from_kinematics`) against the one read off the human path; both are EXECUTED manoeuvres (the eval's own caveat: this is not \"selected vs executed\"). "
  "**Heads**: the lat / lon decision heads and the goal-token head against the v9 labels. Accuracy and kappa of the heads, the goal-token accuracy and the per-token F1 are cells the eval JSON does not carry; they use the same definitions as the plan-derived rows (accuracy, Cohen's kappa) and the eval's own goal threshold (`goal_prob > 0.5`). "
  "The goal-token accuracy is a micro accuracy over the labelled (window, token) cells and is dominated by the all-negative majority; read it beside the macro-F1.")
w("")
tab = pf.render_markdown(R, "X", "B")
tab = re.sub(r"the token's F1 is undefined on at least one resample \(no positive label in the drawn episodes\), so no percentile interval exists; the point values are reported",
             "no interval: the F1 is undefined on some resamples (point difference only)", tab)
tab = re.sub(r"no positive label for this token among its [\d,]+ labelled windows on this grid, so its F1 is undefined \(the eval reports None\)", "no positive label on this grid (the eval reports None)", tab)
parts = tab.split("### ")
for part in parts[1:]:
    title, body = part.split("\n", 1)
    if title.startswith("TACTICAL"):
        w("### " + title)
        w(body.rstrip())
        w("")
w("### Where n is below 2,399, and why")
w("")
w("Each reason below is counted from the mask the cell uses (printed in the json as `n_reason`), not guessed. For the strategic cells the mask is the `ck_pos_err` NaN pattern in the dumps themselves (identical in X and B, and equal to the grid's checkpoint-label mask). "
  "The lateral, longitudinal and goal-head masks are the v9 label join's (`has`, `lat_cls`, `lon_cls`, `goal_w`): the dumps carry no label arrays, so those masks cannot come from the npz.")
w("")
for r, ps in reasons.items():
    w(f"* {r}. Cells: " + ", ".join(f"`{p.split('tactical.')[-1] if p.startswith('tactical.') else p}`" for p in ps) + ".")
w("")
tok = T["heads"]["goal"]["per_token"]
nunav = [k for k, c in tok.items() if c["status"] == "UNAVAILABLE"]
npo = [k for k, c in tok.items() if c["status"] == "POINT_ONLY"]
w(f"**Goal tokens.** {len(nunav)} of 22 tokens have no positive label on this grid and are UNAVAILABLE ({', '.join('`' + k + '`' for k in nunav)}; the eval reports them as None for both arms). "
  f"{len(npo)} more ({', '.join('`' + k + '`' for k in npo)}) have positives in so few episodes that the token's F1 is undefined on some resamples; no percentile interval exists for them, so they carry the point difference only. "
  "The goal macro-F1 handles this the way the eval's definition does: on each resample it averages over the tokens that have a positive label in the drawn episodes.")
w("")
w("**Unavailable by design (both arms, the eval's own reason):** anchor selection (\"each dump's arm commits to ONE path per window: there is no candidate fan and no selection to score\") and selected-vs-executed (both label streams are executed manoeuvres). Both stay work items, not passes.")
w("")
w("## 4. STRATEGIC (S's checkpoint error at 30 / 50 / 80 m; active for v8F-mini)")
w("")
for part in parts[1:]:
    title, body = part.split("\n", 1)
    if title.startswith("STRATEGIC"):
        w(body.rstrip())
        w("")
w("Position error is the dump's own `ck_pos_err`. Heading error is rebuilt from the raw tick-frame `ckpt` and the held-age poses and is used because the rebuild reproduces the stored position error exactly (section 2). "
  "The n of each strategic cell is below 2,399 for the reason in section 3 (the checkpoint label is absent at that arc).")
w("")
w("## 5. Separated cells")
w("")
if sep:
    for p, c in sep:
        w(f"* `{p}`: {c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]")
else:
    w(f"**None.** Of {len(ok)} intervals, 0 exclude 0. UNAVAILABLE: {len(un)} ({len(nunav)} goal tokens with no positive label, plus anchor selection and selected-vs-executed). Point-difference-only: {len(po)} goal tokens.")
w("")
w("## 6. What this does not say")
w("")
w("* **One training seed per arm.** The intervals contain no training-seed variance, so they understate the uncertainty of any lever claim; a separated interval would not have been enough, and a non-separated one is not a no-effect claim. The seed floor F = 0.1207 m (B's two seeds) is an ADE6 figure and is not transferable to these cells.")
w("* **Tier.** Open-loop scoring of each arm's own held outputs (T1-class by the PREREG). The eval JSONs stamp `raw.tier` T0 for this eval; that labelling mismatch is the open item `RESULT_V8F_MIXO.md` section 9 already lists. The plan-derived manoeuvre read is an executed-manoeuvre agreement, not a selected-vs-executed score.")
w(f"* **Multiplicity.** {len(ok)} intervals are listed; they are descriptive, not a family-wise test.")
w("* **Added cells.** The heads' accuracy and kappa, the goal-token accuracy and the per-token F1 are not in the eval JSON, so they have no eval value to reproduce; they are checked by the brute-force bootstrap test and the mutation proof, not by the JSON control.")
w("")
w("## 7. For the next arm (FLAT, and every later v8F eval)")
w("")
w("The same tool takes any two `run` dumps on the eval-141 grid (it refuses otherwise). For FLAT:")
w("")
w("```")
w("python stack/scripts/v8f_paired_families.py --a <FLAT eval_*.npz> --b <B eval_*.npz> --out pairfam_<arm>.json --md pairfam_<arm>_tables.md --label-a <arm> --label-b B")
w("```")
w("")
w("with `PYTHONPATH=stack:taniteval` and `CUDA_VISIBLE_DEVICES=-1`. About 4 minutes on CPU (the identity control doubles the run; `--no-identity-control` skips it). rc 0 = controls passed, rc 2 = refusal, rc 3 = a control failed (the json is still written). "
  "`compute_pairs(A, B, ctx)` is importable for a reader that wants the blocks inline. The held-age options (`--ages`, `--hist`, `--max-age-t`, `--max-age-s`, `--window-start`) default to the M2b family's; the heading cells fall to UNAVAILABLE, with the deviation, if the rebuild does not reproduce a dump's stored `ck_pos_err`.")
w("")
w("## 8. Manifest (nothing staged, committed or pushed)")
w("")
w("| artifact | path (relative to the repo root) |")
w("|---|---|")
w("| the tool | `stack/scripts/v8f_paired_families.py` |")
w("| its tests | `stack/tests/test_v8f_paired_families.py` |")
w("| mutation script and output | `TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/2026-10-25-pairfam2/mutate_pairfam.py`, `mutate_pairfam.out` |")
w("| the banked read | `.../raw/2026-10-25-pairfam2/pairfam_mixo_s0.json` (schema `v8f-paired-families-v1`), `pairfam_mixo_s0_tables.md`, `pairfam_mixo_s0.out` |")
w("| this write-up and its generator | `.../raw/2026-10-25-pairfam2/PAIRFAM_MIXO.md`, `make_pairfam_mixo_md.py` |")
OUT.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
print("wrote", OUT, len("\n".join(L)))
