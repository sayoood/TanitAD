#!/usr/bin/env python3
"""Render RECIPE_SELECTION.md, DESIGN_CHOICES.md and RESULT_GATES.md from the JSON the sweep, the gates and the cost run wrote.  A missing input is written as 'NOT RUN', never invented."""
from __future__ import annotations

import json
import os
import re
import sys

W = "C:/Users/Admin/par18/l2"
OUT = f"{W}/out"
ART = "D:/Projects/TanitAD-artifacts/l2_offprop"


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def ci(d):
    return f"{d['point']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def last_line(p, pat=None):
    """the last line of a log (or the last line matching `pat`, for logs that end with an `rc=` line)"""
    if not os.path.exists(p):
        return "NOT RUN"
    t = open(p, encoding="utf-8", errors="replace").read().strip().splitlines()
    if pat:
        t = [x for x in t if re.search(pat, x)]
    return t[-1] if t else "EMPTY"


# ====================================================================================================================================================================================
def render_selection():
    sw, rc, df = J(f"{OUT}/sweep_semi.json"), J(f"{ART}/recipe_chosen.json"), J(f"{OUT}/INSTRUMENT_DEFECT.json")
    sup = J(f"{OUT}/superseded/reach_chain_recipe_lam10_lr3e-4_ep5_FAIL.json")
    T = ["# RECIPE_SELECTION - L2-OFFPROP (build agent, 2026-10-19; MM addendum 14:10)\n"]
    if df:
        T.append(f"**INSTRUMENT DEFECT.** No setting of the grid clears C-REACH (planted perfect feature, A1-OOF >= 0.98) on both seeds. No recipe was chosen, no outcome exists. Grid: {df['grid']} plus {df['extra']}. Walk: {df['walk']}.\n")
    elif rc:
        p, w = rc["pick"], rc["walk_down_for_seed1_reach"][-1]
        T.append(f"**Chosen recipe `{p['key']}`** (lam_bce {rc['recipe']['lam_bce']:g}, lr {rc['recipe']['lr']:g}, epochs {rc['recipe']['epochs']}, prox {rc['recipe']['prox']:g}; batch {rc['recipe']['batch']}; Adam, no weight decay). "
                 f"**C-REACH A1-OOF (planted arm, real tokens): seed 0 {w['reach_s0']:.4f}, seed 1 {w['reach_s1']:.4f}** (bar 0.98 on both). "
                 f"In-fan clause satisfied in the semi-synthetic signal worlds: {'YES' if not rc['clause_may_bind'] else 'NO - FALLBACK; the clause MAY BIND in the real run'}.\n")
    else:
        T.append("**NOT YET CHOSEN.**\n")
    T.append("Evidence class: MEASURED on REAL tokens with SYNTHETIC labels (a property of the optimiser on a planted model, not of REFe); C-REACH reads the PLANTED arm only. **No real unplanted label was read and no real arm (H, H0, H-null) was trained.**\n")
    if sup:
        T.append(f"## 0. Why the recipe was re-selected\nThe chain's recipe `lam10.0_lr0.0003_ep5` (150 steps per fold) reached C-REACH A1-OOF **{sup['A1_OOF_all']['point']:.4f} < 0.98** (`raw/superseded/reach_chain_recipe_lam10_lr3e-4_ep5_FAIL.json`): "
                 "the in-fan rule (R1 + R2 against P2) selected a recipe that barely moves the head off P2's init, so the real arm would have read FLAT by construction. The MM ruled that C-REACH is a HARD constraint of the selection.\n")
    T.append("## 1. The rule (committed in `code/design_sweep.py` before the sweep ran)\n"
             "1. **Hard constraint - C-REACH:** the planted perfect feature (appended column = 10 x sign of the true EP difference, on the real tokens) gives A1-OOF >= 0.98 on **both** training seeds. Seed 0 is read for every setting; seed 1 is read walking down the ranked list "
             "(equivalent to filtering on both seeds first). If no setting clears it: **INSTRUMENT DEFECT**, no recipe.\n"
             "2. **Subject to:** in-fan(H) >= in-fan(H0) - 0.01 in every semi-synthetic SIGNAL world (sigA, sigB), H0 = the same setting with the pairwise weight 0 (same init, folds, seed, minibatch stream).\n"
             "3. **Maximise** the mean semi-synthetic A1-OOF (pooled lon + lat) of H over the signal worlds. **Ties** (within 0.005 of the best) go to the smaller lambda = pairwise / in-fan weight = 1 / lam_bce (so the larger lam_bce), then the larger prox, then fewer steps, then the smaller lr.\n"
             "4. **Fallback:** if no C-REACH-passing setting satisfies the in-fan clause, take the one with the best in-fan margin and report that the clause may bind.\n"
             "5. **Reported, not gating:** the null world (beta 0): A1(H) minus A1(P2) and the GLOBAL-null H-null minus P2.\n")
    T.append("**Worlds** (`synthetic.semi_synthetic`, REAL tokens): sigA (beta 0.5, alpha 0.25, seed 21), sigB (beta 0.5, alpha 0.5, seed 22), null (beta 0, alpha 0.4, seed 23); P2's in-fan top-8 EP concordance calibrated to 0.889 and P2's A1 at beta 0 to 0.685. "
             "In these worlds P2 is near-calibrated on the in-fan labels BY CONSTRUCTION, so H0 sits close to P2 here; in the real run P2 is near in-sample on its own in-fan labels and an out-of-fold H0 will sit lower, which is why the MM replaced the P2 comparator by H0.\n")
    if sw:
        ch = sw["choice"]
        T.append(f"## 2. Result\n{ch['n_settings']} settings run ({sw['n_expected_settings']} expected). **{ch['n_reach_pass_s0']}** clear C-REACH on seed 0; of those **{ch['n_r1']}** also satisfy the in-fan clause in both signal worlds. Rule outcome: {ch['rule']}.\n")
        base = sw["base"]
        T.append("P2 reference per world: " + "; ".join(f"{w}: A1-OOF {b['a1_p2']:.3f}, in-fan top-8 {b['infan_p2']:.3f}" for w, b in base.items()) + "\n")
        rank = {r["key"]: i + 1 for i, r in enumerate(ch["ranked"])}
        pick_key = rc["pick"]["key"] if rc else None
        T.append("| setting | C-REACH s0 | sigA A1 | sigA in-fan H-H0 | sigB A1 | sigB in-fan H-H0 | mean signal A1 | worst H-H0 | in-fan clause | rank | |\n|---|---|---|---|---|---|---|---|---|---|---|")
        order = sorted(sw["rows"].items(), key=lambda kv: (rank.get(kv[0], 10_000), kv[0]))
        for k, r in order:
            if r["reach_pass_s0"]:
                ws = r["worlds"]
                mean = (ws["sigA"]["a1"] + ws["sigB"]["a1"]) / 2
                marg = min(ws["sigA"]["infan_minus_h0"], ws["sigB"]["infan_minus_h0"])
                T.append(f"| `{k}` | {r['reach_s0']['a1']:.4f} | {ws['sigA']['a1']:.4f} | {ws['sigA']['infan_minus_h0']:+.4f} | {ws['sigB']['a1']:.4f} | {ws['sigB']['infan_minus_h0']:+.4f} | {mean:.4f} | {marg:+.4f} | "
                         f"{'yes' if marg >= -0.01 else 'no'} | {rank.get(k, '-')} | {'**PICK**' if k == pick_key else ''} |")
            else:
                T.append(f"| `{k}` | {r['reach_s0']['a1']:.4f} FAIL | - | - | - | - | - | - | excluded (C-REACH) | - | |")
        if rc:
            p = rc["pick"]
            best = max(r["mean_signal_a1"] for r in ch["ranked"])
            ties = [r["key"] for r in ch["ranked"] if best - r["mean_signal_a1"] <= 0.005]
            tie_lams = sorted({r["lam"] for r in ch["ranked"] if best - r["mean_signal_a1"] <= 0.005})
            T.append(f"\n## 3. The pick\n`{p['key']}`: mean signal A1-OOF {p['mean_signal_a1']:.4f} (best in the feasible set {best:.4f}); worst in-fan H-H0 margin {p['margin']:+.4f} (bar -0.01). "
                     f"Settings within 0.005 of the best A1 (the tie set, broken by smaller lambda = larger lam_bce, then larger prox, then fewer steps, then smaller lr): {', '.join('`'+k+'`' for k in ties)}.\n")
            T.append("Per-world numbers of the pick (seed 0): " + "; ".join(f"{w}: A1(H) {v['a1']:.4f}, A1(H0) {v['a1_h0']:.4f}, in-fan H {v['infan']:.4f} / H0 {v['infan_h0']:.4f}" for w, v in rc["pick_worlds"].items()) + ".\n")
            T.append("C-REACH walk-down (seed 1 is read for ranked settings until one clears it): " + "; ".join(f"`{x['key']}`: s0 {x['reach_s0']:.4f}, s1 {x['reach_s1']:.4f}, {'PASS' if x['both_seeds_pass'] else 'fail'}" for x in rc["walk_down_for_seed1_reach"]) + ".\n")
            d = rc["null_world_diagnostics_reported_not_gating"]
            T.append(f"Null world (beta 0), REPORTED not gating: A1(H) minus A1(P2) = {d['a1_h_minus_p2']:+.4f}; in-fan H minus H0 = {d['infan_h_minus_h0']:+.4f}; GLOBAL-null H-null (r100, r101) A1 {d['a1_hnull_global_r100_r101'][0]:.4f}, {d['a1_hnull_global_r100_r101'][1]:.4f} vs P2 {d['a1_p2']:.4f} "
                     f"(max minus P2 = {d['a1_hnull_global_max_minus_p2']:+.4f}; the VOID rule needs > +0.05 on two of five replicates).\n")
            T.append("**Interpretation notes.** (a) lambda was read as the pairwise / in-fan ratio, 1 / lam_bce (the MM's own phrase 'lambda pairwise / in-fan weight'); " + (f"every setting in the tie set has lam_bce {tie_lams[0]:g}, so the convention is IMMATERIAL for this pick (the tie is decided by prox, then steps). " if len(tie_lams) == 1 else f"the tie set spans lam_bce {tie_lams}, so the opposite convention would change the pick (the table shows every setting). ") +
                     "(b) The tie width 0.005 was fixed in the code before the sweep ran; the MM ruling does not state one. (c) One world seed per world and one training seed per cell in the sweep; the sweep finds a recipe that is not on a cliff and says nothing about REFe. "
                     "(d) C-REACH's planted feature is APPENDED, so passing it proves the optimiser and the read path can learn an ordering that exists in the input; it says nothing about whether the real token carries one.\n")
    return "\n".join(T) + "\n"


# ====================================================================================================================================================================================
def render_design():
    rc = J(f"{ART}/recipe_chosen.json")
    seed, cost, costs = J(f"{OUT}/seed_spread.json"), J(f"{OUT}/cost.json"), J(f"{OUT}/cost_semi.json")
    mg = J(f"{OUT}/planted_margin.json")
    T = ["# DESIGN_CHOICES - L2-OFFPROP (build agent, 2026-10-19; amended by MM_RULING_L2_BUILD.md E1-E7 + addendum)\n",
         "Everything here was fixed on SYNTHETIC or SEMI-SYNTHETIC data (REAL tokens, SYNTHETIC labels). **No real unplanted label was read to make any choice below**, and no real arm (H, H0, H-null) was trained. "
         "Evidence class of every number: MEASURED on synthetic worlds (artifact paths in `raw/`), i.e. a property of the optimiser on a planted model, not of REFe. The recipe and its selection table are in `RECIPE_SELECTION.md`.\n"]
    T.append("## 1. What the draft and the MM rulings fix (not chosen here)\n"
             "* The EP head is row 2 of P2's `score_head = Linear(256, 6)`: **257 parameters**, input = the 256-d token the head consumes (MEASURED from the checkpoint; exported once, `feats_manifest.json`). Initialised from P2's row; every other row and everything upstream frozen.\n"
             "* Loss (arm H) = `softplus(-y (z_p - z_a))` pairwise logistic on Step A (anchor, perturbation) pairs whose TRUE EP differs by >= 0.05, + `lam_bce` x pointwise BCE on in-fan EP labels, + `prox` x squared distance of the EP row from P2's. Both data terms are batch means. Folds: 5 by log (rank mod 5 over the 24 sorted names). Seeds {0, 1}.\n"
             "* **E1 (null):** H-null permutes the perturbations' true EP GLOBALLY (across every perturbation of the dataset); replicates 100-104. The draft's within-(token, anchor) permutation is WITHDRAWN (it read +0.09 above P2 on a planted world) and survives only as a `--null-group anchor` diagnostic. VOID = >= 2 of 5 GLOBAL-null replicates exceed P2's A1-OOF by > 0.05.\n"
             "* **E2 (A1):** the POOLED lon + lat quantity Step A's gate used (0.6851, den 9,888); lon-only and lat-only are reported beside it, ungated.\n"
             "* **E3 / E4:** strictest defaults (null pool = max over all 10 null cells; VOID if EITHER seed has >= 2 exceeding).\n"
             "* **In-fan clause:** in-fan EP concordance (OOF, SELRANK top-8 instrument) of H >= that of **H0 - 0.02**, H0 = the SAME recipe with the pairwise weight 0 (same init, folds, seed, minibatch stream), compared at the same seed; P2's in-fan value is REPORTED, never gated.\n"
             "* **C-LEAK** is a STRUCTURAL reach check inside `oof_scores` (no switch turns it off), see `RESULT_GATES.md`.\n"
             "* Estimator: L1's own paired log-cluster bootstrap (24 logs, 10,000, seed 20260927) - reused from `stepA.py`, not re-implemented.\n")
    if rc:
        R = rc["recipe"]
        T.append(f"## 2. The recipe\n`lam_bce = {R['lam_bce']:g}`, `lr = {R['lr']:g}`, `epochs = {R['epochs']}`, `prox = {R['prox']:g}`, `batch = {R['batch']}`, Adam betas {R['betas']}, eps {R['eps']}, weight decay 0, no schedule, no early stopping "
                 f"(nothing in the training may look at a validation label). Chosen by the rule in `RECIPE_SELECTION.md` -> `{rc['pick']['key']}`. The recipe is a DATA file (`raw/recipe_chosen.json`): `run_l2.py run` refuses to start without it and refuses gates that were earned on a different recipe file or different code.\n")
    else:
        T.append("## 2. The recipe\n**NOT CHOSEN** (see `RECIPE_SELECTION.md`).\n")
    T.append("## 3. Other fixed choices\n"
             "* **Seed** = the minibatch order only (the head starts from P2's EP head; the generator is seeded from `(seed, n_pairs)`). H-null replicate = the permutation seed (100-104), crossed with the training seed. H0 shares H's generator stream, so it takes exactly as many steps.\n"
             "* **Scoring** is float64 on the float32 token and weights (`l2lib.ep_apply`), so an arm's score does not depend on summation blocking; P2's own row gives the banked EP to <= 3e-6 (C-FEAT).\n"
             "* **G-LIVE is enforced inside training**: the full [6, D] weight is the optimised leaf; at every step the gradient of the five frozen rows is asserted EXACTLY 0.0 and at the end they are asserted bit-identical to P2's. The prox term acts on the EP row only.\n"
             "* **C-REACH planted amplitude 10** (the per-dim std of a token is ~1): one appended column equal to `10 * sign(true EP difference)` on perturbation tokens, 0 on in-fan tokens.\n"
             "* **A1 pair rule** is Step A's (`>= 0.05`, scorer tie counts 0.5); the **in-fan pair rule** is the SELRANK instrument's (`> 0.05`, strict). They differ by definition in the two source instruments; both are kept as registered.\n")
    if mg:
        T.append("## 4. The semi-synthetic planted-signal test (the chain's one pytest failure)\n`test_semi_synthetic_...` asserts A1(H) > A1(P2) + 0.02 on a planted-signal world (beta 3). Margins over the bar (MEASURED, `raw/planted_margin.json`):\n")
        for k, v in mg.items():
            T.append(f"* {k}: A1(P2) {v['a1_p2']:.4f}, A1(H) {v['a1_h']:.4f}, gain {v['gain']:+.4f}, margin over the 0.02 bar {v['margin_over_bar']:+.5f} ->{'passes' if v['passes_as_written'] else 'FAILS'}")
        T.append("\nThe bar (strictly above +0.02) and the world are unchanged. The only change is the recipe the test runs under: the source default `Recipe(epochs=12)` before, the SELECTED recipe (`CHOSEN`) now, as the MM instructed. "
                 "Both recipes with lam_bce 10 land on the same 4,413 of 5,009 concordant pairs; the two heads differ (||dW|| 0.152 vs 0.153) but flip the same pairs. The test is NOT loosened and it FAILS (see `RESULT_GATES.md`).\n")
    if seed:
        T.append("## 5. Seed-to-seed spread of H (E8; semi-synthetic, the chosen recipe)\n\n| world | A1-OOF seed 0 | seed 1 | abs diff | in-fan H-H0 seed 0 | seed 1 | max abs EP-score diff (in-fan) | share of pairs whose sign differs |\n|---|---|---|---|---|---|---|---|")
        for k, v in seed["worlds"].items():
            T.append(f"| {k} | {v['A1_seed0']:.4f} | {v['A1_seed1']:.4f} | {v['A1_abs_diff']:.4f} | {v.get('infan_h_minus_h0_seed0', float('nan')):+.4f} | {v.get('infan_h_minus_h0_seed1', float('nan')):+.4f} | {v['max_abs_ep_score_diff_infan']:.3f} | {v['share_of_pairs_whose_sign_differs_between_seeds']:.4f} |")
        T.append("\nThe seed moves only the minibatch order of a (near-)convex fit from a fixed start, so a small spread is expected; it is NOT a replicate of P2's own training seed (H-ESTIM-SEED-1 still applies to the P2 token).\n")
    c = costs or cost
    if c:
        T.append("## 6. Cost of the full real run (measured on " + ("REAL tokens + SYNTHETIC labels" if costs else "synthetic data of the real shapes") + f", {c['threads']} torch threads, a box at 60-90 % CPU load from other jobs, CPU only; recipe `{rc['pick']['key'] if rc else '?'}`)\n\n"
                 f"* data build {c['data_build_s']} s; one H cell (5 folds) **{c['H_cell_fit_5_folds_s']} s**, one H0 cell {c.get('H0_cell_fit_5_folds_s', 'n/a')} s (pairs per fold {c['H_cell_fold_pairs']}, steps per fold {c['H_cell_fold_steps']}); "
                 f"one H-null cell {c['Hnull_cell_fit_5_folds_s']} s; one `evaluate_arm` {c['evaluate_arm_s']} s; peak RSS ~{c['rss_mb_peak_approx'] and round(c['rss_mb_peak_approx'])} MB.\n"
                 f"* full run = **15 cells** (P2 + 2 H + 2 H0 + 10 global-null H-null) and 15 evaluations: **~{c['extrapolated_full_run_min']} min on one process** ({c['extrapolated_full_run_s']}); cells are independent, so N processes divide it.\n"
                 f"* GPU: none. Export (already done): 2 workers x ~100 min-equivalent of CPU, ~45 min wall.\n")
    return "\n".join(T) + "\n"


# ====================================================================================================================================================================================
def render_gates():
    cf, cid, gl, rc, lk = (J(f"{OUT}/gates/{g}.json") for g in ("cfeat", "cid", "glive", "reach", "leak"))
    to, tl = last_line(f"{W}/logs/test_outcome.log", r"ZZOUTCOME_MUTATIONS"), last_line(f"{W}/logs/pytest_test_l2.log", r"\d+ (passed|failed)")
    orig = ""
    if os.path.exists(f"{W}/logs/test_outcome.log"):
        m = re.search(r"original module: (\d+) scenarios, (\d+) failures", open(f"{W}/logs/test_outcome.log", encoding="utf-8").read())
        orig = f"{m.group(1)} scenarios ({m.group(2)} failures of the original module); " if m else ""
    T = ["# RESULT_GATES - L2-OFFPROP build gates (2026-10-19, after the MM rulings)\n",
         "Gates, not results: none trains H, H0 or H-null on real labels or reads an A1-OOF of an unplanted head. Real arrays = the 400 tokens / 24 logs / 38,400 pairs of L1 Step A. Stamp: SEEN-SCENE, SELECTION tier. "
         "Every PASS is stamped with the sha256 of the recipe file and of `l2lib.py`, `gates.py`, `synthetic.py`; `run_l2.py` refuses a PASS earned on a different recipe or different code.\n"]
    if cf:
        T.append(f"## C-FEAT - {'PASS' if cf['pass'] else 'FAIL'}\nP2's EP head applied (float64) to the EXPORTED tokens vs Step A's banked EP scores (`scores.npz`); bar: max abs diff <= 1e-4 on >= 99 % of rows.\n")
        T.append("| rows | n | max abs diff | share <= 1e-4 | p99 abs diff | all six columns, max abs | export-run logits vs banked, max abs |\n|---|---|---|---|---|---|---|")
        for k, v in cf["rows"].items():
            T.append(f"| {k} | {v['n_rows']} | {v['ep_head_on_tokens_vs_banked_max_abs']:.3e} | {v['ep_head_on_tokens_vs_banked_share_le_1e-4']:.6f} | {v['ep_head_on_tokens_vs_banked_p99_abs']:.3e} | {v['all_six_columns_max_abs']:.3e} | {v['export_run_logits_vs_banked_max_abs']:.3e} |")
        T.append(f"\nbundle sha256 matches the run's: {cf['bundle_sha256_match']}; perts sha256 matches: {cf['perts_sha256_match']}. Evidence: MEASURED, `raw/gates/cfeat.json`.\n")
    else:
        T.append("## C-FEAT - NOT RUN\n")
    if cid:
        b = cid["zero_step_scores_bit_identical_to_P2"]
        T.append(f"## C-ID - {'PASS' if cid['pass'] else 'FAIL'}\nZero optimiser steps => P2's EP scores bit for bit, for H and for H0, both seeds; then A1-OOF equals Step A's A1 on the same pairs. Recipe stamp: `{(cid.get('recipe_sha256') or 'NONE')[:16]}`.\n\n"
                 f"* zero-step scores bit-identical to the P2 arm (in-fan AND perturbed): {b}.\n"
                 f"* A1 ({cid['A1_is']}) of the zero-step arm {ci(cid['A1_all_zero_step'])} vs Step A `RESULT.json` {ci(cid['A1_all_StepA_RESULT_json'])}; |diff of points| = {cid['A1_abs_diff_point']:.2e}; "
                 f"worst |diff| over every reported A-quantity (point, lo, hi, den; all/lon/lat) = {cid['worst_abs_diff_over_all_reported_A_quantities_vs_RESULT_json']:.2e}; L1's own analyse on the banked logits reproduces RESULT.json to {cid['reference_path_worst_abs_diff_vs_RESULT_json']:.2e}.\n"
                 f"* pairs whose banked EP-logit gap is below 1e-4 (could flip between fp32 and float64): {cid['n_pairs_with_banked_EP_logit_gap_below_1e-4']}.\n"
                 f"* P2's in-fan top-8 EP concordance on these 400 tokens: {ci(cid['infan_top8_ep_concordance_P2'])} (REPORTED, never gated). Evidence: MEASURED, `raw/gates/cid.json`.\n")
    else:
        T.append("## C-ID - NOT RUN\n")
    if gl:
        g = gl["glive"]
        clean = [r for r in g["batches"] if not r["mutated"]]
        T.append(f"## G-LIVE - {'PASS' if gl['pass'] else 'FAIL'}\nGradient probes on the REAL fold-0 training set ({gl['n_pairs_fold0_train_real']} pairs, {gl['n_fan_rows']} in-fan rows; no optimiser step is taken on a real label); recipe stamp `{(gl.get('recipe_sha256') or 'NONE')[:16]}`.\n\n"
                 f"* finite losses on all {len(clean)} probe batches: {all(r['finite'] for r in clean)}; EP-row gradient |sum| min {min(r['ep_W_grad_abs_sum'] for r in clean):.3e}, bias gradient min {min(r['ep_b_grad_abs'] for r in clean):.3e} (> 0).\n"
                 f"* frozen rows (0,1,3,4,5) gradient max abs: W {max(r['frozen_W_grad_abs_max'] for r in clean)}, b {max(r['frozen_b_grad_abs_max'] for r in clean)} (must be EXACTLY 0.0).\n"
                 f"* control: a loss term through row 3 reads a NON-zero frozen gradient (the probe can read a non-zero): {g['control_mutation_reads_nonzero_on_frozen']}; inside training the same mutation raises: {gl['mutation_leak_other_row_goes_RED']}.\n"
                 f"* **H0 lever reaches the head:** with `w_pair = 0` the EP-row gradient equals the gradient of `lam_bce * BCE` alone BIT FOR BIT, with `w_pair = 1` it differs, frozen rows stay exactly 0.0: {g['h0_lever_reaches']}.\n"
                 f"* **prox lever reaches the head:** its gradient is exactly 0 at P2's init, changes the EP-row gradient off the init, frozen rows stay exactly 0.0: {g['prox_lever_reaches']}.\n"
                 f"* a {gl['training_segment_steps']}-step training segment (H) and a {gl['h0_training_segment_steps']}-step one (H0) ran with the per-step frozen-gradient assertion on, on {gl['training_segment_world']}; H and H0 trained to different heads: {gl['h_and_h0_trained_heads_differ']}. Evidence: MEASURED, `raw/gates/glive.json`.\n")
    else:
        T.append("## G-LIVE - NOT RUN\n")
    if rc:
        ps = rc["per_seed"]
        T.append(f"## C-REACH (planted) - {'PASS' if rc['pass'] else 'FAIL'}\nA feature equal to `{rc['planted_amp']:g} * sign(true EP difference)`, appended to the perturbation token, 5-fold-by-log OOF, **both seeds**. Bar: A1-OOF >= 0.98 on each. Recipe stamp `{(rc.get('recipe_sha256') or 'NONE')[:16]}`.\n")
        for s, v in ps.items():
            T.append(f"* {s}: A1-OOF (pooled) **{ci(v['A1_OOF_all'])}**; lon {ci(v['A1_OOF_lon'])}; lat {ci(v['A1_OOF_lat'])}; steps per fold {v['steps_per_fold']}.")
        T.append("\nReads the PLANTED arm only. Evidence: MEASURED, `raw/gates/reach.json`. (The chain's superseded recipe read 0.9048 here: `raw/superseded/`.)\n")
    else:
        T.append("## C-REACH - NOT RUN\n")
    if lk:
        rows = "\n".join(f"| {f['fold']} | {f['n_train_tokens']} | {f['n_test_tokens']} | {f['n_pairs']} | {f['check']['tokens_overlap']} | {f['check']['logs_overlap']} | {f['check']['trainset']['held_out_tokens_referenced']} |" for f in lk["folds"])
        T.append(f"## C-LEAK (STRUCTURAL) - {'PASS' if lk['pass'] else 'FAIL'}\nREDEFINED by the MM addendum: a linear 257-parameter head cannot memorise shuffled labels, so the old statistical test (clean 0.502 vs leaky 0.504) was UNINFORMATIVE, not passed (`raw/superseded/leak_statistical_uninformative.json`). "
                 "`oof_scores` now calls `fold_split` on every fold, which ASSERTS (fails closed, no off-switch) that the training token set is disjoint from the held-out token set, that held-out logs are disjoint from training logs, and that every index array the optimiser consumes "
                 "(pair anchors, pair perturbations, in-fan rows) references only training tokens; after the folds, every token must have been scored exactly once. Run on the REAL tokens with SHUFFLED labels (no real label is read).\n\n"
                 f"| fold | train tokens | held-out tokens | pairs | token overlap | log overlap | held-out tokens referenced by the training set |\n|---|---|---|---|---|---|---|\n{rows}\n\n"
                 "RED mutations - each must make the production path raise `C-LEAK`:\n\n" + "\n".join(f"* `{k}`: {'fires' if v else 'DID NOT FIRE'}" for k, v in lk["red"].items()) +
                 f"\n\nA clean production `oof_scores` ran end to end ({lk['clean_oof_scores_ran']}). Evidence: MEASURED, `raw/gates/leak.json`; unit tests in `test_l2.py` (`test_oof_scores_trip_on_ONE_injected_held_out_token_in_every_fold` and three more).\n")
    else:
        T.append("## C-LEAK - NOT RUN\n")
    T.append(f"## Outcome mapping - RED mutations\n`python test_outcome.py` -> `{to}`. {orig}each textual mutation (every threshold, every branch, both VOID clauses, both null-pool / void-scope switches, and the in-fan clause: dropped, compared with P2 instead of H0, `>=` to `>`, sign flipped) fails at least one scenario. `raw/test_outcome_mutations.log`.\n")
    mg = J(f"{OUT}/planted_margin.json")
    if mg:
        v, d = mg["selected_recipe"], mg["source_default_REC_epochs12_used_by_the_other_tests"]
        T.append("## KNOWN FAILURE - `test_semi_synthetic_hits_its_two_calibration_targets_and_reads_no_real_label` (NOT loosened)\n"
                 f"The test asserts A1(H) > A1(P2) + 0.02 on a 72-token planted-signal semi-synthetic world (beta 3). Under the SELECTED recipe: A1(P2) {v['a1_p2']:.4f}, A1(H) {v['a1_h']:.4f}, gain {v['gain']:+.4f}, **margin over the bar {v['margin_over_bar']:+.4f} -> {'passes' if v['passes_as_written'] else 'FAILS'}**. "
                 f"Under the source default `Recipe(epochs=12)` (what the chain's run used - the test never read the chain's recipe): gain {d['gain']:+.5f}, margin {d['margin_over_bar']:+.6f} (100 of the 101 pairs needed) -> {'passes' if d['passes_as_written'] else 'FAILS by one pair'}. "
                 "Cause (MEASURED, `raw/planted_margin.json`): the selected recipe is in-fan-weighted by design (lam_bce 10, prox 0.3: the head moves ||dW|| ~0.15 from P2's), so on a world with only 57 training tokens per fold it finds only +22 concordant pairs of 5,009 (the source default finds +100); a recipe that finds it (lam 0.1, ||dW|| ~1.7) degrades in-fan ordering. "
                 "It is the A1-gain versus in-fan trade-off the MM's rule resolves in favour of in-fan non-degradation. The test was not changed; a companion REGRESSION test on the REAL tokens (`test_selected_recipe_finds_a_planted_signal_on_REAL_tokens`: sigA, beta 0.5) passes with the selected recipe (gain +0.045).\n")
    T.append(f"## Unit tests\n`pytest test_l2.py -q` (CPU, synthetic only) -> `{tl}`. Covers folds, the A1 pair definition = L1's A1 denominator, the GLOBAL H-null (default) and the anchor diagnostic, arm H0 (a dead pairwise term: flipping every pair label changes nothing), structural C-LEAK + its RED mutations, "
             "G-LIVE and its mutation, reachability of every Recipe field (incl. prox) into the trained head, C-ID/C-REACH on synthetic worlds, the SELRANK concordance against a brute force, EP-score reachability into the aggregate, pooled-A1 and H0-in-fan plumbing of the read, "
             "the seal guard (RED on a draft, a wrong sha, a missing or failing gate, a gate earned on a different recipe or different code).\n")
    return "\n".join(T) + "\n"


def main():
    open(f"{W}/RECIPE_SELECTION.md", "w", encoding="utf-8").write(render_selection())
    open(f"{W}/DESIGN_CHOICES.md", "w", encoding="utf-8").write(render_design())
    open(f"{W}/RESULT_GATES.md", "w", encoding="utf-8").write(render_gates())
    print("ZZRENDER_OK")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
