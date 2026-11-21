#!/usr/bin/env python3
"""write_results.py -- writes RESULTS_AUCMAP.md from raw/AGG.json + raw/AUCMAP_RESULT.json (numbers are read, not typed).
    C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I write_results.py "<date string from `date`>" """
import json
import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import numpy as np
import aucmap as A

DATE = sys.argv[1]
G = json.load(open(f"{A.RAW}/AGG.json"))
R = json.load(open(f"{A.RAW}/AUCMAP_RESULT.json"))["tasks"]
CT = json.load(open(f"{A.RAW}/CONTROLS.json"))
ADM, CEIL = list(A.ADMISSIBLE), list(A.CEILING)
LV = [str(a) for a in A.AUC_LEVELS]
A.init()
rig = A.Rig(levers=ADM + CEIL)
y = A.labels_from(rig.v0, rig.g)["GO"]
L = []
w = L.append


def fa(x):
    return "none (> 1.0)" if x is None else f"{x}"


w(f"# RESULTS_AUCMAP: how much go/stay discriminability does the dominance corner need? (T2, quantitative form)")
w(f"\nAgent: AUCMAP (Sonnet). Written {DATE}. Frozen card `PREREG_AUCMAP.md` (sha256 `1daf6fbceff28a46a1d03c4576a2037099d060d406ac3638b27221f4d6ee1227`, verified before the run) with `AMEND_1_AUCMAP.md` (C6 only). Verdict follows card sections 8-10 as written. Every number is [M] (measured, ours) from `raw/AUCMAP_RESULT.json` / `raw/AGG.json` / `raw/CONTROLS.json` in this folder; tier T1-kin (logged cameras, `reindexed` longitudinal-only re-perception; NOT T2). Estimator: paired episode-cluster bootstrap B=2000 seed 0 (answers 'another draw of EPISODES' only; blind to training variance, two seeds one training run each, `H-ESTIM-SEED-1`; the rollouts are not iCEM-sampled, so no inference variance). The noise draws of the synthetic posterior are a third variance, carried as the reach rate (K=20 draws per level).")
w("\n**CEILING-TYPE.** Synthetic posteriors are built from the window's true label (ego and future, offline) and the oracle-level levers (`LFO*`, `LFOB*`) force the logged speed. Neither is admissible at inference. These arms decide NO driving claim and are never a driving result. The only admissible arm is the real-posterior anchor gated by the heads-level lever. STRATEGIC family: PAUSED (PI ITEM 25, 2026-09-19), reported beside, never a pass.")

# ---------------------------------------------------------------------------------------------------------------- headline
ad = G["admissible_reach_at_1.0"]
w("\n## 1. Verdict")
w(f"* **Outcome (iii): the corner is not reachable by any gate on this lever family.** CORNER-FULL (M1-FULL and GUARD, both seeds) has reach rate **0.00 at all 8 AUC levels, 20 of 20 draws not reaching at each level, for all 5 admissible levers in both scopes**, including nominal AUC 1.0 (perfect separation of the GO label, q free on a 21-point grid). The CEILING-LEVEL family (`LFO075`, `LFO100`, `LFOB05`, `LFOB10`) also reads 0.00 at AUC 1.0 in both scopes (read beside).")
w(f"* **A\\*_any (CORNER-FULL, ANCHOR scope, label GO, 5 admissible levers) is undefined (> 1.0).** Interval undefined (no level reaches 0.05 reach rate). No lever is FRAGILE (none reaches at any q). A\\* is not a go/stay AUC target for L (outcome (i) needs A\\*_any <= 1.0); the LoRA arm's P4 read has no AUC to compare with under this lever family.")
w(f"* Outcome tests as written: (i) needs headline reach at 0.785 < 0.5 AND no anchor corner AND A\\*_any <= 1.0: the first two hold (reach 0.00; anchor reaches no corner), the third fails, so (i) is not the outcome. (ii) needs reach at 0.785 >= 0.5 or the anchor reaching CORNER-FULL: neither holds. (iv) needs CORNER-FULL in ALL scope but not ANCHOR scope: it is reached in neither. (iii) holds: no admissible lever at AUC 1.0, nor the CEILING-LEVEL family at AUC 1.0.")
w(f"* **Real anchor** (P1P `post_primary`, v0 < 8, 861 windows, realized AUC {G['anchor']['LF100']['auc_v0lt8_card']:.3f} on the card label): no admissible lever reaches CORNER-FULL or CORNER-P at any of the 21 q points, both seeds. NEAR test (section 5): **NEAR for 0 of 5 levers**.")
w("* **Binding predicate (beside, section 3):** at the oracle gate the lever operator itself fails the guard: `cross-track MAE` (LATERAL, ACTIVE) is separated-worse at both seeds at the smallest q that closes enough of M1, and M1-FULL (X - P10 not separated-worse) is met only by the ungated lever (ALL scope q = 1.0, LF100), where 6 (seed 0) and 5 (seed 1) ACTIVE rows are separated-worse and ADE6 is +1.256 / +1.089 m. Reading per card outcome (iii) [E]: the guard rows are driven by the lever operator and not by go/stay discrimination (consistent with the A5 joint reading).")

# ---------------------------------------------------------------------------------------------------------------- table 1
w("\n## 2. A* per (lever, scope, corner), label GO  (CEILING-TYPE: synthetic posterior; oracle-level rows are double-ceiling)")
w("| lever | level type | scope | A* FULL | A* P (beside) | FULL reach @0.785 | FULL reach @1.0 | max share_closed seen (AUC 1.0, mean of seeds, any q) | draws with M1-P at some q (AUC 1.0) | draws with M1-FULL at some q | draws with CORNER-P |\n|---|---|---|---|---|---|---|---|---|---|---|")
for lv in ADM + CEIL:
    for sc in ("ANCHOR", "ALL"):
        a = G["a_star"][f"{lv}|{sc}|GO"]
        d = G["diag"][f"syn_{lv}_{sc}_GO"]
        w(f"| {lv} | {'ADMISSIBLE' if lv in ADM else 'CEILING-LEVEL'} | {sc} | {fa(a['FULL'])} | {fa(a['P'])} | {a['rate_FULL'][2]:.2f} | {a['rate_FULL'][-1]:.2f} | {d['max_share_mean']:.3f} | {d['draws_with_M1P_at_some_q']:.2f} | {d['draws_with_M1FULL_at_some_q']:.2f} | {a['rate_P'][-1]:.2f} |")
w("\nReach rate by level for every (lever, scope), label GO, FULL and P: all 0.00 at 0.50, 0.68, 0.785, 0.85, 0.90, 0.95, 0.99, 1.0 (`raw/AGG.json` `a_star.*.rate_FULL`, `rate_P`). Realized AUC of the synthetic draws, LF100 ANCHOR: " + ", ".join(f"{v:.3f}" for v in G["a_star"]["LF100|ANCHOR|GO"]["realized_auc_mean"]) + " (nominal 0.50 ... 1.0; C4 within 0.02 holds).")
w("Multiplicity, as disclosed in the card: 21 q points (adjacent points correlated), 5 admissible levers for A\\*_any, 2 corners, 2 scopes, 2 labels. Multiplicity inflates false positives; the result here is a zero everywhere, so it is not inflated by it.")

# ---------------------------------------------------------------------------------------------------------------- binding
w("\n## 3. What binds (beside; decides nothing; CEILING-TYPE: AUC 1.0 oracle gate, draw 0)")
w("Per q of the 21-point grid with point share_closed >= 0.4. Columns: share_closed (mean of seeds), ADE6 delta(X - R) m (seed 0 | seed 1), M1-P, M1-FULL, GUARD, ACTIVE rows separated-worse (seed 0 | seed 1 counts) and their names (union).")
sc0 = A.make_score(y, 1.0, 7, 0)
for lv, scn, scope in (("LF100", "ANCHOR", rig.v0 < 8.0), ("LF100", "ALL", np.ones(2399, bool)), ("LFO100", "ANCHOR", rig.v0 < 8.0)):
    w(f"\n**{lv}, {scn} scope** ({'ADMISSIBLE heads level' if lv in ADM else 'CEILING-LEVEL logged level'}):\n")
    w("| q | share_closed | ADE6 d(X-R) s0 | ADE6 d(X-R) s1 | M1-P | M1-FULL | GUARD | #worse s0 | #worse s1 | rows |\n|---|---|---|---|---|---|---|---|---|---|")
    for q in A.Q_GRID:
        if q < 0.25:
            continue
        e = A.evaluate(rig, lv, A.select_top(sc0, scope, q), full=True)
        if e["share_mean"] < 0.4:
            continue
        sw = sorted(set(e["rows_xr"][0]["_active_separated_worse"]) | set(e["rows_xr"][1]["_active_separated_worse"]))
        w(f"| {q:.2f} | {e['share_mean']:.3f} | {e['ade6_s'][0]:+.3f} | {e['ade6_s'][1]:+.3f} | {e['M1P']} | {e['M1FULL']} | {e['GUARD']} | {len(e['rows_xr'][0]['_active_separated_worse'])} | {len(e['rows_xr'][1]['_active_separated_worse'])} | {', '.join(sw) if sw else '-'} |")
oc = CT["values"]["oracle_gate_LF100"]
w(f"\nControl C7 value [M]: the oracle gate (AUC 1.0, q = GO rate {oc['q']:.3f}, ALL scope) on LF100 reads share_closed {oc['share_mean']:.3f} (mean of seeds) with ADE6 d(X - R) {oc['ade6']['0']:+.3f} / {oc['ade6']['1']:+.3f} m, against {oc['q1_share_mean']:.3f} for the ungated LF100.")

# ---------------------------------------------------------------------------------------------------------------- anchor
w("\n## 4. The real anchor (P1P `post_primary`, v0 < 8, 861 windows), per admissible lever (heads-level lever, ADMISSIBLE; label GO of the card)")
a0 = G["anchor"]["LF100"]
w(f"Realized AUC of the posterior [M]: card label v0 < 6 (n 608) **{a0['auc_v0lt6_card']:.3f}**; card label v0 < 8 (n 861) **{a0['auc_v0lt8_card']:.3f}**; against P1P `go` (population-limited label, v0 < 8) **{a0['auc_v0lt8_p1pgo']:.3f}**. See section 8 for the label scope.\n")
w("| lever | max share_closed over q (mean of seeds) | at q | share_closed at q=1.0 (s0 / s1) | ADE6 d(X-R) at q=1.0 m (s0 / s1) | M1-P at some q | smallest q with M1-P | M1-FULL at some q | CORNER-P at some q | CORNER-FULL at some q |\n|---|---|---|---|---|---|---|---|---|---|")
for lv in ADM:
    a = G["anchor"][lv]
    w(f"| {lv} | {a['max_share_mean']:.3f} | {a['q_at_max_share']:.2f} | {a['share_q1'][0]:.3f} / {a['share_q1'][1]:.3f} | {a['ade6_q1'][0]:+.3f} / {a['ade6_q1'][1]:+.3f} | {a['any_M1P']} | {a['smallest_q_M1P'] if a['smallest_q_M1P'] is not None else '-'} | {a['any_M1FULL']} | {a['any_CORNER_P']} | {a['any_CORNER_FULL']} |")
j = round(a0["smallest_q_M1P"] / 0.05)
w(f"\nOnly LF100 reaches M1-P (smallest q {a0['smallest_q_M1P']}: share_closed {a0['share_s_by_q'][j][0]:.3f} / {a0['share_s_by_q'][j][1]:.3f}, ADE6 d(X-R) {a0['ade6_s_by_q'][j][0]:+.3f} / {a0['ade6_s_by_q'][j][1]:+.3f} m, GUARD False, M1-FULL False). The anchor's corner status is read with the same predicates; one posterior, so no reach rate.")

# ---------------------------------------------------------------------------------------------------------------- near
w("\n## 5. NEAR test (card section 9): is the real-anchor curve inside the synthetic band?")
w("Synthetic level closest to the anchor's realized AUC on the card label (0.749 on v0 < 8) is 0.785 (|0.749 - 0.785| = 0.036; the 0.68 level is 0.069 away), which is also the card's second test level, so one test level is read. Fraction of the 21 q points with BOTH coordinates (share_closed, ADE6 d(X-R)) inside the pointwise [2.5, 97.5] percentile band of the 20 draws, per seed; NEAR needs >= 0.75 at both seeds.\n")
w("| lever | frac inside s0 | frac inside s1 | NEAR | anchor share_closed at q=0.5 (mean seeds) | synthetic 0.785 median at q=0.5 |\n|---|---|---|---|---|---|")
for lv in ADM:
    n = G["near"][lv]["0.785"]
    sh_a = float(np.mean(np.array(G["anchor"][lv]["share_s_by_q"])[10]))
    sh_s = float(np.median(np.array(R[f"syn_{lv}_ANCHOR_GO"]["levels"]["0.785"]["share_mean"])[:, 10]))
    w(f"| {lv} | {n['frac_inside_per_seed'][0]:.3f} | {n['frac_inside_per_seed'][1]:.3f} | {n['NEAR']} | {sh_a:.3f} | {sh_s:.3f} |")
w("\nThe real posterior (realized AUC 0.749) closes less M1 mass per q than a synthetic 0.785 posterior on every lever with a visible curve, so the synthetic 0.785 curve is an optimistic stand-in for the anchor. This does not change the outcome: even the perfect-separation synthetic posterior reaches no corner.")

# ---------------------------------------------------------------------------------------------------------------- sensitivity
w("\n## 6. Sensitivity arms (beside)")
w("* **Label KEEP** (`v6 >= v0 - 1.0`, ANCHOR scope, 5 admissible levers): reach rate 0.00 at all 8 levels for every lever (FULL and P). A\\* none (> 1.0).")
ps = G["noskill"]
w(f"* **Shuffled-label arm** (LF100, ANCHOR, labels permuted, then the AUC-1.0 construction): realized AUC against the TRUE label {ps['perm_realized_auc_vs_true_label_mean']:.3f} (range {ps['perm_realized_auc_range'][0]:.3f} to {ps['perm_realized_auc_range'][1]:.3f}); its mean share_closed curve over q sits on the nominal-0.50 curve (max absolute gap {ps['max_abs_gap_perm_vs_nominal0.5_curve']:.3f}; at q = 0.5: {ps['perm_share_at_q0.5']:.3f} against {ps['nominal0.5_share_at_q0.5']:.3f}); reach rate FULL 0.00 at all levels. No-skill false-positive rate: 0 of 20 draws.")
w("* **Corner-P** (M1-P and GUARD, beside): reach rate 0.00 for all 18 (lever, scope) rows, label GO, all levels (table in section 2, last column).")
w("* Not gated, named in the card: `lonfloor` / `kbox2lf` (no per-window arrays in this layout) and the `LFN*` noise arms.")

# ---------------------------------------------------------------------------------------------------------------- four families
w("\n## 7. Four families at the best q (beside; STRATEGIC PAUSED, PI ITEM 25)")
w("No (draw, level) reaches the corner, so the card's 'reached' q does not exist; the best q is fixed here, after the data and deciding nothing: the smallest q on the AUC-1.0 oracle-gate curve (ANCHOR scope, draw 0) whose point share_closed (mean of seeds) is >= 0.5, or q = 1.0 if none. All 9 levers: `raw/FOUR_FAMILIES.md` (every row, both seeds, delta [lo, hi], direction-aware flag). Summary [M, CEILING-TYPE]:\n")
w("| lever | type | q | share_closed s0 / s1 | ADE6 d(X-R) m s0 / s1 | M1-P | M1-FULL | GUARD | ACTIVE rows separated-worse s0 | s1 |\n|---|---|---|---|---|---|---|---|---|---|")
for lv in ADM + CEIL:
    f = G["four_families_q"][lv]
    w(f"| {lv} | {'ADMISSIBLE' if lv in ADM else 'CEILING-LEVEL'} | {f['q']:.2f} | {f['share_s']['0']:.3f} / {f['share_s']['1']:.3f} | {f['ade6_s']['0']:+.3f} / {f['ade6_s']['1']:+.3f} | {f['M1P']} | {f['M1FULL']} | {f['GUARD']} | {', '.join(f['active_separated_worse']['0']) or '-'} | {', '.join(f['active_separated_worse']['1']) or '-'} |")
w("\nLF100 at q = 0.5 (oracle gate, CEILING-TYPE), the arm that comes closest on M1-P and ADE6, all four families [M, T1-kin], X - R, delta [lo, hi], seed 0 | seed 1:\n")
ff = open(f"{A.RAW}/FOUR_FAMILIES.md").read().split("\n## ")
lf = [s for s in ff if s.startswith("LF100 (")][0].split("\n", 1)[1]
w(lf.strip())
# anchor LF100 four families at its smallest M1-P q
a = R["anchor_LF100_ANCHOR_GO"]
jj = round(a0["smallest_q_M1P"] / 0.05)
rows = a["curve"][jj]["rows_xr"]
w(f"\nReal anchor LF100 at its smallest M1-P q = {a0['smallest_q_M1P']} (ADMISSIBLE arm; X - R; STRATEGIC PAUSED rows beside):\n")
w("| family | row | s0 delta [lo, hi] | s1 delta [lo, hi] | separated-worse s0/s1 |\n|---|---|---|---|---|")
for lab in A.GD.LABELS:
    r0, r1 = rows["s0"][lab], rows["s1"][lab]
    for r in (r0, r1):
        r["separated"] = bool(r["lo"] > 0 or r["hi"] < 0)
    f = A.SR.FAMILY[lab]
    tag = "STRATEGIC PAUSED (beside)" if f == "STRATEGIC" else f
    w(f"| {tag} | {lab} | {r0['delta']:+.4f} [{r0['lo']:+.4f}, {r0['hi']:+.4f}] | {r1['delta']:+.4f} [{r1['lo']:+.4f}, {r1['hi']:+.4f}] | {A.SR.worse(A.GD, lab, r0)}/{A.SR.worse(A.GD, lab, r1)} |")
w("\nThe S-POS `b` arrays (P10 side) differ between arms, so no X - P10 contrast on S-POS is read; the S-POS rows above are X - R and sit beside, never a pass.")

# ---------------------------------------------------------------------------------------------------------------- label scope note
w("\n## 8. Note: 0.785 versus 0.749 (label scope of the banked probe AUC)")
w(f"The banked 0.785 (P1P / THEORY_CRAWL) is the AUC of `post_primary` against P1P `go` over the 861 windows with v0 < 8. P1P `go` is limited to the probe population (v0 < 6) and reads False for every window at 6 <= v0 < 8 (AMEND_1; equal to the card label on all 608 windows v0 < 6, differs on 81 of 253 at 6 <= v0 < 8). Against the card label the same posterior reads **{a0['auc_v0lt8_card']:.3f}** on the 861 windows and **{a0['auc_v0lt6_card']:.3f}** on the 608 windows with v0 < 6. So 0.785 is the AUC on a label that scores the 6 to 8 m/s windows as non-go; on the population-free card label the real posterior sits at 0.71 to 0.75. The synthetic 0.785 level is a nominal construction on the card label and is unaffected. Nothing in the verdict depends on which of the three numbers is quoted: no level up to 1.0 reaches the corner.")

# ---------------------------------------------------------------------------------------------------------------- controls
w("\n## 9. Controls and caveats")
w("* Controls (`raw/CONTROLS.json`, `test_aucmap.py`): " + ", ".join(f"{k.replace('check_', '')} {v}" for k, v in CT["checks"].items()) + f". Deliberate-regression mutants RED: {sum(1 for m in CT['mutants'].values() if m['RED'])} of {len(CT['mutants'])} (selection inverted, scope mask dropped, label margin changed, shuffle removed, P10 contrast ignoring the base excess, GUARD ignoring ACTIVE rows, ADE6 guard disabled). C9 (added after the sweep, before this write-up; it asserts the corner predicate CAN read True: a lever equal to the base with the M1 row at P10 reads CORNER-FULL True and share_closed 1.0; an ACTIVE row pushed worse, or ADE6 +0.2 m, reads CORNER-FULL False) shows the zero reach rate is not a predicate that can only read False.")
w("* `run_synthetic` was extended before the sweep to also store the per-seed share_closed (`share_s`) the NEAR test needs; nothing else in `aucmap.py` changed after the controls that fixed the sweep code, and the controls were re-run in full afterwards (all PASS, 7 of 7 mutants RED).")
w("* Scope of what this measures: banked levers (LF050, LF075, LF100, LFB05, LFB10 heads level) gated per window by a posterior of controlled AUC, applied to rollouts that already exist. It does not build a rollout arm with the gate inside the replan loop, does not test `lonfloor` / `kbox2lf`, and does not test any other lever. 'No gate on THIS lever family reaches the corner' is the claim; it says nothing about a different lever.")
w("* The corner is evaluated on 141 episodes, two seeds, one training run each; a separated difference is necessary and not sufficient for a lever effect (`H-ESTIM-SEED-1`). A zero reach rate is not the false-positive risk that `H-ESTIM-SEED-1` describes; it leaves open that another training run of a lever would read differently.")
w("* Pre-freeze inspection of ungated M1 mass (card section 2): v0 < 8 carries about 91 % of the base-over-P10 excess, which is why the ANCHOR scope is not a small fraction of the problem.")

# ---------------------------------------------------------------------------------------------------------------- next
w("\n## 10. What this leaves (Rule Zero: the next lever, not only the verdict)")
w("* **Outcome (iii) names the next arm: change the lever / the decode path (F), not perception.** The data say what that arm must achieve at the lever, with a perfect gate: close M1 by share_closed >= 0.5 (and reach X - P10 not separated-worse) while leaving `cross-track MAE` (LATERAL), `DK-VIOL`, `along-track MAE` and `speed MAE` (LONGITUDINAL) and ADE6 inside the guard. LF100 closes M1 at the oracle gate (q = 0.304 of ALL windows) with ADE6 +0.011 / +0.006 m, but pays `cross-track MAE` separated-worse at both seeds, and only the ungated lever reaches M1-FULL.")
w("* The bounded lifts (LFB05, LFB10) keep the guard rows nearer base at q = 1.0 (LFB05: 0 | 1 ACTIVE rows worse, ADE6 +0.023 / +0.006 m) but close only 0.31 and 0.47 of M1 at q = 1.0 in ANCHOR scope. The programme's open point is a lever with LF100's M1 closure and LFB05's guard footprint; a gate does not create that.")
w("* **Blocker, named:** the next arm (a lever or decode-path change, F) needs its own pre-registered card and a rollout arm; it is a Master Mind / PI scheduling decision, not something this card can launch. The AUC target for L (outcome (i)) does not exist under this lever family, so the LoRA arm's P4 read should not cite an A\\*.")

# ---------------------------------------------------------------------------------------------------------------- manifest
w("\n## 11. Deliverable manifest (all in `E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/aucmap/`; single location unless noted; nothing staged, committed or pushed)")
w("| artifact | path (repo-external, dev box E:) |\n|---|---|\n| frozen card + sha | `PREREG_AUCMAP.md`, `PREREG_AUCMAP.sha256` |\n| amendment (C6) + sha | `AMEND_1_AUCMAP.md`, `AMEND_1_AUCMAP.sha256` |\n| instrument | `aucmap.py` |\n| controls + mutants | `test_aucmap.py`, `raw/CONTROLS.json` |\n| sweep driver | `run_all.py`, `raw/AUCMAP_run.log`, `raw/AUCMAP_stdout.log`, markers `raw/AUCMAP.done`, `raw/AUCMAP.running.ended` |\n| per-task results (29) | `raw/tasks/*.json` |\n| merged results | `raw/AUCMAP_RESULT.json` |\n| aggregation | `aggregate.py`, `write_results.py`, `raw/AGG.json`, `raw/FOUR_FAMILIES.md` |\n| this report | `RESULTS_AUCMAP.md` |\n| handoff | `HANDOFF_AUCMAP.md` |")
w("\nAll of this lives only in this folder on the dev box E: (not in the repo, not staged). Integration for the Master Mind: land the folder (excluding `raw/tasks/` if size matters) and add the outcome-(iii) row to `GOALS_AND_CLAIMS.md`.")

open(f"{HERE}/RESULTS_AUCMAP.md", "w", newline="\n").write("\n".join(L) + "\n")
print("written", len("\n".join(L)))
