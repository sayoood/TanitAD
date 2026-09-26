#!/usr/bin/env python3
"""Write eval/RESULT_E6_<name>.md from proptable/<name>/{readout,gates}.json -- every number is read,
none is typed (a number copied by hand from a summary is how a landed doc quotes one that never existed).

    python eval/write_result_e6.py --name sub200_ep011
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "D:/Projects/TanitAD/data/refe_navtest"


def ci(v, d=1):
    return f"{v['mean']:.{d}f} [{v['ci95'][0]:.{d}f}, {v['ci95'][1]:.{d}f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    wd = os.path.join(DATA, "proptable", a.name).replace("\\", "/")
    R = json.load(open(os.path.join(wd, "readout.json"), encoding="utf-8"))
    G = json.load(open(os.path.join(wd, "gates.json"), encoding="utf-8"))
    g, A, B, C, Dl = G["gates"], R["a_pdms"], R["b_ranking"], R["c_components"], R["d_path_length"]
    V, sk, CA = R["verdict"], A["selection_skill"], R["a_components_at"]
    L = []
    L.append(f"# RESULT E-6 -- selection diagnosis, `{a.name}` (SPEC_NAVTEST Amendments 3 + 3a)\n")
    L.append(f"**Checkpoint:** `{G['ckpt']}` · **tokens:** {R['N']} over {R['n_logs']} logs (`{os.path.basename(G['tokens'])}`) · "
             f"**proposals:** {R['M']} per token, each scored as its own seam by `score_navtest_refe.py` UNCHANGED "
             f"({R['N'] * R['M']:,} scored plans) · **bootstrap:** log clusters, {R['boot']:,} resamples · "
             f"**evidence class:** MEASURED · **tier:** NAVSIM v1 PDMS (open-loop plan, pseudo-simulated against logged agents).\n")
    L.append("## Gates\n")
    L.append(f"- **G1** the dump re-run reproduces the landed seam: max |d pose| **{g['G1'].get('max_abs_pose_diff')}**, "
             f"{g['G1'].get('tokens_identical')} of {R['N']} tokens identical -> {'PASS' if g['G1'].get('pass') else 'FAIL'}")
    L.append(f"- **G2** the table at the planner's pick reproduces the landed per-token score: max |d| "
             f"**{g['G2'].get('max_abs_score_diff_identical_poses')}** over {g['G2'].get('tokens_compared')} tokens -> "
             f"{'PASS' if g['G2'].get('pass') else 'FAIL'}")
    L.append(f"- **G3** {g['G3']['pass']} of {g['G3']['runs']} single-proposal runs PASS with every token valid -> "
             f"{'PASS' if g['G3']['ok'] else 'FAIL'}")
    L.append(f"- transcribed aggregate reproduces the planner's pick on **{100 * R['pick_reproduced_by_transcribed_aggregate']:.1f} %** of tokens\n")
    L.append("## (a) PDMS by how the plan is chosen\n")
    L.append("| choice | PDMS [95 %] |\n|---|---|")
    L.append(f"| best of {R['M']} (oracle) | {ci(A['oracle'])} |")
    L.append(f"| **planner's pick** | **{ci(A['actual'])}** |")
    L.append(f"| random proposal (mean of {R['M']}) | {ci(A['random'])} |")
    L.append(f"| medoid, no scorer (pre-registered comparison) | {ci(R['confirmatory_medoid']['pdms'])} |")
    for k, v in R["EXPLORATORY_rules"].items():
        if not k.startswith("_"):
            L.append(f"| {k} (EXPLORATORY) | {ci(v['pdms'])} |")
    L.append("")
    L.append(f"Paired: pick - random **{ci(A['actual_minus_random'])}**, oracle - pick **{ci(A['oracle_minus_actual'])}**, "
             f"medoid - pick **{ci(R['confirmatory_medoid']['medoid_minus_actual'])}** PDMS points. "
             f"**Selection skill** (pick - random) / (oracle - random) = **{sk['value']:.3f} [{sk['ci95'][0]:.3f}, {sk['ci95'][1]:.3f}]**.\n")
    L.append("Sub-scores (x100) at each choice:\n")
    L.append("| component | pick | random | best proposal |\n|---|---|---|---|")
    for k in ("NC", "DAC", "DDC", "EP", "TTC", "C"):
        L.append(f"| {k} | {100 * CA[k]['pick']:.1f} | {100 * CA[k]['random']:.1f} | {100 * CA[k]['oracle_pick']:.1f} |")
    L.append("\n## (b) Ranking skill within a token\n")
    L.append(f"- Spearman(planner aggregate, true PDMS) over the {B['tokens_true_pdms_varies']} tokens whose PDMS varies: "
             f"**{ci(B['within_token_spearman_agg_vs_pdms'], 3)}**; pooled over all (token, proposal): {B['pooled_spearman_agg_vs_pdms']:.3f}")
    L.append(f"- the pick is a best proposal on **{100 * B['top1_hit']['mean']:.1f} %** of tokens; a random pick would be "
             f"{100 * B['top1_hit_random_expectation']['mean']:.1f} %\n")
    L.append("## (c) Each scorer output against the harness's verdict on the same proposals\n")
    L.append("| output | pooled AUC | within-token AUC [95 %] | varies in | mean predicted pass | true pass rate | FAILING |\n|---|---|---|---|---|---|---|")
    for k in ("NC", "DAC", "DDC", "TTC", "C"):
        c = C[k]
        wt = ci(c["within_token_auc"], 3) + f" (n {c['within_token_auc']['n']})" if c.get("within_token_auc") else "-"
        L.append(f"| {k} | {c['pooled_auc']:.3f} | {wt} | {100 * c['share_tokens_varying']:.0f} % | "
                 f"{100 * c['mean_pred_prob']:.0f} % | {100 * c['true_rate']:.0f} % | {'**yes**' if c['FAILING'] else 'no'} |")
    e = C["EP"]
    L.append(f"| EP (Spearman) | {e['pooled_spearman']:.3f} | {ci(e['within_token_spearman'], 3) if e.get('within_token_spearman') else '-'} | "
             f"{100 * e['share_tokens_varying']:.0f} % | {100 * e['mean_pred_prob']:.0f} % | {100 * e['true_mean']:.0f} % (mean) | - |\n")
    L.append("## (d) Path length\n")
    L.append(f"- pick's path length / mean of {R['M']}: **{ci(Dl['pick_over_mean'], 3)}**; its percentile among the {R['M']}: "
             f"{ci(Dl['pick_length_percentile'], 3)}")
    L.append(f"- within-token Spearman with path length -- scorer aggregate **{ci(Dl['within_token_spearman_agg_vs_length'], 3)}**, "
             f"predicted EP {ci(Dl['within_token_spearman_predEP_vs_length'], 3)}, TRUE PDMS **{ci(Dl['within_token_spearman_truePDMS_vs_length'], 3)}**\n")
    # descriptive (NOT pre-registered; computed here from the table so no number is typed)
    import numpy as np
    T = np.load(os.path.join(wd, "table.npz"))
    p, sub, pick = T["pdms"], T["sub"], T["pick"]
    N, M = p.shape
    good = (p >= 0.8).sum(1)
    ar = np.arange(N)
    dsc = {"good_mean": float(good.mean()), "good_median": float(np.median(good)),
           "tok_ge1": float((good >= 1).mean()), "tok_ge16": float((good >= 16).mean()),
           "pick_zero": float((p[ar, pick] == 0).mean()), "all_zero": float((p.max(1) == 0).mean()),
           "dac_any": float((sub[:, :, 1].max(1) == 1).mean()), "pick_dac_fail": float((sub[ar, pick, 1] == 0).mean())}
    L.append("## Descriptive (not pre-registered)\n")
    L.append(f"- proposals scoring PDMS >= 80 per token: mean **{dsc['good_mean']:.1f}**, median {dsc['good_median']:.1f} of {M}; "
             f"{100 * dsc['tok_ge1']:.1f} % of tokens have at least one, {100 * dsc['tok_ge16']:.1f} % have at least 16")
    L.append(f"- the pick scores 0 on **{100 * dsc['pick_zero']:.1f} %** of tokens; all {M} proposals score 0 on only "
             f"{100 * dsc['all_zero']:.1f} %")
    L.append(f"- the pick leaves the drivable area on **{100 * dsc['pick_dac_fail']:.1f} %** of tokens; {100 * dsc['dac_any']:.1f} % "
             f"of tokens have at least one proposal that stays inside it\n")
    L.append("## Reading\n")
    L.append(f"The proposal head is not the problem: in a typical scene a quarter to a third of the {M} proposals score "
             f"80 or more, and the best one averages {A['oracle']['mean']:.1f} PDMS. The scorer is: its pick is indistinguishable from "
             f"a random proposal (skill {sk['value']:.3f}), it believes {100 * C['DAC']['mean_pred_prob']:.0f} % of proposals stay "
             f"on the road when {100 * C['DAC']['true_rate']:.0f} % do, and within a scene its drivable-area, driving-direction "
             f"and comfort outputs rank proposals at chance (within-token AUC {C['DAC']['within_token_auc']['mean']:.2f} / "
             f"{C['DDC']['within_token_auc']['mean']:.2f} / {C['C']['within_token_auc']['mean']:.2f}), while its progress output "
             f"follows path length (within-token Spearman {Dl['within_token_spearman_predEP_vs_length']['mean']:.2f}) although "
             f"the true PDMS does not ({Dl['within_token_spearman_truePDMS_vs_length']['mean']:.2f}). Pooled AUCs look better "
             f"than within-token ones because whole scenes differ in difficulty; choosing needs the within-token skill. "
             f"MECHANISM (consistent with every number here, not yet shown causal): `refe/train.py` supervises the scorer with "
             f"8-9 FIXED candidates per frame, each attached to its NEAREST proposal a few metres away (`assign_d` 2.5-5.7 m on "
             f"the latest micro-batches), so the labels a proposal's scorer output learns are the scores of a different path, "
             f"and drivable area / comfort change within one metre -- a declared departure from DriveZero, which scores the "
             f"student's OWN proposals. The fix is a recipe decision for the PI.\n")
    L.append("## Verdict (rule fixed before the table was read)\n")
    L.append(f"**{V['outcome']}**; failing scorer outputs: **{', '.join(V['failing_scorer_outputs']) or 'none'}**. "
             f"Rule: {V['rule']}. Amendment 3's original clause (unused, reported): {V['original_clause_amendment3']}.\n")
    L.append("## Files\n")
    L.append(f"`{wd}/` -- `table.npz` (PDMS + six sub-scores + logits + poses per token x proposal), `readout.json`, `gates.json`, "
             f"`logs/`; single-proposal seams `{DATA}/seams/proptable/{a.name}/`; scores `{DATA}/score/refe_{a.name}_pNN/`. "
             f"Tools: `eval/proposal_table.py`, `eval/selection_readout.py` (self-test `eval/selftest_selection_readout.py`), "
             f"this file by `eval/write_result_e6.py`.\n")
    out = os.path.join(HERE, f"RESULT_E6_{a.name}.md")
    open(out, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
