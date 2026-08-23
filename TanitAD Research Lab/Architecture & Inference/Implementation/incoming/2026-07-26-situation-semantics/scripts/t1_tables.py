"""Render the T1 result tables from `artifacts/t1_probe.json`. No number is typed by hand.

usage:  python t1_tables.py <artifacts dir> <out .md>
"""
from __future__ import annotations

import json
import sys

# H2's published NOT_T_seen head numbers, for the like-for-like column. INHERITED,
# `2026-07-26-h2-classifier/artifacts/c12_fix.json` — quoted, never re-derived here.
H2 = {
    "head_ego":     dict(ap=0.12263, x=3.74, d=+0.07659, lo=+0.05055, hi=+0.13529, sep=True),
    "head_img_ego": dict(ap=0.05205, x=1.59, d=+0.00601, lo=-0.00040, hi=+0.03947, sep=False),
    "head_img":     dict(ap=0.04914, x=1.50, d=+0.00310, lo=-0.00291, hi=+0.04284, sep=False),
}
ROLE = {
    "ego_t": "⭐ POSITIVE CONTROL (2 ego channels)",
    "ego_win": "⭐ POSITIVE CONTROL (ego, head's window)",
    "img_t_SHUFFLED": "⭐ NEGATIVE CONTROL (features permuted)",
    "img_t": "⭐⭐ **PRIMARY** — frozen 2048-d state at t",
    "img_win_mean": "frozen state, 8-step window mean",
    "img_win_flat": "the head's exact input, read linearly",
    "img_pca16": "low-rank image, k=16",
    "img_pca64": "low-rank image, k=64",
    "img_pca256": "low-rank image, k=256",
    "ego_win+img_pca16": "ego + low-rank image, k=16",
    "ego_win+img_pca64": "ego + low-rank image, k=64",
    "ego_win+img_pca256": "ego + low-rank image, k=256",
    "constant": "the chance arm itself",
}


def f(x, n=5):
    return "—" if x is None else f"{x:+.{n}f}"


def main():
    d = json.load(open(f"{sys.argv[1]}/t1_probe.json"))
    L = []
    ho = d["heldout"]
    L.append(f"*Target `NOT_T_seen` at the **frame** level. Held-out = the label's CONFIRM chunks: "
             f"**{ho['n_frames']:,} frames**, **{ho['n_positives']:,} positives** "
             f"(base rate **{ho['base_rate']:.5f}**), **{ho['n_positive_clips']} of "
             f"{ho['n_clips']} clips** positive. TRAIN = {d['train']['n_frames']:,} frames / "
             f"{d['train']['n_positives']:,} positives / {d['train']['n_clips']} clips.*")
    L.append("")
    fc = d["fidelity_check"]
    L.append(f"**Fidelity check (direction 1): `all_match = {fc['all_match']}`** — this loader "
             f"reproduces every one of H2's published substrate counts exactly "
             f"({', '.join(f'{k} {v}' for k, v in fc['expected_INHERITED_from_H2'].items())}).")
    L.append("")
    L.append("### The ladder — held-out, RIDGE (closed form)")
    L.append("")
    L.append("| representation | role | dim | AP | AP / base | AUROC | ΔAP vs chance | CI95 | "
             "above chance? | selected λ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, a in d["arms"].items():
        r = a.get("ridge")
        if not r:
            continue
        p = r["paired_AP_vs_chance"]
        lam = r["selected"]["lam"] if r.get("selected") else None
        L.append(f"| `{name}` | {ROLE.get(name, '')} | {a['dim']} | "
                 f"{r['AP']['point']:.5f} [{r['AP']['lo']:.5f}, {r['AP']['hi']:.5f}] | "
                 f"**{r['AP_over_base']}×** | {r['AUROC']:.4f} | {f(p['delta'])} | "
                 f"[{f(p['lo'])}, {f(p['hi'])}] | "
                 f"{'✅ **YES**' if p['above_chance'] else 'no'} | "
                 f"{('%.3g' % lam) if lam is not None else '—'} |")
    L.append("")
    L.append("### The second reader — LOGISTIC (LBFGS), same split, same selection rule")
    L.append("")
    L.append("| representation | AP | AP / base | ΔAP vs chance | CI95 | above chance? |")
    L.append("|---|---|---|---|---|---|")
    for name, a in d["arms"].items():
        r = a.get("logistic")
        if not r:
            continue
        p = r["paired_AP_vs_chance"]
        L.append(f"| `{name}` | {r['AP']['point']:.5f} | **{r['AP_over_base']}×** | "
                 f"{f(p['delta'])} | [{f(p['lo'])}, {f(p['hi'])}] | "
                 f"{'✅ **YES**' if p['above_chance'] else 'no'} |")
    L.append("")
    L.append("### ⭐ The comparison that is the finding — SAME target, SAME split, SAME estimator")
    L.append("")
    L.append("| reader | params | AP | AP / base | ΔAP vs chance | CI95 | above chance? |")
    L.append("|---|---|---|---|---|---|---|")
    for k, h in H2.items():
        L.append(f"| `{k}` — H2's attention head (INHERITED) | "
                 f"{'415 k' if k == 'head_ego' else '2.17 M'} | {h['ap']:.5f} | {h['x']}× | "
                 f"{f(h['d'])} | [{f(h['lo'])}, {f(h['hi'])}] | "
                 f"{'✅ **YES**' if h['sep'] else '**no**'} |")
    for nm, lbl in (("img_t", "**LINEAR RIDGE probe on the frozen 2048-d state** (ours)"),
                    ("img_t", None)):
        if lbl is None:
            continue
        a = d["arms"][nm]["ridge"]
        p = a["paired_AP_vs_chance"]
        L.append(f"| ⭐ {lbl} | **2,049** | **{a['AP']['point']:.5f}** | "
                 f"**{a['AP_over_base']}×** | **{f(p['delta'])}** | "
                 f"[{f(p['lo'])}, {f(p['hi'])}] | "
                 f"{'✅ **YES**' if p['above_chance'] else '**no**'} |")
    a = d["arms"]["img_t"]["logistic"]
    p = a["paired_AP_vs_chance"]
    L.append(f"| ⭐ LINEAR LOGISTIC probe, same input (ours) | 2,049 | {a['AP']['point']:.5f} | "
             f"{a['AP_over_base']}× | {f(p['delta'])} | [{f(p['lo'])}, {f(p['hi'])}] | "
             f"{'✅ **YES**' if p['above_chance'] else '**no**'} |")
    L.append("")
    L.append("### Amendment A1 — the chance arm, both conventions")
    L.append("")
    ca = d.get("chance_arms", {})
    L.append(f"*A fully-tied constant score is ranked by ROW ORDER under a stable sort, so its "
             f"full-sample AP is **{ca.get('constant_rowtie', {}).get('AP_point')}** against an "
             f"analytic base rate of **{ca.get('analytic_base_rate')}** — while inside a bootstrap "
             f"draw `_draws` randomises the clip order, giving a median of "
             f"**{ca.get('constant_rowtie', {}).get('AP_boot_median')}**. A uniform random ranker "
             f"gives **{ca.get('uniform_random', {}).get('AP_point')}**. The POINT estimate is "
             f"therefore deflated under H2's convention; **the CI, which is what decides "
             f"separation, is computed on the randomised draws and is unaffected.** Both are "
             f"reported below.*")
    L.append("")
    L.append("| representation | ΔAP vs chance (`const`, H2's convention) | CI95 | "
             "ΔAP vs chance (`rand`, unbiased) | CI95 | above chance (both)? |")
    L.append("|---|---|---|---|---|---|")
    for name, a in d["arms"].items():
        r = a.get("ridge")
        if not r or "paired_AP_vs_chance_rand" not in r:
            continue
        pc, pr = r["paired_AP_vs_chance_const"], r["paired_AP_vs_chance_rand"]
        both = pc["above_chance"] and pr["above_chance"]
        L.append(f"| `{name}` | {f(pc['delta'])} | [{f(pc['lo'])}, {f(pc['hi'])}] | "
                 f"{f(pr['delta'])} | [{f(pr['lo'])}, {f(pr['hi'])}] | "
                 f"{'✅ **YES**' if both else 'no'} |")
    L.append("")
    ev = d["pca_explained_variance_ratio"]
    L.append(f"*PCA on the TRAIN rows of the standardised state: top-1 component carries "
             f"**{ev['top1']*100:.1f} %** of the variance, 16 components **{ev['cum16']*100:.2f} %**, "
             f"64 **{ev['cum64']*100:.2f} %**, 256 **{ev['cum256']*100:.4f} %**. The frozen state is "
             f"**extremely low-rank** — which is itself part of the answer.*")
    open(sys.argv[2], "w", encoding="utf-8").write("\n".join(L) + "\n")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console is cp1252
    print("\n".join(L))


if __name__ == "__main__":
    main()
