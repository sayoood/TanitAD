#!/usr/bin/env python3
"""perc2plan_result_md.py -- render RESULT.md from PERC2PLAN.json (+ SMOKE_CONTROLS.json). Reads numbers, decides nothing:
every verdict cell is copied from the JSON the analysis wrote by the pre-registered rule.

    python perc2plan_result_md.py --json PERC2PLAN.json --smoke SMOKE_CONTROLS.json --out RESULT.md
"""
import argparse
import json
import time

ARMS = ("AGT_ZERO", "AGT_SHUF", "BEV_ZERO", "BEV_SHUF", "BOTH_ZERO")


def ci(r):
    if not isinstance(r, dict) or "delta" not in r:
        return "n/a"
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]{' **sep**' if r.get('separated') else ''}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.load(open(a.json, encoding="utf-8"))
    sm = json.load(open(a.smoke, encoding="utf-8")) if a.smoke else None
    L = ["# PERC2PLAN-R8 — does refcv8's planner USE its perception inputs?", "",
         f"* Rendered {time.strftime('%Y-%m-%d %H:%M:%S %z')} from `{a.json}`; prereg `PREREG_PERC2PLAN.md` sha256 `{r['prereg_sha256']}`.",
         f"* **Tier: {r['tier']}.**",
         f"* **Interval:** {r['interval']}.",
         "* Model: refcv8 EMA-48,852, per-frame nav feed, eval139 (4,754 windows / 139 episodes), os-only arm set, inference seeds 0 and 1, Thor.",
         ""]
    L += ["## Verdicts (mechanical, PREREG §4)", "",
          "| arm | headline | S2 verdict | S6 verdict | S2 ADE Δ s0 | S2 ADE Δ s1 | S6 ADE Δ s0 | S6 ADE Δ s1 |", "|---|---|---|---|---|---|---|---|"]
    for x in ARMS:
        v = r["verdicts"].get(x)
        if not v:
            L.append(f"| {x} | NOT RUN | | | | | | |")
            continue
        ps = v["per_surface"]
        cells = []
        for surf in ("S2", "S6"):
            for s in ("s0", "s1"):
                row = ps.get(surf, {}).get("rows", {}).get(s)
                cells.append("n/a" if not row else f"{row['ade_delta']:+.4f} [{row['ade_lo']:+.4f}, {row['ade_hi']:+.4f}]")
        L.append(f"| {x} | **{v['headline']}** | {ps.get('S2', {}).get('verdict')} | {ps.get('S6', {}).get('verdict')} | "
                 + " | ".join(cells) + " |")
    L += ["", "ADE Δ = arm − BASE at the same inference seed, metres, 95 % paired episode-cluster CI.", ""]
    L += ["### Inference-seed floor (BASE_s1 − BASE_s0)", ""]
    for surf, f in r.get("floor", {}).items():
        if "floor_m" in f:
            L.append(f"* {surf}: floor = **{f['floor_m']:.4f} m** (block {ci(f['ade_block'])}); seed plan displacement "
                     f"mean {f['seed_displacement']['mean_m']} m, p95 {f['seed_displacement']['p95_m']} m")
        else:
            L.append(f"* {surf}: {f}")
    L += ["", "### Separated components at BOTH seeds", ""]
    for x in ARMS:
        v = r["verdicts"].get(x)
        if not v:
            continue
        for surf in ("S2", "S6"):
            p = v["per_surface"].get(surf, {})
            L.append(f"* {x} {surf}: worse {p.get('separated_worse_both_seeds')} · better {p.get('separated_better_both_seeds')}")
    L += ["", "## All four families, per seed (arm − BASE)", ""]
    for sk in sorted(r["seeds"]):
        blk = r["seeds"][sk]
        L += [f"### {sk} — n_windows S2 {blk['n_windows_S2']}, S6 {blk['n_windows_S6']}, episodes {blk['n_episodes']}", ""]
        for surf in ("S2", "S6"):
            pairs = {k.split("|", 1)[1]: v for k, v in blk["pairs"].items() if k.startswith(surf + "|") and "BANKED" not in k}
            if not pairs:
                continue
            mets = [(fam, m) for fam, b in next(iter(pairs.values()))["families"].items() for m in b]
            L += [f"**{surf}**", "", "| family.metric | BASE mean | " + " | ".join(pairs) + " |",
                  "|---|---|" + "---|" * len(pairs)]
            for fam, m in mets:
                base = blk["means"][surf].get("BASE", {}).get(m)
                L.append(f"| {fam}.{m} | {base} | " + " | ".join(ci(pairs[x]["families"][fam].get(m)) for x in pairs) + " |")
            L.append("")
        L += ["Plan displacement vs BASE (mean / p95, m) · sel_idx changed · tac_decoder_v6 argmax changed:", ""]
        for k, d in blk["displacement"].items():
            L.append(f"* {k}: {d['mean_m']} / {d['p95_m']}")
        L.append(f"* sel_idx changed: {blk['sel_idx_changed']}")
        L.append(f"* tacv6: {blk['tacv6_argmax_changed']}")
        L.append("")
    L += ["**Strategic family:** " + r["strategic_family"]["status"] + f" (n = {r['strategic_family']['n']}) — "
          + r["strategic_family"]["reason"] + ".", ""]
    L += ["## Controls", "", f"* VOID gates all pass: **{r['controls'].get('void_gates_all_pass')}**; failed: {r['controls'].get('void_gates_failed')}",
          f"* C2 (BASE_s1 vs banked dev-box battery s1, ≤ 1e-5): **{r['controls'].get('C2_stamp')}**"]
    for sk in ("s0", "s1"):
        c = r["controls"].get(sk, {})
        if c:
            L.append(f"* {sk}: C2 {json.dumps(c.get('C2_vs_banked'))} · C3 {json.dumps(c.get('C3_vs_fanroll'))} · "
                     f"S6 GT independent max |Δ| {c.get('S6_gt_independent_max_abs_m')}")
            L.append(f"  * C4/C5/C6: " + ", ".join(f"{k}={v}" for k, v in c.items() if k.startswith(("C4_", "C5_", "C6_"))))
    if sm:
        L += ["", "### CPU smoke (dev box, 6 windows — controls only, not a result)", "",
              f"* C1 as registered (bit-exact vs the unmodified child): **{sm['C1_as_registered_bitexact']['verdict']}** — "
              f"max |Δ| os {sm['C1_as_registered_bitexact']['max_abs_os_m']:.3g} m, plan_full {sm['C1_as_registered_bitexact']['max_abs_plan_full_m']:.3g} m, sel_idx identical {sm['C1_as_registered_bitexact']['sel_idx_identical']}",
              f"* Attribution: {sm['C1_attribution']['reading']}",
              f"* Zero is exactly zero at the hook: {sm['zero_is_exactly_zero_at_hook']}"]
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
