"""POST-HOC analyses added AFTER the pre-committed run (NOT in SIM_DESIGN.md; labelled as such in RESULT.md):
  (a) A2-r1 with a margin on the lower CI (threshold t in 1.00..1.25) on the SAME replicates: false-refusal and power;
  (b) A1-r1 power at SMALL planted over-prediction on the small-n_changed geometries (G25, G05), where the lower-CI conjunct is a very sensitive significance test.
Same code, same draws, same generator A, N = 1,000.   python posthoc.py --workdir D:/epadm1/run --workers 12"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ep_admission_sim as S                                                              # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    S.design_guard()
    z = np.load(os.path.join(a.workdir, "replicates.npz"))
    get = lambda n, k: z[f"{n}|{k}".replace("|", "__")]             # noqa: E731
    out = {"what": "POST-HOC (not pre-registered in SIM_DESIGN.md); same replicates / same code", "A2_margin": {}, "small_delta": {}}
    thr = (1.00, 1.05, 1.10, 1.15, 1.20, 1.25)
    for g in ("G78", "G25", "G05"):
        a2 = get(f"{g}|A|CAL", "a2")                                      # [n, seed, head, (point, lo, hi)]
        out["A2_margin"][f"{g}_false_refusal"] = {str(t): S.rate(int((a2[:, :, :, 1] > t).any(axis=(1, 2)).sum()), len(a2)) for t in thr}
    for h in S.HEADS:
        for rho in S.RHOS:
            a2 = get(f"G78|A|PLANT_A2_{h}_{rho}", "a2")[:, :, S.HEADS.index(h), 1]      # [n, seed] lower CI of the planted head
            out["A2_margin"][f"G78_power_{h}_rho{rho}"] = {str(t): S.rate(int((a2 > t).any(axis=1).sum()), len(a2)) for t in thr}
    vec = S.load_vec(a.workdir)
    conds = {}
    for g, ds in (("G25", (0.005, 0.01, 0.02)), ("G05", (0.002, 0.005, 0.01))):
        for d in ds:
            conds[f"{g}|A|PLANT_EP_{d}"] = {"name": f"{g}|A|PLANT_EP_{d}", "gen": "A", "geom": g, "a1": True, "a2": False, "kind": "plant_ep", "kw": {"delta": d},
                                            "vec": {s: S.build_condition(vec[g][s], "plant_ep", delta=d) for s in S.SEEDS}}
    recs = S.run_condition_set(conds, 1000, a.workers)
    for n, c in conds.items():
        s = S.summarize(c, recs[n])
        out["small_delta"][n] = {"A1_r1_any_seed": s["A1"]["r1_any_seed"], "A1_draft_any_seed": s["A1"]["draft_any_seed"], "mean_half_width": s["A1"]["D_M_half_width_mean"], "D_M_point_mean": s["A1"]["D_M_point_mean"]}
    json.dump(out, open(os.path.join(a.workdir, "posthoc.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)

    def pc(r):
        return f"{100 * r['rate']:.1f} % [{100 * r['wilson95'][0]:.1f}, {100 * r['wilson95'][1]:.1f}]"
    lines = ["#### P1. A2-r1 with a margin t on the lower CI (FAIL-A2 iff lower CI > t for any head, any seed); POST-HOC\n", "| t | " + " | ".join(f"FR {g}" for g in ("G78", "G25", "G05")) + " | " + " | ".join(f"power {h} rho {r}" for h in S.HEADS for r in S.RHOS) + " |",
             "|---|" + "---|" * (3 + 6)]
    for t in thr:
        lines.append(f"| {t:.2f} | " + " | ".join(pc(out["A2_margin"][f"{g}_false_refusal"][str(t)]) for g in ("G78", "G25", "G05")) + " | " + " | ".join(pc(out["A2_margin"][f"G78_power_{h}_rho{r}"][str(t)]) for h in S.HEADS for r in S.RHOS) + " |")
    lines += ["", "#### P2. A1-r1 power at small planted over-prediction where the interval is narrow (generator A, N = 1000); POST-HOC\n", "| condition | mean D_M half-width (seed 0) | A1-r1 refuses | A1 DRAFT refuses |", "|---|---|---|---|"]
    for n, v in out["small_delta"].items():
        lines.append(f"| {n} | {v['mean_half_width'][0]:.4f} | {pc(v['A1_r1_any_seed'])} | {pc(v['A1_draft_any_seed'])} |")
    open(os.path.join(a.workdir, "posthoc_tables.md"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))
