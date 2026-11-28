"""Read SH5 grid 5 against PREREG_GRID5.md (literal thresholds). CPU, stdlib only, reads work/diag5/diag_G5_s*.json, writes raw/grid5_cells.json + raw/grid5_tables.txt.
Thresholds are copied from diag_pre3.learner_pass (PASS_F .97, PASS_ALL .95, PASS_R2 .50, COLLAPSE_MAX .80); a separate step cross-checks against diag_pre3.passing_cells."""
import json, os, sys
ROOT = "C:/Users/Admin/par18/sh5diag"
PASS_F, PASS_ALL, PASS_R2, COLL = 0.97, 0.95, 0.50, 0.80
MASS, BETA, TOP1, STEP = 0.8, 16.0, 0.5, 0.03

def lp(rec):
    try:
        ok = rec["auc_F"] >= PASS_F and rec["auc_all_strata"] >= PASS_ALL and rec["r2_reg"] >= PASS_R2
        for a in ("ablate_zero_clear", "ablate_shuffle_clear"):
            ok = ok and rec[a]["auc_all_strata"] <= COLL and rec[a]["auc_F"] <= COLL
        return bool(ok)
    except (KeyError, TypeError):
        return False

def heads(rec):
    """all (layer, head, plant_mass, beta_eff, top1) of one eval record"""
    out = []
    for lay in rec.get("attn", []):
        for h in range(len(lay["beta_eff"])):
            out.append((lay["layer"], h, lay["plant_mass"][h], lay["beta_eff"][h], lay["top1_within_0p5m"][h]))
    return out

def committed(rec):
    return [x for x in heads(rec) if x[2] >= MASS]

def sharp(rec):
    return [x for x in committed(rec) if x[3] >= BETA and x[4] >= TOP1]

R = {}
for s in (0, 1, 2):
    p = f"{ROOT}/work/diag5/diag_G5_s{s}.json"
    if os.path.exists(p):
        R[s] = json.load(open(p))
cells = sorted({c for r in R.values() for c in r["curve"]}, key=lambda c: ["cos16_k8o8_120ep", "cos16_k8o8_wd1_120ep", "BEST_120ep", "base_k8o8_120ep"].index(c))
out = {"cells": {}, "G1": {}, "G3": {}, "G2": {}, "G4": {}}
lines = []
for c in cells:
    out["cells"][c] = {}
    for s, r in R.items():
        if c not in r["curve"]:
            continue
        cur = {e["cycle"]: e for e in r["curve"][c]}
        fin = r.get("final", {}).get(c)
        rows = []
        for cy in sorted(cur):
            e = cur[cy]
            ch = committed(e)
            top = max(heads(e), key=lambda x: x[2])          # the head with the largest plant mass
            rows.append({"cycle": cy, "F": e["auc_F"], "ALLQ": e["auc_all_strata"], "R2": e["r2_reg"], "n_committed": len(ch),
                         "committed": [dict(layer=a, head=b, plant_mass=m, beta=be, top1=t) for a, b, m, be, t in ch],
                         "maxmass_head": dict(layer=top[0], head=top[1], plant_mass=top[2], beta=top[3], top1=top[4]),
                         "max_beta_any_head": max(x[3] for x in heads(e)),
                         "sharp_committed": [dict(layer=a, head=b, plant_mass=m, beta=be, top1=t) for a, b, m, be, t in sharp(e)]})
        d = {"rows": rows, "has_final": fin is not None, "complete": fin is not None}
        if fin is not None:
            d["final_pass"] = lp(fin)
            d["final_ablate"] = {a: {k: fin[a][k] for k in ("auc_F", "auc_all_strata", "r2_reg")} for a in ("ablate_zero_clear", "ablate_shuffle_clear", "ablate_no_plant")}
            d["final_epochs"] = fin.get("epochs"); d["final_steps"] = fin.get("steps")
        out["cells"][c][s] = d
        lines.append(f"\n== {c}  seed {s}  ({'complete, 30 cycles' if fin else 'PARTIAL: last eval cycle %d' % max(cur)}) ==")
        lines.append("cyc    F      ALLQ    R2   | nCommit  maxmass-head(L,h):mass beta top1 | committed heads (L,h,mass,beta,top1) | maxbeta(any head)")
        for x in rows:
            m = x["maxmass_head"]
            cs = "; ".join(f"L{k['layer']}h{k['head']} m{k['plant_mass']:.2f} b{k['beta']:.1f} t{k['top1']:.2f}" for k in x["committed"]) or "-"
            lines.append(f"{x['cycle']:>3}  {x['F']:.3f}  {x['ALLQ']:.3f}  {x['R2']:.3f} | {x['n_committed']}  L{m['layer']}h{m['head']}:{m['plant_mass']:.2f} b{m['beta']:.1f} t{m['top1']:.2f} | {cs} | {x['max_beta_any_head']:.1f}")
        if fin is not None:
            a = d["final_ablate"]
            lines.append(f"final c30: PASS_cell(seed)={d['final_pass']}  zero_clear F {a['ablate_zero_clear']['auc_F']:.3f} ALLQ {a['ablate_zero_clear']['auc_all_strata']:.3f} R2 {a['ablate_zero_clear']['r2_reg']:.3f} | shuffle_clear F {a['ablate_shuffle_clear']['auc_F']:.3f} ALLQ {a['ablate_shuffle_clear']['auc_all_strata']:.3f} R2 {a['ablate_shuffle_clear']['r2_reg']:.3f} | no_plant F {a['ablate_no_plant']['auc_F']:.3f} ALLQ {a['ablate_no_plant']['auc_all_strata']:.3f} R2 {a['ablate_no_plant']['r2_reg']:.3f}")

# outcomes, in PREREG order
fc = {}   # cell -> seed -> {cycle: row}
for c in cells:
    fc[c] = {s: {x["cycle"]: x for x in out["cells"][c][s]["rows"]} for s in R if s in out["cells"][c]}
for c in cells:
    s01 = [s for s in (0, 1) if s in R and out["cells"][c][s]["has_final"]]
    out["G1"][c] = {"seeds_with_final": s01, "pass_s0": out["cells"][c][0].get("final_pass"), "pass_s1": out["cells"][c][1].get("final_pass"),
                    "PASS_REG": len(s01) == 2 and all(out["cells"][c][s]["final_pass"] for s in s01)}
    # G3: committed head with beta>=16 AND top1>=0.5, on both seeds; (a) at the FINAL cycle, (b) at ANY eval cycle of each seed
    fin_ok = all(len(fc[c][s][30]["sharp_committed"]) > 0 for s in (0, 1))
    any_ok = all(any(len(x["sharp_committed"]) > 0 for x in fc[c][s].values()) for s in (0, 1))
    out["G3"][c] = {"final_cycle_sharp_committed_both_seeds": fin_ok, "any_eval_sharp_committed_both_seeds": any_ok,
                    "per_seed_final": {s: fc[c][s][30]["sharp_committed"] for s in (0, 1)},
                    "per_seed_any_eval_cycles": {s: [cy for cy, x in sorted(fc[c][s].items()) if x["sharp_committed"]] for s in (0, 1)}}
    # G2: committed head on both seeds (at the final cycle; and at any eval) AND F(c30)-F(c24) >= +0.03 on both
    dF = {s: fc[c][s][30]["F"] - fc[c][s][24]["F"] for s in (0, 1)}
    com_fin = all(fc[c][s][30]["n_committed"] > 0 for s in (0, 1)); com_any = all(any(x["n_committed"] > 0 for x in fc[c][s].values()) for s in (0, 1))
    out["G2"][c] = {"dF_c30_minus_c24": dF, "committed_head_both_seeds_at_c30": com_fin, "committed_head_both_seeds_any_eval": com_any,
                    "G2_literal_final": com_fin and all(v >= STEP for v in dF.values()), "G2_literal_anyeval_commit": com_any and all(v >= STEP for v in dF.values())}
    out["G4"][c] = {"s1_any_committed_eval_cycles": [cy for cy, x in sorted(fc[c][1].items()) if x["n_committed"] > 0]}
out["G4_seed1_commits_no_head_in_any_cell_at_any_eval"] = all(len(v["s1_any_committed_eval_cycles"]) == 0 for v in out["G4"].values())
out["PASS_REG_cells"] = [c for c in cells if out["G1"][c]["PASS_REG"]]
json.dump(out, open(f"{ROOT}/raw/grid5_cells.json", "w"), indent=1)
open(f"{ROOT}/raw/grid5_tables.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nPASS_REG cells:", out["PASS_REG_cells"])
for c in cells:
    print(c, "G1", {k: v for k, v in out["G1"][c].items()}, "\n   G3", {k: out["G3"][c][k] for k in ("final_cycle_sharp_committed_both_seeds", "any_eval_sharp_committed_both_seeds", "per_seed_any_eval_cycles")}, "\n   G2", out["G2"][c])
print("G4 seed1 commits no head anywhere:", out["G4_seed1_commits_no_head_in_any_cell_at_any_eval"])
