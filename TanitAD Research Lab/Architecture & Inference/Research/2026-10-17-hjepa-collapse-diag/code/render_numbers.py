#!/usr/bin/env python
"""Render raw/readout.json -> raw/RESULT_numbers.md (tables only; the narrative lives in RESULT.md).

    python render_numbers.py <pkg>/raw/readout.json
"""
import json
import os
import sys


def f(x, d=4):
    return "n/a" if x is None else f"{x:.{d}f}"


def main(p):
    r = json.load(open(p))
    L = []
    A = L.append
    A(f"# H-HJ-COLLAPSE-1 numbers (generated from readout.json; n_episodes={r['n_episodes']}, "
      f"ckpt step {r['ckpt_step']}, strict load {r['strict']})\n")
    A(f"Estimator: {r['estimator']}\n")
    pr = r["prereg"]
    A("## PREREG outcome (operative predictor, k = 10 = 2.0 s)\n")
    A(f"* steer-sens = {f(pr['steer_sens_k10_op']['mean'])} "
      f"[{f(pr['steer_sens_k10_op']['lo'])}, {f(pr['steer_sens_k10_op']['hi'])}]")
    A(f"* v-sens (speed-channel swap) = {f(pr['v_sens_k10_op']['mean'])} "
      f"[{f(pr['v_sens_k10_op']['lo'])}, {f(pr['v_sens_k10_op']['hi'])}]; "
      f"v/steer = {f(pr['v_over_steer_ratio'], 2)}")
    A(f"* n = {pr['n_windows']} windows / {pr['n_episodes']} episodes")
    A(f"* **point-estimate outcome: {pr['outcome_point_estimate']}**; steer CI straddles "
      f"thresholds: {pr['steer_ci_straddles_thresholds'] or 'none'}\n")

    for level, title in (("op", "Operative (k = 1/5/10 steps = 0.2/1.0/2.0 s)"),
                         ("tac", "Tactical (k = 1/5/10 steps = 0.6/3.0/6.0 s)")):
        A(f"## sens_k(c), {title} -- mean of per-window ratios, episode-cluster 95 % CI\n")
        A("| swap | k=1 | k=5 | k=10 | median k=10 | ratio-of-means k=10 |")
        A("|---|---|---|---|---|---|")
        for var, d in r["sens"][level].items():
            if not d:
                continue
            cells = []
            for k in (1, 5, 10):
                m = d.get(f"k{k}", {}).get("mean_of_ratios")
                cells.append("n/a" if not m else f"{f(m['mean'])} [{f(m['lo'])}, {f(m['hi'])}]")
            k10 = d.get("k10", {})
            med = (k10.get("median_of_ratios") or {}).get("mean")
            rom = (k10.get("ratio_of_means") or {}).get("point")
            A(f"| {var} | " + " | ".join(cells) + f" | {f(med)} | {f(rom)} |")
        A("")
    A("## EXPLORATORY: sens after the model's linear decoder to_enc (operative; the space the loss reads)\n")
    A("| swap | k=1 | k=5 | k=10 |")
    A("|---|---|---|---|")
    for var, d in r.get("sens_enc_op", {}).items():
        cells = []
        for k in (1, 5, 10):
            m = d.get(f"k{k}", {}).get("mean_of_ratios")
            cells.append("n/a" if not m else f"{f(m['mean'])} [{f(m['lo'])}, {f(m['hi'])}]")
        A(f"| {var} | " + " | ".join(cells) + " |")
    A("")
    A("## rho (paper Eq. 10; within-episode variance share), episode-cluster 95 % CI\n")
    A("| quantity | rho | CI | n_ep | n_vec |")
    A("|---|---|---|---|---|")
    for k, d in r["rho"].items():
        A(f"| {k} | {f(d['rho'])} | [{f(d['lo'])}, {f(d['hi'])}] | {d['n_episodes']} | "
          f"{d['n_vectors']} |")
    A("\nReference (paper App. L, Table 12, 5 fps, DROID): collapsed rho 0.008-0.016; IDM-on "
      "healthy 0.12; time-repulsion 0.17-0.92. (Different encoder, corpus and latent shape.)\n")
    A("## Input perturbation size of each swap (RMS over windows and horizon)\n")
    A("| level | swap | rms d(a) m/s^2 | rms d(steer) rad | rms d(v/30) |")
    A("|---|---|---|---|---|")
    for level, d in r["perturbation_input_units"].items():
        for var, x in d.items():
            A(f"| {level} | {var} | {f(x['rms_a_mps2'], 4)} | {f(x['rms_steer_rad'], 5)} | "
              f"{f(x['rms_vchan_(v/30)'], 4)} |")
    A("\n## Controls\n")
    c = r["controls"]
    for k in ("self_swap_op", "self_swap_tac"):
        A(f"* {k}: {json.dumps(c[k])}")
    for kind in ("identity", "drift", "linear"):
        for level in ("op", "tac"):
            for var in ("self", "all", "steer", "v", "a"):
                d = c[f"stub_{kind}"][level].get(var, {}).get("k10")
                if d:
                    A(f"* stub {kind} / {level} / {var} / k=10: num_max_abs={d['num_max_abs']:.3e} "
                      f"den_min={d['den_min']:.3e} sens_mean={f(d['sens_mean'], 6)} "
                      f"[{f(d['sens_lo'], 6)}, {f(d['sens_hi'], 6)}] n_den_zero={d['n_den_zero']}")
    A("\n## Rollout displacement vs field norm (variant `all`)\n")
    A(json.dumps(r.get("rollout_displacement_vs_field_norm"), indent=1))
    A("\n## WM-vs-persistence (T0, teacher-forced, frozen target space, S_10 windows)\n")
    A(json.dumps(r.get("wm_mse_T0"), indent=1))
    A("\n## Exploratory terciles of the swap magnitude (op, k=10)\n")
    A("```\n" + json.dumps(r.get("exploratory_terciles_op_k10"), indent=1) + "\n```")
    A("\n## Tripwire (probe rollouts vs model.forward())\n")
    A(json.dumps(r.get("tripwire_vs_forward")))
    out = os.path.join(os.path.dirname(p), "RESULT_numbers.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
