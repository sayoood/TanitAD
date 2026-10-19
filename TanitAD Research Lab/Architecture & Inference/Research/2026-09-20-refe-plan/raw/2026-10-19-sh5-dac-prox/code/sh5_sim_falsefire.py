"""R51 / R52 / R53 discipline: the FALSE-FIRE rate of every registered control, simulated on NOISE-TRAINED replicates BEFORE the prereg is sealed.  CPU, synthetic (sh5_synth), the SAME code path as the real
run (Trainer -> save/predict -> oof_stack -> Analyzer -> the registered decision functions).  A noise-trained replicate trains the branch on PERMUTED labels and targets (the machinery null of the prereg):
its OOF logit carries no WITHIN-SET ranking (it can carry a set-level offset, which the stacker may weight and which a within-set AUC cannot see), so Delta_F must scatter about 0.  We read EVERY registered decision rule on every replicate - the CTX-REACHABLE per-seed rule (lower CI > +0.02), the NOT-REACHABLE
rule, C-REACH, C-POWER, PICK-GAIN - and count how often each FIRES, next to a POSITIVE control (the real branch in the world where the ctx does carry the information) so that "never fired" cannot be a
rule that cannot fire.  We also record how wide the training noise is next to the interval the bootstrap reports (`H-ESTIM-SEED-1`: the bootstrap is blind to training variance).

    python sh5_sim_falsefire.py --reps 16 --pos-reps 4 --out F
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sh5_lib as L                                                                                    # noqa: E402
import sh5_branch as B                                                                                 # noqa: E402
import sh5_synth as SY                                                                                 # noqa: E402
import sh5_train_eval as TE                                                                            # noqa: E402

torch.set_num_threads(2)
CFG = B.SH5Cfg(d=32, heads=2, layers=2, ffn=64, d_ctx=40, fourier_k=3)


def one_replicate(D, ctx, rep: int, epochs: int, lr: float, arm: str = "nulllabel"):
    """train one replicate and read EVERY registered decision rule on it.  arm 'nulllabel' = a noise-trained branch (labels permuted within set AND targets permuted within token): every rule that can FIRE
    is counted (CTX-REACHABLE per-seed rule, C-REACH, C-POWER, PICK-GAIN) - and the NOT-REACHABLE rule, which a useless branch satisfies by construction (that is why NOT-REACHABLE needs C-POWER).
    arm 'S' = the real branch in the world where the ctx DOES carry the information: the POSITIVE control (the same rules must be able to fire)."""
    L.LR = lr
    seed = rep % 2
    spec = [TE.Spec(arm, k, seed, rep if arm == "nulllabel" else 0) for k in range(L.FOLDS)]
    rc = TE.RunCfg(device="cpu", autocast=False, seed=1000 + rep, p_sub=8, tokens_per_step=4, epochs_override=epochs, shard_gib=10.0)
    tr = TE.Trainer(D, spec, rc, TE.MemCtx(ctx), cfg=CFG, log=lambda m: None)
    tr.train()
    g = tr.predict()[spec[0].group]
    FL, PK = L.refs()
    zL = D["z6"][..., L.DAC_COL]
    lab = D["lab"][..., L.DAC_COL]
    an = L.Analyzer(lab, D["code"], D["log_idx"], D["n_logs"], B=L.BOOT_B)
    stats = PK.Stats(D["log_idx"], an.draws, FL, n_logs=D["n_logs"])
    zc, par = L.oof_stack(zL, g["z"], lab, D["fold"])
    stL = an.stats(zL)
    stLS = an.stats(zc)
    out = {"rep": rep, "arm": arm, "seed": seed}
    for name, stratum in (("F", "Q4"), ("Q123", "Q123")):
        ci = an.diff(stLS, stL, stratum)
        out[f"delta_{name}"] = ci
        out[f"fires_{name}"] = bool(ci["lo"] > L.INFO_MARGIN)                                   # the CTX-REACHABLE per-seed rule
        out[f"not_{name}"] = bool(ci["hi"] < L.NOTREACH_MARGIN)                                 # the NOT-REACHABLE per-seed rule
    out["c_power_pass_if_both_seeds_read_this"] = bool(L.power_ok([out["delta_F"]] * len(L.SEEDS)))
    sa = an.auc(an.stats(g["z"]), "Q4")
    out["S_alone_F_auc"] = sa
    out["r2_reg_vs_real_targets"] = TE.r2_reg(g["reg"], D["tgt"])
    out["c_reach_pass"] = bool(sa["lo"] >= L.REACH_MIN_AUC and out["r2_reg_vs_real_targets"] >= L.REACH_MIN_R2)
    pl = L.pick_level(D["z6"], D["lab"], zc, D[f"ep_s{seed}"], stats)
    out["pick_d_pdms_x100"], out["pick_d_dac0"] = pl["rule[S]"]["vs_P2"]["d_pdms_x100"], pl["rule[S]"]["vs_P2"]["d_dac0"]
    out["pick_gain_seed"] = L.pick_gain_seed(out["pick_d_pdms_x100"], out["pick_d_dac0"])
    out["stacker_b"] = [p[1] for p in par]
    out["stacker_a"] = [p[0] for p in par]
    return out


def rate(rows, key):
    return {"n": len(rows), "k": int(sum(bool(r[key]) for r in rows)), "rate": float(np.mean([bool(r[key]) for r in rows])) if rows else None}


def upper95(k: int, n: int) -> float:
    """one-sided 95 % Clopper-Pearson upper bound on a rate with k events in n (k = 0: 1 - 0.05^(1/n))"""
    from scipy.stats import beta
    return float(beta.ppf(0.95, k + 1, n - k)) if k < n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--pos-reps", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--pos-epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D, ctx = SY.make_synth(T=a.T, seed=77, info=True, zl_noise=0.3)
    t0 = time.time()
    rows, pos = [], []
    for r in range(a.reps):
        rows.append(one_replicate(D, ctx, r, a.epochs, a.lr))
        x = rows[-1]
        print(f"  null rep {r + 1}/{a.reps} {time.time() - t0:.0f} s: Delta_F {x['delta_F']['point']:+.4f} [{x['delta_F']['lo']:+.4f}, {x['delta_F']['hi']:+.4f}] Delta_Q123 {x['delta_Q123']['point']:+.4f} "
              f"fire(F) {x['fires_F']} fire(Q123) {x['fires_Q123']} NOT(F) {x['not_F']} reach {x['c_reach_pass']} power {x['c_power_pass_if_both_seeds_read_this']} pickgain {x['pick_gain_seed']}", flush=True)
    for r in range(a.pos_reps):
        pos.append(one_replicate(D, ctx, r, a.pos_epochs, a.lr, arm="S"))
        x = pos[-1]
        print(f"  POS rep {r + 1}/{a.pos_reps} {time.time() - t0:.0f} s: Delta_F {x['delta_F']['point']:+.4f} [{x['delta_F']['lo']:+.4f}, {x['delta_F']['hi']:+.4f}] Delta_Q123 {x['delta_Q123']['point']:+.4f} "
              f"fire(F) {x['fires_F']} fire(Q123) {x['fires_Q123']} pickgain {x['pick_gain_seed']}", flush=True)
    pts = np.array([r["delta_Q123"]["point"] for r in rows])
    ptsF = np.array([r["delta_F"]["point"] for r in rows])
    width = np.array([r["delta_Q123"]["hi"] - r["delta_Q123"]["lo"] for r in rows])
    kF = int(sum(r["fires_F"] for r in rows))
    n_fire = int(sum(r["fires_Q123"] for r in rows) + kF)
    p_rep = n_fire / (2 * len(rows))
    up = upper95(kF, len(rows))
    res = {"reps": a.reps, "pos_reps": a.pos_reps, "epochs": a.epochs, "pos_epochs": a.pos_epochs, "lr_cpu_synthetic": a.lr, "T_tokens": a.T, "N_sets": int(D["N"]),
           "world": "asymmetric-boundary synthetic (sh5_synth), zl_noise 0.3",
           "NULL_noise_trained": {
               "ctx_reachable_rule_fires_F": rate(rows, "fires_F"), "ctx_reachable_rule_fires_Q123": rate(rows, "fires_Q123"),
               "per_replicate_false_fire_rate_lower_CI_gt_0.02_pooled_F_and_Q123": p_rep, "n_fires": n_fire, "upper95_of_the_F_false_fire_rate": up,
               "P_any_of_3_label_null_replicates_fires_at_the_observed_rate": 1 - (1 - p_rep) ** L.NULL_LABEL_REPS,
               "P_any_of_3_label_null_replicates_fires_at_the_upper95_rate": 1 - (1 - up) ** L.NULL_LABEL_REPS,
               "not_reachable_rule_satisfied_F": rate(rows, "not_F"), "c_reach_false_pass": rate(rows, "c_reach_pass"), "c_power_false_pass": rate(rows, "c_power_pass_if_both_seeds_read_this"),
               "pick_gain_false_fire_per_seed": rate(rows, "pick_gain_seed"),
               "delta_Q123_points": {"mean": float(pts.mean()), "sd": float(pts.std(ddof=1)) if len(pts) > 1 else None, "min": float(pts.min()), "max": float(pts.max())},
               "delta_F_points": {"mean": float(ptsF.mean()), "sd": float(ptsF.std(ddof=1)) if len(ptsF) > 1 else None, "min": float(ptsF.min()), "max": float(ptsF.max())},
               "bootstrap_interval_width_Q123_mean": float(width.mean()),
               "training_sd_over_interval_halfwidth": (float(pts.std(ddof=1) / (width.mean() / 2)) if len(pts) > 1 else None)},
           "POSITIVE_control_branch_reads_the_ctx": {"ctx_reachable_rule_fires_F": rate(pos, "fires_F"), "ctx_reachable_rule_fires_Q123": rate(pos, "fires_Q123"), "pick_gain_per_seed": rate(pos, "pick_gain_seed"),
                                                     "delta_F_points": [r["delta_F"]["point"] for r in pos], "delta_Q123_points": [r["delta_Q123"]["point"] for r in pos]},
           "rows": rows, "pos_rows": pos,
           "note": "synthetic world, SAME stacker + bootstrap (B = 10,000) + decision functions as the real run; ESTIMATED false-fire rates for a branch that cannot read anything; the real run's rates can differ"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in res if k not in ("rows", "pos_rows")}, indent=1))


if __name__ == "__main__":
    main()
