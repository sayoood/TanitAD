#!/usr/bin/env python3
"""H-ADAM sections 3.1-3.2: open-loop Adam simulation on the gradient bank.  Real torch AdamW + real clip_grad_norm_ (not re-implemented).

  g_t(lam) = clip( sum_{8 micro of draw t} gB + lam_t * sum gE )  (each bank row is already divided by accum = 8, as the trainer does, so the SUM of 8 rows is the trainer's update gradient),  AdamW(lr_t, wd 0.01, betas .9/.999, eps 1e-8) -- the trainer's step order (clip, set lr, step).
  regime T: lam_t = lam * min(1, t/200), lr_t = lr_at(t) (warm-up);  regime P: lam_t = lam, lr = 1e-4, cold state.
  Delta(lam) = D(lam) - D(0);  S = sqrt(sum_C80 |Delta(.25)|^2 / sum_C80 |Delta(1)|^2);  C80 = tensors carrying 80 % of |Delta(1)|^2.

    python adam_sim.py --bank-dir <dir> --out adam_sim.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adam_common as AC  # noqa: E402
from disp_real import group_of, c80  # noqa: E402

SNAPS = (25, 50, 100, 250)
LAMS = (0.0, 0.25, 1.0)


def draws_for(seed, K, n_micro):
    rng = np.random.default_rng(seed * 1000 + 7)
    return [rng.choice(n_micro, 8, replace=False) for _ in range(K)]


def run(bank, theta0, a, draws, regime, lam, K, gb_scale=1.0, zero_b=False, clip=True, snaps=SNAPS):
    params = [torch.nn.Parameter(t.clone()) for t in theta0]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    D, norms, clipped = {}, [], 0
    for t in range(K):
        ix = torch.as_tensor(draws[t])
        gb = torch.zeros_like(bank["gB"][0]) if zero_b else bank["gB"][ix].sum(0) * gb_scale
        ge = bank["gE"][ix].sum(0)
        l = AC.lam_t(lam, t, a) if regime == "T" else lam
        g = gb + l * ge
        for p, gi in zip(params, AC.unflat(g, theta0)):
            p.grad = gi.clone()
        n = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if clip else float(g.norm())
        norms.append(n)
        clipped += int(n > 1.0 and clip)
        lr = AC.lr_at(t, a) if regime == "T" else a.lr
        for gr in opt.param_groups:
            gr["lr"] = lr
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (t + 1) in snaps:
            D[t + 1] = [(p.detach() - t0).clone() for p, t0 in zip(params, theta0)]
    return D, {"grad_norm_mean": float(np.mean(norms)), "grad_norm_max": float(np.max(norms)), "clip_engaged_steps": clipped, "steps": K}


def stats(D0, D25, D1, names):
    """per-tensor S_p, C80, energy-weighted S, cos, and the suggested U / cos on the full displacement"""
    d25 = [a_ - b_ for a_, b_ in zip(D25, D0)]
    d1 = [a_ - b_ for a_, b_ in zip(D1, D0)]
    e1 = {n: float((x * x).sum()) for n, x in zip(names, d1)}
    e25 = {n: float((x * x).sum()) for n, x in zip(names, d25)}
    core = c80(e1)
    S = math.sqrt(sum(e25[n] for n in core) / sum(e1[n] for n in core))
    S_all = math.sqrt(sum(e25.values()) / sum(e1.values()))
    v25 = torch.cat([x.reshape(-1) for n, x in zip(names, d25) if n in core])
    v1 = torch.cat([x.reshape(-1) for n, x in zip(names, d1) if n in core])
    cos = float(v25 @ v1 / (v25.norm() * v1.norm() + 1e-30))
    f25, f1 = AC.flat(D25), AC.flat(D1)
    U = float(f25.norm() / f1.norm())
    cosU = float(f25 @ f1 / (f25.norm() * f1.norm()))
    grp = {}
    for n in names:
        g = group_of(n)
        grp.setdefault(g, [0.0, 0.0])
        grp[g][0] += e25[n]
        grp[g][1] += e1[n]
    tot1 = sum(e1.values())
    return {"S_C80": S, "S_all": S_all, "elasticity_e": (math.log(S) / math.log(0.25)) if S > 0 else None, "cos_C80": cos, "U_full_norm_ratio": U, "cos_full": cosU,
            "n_C80": len(core),
            "groups": {g: {"share_of_EPO_energy_at_lam1": v[1] / tot1, "S_group": math.sqrt(v[0] / v[1]) if v[1] > 0 else None} for g, v in grp.items()},
            "S_p": {n: (math.sqrt(e25[n] / e1[n]) if e1[n] > 0 else None) for n in names}, "C80_tensors": core}


def ratios_r(bank, names, shapes, a, draws, K):
    """section 3.1: r_p(lam) = lam * rms_t|gE_p| / rms_t|gB_p| over K draws; coordinate share |lam gE| > |gB| (rms over t)"""
    ix = [torch.as_tensor(d) for d in draws[:K]]
    GB = torch.stack([bank["gB"][i].sum(0) for i in ix])
    GE = torch.stack([bank["gE"][i].sum(0) for i in ix])      # [K, P]
    out, o = {}, 0
    for n, sh in zip(names, shapes):
        m = int(np.prod(sh))
        b = GB[:, o:o + m]
        e = GE[:, o:o + m]
        o += m
        nb = float(b.pow(2).sum(1).mean().sqrt())
        ne = float(e.pow(2).sum(1).mean().sqrt())
        rb = b.pow(2).mean(0).sqrt()
        re = e.pow(2).mean(0).sqrt()
        out[n] = {"rms_norm_gB": nb, "rms_norm_gE": ne, "r_lam1": ne / nb if nb > 0 else None, "r_lam025": 0.25 * ne / nb if nb > 0 else None,
                  "coord_share_EPO_dominates_lam1": float((re > rb).float().mean()), "coord_share_EPO_dominates_lam025": float((0.25 * re > rb).float().mean()), "numel": m}
    tot_b = float(GB.pow(2).sum(1).mean().sqrt())
    tot_e = float(GE.pow(2).sum(1).mean().sqrt())
    return out, {"rms_total_norm_gB": tot_b, "rms_total_norm_gE": tot_e, "r_total_lam1": tot_e / tot_b, "r_total_lam025": 0.25 * tot_e / tot_b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", type=int, nargs="+", default=[0, 1, 2])
    q = ap.parse_args()
    t0 = time.time()
    SF = AC.load_trainer()
    a, _ = AC.r0_args(SF)
    holder, cfg = AC.build_scorer(SF, a)
    armB = SF.ScorerCopy(holder)
    names = [n for n, _ in armB.named_parameters()]
    theta0 = [p.detach().clone() for p in armB.parameters()]
    shapes = [tuple(p.shape) for p in theta0]
    res = {"tool": "adam_sim/1", "class": "ESTIMATED (real scorer weights / labels / trajectories; substituted sctx; open-loop at theta_0; CPU fp32)", "argv_lr": a.lr, "warmup": a.warmup,
           "epo_ramp": a.epo_ramp, "n_params": int(sum(t.numel() for t in theta0)), "n_tensors": len(names), "streams": {}}
    ctrl = {}
    keep = ("S_C80", "S_all", "cos_C80", "U_full_norm_ratio", "cos_full")
    for s in q.streams:
        bank = torch.load(os.path.join(q.bank_dir, f"bank_s{s}.pt"), weights_only=False)
        nm = bank["gB"].shape[0]
        draws = draws_for(s, 250, nm)
        r_p, r_tot = ratios_r(bank, names, shapes, a, draws, 50)
        R = {"r": {"per_tensor": r_p, "total": r_tot}, "modes": {}}
        for regime, K in (("T", 250), ("P", 100)):
            runs, meta = {}, {}
            for lam in LAMS:
                runs[lam], meta[lam] = run(bank, theta0, a, draws, regime, lam, K)
            R["modes"][regime] = {"clip": {str(lam): meta[lam] for lam in LAMS}, "at": {}}
            for k in SNAPS:
                if k in runs[0.0]:
                    R["modes"][regime]["at"][str(k)] = stats(runs[0.0][k], runs[0.25][k], runs[1.0][k], names)
            if s == q.streams[0]:                       # controls on the first stream only (they are exact, not statistical)
                c2 = {lam: run(bank, theta0, a, draws, regime, lam, 50, zero_b=True, snaps=(50,))[0][50] for lam in LAMS}
                c2n = {lam: run(bank, theta0, a, draws, regime, lam, 50, zero_b=True, clip=False, snaps=(50,))[0][50] for lam in LAMS}
                c3 = {lam: run(bank, theta0, a, draws, regime, lam, 50, gb_scale=1000.0, snaps=(50,))[0][50] for lam in LAMS}
                rr = run(bank, theta0, a, draws, regime, 1.0, 50, snaps=(50,))[0][50]
                same = all(torch.equal(x, y) for x, y in zip(rr, runs[1.0][50]))
                ctrl[regime] = {"C2_BCE_removed_clip_on": {k: v for k, v in stats(c2[0.0], c2[0.25], c2[1.0], names).items() if k in keep},
                                "C2_BCE_removed_clip_off": {k: v for k, v in stats(c2n[0.0], c2n[0.25], c2n[1.0], names).items() if k in keep},
                                "C3_BCE_x1000": {k: v for k, v in stats(c3[0.0], c3[0.25], c3[1.0], names).items() if k in keep},
                                "C4_rerun_bit_identical": bool(same)}
        res["streams"][str(s)] = R
        print(f"stream {s} done {time.time()-t0:.0f}s | S_P50 {R['modes']['P']['at']['50']['S_C80']:.3f} S_T50 {R['modes']['T']['at']['50']['S_C80']:.3f} "
              f"S_T250 {R['modes']['T']['at']['250']['S_C80']:.3f}", flush=True)
    res["controls"] = ctrl
    json.dump(res, open(q.out, "w"), indent=1)
    print("ZZSIM_OK", q.out, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
