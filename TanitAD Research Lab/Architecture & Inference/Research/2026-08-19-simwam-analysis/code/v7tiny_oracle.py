"""Is the one-tick latent movement PREDICTABLE AT ALL? — the ceiling for G2.

⛔ WHY THIS EXISTS. v7-tiny's fixed arm scores EM = -443 at h=1 on held-out
clips: its delta is ~15x the true per-tick movement. The obvious read is "the
predictor is broken". But there is a competing read that would make that verdict
meaningless, and it has to be killed or confirmed BEFORE the gate is interpreted:

    if the per-tick movement of the latent is mostly UNPREDICTABLE (encoder
    jitter, or a representation regularised toward isotropy by O6/SIGReg),
    then NO predictor can beat HOLD, EM<=0 is the correct answer, and the
    residual-init story explains nothing.

Two cheap, independent probes, both on HELD-OUT clips:

  A. DELTA AUTOCORRELATION. Real motion is smooth: consecutive deltas
     dz_t = z_{t+1}-z_t and dz_{t+1} point in similar directions. White noise
     does not. corr ~ 0 means the movement carries no exploitable structure;
     corr >> 0 means it does.

  B. LINEAR ORACLE. Fit the best RIDGE map [z, a] -> dz on one set of clips and
     score it on a DISJOINT set. This is a lower bound on what is learnable --
     a linear map is far weaker than a 7.5 M-parameter transformer, so if the
     LINEAR map beats HOLD, the task is learnable and the trained predictor is
     genuinely failing at it. If even the fitted oracle cannot beat HOLD, the
     target is noise and G2 is unreachable BY CONSTRUCTION.

⚠️ EPISODE-DISJOINT by construction: the ridge is fit on clips it is never
scored on. Fitting and scoring on the same clips would let the oracle memorise
and would invert the conclusion.

⚠️ Reported alongside a HOLD control (dz_hat = 0, EM = 0 exactly) and a MEAN
control (dz_hat = mean training delta) so a "beats hold" reading cannot come
from a constant offset.

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SP = Path(r"C:\Users\Admin\AppData\Local\Temp\claude"
          r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
          r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
sys.path.insert(0, str(SP / "sp2"))
sys.path.insert(0, str(SP))
sys.path.insert(0, str(Path(r"G:\Meine Ablage\SayBouBase\raw\Projects"
                            r"\TanitAD\stack")))


def main() -> int:
    ap = argparse.ArgumentParser(description="G2 ceiling")
    ap.add_argument("--arm", default="fixed")
    ap.add_argument("--cache", default=str(
        SP / "sp2/cache/v7tiny-heldout24-w120-256x640cyl"))
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--frames-per-clip", type=int, default=140)
    ap.add_argument("--out", default=str(SP / "v7tiny_oracle.json"))
    #: ⭐ the cross-check that decides SCOPE. If v7-tiny's latent is noise but
    #: v6F's is not, the finding is "2,000 steps is too few"; if BOTH are noise,
    #: it is a property of the recipe and it explains the programme's stall.
    ap.add_argument("--v6f", action="store_true",
                    help="score v6F's real 20k checkpoint instead of a v7-tiny arm")
    ap.add_argument("--v6f-ckpt",
                    default=str(SP / "ckpt/v6F_sw_step020000.fp16.pt"))
    ap.add_argument("--v6f-config", default=str(SP / "sp2/v6F_config.json"))
    a = ap.parse_args()

    import v7tiny_g2 as G
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.v6f:
        import e_pred_probe as E
        world, step = E.load_world(Path(a.v6f_ckpt), Path(a.v6f_config), dev)
        a.arm = f"v6F@{Path(a.v6f_ckpt).name}"
    else:
        world, step = G.load_arm(a.arm, dev)
    clips = sorted(Path(a.cache).glob("*.v2ep.pt"))[:a.clips]
    if len(clips) < 6:
        print(f"  [FATAL] need >=6 clips, have {len(clips)}")
        return 1

    Z, A, cid = [], [], []
    for n, cp in enumerate(clips, 1):
        z, act, _spd = G.encode_clip(world, cp, dev, a.frames_per_clip)
        Z.append(z.numpy())
        A.append(act.numpy())
        cid.append(cp.name[:8])
        print(f"    [{n}/{len(clips)}] {cp.name[:10]} {len(z)} frames",
              flush=True)

    # ---- A. delta autocorrelation -------------------------------------------
    cos1, cos5 = [], []
    for z in Z:
        dz = z[1:] - z[:-1]
        nz = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-12)
        cos1.append(float((nz[:-1] * nz[1:]).sum(1).mean()))
        if len(nz) > 5:
            cos5.append(float((nz[:-5] * nz[5:]).sum(1).mean()))
    print(f"\n  A. DELTA DIRECTION AUTOCORRELATION (held-out, per clip)")
    print(f"     lag 1 (0.1s): cos {np.mean(cos1):+.4f} "
          f"[{np.min(cos1):+.4f}, {np.max(cos1):+.4f}]")
    print(f"     lag 5 (0.5s): cos {np.mean(cos5):+.4f} "
          f"[{np.min(cos5):+.4f}, {np.max(cos5):+.4f}]")
    print(f"     ~0 => movement is directionally unstructured (noise-like);"
          f"  >>0 => smooth, exploitable motion")

    # ---- B. linear oracle, EPISODE-DISJOINT ---------------------------------
    half = len(Z) // 2
    def build(ix):
        X, Y = [], []
        for i in ix:
            z, ac = Z[i], A[i]
            n = min(len(z), len(ac)) - 1
            X.append(np.concatenate([z[:n], ac[:n]], 1))
            Y.append(z[1:n + 1] - z[:n])
        return np.concatenate(X), np.concatenate(Y)
    Xtr, Ytr = build(range(half))
    Xte, Yte = build(range(half, len(Z)))
    print(f"\n  B. LINEAR ORACLE  fit on clips 0..{half-1} "
          f"({len(Xtr):,} rows) -> scored on {half}..{len(Z)-1} "
          f"({len(Xte):,} rows), EPISODE-DISJOINT")

    mov = float((Yte ** 2).sum())
    res = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "arm": a.arm, "step": step,
           "n_clips": len(Z), "parity": False,
           "delta_autocorr": {"lag1_cos_mean": round(float(np.mean(cos1)), 4),
                              "lag5_cos_mean": round(float(np.mean(cos5)), 4)},
           "controls": {}, "oracle": {}}

    em_hold = 1.0 - float((Yte ** 2).sum()) / mov
    mu = Ytr.mean(0, keepdims=True)
    em_mean = 1.0 - float(((Yte - mu) ** 2).sum()) / mov
    res["controls"] = {"hold": round(em_hold, 6),
                       "mean_delta": round(em_mean, 6)}
    print(f"     control HOLD (dz=0)        EM {em_hold:+.4f}   (must be 0.0000)")
    print(f"     control MEAN (dz=mean)     EM {em_mean:+.4f}")

    # per-CLIP test terms so the ceiling carries an episode-cluster CI. A bare
    # point estimate near zero is exactly where a CI decides the verdict, and
    # `EM > 0` as a pass rule fires on +0.0018, which is nothing.
    te_ix = list(range(half, len(Z)))
    def clip_rows(i):
        z, ac = Z[i], A[i]
        n = min(len(z), len(ac)) - 1
        return (np.concatenate([z[:n], ac[:n]], 1), z[1:n + 1] - z[:n])

    Xc = Xtr - Xtr.mean(0, keepdims=True)
    Yc = Ytr - mu
    G_ = Xc.T @ Xc
    C = Xc.T @ Yc
    #: materiality floor. Below this the "oracle" is indistinguishable from
    #: HOLD and calling it "learnable" would be reading noise as a result.
    MATERIAL = 0.01
    best, best_lam, best_ci = None, None, (0.0, 0.0)
    rng = np.random.default_rng(0)
    for lam in (1e-4, 1e-2, 1.0, 1e2, 1e4):
        Wr = np.linalg.solve(G_ + lam * np.eye(G_.shape[0]), C)
        errs, movs = [], []
        for i in te_ix:
            Xi, Yi = clip_rows(i)
            pi = (Xi - Xtr.mean(0, keepdims=True)) @ Wr + mu
            errs.append(float(((Yi - pi) ** 2).sum()))
            movs.append(float((Yi ** 2).sum()))
        errs, movs = np.array(errs), np.array(movs)
        e = 1.0 - errs.sum() / movs.sum()
        bs = np.empty(4000)
        for b in range(4000):
            j = rng.integers(0, len(errs), len(errs))
            bs[b] = 1.0 - errs[j].sum() / movs[j].sum()
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        res["oracle"][f"ridge_{lam:g}"] = {"em": round(float(e), 6),
                                           "ci95": [round(lo, 6), round(hi, 6)]}
        tag = ("MATERIAL" if lo > MATERIAL else
               "== hold (immaterial)" if lo > -MATERIAL else "worse than hold")
        print(f"     ridge lambda={lam:<7g} EM {e:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  {tag}")
        # ⛔ select on the CI LOWER BOUND, not the point estimate. Selecting on
        # the point estimate picks the lambda with the widest CI (measured: v6F
        # lambda=0.01 reads +0.1298 with CI [-0.0224, +0.2349], i.e. not
        # distinguishable from hold, while lambda=1 reads a SMALLER +0.0203
        # whose CI [+0.0056, +0.0317] EXCLUDES zero). Point-estimate selection
        # would have reported the unfalsifiable number and hidden the real one.
        if best is None or lo > best_ci[0]:
            best, best_lam, best_ci = e, lam, (lo, hi)
    res["oracle_best_em"] = round(float(best), 6)
    res["oracle_best_lambda"] = best_lam
    res["oracle_best_ci95"] = [round(best_ci[0], 6), round(best_ci[1], 6)]
    res["materiality_floor"] = MATERIAL
    res["verdict"] = (
        f"LEARNABLE: the linear oracle explains {best:.1%} of held-out per-tick "
        f"movement (CI lower bound {best_ci[0]:+.4f} > {MATERIAL}), so the "
        f"movement carries exploitable structure and a predictor scoring EM<0 "
        f"is failing a learnable task"
        if best_ci[0] > MATERIAL else
        f"NOT LINEARLY PREDICTABLE: the best fitted LINEAR oracle explains only "
        f"{best:.2%} of held-out per-tick movement (CI [{best_ci[0]:+.4f}, "
        f"{best_ci[1]:+.4f}], not materially above HOLD). "
        f"⛔ THIS IS A ONE-DIRECTIONAL INFERENCE AND MUST NOT BE READ AS "
        f"'unpredictable by any predictor'. A linear oracle BEATING hold proves "
        f"the target is learnable; a linear oracle FAILING proves only that it "
        f"is not learnable BY A LINEAR MAP. In 2048 dims with nonlinear scene "
        f"motion that is a WEAK lower bound -- the nonlinear question needs a "
        f"nonlinear probe (v7tiny_mlp_oracle.py). "
        f"⚠️ SCOPE: measured on this arm's own latents at this step count; it "
        f"does NOT transfer to another model or training length unmeasured.")
    print(f"\n  VERDICT: {res['verdict']}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
