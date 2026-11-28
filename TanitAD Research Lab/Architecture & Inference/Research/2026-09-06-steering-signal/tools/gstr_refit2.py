"""D-GSTR-1 P2 (v2) -- can the steering signal turn left, and what would fix it.

⚠️ v1 OF THIS PROBE WAS WRONG AND IS RETRACTED HERE RATHER THAN QUIETLY FIXED:
its ridge had NO INTERCEPT, so its no-information limit was the ZERO VECTOR and
not the constant predictor. It therefore scored WORSE than its own control
(+0.6738) and "concluded" M5b. That is the 2026-08-22 ridge family exactly --
a probe whose degenerate limit is not the control it is compared against.
⛔ THE FIX IS A CONTROL, NOT A PATCH: this version asserts that at the LARGEST
lambda the ridge's score EQUALS the constant control's, and refuses to report a
verdict if it does not.

Arms, all on EPISODE-DISJOINT fit / scored splits:
  A  ORACLE RIDGE  ctx -> unit bearing, WITH intercept. lambda selected on a
                   fit-INTERNAL, episode-disjoint validation split.
  A2 TURN-SUBSET   the same, restricted to windows where the LAN target
                   actually turns (|lateral| > 0.10). The full-set regression
                   is 81 % near-straight and hides this question.
  B  HEAD REFIT    the SAME Linear(256, 3), from the checkpoint's own weights,
                   retrained on `strategic_goal_loss`, trunk FROZEN. The FIX.
  C  REPLICATE     B with a different seed -- the rig's own noise floor.
  D  ONESIDED      the DELIBERATE REGRESSION; must FAIL G1 and G2.

Gates G1/G2/G3 are PRE-REGISTERED in PREREG.md section 2 and are not edited
here. GT-left is the 6 s terminal ego-frame y > +1.0 m -- the same horizon as
the 1,597-window figure this work package inherited.
ASCII-only.
"""
import json
import os

import numpy as np
import torch
from torch import nn

BANK = os.environ.get("CTX_BANK", "/home/nvidia/navroute/CTX_BANK2.npz")
OUT = os.environ.get("REFIT_OUT", "/home/nvidia/navroute/GSTR_REFIT.json")
NBOOT = int(os.environ.get("NBOOT", "2000"))

G1_MIN_LEFT_FRAC = 0.150
G2_MIN_LEFT_ON_GTLEFT = 0.300
GT_LEFT_M = 1.0
TURN_TAU = 0.10


def boot(ep, x, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    ue = np.unique(ep)
    idx = {e: np.where(ep == e)[0] for e in ue}
    d = np.empty(nboot)
    for i in range(nboot):
        pick = rng.choice(ue, size=ue.size, replace=True)
        d[i] = x[np.concatenate([idx[e] for e in pick])].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean": round(float(x.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n": int(x.size), "n_ep": int(ue.size)}


def l_bear(p, t):
    u = p / np.linalg.norm(p, axis=1, keepdims=True).clip(1e-9)
    return 1.0 - (u * t).sum(1)


def ridge(Xf, Yf, Xs, lam):
    """WITH INTERCEPT: centre both, solve, add the fit mean back."""
    mx, my = Xf.mean(0), Yf.mean(0)
    A = (Xf - mx).T @ (Xf - mx) + lam * np.eye(Xf.shape[1])
    Wr = np.linalg.solve(A, (Xf - mx).T @ (Yf - my))
    return (Xs - mx) @ Wr + my


def fit_head(ctx_f, tgt_f, dist_f, ctx_s, W0, b0, *, seed, epochs=60,
             lr=1e-3, onesided=False, turn_weight=0.0):
    torch.manual_seed(seed)
    lin = nn.Linear(ctx_f.shape[1], 3)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(W0))
        lin.bias.copy_(torch.from_numpy(b0))
        if seed != 0:
            lin.weight.add_(torch.randn_like(lin.weight) * 1e-3)
    X = torch.from_numpy(ctx_f).float()
    T = torch.from_numpy(tgt_f).float()
    D = torch.from_numpy(dist_f).float()
    # ⭐ THE NEXT LEVER, NAMED BY A2. The full-set cosine loss is 81 % NEAR-
    # STRAIGHT windows, so the turns -- the only rows that carry left/right --
    # contribute almost none of the gradient. `turn_weight` upweights a row by
    # its own |lateral|, which is a property of the LABEL and of nothing else:
    # no new input, no threshold tuned on the scored split.
    Wt_ = (1.0 + turn_weight * T[:, 1].abs()) if turn_weight else None
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)

    def head(raw):
        xy = raw[:, :2]
        if onesided:
            xy = torch.stack([xy[:, 0],
                              -nn.functional.softplus(xy[:, 1])], dim=-1)
        return xy / xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, 256):
            j = perm[i:i + 256]
            raw = lin(X[j])
            lb = 1.0 - (head(raw) * T[j]).sum(-1)
            ld = (torch.tanh(raw[:, 2]) - D[j]).abs()
            if Wt_ is None:
                loss = lb.mean() + ld.mean()
            else:
                w = Wt_[j]
                loss = (lb * w).sum() / w.sum() + ld.mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        u = head(lin(torch.from_numpy(ctx_s).float()))
    return u.numpy().astype(np.float64), lin


def main():
    z = np.load(BANK)
    ctx = z["ctx"].astype(np.float64)
    gs = z["g_str"].astype(np.float64)
    tgt = z["target"].astype(np.float64)
    tval = z["tvalid"].astype(bool)
    ep = z["ep"].astype(np.int64)
    nav = z["nav"].astype(np.int64)
    gty6 = z["gt_y6"].astype(np.float64)
    gtv6 = z["gt_valid6"].astype(bool)
    W0, b0 = z["W"].astype(np.float32), z["b"].astype(np.float32)
    n, d = ctx.shape

    ue = np.unique(ep)
    perm = np.random.default_rng(0).permutation(ue)
    fit_eps = set(perm[:int(round(0.70 * ue.size))].tolist())
    inner = set(perm[:int(round(0.55 * ue.size))].tolist())
    IS_FIT = np.array([e in fit_eps for e in ep])
    IS_INNER = np.array([e in inner for e in ep])
    IS_VAL = IS_FIT & ~IS_INNER
    Sall = ~IS_FIT
    F, I, V, S = IS_FIT & tval, IS_INNER & tval, IS_VAL & tval, Sall & tval
    eps_s = ep[S]

    res = {"tool": "D-GSTR-1 P2 oracle probe + head refit (v2, with intercept)",
           "bank": BANK, "evidence_class": "MEASURED (ours)",
           "tier": "T1 for the model output; the LAN corridor is a LABEL",
           "estimator": f"episode-cluster bootstrap, {NBOOT} resamples, seed 0",
           "n_windows": int(n), "d_ctx": int(d), "n_episodes": int(ue.size),
           "RETRACTION": ("v1 of this probe had NO INTERCEPT; its degenerate "
                          "limit was the zero vector, not the constant, so it "
                          "scored +0.6738 WORSE than its own control and "
                          "'concluded' M5b. Retracted, not patched: v2 carries "
                          "the LARGE-LAMBDA control that makes that failure "
                          "impossible to repeat silently."),
           "SPLIT": {"rule": "EPISODE-DISJOINT; lambda on a fit-INTERNAL "
                             "episode-disjoint validation split; the scored "
                             "split is scored and never tuned on",
                     "n_ep_fit": len(fit_eps),
                     "n_ep_scored": int(ue.size - len(fit_eps)),
                     "n_win_fit": int(IS_FIT.sum()),
                     "n_win_scored": int(Sall.sum())},
           "_frozen_trunk": ("⚠️ THE TRUNK IS FROZEN. This shows what a linear "
                             "map on the SAME ctx can express; it is NOT a "
                             "retrained arm and is never quoted as one.")}

    # ---- constants / floors ----------------------------------------------
    cmu = tgt[F, :2].mean(0)
    cmu = cmu / max(np.linalg.norm(cmu), 1e-12)
    lb_const = 1.0 - (tgt[S, :2] @ cmu)
    lb_straight = 1.0 - tgt[S, 0]
    lb_model = l_bear(gs[S, :2], tgt[S, :2])
    navs = np.unique(nav)
    Nf = np.stack([(nav[F] == v).astype(float) for v in navs], 1)
    Ns = np.stack([(nav[S] == v).astype(float) for v in navs], 1)
    lb_nav = l_bear(ridge(Nf, tgt[F, :2], Ns, 1e-6), tgt[S, :2])

    lams = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e8]
    best = min(((lam, float(l_bear(ridge(ctx[I], tgt[I, :2], ctx[V], lam),
                                   tgt[V, :2]).mean()))
                for lam in lams), key=lambda kv: kv[1])
    lam = best[0]
    lb_or = l_bear(ridge(ctx[F], tgt[F, :2], ctx[S], lam), tgt[S, :2])
    lb_huge = l_bear(ridge(ctx[F], tgt[F, :2], ctx[S], 1e12), tgt[S, :2])
    ctrl_ok = bool(abs(float(lb_huge.mean()) - float(lb_const.mean())) < 1e-3)

    res["A_ORACLE_RIDGE"] = {
        "n_fit": int(F.sum()), "n_scored": int(S.sum()), "d": int(d),
        "lambda_selected": lam,
        "CONTROL_large_lambda_equals_constant": {
            "l_bear_at_lambda_1e12": round(float(lb_huge.mean()), 6),
            "l_bear_constant_control": round(float(lb_const.mean()), 6),
            "PASS": ctrl_ok,
            "rule": "the ridge's no-information limit MUST be the constant "
                    "predictor; if it is not, the probe is measuring its own "
                    "misconfiguration"},
        "l_bear_ORACLE_ridge_ctx": boot(eps_s, lb_or),
        "l_bear_CONTROL_constant_only": boot(eps_s, lb_const),
        "l_bear_CONTROL_constant_straight": boot(eps_s, lb_straight),
        "l_bear_FLOOR_nav_token_onehot": boot(eps_s, lb_nav),
        "l_bear_LIVE_MODEL_g_str": boot(eps_s, lb_model),
        "oracle_minus_constant": boot(eps_s, lb_or - lb_const),
        "model_minus_constant": boot(eps_s, lb_model - lb_const),
    }
    om = res["A_ORACLE_RIDGE"]["oracle_minus_constant"]
    res["A_ORACLE_RIDGE"]["_verdict"] = (
        "INADMISSIBLE -- the large-lambda control did not read the constant"
        if not ctrl_ok else
        ("M5a OPTIMISATION FAILURE: a linear map on the SAME ctx BEATS the "
         "constant control on the scored split, so the information is there "
         "and training did not find it" if om["hi"] < 0 else
         "the ridge does NOT beat the constant control: ctx does not carry "
         "the bearing LINEARLY. ⚠️ A negative from a linear probe is a "
         "negative about LINEAR decodability only, never about learnability"))

    # ---- A2: the TURN subset ---------------------------------------------
    turn_f = F & (np.abs(tgt[:, 1]) > TURN_TAU)
    turn_s = S & (np.abs(tgt[:, 1]) > TURN_TAU)
    if turn_f.sum() > 50 and turn_s.sum() > 20:
        pr = ridge(ctx[turn_f], tgt[turn_f, :2], ctx[turn_s], lam)
        sgn = np.sign(pr[:, 1]) == np.sign(tgt[turn_s, 1])
        base = max(float((tgt[turn_s, 1] > 0).mean()),
                   float((tgt[turn_s, 1] < 0).mean()))
        res["A2_TURN_SUBSET"] = {
            "question": "when the LAN route ACTUALLY turns, can a linear map "
                        "on ctx tell WHICH WAY?",
            "n_fit": int(turn_f.sum()), "n_scored": int(turn_s.sum()),
            "tau": TURN_TAU,
            "sign_accuracy_ridge": round(float(sgn.mean()), 4),
            "CONTROL_majority_class": round(base, 4),
            "sign_acc_bootstrap": boot(ep[turn_s], sgn.astype(float)),
            "live_model_sign_accuracy": round(float(
                (np.sign(gs[turn_s, 1]) == np.sign(tgt[turn_s, 1])).mean()), 4),
            "_reads": "a ridge at or below the majority-class control has "
                      "shown nothing; the live model's row is on the SAME rows",
        }

    # ---- the gates --------------------------------------------------------
    gt_left = gtv6 & (gty6 > GT_LEFT_M)
    lbf_hi = res["A_ORACLE_RIDGE"]["l_bear_CONTROL_constant_straight"]["hi"]
    tval_s = tval[Sall]
    gl_s = gt_left[Sall]

    def score(u, tag):
        lat = u[:, 1]
        lb = l_bear(u[tval_s], tgt[S, :2])
        g1 = float((lat > 0).mean())
        g2 = float((u[gl_s, 1] >= 0).mean()) if gl_s.sum() else float("nan")
        lbm = float(lb.mean())
        r = {"arm": tag,
             "G1_frac_lateral_positive": round(g1, 4),
             "G1_threshold": G1_MIN_LEFT_FRAC,
             "G1_PASS": bool(g1 >= G1_MIN_LEFT_FRAC),
             "G2_frac_nonneg_on_GT_left_6s": (round(g2, 4) if g2 == g2
                                              else None),
             "G2_threshold": G2_MIN_LEFT_ON_GTLEFT,
             "G2_PASS": bool(g2 >= G2_MIN_LEFT_ON_GTLEFT) if g2 == g2
                        else False,
             "G3_l_bear": round(lbm, 4),
             "G3_threshold_const_floor_hi": round(float(lbf_hi), 4),
             "G3_PASS": bool(lbm <= lbf_hi),
             "n_scored": int(lat.size), "n_gt_left_6s": int(gl_s.sum()),
             "G1_bootstrap": boot(ep[Sall], (lat > 0).astype(float))}
        tsub = np.abs(tgt[Sall, 1]) > TURN_TAU
        if tsub.sum() > 20:
            acc = (np.sign(u[tsub, 1]) == np.sign(tgt[Sall, 1][tsub]))
            r["turn_sign_accuracy"] = round(float(acc.mean()), 4)
            r["turn_sign_n"] = int(tsub.sum())
        r["VERDICT"] = ("PASS" if (r["G1_PASS"] and r["G2_PASS"]
                                   and r["G3_PASS"]) else "FAILED")
        return r

    res["LIVE_ARM_refcv4b"] = score(
        gs[Sall, :2] / np.linalg.norm(gs[Sall, :2], axis=1,
                                      keepdims=True).clip(1e-9),
        "LIVE refcv4b ckpt_40284_FINAL")
    res["CONTROL_constant_only"] = score(
        np.repeat(cmu[None], int(Sall.sum()), 0), "CONTROL constant-only")
    args = (ctx[F].astype(np.float32), tgt[F, :2].astype(np.float32),
            tgt[F, 2].astype(np.float32), ctx[Sall].astype(np.float32),
            W0, b0)
    # ⛔ PRE-REGISTERED BEFORE THIS ARM RAN (RESULT section 4): E must clear
    # G1, G2 AND G3 on BOTH seeds. A one-seed pass is necessary, not
    # sufficient -- C_REPLICATE is why: it cleared G1 and MISSED G2.
    TURN_W = 20.0
    for tag, seed, one, tw in (("B_REFIT", 0, False, 0.0),
                               ("C_REPLICATE", 1, False, 0.0),
                               ("D_ONESIDED_regression", 0, True, 0.0),
                               ("E_TURNWEIGHTED", 0, False, TURN_W),
                               ("E2_TURNWEIGHTED_REPLICATE", 1, False, TURN_W)):
        _u, _lin = fit_head(*args, seed=seed, onesided=one,
                            turn_weight=tw)
        res[tag] = score(_u, tag)
        if tag == "E_TURNWEIGHTED":
            torch.save({"weight": _lin.weight.detach(),
                        "bias": _lin.bias.detach(),
                        "turn_weight": tw, "seed": seed,
                        "_reads": "frozen-trunk refit of str_goal_head; NOT a "
                                  "retrained arm"},
                       "/home/nvidia/navroute/gstr_head_E.pt")
    res["E_TURNWEIGHTED"]["turn_weight"] = TURN_W
    res["E2_TURNWEIGHTED_REPLICATE"]["turn_weight"] = TURN_W
    res["SEED_ROBUSTNESS"] = {
        "rule": "a one-seed separated result is necessary, not sufficient "
                "(measured false-positive rate 6/42 = 14.3 %). An arm is "
                "certified only if BOTH seeds clear all three gates.",
        "B_vs_C_unweighted": {
            "seed0": res["B_REFIT"]["VERDICT"],
            "seed1": res["C_REPLICATE"]["VERDICT"],
            "CERTIFIED": bool(res["B_REFIT"]["VERDICT"] == "PASS"
                              and res["C_REPLICATE"]["VERDICT"] == "PASS")},
        "E_vs_E2_turnweighted": {
            "seed0": res["E_TURNWEIGHTED"]["VERDICT"],
            "seed1": res["E2_TURNWEIGHTED_REPLICATE"]["VERDICT"],
            "CERTIFIED": bool(res["E_TURNWEIGHTED"]["VERDICT"] == "PASS"
                              and res["E2_TURNWEIGHTED_REPLICATE"]["VERDICT"]
                              == "PASS")}}
    res["D_ONESIDED_regression"]["_reads"] = (
        "⛔ THE DELIBERATE REGRESSION. It must FAIL G1 and G2. A gate that "
        "cannot fail measures nothing.")
    res["GATE_VALIDITY"] = {
        "regression_failed_as_required":
            bool(not res["D_ONESIDED_regression"]["G1_PASS"]
                 and not res["D_ONESIDED_regression"]["G2_PASS"]),
        "constant_control_failed_G1_as_required":
            bool(not res["CONTROL_constant_only"]["G1_PASS"]),
        "live_arm_failed_as_expected":
            bool(not res["LIVE_ARM_refcv4b"]["G1_PASS"]),
        "ridge_no_information_limit_control":
            res["A_ORACLE_RIDGE"]["CONTROL_large_lambda_equals_constant"]
               ["PASS"],
    }
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
