#!/usr/bin/env python3
"""H-RL-SELECT-HEADROOM-1 -- is the 2.11x selection gap REACHABLE, or irreducible?

⛔ WHY THIS RUNS IN THE SAME TURN AS THE PROJECTION'S VERDICT. `H-PROJ-CONTACT-1` is
SUPPORTED, which retires RL-for-collision-avoidance and names the selector as the
place the effort goes. Naming it is not enough: an RL selection arm costs GPU-days,
and the question that decides whether to spend them is answerable HERE, at zero GPU,
on the fan that is now provably collision-free.

THE QUESTION: the fan CONTAINS a member 2.11x closer to the human than the one the
model picks (oracle 0.2306 m vs selected 0.4871 m). Is that gap reachable by BETTER
RANKING OVER WHAT THE MODEL ALREADY EMITS, or does closing it need information that
is not in the fan at all?

  * a rule fit on banked per-candidate features closes a large share on HELD-OUT
    episodes  ==>  the gap is a RANKING problem. An RL selection stage has a target
    it can actually hit, and the linear rule is its floor -- RL must beat it.
  * it closes ~nothing  ==>  the gap needs SCENE information the fan does not carry.
    An RL stage over these features cannot close it either, and the work belongs in
    the representation, not in the selector. That would REFUTE the redirect this
    panel exists to justify, in the same turn it was proposed.

⛔ FOUR CONTROLS, EACH WITH A KNOWN VALUE, because a probe that tunes on the data it
scores manufactures results (MEASURED four distinct ways on 2026-08-22):
  C-CONST    always pick candidate index 0        -- a no-information rule
  C-RANDOM   pick uniformly at random (20 seeds)  -- the chance floor
  C-SHUFFLE  the fitted rule with its FEATURES ROW-SHUFFLED -- must collapse to chance
  C-ORACLE   min ADE                              -- the ceiling, by construction
Every hyper-parameter (the ridge lambda) is chosen on the FIT episodes only; the
held-out episodes are scored and never tuned on. `n` and `d` are printed.

⚠️ Tier: T0 selection-quality readout on the EMITTED fan. Never a driving claim.
⚠️ Leak check, stated: every feature is available at inference (model outputs, path
geometry, and reward components computed from v0 and the agent track). NONE is
computed from the human future. The ADE target is used for FITTING on the fit split
only -- which is what a learned selector's training signal legitimately is.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = r"C:\Users\Admin\collproj"
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from tanitad.refs import contact_projection as CP           # noqa: E402
import taniteval.ci as CI                                    # noqa: E402

BANK = r"C:\Users\Admin\collproj\out\fan_bank_base_240w.npz"
GT = r"C:\Users\Admin\collproj\out\gt_240w.npz"
OUT = r"C:\Users\Admin\collproj\out\selection_headroom.json"
N_BOOT = 4000
SEED = 11
FIT_FRAC = 0.6


def main():
    z = np.load(BANK, allow_pickle=True)
    gz = np.load(GT, allow_pickle=True)
    fan = torch.tensor(np.asarray(z["fan2"]), dtype=torch.float64)
    lead = torch.tensor(np.asarray(z["lead5"]), dtype=torch.float64)[:, None]
    has = torch.tensor(np.asarray(z["has_lead"]).astype(bool))
    v0 = torch.tensor(np.asarray(z["v0"], dtype=np.float64))
    sel = torch.tensor(np.asarray(z["sel_idx"]).astype(np.int64))
    rank = torch.tensor(np.asarray(z["rank"]), dtype=torch.float64)
    conf = torch.tensor(np.asarray(z["conf"]), dtype=torch.float64)
    gt = torch.tensor(np.asarray(gz["gt"]), dtype=torch.float64)
    eid = np.asarray(z["eid"]).astype(int).ravel()
    B, N, S, _ = fan.shape

    # ⭐ the arena is the PROJECTED fan: the safety axis is now a constant, which is
    # the whole point of running this after the projection rather than before it.
    proj, info = CP.project_contact_free(fan, lead, has, margin_m=0.0)
    ade = (proj[..., 1:, :] - gt[:, None, :, :]).norm(dim=-1).mean(dim=-1)   # [B, N]

    # ---- features, all inference-available, none from the human future ---- #
    d = proj[:, :, 1:, :] - proj[:, :, :-1, :]
    spd = d.norm(dim=-1) / 0.5                                   # [B,N,4]
    hd = torch.atan2(d[..., 1], d[..., 0])
    dth = torch.atan2(torch.sin(hd[..., 1:] - hd[..., :-1]),
                      torch.cos(hd[..., 1:] - hd[..., :-1]))
    # ⛔⛔ `rank` IS `sel_score_v3` MASKED WITH -inf BY `reach_keep`, AND 7,447 OF THE
    # 30,720 ENTRIES ARE -inf. Standardising that column produces NaN for EVERY row,
    # every ridge prediction becomes NaN, and `argmin` silently returns index 0 --
    # which is exactly the C_CONST control. MEASURED 2026-09-05: `ridge_rerank`,
    # `C_CONST_index0` and `C_SHUFFLE_features` all read 5.6688 to four decimals, and
    # that three-way tie is the ONLY reason it was caught. ⇒ the mask is INFORMATION,
    # not a defect: it is carried as a separate BINARY feature and the score column is
    # made finite. An `assert` on the feature matrix pins it.
    reach_keep = torch.isfinite(rank).double()
    rank_finite = torch.where(torch.isfinite(rank), rank,
                              torch.full_like(rank, float(conf.min()) - 1.0))
    feats = {
        "rank": rank_finite,
        "reach_keep": reach_keep,
        "conf": conf,
        "peak_g": torch.tensor(np.asarray(z["peak_g"]), dtype=torch.float64),
        "c_collision": torch.tensor(np.asarray(z["c_collision"]), dtype=torch.float64),
        "c_comfort": torch.tensor(np.asarray(z["c_comfort"]), dtype=torch.float64),
        "c_feasibility": torch.tensor(np.asarray(z["c_feasibility"]), dtype=torch.float64),
        "c_headway": torch.tensor(np.asarray(z["c_headway"]), dtype=torch.float64),
        "c_progress": torch.tensor(np.asarray(z["c_progress"]), dtype=torch.float64),
        "v_mean_2s": proj[:, :, -1, :].norm(dim=-1) / 2.0,
        "v_mean_minus_v0": proj[:, :, -1, :].norm(dim=-1) / 2.0 - v0[:, None],
        "disp": proj[:, :, -1, :].norm(dim=-1),
        "lateral_end": proj[:, :, -1, 1].abs(),
        "heading_total": dth.sum(dim=-1).abs(),
        "heading_max": dth.abs().amax(dim=-1),
        "speed_std": spd.std(dim=-1),
        "min_clear": CP.min_rel_distance(proj, lead.expand_as(proj)).clamp(max=50.0),
        "sigma": info["sigma"].double(),
    }
    names = sorted(feats)
    X = torch.stack([feats[k] for k in names], dim=-1)            # [B, N, d]
    dd = X.shape[-1]
    bad = ~torch.isfinite(X)
    if bool(bad.any()):
        raise SystemExit("[sel] non-finite features in columns "
                         + str(sorted({names[i] for i in
                                       torch.nonzero(bad)[:, -1].tolist()})))
    print(f"[sel] feature matrix finite: {X.shape} -- asserted, not assumed", flush=True)

    # ---- the episode-disjoint split ---------------------------------------- #
    eps = sorted(set(eid.tolist()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(eps))
    n_fit = int(round(FIT_FRAC * len(eps)))
    fit_eps = {eps[i] for i in perm[:n_fit]}
    fit_w = np.array([e in fit_eps for e in eid])
    hel_w = ~fit_w
    print(f"[sel] {B} windows / {len(eps)} episodes -> fit {fit_w.sum()} w "
          f"({len(fit_eps)} ep) | held-out {hel_w.sum()} w "
          f"({len(eps) - len(fit_eps)} ep) | d = {dd} features, "
          f"n_fit_rows = {int(fit_w.sum()) * N}", flush=True)

    # ---- the ridge, fit on the FIT split, lambda chosen there too ---------- #
    def fit_ridge(mask, lam, Xs, ys):
        Xf = Xs[mask].reshape(-1, dd)
        yf = ys[mask].reshape(-1)
        Xf = torch.cat([Xf, torch.ones(Xf.shape[0], 1, dtype=Xf.dtype)], 1)
        A = Xf.T @ Xf + lam * torch.eye(dd + 1, dtype=Xf.dtype)
        return torch.linalg.solve(A, Xf.T @ yf)

    mu = X[fit_w].reshape(-1, dd).mean(0)
    sd = X[fit_w].reshape(-1, dd).std(0).clamp_min(1e-9)
    Xs = (X - mu) / sd
    ys = ade

    # inner split of the FIT episodes to choose lambda -- NEVER the held-out set
    fe = sorted(fit_eps)
    inner = {fe[i] for i in rng.permutation(len(fe))[:max(1, int(0.7 * len(fe)))]}
    in_w = np.array([e in inner for e in eid]) & fit_w
    val_w = fit_w & ~in_w
    best = (None, float("inf"))
    for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4):
        w = fit_ridge(in_w, lam, Xs, ys)
        pred = (torch.cat([Xs[val_w], torch.ones(int(val_w.sum()), N, 1,
                                                 dtype=Xs.dtype)], -1) @ w)
        pick = pred.argmin(dim=1)
        v = float(ys[val_w].gather(1, pick[:, None])[:, 0].mean())
        if v < best[1]:
            best = (lam, v)
    lam = best[0]
    w = fit_ridge(fit_w, lam, Xs, ys)
    print(f"[sel] lambda {lam} chosen on an INNER split of the fit episodes "
          f"(inner-val ADE {best[1]:.4f}); never on the held-out set", flush=True)

    def pred_pick(mask, weights, Xuse=None):
        Xu = Xs if Xuse is None else Xuse
        n = int(mask.sum())
        pr = torch.cat([Xu[mask], torch.ones(n, N, 1, dtype=Xu.dtype)], -1) @ weights
        return pr.argmin(dim=1)

    # ---- the arms ---------------------------------------------------------- #
    hw = hel_w
    e_h = eid[hw]
    ade_h = ade[hw]
    arms = {}
    arms["model_argmax"] = ade_h.gather(1, sel[hw][:, None])[:, 0]
    arms["conf_argmax"] = ade_h.gather(1, conf[hw].argmax(1)[:, None])[:, 0]
    order_h = rank[hw].argsort(1, descending=True)
    arms["C_ORACLE_ceiling"] = ade_h.min(dim=1).values
    arms["oracle_in_top32"] = ade_h.gather(1, order_h[:, :32]).min(dim=1).values
    arms["C_CONST_index0"] = ade_h[:, 0]
    rr = np.random.default_rng(7)
    rnd = torch.stack([ade_h.gather(1, torch.tensor(
        rr.integers(0, N, size=int(hw.sum())))[:, None])[:, 0] for _ in range(20)])
    arms["C_RANDOM_20seed"] = rnd.mean(0)
    arms["ridge_rerank"] = ade_h.gather(1, pred_pick(hw, w)[:, None])[:, 0]
    # C-SHUFFLE: the same fitted rule on ROW-SHUFFLED features must collapse to chance
    idx = torch.tensor(rr.permutation(int(hw.sum()) * N)).reshape(int(hw.sum()), N)
    Xsh = Xs[hw].reshape(-1, dd)[idx.reshape(-1)].reshape(int(hw.sum()), N, dd)
    Xfull = Xs.clone()
    Xfull[hw] = Xsh
    arms["C_SHUFFLE_features"] = ade_h.gather(1, pred_pick(hw, w, Xfull)[:, None])[:, 0]
    # the rule RESTRICTED to the model's own top-32 (a deployable re-rank)
    pr = torch.cat([Xs[hw], torch.ones(int(hw.sum()), N, 1, dtype=Xs.dtype)], -1) @ w
    masked = pr.gather(1, order_h[:, :32])
    arms["ridge_rerank_top32"] = ade_h.gather(
        1, order_h.gather(1, masked.argmin(1)[:, None]))[:, 0]
    # ⛔ the model's OWN selection is restricted to `reach_keep`; an arm allowed to
    # pick a masked-out candidate is not comparable to it, so the restricted arm is
    # reported beside the free one rather than instead of it.
    keep_h = torch.isfinite(rank[hw])
    pr_k = pr.masked_fill(~keep_h, float("inf"))
    arms["ridge_rerank_reach_only"] = ade_h.gather(1, pr_k.argmin(1)[:, None])[:, 0]
    arms["C_ORACLE_reach_only"] = ade_h.masked_fill(~keep_h, float("inf")).min(dim=1).values

    # ⛔ A NEGATIVE FROM A LINEAR PROBE IS NOT A NEGATIVE ABOUT LEARNABILITY. The
    # implication runs one way only: a linear oracle BEATING the baseline proves the
    # target is reachable; a linear oracle FAILING proves only "not by a linear map".
    # So the same question is asked again with a NONLINEAR learner, on the same split,
    # with the same shuffled-feature control -- because "closes 2.3 %" is a claim about
    # a function class until it is asked of a second one.
    from sklearn.ensemble import HistGradientBoostingRegressor          # noqa: E402
    Xf = X[fit_w].reshape(-1, dd).numpy()
    yf = ade[fit_w].reshape(-1).numpy()
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                       max_depth=6, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.2,
                                       random_state=SEED).fit(Xf, yf)
    Ph = torch.tensor(gb.predict(X[hw].reshape(-1, dd).numpy()),
                      dtype=torch.float64).reshape(int(hw.sum()), N)
    arms["gbdt_rerank"] = ade_h.gather(1, Ph.argmin(1)[:, None])[:, 0]
    arms["gbdt_rerank_reach_only"] = ade_h.gather(
        1, Ph.masked_fill(~keep_h, float("inf")).argmin(1)[:, None])[:, 0]
    Psh = torch.tensor(gb.predict(Xsh.reshape(-1, dd).numpy()),
                       dtype=torch.float64).reshape(int(hw.sum()), N)
    arms["C_SHUFFLE_gbdt"] = ade_h.gather(1, Psh.argmin(1)[:, None])[:, 0]
    print(f"[sel] GBDT: {gb.n_iter_} iters, fit rows {Xf.shape[0]} x {dd} features",
          flush=True)

    def boot(v):
        r = CI.episode_cluster_bootstrap(np.asarray(v, dtype=np.float64), e_h,
                                         n_boot=N_BOOT, seed=SEED)
        return {k: r[k] for k in ("mean", "lo", "hi", "n_windows", "n_episodes")}

    def pboot(a, b):
        r = CI.paired_episode_cluster_bootstrap(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64),
            e_h, n_boot=N_BOOT, seed=SEED)
        return {k: r[k] for k in ("delta", "lo", "hi", "separated")}

    base = arms["model_argmax"].numpy()
    ceil = arms["C_ORACLE_ceiling"].numpy()
    gap = float(base.mean() - ceil.mean())
    rec = {"_what": "H-RL-SELECT-HEADROOM-1: how much of the selection gap is reachable "
                    "by re-ranking what the model already emits, on the PROJECTED "
                    "(collision-free) fan",
           "_evidence_class": "MEASURED (ours; 0 GPU)",
           "_tier": "T0 selection-quality readout on the EMITTED fan; never a driving claim",
           "arena": "contact-projected fan (fan_contact = 0.000000 by construction)",
           "n_features": dd, "features": names, "ridge_lambda": lam,
           "split": {"fit_windows": int(fit_w.sum()), "heldout_windows": int(hw.sum()),
                     "fit_episodes": len(fit_eps),
                     "heldout_episodes": len(eps) - len(fit_eps),
                     "episode_disjoint": True, "fit_rows": int(fit_w.sum()) * N},
           "selection_gap_m": gap, "arms": {}}
    for k, v in arms.items():
        rec["arms"][k] = {"ade_m": float(v.mean()), **boot(v.numpy()),
                          "paired_vs_model": pboot(v.numpy(), base),
                          "gap_closed_pct": 100.0 * (base.mean() - float(v.mean())) / gap
                          if gap > 0 else float("nan")}
        a = rec["arms"][k]
        print(f"[sel] {k:24s} ADE {a['ade_m']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}]  "
              f"vs model {a['paired_vs_model']['delta']:+.4f} "
              f"[{a['paired_vs_model']['lo']:+.4f}, {a['paired_vs_model']['hi']:+.4f}] "
              f"sep={a['paired_vs_model']['separated']}  gap closed "
              f"{a['gap_closed_pct']:+.1f} %", flush=True)

    # ------------------------------------------------------------------ #
    # ⛔⛔ THE CONTROL THAT DECIDES WHETHER THE "GAP" IS AN OBJECTIVE AT ALL.
    # `oracle_in_fan` is a BEST-OF-N statistic: min ADE over N samples falls with N
    # by construction, whether or not any selector could have known which sample to
    # take. If the curve is still falling steeply at N = 128, then "close the gap to
    # the oracle" is not a reachable target -- it is asking a selector to guess which
    # of 128 plausible futures happens to match THIS human, which is unpredictable in
    # principle. ⇒ measure the decay, and read the gap against it rather than against
    # zero. Random subsets, 40 draws each, on the HELD-OUT windows only.
    # ------------------------------------------------------------------ #
    rr2 = np.random.default_rng(23)
    curve = []
    for k in (1, 2, 4, 8, 16, 32, 64, 128):
        vals = []
        for _ in range(40 if k < N else 1):
            cols = torch.tensor(rr2.choice(N, size=k, replace=False))
            vals.append(ade_h[:, cols].min(dim=1).values)
        v = torch.stack(vals).mean(0)
        curve.append({"n": k, "min_ade_m": float(v.mean()),
                      "median_ade_m": float(ade_h[:, cols].median(dim=1).values.mean())})
        print(f"[sel] best-of-N  N={k:4d}  min-ADE {curve[-1]['min_ade_m']:.4f}",
              flush=True)
    import math
    lx = np.log([c["n"] for c in curve][1:])
    ly = np.log([c["min_ade_m"] for c in curve][1:])
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    ss = float(1.0 - res[0] / ((ly - ly.mean()) ** 2).sum()) if len(res) else float("nan")
    rec["best_of_n_control"] = {
        "_what": "min ADE over a RANDOM subset of k candidates, 40 draws, held-out windows",
        "curve": curve, "exponent": float(coef[0]), "r2": ss,
        "fit_window": "N = 2..128", "n_points": len(lx),
        "still_falling_at_128": bool(curve[-1]["min_ade_m"] < curve[-2]["min_ade_m"] - 1e-4),
        "random_best_of_128_m": curve[-1]["min_ade_m"],
        "model_argmax_m": float(base.mean()),
        "_reading": ("a RANDOM best-of-128 that already beats the model's argmax means "
                     "the 'gap' is substantially a best-of-N statistic and not a "
                     "selection skill a ranker could acquire; the exponent carries its "
                     "fit window, R2 and n, per the programme's own rule"),
    }
    print(f"[sel] best-of-N exponent {coef[0]:+.3f} (R2 {ss:.3f}, n={len(lx)}, "
          f"window N=2..128) | random best-of-128 {curve[-1]['min_ade_m']:.4f} vs "
          f"model argmax {base.mean():.4f}", flush=True)

    rec["controls"] = {
        "C_CONST_must_be_far_worse_than_model": bool(
            rec["arms"]["C_CONST_index0"]["ade_m"] > rec["arms"]["model_argmax"]["ade_m"]),
        "C_RANDOM_must_be_far_worse_than_model": bool(
            rec["arms"]["C_RANDOM_20seed"]["ade_m"] > rec["arms"]["model_argmax"]["ade_m"]),
        "C_SHUFFLE_gbdt_must_collapse_to_chance": bool(
            rec["arms"]["C_SHUFFLE_gbdt"]["ade_m"]
            > 0.5 * (rec["arms"]["C_RANDOM_20seed"]["ade_m"]
                     + rec["arms"]["model_argmax"]["ade_m"])),
        "C_SHUFFLE_must_collapse_to_chance": bool(
            rec["arms"]["C_SHUFFLE_features"]["ade_m"]
            > 0.5 * (rec["arms"]["C_RANDOM_20seed"]["ade_m"]
                     + rec["arms"]["model_argmax"]["ade_m"])),
        "C_ORACLE_is_the_minimum": bool(
            all(rec["arms"]["C_ORACLE_ceiling"]["ade_m"] <= v["ade_m"]
                for v in rec["arms"].values())),
    }
    rec["controls"]["ALL_PASS"] = all(rec["controls"].values())
    print("[sel] CONTROLS:", rec["controls"], flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print(f"[sel] -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
