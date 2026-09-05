#!/usr/bin/env python
"""STEP 2, RE-SPECIFIED: a CURVATURE-MAGNITUDE head on the banked `intent`.

⭐ WHY NOT THE CLASSIFIER RE-FIT THAT WAS PLANNED. `ROOT_CAUSE.md` MEASURED that
on 90.15 % of GT-turn windows `LANE_KEEP` is the VOCABULARY-OPTIMAL token: the
lateral goal can command exactly two SUSTAINED curvatures, 0 and 0.08 (R 12.5 m),
and the corpus curves at R 100-1000 m. Re-fitting a classifier over that
vocabulary cannot express a 200 m curve, so it cannot fix the defect. What is
missing is a MAGNITUDE, and the same banked `intent` is where it must come from.

⭐ WHY IT IS CHEAP AND WHY THE TRUNK IS UNTOUCHED. `lat_head` is
`LayerNorm(1024) -> Linear(1024, 8)` over `intent` (`refa_v1.py:1218`, applied at
`:1956`). This fits a `Linear(1024, 1)` on the SAME frozen `LayerNorm(intent)`.
The trunk and the world model are not merely frozen -- they are NOT IN THE
OPTIMISATION AT ALL, which is asserted here by construction (only the new
weights are created) rather than by a parameter count.

⛔ ADMISSIBILITY (PI 2026-08-03). The TARGET is `gt_kappa`, derived from the
ego's FUTURE poses -- a LABEL, and labels may use ego. The INPUT is `intent`,
which `plan()` builds at `refa_v1.py:2156` with `ego=None`, so at inference it is
VISION + NAV only. The situation classifier's output enters nowhere. nav is
weakly route-informative and INHERITED; the `nav_shuffled` arm below scores the
same fit on `intent` recomputed under a permuted nav, so a fit that leans on the
route signal is visible rather than assumed away.

⛔ CONTROLS -- CLAUDE.md's four probe traps, each one instrumented:
  * TRAP 1 (normalise by raw energy): R^2 is reported against the TEST split's
    own mean, and a CONSTANT-ONLY arm is scored in the same table. It must read
    R^2 <= 0 exactly; if a learned arm ties it, the panel says NO INFORMATION.
  * TRAP 2/3 (tune on the scored split): lambda is chosen on a FIT-INTERNAL
    episode-disjoint TUNE split. The TEST split is scored ONCE and never tuned
    on. Both are printed.
  * TRAP 4 (n << d): `n_train`, `n_episodes_train` and `d` are printed for every
    cell. The EFFECTIVE n is EPISODES, not windows, and that is stated.
  * a SHUFFLED-TARGET arm (targets permuted within train) must collapse to the
    constant predictor on TEST.
  * a RAW-INPUT floor: the SHIPPED head's own hard kappa, and the ZERO
    predictor. A learned magnitude that cannot beat zero has added nothing.

⛔ SPLIT BY EPISODE, never by window: windows from one episode are near
duplicates and a window-level split would manufacture a success.

Run: python gm_kappa_head.py --intent <npz> --out <json> --save-head <pt>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

GOAL_KAPPA_MAX = 0.2            # refa_v1.py:118 — PlanConfig.kappa_max


def episode_split(eids, seed, fracs=(0.6, 0.2, 0.2)):
    u = np.unique(eids)
    rng = np.random.default_rng(seed)
    p = rng.permutation(u)
    n1 = int(round(fracs[0] * len(u)))
    n2 = n1 + int(round(fracs[1] * len(u)))
    tr, tu, te = set(p[:n1]), set(p[n1:n2]), set(p[n2:])
    return (np.array([e in tr for e in eids]),
            np.array([e in tu for e in eids]),
            np.array([e in te for e in eids]))


def ridge_fit(X, y, lam):
    """Closed-form ridge with an unpenalised intercept."""
    mu, ym = X.mean(0), y.mean()
    Xc, yc = X - mu, y - ym
    d = Xc.shape[1]
    A = Xc.T @ Xc + lam * np.eye(d)
    w = np.linalg.solve(A, Xc.T @ yc)
    b = ym - mu @ w
    return w, b


def metrics(pred, y, gt_raw, name, n_tr, n_ep_tr, d, lam):
    err = pred - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    turn = np.abs(gt_raw) > 1e-2
    nz_sign = (np.abs(pred) > 1e-3) & (np.sign(pred) == np.sign(gt_raw))
    return {
        "arm": name, "lambda": lam,
        "n_test": int(len(y)), "n_train": int(n_tr),
        "n_episodes_train": int(n_ep_tr), "d": int(d),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1 - np.sum(err ** 2) / ss_tot) if ss_tot > 0 else None,
        "pearson_r": (float(np.corrcoef(pred, y)[0, 1])
                      if pred.std() > 1e-12 and y.std() > 1e-12 else 0.0),
        "n_real_turns_in_test": int(turn.sum()),
        "frac_real_turns_nonzero_correct_sign": (float(nz_sign[turn].mean())
                                                 if turn.sum() else None),
        "frac_straights_nonzero": (float((np.abs(pred) > 1e-3)[~turn].mean())
                                   if (~turn).sum() else None),
        "mean_abs_pred": float(np.abs(pred).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint whose lat_head.0 LayerNorm is applied "
                         "(the head's own frozen normaliser)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=GOAL_KAPPA_MAX)
    ap.add_argument("--save-head", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Z = np.load(a.intent, allow_pickle=False)
    I = Z["intent"].astype(np.float64)
    Is = Z["intent_navshuf"].astype(np.float64)
    gt = Z["gt_kappa"].astype(np.float64)
    eid = Z["clip_index"].astype(int)
    lg = Z["lat_logits"].astype(np.float64)
    names = [str(x) for x in Z["lat_names"]]
    TL, TR = names.index("TURN_L"), names.index("TURN_R")
    n, d = I.shape

    # the head's own frozen LayerNorm, so the new Linear is a drop-in sibling
    ln_info = "none (raw intent)"
    if a.ckpt:
        import torch
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        gw = [k for k in sd if k.endswith("lat_head.0.weight")]
        if gw:
            k0 = gw[0]
            w = sd[k0].float().numpy()
            b = sd[k0.replace("weight", "bias")].float().numpy()
            mu = I.mean(-1, keepdims=True)
            sg = I.std(-1, keepdims=True)
            I = (I - mu) / (sg + 1e-5) * w + b
            mus = Is.mean(-1, keepdims=True); sgs = Is.std(-1, keepdims=True)
            Is = (Is - mus) / (sgs + 1e-5) * w + b
            ln_info = f"applied from {k0}"

    y = np.clip(gt, -a.clip, a.clip)
    m_tr, m_tu, m_te = episode_split(eid, a.seed)
    n_ep_tr = len(np.unique(eid[m_tr]))
    rows = []

    # ---- lambda on the FIT-INTERNAL TUNE split ONLY ---------------------- #
    lams = [10.0 ** e for e in range(-4, 9)]
    tune = []
    for lam in lams:
        w, b = ridge_fit(I[m_tr], y[m_tr], lam)
        p = I[m_tu] @ w + b
        tune.append((float(np.sqrt(np.mean((p - y[m_tu]) ** 2))), lam))
    tune.sort()
    lam_star = tune[0][1]

    # ---- the arms, scored ONCE on TEST ----------------------------------- #
    w, b = ridge_fit(I[m_tr], y[m_tr], lam_star)
    pred = np.clip(I[m_te] @ w + b, -a.clip, a.clip)
    rows.append(metrics(pred, y[m_te], gt[m_te], "RIDGE on intent (LEARNED)",
                        m_tr.sum(), n_ep_tr, d, lam_star))

    rows.append(metrics(np.full(m_te.sum(), y[m_tr].mean()), y[m_te], gt[m_te],
                        "CONSTANT-ONLY (control)", m_tr.sum(), n_ep_tr, 0, None))
    rows.append(metrics(np.zeros(m_te.sum()), y[m_te], gt[m_te],
                        "ZERO (floor)", m_tr.sum(), n_ep_tr, 0, None))

    hk = np.zeros(n)
    hk[lg.argmax(-1) == TL] = 0.08
    hk[lg.argmax(-1) == TR] = -0.08
    rows.append(metrics(hk[m_te], y[m_te], gt[m_te],
                        "SHIPPED head hard kappa (floor)",
                        m_tr.sum(), n_ep_tr, 0, None))

    rng = np.random.default_rng(a.seed)
    ysh = y.copy()
    ysh[m_tr] = rng.permutation(ysh[m_tr])
    ws, bs = ridge_fit(I[m_tr], ysh[m_tr], lam_star)
    rows.append(metrics(np.clip(I[m_te] @ ws + bs, -a.clip, a.clip),
                        y[m_te], gt[m_te], "SHUFFLED-TARGET (control)",
                        m_tr.sum(), n_ep_tr, d, lam_star))

    wn, bn = ridge_fit(Is[m_tr], y[m_tr], lam_star)
    rows.append(metrics(np.clip(Is[m_te] @ wn + bn, -a.clip, a.clip),
                        y[m_te], gt[m_te], "RIDGE on NAV-SHUFFLED intent (P3)",
                        m_tr.sum(), n_ep_tr, d, lam_star))

    learned = rows[0]
    const = rows[1]
    zero = rows[2]
    ship = rows[3]
    shuf = rows[4]
    ctrl = {
        "constant_reads_no_information": {
            "r2": const["r2"],
            "passes": bool(const["r2"] is not None and const["r2"] <= 1e-9)},
        "shuffled_target_collapses": {
            "r2": shuf["r2"],
            "passes": bool(shuf["r2"] is not None and shuf["r2"] <= 0.02)},
        "learned_beats_zero_floor": {
            "learned_rmse": learned["rmse"], "zero_rmse": zero["rmse"],
            "passes": bool(learned["rmse"] < zero["rmse"])},
        "learned_beats_shipped_head": {
            "learned_rmse": learned["rmse"], "shipped_rmse": ship["rmse"],
            "passes": bool(learned["rmse"] < ship["rmse"])},
        "lambda_not_at_grid_edge": {
            "lambda": lam_star, "grid": [lams[0], lams[-1]],
            "passes": bool(lams[0] < lam_star < lams[-1]),
            "note": ("lambda pinned at the top of the grid is the n<<d "
                     "signature: maximal shrinkage wins because there is no "
                     "power, which reads as absence but is underpowering")},
        "power": {"n_train_windows": int(m_tr.sum()),
                  "n_train_EPISODES_effective_n": int(n_ep_tr),
                  "d": int(d),
                  "windows_per_dim": float(m_tr.sum() / d),
                  "EPISODES_per_dim": float(n_ep_tr / d),
                  "note": ("windows within an episode are near-duplicates, so "
                           "the effective n is EPISODES. episodes/d << 1 means "
                           "this fit is underpowered BY CONSTRUCTION and a "
                           "null must not be read as absence")},
    }

    navsh = rows[5]
    ctrl["beats_constant_on_turn_sign"] = {
        "learned": learned["frac_real_turns_nonzero_correct_sign"],
        "constant": const["frac_real_turns_nonzero_correct_sign"],
        "passes": bool(learned["frac_real_turns_nonzero_correct_sign"] is not None
                       and const["frac_real_turns_nonzero_correct_sign"] is not None
                       and learned["frac_real_turns_nonzero_correct_sign"]
                       > const["frac_real_turns_nonzero_correct_sign"] + 0.10),
        "note": ("a CONSTANT non-zero prediction already gets the sign right "
                 "about half the time by chance; beating ZERO is not enough")}
    ctrl["not_a_nav_echo"] = {
        "r2_true_nav": learned["r2"], "r2_nav_shuffled": navsh["r2"],
        "frac_of_r2_surviving_nav_shuffle": (
            float(navsh["r2"] / learned["r2"]) if learned["r2"] else None),
        "passes": bool(learned["r2"] and navsh["r2"] is not None
                       and navsh["r2"] / learned["r2"] > 0.5),
        "note": ("nav on PhysicalAI is ultimately supplied from the ego's own "
                 "FUTURE path. A fit whose skill vanishes when nav is permuted "
                 "is reading the route, not the road.")}
    verdict = ("SUCCEEDS" if all(
        ctrl[k]["passes"] for k in
        ("constant_reads_no_information", "shuffled_target_collapses",
         "learned_beats_zero_floor", "learned_beats_shipped_head",
         "beats_constant_on_turn_sign", "not_a_nav_echo"))
        else "FAILS")

    out = {"intent": a.intent, "layernorm": ln_info,
           "n_windows": int(n), "d": int(d),
           "n_episodes": int(len(np.unique(eid))),
           "split": {"episodes_train": int(n_ep_tr),
                     "episodes_tune": int(len(np.unique(eid[m_tu]))),
                     "episodes_test": int(len(np.unique(eid[m_te]))),
                     "windows_train": int(m_tr.sum()),
                     "windows_tune": int(m_tu.sum()),
                     "windows_test": int(m_te.sum()),
                     "by": "EPISODE (clip_index), never by window", "seed": a.seed},
           "lambda_selection": {"grid": lams, "chosen": lam_star,
                                "chosen_on": "TUNE split (fit-internal), never TEST",
                                "tune_rmse_sorted": [[r, l] for r, l in tune[:4]]},
           "target": f"gt_kappa clipped to +/-{a.clip}",
           "rows": rows, "controls": ctrl, "verdict": verdict}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    if a.save_head and verdict == "SUCCEEDS":
        import torch
        torch.save({"weight": torch.tensor(w, dtype=torch.float32),
                    "bias": torch.tensor(float(b), dtype=torch.float32),
                    "layernorm": ln_info, "lambda": lam_star,
                    "clip": a.clip, "d": int(d),
                    "target": "gt_kappa (sustained curvature, 1/m)",
                    "units": "1/m", "control_units": "kappa",
                    "trained_on": a.intent,
                    "episodes_train": int(n_ep_tr),
                    "provenance": "gm_kappa_head.py ridge, episode-disjoint"},
                   a.save_head)
        print(f"[saved] {a.save_head}")

    print(f"n={n} d={d} episodes={len(np.unique(eid))}  LN: {ln_info}")
    print(f"split: train {int(m_tr.sum())}w/{n_ep_tr}ep  tune {int(m_tu.sum())}w  "
          f"test {int(m_te.sum())}w/{len(np.unique(eid[m_te]))}ep   lambda*={lam_star:g}")
    print(f"{'arm':<36}{'RMSE':>10}{'MAE':>9}{'R2':>9}{'r':>7}"
          f"{'turn_nz_sgn':>12}{'str_nz':>9}")
    for r in rows:
        f1 = r["frac_real_turns_nonzero_correct_sign"]
        f2 = r["frac_straights_nonzero"]
        print(f"{r['arm']:<36}{r['rmse']:>10.5f}{r['mae']:>9.5f}"
              f"{(r['r2'] if r['r2'] is not None else float('nan')):>9.4f}"
              f"{r['pearson_r']:>7.3f}"
              f"{(f1 if f1 is not None else float('nan')):>12.4f}"
              f"{(f2 if f2 is not None else float('nan')):>9.4f}")
    print()
    print("CONTROLS: " + json.dumps({k: v.get("passes") for k, v in ctrl.items()
                                     if "passes" in v}))
    print(f"POWER: {ctrl['power']['n_train_EPISODES_effective_n']} episodes / "
          f"{d} dims = {ctrl['power']['EPISODES_per_dim']:.4f} episodes per dim")
    print(f"VERDICT: {verdict}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
