#!/usr/bin/env python
"""THE FIX THAT NEEDS NO NEW PARAMETERS: the head's EXPECTED sustained curvature.

⭐ THE DEFECT (MEASURED, `ROOT_CAUSE.md`). `canonical_controls` can command
exactly two SUSTAINED curvatures, 0 and +/-GOAL_KAPPA_TURN = 0.08 (R 12.5 m),
because NUDGE_* / LANE_CHANGE_* are S-curves with zero net heading change. The
corpus curves at R 100-1000 m. So the goal is a **3-level quantisation of a
continuous quantity**, and on 90 % of GT-turn windows `LANE_KEEP` is the
vocabulary-optimal token. No re-fit of `lat_head` changes that.

⭐ THE FIX. The head already emits a full posterior over the 8 tokens. Define
the per-token SUSTAINED curvature (TURN_L +K, TURN_R -K, every other token 0 --
S-curves contribute zero net heading by construction) and take its EXPECTATION:

    kappa_hat = K * (p[TURN_L] - p[TURN_R]),      p = softmax(lat_logits / T)

That is a CONTINUOUS signed curvature in [-K, +K] -- exactly the missing range --
computed from the SHIPPED head with **zero new parameters and zero training**.
As T -> 0 the posterior concentrates and kappa_hat -> the current hard decode,
so the change is a strict generalisation with the legacy behaviour at one end.

⛔ WHAT IS BEING TESTED, AND BOTH OUTCOMES ARE COMMITTED (SPEC.md amendment):
  * SUCCEEDS if `kappa_hat` tracks the road's curvature better than the hard
    decode does -- lower RMSE against the REACHABLE target (gt clipped to
    +/-K), and a materially higher fraction of GT-turn windows carrying a
    NON-ZERO, CORRECTLY SIGNED sustained curvature.
  * FAILS if it does not beat the hard decode, in which case the posterior
    carries no usable magnitude information and a trained magnitude head (a
    Linear on the banked `intent`) is the next step rather than this.

⛔ CONTROLS, each of which must read a value fixed in advance:
  C1 T -> 0 must reproduce the HARD decode's kappa exactly (the legacy path).
  C2 a UNIFORM-POSTERIOR control (logits replaced by zeros) must score
     kappa_hat == 0 everywhere and therefore EXACTLY the LANE_KEEP baseline --
     without it, "soft beats hard" could be an artefact of shrinking magnitudes
     toward a target whose median is ~0.
  C3 a SHUFFLED-LOGIT control (rows permuted across windows) must destroy the
     correlation; if it does not, the win is not about this head's evidence.
  C4 the ZERO baseline (kappa == 0 everywhere) is reported in every table --
     a predictor of a near-zero-median quantity that cannot beat zero has
     added nothing. n printed for every cell.

Run: python gm_soft_kappa.py --logits <npz> --gt-kappa <npz> --out <json>
     python gm_soft_kappa.py --intent <intent npz> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

GOAL_KAPPA_TURN = 0.08          # refa_v1.py:119


def softmax(z, T):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default=None)
    ap.add_argument("--gt-kappa", default=None)
    ap.add_argument("--intent", default=None)
    ap.add_argument("--kappa-turn", type=float, default=GOAL_KAPPA_TURN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.intent:
        Z = np.load(a.intent, allow_pickle=False)
        gt = Z["gt_kappa"].astype(np.float64)
        lg = Z["lat_logits"].astype(np.float64)
        names = [str(x) for x in Z["lat_names"]]
        eid = Z["clip_index"].astype(int)
        src = a.intent
    else:
        Z = np.load(a.logits, allow_pickle=False)
        G = np.load(a.gt_kappa, allow_pickle=False)
        names = [str(x) for x in Z["lat_names"]]
        key = {(int(c), int(w)): i for i, (c, w) in
               enumerate(zip(Z["clip_index"], Z["ws"]))}
        g = np.full(len(Z["ws"]), np.nan)
        for c, w, kk in zip(G["clip_index"], G["ws"], G["gt_kappa"]):
            i = key.get((int(c), int(w)))
            if i is not None:
                g[i] = kk
        m = np.isfinite(g)
        gt = g[m]
        lg = Z["lat_logits"].astype(np.float64)[m]
        eid = Z["clip_index"].astype(int)[m]
        src = a.logits

    K = float(a.kappa_turn)
    TL, TR = names.index("TURN_L"), names.index("TURN_R")
    n = len(gt)
    # the REACHABLE target: the goal cannot command more than +/-K
    tgt = np.clip(gt, -K, K)

    def hard_kappa(logits):
        d = logits.argmax(-1)
        k = np.zeros(len(d))
        k[d == TL] = K
        k[d == TR] = -K
        return k

    def scores(k, label):
        err = k - tgt
        turn = np.abs(gt) > 1e-2                     # a real turn: R <= 100 m
        nz_signed = (np.abs(k) > 1e-9) & (np.sign(k) == np.sign(gt))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((tgt - tgt.mean()) ** 2))
        return {
            "arm": label, "n": int(n),
            "rmse_vs_reachable_target": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
            "pearson_r_vs_gt": (float(np.corrcoef(k, gt)[0, 1])
                                if k.std() > 0 else 0.0),
            "n_gt_real_turn_kappa_gt_1e-2": int(turn.sum()),
            "frac_real_turns_with_nonzero_correct_sign": (
                float(nz_signed[turn].mean()) if turn.sum() else None),
            "frac_all_windows_nonzero_kappa": float((np.abs(k) > 1e-9).mean()),
            "frac_straight_windows_nonzero": float(
                (np.abs(k) > 1e-9)[~turn].mean()) if (~turn).sum() else None,
            "mean_abs_kappa": float(np.abs(k).mean()),
        }

    rows = [scores(np.zeros(n), "ZERO baseline (C4)"),
            scores(hard_kappa(lg), "HARD argmax (shipped)")]
    temps = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    for T in temps:
        p = softmax(lg, T)
        rows.append(scores(K * (p[:, TL] - p[:, TR]), f"SOFT T={T}"))

    # ------------------------------- CONTROLS ------------------------------ #
    ctrl = {}
    # C1 — T -> 0 reproduces the hard decode
    p0 = softmax(lg, 1e-3)
    k0 = K * (p0[:, TL] - p0[:, TR])
    kh = hard_kappa(lg)
    ctrl["C1_T_to_zero_reproduces_hard"] = {
        "max_abs_diff": float(np.abs(k0 - kh).max()),
        "passes": bool(np.abs(k0 - kh).max() < 1e-9)}
    # C2 — uniform posterior must give exactly zero
    pu = softmax(np.zeros_like(lg), 1.0)
    ku = K * (pu[:, TL] - pu[:, TR])
    ctrl["C2_uniform_posterior_is_zero"] = {
        "max_abs_kappa": float(np.abs(ku).max()),
        "rmse_equals_zero_baseline": bool(
            abs(np.sqrt(np.mean((ku - tgt) ** 2))
                - rows[0]["rmse_vs_reachable_target"]) < 1e-12),
        "passes": bool(np.abs(ku).max() < 1e-12)}
    # C3 — shuffled logits must destroy the correlation
    rng = np.random.default_rng(a.seed)
    rs = []
    for _ in range(200):
        ps = softmax(lg[rng.permutation(n)], 1.0)
        ks = K * (ps[:, TL] - ps[:, TR])
        rs.append(float(np.corrcoef(ks, gt)[0, 1]) if ks.std() > 0 else 0.0)
    best_soft = min((r for r in rows if r["arm"].startswith("SOFT")),
                    key=lambda r: r["rmse_vs_reachable_target"])
    ctrl["C3_shuffled_logits_destroy_correlation"] = {
        "mean_r": float(np.mean(rs)), "sd_r": float(np.std(rs)),
        "real_r_at_best_soft": best_soft["pearson_r_vs_gt"],
        "passes": bool(abs(np.mean(rs)) < 0.06
                       and best_soft["pearson_r_vs_gt"] > np.mean(rs) + 4 * np.std(rs))}

    hard = rows[1]
    zero = rows[0]
    verdict = ("SUCCEEDS" if (best_soft["rmse_vs_reachable_target"]
                              < hard["rmse_vs_reachable_target"]
                              and best_soft["frac_real_turns_with_nonzero_correct_sign"]
                              > hard["frac_real_turns_with_nonzero_correct_sign"])
               else "FAILS")
    out = {"source": src, "n_windows": int(n), "kappa_turn": K,
           "n_episodes": int(len(np.unique(eid))),
           "target": "gt_kappa clipped to +/-GOAL_KAPPA_TURN (the reachable range)",
           "rows": rows, "controls": ctrl,
           "best_soft": best_soft, "hard": hard, "zero": zero,
           "verdict": verdict}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"n={n}  episodes={len(np.unique(eid))}  K={K}  source={src}")
    print(f"{'arm':<24}{'RMSE':>9}{'MAE':>9}{'R2':>8}{'r':>7}"
          f"{'turn_nz_sign':>14}{'straight_nz':>12}{'mean|k|':>9}")
    for r in rows:
        f1 = r["frac_real_turns_with_nonzero_correct_sign"]
        f2 = r["frac_straight_windows_nonzero"]
        print(f"{r['arm']:<24}{r['rmse_vs_reachable_target']:>9.5f}"
              f"{r['mae']:>9.5f}{r['r2']:>8.4f}{r['pearson_r_vs_gt']:>7.3f}"
              f"{(f1 if f1 is not None else float('nan')):>14.4f}"
              f"{(f2 if f2 is not None else float('nan')):>12.4f}"
              f"{r['mean_abs_kappa']:>9.5f}")
    print()
    print("CONTROLS: " + json.dumps({k: v["passes"] for k, v in ctrl.items()}))
    print(f"C3 shuffled r = {ctrl['C3_shuffled_logits_destroy_correlation']['mean_r']:+.4f}"
          f" +/- {ctrl['C3_shuffled_logits_destroy_correlation']['sd_r']:.4f}"
          f"   real r = {best_soft['pearson_r_vs_gt']:+.4f}")
    print(f"VERDICT: {verdict}  (best {best_soft['arm']})")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
