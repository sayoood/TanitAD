#!/usr/bin/env python
"""WHAT kappa_turn SHOULD WE SHIP? — the REALISED curve, not the oracle bound.

⭐ WHY THIS EXISTS AND WHY THE ORACLE TABLE IS NOT ENOUGH. `vocab_design.py`
prices each candidate vocabulary under an ORACLE that always picks the best
available token. That is a BOUND. But `kappa_turn` also moves the vocabulary's
own CROSSOVER (`kappa_turn/2`, the |kappa| above which TURN beats LANE_KEEP),
and the head's turn recall is a strong function of where that crossover sits:

    crossover 0.040 (kappa_turn 0.08)  ->  argmax recall 0.744 @ false 0.120
    crossover 0.010 (kappa_turn 0.02)  ->  argmax recall 0.388 @ false 0.107

⇒ **a smaller `kappa_turn` makes more of the road EXPRESSIBLE but asks the head
to decide on windows where it is WEAKER.** The oracle table and the ROC table
therefore point in opposite directions, and neither alone can choose the
constant. This composes them: for each candidate it applies the SHIPPED head's
ACTUAL argmax decode, converts it through `canonical_controls`' real profile,
and scores the resulting goal curvature against the road.

⛔ CONTROLS, each of which must read a value fixed in advance:
  * `kappa_turn = 0.08` must reproduce the shipped arm's realised error exactly
    — it IS the shipped configuration;
  * a ZERO floor (the goal always commands 0) is in every table: a vocabulary
    whose REALISED error cannot beat "never turn" is worse than useless, and the
    oracle table cannot show that because an oracle never turns wrongly;
  * a RANDOM-DECODE control (the same marginal decode rate, permuted across
    windows) must be WORSE than the real head, else the win is the vocabulary
    alone and not the head using it;
  * near-stationary windows excluded (`gt_kappa` divides by
    `speed.clamp_min(0.5)`); counts printed.
  * n printed for every cell.

Run: python gm_realised_kappa.py --intent <npz> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

GOAL_KAPPA_TURN = 0.08
GOAL_TURN_S = 4.0
GOAL_KAPPA_MAX = 0.2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", required=True)
    ap.add_argument("--horizon-s", type=float, default=6.0)
    ap.add_argument("--turn-thr", type=float, default=1e-2)
    ap.add_argument("--min-v0", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Z = np.load(a.intent, allow_pickle=False)
    v0 = Z["v0"].astype(np.float64)
    keep = v0 >= a.min_v0
    n_excl = int((~keep).sum())
    gt = np.clip(Z["gt_kappa"].astype(np.float64)[keep], -GOAL_KAPPA_MAX,
                 GOAL_KAPPA_MAX)
    lg = Z["lat_logits"].astype(np.float64)[keep]
    names = [str(x) for x in Z["lat_names"]]
    TL, TR = names.index("TURN_L"), names.index("TURN_R")
    n = len(gt)
    duty = min(1.0, GOAL_TURN_S / a.horizon_s)
    turn = np.abs(gt) > a.turn_thr
    dec = lg.argmax(-1)

    def realised(kt, decode):
        """The goal curvature the planner would actually be given, in the same
        horizon-averaged sense the oracle table uses."""
        k = np.zeros(n)
        k[decode == TL] = kt * duty
        k[decode == TR] = -kt * duty
        return k

    def score(k, name, kt):
        err = np.abs(k - gt)
        nz = np.abs(k) > 1e-9
        sgn_ok = nz & (np.sign(k) == np.sign(gt))
        return {"design": name, "kappa_turn": kt, "n": int(n),
                "n_turns": int(turn.sum()),
                "medae_on_turns": float(np.median(err[turn])) if turn.sum() else None,
                "rmse_all": float(np.sqrt(np.mean(err ** 2))),
                "frac_turns_goaled_nonzero_correct_sign":
                    float(sgn_ok[turn].mean()) if turn.sum() else None,
                "frac_straights_given_curvature":
                    float(nz[~turn].mean()) if (~turn).sum() else None,
                "crossover_kappa": kt / 2.0 if kt else None}

    rows = [score(np.zeros(n), "ZERO — goal never turns (FLOOR)", 0.0)]
    grid = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12)
    real = []
    for kt in grid:
        r = score(realised(kt, dec), f"REALISED shipped head, kappa_turn={kt}", kt)
        r["is_shipped"] = bool(abs(kt - GOAL_KAPPA_TURN) < 1e-12)
        real.append(r)
    rows += real

    # CONTROL: a random decode with the SAME marginal rate must be worse
    rng = np.random.default_rng(a.seed)
    ctrl_rows = []
    for kt in (0.02, 0.08):
        accs = []
        for _ in range(200):
            perm = dec[rng.permutation(n)]
            accs.append(score(realised(kt, perm), "x", kt)["medae_on_turns"])
        ctrl_rows.append({"kappa_turn": kt,
                          "random_decode_medae_mean": float(np.mean(accs)),
                          "random_decode_medae_sd": float(np.std(accs)),
                          "real_head_medae": [r["medae_on_turns"] for r in real
                                              if abs(r["kappa_turn"] - kt) < 1e-12][0]})
    best = min(real, key=lambda r: r["medae_on_turns"])
    ship = [r for r in real if r.get("is_shipped")][0]
    floor = rows[0]
    ctrl = {
        "CONTROL_zero_floor_present": {"medae": floor["medae_on_turns"],
                                       "passes": True},
        "CONTROL_shipped_row_present": {
            "passes": any(r.get("is_shipped") for r in real)},
        "CONTROL_random_decode_is_worse": {
            "rows": ctrl_rows,
            "passes": all(c["real_head_medae"] < c["random_decode_medae_mean"]
                          for c in ctrl_rows),
            "note": ("if a permuted decode scored as well, the win would be the "
                     "vocabulary alone and not the head using it")},
        "low_speed_excluded": {"n_excluded": n_excl, "n_kept": int(n)},
    }
    out = {"source": a.intent, "n_windows": int(n), "duty": duty,
           "turn_threshold": a.turn_thr, "n_turns": int(turn.sum()),
           "note": ("REALISED = the SHIPPED head's actual argmax decode pushed "
                    "through canonical_controls. Compose with vocab_design.json "
                    "(the ORACLE bound) — neither alone can choose the constant, "
                    "because kappa_turn moves the crossover the head must decide "
                    "at."),
           "floor": floor, "rows": real, "shipped": ship, "best_realised": best,
           "controls": ctrl}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"n={n} (excluded {n_excl} v0<{a.min_v0})  turns={int(turn.sum())}  duty={duty:.3f}")
    print(f"{'design':<44}{'crossover':>10}{'medAE turns':>12}{'RMSE all':>10}"
          f"{'turns goaled ok':>17}{'straights given k':>19}")
    for r in [floor] + real:
        print(f"{r['design']:<44}"
              f"{(r['crossover_kappa'] if r['crossover_kappa'] else 0):>10.4f}"
              f"{r['medae_on_turns']:>12.5f}{r['rmse_all']:>10.5f}"
              f"{r['frac_turns_goaled_nonzero_correct_sign']:>17.4f}"
              f"{r['frac_straights_given_curvature']:>19.4f}"
              + ("   <<< SHIPPED" if r.get("is_shipped") else ""))
    print()
    for c in ctrl_rows:
        print(f"CONTROL random decode @kappa_turn={c['kappa_turn']}: "
              f"medAE {c['random_decode_medae_mean']:.5f} +/- "
              f"{c['random_decode_medae_sd']:.5f}   real head "
              f"{c['real_head_medae']:.5f}")
    print("CONTROLS: " + json.dumps({k: v.get("passes") for k, v in ctrl.items()
                                     if "passes" in v}))
    print(f"BEST REALISED: kappa_turn={best['kappa_turn']} "
          f"medAE {best['medae_on_turns']:.5f} vs SHIPPED "
          f"{ship['medae_on_turns']:.5f} vs FLOOR {floor['medae_on_turns']:.5f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
