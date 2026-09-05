#!/usr/bin/env python
"""IS `LANE_KEEP` THE WRONG ANSWER, OR THE BEST ANSWER THE VOCABULARY HAS?

⭐ THE QUESTION. The predecessor MEASURED that the goal head decodes `LANE_KEEP`
on 86.5 % of windows and that a `LANE_KEEP` decode makes a turn unreachable. It
read that as a HEAD defect. This asks the prior question, which nobody asked:
**given the tokens that exist, what SHOULD the head emit on a GT-turn window?**

The lateral goal vocabulary's sustained curvature is not free. From
`refa_v1.py:canonical_controls` (:347) and its constants (:118-123):

    TURN_L/R      k[0 : 4.0/dt] = +/- 0.08   (GOAL_KAPPA_TURN, R = 12.5 m)
                  SUSTAINED for GOAL_TURN_S = 4 s, then 0
    NUDGE_L/R     S-CURVE  +kap then -kap,  kap = 2*0.5/(v_ref^2 * 1.0^2)
    LANE_CHANGE   S-CURVE  +kap then -kap,  kap = 2*1.75/(v_ref^2 * 2.0^2)
    LANE_KEEP     k == 0

⇒ An S-curve integrates to ZERO net heading change, so it cannot track a
sustained curve however large its peak. **The only SUSTAINED curvatures the
vocabulary can command are 0 and +/-0.08.** A road at R = 200 m (k = 0.005) has
no token.

⇒ THE TEST. For every window, score each lateral token by the L2 error between
its canonical curvature profile and the CONSTANT GT curvature over the same
horizon, and report which token is the argmin. If `LANE_KEEP` is the argmin on
most GT-turn windows, then the head emitting `LANE_KEEP` is **not a decision
error** -- it is the vocabulary having no token for the road, and no re-fit of
the head can fix it.

⛔ CONTROLS (each must read a value fixed before the run):
  C1 on a window with |k_gt| ~ 0.08 the argmin MUST be TURN, not LANE_KEEP --
     otherwise the scoring is broken rather than the vocabulary. Reported as
     the argmin histogram restricted to |k_gt| > 0.06.
  C2 the crossover threshold must equal GOAL_KAPPA_TURN / 2 = 0.04 analytically
     (for a constant target the L2 argmin between 0 and K is the midpoint,
     scaled by the duty cycle); the measured crossover is printed beside it.
  C3 n printed for every band; a band with n < 10 is marked UNDERPOWERED and
     no verdict is read off it.

Run: python gm_vocab_fit.py --logits <npz> --gt-kappa <npz> --out <json>
     python gm_vocab_fit.py --intent <intent npz> --out <json>     (dense panel)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

# ---- constants read from stack/tanitad/refs/refa_v1.py:118-123 ------------ #
GOAL_KAPPA_MAX = 0.2
GOAL_KAPPA_TURN = 0.08
GOAL_TURN_S = 4.0
GOAL_LANE_CHANGE = (2.0, 1.75)
GOAL_NUDGE = (1.0, 0.5)
GOAL_V_REF_MIN_MPS = 3.0


def canonical_kappa(lat: str, v0: float, n: int, dt: float) -> np.ndarray:
    """The curvature column of `canonical_controls`, re-stated. ⛔ Re-stated, so
    it carries control C1: on a genuinely sharp window it must select TURN."""
    k = np.zeros(n)
    sign = 1.0 if lat.endswith("_L") else -1.0
    v_ref = max(float(v0), GOAL_V_REF_MIN_MPS)

    def s_curve(half_s, half_m):
        m = int(round(half_s / dt))
        kap = min(GOAL_KAPPA_MAX, 2.0 * half_m / (v_ref ** 2 * half_s ** 2))
        k[:m] = sign * kap
        k[m:2 * m] = -sign * kap

    if lat.startswith("TURN_"):
        k[:int(round(GOAL_TURN_S / dt))] = sign * GOAL_KAPPA_TURN
    elif lat.startswith("LANE_CHANGE_"):
        s_curve(*GOAL_LANE_CHANGE)
    elif lat.startswith("NUDGE_"):
        s_curve(*GOAL_NUDGE)
    return k


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", default=None)
    ap.add_argument("--gt-kappa", default=None)
    ap.add_argument("--intent", default=None,
                    help="dense panel npz carrying gt_kappa + v0 directly")
    ap.add_argument("--op-steps", type=int, default=30)
    ap.add_argument("--op-dt", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.intent:
        Z = np.load(a.intent, allow_pickle=False)
        gt = Z["gt_kappa"].astype(np.float64)
        v0 = Z["v0"].astype(np.float64)
        lat_names = [str(x) for x in Z["lat_names"]]
        lg = Z["lat_logits"].astype(np.float64)
        src = a.intent
    else:
        Z = np.load(a.logits, allow_pickle=False)
        G = np.load(a.gt_kappa, allow_pickle=False)
        lat_names = [str(x) for x in Z["lat_names"]]
        key = {(int(c), int(w)): i for i, (c, w) in
               enumerate(zip(Z["clip_index"], Z["ws"]))}
        gtf = np.full(len(Z["ws"]), np.nan)
        for c, w, kk in zip(G["clip_index"], G["ws"], G["gt_kappa"]):
            i = key.get((int(c), int(w)))
            if i is not None:
                gtf[i] = kk
        m = np.isfinite(gtf)
        gt = gtf[m]
        v0 = Z["v0"].astype(np.float64)[m]
        lg = Z["lat_logits"].astype(np.float64)[m]
        src = a.logits

    n = len(gt)
    nstep, dt = int(a.op_steps), float(a.op_dt)
    LK = lat_names.index("LANE_KEEP")

    # argmin token per window against a CONSTANT GT curvature profile
    best = np.zeros(n, dtype=int)
    err_lk = np.zeros(n)
    err_best = np.zeros(n)
    for i in range(n):
        tgt = np.full(nstep, gt[i])
        e = np.array([np.sum((canonical_kappa(t, v0[i], nstep, dt) - tgt) ** 2)
                      for t in lat_names])
        best[i] = int(np.argmin(e))
        err_lk[i] = e[LK]
        err_best[i] = e[best[i]]

    def hist(mask):
        return {lat_names[c]: int((best[mask] == c).sum())
                for c in range(len(lat_names)) if (best[mask] == c).sum()}

    bands = []
    edges = [(0.0, 1e-3), (1e-3, 5e-3), (5e-3, 1e-2), (1e-2, 2e-2),
             (2e-2, 4e-2), (4e-2, 6e-2), (6e-2, 1e9)]
    for lo, hi in edges:
        mk = (np.abs(gt) > lo) & (np.abs(gt) <= hi)
        if not mk.sum():
            continue
        bands.append({
            "kappa_band": [lo, (None if hi > 1e8 else hi)],
            "radius_m": [(None if lo == 0 else round(1 / lo, 1)),
                         (None if hi > 1e8 else round(1 / hi, 1))],
            "n": int(mk.sum()),
            "UNDERPOWERED_n_lt_10": bool(mk.sum() < 10),
            "vocab_argmin_hist": hist(mk),
            "frac_vocab_argmin_is_LANE_KEEP": float((best[mk] == LK).mean()),
            "frac_head_decodes_LANE_KEEP": float(
                (lg[mk].argmax(-1) == LK).mean()),
        })

    turn_ids = [i for i, t in enumerate(lat_names) if t.startswith("TURN_")]
    gt_turn = np.abs(gt) > 1e-3
    # the measured crossover: smallest |k| at which TURN beats LANE_KEEP
    o = np.argsort(np.abs(gt))
    cross = None
    for i in o:
        if best[i] in turn_ids:
            cross = float(abs(gt[i]))
            break

    # C1 — on genuinely sharp windows the argmin MUST be TURN
    sharp = np.abs(gt) > 0.06
    c1 = {"n_sharp": int(sharp.sum()),
          "argmin_hist_on_sharp": hist(sharp) if sharp.sum() else {},
          "frac_TURN_on_sharp": (float(np.isin(best[sharp], turn_ids).mean())
                                 if sharp.sum() else None),
          "passes": bool(sharp.sum() >= 10
                         and np.isin(best[sharp], turn_ids).mean() > 0.9)}

    out = {
        "source": src, "n_windows": int(n),
        "vocabulary": {
            "GOAL_KAPPA_TURN": GOAL_KAPPA_TURN,
            "turn_radius_m": round(1 / GOAL_KAPPA_TURN, 2),
            "GOAL_TURN_S": GOAL_TURN_S,
            "sustained_curvatures_available": [0.0, GOAL_KAPPA_TURN],
            "note": ("NUDGE_* and LANE_CHANGE_* are S-CURVES (+kap then -kap): "
                     "net heading change ZERO, so they cannot track a sustained "
                     "curve however large their peak. The only SUSTAINED "
                     "curvatures the vocabulary can command are 0 and 0.08."),
            "source": "stack/tanitad/refs/refa_v1.py:118-123, :347-383",
        },
        "bands": bands,
        "crossover": {
            "analytic_midpoint_GOAL_KAPPA_TURN_over_2": GOAL_KAPPA_TURN / 2,
            "measured_smallest_abs_kappa_where_TURN_wins": cross,
        },
        "headline": {
            "n_gt_turn_kappa_gt_1e-3": int(gt_turn.sum()),
            "of_those_frac_vocab_argmin_is_LANE_KEEP": float(
                (best[gt_turn] == LK).mean()) if gt_turn.sum() else None,
            "frac_of_ALL_windows_where_vocab_argmin_is_LANE_KEEP": float(
                (best == LK).mean()),
            "frac_of_ALL_windows_head_decodes_LANE_KEEP": float(
                (lg.argmax(-1) == LK).mean()),
            "head_agrees_with_vocab_argmin": float((lg.argmax(-1) == best).mean()),
        },
        "CONTROL_C1_sharp_windows_select_TURN": c1,
        "verdict": None,
    }
    f = out["headline"]["of_those_frac_vocab_argmin_is_LANE_KEEP"]
    out["verdict"] = (
        f"On {f:.1%} of GT-turn windows (|kappa|>1e-3) LANE_KEEP is the "
        f"VOCABULARY-OPTIMAL token: no available token is closer to the road's "
        f"curvature. The head emitting LANE_KEEP there is therefore NOT a "
        f"decision error, and NO re-fit of the head can change it -- the "
        f"binding constraint is that the vocabulary has no sustained curvature "
        f"between 0 and {GOAL_KAPPA_TURN} (R {1/GOAL_KAPPA_TURN:.1f} m)."
        if f is not None and f > 0.5 else
        "LANE_KEEP is NOT the vocabulary-optimal token on most GT-turn windows: "
        "the head IS mis-deciding and a re-fit is indicated.")

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print(f"n={n}  source={src}")
    print(f"{'kappa band':<22} {'R(m)':<16} {'n':>5} {'vocab->LK':>10} {'head->LK':>9}")
    for b in bands:
        lo, hi = b["kappa_band"]
        r = b["radius_m"]
        print(f"({lo:.0e},{hi if hi is None else f'{hi:.0e}'}]".ljust(22)
              + f"{str(r[0]) + '-' + str(r[1]):<16}"
              + f"{b['n']:>5} {b['frac_vocab_argmin_is_LANE_KEEP']:>10.4f}"
              + f" {b['frac_head_decodes_LANE_KEEP']:>9.4f}"
              + ("  UNDERPOWERED" if b["UNDERPOWERED_n_lt_10"] else ""))
    print()
    print(f"C1 sharp(|k|>0.06) n={c1['n_sharp']} frac_TURN={c1['frac_TURN_on_sharp']} "
          f"passes={c1['passes']}")
    print(f"crossover: analytic {GOAL_KAPPA_TURN/2}  measured {cross}")
    print(f"HEADLINE: of GT-turn windows, vocab argmin is LANE_KEEP on "
          f"{out['headline']['of_those_frac_vocab_argmin_is_LANE_KEEP']:.4f}")
    print(f"          head decodes LANE_KEEP on "
          f"{out['headline']['frac_of_ALL_windows_head_decodes_LANE_KEEP']:.4f} of ALL windows; "
          f"vocab argmin is LANE_KEEP on "
          f"{out['headline']['frac_of_ALL_windows_where_vocab_argmin_is_LANE_KEEP']:.4f}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
