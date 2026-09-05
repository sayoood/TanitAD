#!/usr/bin/env python
"""The `--lat-logit-bias` LADDER: what each setting does to the decode, exactly.

⭐ WHY. `lat_logit_bias` is now the lever, and a GPU A/B costs ~37 s per window
per arm. Choosing its values by fishing would waste that. The DECODE, however,
is a deterministic function of already-extracted logits, so the decode side of
every candidate setting can be computed EXACTLY at zero GPU — and only the
plan-level consequence then needs the GPU.

Two rule families are laid out:

  * ``commit b``  — a single NEGATIVE bias on LANE_KEEP. One scalar, monotone,
    and it CANNOT boost a class the head never emits.
  * ``prior tau`` — the textbook logit adjustment ``-tau * log(pi)``.
    ⚠️ REPORTED WITH A WARNING: the sweep's priors are eps-smoothed at 1e-6, so
    the three NEVER-LABELLED classes (LANE_CHANGE_L/R, ABORT_LC) receive a bias
    of ``-tau*log(1e-6/Z)`` ~ +10 at tau=0.75 — i.e. the rule most strongly
    promotes exactly the tokens with no evidence behind them. That is very
    likely why `prior:label tau>=1.0` COLLAPSED in the banked sweep
    (balanced_acc 0.133). The ladder therefore also reports a MASKED variant
    that leaves never-emitted classes alone.

⛔ CONTROLS: b=0 / tau=0 must reproduce the banked argmax decode EXACTLY
(histogram and turn recall), and the never-emitted-class count is printed for
every row so a rule that wins by inventing LANE_CHANGE cannot look good.

Run: python bias_ladder.py --logits <npz> --gt-kappa <npz> --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits", required=True)
    ap.add_argument("--gt-kappa", default=None,
                    help="npz with clip_index/ws/gt_kappa (optional; when "
                         "absent only decode statistics are reported)")
    ap.add_argument("--kappa-turn", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    Z = np.load(a.logits, allow_pickle=False)
    lg = Z["lat_logits"].astype(np.float64)
    names = [str(x) for x in Z["lat_names"]]
    n, d = lg.shape
    LK = names.index("LANE_KEEP")
    TURN = [names.index("TURN_L"), names.index("TURN_R")]
    CURV = [i for i, t in enumerate(names)
            if t.startswith(("TURN_", "NUDGE_", "LANE_CHANGE_"))]
    lab = Z["lat_label"].astype(int)

    gt = None
    if a.gt_kappa:
        G = np.load(a.gt_kappa, allow_pickle=False)
        key_l = {(int(c), int(w)): i for i, (c, w) in
                 enumerate(zip(Z["clip_index"], Z["ws"]))}
        gt = np.full(n, np.nan)
        for c, w, k in zip(G["clip_index"], G["ws"], G["gt_kappa"]):
            i = key_l.get((int(c), int(w)))
            if i is not None:
                gt[i] = k
    have_gt = gt is not None and np.isfinite(gt).any()

    # priors, eps-smoothed exactly as the banked sweep does
    eps = 1e-6
    m_lab = lab >= 0
    pi = np.array([(lab[m_lab] == c).sum() for c in range(d)], dtype=np.float64)
    pi = np.maximum(pi, eps)
    pi /= pi.sum()
    never = [i for i in range(d) if (lab[m_lab] == i).sum() == 0]

    def row(name, bias):
        dec = (lg + bias[None, :]).argmax(-1)
        hist = {names[i]: int((dec == i).sum()) for i in range(d)
                if (dec == i).sum()}
        out = {"setting": name,
               "bias": [round(float(x), 6) for x in bias],
               "hist": hist,
               "n_never_labelled_emitted": int(sum(
                   (dec == i).sum() for i in never)),
               "frac_curvature_carrying": float(np.isin(dec, CURV).mean()),
               "frac_TURN": float(np.isin(dec, TURN).mean())}
        if have_gt:
            m = np.isfinite(gt)
            turning = m & (np.abs(gt) > a.kappa_turn)
            straight = m & (np.abs(gt) <= a.kappa_turn)
            prop = np.isin(dec, CURV)
            out["gt"] = {
                "n_gt_turn": int(turning.sum()),
                "n_gt_straight": int(straight.sum()),
                "turn_recall": float(prop[turning].mean()) if turning.sum() else None,
                "false_turn_on_straight": float(prop[straight].mean())
                if straight.sum() else None}
            both = turning & prop
            if both.sum():
                sgn_ok = np.array([
                    (gt[i] > 0 and names[dec[i]].endswith("_L")) or
                    (gt[i] < 0 and names[dec[i]].endswith("_R"))
                    for i in np.where(both)[0]])
                out["gt"]["direction_acc"] = float(sgn_ok.mean())
                out["gt"]["n_both_turn"] = int(both.sum())
        return out

    rows = [row("argmax (CONTROL, b=0)", np.zeros(d))]
    for b in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        v = np.zeros(d)
        v[LK] = -b
        rows.append(row(f"commit b={b}", v))
    for tau in (0.25, 0.5, 0.75, 1.0):
        rows.append(row(f"prior tau={tau} (EPS-SMOOTHED)", -tau * np.log(pi)))
        v = -tau * np.log(pi)
        v[never] = 0.0                       # the masked variant
        rows.append(row(f"prior tau={tau} MASKED", v))

    R = {"logits": a.logits, "n": int(n), "d": int(d), "lat_names": names,
         "prior_eps_smoothed": [round(float(x), 8) for x in pi],
         "never_labelled_classes": [names[i] for i in never],
         "gt_kappa_joined": bool(have_gt), "rows": rows}
    with open(a.out, "w") as f:
        json.dump(R, f, indent=1)

    print(f"[in] n={n} windows, d={d}; never-labelled classes: "
          f"{[names[i] for i in never]}")
    hdr = (f"{'setting':<32}{'curv%':>8}{'TURN%':>8}{'recall':>8}"
           f"{'falseT':>8}{'dir':>7}{'nevr':>6}  hist")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        g = r.get("gt", {})
        def f(x, w=8):
            return f"{x:>{w}.3f}" if isinstance(x, float) else f"{'-':>{w}}"
        print(f"{r['setting']:<32}{r['frac_curvature_carrying']:>8.3f}"
              f"{r['frac_TURN']:>8.3f}{f(g.get('turn_recall'))}"
              f"{f(g.get('false_turn_on_straight'))}"
              f"{f(g.get('direction_acc'), 7)}"
              f"{r['n_never_labelled_emitted']:>6}  "
              f"{ {k: v for k, v in r['hist'].items()} }")
    print(f"\n[wrote] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
