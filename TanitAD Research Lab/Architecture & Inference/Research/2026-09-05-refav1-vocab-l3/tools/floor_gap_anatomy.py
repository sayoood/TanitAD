#!/usr/bin/env python
"""WHERE THE `3.6-5.9x BELOW THE TRIVIAL FLOORS ON TURNING WINDOWS` COMES FROM.

`PREREG_D-VOCAB-L3_FLOOR_GAP` §1 rests on one MEASURED fact, taken from the
28-window `abt` A/B: *"both A/B arms sit 3.6-5.9x BELOW the trivial `ha` /
`ha0_ext` floors on TURNING windows"*. Everything the prereg predicts is a
prediction about closing THAT gap.

⛔ THE `abt` PANEL IS NOT A SAMPLE OF THE ROAD. Its 14 episodes were selected
because the `prior tau=0.5` decode CHANGED there — selected on the LEVER, not on
the road (`raw/ab_power_analysis.json`: 18 of 282 windows change, and the
targeted view keeps the 14 episodes carrying them). Whether the floor gap
survives on a representative panel is therefore an open question, and it is the
question this instrument answers.

It crosses TWO axes, because either alone can produce a wrong reading:

  * PANEL — the targeted `abt` grid (28 windows) against the representative
    `dump_ccos_comp` grid (282 windows / 141 episodes, same checkpoint, same
    `ccos`, stride 40);
  * THRESHOLD — the inherited 1e-3 (R 1000 m), 1e-2, and the MEASURED
    vocabulary crossover 4e-2 (R 25 m). ⛔ The prereg itself forbids 1e-3: it is
    the label-set error behind the retracted "20.5 % turn recall".

⛔ `gt_kappa = yaw_rate.mean / speed.mean.clamp_min(0.5)` is NOT a curvature
below ~1 m/s — it divides by the FLOOR. Windows with v0 < 1 m/s are excluded and
counted (the goal-margin stream's §0 defect, which swamped an RMS by 7x). Both
filtered and unfiltered figures are printed, because the retraction it caused
there means the difference must be visible rather than trusted.

⛔ `ha0_ext` is the INTEGRATOR form (`refav1_arm.hold_ext_controls`) per M11 —
it is what the dumps bank; the closed form differs by 1.862923 m at 15 s, more
than the whole quantity being measured.

CONTROLS, each of which must read a known value or the table is void:
 1. `n` printed for every cell, both sides of every split;
 2. the STRAIGHT complement printed beside every turning cell — a claim about
    turning windows that is equally true of straights is a claim about the
    panel, not about turning;
 3. `ha0` (constant velocity) carried beside `ha0_ext`, so a floor that moves
    with the split is visible;
 4. ⛔⛔ EACH PANEL'S OWN `manifest["cost"]` IS READ AND PRINTED. The two panels
    do NOT share a cost triple — `dump_ccos_comp` ran at
    `W_KAPPA = 32.149` and the `abt` grid at `W_KAPPA = 0.0` — so a
    panel-vs-panel difference is confounded between the WINDOW SELECTION and the
    CURVATURE PENALTY, and the table must say so rather than let a reader
    attribute it. (`D-REFAV1-P4-COS-THRASH`, MEASURED the same day: at
    `W_KAPPA = 0` the planner executes curvature at the `kappa_max = 0.2` clip
    bound on 90 % of STRAIGHT windows.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def load(dump: str) -> dict:
    G, V0, arms = [], [], {}
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        with np.load(f) as z:
            G.append(z["g"]); V0.append(z["v0"])
            for k in z.files:
                if k in ("g", "ws", "eid", "clip_index", "v0"):
                    continue
                if z[k].ndim == 3:
                    arms.setdefault(k, []).append(z[k])
    if not G:
        raise SystemExit(f"no ep*.npz under {dump}")
    return dict(g=np.concatenate(G), v0=np.concatenate(V0),
                arms={k: np.concatenate(v) for k, v in arms.items()})


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=1)


def gt_kappa(g: np.ndarray, stack: str, taniteval: str, dt: float) -> np.ndarray:
    sys.path.insert(0, stack); sys.path.insert(0, taniteval)
    import torch
    from taniteval import four_families as ff
    G = ff._seq_geometry(torch.as_tensor(g).float(), dt)
    return (G["yaw_rate"].mean(1) / G["speed"].mean(1).clamp_min(0.5)).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    metavar="NAME=DUMP", help="repeatable")
    ap.add_argument("--stack", default="C:/Users/Admin/tanitad-wt/stack")
    ap.add_argument("--taniteval", default="C:/Users/Admin/tanitad-wt/taniteval")
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--thresholds", default="1e-3,1e-2,4e-2")
    ap.add_argument("--v0-floor", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    thrs = [float(x) for x in a.thresholds.split(",")]

    out: dict = {
        "why": "does the prereg's motivating floor gap survive a "
               "representative panel and the MEASURED crossover?",
        "thresholds": thrs, "v0_floor_mps": a.v0_floor,
        "ha0_ext_form": "INTEGRATOR (refav1_arm.hold_ext_controls), M11",
        "kappa_recipe": "yaw_rate.mean(1)/speed.mean(1).clamp_min(0.5) on the "
                        "dump's own GT waypoints — the same recipe as "
                        "tools/gt_kappa.py, so the split joins the banked "
                        "decode statistics",
        "panels": {}}

    for spec in a.panel:
        name, _, dump = spec.partition("=")
        D = load(dump)
        k = gt_kappa(D["g"], a.stack, a.taniteval, a.dt)
        A = {n: ade(v, D["g"]) for n, v in D["arms"].items()
             if n in ("cl", "ha", "ha0", "ha0_ext", "ol")}
        if "cl" not in A or "ha0_ext" not in A:
            raise SystemExit(f"{name}: dump lacks cl or ha0_ext")
        man = os.path.join(dump, "manifest.json")
        cost = None
        if os.path.exists(man):
            with open(man, encoding="utf-8") as fh:
                cost = json.load(fh).get("cost")
        row: dict = {"dump": dump,
                     "cost": cost or "ABSENT (arm predates cost stamping)",
                     "n_windows": int(len(k)),
                     "n_excluded_low_v0": int((D["v0"] < a.v0_floor).sum()),
                     "kappa_abs_quantiles": {
                         str(q): float(np.quantile(np.abs(k), q))
                         for q in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)}}
        for lbl, base in (("v0_filtered", D["v0"] >= a.v0_floor),
                          ("unfiltered", np.ones(len(k), bool))):
            cells = {}
            for thr in thrs:
                t = (np.abs(k) > thr) & base
                s = (np.abs(k) <= thr) & base
                c: dict = {"n_turning": int(t.sum()),
                           "n_straight": int(s.sum())}
                for tag, m in (("turning", t), ("straight", s)):
                    if not m.sum():
                        c[tag] = None
                        continue
                    c[tag] = {n: float(v[m].mean()) for n, v in A.items()}
                    c[tag]["deficit_cl_minus_ha0ext"] = float(
                        (A["cl"][m] - A["ha0_ext"][m]).mean())
                    c[tag]["ratio_cl_over_ha0ext"] = float(
                        A["cl"][m].mean() / A["ha0_ext"][m].mean())
                    c[tag]["ratio_cl_over_ha"] = float(
                        A["cl"][m].mean() / A["ha"][m].mean())
                cells[f"thr_{thr:g}"] = c
            row[lbl] = cells
        out["panels"][name] = row

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    # a compact table, because the JSON is where the evidence is but the table
    # is what a reader will act on
    for name, row in out["panels"].items():
        c = row.get("cost")
        w = (c or {}).get("weights") if isinstance(c, dict) else None
        print(f"[cost] {name}: metric="
              f"{(c or {}).get('metric') if isinstance(c, dict) else c} "
              f"weights={w}")
    print(f"{'panel':>16} {'thr':>6} {'n_t':>4} "
          f"{'cl_t':>7} {'ha0ext_t':>9} {'ratio_t':>8} | "
          f"{'n_s':>4} {'cl_s':>7} {'ha0ext_s':>9} {'ratio_s':>8}")
    for name, row in out["panels"].items():
        for key, c in row["v0_filtered"].items():
            t, s = c["turning"], c["straight"]
            print(f"{name:>16} {key[4:]:>6} {c['n_turning']:>4} "
                  f"{(t['cl'] if t else float('nan')):>7.4f} "
                  f"{(t['ha0_ext'] if t else float('nan')):>9.4f} "
                  f"{(t['ratio_cl_over_ha0ext'] if t else float('nan')):>8.3f} | "
                  f"{c['n_straight']:>4} "
                  f"{(s['cl'] if s else float('nan')):>7.4f} "
                  f"{(s['ha0_ext'] if s else float('nan')):>9.4f} "
                  f"{(s['ratio_cl_over_ha0ext'] if s else float('nan')):>8.3f}")
    print(f"[wrote] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
