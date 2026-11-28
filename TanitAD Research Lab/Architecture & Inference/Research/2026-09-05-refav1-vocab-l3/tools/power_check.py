#!/usr/bin/env python
"""ERRATUM-1 §E2 — THE BLOCKING POWER CHECK, before any panel GPU is spent.

Question: is `the L=1 turning-window deficit to ha0_ext shrinks by >= 50 %`
measurable at all on this rig? It is only meaningful if HALF THE DEFICIT
exceeds the planner's own INFERENCE-seed noise floor on the SAME windows
(§E1: iCEM samples, so one checkpoint evaluated twice disagrees — no
retraining involved).

⛔ The turning split uses the MEASURED crossover |gt_k| > 4e-2 (R 25 m), never
the inherited 1e-3 (R 1000 m).
⛔ `gt_kappa = yaw_rate.mean / speed.mean.clamp_min(0.5)` is NOT a curvature
below ~1 m/s — it divides by the floor. Windows with v0 < 1 m/s are excluded
and counted (the goal-margin stream's §0 defect, which swamped an RMS by 7x).
⛔ `ha0_ext` is the INTEGRATOR form (`refav1_arm.hold_ext_controls`) per M11 —
it is what the dump banked; the closed form differs by 1.86 m at 15 s.

CONTROLS, each of which must read a KNOWN value or the check is void:
 1. the floors (ha / ha0 / ha0_ext) read no plan seed, so they must be
    BIT-IDENTICAL between the seed-0 and seed-1 dumps. If a floor moves, the
    harness is not holding what it claims and no number below counts.
 2. the straight-window deficit is reported beside the turning one — a
    vocabulary change touches nothing there.
 3. `n` and the excluded count are printed for every cell.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np


def load(dump: str) -> dict:
    G, WS, CIX, V0, arms = [], [], [], [], {}
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        with np.load(f) as z:
            m = len(z["ws"])
            G.append(z["g"]); WS.append(z["ws"]); V0.append(z["v0"])
            CIX.append(np.repeat(np.asarray(z["clip_index"]).ravel()[0], m))
            for k in z.files:
                if k in ("g", "ws", "eid", "clip_index", "v0"):
                    continue
                if z[k].ndim == 3:
                    arms.setdefault(k, []).append(z[k])
    return dict(g=np.concatenate(G), ws=np.concatenate(WS),
                v0=np.concatenate(V0), clip=np.concatenate(CIX),
                arms={k: np.concatenate(v) for k, v in arms.items()})


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-window ADE (m) — mean over the K horizon steps of the L2 error."""
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=1)


def gt_kappa(g: np.ndarray, stack: str, taniteval: str, dt: float) -> np.ndarray:
    sys.path.insert(0, stack); sys.path.insert(0, taniteval)
    import torch
    from taniteval import four_families as ff
    Gg = ff._seq_geometry(torch.as_tensor(g).float(), dt)
    return (Gg["yaw_rate"].mean(1) / Gg["speed"].mean(1).clamp_min(0.5)).numpy()


def cell(x: np.ndarray) -> dict:
    return {"n": int(x.size), "mean": float(x.mean()) if x.size else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="the REPRESENTATIVE panel")
    ap.add_argument("--seed-a", required=True, help="replicate arm A (seed 0)")
    ap.add_argument("--seed-b", required=True, help="replicate arm B (seed 1)")
    ap.add_argument("--stack", default="C:/Users/Admin/tanitad-wt/stack")
    ap.add_argument("--taniteval", default="C:/Users/Admin/tanitad-wt/taniteval")
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--kappa-thr", type=float, default=4e-2)
    ap.add_argument("--v0-floor", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out: dict = {"why": __doc__.strip().splitlines()[0],
                 "kappa_threshold": a.kappa_thr,
                 "kappa_threshold_why": "the MEASURED vocabulary crossover "
                                        "(R 25 m); the inherited 1e-3 is "
                                        "R 1000 m and is the label-set error "
                                        "behind the retracted 20.5 % recall",
                 "v0_floor_mps": a.v0_floor,
                 "ha0_ext_form": "INTEGRATOR (refav1_arm.hold_ext_controls), M11"}

    # ---- 1. the representative panel -------------------------------------- #
    D = load(a.dump)
    k = gt_kappa(D["g"], a.stack, a.taniteval, a.dt)
    ok = D["v0"] >= a.v0_floor
    turn = (np.abs(k) > a.kappa_thr) & ok
    strt = (np.abs(k) <= a.kappa_thr) & ok
    out["panel"] = {
        "dump": a.dump, "n_windows": int(len(k)),
        "n_excluded_low_v0": int((~ok).sum()),
        "n_turning": int(turn.sum()), "n_straight": int(strt.sum()),
        "turn_fraction_of_valid": float(turn.sum() / max(int(ok.sum()), 1)),
    }
    A = {n: ade(v, D["g"]) for n, v in D["arms"].items() if n in
         ("cl", "ha", "ha0", "ha0_ext", "ol")}
    if "cl" not in A or "ha0_ext" not in A:
        print("REFUSE: dump lacks cl or ha0_ext"); return 2
    out["ade"] = {sel_name: {n: cell(v[sel]) for n, v in A.items()}
                  for sel_name, sel in (("all_valid", ok), ("turning", turn),
                                        ("straight", strt))}
    dfc_t = float((A["cl"][turn] - A["ha0_ext"][turn]).mean())
    dfc_s = float((A["cl"][strt] - A["ha0_ext"][strt]).mean())
    out["deficit_cl_minus_ha0ext"] = {
        "turning": dfc_t, "straight": dfc_s,
        "turning_ratio_cl_over_ha0ext":
            float(A["cl"][turn].mean() / A["ha0_ext"][turn].mean()),
        "target_50pct_shrink": 0.5 * dfc_t}

    # ---- 2. the INFERENCE-seed floor, on the same split ------------------- #
    SA, SB = load(a.seed_a), load(a.seed_b)
    same = (SA["ws"].shape == SB["ws"].shape and
            bool((SA["ws"] == SB["ws"]).all()) and
            bool((SA["clip"] == SB["clip"]).all()))
    # ⛔ CONTROL 1: the floors read no plan seed => BIT-IDENTICAL or void.
    floors = {}
    for n in ("ha", "ha0", "ha0_ext"):
        if n in SA["arms"] and n in SB["arms"]:
            d = np.abs(SA["arms"][n] - SB["arms"][n]).max()
            floors[n] = {"max_abs_diff": float(d), "bit_identical": bool(d == 0)}
    out["CONTROL_floors_bit_identical_across_seeds"] = {
        "same_grid": same, "per_arm": floors,
        "passes": bool(same and all(v["bit_identical"] for v in floors.values())),
        "why": "ha/ha0/ha0_ext read no plan seed; if one moves the harness is "
               "not holding what it claims and nothing below counts"}

    ks = gt_kappa(SA["g"], a.stack, a.taniteval, a.dt)
    oks = SA["v0"] >= a.v0_floor
    turn_s = (np.abs(ks) > a.kappa_thr) & oks
    ca, cb = ade(SA["arms"]["cl"], SA["g"]), ade(SB["arms"]["cl"], SB["g"])
    out["inference_seed_floor"] = {
        "arm_a": a.seed_a, "arm_b": a.seed_b,
        "n_windows": int(len(ca)), "n_turning": int(turn_s.sum()),
        "n_excluded_low_v0": int((~oks).sum()),
        "abs_delta_all_valid": float(abs((ca[oks] - cb[oks]).mean())),
        "abs_delta_turning": (float(abs((ca[turn_s] - cb[turn_s]).mean()))
                              if turn_s.sum() else None),
        "paired_absmean_turning": (float(np.abs(ca[turn_s] - cb[turn_s]).mean())
                                   if turn_s.sum() else None),
        "what_it_answers": "would ANOTHER INFERENCE RUN say this? (ERRATUM E1) "
                           "— not another training run (H-ESTIM-SEED-1) and not "
                           "another draw of episodes (the bootstrap)"}

    floor = out["inference_seed_floor"]["abs_delta_turning"]
    if floor is None:
        floor = out["inference_seed_floor"]["abs_delta_all_valid"]
    tgt = out["deficit_cl_minus_ha0ext"]["target_50pct_shrink"]
    out["VERDICT"] = {
        "target_50pct_shrink_m": tgt,
        "inference_seed_floor_m": floor,
        "target_over_floor": float(tgt / floor) if floor else None,
        "runnable_as_preregistered": bool(tgt > floor),
        "reading": ("the 50 % criterion clears the inference-seed floor — the "
                    "panel can express its own committed effect"
                    if tgt > floor else
                    "UNDERPOWERED BY CONSTRUCTION — half the deficit is inside "
                    "the planner's own run-to-run noise; redesign (more "
                    "windows, more inference seeds) before spending GPU")}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["panel"], indent=1))
    print(json.dumps(out["deficit_cl_minus_ha0ext"], indent=1))
    print(json.dumps(out["CONTROL_floors_bit_identical_across_seeds"], indent=1))
    print(json.dumps(out["inference_seed_floor"], indent=1))
    print(json.dumps(out["VERDICT"], indent=1))
    print(f"[wrote] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
