#!/usr/bin/env python
"""⭐ THE DE-CONFOUNDED ORACLE, READ OFF ITS OWN DUMP.

The banked `cl_oraclegoal` arm read **ADE 7.4708** with a perfect goal and was
called a bound. It is not one: supplying `goal_field` leaves `goal_action = None`
so the canonical SEED never enters the iCEM pool (`goal_source` `supplied`
142/142 against `tactical_imagined` 142/142 for `cl`). It differs from the
shipped arm in TWO things, so it bounds nothing.

`cl_oracleseed` keeps the head's own seed and replaces ONLY the goal field. This
reads all three arms off ONE dump, so they share the window grid, the plan seed,
the cost metric and the weight triple by construction.

WHAT IT REPORTS
  * ADE per arm, on ALL / TURNING / STRAIGHT windows, split at the MEASURED
    crossover |gt_k| > 4e-2 (never the inherited 1e-3 = R 1000 m);
  * the trivial floors `ha` / `ha0` / `ha0_ext` (the INTEGRATOR form, M11) on the
    same splits — the bar the headline question is about;
  * paired deltas, window by window (the arms are on identical windows, so the
    pairing is exact and a per-window difference of 0 is a genuine tie);
  * ⭐ ERRATUM §E3's SATURATION RATE — the fraction of windows whose planned
    controls touch `kappa_max` / `a_max` — per arm, because a deficit that
    shrinks while saturation stays pinned was not fixed by what you think.

CONTROLS, each of which must read a KNOWN value or the table is void
  1. `goal_source` per arm: `cl` must be `tactical_imagined` on EVERY window and
     `cl_oracleseed` must be `supplied+seed` on EVERY window. If not, the arm
     that ran is not the arm being reported.
  2. the seed identity: `cl` and `cl_oracleseed` must decode the SAME (lat, lon)
     token on every window — that is what makes the goal the only difference.
  3. the floors are single-valued in the dump (one `ha0_ext` per window for all
     arms), so a floor cannot move between arms by construction; the check that
     remains is against the BANKED floors of another run, done separately.
  4. `n` printed for every cell; windows with v0 < 1 m/s excluded and counted,
     because `yaw_rate/speed` is not a curvature below the speed floor.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

GOAL_SOURCE_NAMES = ["none", "supplied", "tactical_imagined", "supplied+seed"]
LAT = ["LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC",
       "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R"]


def load(dump: str):
    traj, dec = {}, {}
    G, V0 = [], []
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        with np.load(f) as z:
            G.append(z["g"]); V0.append(z["v0"])
            for k in z.files:
                if k in ("g", "ws", "eid", "clip_index", "v0"):
                    continue
                if z[k].ndim == 3:
                    traj.setdefault(k, []).append(z[k])
    for f in sorted(glob.glob(os.path.join(dump, "decisions", "ep*.npz"))):
        with np.load(f, allow_pickle=True) as z:
            for k in z.files:
                dec.setdefault(k, []).append(z[k])
    if not G:
        raise SystemExit(f"no ep*.npz under {dump} yet")
    return (np.concatenate(G), np.concatenate(V0),
            {k: np.concatenate(v) for k, v in traj.items()},
            {k: np.concatenate(v) for k, v in dec.items()})


def ade(p, g):
    return np.linalg.norm(p - g, axis=-1).mean(axis=1)


def gt_kappa(g, stack, taniteval, dt):
    sys.path.insert(0, stack); sys.path.insert(0, taniteval)
    import torch
    from taniteval import four_families as ff
    G = ff._seq_geometry(torch.as_tensor(g).float(), dt)
    return (G["yaw_rate"].mean(1) / G["speed"].mean(1).clamp_min(0.5)).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--stack", default="C:/Users/Admin/tanitad-wt/stack")
    ap.add_argument("--taniteval", default="C:/Users/Admin/tanitad-wt/taniteval")
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--kappa-thr", type=float, default=4e-2)
    ap.add_argument("--v0-floor", type=float, default=1.0)
    ap.add_argument("--kappa-max", type=float, default=0.2)
    ap.add_argument("--a-max", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    g, v0, traj, dec = load(a.dump)
    k = gt_kappa(g, a.stack, a.taniteval, a.dt)
    ok = v0 >= a.v0_floor
    turn = (np.abs(k) > a.kappa_thr) & ok
    strt = (np.abs(k) <= a.kappa_thr) & ok
    man = os.path.join(a.dump, "manifest.json")
    cost = vocab = None
    if os.path.exists(man):
        with open(man, encoding="utf-8") as fh:
            m = json.load(fh)
        cost, vocab = m.get("cost"), m.get("goal_vocab")

    out: dict = {
        "dump": a.dump, "cost": cost, "goal_vocab": vocab,
        "n_windows": int(len(k)), "n_excluded_low_v0": int((~ok).sum()),
        "kappa_threshold": a.kappa_thr,
        "n_turning": int(turn.sum()), "n_straight": int(strt.sum()),
        "ha0_ext_form": "INTEGRATOR (refav1_arm.hold_ext_controls), M11",
        "tier_note": "cl_oraclegoal and cl_oracleseed read the TRUE FUTURE => "
                     "T0, diagnostics/bounds, never driving numbers "
                     "(EVAL_DOCTRINE). cl / ha / ha0 / ha0_ext are T1."}

    # ---- CONTROL 1: goal provenance per arm ------------------------------- #
    want = {"cl": "tactical_imagined", "cl_oraclegoal": "supplied",
            "cl_oracleseed": "supplied+seed"}
    prov, prov_ok = {}, True
    for arm, exp in want.items():
        key = f"goal_source_{arm}"
        if key not in dec:
            continue
        codes = dec[key]
        names = [GOAL_SOURCE_NAMES[c] if c < len(GOAL_SOURCE_NAMES) else str(c)
                 for c in codes.tolist()]
        frac = float(np.mean([n == exp for n in names]))
        prov[arm] = {"expected": exp, "frac_matching": frac,
                     "n": int(len(names))}
        prov_ok &= (frac == 1.0)
    out["CONTROL_goal_provenance"] = {
        "per_arm": prov, "passes": bool(prov_ok),
        "why": "if cl is not tactical_imagined on every window it is not the "
               "shipped pipeline, and if cl_oracleseed is not supplied+seed the "
               "seed did NOT enter the pool and this is the CONFOUNDED arm"}

    # ---- CONTROL 2: the seed token is identical between cl and oracleseed -- #
    if "goal_lat_cl" in dec and "goal_lat_cl_oracleseed" in dec:
        same_lat = bool((dec["goal_lat_cl"] ==
                         dec["goal_lat_cl_oracleseed"]).all())
        same_lon = bool((dec["goal_lon_cl"] ==
                         dec["goal_lon_cl_oracleseed"]).all())
        import collections
        hist = collections.Counter(dec["goal_lat_cl"].tolist())
        out["CONTROL_seed_token_identical"] = {
            "lat_identical": same_lat, "lon_identical": same_lon,
            "passes": bool(same_lat and same_lon),
            "cl_lat_decode": {LAT[i] if i < len(LAT) else str(i): int(c)
                              for i, c in sorted(hist.items())},
            "why": "the goal must be the ONLY difference; a different decode "
                   "would mean the head saw different inputs"}

    # ---- ADE per arm on the three splits ---------------------------------- #
    arms = [n for n in ("cl", "cl_oraclegoal", "cl_oracleseed",
                        "ha", "ha0", "ha0_ext", "ol") if n in traj]
    A = {n: ade(traj[n], g) for n in arms}
    out["ade"] = {
        lbl: {n: {"n": int(m.sum()),
                  "mean": (float(A[n][m].mean()) if m.sum() else None)}
              for n in arms}
        for lbl, m in (("all_valid", ok), ("turning", turn), ("straight", strt))}

    # ---- the headline: does a perfect goal beat the floor on turns? -------- #
    if turn.sum() and "cl_oracleseed" in A:
        out["HEADLINE_turning"] = {
            "n": int(turn.sum()),
            "cl_oracleseed": float(A["cl_oracleseed"][turn].mean()),
            "ha0_ext": float(A["ha0_ext"][turn].mean()),
            "ha": float(A["ha"][turn].mean()),
            "ratio_oracleseed_over_ha0ext":
                float(A["cl_oracleseed"][turn].mean() / A["ha0_ext"][turn].mean()),
            "beats_ha0_ext": bool(A["cl_oracleseed"][turn].mean()
                                  < A["ha0_ext"][turn].mean()),
            "cl_ratio_over_ha0ext":
                float(A["cl"][turn].mean() / A["ha0_ext"][turn].mean())}

    # ---- paired deltas, exact (identical windows) ------------------------- #
    pairs = [("cl_oracleseed", "cl"), ("cl_oraclegoal", "cl"),
             ("cl_oracleseed", "cl_oraclegoal")]
    out["paired"] = {}
    for x, y in pairs:
        if x not in A or y not in A:
            continue
        d = A[x] - A[y]
        out["paired"][f"{x}_minus_{y}"] = {
            lbl: {"n": int(m.sum()),
                  "mean": (float(d[m].mean()) if m.sum() else None),
                  "n_exact_ties": int((d[m] == 0).sum()) if m.sum() else None}
            for lbl, m in (("all_valid", ok), ("turning", turn),
                           ("straight", strt))}

    # ---- ⭐ ERRATUM §E3: the SATURATION RATE ------------------------------- #
    sat = {}
    for arm in ("cl", "cl_oraclegoal", "cl_oracleseed"):
        key = f"{arm}_controls"
        if key not in dec:
            continue
        c = dec[key]                                   # [N, H, 2] = (a, kappa)
        amax = np.abs(c[..., 0]).max(axis=1)
        kmax = np.abs(c[..., 1]).max(axis=1)
        row = {}
        for lbl, m in (("all_valid", ok), ("turning", turn),
                       ("straight", strt)):
            if not m.sum():
                row[lbl] = None
                continue
            row[lbl] = {
                "n": int(m.sum()),
                "frac_at_kappa_max": float((kmax[m] >= a.kappa_max - 1e-6).mean()),
                "frac_at_a_max": float((amax[m] >= a.a_max - 1e-6).mean()),
                "mean_max_abs_kappa": float(kmax[m].mean()),
                "mean_max_abs_accel": float(amax[m].mean()),
                "frac_curvature_nonzero": float((kmax[m] > 1e-3).mean())}
        sat[arm] = row
    out["SATURATION_E3"] = {
        "per_arm": sat, "kappa_max": a.kappa_max, "a_max": a.a_max,
        "reading": "deficit shrinks AND saturation falls => the named mechanism; "
                   "deficit shrinks AND saturation stays pinned => something "
                   "else did the work and the claim is NOT established"}

    # ---- ⭐ THE PER-WINDOW TABLE, because an aggregate invites a story ---- #
    # MEASURED 2026-09-05: an explanation of the exact ties ("they are the
    # LANE_KEEP decodes, whose seed is all-zero") was written from a 5-window
    # aggregate and REFUTED by this table -- all 3 turning windows decode TURN_*
    # and only 1 of 5 tied windows is LANE_KEEP. A mechanism claim needs the
    # JOINT, not the marginals.
    rows = []
    tie = None
    if ("cl_oracleseed_controls" in dec and "cl_oraclegoal_controls" in dec):
        cs, cg = dec["cl_oracleseed_controls"], dec["cl_oraclegoal_controls"]
        tie = np.array([float(np.abs(cs[i] - cg[i]).max()) == 0.0
                        for i in range(len(cs))])
    for i in range(len(k)):
        r = {"i": i, "v0": float(v0[i]), "gt_kappa": float(k[i]),
             "valid": bool(ok[i]), "turning": bool(turn[i]),
             "lat_decode": (LAT[int(dec["goal_lat_cl"][i])]
                            if "goal_lat_cl" in dec else None)}
        if tie is not None:
            r["oracleseed_tied_with_oraclegoal"] = bool(tie[i])
        for n_ in arms:
            r[f"ade_{n_}"] = float(A[n_][i])
        rows.append(r)
    out["per_window"] = rows
    if tie is not None and "goal_lat_cl" in dec:
        lat_i = dec["goal_lat_cl"]
        out["JOINT_tie_vs_decode"] = {
            "n_tied": int(tie.sum()),
            "n_tied_that_are_LANE_KEEP": int((tie & (lat_i == 0)).sum()),
            "n_turning": int(turn.sum()),
            "n_turning_that_are_LANE_KEEP": int((turn & (lat_i == 0)).sum()),
            "n_turning_that_are_TURN": int((turn & (lat_i >= 6)).sum()),
            "why": "the joint that refuted the 'ties are the zero-seed "
                   "LANE_KEEP windows' story; a mechanism claim needs it"}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for key in ("CONTROL_goal_provenance", "CONTROL_seed_token_identical",
                "ade", "HEADLINE_turning", "paired", "SATURATION_E3",
                "JOINT_tie_vs_decode"):
        if key in out:
            print(f"--- {key} ---")
            print(json.dumps(out[key], indent=1))
    print(f"[wrote] {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
