#!/usr/bin/env python
"""Analyse the TARGETED A/B — control first, then the conditional effect.

⛔ THE ORDER MATTERS. The targeted design rests on one premise: a window whose
decoded token does NOT change gives BIT-IDENTICAL arms (icem_plan is seeded), so
it contributes exactly 0 to a paired delta and the aggregate factorises as
``marginal = (18/282) x conditional``. **That premise is checked FIRST, and if
it fails nothing else in this file is admissible.**

Then three quantities, in the order they should be read:

  1. CONTROL   unchanged-decode windows: must be bit-identical between arms.
  2. REPLICATE argmax@seed1 - argmax@seed0 on the SAME windows: the rig's own
     run-to-run floor. `icem_plan` draws seeded coloured noise, so this is NOT
     zero, and H-ESTIM-SEED-1 says the lever must be read against it.
  3. LEVER     prior050 - argmax@seed0, per metric family.

⚠️ A lever delta that is not clearly larger than the replicate delta has NOT
been shown to move anything, however tight its own interval.

Run: python analyze_targeted.py --dir <abt dir> --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _load(dump: str):
    CL, GL, WS, CIX, G, HA, HAX, V0 = [], [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        d_f = os.path.join(dump, "decisions", os.path.basename(f))
        if not os.path.exists(d_f):
            continue
        with np.load(f) as z, np.load(d_f) as d:
            m = len(z["ws"])
            CL.append(d["cl_controls"]); GL.append(d["goal_lat_cl"])
            WS.append(z["ws"]); G.append(z["g"]); HA.append(z["ha"])
            HAX.append(z["ha0_ext"]); V0.append(z["v0"])
            CIX.append(np.repeat(np.asarray(z["clip_index"]).ravel()[0], m))
    if not CL:
        return None
    return dict(cl=np.concatenate(CL), glat=np.concatenate(GL),
                ws=np.concatenate(WS), cix=np.concatenate(CIX),
                g=np.concatenate(G), ha=np.concatenate(HA),
                ha0_ext=np.concatenate(HAX), v0=np.concatenate(V0))


def _paths(controls, v0, dt=0.2):
    """Integrate (a, kappa) to xy with the ARM'S OWN `paths_from_controls`, so
    this analysis cannot drift from how the dumps were produced.

    ⚠️ Two things a preflight caught before any GPU time was spent on them:
    the module lives in `taniteval/TOOLS`, not `taniteval/`; and it returns a
    LEADING BATCH DIM `[1, K, 2]`, so the squeeze is required or every ADE
    below broadcasts into nonsense."""
    import sys
    import torch
    p = "C:/Users/Admin/tanitad-wt/taniteval/tools"
    if p not in sys.path:
        sys.path.insert(0, p)
    from refav1_arm import paths_from_controls
    out = paths_from_controls(torch.as_tensor(controls).float(), float(v0),
                              dt, controls.shape[0]).numpy()
    return out[0] if out.ndim == 3 else out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = {}
    for nm in ("argmax", "prior050", "argmax_seed1"):
        d = _load(os.path.join(a.dir, f"dump_{nm}"))
        if d is None:
            print(f"MISSING arm {nm} — inconclusive")
            return 2
        arms[nm] = d
    A, P, S1 = arms["argmax"], arms["prior050"], arms["argmax_seed1"]
    n = len(A["ws"])

    # the windows must be the same, in the same order
    same_grid = (np.array_equal(A["ws"], P["ws"]) and
                 np.array_equal(A["cix"], P["cix"]) and
                 np.array_equal(A["ws"], S1["ws"]))
    changed = A["glat"] != P["glat"]

    R = {"dir": a.dir, "n_windows": int(n), "same_grid": bool(same_grid),
         "n_decode_changed": int(changed.sum()),
         "n_decode_unchanged": int((~changed).sum())}

    # ---- 1. THE CONTROL that the whole design rests on ------------------- #
    unc = ~changed
    if unc.sum():
        bit_id = bool(np.array_equal(A["cl"][unc], P["cl"][unc]))
        maxd = float(np.abs(A["cl"][unc] - P["cl"][unc]).max())
    else:
        bit_id, maxd = None, None
    R["CONTROL_unchanged_windows_bit_identical"] = {
        "n": int(unc.sum()), "bit_identical": bit_id, "max_abs_diff": maxd,
        "why": "the targeted design's premise; if False the factorisation "
               "marginal=(18/282)*conditional is VOID and nothing below counts"}
    R["control_passes"] = bool(same_grid and bit_id)

    # ---- per-window ADE against GT, for each arm ------------------------- #
    def ade(arm):
        out = np.full(n, np.nan)
        for i in range(n):
            p = _paths(arm["cl"][i], arm["v0"][i])
            out[i] = float(np.linalg.norm(p - arm["g"][i], axis=-1).mean())
        return out

    def ade_of(traj, ref):
        return np.linalg.norm(traj - ref, axis=-1).mean(axis=-1)

    ade_A, ade_P, ade_S1 = ade(A), ade(P), ade(S1)
    ade_ha = ade_of(A["ha"], A["g"])
    ade_hax = ade_of(A["ha0_ext"], A["g"])

    def blk(x, m, name):
        v = x[m]
        return {"name": name, "n": int(m.sum()),
                "mean": float(np.mean(v)) if m.sum() else None}

    for tag, m in (("ALL", np.ones(n, bool)), ("CHANGED", changed),
                   ("UNCHANGED", unc)):
        R[f"ade_{tag}"] = {
            "argmax": blk(ade_A, m, "argmax"),
            "prior050": blk(ade_P, m, "prior050"),
            "argmax_seed1_REPLICATE": blk(ade_S1, m, "argmax@seed1"),
            "ha": blk(ade_ha, m, "ha"),
            "ha0_ext": blk(ade_hax, m, "ha0_ext"),
            "LEVER_delta_prior050_minus_argmax":
                float(np.mean(ade_P[m] - ade_A[m])) if m.sum() else None,
            "REPLICATE_delta_seed1_minus_seed0":
                float(np.mean(ade_S1[m] - ade_A[m])) if m.sum() else None}

    # ⭐ the comparison that decides it
    if changed.sum():
        lev = abs(float(np.mean(ade_P[changed] - ade_A[changed])))
        rep = abs(float(np.mean(ade_S1[changed] - ade_A[changed])))
        R["VERDICT_on_changed_windows"] = {
            "abs_lever_delta": lev, "abs_replicate_delta": rep,
            "lever_over_replicate": (lev / rep) if rep > 0 else None,
            "reading": ("the lever's effect is INSIDE the rig's own run-to-run "
                        "noise — not shown to move anything"
                        if rep > 0 and lev <= rep else
                        "the lever's effect EXCEEDS the run-to-run floor on "
                        "this panel (necessary, still not sufficient at n="
                        f"{int(changed.sum())})")}
    json.dump(R, open(a.out, "w"), indent=1)
    print(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
