#!/usr/bin/env python3
"""T-H -- the DECISION PROBE. Does the plan's 0-2 s acceleration depend on the GAP?

    python t_h_decision_probe.py --dump <DUMP> [--dump2 <SEED1 DUMP>] --out out.json

⛔ ZERO GPU. Everything here is read from a banked dump + the B1 EVAL lead block.

WHAT IS REGRESSED. For each lead window (the lead block's own LEAD state, on the dump's
own window grid), the planned 0-2 s longitudinal acceleration

    a_plan = 2 * (s(T) - v0*T) / T^2 ,  s = arc length along the planned polyline, T = 2 s

is regressed on ``[gap0_m, closing_rate_mps, v0_mps, a0_mps2]`` with an intercept.
`closing_rate = -rel_speed_mps` of the block (rel_speed + = the lead is pulling away),
so a POSITIVE closing rate means the lead is being approached.

⭐ a0 IS NOT RE-DERIVED. It is read out of the `ha0_ext` arm of the SAME dump with the
SAME estimator: `ha0_ext` integrates a constant (a0, k0) from the measured v0
(`refav1_arm.hold_ext_controls`), so a_plan(`ha0_ext`) IS a0 by construction. That is
also what makes the control exact.

THE CONTROLS, and their KNOWN values:
  * `ha0_ext`  -- plans accel == a0, so with a0 among the regressors its gap coefficient
                  is EXACTLY 0 and its a0 coefficient is EXACTLY 1;
  * `ha0`      -- plans accel == 0, so EVERY coefficient is EXACTLY 0;
  * `g` (human)-- the reference coefficient the model's is compared against.

Intervals: episode-cluster bootstrap over CLIPS (`taniteval.ci`), 2,000 resamples.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

REPO = os.environ.get("D3_REPO", r"C:\Users\Admin\refcv5cmp\repo")
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
LEAD_BLOCK = r"C:\Users\Admin\refcv5cmp\data\b1_eval_lead_block.npz"
REGRESSORS = ("gap_m", "closing_rate_mps", "v0_mps", "a0_mps2")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def plan_accel(P, v0, T):
    """2*(arc length - v0*T)/T^2 -- the ONE definition used for every arm."""
    s = np.zeros(P.shape[0])
    prev = np.zeros((P.shape[0], 2))
    for j in range(P.shape[1]):
        s += np.linalg.norm(P[:, j] - prev, axis=-1)
        prev = P[:, j]
    return 2.0 * (s - v0 * T) / (T * T)


def ols(X, y):
    Xd = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta                                   # [intercept, *REGRESSORS]


def boot_coeffs(X, y, eid, n_boot=2000, seed=0, alpha=0.05):
    from taniteval.ci import episode_index
    uniq, idx_by_ep = episode_index(list(eid))
    point = ols(X, y)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_ep[uniq[p]] for p in pick])
        try:
            draws.append(ols(X[sel], y[sel]))
        except np.linalg.LinAlgError:
            continue
    D = np.asarray(draws)
    lo = np.percentile(D, 100 * alpha / 2, axis=0)
    hi = np.percentile(D, 100 * (1 - alpha / 2), axis=0)
    out = {}
    for i, nm in enumerate(("intercept",) + REGRESSORS):
        out[nm] = {"coef": float(point[i]), "lo": float(lo[i]), "hi": float(hi[i]),
                   "separated_from_0": bool(lo[i] > 0 or hi[i] < 0)}
    out["_n_boot"] = int(D.shape[0])
    out["_n"] = int(X.shape[0])
    out["_n_episodes"] = int(len(uniq))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--dump2", default=None, help="a second inference seed, for the floor")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arm_mod = _mod("refcv3_arm_real",
                   os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py"))
    ra = _mod("refav1_arm_real",
              os.path.join(REPO, "taniteval", "tools", "refav1_arm.py"))

    files = sorted(glob.glob(os.path.join(a.dump, "ep*.npz")))
    man = json.load(open(os.path.join(a.dump, "manifest.json"), encoding="utf-8"))
    arms = ("os", "ha0", "ha0_ext")
    P = {k: [] for k in arms + ("g",)}
    WS, CID = [], []
    by_fi = {int(e["file_index"]): e["clip_id"] for e in man["episodes"]}
    for f in files:
        fi = int(os.path.basename(f)[2:5])
        with np.load(f) as z:
            for k in arms + ("g",):
                P[k].append(np.asarray(z[k], dtype=np.float64))
            WS.append(np.asarray(z["ws"], dtype=np.int64).reshape(-1))
            CID.extend([by_fi[fi]] * len(z["ws"]))
    P = {k: np.concatenate(v) for k, v in P.items()}
    WS = np.concatenate(WS)
    CID = np.asarray(CID, dtype=object)

    if a.dump2:
        P2 = []
        for f in sorted(glob.glob(os.path.join(a.dump2, "ep*.npz"))):
            with np.load(f) as z:
                P2.append(np.asarray(z["os"], dtype=np.float64))
        P["os_seed1"] = np.concatenate(P2)

    # -- the lead join (harness's own) ---------------------------------------
    dt, k = 0.5, 4
    raw_off = int(man["corpus"]["frames"]["provider_to_raw_frame_offset"])
    view, idx, meta, info = arm_mod.lead_block_common_grid(LEAD_BLOCK, dt, k)
    shim_dir = tempfile.mkdtemp(prefix="d3_th_")
    shim = []
    for f in files:
        with np.load(f) as d:
            ws = np.asarray(d["ws"]).astype(np.int64).reshape(-1)
            v0 = np.asarray(d["v0"], dtype=np.float32).reshape(-1)
        sf = os.path.join(shim_dir, os.path.basename(f))
        np.savez(sf, ws=ws + raw_off, v0=v0)
        shim.append(sf)
    lead = ra.join_lead_block(shim, man, view, idx, k=info["k"], dt=info["dt_s"],
                              frame_of_t=lambda t: t)
    lead.pop("coverage")
    cols, dtv = info["dump_cols"], info["dt_s"]
    T = dtv * len(cols)
    v0 = np.asarray(lead["speeds"], dtype=np.float64)
    gap = np.asarray(lead["gap0_m"], dtype=np.float64)
    state = np.asarray(lead["state"]).astype(str)
    eid = np.asarray(lead["eid"], dtype=object)

    # -- rel_speed, joined on the SAME key (clip_id, RAW frame) ---------------
    z = np.load(LEAD_BLOCK, allow_pickle=True)
    rs_map = {(c, int(f)): float(r) for c, f, r in
              zip(z["clip_id"].astype(str), z["frame"].astype(np.int64),
                  z["rel_speed_mps"].astype(np.float64))}
    closing = np.array([-rs_map.get((CID[i], int(WS[i] + raw_off)), np.nan)
                        for i in range(len(WS))])

    a0 = plan_accel(P["ha0_ext"][:, cols], v0, T)
    rows = {}
    m = (state == "LEAD") & np.isfinite(gap) & np.isfinite(closing) & np.isfinite(a0)
    X = np.column_stack([gap[m], closing[m], v0[m], a0[m]])
    print(f"[t-h] n_lead_windows={int(m.sum())} of {len(m)} · "
          f"n_ep={len(set(eid[m].tolist()))}")
    for nm in ("os", "os_seed1", "g", "ha0", "ha0_ext"):
        if nm not in P:
            continue
        y = plan_accel(P[nm][:, cols], v0, T)[m]
        rows[nm] = boot_coeffs(X, y, eid[m], n_boot=a.n_boot, seed=a.seed)
        rows[nm]["_mean_plan_accel_mps2"] = float(y.mean())
        rows[nm]["_std_plan_accel_mps2"] = float(y.std())

    out = {"_what": "T-H decision probe: planned 0-2 s accel ~ gap + closing + v0 + a0",
           "tier": "T1 (self-action OPEN loop)",
           "n_windows_scored": int(m.sum()),
           "n_windows_total": int(len(m)),
           "n_episodes": int(len(set(eid[m].tolist()))),
           "dump": a.dump, "dump_seed1": a.dump2,
           "regressors": list(REGRESSORS),
           "response": "2*(arc length of the plan over 2 s - v0*2)/4  [m/s^2]",
           "controls_known_values": {
               "ha0": "accel == 0 -> EVERY coefficient EXACTLY 0",
               "ha0_ext": "accel == a0 -> gap/closing/v0 coefficients EXACTLY 0, "
                          "a0 coefficient EXACTLY 1"},
           "estimator": (f"OLS + episode-cluster bootstrap over clips "
                         f"(taniteval.ci.episode_index), n_boot={a.n_boot}, "
                         f"seed={a.seed}"),
           "arms": rows}
    print(json.dumps({k: {"gap_m": v["gap_m"], "closing_rate_mps": v["closing_rate_mps"],
                          "a0_mps2": v["a0_mps2"]}
                      for k, v in rows.items()}, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
        print("[out]", a.out)


if __name__ == "__main__":
    main()
