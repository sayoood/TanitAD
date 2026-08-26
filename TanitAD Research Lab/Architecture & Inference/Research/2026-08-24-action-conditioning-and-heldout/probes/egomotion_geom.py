"""E-DEC-58 — EGO-MOTION STATE vs GEOMETRIC scene change. (PI directive, 2026-08-26)

⭐⭐⭐ THE PI'S CORRECTION, AND IT REFRAMES THE WHOLE CAMPAIGN:
    "why are you using curvature, use just the values: yaw rate, long acc and v0
     as measured state not an action. v0 is required to predict the state of the
     environment..."

TWO THINGS FOLLOW.

(1) ⛔ THE CHANNEL WAS BADLY PARAMETERISED. We fed `steer = atan(L·curvature)` with
    L a LEGACY CONSTANT 2.9 m applied to every clip — a bicycle-model steering
    proxy, i.e. a fake per-clip constant baked into a quantity that only ever needed
    to be a rotation rate. And `steer` CONTAINS NO SPEED: the geometrically
    meaningful quantity is the yaw RATE, omega = v*kappa, which the model had to
    reconstruct from two separate channels through a FiLM bottleneck. ⇒ Feed the
    MEASURED values directly: **[yaw_rate, a_long, v]**.

(2) ⛔⛔ AND WE TESTED THE WRONG TARGETS. E-DEC-48b asked whether the "action"
    predicts `n_agents`, `occ_center`, `n_free_cols` — **COUNTING descriptors that
    are EGO-MOTION-INVARIANT BY CONSTRUCTION.** The number of agents in view does
    not change because you yaw. Concluding "the action carries no information about
    the scene" from those targets was answering a question ego motion was never
    going to affect.

⭐ WHAT EGO MOTION ACTUALLY DETERMINES IS **GEOMETRY**: bearings ROTATE with omega,
ranges CLOSE with v. Over k ticks a static object's bearing shifts by ~ -omega*k*dt
and its range shrinks by ~ v*k*dt. **Those are the targets.**

THE PANEL — targets at t+k, all GEOMETRIC:
    d_bearing   change in mean agent bearing        <- should track -omega*k*dt
    d_range     change in mean in-lane range        <- should track -v*k*dt
    bearing_tp1 the bearing itself at t+k
    range_tp1   the range itself at t+k

COLUMNS:
    scene_t                the same geometry at t — the POSITIVE CONTROL
    ego_state_t            ⭐ [yaw_rate, a_long, v] — the PI's channels, RAW
    scene_t + ego_state_t  the joint
    CLOSED FORM            ⭐⭐ -omega*k*dt and -v*k*dt: pure kinematics, NO fitting.
                           If this predicts as well as the fit, the relation is
                           geometry and needs no learning — which is the POINT, not
                           a defect: it is what the predictor should be reproducing.
    constant               reads EXACTLY 0.0000

⚠️ RUN AT FULL CORPUS (129 clips), not 20. The 20-clip panels that produced this
campaign's negatives were underpowered — the null reached |t| 3.49 largely because
n was small. That mistake is not repeated here.

CPU only: needs the agent labels and the ego channels, no encoder, no GPU.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np
import torch

SP = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP / "sp2"))
LEAD = pathlib.Path(os.environ.get(
    "SPD_CORPUS", str(SP / "sp2/cache/physicalai-val130-heldout")))
LABELS = pathlib.Path(os.environ.get("SPD_LABELS", str(SP / "sp2/val130_agents.jsonl")))
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "egomotion_geom.json")))
N_CLIPS, F, K, DT = 129, 100, 4, 0.1
K_FOLDS = 10


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import panel_kfold as PK
    from rangeprobe_rff import rff_fold, within_clip_r

    LAB = {}
    with open(LABELS, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                LAB.setdefault(r["clip_id"], {})[int(r["frame_idx"])] = r.get("agents", [])

    def geom(cid, m):
        """mean agent BEARING and mean in-lane RANGE per frame — the two
        quantities ego motion actually moves."""
        b = np.full(m, np.nan)
        rg = np.full(m, np.nan)
        for i in range(m):
            a = LAB.get(cid, {}).get(i, [])
            if not a:
                continue
            cx = np.array([float(q.get("cx", np.nan)) for q in a])
            cy = np.array([float(q.get("cy", np.nan)) for q in a])
            ok = np.isfinite(cx) & np.isfinite(cy) & (cx > 0.5)
            if ok.any():
                b[i] = float(np.mean(np.arctan2(cy[ok], cx[ok])))     # bearing, rad
                inl = ok & (np.abs(cy) < 3.0)
                if inl.any():
                    rg[i] = float(np.min(cx[inl]))                     # nearest range
        return b, rg

    clips = []
    for c in sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]:
        d = torch.load(c, map_location="cpu", weights_only=False)
        if d["clip_id"] in LAB:
            clips.append((c, d))
    print("\n  E-DEC-58 — EGO-MOTION STATE vs GEOMETRIC scene change")
    print("  channels: [yaw_rate, a_long, v] as MEASURED STATE (PI directive)")
    print(f"  {len(clips)} held-out clips, k={K} ({K*DT:.1f} s)\n", flush=True)

    SCN, EGO, KIN = [], [], []
    T = {k: [] for k in ("d_bearing", "d_range", "bearing_t+k", "range_t+k")}
    for c, d in clips:
        pos = np.asarray(d["poses"], dtype=np.float64)
        act = np.asarray(d["actions"], dtype=np.float64)
        yaw, v = pos[:, 2], pos[:, 3]
        n = min(len(yaw), F)
        b, rg = geom(d["clip_id"], n)
        m = n - K - 1
        if m < 25:
            continue
        i = np.arange(m)
        omega = wrap(yaw[i + 1] - yaw[i]) / DT                # MEASURED yaw rate
        alon = act[i, 1]                                       # measured a_long
        vt = v[i]
        ok = np.isfinite(b[i]) & np.isfinite(b[i + K]) & np.isfinite(rg[i]) \
            & np.isfinite(rg[i + K])
        if ok.sum() < 25:
            continue
        j = i[ok]
        SCN.append(np.column_stack([b[j], rg[j]]))
        EGO.append(np.column_stack([omega[ok], alon[ok], vt[ok]]))
        # ⭐ pure kinematics, no fitting: bearing rotates by -omega*k*dt,
        #    range closes by -v*k*dt
        KIN.append(np.column_stack([-omega[ok] * K * DT, -vt[ok] * K * DT]))
        T["d_bearing"].append(wrap(b[j + K] - b[j])[:, None])
        T["d_range"].append((rg[j + K] - rg[j])[:, None])
        T["bearing_t+k"].append(b[j + K][:, None])
        T["range_t+k"].append(rg[j + K][:, None])
    if len(SCN) < 8:
        print("  too few usable clips"); return 1

    # ⭐ THE MATCHED NULL, THROUGH THE IDENTICAL CODE PATH. SPD_NULL=1 replaces the
    # INPUT columns with Gaussian noise of the same shape while keeping the real
    # targets, so every t it produces is a draw from this estimator's null. Running
    # it from the SAME script is deliberate: a null measured by a separate program
    # can drift from the panel it is supposed to calibrate.
    if os.environ.get("SPD_NULL") == "1":
        _g = np.random.default_rng(int(os.environ.get("SPD_NULL_SEED", "0")))
        SCN = [_g.standard_normal(x.shape) for x in SCN]
        EGO = [_g.standard_normal(x.shape) for x in EGO]
        KIN = [_g.standard_normal(x.shape) for x in KIN]
        print("  NULL MODE - inputs are Gaussian noise; every t below is a "
              "null draw\n", flush=True)

    COL = {"scene_t (POSITIVE CONTROL)": SCN,
           "ego_state [w, a, v]": EGO,
           "scene_t + ego_state": [np.concatenate([s, e], 1) for s, e in zip(SCN, EGO)],
           "CLOSED FORM (kinematics)": KIN,
           "constant (control)": [np.ones((len(x), 1)) for x in SCN]}
    nrow = sum(len(x) for x in SCN)
    print(f"  {len(SCN)} clips, {nrow} rows\n")
    print(f"  {'target':<15}{'column':<28}{'r':>9}{'shuf':>9}{'t-shuf':>9}{'t':>7}")
    print("  " + "-" * 80)
    rng = np.random.default_rng(0)
    rep = {"_evidence_class": "MEASURED (ours; dev-box, CPU)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT", "k": K,
           "n_clips": len(SCN), "n_rows": nrow,
           "channels": "[yaw_rate, a_long, v] as MEASURED STATE (PI directive)",
           "targets": {}}
    for tn, Y in T.items():
        Ysh = [y.ravel()[rng.permutation(len(y))][:, None] for y in Y]
        cells = {}
        for cn, X in COL.items():
            # ⭐ K-FOLD FIT, PER-CLIP SCORE. Leave-one-clip-out is O(n^2) and at
            # n=129 that is 42x the 20-clip cost, not the 6.5x I first quoted —
            # three jobs burned ~90k CPU-seconds proving it. The t-test only needs
            # per-CLIP scores; the FIT does not have to be per-clip.
            tr = PK.kfold_clip_scores(X, Y, rff_fold, within_clip_r, K_FOLDS)
            sh = PK.kfold_clip_scores(X, Ysh, rff_fold, within_clip_r, K_FOLDS)
            cells[cn] = (tr, sh)
            dd = tr - sh
            t = float(dd.mean()) / max(
                float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            print(f"  {tn:<15}{cn:<28}{tr.mean():>+9.4f}{sh.mean():>+9.4f}"
                  f"{dd.mean():>+9.4f}{t:>7.2f}", flush=True)
        def tt(x):
            return float(x.mean()) / max(float(x.std(ddof=1) / np.sqrt(len(x))), 1e-12)
        marg = cells["scene_t + ego_state"][0] - cells["scene_t (POSITIVE CONTROL)"][0]
        rep["targets"][tn] = {
            cn: {"r": round(float(v2[0].mean()), 4), "t": round(tt(v2[0] - v2[1]), 2)}
            for cn, v2 in cells.items()}
        rep["targets"][tn]["ego_marginal_given_scene"] = {
            "delta": round(float(marg.mean()), 4), "t": round(tt(marg), 2)}
        print(f"  {'':15}-> EGO-STATE's marginal given the scene: "
              f"{marg.mean():+.4f} (t {tt(marg):+.2f})\n", flush=True)

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
