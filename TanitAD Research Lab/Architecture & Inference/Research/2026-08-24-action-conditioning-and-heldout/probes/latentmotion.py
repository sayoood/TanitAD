"""E-DEC-59 — DOES EGO MOTION PREDICT THE **LATENT'S OWN CHANGE**?

⭐ THE TARGET E-DEC-58 POINTED AT. The geometric panel failed because its targets
were LABEL-DERIVED: "mean bearing of detected agents" churns as detections appear
and disappear, and over 0.4 s that noise buried a rotation that is a physical
certainty — the CLOSED FORM could not clear its own null either, which is what
diagnosed the target rather than the hypothesis. ⇒ **Use a target that cannot
churn: the latent itself.** Ego motion moves the image by construction, so it must
move the latent.

⚠️ THIS IS NOT NEW GROUND — AND SAYING SO IS THE POINT. `deltaz.py` (E-DEC-40)
already asked almost this question and got **action −0.0109 (t −0.57)** against
drift **+0.1952 (t 8.38)**. What is new is THREE FIXES APPLIED TOGETHER, each of
which was a measured defect in that panel:

  1. ⭐ the **ω parameterisation** (PI directive): `[yaw_rate, a_long, v]` instead
     of `atan(L·κ)`, which is speed-blind and carries a legacy 2.9 m wheelbase.
     ω = v·κ is what actually moves the image.
  2. ⭐ **K-fold fit / per-clip score at ALL usable clips** instead of 20 —
     leave-one-out at n=129 is O(n²) and unaffordable; this is 13× cheaper for the
     same statistic (`panel_kfold.py`).
  3. ⭐ a **MATCHED NULL through the identical code path** (`SPD_NULL=1`). The
     20-clip panels had none, and a null measured under a different estimator does
     not transfer — E-DEC-58's K-fold null reached 4.15 where the leave-one-out one
     reached 3.49.

TARGETS — Δz = z_{t+k} − z_t projected on its top PCA directions, the same
construction E-DEC-40 used, so the numbers are comparable to the banked ones.

COLUMNS
    z_t                    the DRIFT baseline, and the POSITIVE CONTROL — E-DEC-40
                           measured it at t 8.38, so a panel that cannot reproduce
                           that is broken and must not be read.
    ego_state [ω, a, v]    ⭐ the PI's channels, as MEASURED STATE
    z_t + ego_state        the joint; the readable quantity is its MARGINAL over z_t
    constant               reads EXACTLY 0.0000

⭐ THE READ: `(z_t + ego) − z_t` is what ego motion adds to the latent's own drift.
If that clears the matched null, ego-motion conditioning has a real target at last.
If it does not — with the right channels, the right target and adequate power — then
the transition genuinely does not respond to ego motion, and that is a much stronger
negative than anything this campaign has produced so far.
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
sys.path.insert(0, r"C:\Users\Admin\tanitad-mirror\stack")
LEAD = pathlib.Path(os.environ.get(
    "SPD_CORPUS", str(SP / "sp2/cache/physicalai-val130-heldout")))
ARMS = os.environ.get("SPD_ARMS", "rdw8p30k").split(",")
OUT = pathlib.Path(os.environ.get("SPD_OUT", str(SP / "latentmotion.json")))
N_CLIPS = int(os.environ.get("SPD_NCLIPS", "80"))
F, K, DT, K_FOLDS = 100, 4, 0.1, 10
# ⭐ SPD_BAND selects WHICH PCA directions of dz are scored. The default is
# the top 8, matching E-DEC-40. "8:16" is the band where E-DEC-40's own band
# control found the ONLY hint of action content (+0.0258, t 2.73) — a
# LOW-VARIANCE subspace the top-8 projection is blind to by construction.
_b = os.environ.get("SPD_BAND", "0:8").split(":")
BAND_LO, BAND_HI = int(_b[0]), int(_b[1])
N_DIR = BAND_HI - BAND_LO


def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import panel_kfold as PK
    import v7tiny_g2 as G
    from rangeprobe_rff import rff_fold, within_clip_r

    dev = torch.device("cuda")
    clips = sorted(LEAD.glob("*.v2ep.pt"))[:N_CLIPS]
    arms = [a for a in ARMS if (SP / f"v7tiny_{a}" / "ckpt.pt").is_file()]
    null = os.environ.get("SPD_NULL") == "1"
    print("\n  E-DEC-59 — DOES EGO MOTION PREDICT THE LATENT'S OWN CHANGE?")
    print("  channels [yaw_rate, a_long, v] · K-fold fit / per-clip score")
    if null:
        print("  NULL MODE - inputs are Gaussian noise; every t is a null draw")
    print(flush=True)
    rep = {"_evidence_class": "MEASURED (ours; dev-box RTX 4060)",
           "eval_tier": "T0-DIAGNOSTIC", "split": "HELD-OUT", "k": K,
           "null_mode": null, "arms": {}}

    for arm in arms:
        w, st = G.load_arm(arm, dev)
        ZT, EGO, DZ = [], [], []
        with torch.no_grad():
            for c in clips:
                d = torch.load(c, map_location="cpu", weights_only=False)
                yaw = np.asarray(d["poses"], dtype=np.float64)[:, 2]
                z, act, spd = G.encode_clip(w, c, dev, F)
                zt = z.float().numpy().astype(np.float64)
                a = act.float().numpy().astype(np.float64)
                v = spd.float().numpy().astype(np.float64).ravel()
                m = min(len(zt) - K, len(a) - K, len(v) - K, len(yaw) - K - 1)
                if m < 30:
                    continue
                i = np.arange(m)
                omega = wrap(yaw[i + 1] - yaw[i]) / DT        # MEASURED yaw rate
                ZT.append(zt[i])
                EGO.append(np.column_stack([omega, a[i, 1], v[i]]))
                DZ.append(zt[i + K] - zt[i])
        del w
        torch.cuda.empty_cache()
        if len(DZ) < 10:
            print(f"  {arm}: too few clips"); continue

        if null:
            g = np.random.default_rng(int(os.environ.get("SPD_NULL_SEED", "0")))
            ZT = [g.standard_normal(x.shape) for x in ZT]
            EGO = [g.standard_normal(x.shape) for x in EGO]

        ALL = np.concatenate(DZ)
        mu = ALL.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(ALL - mu, full_matrices=False)
        COL = {"z_t (DRIFT / POSITIVE CONTROL)": ZT,
               "ego_state [w, a, v]": EGO,
               "z_t + ego_state": [np.concatenate([a1, b1], 1) for a1, b1 in zip(ZT, EGO)],
               "constant (control)": [np.ones((len(x), 1)) for x in ZT]}
        nrow = sum(len(x) for x in ZT)
        print(f"  === {arm} (step {st}) — {len(ZT)} clips, {nrow} rows, "
              f"top-{N_DIR} PCs of Δz ===")
        print(f"  {'column':<32}{'r':>9}{'shuf':>9}{'t-shuf':>9}{'t':>7}")
        print("  " + "-" * 68)
        cells = {}
        for cn, X in COL.items():
            tr, sh = [], []
            for j in range(BAND_LO, BAND_HI):
                Y = [(dz - mu) @ Vt[j][:, None] for dz in DZ]
                rngj = np.random.default_rng(100 + j)
                Ysh = [y.ravel()[rngj.permutation(len(y))][:, None] for y in Y]
                tr.append(PK.kfold_clip_scores(X, Y, rff_fold, within_clip_r, K_FOLDS))
                sh.append(PK.kfold_clip_scores(X, Ysh, rff_fold, within_clip_r, K_FOLDS))
            tr, sh = np.concatenate(tr), np.concatenate(sh)
            cells[cn] = (tr, sh)
            dd = tr - sh
            t = float(dd.mean()) / max(
                float(dd.std(ddof=1) / np.sqrt(len(dd))), 1e-12)
            print(f"  {cn:<32}{tr.mean():>+9.4f}{sh.mean():>+9.4f}"
                  f"{dd.mean():>+9.4f}{t:>7.2f}", flush=True)

        def tt(x):
            return float(x.mean()) / max(float(x.std(ddof=1) / np.sqrt(len(x))), 1e-12)
        marg = cells["z_t + ego_state"][0] - cells["z_t (DRIFT / POSITIVE CONTROL)"][0]
        ctrl = tt(cells["z_t (DRIFT / POSITIVE CONTROL)"][0]
                  - cells["z_t (DRIFT / POSITIVE CONTROL)"][1])
        rep["arms"][arm] = {
            "step": int(st), "n_clips": len(ZT), "n_rows": nrow,
            "columns": {cn: {"r": round(float(v2[0].mean()), 4),
                             "t": round(tt(v2[0] - v2[1]), 2)}
                        for cn, v2 in cells.items()},
            "drift_control_t": round(ctrl, 2),
            "ego_marginal_over_drift": {"delta": round(float(marg.mean()), 4),
                                        "t": round(tt(marg), 2)}}
        print(f"\n  drift control t {ctrl:+.2f}  (E-DEC-40 banked it at 8.38)")
        print(f"  EGO-STATE's marginal over the drift: {marg.mean():+.4f} "
              f"(t {tt(marg):+.2f})\n", flush=True)

    OUT.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
