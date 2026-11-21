"""EXPLORATORY, zero GPU: what the R5 prior is at 2 s when it is built at K=30.
kd_x built on the 30-step paths (the R5 / trunk-probe 6 s definition) and truncated to its first
10 steps is NOT the banked 2 s kd_x (built on the 10-step paths) wherever the 10-step version
extrapolates past its own end. This measures the consequence on the same 2,399 windows:
mean ADE@2s of each prior vs the 2 s GT (`g`), and ADE/FDE@6s vs `g6` -- plain means,
no CI (the paired bootstrap is the coordinator's floors_6s_exploratory.json)."""
import glob, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kdx_reference_check import retime

FEAT = os.environ.get("REFAV1_PROBE_FEAT", "C:/Users/Admin/refav1_probe/feat_s4")


def main():
    from tanitad.refs.refav1_traj import kinematic_prior
    zs = [np.load(f) for f in sorted(glob.glob(f"{FEAT}/ep*.npz"))]
    cat = lambda k: np.concatenate([z[k] for z in zs])
    kin, g, g6, ok6 = cat("kin"), cat("g").astype(np.float64), cat("g6").astype(np.float64), cat("g6_ok").astype(bool)
    h0, hx = cat("ha0_6").astype(np.float32), cat("ha0_ext_6").astype(np.float32)
    v0, a0, k0 = (torch.from_numpy(kin[:, i].copy()) for i in range(3))
    pri = {p: kinematic_prior(p, v0, a0, k0, steps=30, dt=0.2).numpy().astype(np.float64)
           for p in ("kdx", "damp50", "cv")}
    damp10 = 0.5 * h0[:, :10] + 0.5 * hx[:, :10]
    kdx10 = np.stack([retime(damp10[i], hx[i, :10]) for i in range(len(kin))]).astype(np.float64)
    ade = lambda p, t: float(np.linalg.norm(p - t, axis=-1).mean())
    fde = lambda p, t: float(np.linalg.norm(p[:, -1] - t[:, -1], axis=-1).mean())
    rep = {"n_windows": int(len(kin)), "n_6s_ok": int(ok6.sum()),
           "ade2_kdx10_banked_2s_floor": ade(kdx10, g),
           "ade2_kdx30_first10": ade(pri["kdx"][:, :10], g),
           "ade2_damp50_first10": ade(pri["damp50"][:, :10], g),
           "ade2_cv_first10": ade(pri["cv"][:, :10], g),
           "ade2_ha0_ext": ade(hx[:, :10].astype(np.float64), g)}
    for p in ("kdx", "damp50", "cv"):
        rep[f"ade6_{p}"] = ade(pri[p][ok6], g6[ok6])
        rep[f"fde6_{p}"] = fde(pri[p][ok6], g6[ok6])
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
