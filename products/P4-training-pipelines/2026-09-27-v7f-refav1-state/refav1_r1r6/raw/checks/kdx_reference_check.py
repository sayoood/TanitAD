"""Known-value control on REAL data (zero GPU): the batched torch priors of
`tanitad.refs.refav1_traj` against the banked reference implementation.

Reference = `.../2026-09-27-refav1-fullgrid-loncomb3/raw/analysis/explore/build_explore.py::retime`
(copied VERBATIM below) applied to the arm's own per-window 30-step paths
(`ha0_6`, `ha0_ext_6`), banked by the trunk probe's extractor in
C:/Users/Admin/refav1_probe/feat_s4/ep*.npz together with the t0 kinematics
`kin` = (v0, a0, kappa0). Our priors are rebuilt from `kin` ALONE.
Reports max |delta| per quantity; the claim is bit-identity (0.0).
"""
import glob, json, os, sys
import numpy as np
import torch

FEAT = os.environ.get("REFAV1_PROBE_FEAT", "C:/Users/Admin/refav1_probe/feat_s4")


# ---- VERBATIM from build_explore.py (the banked kd_x definition) ----------
def arclen(p):                      # p [10,2] positions at k=1..10, origin at k=0
    P = np.concatenate([np.zeros((1, 2), p.dtype), p], 0)
    seg = np.diff(P, axis=0)
    L = np.linalg.norm(seg, axis=-1)
    return P, seg, L, np.concatenate([[0.0], np.cumsum(L)])


def retime(path, lon_src):
    P, seg, L, S = arclen(path.astype(np.float64))
    s_src = arclen(lon_src.astype(np.float64))[3][1:]
    out = np.zeros((len(s_src), 2))
    nz = np.nonzero(L > 1e-9)[0]
    for i, s in enumerate(s_src):
        if S[-1] < 1e-6 or len(nz) == 0:          # degenerate damped path: straight ahead (+x = heading at t0)
            out[i] = (s, 0.0); continue
        if s <= S[-1]:
            j = int(np.clip(np.searchsorted(S, s, side="right") - 1, 0, len(L) - 1))
            while L[j] <= 1e-9 and j + 1 < len(L):
                j += 1
            out[i] = P[j] + (s - S[j]) / max(L[j], 1e-12) * seg[j] if L[j] > 1e-9 else P[j]
        else:
            jj = nz[-1]
            out[i] = P[-1] + (s - S[-1]) * seg[jj] / L[jj]
    return out.astype(np.float32)
# ---------------------------------------------------------------------------


def main():
    from tanitad.refs.refav1_traj import kinematic_floor_paths, kinematic_prior, retime_paths
    files = sorted(glob.glob(f"{FEAT}/ep*.npz"))
    if not files:
        print("NO BANKED FEATURES at", FEAT); return 2
    kin, h0, hx = [], [], []
    for f in files:
        z = np.load(f)
        kin.append(z["kin"]); h0.append(z["ha0_6"]); hx.append(z["ha0_ext_6"])
    kin = np.concatenate(kin); h0 = np.concatenate(h0).astype(np.float32)
    hx = np.concatenate(hx).astype(np.float32)
    n = len(kin)
    v0, a0, k0 = (torch.from_numpy(kin[:, i].copy()) for i in range(3))
    # 1) the integrated floors, batched, vs the arm's per-window paths
    t_h0, t_hx = kinematic_floor_paths(v0, a0, k0, 30, 0.2, units="kappa")
    d_h0 = float(np.abs(t_h0.numpy() - h0).max())
    d_hx = float(np.abs(t_hx.numpy() - hx).max())
    # 2) retime alone, on the SAME float32 inputs
    damp_np = 0.5 * h0 + 0.5 * hx
    ref_kdx = np.stack([retime(damp_np[i], hx[i]) for i in range(n)])
    t_kdx_on_ref = retime_paths(torch.from_numpy(damp_np), torch.from_numpy(hx)).numpy()
    d_retime = float(np.abs(t_kdx_on_ref - ref_kdx).max())
    # 3) end to end, from kin alone
    t_kdx = kinematic_prior("kdx", v0, a0, k0, steps=30, dt=0.2).numpy()
    t_d50 = kinematic_prior("damp50", v0, a0, k0, steps=30, dt=0.2).numpy()
    t_cv = kinematic_prior("cv", v0, a0, k0, steps=30, dt=0.2).numpy()
    rep = {"n_windows": int(n), "n_episodes": len(files),
           "max_abs_ha0": d_h0, "max_abs_ha0_ext": d_hx,
           "max_abs_retime_same_inputs": d_retime,
           "max_abs_kdx_end_to_end": float(np.abs(t_kdx - ref_kdx).max()),
           "max_abs_damp50_end_to_end": float(np.abs(t_d50 - damp_np).max()),
           "max_abs_cv_end_to_end": float(np.abs(t_cv - h0).max()),
           "n_windows_kdx_not_bitexact": int((np.abs(t_kdx - ref_kdx).reshape(n, -1).max(1) > 0).sum()),
           "kdx_first10_vs_2s_kdx_note": "the 2 s kd_x is retime on the 10-step paths; see max_abs_kdx30_first10_vs_kdx10"}
    ref10 = np.stack([retime(damp_np[i, :10], hx[i, :10]) for i in range(n)])
    rep["max_abs_kdx30_first10_vs_kdx10"] = float(np.abs(t_kdx[:, :10] - ref10).max())
    rep["n_windows_kdx30_first10_differs_from_kdx10"] = int(
        (np.abs(t_kdx[:, :10] - ref10).reshape(n, -1).max(1) > 1e-6).sum())
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
