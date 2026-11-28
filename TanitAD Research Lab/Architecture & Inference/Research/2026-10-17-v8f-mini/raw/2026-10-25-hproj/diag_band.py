#!/usr/bin/env python3
"""diag_band.py -- POST-HOC DIAGNOSTIC #2, run AFTER the HPROJ read was fixed. NOT a PREREG row, NO bar, changes NO outcome.

Question it answers (one, discriminating): `diag_vhi.py` showed that even an ORACLE speed cap (min(posted limit, GT v_hi)) fixes CONSISTENCY but costs ADE2
(+0.22 / +0.24 m, separated worse than B) although the eval's SPEED rule reads only the intervals that START at tau >= 2.0 s (`_BAND_KNOT` = 10 knots of 0.2 s).
`project_speed` caps EVERY interval from t = 0, so on a window where the ego starts above the cap it brakes at once. Is the ADE cost the 0-2 s part of the cap?

The banded variant below is `project_speed` verbatim with ONE change: the speed cap applies from interval index 20 (tau >= 2.0 s) onward; the Kamm circle applies everywhere.
Nothing is fitted or tuned: 2.0 s is the eval rule's own band start. The identity control (band start 0 == `project_speed`) must read <= 1e-6 m.
"""
from __future__ import annotations

import json
import sys

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

import v8f_hproj as HP  # noqa: E402

H = HP.H
BAND_I = 20           # 0.1 s intervals: interval j covers [0.1 j, 0.1 (j+1)); the rule's band is intervals starting at tau >= 2.0 s


def project_banded(traj60: np.ndarray, cap: np.ndarray, start: int = BAND_I, mu: float = 0.7, dt: float = 0.1, g: float = 9.81) -> np.ndarray:
    """`tanitad.models.v8f_mini.project_speed`, verbatim, except the CAP binds only on intervals j >= `start` (Kamm everywhere)."""
    import torch
    from tanitad.models import v8f_mini as M
    traj = torch.as_tensor(np.asarray(traj60), dtype=torch.float32)
    cap_ms = torch.as_tensor(np.asarray(cap), dtype=torch.float32)
    B, N, _ = traj.shape
    P = torch.cat([traj.new_zeros(B, 1, 2), traj], 1)
    seg = P[:, 1:] - P[:, :-1]
    L = torch.sqrt((seg ** 2).sum(-1) + 1e-12)
    v = L / dt
    ang = torch.atan2(seg[..., 1], seg[..., 0])
    dpsi = torch.zeros_like(ang)
    dpsi[:, 1:] = M.wrap(ang[:, 1:] - ang[:, :-1])
    kappa = dpsi.abs() / L.clamp_min(1e-3)
    kappa = torch.where(L > 1e-3, kappa, torch.zeros_like(kappa))
    v_kamm = torch.sqrt(mu * g / kappa.clamp_min(1e-6))
    capN = cap_ms.reshape(B, 1).expand(B, N).clone()
    capN[:, :start] = 1e9
    vcap = torch.minimum(v_kamm, capN)
    v_new = torch.minimum(v, vcap)
    S_old = torch.cat([L.new_zeros(B, 1), torch.cumsum(L, 1)], 1)
    S_new = torch.cumsum(v_new * dt, 1)
    idx = (torch.searchsorted(S_old.contiguous(), S_new.contiguous(), right=True) - 1).clamp(0, N - 1)
    L_i = torch.gather(L, 1, idx)
    frac = (S_new - torch.gather(S_old, 1, idx)) / L_i.clamp_min(1e-9)
    frac = torch.where(L_i > 1e-9, frac.clamp(0.0, 1.0), torch.zeros_like(frac))
    P0 = torch.gather(P, 1, idx[..., None].expand(-1, -1, 2))
    sg = torch.gather(seg, 1, idx[..., None].expand(-1, -1, 2))
    return (P0 + frac[..., None] * sg).numpy().astype(np.float64)


def main():
    ev = H.load_eval_module()
    from taniteval import v8f_beta_eval as BE
    from tanitad.data import v8f_mini_data as D
    res = {"label": "POST-HOC DIAGNOSTIC #2 (not a PREREG row, no bar, no outcome change)", "band_start_interval": BAND_I, "seeds": {}}
    for seed in H.SEEDS:
        ctx = HP.load_ctx(seed)
        X, z, gt, eid, tau = ctx["X"], ctx["z"], ctx["gt"], ctx["eid"], ctx["tau"]
        S = np.load(f"{HP.OUT}/score_{seed}_perwindow.npz")
        H60, cap_lim = S["H60"], S["cap"].astype(np.float64)
        tau60, _, _ = HP.recon_traj60(tau, ctx["P"], HP.basis_np(ctx["cfg"].nb))
        lim = np.asarray(X["limit_ms"], np.float64)
        vhi_gt = np.where(np.asarray(X["has"], bool), np.asarray(X["cons"][:, D.CONS_IDX["v_hi_ms"]], np.float64), np.nan)
        vhi_pr = np.maximum(z["cons_pred"][:, D.CONS_IDX["v_hi_ms"]].astype(np.float64), 0.0)
        caps = {"lim": cap_lim, "pr": np.fmin(lim, vhi_pr), "or": np.where(np.isfinite(vhi_gt), np.fmin(lim, vhi_gt), lim)}
        info = {"identity_control_band0_vs_project_speed_maxabs_m": float(np.abs(project_banded(H60, cap_lim, start=0) - HP.project_f32(H60, cap_lim)).max())}
        assert info["identity_control_band0_vs_project_speed_maxabs_m"] <= 1e-6, info
        arms = {}
        for kname, cap in caps.items():
            arms[f"HPb_{kname}"] = project_banded(H60, cap.astype(np.float32))[:, 1::2]
            arms[f"Bpb_{kname}"] = project_banded(tau60, cap.astype(np.float32))[:, 1::2]
        pwB = H.arm_pw(tau, gt, BE)
        vB = {k: z[k].astype(float) for k in HP.CK}
        ex = H.extras_of(z, D)
        out = {}
        for n, p in arms.items():
            r, pw = ev.evaluate(p, X, ex, HP.B_BOOT, with_families=False)
            pwA = H.arm_pw(p, gt, BE)
            V = {k: pw[k].astype(float) for k in HP.CK}
            out[n] = {"rate": {k: float(V[k].mean()) for k in HP.CK},
                      "ade2_minus_B": BE.paired_row(pwA["ade2"], pwB["ade2"], eid, HP.B_BOOT), "ade6_minus_B": BE.paired_row(pwA["ade6"], pwB["ade6"], eid, HP.B_BOOT),
                      "cons_minus_B": {k: BE.paired_row(V[k], vB[k], eid, HP.B_BOOT) for k in HP.CK},
                      "accel_minus_B": BE.paired_row(pwA["fw_lon_accel_mae"], pwB["fw_lon_accel_mae"], eid, HP.B_BOOT)}
        info["arms"] = out
        res["seeds"][seed] = info
        print(seed, {n: {"ade2": round(o["ade2_minus_B"]["mean"], 4), "ade6": round(o["ade6_minus_B"]["mean"], 4),
                         "cons_any": f"{o['cons_minus_B']['viol_any']['mean']:+.4f} hi {o['cons_minus_B']['viol_any']['hi']:+.4f}"} for n, o in out.items()}, flush=True)
    H.jdump(res, f"{HP.OUT}/diag_band.json")


if __name__ == "__main__":
    main()
