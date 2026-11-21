"""progress_retime_torch.py -- refcv9 L1 annex E0-next: `ProgressRetime`, the in-forward re-timing layer, as a torch module
(a port of the shipped stage's numpy geometry: a0k0_headroom_retime_r8.polyline_geometry / eval_polyline / retime_progress).

Re-times the SELECTED path to a target arc length s' at every slot; the path vertices are DETACHED (no gradient reaches selection);
s' is differentiable, so a progress head trains through it.  Every re-timed point lies on the selected polyline or its end ray, so the
path shape (lateral) is untouched by construction.
    mode="fixed": s' = s_P + g(t) (s_sel - s_P)      g = the frozen deploy table (8 knots)            == refcv8 + stage   (mechanism a)
    mode="head" : s' = s_P + g_cal(t) * dshat         dshat [B, 8] from a progress head, g_cal init 1  (mechanism c4)
s' is clipped at >= 0.  NOT wired into any model here (a design-annex deliverable; the repo port is a Master Mind decision).

    python -X utf8 progress_retime_torch.py --check <DEPLOY8_POS_*.npz> <deploy_gains.json>   # known-value controls, prints ZZPRTCHECK..ZZ
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch


def polyline_geometry(O: torch.Tensor) -> dict:
    """O [n, S, 2] -> V [n, S+1, 2] (V0 = 0), L [n, S+1] cumulative arc, u [n, S, 2] unit dirs (0 on zero-length), u_end [n, 2]."""
    if O.ndim != 3 or O.shape[-1] != 2:
        raise ValueError(f"plans must be [n, S, 2], got {tuple(O.shape)}")
    if not torch.isfinite(O).all():
        raise ValueError("non-finite plan position: the re-timing is undefined")
    n, S, _ = O.shape
    V = torch.cat([O.new_zeros(n, 1, 2), O], 1)
    seg = V[:, 1:] - V[:, :-1]
    ln = torch.linalg.vector_norm(seg, dim=-1)
    L = torch.cat([O.new_zeros(n, 1), torch.cumsum(ln, 1)], 1)
    pos = ln > 0
    u = torch.where(pos[..., None], seg / torch.where(pos, ln, torch.ones_like(ln))[..., None], torch.zeros_like(seg))
    rev = torch.flip(pos, [1]).to(torch.int64)
    last = torch.where(pos.any(1), S - 1 - torch.argmax(rev, 1), torch.zeros(n, dtype=torch.int64, device=O.device))
    u_end = u[torch.arange(n, device=O.device), last].clone()
    fb = ~pos.any(1)
    u_end[fb] = torch.tensor([1.0, 0.0], dtype=O.dtype, device=O.device)
    return {"V": V, "L": L, "u": u, "u_end": u_end}


def eval_polyline(geo: dict, s: torch.Tensor) -> torch.Tensor:
    """C(s) for s [n, m] (>= 0): linear along the polyline; beyond L_S along the end ray anchored at V_S.  -> [n, m, 2]."""
    V, L, u, u_end = geo["V"], geo["L"], geo["u"], geo["u_end"]
    n, S = u.shape[0], u.shape[1]
    idx = torch.zeros(s.shape, dtype=torch.int64, device=s.device)
    for j in range(1, S):
        idx = idx + (s >= L[:, j:j + 1]).to(torch.int64)
    rows = torch.arange(n, device=s.device)[:, None].expand_as(idx)
    P = V[rows, idx] + (s - L[rows, idx])[..., None] * u[rows, idx]
    beyond = s >= L[:, S:S + 1]
    Pe = V[:, S:S + 1] + (s - L[:, S:S + 1])[..., None] * u_end[:, None, :]
    return torch.where(beyond[..., None], Pe, P)


class ProgressRetime(torch.nn.Module):
    def __init__(self, gains8, mode: str = "fixed"):
        super().__init__()
        if mode not in ("fixed", "head"):
            raise ValueError(mode)
        self.mode = mode
        self.register_buffer("g_fixed", torch.as_tensor(np.asarray(gains8, np.float64), dtype=torch.float64))
        self.g_cal = torch.nn.Parameter(torch.ones(len(gains8), dtype=torch.float64))   # (c4) calibration, init 1

    def forward(self, plan8: torch.Tensor, prior8: torch.Tensor, dshat: torch.Tensor | None = None) -> torch.Tensor:
        geo = polyline_geometry(plan8.detach().to(torch.float64))          # the SELECTED path: no gradient to selection
        s_sel = geo["L"][:, 1:]
        s_P = polyline_geometry(prior8.detach().to(torch.float64))["L"][:, 1:]
        if self.mode == "fixed":
            s = s_sel + (1.0 - self.g_fixed) * (s_P - s_sel)                 # == s_P + g (s_sel - s_P); g == 1 -> s_sel exactly
        else:
            if dshat is None:
                raise ValueError("mode='head' needs dshat [B, 8]")
            s = s_P + self.g_cal * dshat.to(torch.float64)
        s = torch.clamp(s, min=0.0)
        return eval_polyline(geo, s).to(plan8.dtype)


def check(npz: str, gains_json: str) -> int:
    z = np.load(npz, allow_pickle=True)
    g = json.load(open(gains_json, encoding="utf-8"))["gains"]
    plan8, prior8 = torch.from_numpy(z["S2_plan8"]), torch.from_numpy(z["S2_prior8"])
    want = z["S2_pos8"].astype(np.float64)
    m = ProgressRetime(g, "fixed")
    with torch.no_grad():
        got = m(plan8.double(), prior8.double()).numpy()
        r1 = float(np.abs(got - want).max())                                  # K-a: the deployed stage, all 8 slots
        m1 = ProgressRetime([1.0] * 8, "fixed")
        r2 = float(np.abs(m1(plan8.double(), prior8.double()).numpy() - z["S2_plan8"].astype(np.float64)).max())   # g=1 -> plan
        sP = polyline_geometry(prior8.double())["L"][:, 1:]
        sel = polyline_geometry(plan8.double())["L"][:, 1:]
        mh = ProgressRetime(g, "head")
        p_h0 = mh(plan8.double(), prior8.double(), dshat=sel - sP)            # head with dshat = s_sel - s_P, g_cal 1 -> plan
        r4 = float(np.abs(p_h0.numpy() - z["S2_plan8"].astype(np.float64)).max())
    # gradient: d(loss)/d(dshat) exists, d/d(plan) does not
    plan_g = plan8.double().clone().requires_grad_(True)
    dsh = (sel - sP).clone().requires_grad_(True)
    out = ProgressRetime(g, "head")(plan_g, prior8.double(), dshat=dsh)
    out.sum().backward()
    grad_ok = bool(dsh.grad is not None and torch.isfinite(dsh.grad).all() and dsh.grad.abs().sum() > 0 and plan_g.grad is None)
    # mutation (must go RED): a table shifted by +0.05 must NOT reproduce the deployed positions
    with torch.no_grad():
        rm = float(np.abs(ProgressRetime([x + 0.05 for x in g], "fixed")(plan8.double(), prior8.double()).numpy() - want).max())
    res = {"n": int(plan8.shape[0]), "Ka_fixed_vs_banked_pos8_maxabs_m": r1, "Kb_g1_vs_plan_maxabs_m": r2,
           "Kc_head_identity_vs_plan_maxabs_m": r4, "Kd_grad_to_dshat_only": grad_ok, "Km_mutation_maxabs_m": rm}
    res["PASS"] = bool(r1 <= 1e-5 and r2 <= 1e-9 and r4 <= 1e-6 and grad_ok and rm > 1e-3)
    print(json.dumps(res), "ZZPRTCHECK" + ("PASS" if res["PASS"] else "FAIL") + "ZZ")
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--check":
        raise SystemExit(check(sys.argv[2], sys.argv[3]))
    print(__doc__)
