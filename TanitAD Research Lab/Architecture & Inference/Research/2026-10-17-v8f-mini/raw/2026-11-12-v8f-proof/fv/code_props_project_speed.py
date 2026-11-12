"""Property check of the EXISTING project_speed (v8f_mini.py at 3e264bbf) against FV_SPEC_V3_CODE.md. The code under test is imported, never edited.
Usage: PYTHONPATH=<tree>/stack python code_props_project_speed.py FV_SPEC_V3_CODE.md CODE_PROPS_RESULT.json"""
import hashlib, inspect, json, math, sys
import torch
import tanitad.models.v8f_mini as M

SPEC, OUT = sys.argv[1], sys.argv[2]
torch.set_num_threads(2)
DT, MU, G = 0.1, 0.7, 9.81
N, B = 60, 20000
gen = torch.Generator().manual_seed(20261113)
D = torch.float64


def paths_from(v, dpsi0, dpsi):
    """v [B,N] interval speeds, heading of segment 0 = dpsi0, later heading increments dpsi [B,N] (dpsi[:,0] ignored)."""
    ang = dpsi0[:, None] + torch.cumsum(torch.cat([torch.zeros(v.shape[0], 1, dtype=D), dpsi[:, 1:]], 1), 1)
    seg = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (v * DT)[..., None]
    return torch.cumsum(seg, 1)


def fam(name):
    r = lambda *s: torch.rand(*s, generator=gen, dtype=D)
    if name == "F1_smooth":
        v = (2 + 28 * r(B, 1)) * (0.6 + 0.8 * r(B, N)).cumsum(1) / torch.arange(1, N + 1, dtype=D)
        kap = (r(B, 1) - 0.5) * 0.2
        return paths_from(v, torch.zeros(B, dtype=D), kap * v * DT)
    if name == "F2_vertex":
        v = (3 + 25 * r(B, 1)).expand(B, N).clone()
        dpsi = torch.zeros(B, N, dtype=D)
        for _ in range(2):
            j = torch.randint(2, N - 2, (B,), generator=gen)
            a = torch.deg2rad(5 + 55 * r(B)) * torch.where(r(B) < 0.5, -1.0, 1.0)
            dpsi[torch.arange(B), j] = a
        return paths_from(v, torch.zeros(B, dtype=D), dpsi)
    if name == "F3_stopgo":
        v = (15 * r(B, N)) * (r(B, N) > 0.3)
        v[:, 20:30] = 0.0
        return paths_from(v, torch.zeros(B, dtype=D), (r(B, N) - 0.5) * 0.05)
    if name == "F4_straight":
        v = (1 + 39 * r(B, 1)).expand(B, N).clone()
        return paths_from(v, torch.zeros(B, dtype=D), torch.zeros(B, N, dtype=D))
    if name == "F5_first_turn":
        v = (5 + 20 * r(B, 1)).expand(B, N).clone()
        return paths_from(v, torch.deg2rad(10 + 50 * r(B)), torch.zeros(B, N, dtype=D))
    raise KeyError(name)


def kin(traj):
    Bn = traj.shape[0]
    P = torch.cat([traj.new_zeros(Bn, 1, 2), traj], 1)
    seg = P[:, 1:] - P[:, :-1]
    L = torch.sqrt((seg ** 2).sum(-1) + 1e-12)
    ang = torch.atan2(seg[..., 1], seg[..., 0])
    dpsi = torch.zeros_like(ang)
    dpsi[:, 1:] = M.wrap(ang[:, 1:] - ang[:, :-1])
    kappa = dpsi.abs() / L.clamp_min(1e-3)
    kappa = torch.where(L > 1e-3, kappa, torch.zeros_like(kappa))
    return P, seg, L, L / DT, kappa


def dist_to_polyline(pts, P):
    """pts [B,N,2], P [B,N+1,2] -> min distance of each point to the polyline."""
    a, b = P[:, None, :-1], P[:, None, 1:]
    p = pts[:, :, None]
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab ** 2).sum(-1).clamp_min(1e-18)).clamp(0, 1)
    return (((a + t[..., None] * ab) - p) ** 2).sum(-1).sqrt().min(-1).values


def arc_pos(pts, P, L):
    a, b = P[:, None, :-1], P[:, None, 1:]
    p = pts[:, :, None]
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab ** 2).sum(-1).clamp_min(1e-18)).clamp(0, 1)
    d = (((a + t[..., None] * ab) - p) ** 2).sum(-1)
    S0 = torch.cat([L.new_zeros(L.shape[0], 1), torch.cumsum(L, 1)], 1)[:, :-1]
    s = S0[:, None] + t * L[:, None]
    # among segments within 1e-9 of the best distance take the largest arc position that keeps monotonicity checkable
    best = d.min(-1, keepdim=True).values
    return torch.where(d <= best + 1e-12, s, torch.full_like(s, -1.0)).max(-1).values


def worst(viol, traj, cap, extra=None):
    """viol [B] >= 0: amount of violation per path."""
    i = int(torch.argmax(viol))
    ce = None
    if float(viol[i]) > 0:
        _, _, L, v, k = kin(traj[i:i + 1])
        ce = {"index": i, "cap_ms": None if cap is None else float(cap[i]), "v_in_first8": [round(float(x), 4) for x in v[0, :8]],
              "kappa_in_max": float(k.max()), "v_in_max": float(v.max())}
        if extra is not None:
            ce.update({k_: (float(v_[i]) if torch.is_tensor(v_) else v_) for k_, v_ in extra.items()})
    return {"pass_share": float((viol <= 0).double().mean()), "worst": float(viol[i]), "counterexample": ce}


def check(fn, name, with_cap=True):
    traj = fam(name)
    cap = (5 + 20 * torch.rand(B, generator=gen, dtype=D)) if with_cap else None
    out = fn(traj, cap, MU, DT, G)
    P, seg, L, v, kap = kin(traj)
    Po, sego, Lo, vo, kapo = kin(out)
    v_kamm = torch.sqrt(MU * G / kap.clamp_min(1e-6))
    vcap = v_kamm if cap is None else torch.minimum(v_kamm, cap[:, None])
    v_new = torch.minimum(v, vcap)
    res = {}
    if cap is not None:
        res["C-CAP"] = worst((vo - cap[:, None] - 1e-6).clamp_min(0).max(1).values, traj, cap)
    res["C-GEOM"] = worst((dist_to_polyline(out, P) - 1e-6).clamp_min(0).max(1).values, traj, cap)
    s = arc_pos(out, P, L)
    res["C-MONO"] = worst((s[:, :-1] - s[:, 1:] - 1e-6).clamp_min(0).max(1).values, traj, cap)
    ok_in = (v <= vcap + 1e-12).all(1)
    dn = ((out - traj).abs().amax((1, 2)) - 1e-6).clamp_min(0)
    res["C-NOOP"] = worst(torch.where(ok_in, dn, torch.zeros_like(dn)), traj, cap, {"n_applicable": int(ok_in.sum())})
    res["C-NOOP"]["n_applicable"] = int(ok_in.sum())
    res["C-KAMM-IN"] = worst(((v_new ** 2 * kap) / (MU * G) - (1 + 1e-9)).clamp_min(0).max(1).values, traj, cap)
    ratio_out = (vo ** 2 * kapo) / (MU * G)
    res["C-KAMM-OUT"] = worst((ratio_out - (1 + 1e-3)).clamp_min(0).max(1).values, traj, cap, {"ratio_out_max": ratio_out.max(1).values})
    res["C-KAMM-OUT"]["median_of_violating_ratio"] = float(ratio_out.max(1).values[ratio_out.max(1).values > 1.001].median()) if (ratio_out.max(1).values > 1.001).any() else None
    out2 = fn(out, cap, MU, DT, G)
    res["C-IDEM"] = worst(((out2 - out).abs().amax((1, 2)) - 1e-6).clamp_min(0), traj, cap)
    return res


def first_turn():
    """C-FIRST: a plan turning at step 0 by angle a, against the same plan turning by a at step 5; equal speed."""
    traj0 = fam("F5_first_turn")
    _, _, L0, v0, _ = kin(traj0)
    ang0 = torch.atan2(traj0[:, 0, 1], traj0[:, 0, 0])
    dpsi = torch.zeros(B, N, dtype=D); dpsi[:, 5] = ang0
    traj5 = paths_from(v0, torch.zeros(B, dtype=D), dpsi)
    o0, o5 = M.project_speed(traj0, None, MU, DT, G), M.project_speed(traj5, None, MU, DT, G)
    slowed0 = ((traj0 - o0).abs().amax((1, 2)) > 1e-6)
    slowed5 = ((traj5 - o5).abs().amax((1, 2)) > 1e-6)
    viol = (slowed5 & ~slowed0).double()
    return {"pass_share": float(1 - viol.mean()), "share_slowed_when_turn_at_step5": float(slowed5.double().mean()), "share_slowed_when_turn_at_step0": float(slowed0.double().mean()),
            "counterexample": None if viol.sum() == 0 else {"index": int(torch.argmax(viol)), "angle_deg": float(torch.rad2deg(ang0[int(torch.argmax(viol))])), "speed_ms": float(v0[int(torch.argmax(viol)), 0])}}


def mutated(traj, cap_ms, mu=0.7, dt=0.1, g=9.81):
    """project_speed with the cap line removed (mutation control): the cap is ignored."""
    return M.project_speed(traj, None, mu, dt, g)


def solver():
    from z3 import Reals, Solver, Not, And, unsat, sat
    r = {}
    s1, s2, v, cap, dt = Reals("s1 s2 v cap dt")
    S = Solver(); S.add(dt > 0, v >= 0, v <= cap, s2 - s1 == v * dt, Not(s2 - s1 <= cap * dt)); r["VC-K1"] = "PROVED" if S.check() == unsat else "NOT-PROVED"
    ax, ay, bx, by, cx, cy, x, y = Reals("ax ay bx by cx cy x y")
    S = Solver(); S.set("timeout", 120000)
    S.add(x >= 0, y >= 0, (ax - bx) ** 2 + (ay - by) ** 2 <= x * x, (bx - cx) ** 2 + (by - cy) ** 2 <= y * y, Not((ax - cx) ** 2 + (ay - cy) ** 2 <= (x + y) ** 2))
    c = S.check(); r["VC-K2"] = "PROVED" if c == unsat else ("COUNTEREXAMPLE" if c == sat else "UNKNOWN")
    S = Solver(); S.add(x >= 0, y >= 0, (ax - bx) ** 2 + (ay - by) ** 2 <= x * x, (bx - cx) ** 2 + (by - cy) ** 2 <= y * y, Not((ax - cx) ** 2 + (ay - cy) ** 2 <= x * x + y * y))
    r["M-K2 (drop the cross term)"] = "REFUTED" if S.check() == sat else "NOT-REFUTED"
    return r


res = {"spec_sha256": hashlib.sha256(open(SPEC, "rb").read()).hexdigest(), "code_file": inspect.getsourcefile(M).replace("\\", "/"),
       "code_sha256": hashlib.sha256(open(inspect.getsourcefile(M), "rb").read()).hexdigest(), "n_per_family": B, "families": {}}
for f in ("F1_smooth", "F2_vertex", "F3_stopgo", "F4_straight"):
    res["families"][f] = check(M.project_speed, f)
res["families"]["F2_vertex_no_cap"] = check(M.project_speed, "F2_vertex", with_cap=False)
res["C-FIRST"] = first_turn()
res["mutation_cap_removed"] = {"C-CAP": check(mutated, "F4_straight")["C-CAP"]["pass_share"]}
try:
    res["solver"] = solver()
except Exception as e:  # z3 may be absent on this box; the solver part is then run on the pod
    res["solver"] = {"error": type(e).__name__}
json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
for f, r in res["families"].items():
    print(f, {k: (round(v["pass_share"], 4), round(v["worst"], 4)) for k, v in r.items()})
print("C-FIRST", {k: v for k, v in res["C-FIRST"].items() if k != "counterexample"})
print("mutation", res["mutation_cap_removed"], "solver", res["solver"])
