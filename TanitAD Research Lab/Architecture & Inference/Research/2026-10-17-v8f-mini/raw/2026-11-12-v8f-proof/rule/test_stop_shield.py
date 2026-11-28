"""Property tests of stop_shield.py (PREREG_V8F_PROOF_ADD3).  CPU, numpy only.  Run: python test_stop_shield.py  (or pytest)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stop_shield as SH  # noqa: E402

N = 20000
DT, Q = 0.2, 0.6
TOL = 1e-9


def _cases(seed=0, n=N):
    r = np.random.default_rng(seed)
    v = np.where(r.random(n) < 0.1, 0.0, r.uniform(0, 40, n))
    slack = np.where(r.random(n) < 0.3, 0.0, r.exponential(5.0, n))          # d = B(v) + slack: the invariant holds on entry, often exactly tight
    d = SH.brake_dist(v, Q, DT) + slack
    kind = r.integers(0, 5, n)
    u = np.where(kind == 0, r.uniform(-50, 50, n), np.where(kind == 1, r.uniform(-1e6, 1e6, n), np.where(kind == 2, v + r.normal(0, 1, n), np.where(kind == 3, np.maximum(v - Q, 0) + r.normal(0, 0.2, n), r.uniform(0, 60, n)))))
    return v, u, d


def test_closed_form_equals_recursion():
    r = np.random.default_rng(1)
    v = np.concatenate([r.uniform(0, 60, N), np.arange(0, 120) * Q, np.arange(0, 120) * Q + 1e-3, np.arange(1, 120) * Q - 1e-3])
    cf = SH.brake_dist(v, Q, DT)
    rec = np.array([SH.brake_dist_rec(x, Q, DT) for x in v])
    assert np.max(np.abs(cf - rec)) < 1e-9, np.max(np.abs(cf - rec))
    assert float(SH.brake_dist(0.0, Q, DT)) == 0.0 and float(SH.brake_dist(-5.0, Q, DT)) == 0.0
    # VC-C2 recursion identity at n >= 1, n*q <= v
    for x in r.uniform(1, 40, 200):
        k = int(x // Q)
        for n in range(1, min(k, 8) + 1):
            f = lambda vv, nn: DT * ((nn + 1) * vv - Q * nn * (nn + 1) / 2)
            assert abs(f(x, n) - (x * DT + f(x - Q, n - 1))) < 1e-9


def test_vc_c1_conclusions_every_step():
    v, u, d = _cases()
    bad = 0
    for vi, ui, di in zip(v, u, d):
        vn, d2 = SH.shield_step(vi, ui, di, Q, DT)
        vb = max(vi - Q, 0.0)
        ok = (float(SH.brake_dist(vn, Q, DT)) <= d2 + TOL) and (d2 >= -TOL) and (vn >= vb - TOL) and (vn >= 0)
        bad += (not ok)
    assert bad == 0, bad


def test_minimal_intervention():
    v, u, d = _cases(seed=2)
    n_resp = 0
    for vi, ui, di in zip(v, u, d):
        vb, d2 = max(vi - Q, 0.0), di - vi * DT
        if ui >= vb and float(SH.brake_dist(ui, Q, DT)) <= d2:
            n_resp += 1
            vn, _ = SH.shield_step(vi, ui, di, Q, DT)
            assert vn == ui
    assert n_resp > 1000, n_resp
    # whole-profile form: a profile that respects the envelope at every step comes back unchanged, and so does its re-timed plan
    P = np.cumsum(np.column_stack([np.full(30, 1.0), np.zeros(30)]), axis=0)          # constant 5 m/s along x
    out = SH.shield_plan(P, 1e6)
    assert np.array_equal(out["v"], SH.seg_speeds(P, DT)) and np.max(np.abs(out["plan"] - P)) < 1e-9


def test_chain_invariant_many_steps():
    r = np.random.default_rng(3)
    for _ in range(4000):
        n = 30
        s = np.where(r.random(n) < 0.2, r.uniform(-30, 80, n), r.uniform(0, 15, n))
        v0 = abs(s[0])
        d0 = float(SH.brake_dist(v0, Q, DT)) + r.exponential(10.0)
        v, feas, dl = SH.shield_speeds(s, d0, Q, DT)
        assert feas
        d = d0
        for k in range(n):
            assert float(SH.brake_dist(v[k], Q, DT)) <= d + 1e-8, (k,)
            d -= v[k] * DT
            assert d >= -1e-8
        assert abs(d - dl) < 1e-8 and dl >= -1e-8
        assert np.all(np.diff(v) >= -Q - 1e-12)                                             # never brakes harder than q per step


def test_mutation_breaks_the_invariant():
    v, u, d = _cases(seed=4)
    viol = 0
    for vi, ui, di in zip(v, u, d):
        vn, d2 = SH.shield_step(vi, ui, di, Q, DT, fallback=False)
        viol += float(SH.brake_dist(vn, Q, DT)) > d2 + 1e-6
    assert viol > 0, viol
    return viol


def test_infeasible_start_brakes():
    s = np.full(30, 10.0)
    v, feas, _ = SH.shield_speeds(s, 1.0, Q, DT)
    assert not feas and np.allclose(v[1:], np.maximum(10.0 - Q * np.arange(1, 30), 0.0))


def test_retime_geometry():
    r = np.random.default_rng(5)
    for _ in range(200):
        th = np.cumsum(r.normal(0, 0.05, 30))
        sp = r.uniform(0, 1.6, 30)
        P = np.cumsum(np.column_stack([sp * np.cos(th), sp * np.sin(th)]), axis=0)
        v = r.uniform(0, 12, 30)
        out = SH.retime(P, v, DT)
        Pp, L, _, ext = SH.path_of(P)
        assert np.max(SH.dist_to_path(Pp, L, ext, out)) < 1e-6
        same = SH.retime(P, SH.seg_speeds(P, DT), DT)
        assert np.max(np.abs(same - P)) < 1e-9


if __name__ == "__main__":
    fs = [k for k in sorted(globals()) if k.startswith("test_")]
    for k in fs:
        res = globals()[k]()
        print("PASS", k, "" if res is None else f"(mutation violations on {res} of {N} cases)")
    print(f"ALL {len(fs)} PASS, {N} random cases per property")
