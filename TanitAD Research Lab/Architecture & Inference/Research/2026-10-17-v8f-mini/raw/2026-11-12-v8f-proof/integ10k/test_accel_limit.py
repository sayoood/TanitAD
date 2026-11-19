"""test_accel_limit.py -- pure tests of accel_limit of v8f_t1_eval_reader_rule_a.py (card PREREG_V8F_PROOF_ADD53), no model, no GPU. Run on the pod:
  DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python test_accel_limit.py [DUMP_DIR]
Each case has an arm that must fail (a planted wrong limit, a wrong start speed, a wrong path). With DUMP_DIR (a banked dump of the no-rule run) the function is applied to
the banked first-tick plans and only instrument quantities are printed (how far a chord speed can lie from the arc speed the limit acts on); no rate, no error against the log.
Last line: ZZ TEST-ACCEL-LIMIT PASS n/n or FAIL."""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v8f_t1_eval_reader_rule_a as A  # noqa: E402

DT = 0.2
N = 30
RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def straight(speeds):
    """A plan on the x axis with the given step speeds [N]."""
    x = np.cumsum(np.asarray(speeds, np.float64) * DT)
    return np.stack([x, np.zeros_like(x)], -1)


def arc_speeds(p):
    q = np.concatenate([np.zeros((1, 2)), np.asarray(p, np.float64)], 0)
    return np.linalg.norm(np.diff(q, axis=0), axis=-1) / DT


def expected(v_plan, v0, a_max, b_lim):
    """The rate limit written out step by step: v'(i) = clip(v_plan(i), max(0, v'(i-1) - b dt), v'(i-1) + a dt)."""
    out, v = [], float(v0)
    for w in v_plan:
        v = min(max(float(w), max(0.0, v - b_lim * DT)), v + a_max * DT)
        out.append(v)
    return np.asarray(out)


def dist_to_polyline(pts, poly):
    P = np.concatenate([np.zeros((1, 2)), poly], 0)
    a, b = P[:-1], P[1:]
    ab = b - a
    den = np.maximum((ab ** 2).sum(-1), 1e-18)
    best = np.full(len(pts), np.inf)
    for i in range(len(a)):
        t = np.clip(((pts - a[i]) * ab[i]).sum(-1) / den[i], 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(pts - (a[i] + t[:, None] * ab[i]), axis=-1))
    return best


# 1 limits out of reach: the very same object comes back
p = np.stack([straight(np.linspace(8, 20, N)), straight(np.full(N, 3.0))]).astype(np.float32)
out, info = A.accel_limit(p, np.array([5.0, 9.0]), 1e9, 1e9, 10)
check("1 out-of-reach limits return the input object", out is p and int(info["n_up"].sum()) == 0 and int(info["n_dn"].sum()) == 0)
out2, _ = A.accel_limit(p, np.array([5.0, 9.0]), 3.0, 8.0, 10)
check("1x planted arm: real limits do change this plan", out2 is not p and bool((out2 != p).any()))

# 2 a constant-speed plan at the ego speed is left alone
p = straight(np.full(N, 12.0))[None].astype(np.float32)
out, info = A.accel_limit(p, np.array([12.0]), 3.0, 8.0, 5)
check("2 constant speed at v0 is unchanged", out is p)

# 3 a plan that wants +5 m/s at once: 0.6 m/s per step, then the plan's own speed
vp = np.full(N, 15.0)
p = straight(vp)[None]
out, info = A.accel_limit(p, np.array([10.0]), 3.0, 8.0, 5)
got, exp = arc_speeds(out[0]), expected(vp, 10.0, 3.0, 8.0)
check("3 acceleration limit step by step", np.allclose(got, exp, atol=1e-9) and int(info["n_up"][0]) == 5 and abs(float(info["req"][0]) - 25.0) < 1e-9,
      "first speeds %s n_up %d req %.2f" % (np.round(got[:10], 3).tolist(), int(info["n_up"][0]), float(info["req"][0])))
check("3x planted arm: twice the limit gives other speeds", not np.allclose(got, expected(vp, 10.0, 6.0, 8.0), atol=1e-6))
check("3y planted arm: a start speed off by 1 m/s gives other speeds", not np.allclose(got, expected(vp, 11.0, 3.0, 8.0), atol=1e-6))

# 4 a plan that wants -8 m/s at once: 1.6 m/s per step
vp = np.full(N, 2.0)
p = straight(vp)[None]
out, info = A.accel_limit(p, np.array([10.0]), 3.0, 8.0, 5)
got, exp = arc_speeds(out[0]), expected(vp, 10.0, 3.0, 8.0)
check("4 braking limit step by step", np.allclose(got, exp, atol=1e-9) and int(info["n_dn"][0]) == 5 and int(info["n_up"][0]) == 0, "first speeds %s" % np.round(got[:8], 3).tolist())

# 5 a plan that stands still from the ego speed 10: braked along the extension ray, finite
p = np.zeros((1, N, 2))
out, info = A.accel_limit(p, np.array([10.0]), 3.0, 8.0, 5)
got, exp = arc_speeds(out[0]), expected(np.zeros(N), 10.0, 3.0, 8.0)
check("5 standing plan is braked to 0, finite", bool(np.isfinite(out).all()) and np.allclose(got, exp, atol=1e-9) and got[-1] == 0.0, "first speeds %s" % np.round(got[:8], 3).tolist())

# 6 a curve (radius 30 m, 12 m/s) entered at 8 m/s: points stay on the plan's own path, arc speeds obey the limit
th = 12.0 * DT * np.arange(1, N + 1) / 30.0
p = np.stack([30.0 * np.sin(th), 30.0 * (1.0 - np.cos(th))], -1)[None]
out, info = A.accel_limit(p, np.array([8.0]), 3.0, 8.0, 5)
d = dist_to_polyline(out[0], p[0])
P, L, ext = A.R.plan_path(p[0])
a_exp = np.cumsum(expected(np.diff(L) / DT, 8.0, 3.0, 8.0) * DT)
exp_pts = A.R.sample_path(P, L, ext, a_exp)
check("6 curve: output lies on the plan's own path at the expected arc lengths", float(d.max()) < 1e-9 and np.allclose(out[0], exp_pts, atol=1e-9), "max distance to the path %.2e" % float(d.max()))
check("6x planted arm: the unlimited plan is not at those arc lengths", not np.allclose(p[0], exp_pts, atol=1e-3))

# 7 a row without a finite ego speed is left alone, the other rows are limited
p = np.stack([straight(np.full(N, 15.0)), straight(np.full(N, 15.0))])
out, info = A.accel_limit(p, np.array([np.nan, 10.0]), 3.0, 8.0, 5)
check("7 non-finite v0 row unchanged, other row limited", np.array_equal(out[0], p[0]) and not np.array_equal(out[1], p[1]) and int(info["n_up"][0]) == 0)

# 8 dtype kept; an untouched row is bit-identical; a step that is not clipped keeps the original knot
p = np.stack([straight(np.full(N, 12.0)), straight(np.concatenate([np.full(3, 14.0), np.full(N - 3, 10.0)]))]).astype(np.float32)
out, info = A.accel_limit(p, np.array([12.0, 10.0]), 3.0, 8.0, 5)
check("8 dtype float32 kept, untouched row bit-identical", out.dtype == np.float32 and np.array_equal(out[0], p[0]) and bool(info["changed"][1]) and not bool(info["changed"][0]))

# 9 executed-step counters stop at m
vp = np.full(N, 30.0)
out, info = A.accel_limit(straight(vp)[None], np.array([10.0]), 3.0, 8.0, 10)
check("9 n_up counts executed steps only", int(info["n_up"][0]) == 10, "n_up %d" % int(info["n_up"][0]))

# 10 the limited plan is a fixed point: limiting it again from the same start speed changes nothing beyond rounding
vp = np.concatenate([np.full(10, 16.0), np.full(10, 4.0), np.full(10, 12.0)])
p = straight(vp)[None]
o1, _ = A.accel_limit(p, np.array([10.0]), 3.0, 8.0, 5)
o2, _ = A.accel_limit(o1, np.array([10.0]), 3.0, 8.0, 5)
check("10 limiting twice equals limiting once", np.allclose(o1, o2, atol=1e-9), "max diff %.2e" % float(np.abs(np.asarray(o1) - np.asarray(o2)).max()))

if len(sys.argv) > 1:
    files = sorted(glob.glob(os.path.join(sys.argv[1], "ep*.npz")))
    ol = np.concatenate([np.load(f)["ol"] for f in files])
    v0 = np.concatenate([np.load(f)["v0"] for f in files]).astype(np.float64)
    for a_max in (3.0, 1.5):
        out, info = A.accel_limit(ol, v0, a_max, 8.0, 5)
        o64 = np.asarray(out, np.float64)
        q = np.concatenate([np.zeros((len(o64), 1, 2)), o64], 1)
        chord = np.linalg.norm(np.diff(q, axis=1), axis=-1) / DT
        dv = np.diff(np.concatenate([v0[:, None], chord], 1), axis=1) / DT
        arc = np.diff(np.concatenate([np.zeros((len(o64), 1)), info["_a_new"]], 1), axis=1) / DT          # speeds along the plan's own path: what the limit acts on
        da = np.diff(np.concatenate([v0[:, None], arc], 1), axis=1) / DT
        far = np.abs(chord - arc) > 0.05
        print("banked first-tick plans n %d a_max %.1f: changed %.4f | arc acceleration max %.4f min %.4f | chord acceleration max %.4f min %.4f | share of steps with chord acceleration above a_max + 0.25: %.5f, below -(b_lim + 0.25): %.5f | steps with |chord - arc| speed above 0.05 m/s: share %.5f in %d plans, max %.4f m/s | dtype %s" % (
            len(ol), a_max, float(info["changed"].mean()), float(da.max()), float(da.min()), float(dv.max()), float(dv.min()), float((dv > a_max + 0.25).mean()), float((dv < -8.25).mean()),
            float(far.mean()), int(far.any(1).sum()), float(np.abs(chord - arc).max()), out.dtype), flush=True)
        check("11 banked plans a_max %.1f: arc acceleration within the limits (1e-6)" % a_max, float(da.max()) <= a_max + 1e-6 and float(da.min()) >= -8.0 - 1e-6)
        check("12 banked plans a_max %.1f: logged executed speeds equal the first 5 arc speeds" % a_max, np.array_equal(info["v_exec"], arc[:, :5]))

print("ZZ TEST-ACCEL-LIMIT %s %d/%d" % ("PASS" if all(RES) else "FAIL", sum(RES), len(RES)))
sys.exit(0 if all(RES) else 1)
