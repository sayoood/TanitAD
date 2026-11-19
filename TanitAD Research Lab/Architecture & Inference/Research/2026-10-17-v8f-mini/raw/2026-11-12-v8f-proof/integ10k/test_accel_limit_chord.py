"""test_accel_limit_chord.py -- pure tests of accel_limit_chord of v8f_t1_eval_reader_rule_ac.py (card PREREG_V8F_PROOF_ADD56), no model, no GPU. Run on the pod:
  DISTRULE_DIR=<build> PYTHONPATH=<tree>/stack python test_accel_limit_chord.py DUMP_DIR
DUMP_DIR = a banked dump of card ADD52 (the no-rule run at 5 steps). Its executed tracks are bent and serve as path material only: every quantity printed is a property of
the limiter (chord speeds of its output against the limit); no rate, no error against the log, nothing of card ADD53's runs. Each check has an arm that must fail.
Last line: ZZ TEST-ACCEL-LIMIT-CHORD PASS n/n or FAIL."""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v8f_t1_eval_reader_rule_a as A  # noqa: E402
import v8f_t1_eval_reader_rule_ac as AC  # noqa: E402

DT = 0.2
N = 30
RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print("%s %s %s" % ("PASS" if ok else "FAIL", name, detail), flush=True)


def straight(speeds):
    x = np.cumsum(np.asarray(speeds, np.float64) * DT)
    return np.stack([x, np.zeros_like(x)], -1)


def chord_speeds(p):
    q = np.concatenate([np.zeros((len(p), 1, 2)), np.asarray(p, np.float64)], 1)
    return np.linalg.norm(np.diff(q, axis=1), axis=-1) / DT                     # [B, N]


def excess(p, v0, a_max, b_lim):
    """largest amount (m/s) by which a chord speed of the plan leaves [max(0, v_prev - b dt), v_prev + a dt], v_prev(0) = v0, over all N steps."""
    s = chord_speeds(p)
    prev = np.concatenate([np.asarray(v0, np.float64)[:, None], s[:, :-1]], 1)
    return np.maximum(s - (prev + a_max * DT), np.maximum(0.0, prev - b_lim * DT) - s)


def block_share(cl, m, a_max, b_lim=8.0, tol=0.25):
    """the definition of control f5 of card ADD53 (reader), copied: inside each block of m steps the share of steps whose chord acceleration leaves the limit by more than tol."""
    s = chord_speeds(cl)
    acc = np.diff(s, axis=1) / DT
    inside = np.array([(i + 1) % m != 0 for i in range(acc.shape[1])])
    x = acc[:, inside]
    return float(((x > a_max + tol) | (x < -(b_lim + tol))).sum() / max(x.size, 1))


def dist_to_path(pts, plan):
    """largest distance of the points [N,2] from the polyline origin -> plan knots (and its straight extension)."""
    P = np.vstack([np.zeros((1, 2)), plan])
    seg = np.diff(P, axis=0)
    L2 = np.maximum((seg ** 2).sum(-1), 1e-30)
    worst = 0.0
    for q in pts:
        t = np.clip(((q - P[:-1]) * seg).sum(-1) / L2, 0.0, 1.0)
        t[-1] = max(((q - P[-2]) * seg[-1]).sum() / L2[-1], 0.0)             # the last segment goes on as a ray
        d = np.linalg.norm(P[:-1] + t[:, None] * seg - q, axis=-1)
        worst = max(worst, float(d.min()))
    return worst


rng = np.random.default_rng(56)

# 1 straight plans: the chord form is the path form
sp = np.clip(8.0 + np.cumsum(rng.normal(0.0, 1.2, (40, N)), 1), 0.0, None)
pl = np.stack([straight(s) for s in sp])
v0 = sp[:, 0] + rng.normal(0.0, 1.0, 40)
v0 = np.clip(v0, 0.0, None)
o_p, i_p = A.accel_limit(pl, v0, 3.0, 8.0, 5)
o_c, i_c = AC.accel_limit_chord(pl, v0, 3.0, 8.0, 5)
d1 = float(np.abs(o_p - o_c).max())
check("1 straight plans: chord form equals path form", d1 <= 1e-9 and np.array_equal(i_p["n_up"], i_c["n_up"]) and np.array_equal(i_p["n_dn"], i_c["n_dn"]) and int(i_p["n_up"].sum()) > 0
      and int(i_p["n_dn"].sum()) > 0, "largest knot difference %.2e m; clipped steps up %d down %d" % (d1, int(i_c["n_up"].sum()), int(i_c["n_dn"].sum())))
bad = pl.copy()
bad[:, :, 1] += 0.3 * np.sin(np.arange(N))[None]                                # arm that must fail: on a bent path the two forms differ
check("1b arm: on bent plans the two forms differ", float(np.abs(A.accel_limit(bad, v0, 3.0, 8.0, 5)[0] - AC.accel_limit_chord(bad, v0, 3.0, 8.0, 5)[0]).max()) > 1e-3)

# 2 limits out of reach: the input object comes back
o, info = AC.accel_limit_chord(pl, v0, 1e9, 1e9, 5)
check("2 limits out of reach return the input object", o is pl and not info["changed"].any() and int(info["n_up"].sum() + info["n_dn"].sum()) == 0)

# 3 unclipped steps keep the knot bit for bit (float32 plan with one late jump)
s3 = np.full(N, 10.0)
s3[20:] = 14.0
p3 = straight(s3)[None].astype(np.float32)
o3, i3 = AC.accel_limit_chord(p3, np.array([10.0]), 3.0, 8.0, 5)
check("3 steps before the first clip are the original knots bit for bit", o3.dtype == np.float32 and np.array_equal(o3[0, :20], p3[0, :20]) and not np.array_equal(o3[0, 20:], p3[0, 20:]) and int(i3["n_up"][0]) == 0,
      "(the jump lies after the executed steps: n_up over the executed steps %d)" % int(i3["n_up"][0]))

# 4 the batched path sampler equals dist_rule.sample_path
PP, aP, EXT = np.zeros((40, N + 1, 2)), np.zeros((40, N + 1)), np.zeros((40, 2))
for b in range(40):
    PP[b], aP[b], EXT[b] = AC.R.plan_path(bad[b])
arcs = rng.uniform(0.0, 1.3, 40) * aP[:, -1]
ref = np.stack([AC.R.sample_path(PP[b], aP[b], EXT[b], [arcs[b]])[0] for b in range(40)])
d4 = float(np.abs(AC._sample_rows(PP, aP, EXT, arcs) - ref).max())
d4k = float(np.abs(AC._sample_rows(PP, aP, EXT, aP[:, 7]) - PP[:, 7]).max())
check("4 batched sampler equals dist_rule.sample_path (and returns a knot at its own arc length)", d4 <= 1e-12 and d4k == 0.0, "largest difference %.2e m; at a knot %.1e" % (d4, d4k))

# 5 to 9 on bent path material (executed tracks of the banked no-rule run at 5 steps, used as plans)
files = sorted(glob.glob(os.path.join(sys.argv[1], "ep*.npz")))
assert len(files) == 141, len(files)
cl = np.concatenate([np.asarray(np.load(f)["cl"], np.float64) for f in files])
cl = cl[np.isfinite(cl).all(axis=(1, 2))][::3]
vb = chord_speeds(cl)[:, 0]
print("path material: %d bent tracks of 30 knots" % len(cl))
for a_max, m in ((3.0, 5), (3.0, 10), (1.5, 5)):
    oc, ic = AC.accel_limit_chord(cl, vb, a_max, 8.0, m)
    op, ip = A.accel_limit(cl, vb, a_max, 8.0, m)
    nofail = ic["n_fail"] == 0
    e_c = excess(oc, vb, a_max, 8.0)
    e_p = excess(op, vb, a_max, 8.0)
    tag = "a_max %.1f m %d" % (a_max, m)
    check("5 chord form holds the limit on the chord speeds of every step (%s)" % tag, float(e_c[nofail].max()) <= 1e-6 and int((~nofail).sum()) <= 0.01 * len(cl),
          "largest excess %.2e m/s; tracks with a lower bound not reached %d of %d; clipped up %d down %d" % (float(e_c[nofail].max()), int((~nofail).sum()), len(cl), int(ic["n_up"].sum()), int(ic["n_dn"].sum())))
    check("5b arm: the path form leaves the limit on the chord speeds of the same tracks (%s)" % tag, float(e_p.max()) > 0.05, "largest excess %.3f m/s" % float(e_p.max()))
    sh_c, sh_p = block_share(oc, m, a_max), block_share(op, m, a_max)
    check("6 the f5 quantity of the reader is 0 on the chord form and above 0 on the path form (%s)" % tag, sh_c <= 1e-4 and sh_p > sh_c, "chord form %.5f, path form %.5f" % (sh_c, sh_p))
    check("7 the log speeds are the chord speeds of the output (%s)" % tag, float(np.abs(ic["v_exec"] - chord_speeds(oc)[:, :m]).max()) <= 1e-9)
oc, ic = AC.accel_limit_chord(cl, vb, 3.0, 8.0, 5)
check("5c arm: a wrong limit is seen (output of 3.0 checked against 2.0)", float(excess(oc, vb, 2.0, 8.0).max()) > 0.05)
idx = np.nonzero(ic["changed"])[0][:60]
d8 = max(dist_to_path(oc[b], cl[b]) for b in idx)
check("8 every knot of the limited plan lies on the plan's own path", len(idx) > 0 and d8 <= 1e-9, "largest distance %.2e m over %d changed plans" % (d8, len(idx)))
check("8b arm: a knot moved off the path is seen", dist_to_path(oc[idx[0]] + np.array([0.0, 0.05]), cl[idx[0]]) > 1e-3)
check("9 the arc positions never run backwards", bool((np.diff(np.concatenate([np.zeros((len(cl), 1)), ic["_a_new"]], 1), axis=1) >= -1e-12).all()))
o32, i32 = AC.accel_limit_chord(cl.astype(np.float32), vb, 3.0, 8.0, 5)
e32 = excess(o32, vb, 3.0, 8.0)[i32["n_fail"] == 0]
check("10 float32 plans: the limit holds on the stored knots to 1e-3 m/s", o32.dtype == np.float32 and float(e32.max()) <= 1e-3, "largest excess %.2e m/s" % float(e32.max()))

# 11 rows without a finite start speed are not touched
vn = vb.copy()
vn[::2] = np.nan
on, inn = AC.accel_limit_chord(cl, vn, 3.0, 8.0, 5)
check("11 rows without a finite start speed keep their plan", np.array_equal(on[::2], cl[::2]) and not inn["changed"][::2].any() and inn["changed"][1::2].any())

# 12 the default form of the wrapper is the function of card ADD53, unchanged
o_a, i_a = A.accel_limit(cl, vb, 3.0, 8.0, 5)
o_b, i_b = AC.accel_limit(cl, vb, 3.0, 8.0, 5)
check("12 default form path: the wrapper's accel_limit is that of card ADD53 bit for bit", AC.STATE["aform"] == "path" and np.array_equal(o_a, o_b) and all(np.array_equal(i_a[k], i_b[k]) for k in i_a))
print("ZZ TEST-ACCEL-LIMIT-CHORD %s %d/%d" % ("PASS" if all(RES) else "FAIL", sum(RES), len(RES)))
