"""test_zt3.py -- self-tests AND mutation checks for every new ZT3 metric (PREREG_ZT3 section 3/4).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_zt3.py            # unit + mutation (fast)
    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe test_zt3.py --pipeline # + geometry-pipeline zero controls (NAVSIM venv, ~30 s)

Pattern: each metric has a checker `chk_*(fn) -> bool`. The REAL implementation must pass; a deliberate
MUTANT (one-line regression) must FAIL. A checker that passes the mutant proves nothing, so that is a failure too.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zt3_lib as L  # noqa: E402

ZERO_TAIL_SHA256 = "149c3e97927ed219323f8f070b67a36d9172a575e4ec23162241c417f8501a3d"   # git blob f0c9bc10 @ origin 605711b
RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name)


def real_vs_mutant(name, chk, real, mutant):
    check(f"{name}: real passes", chk(real))
    check(f"{name}: mutant is caught (RED)", not chk(mutant))


# ------------------------------------------------------------------ checkers
def chk_nondrivable(fn):
    ok = True
    a = np.zeros((1, 2, 4), bool)
    a[0, 0, :] = True                                        # all 4 corners in polygon 0
    ok &= not bool(fn(a))
    b = np.zeros((1, 2, 4), bool)
    b[0, 0, :2] = True
    b[0, 1, 2:] = True                                       # 2 corners in poly 0, 2 in poly 1 -> still drivable
    ok &= not bool(fn(b))
    c = np.zeros((1, 2, 4), bool)
    c[0, 0, :3] = True                                       # 3 of 4 -> non-drivable
    ok &= bool(fn(c))
    ok &= bool(fn(np.zeros((1, 0, 4), bool)))                # no polygons -> non-drivable
    return ok


def mut_nondrivable(ci):                                     # "<= 4": everything non-drivable
    return (np.asarray(ci).sum(axis=-2) > 0).sum(axis=-1) <= 4


def chk_first_true(fn):
    return fn(np.array([0, 0, 1, 1], bool)) == 2 and fn(np.zeros(5, bool)) == -1 and fn(np.array([1, 0], bool)) == 0


def mut_first_true(flags):
    idx = np.flatnonzero(flags)
    return int(idx[-1]) if idx.size else -1


def chk_signed_offset(fn):
    line = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    ok = abs(fn([5.0, 2.0], line)[0] - 2.0) < 1e-9 and abs(fn([5.0, -3.0], line)[0] + 3.0) < 1e-9
    e, s = fn([30.0, 1.0], line)                             # beyond the end: last segment extended
    ok &= abs(e - 1.0) < 1e-9 and abs(s - 30.0) < 1e-9
    e2, _ = fn([-3.0, 4.0], line)                            # before the start: NOT extended -> distance to the start point
    ok &= abs(abs(e2) - 5.0) < 1e-9
    turn = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])  # left turn: inside of the turn is LEFT (+)
    ok &= fn([8.0, 3.0], turn)[0] > 0
    ok &= np.isnan(fn([1.0, 1.0], np.zeros((5, 2)))[0])      # standstill reference
    return bool(ok)


def mut_signed_offset(pt, poly):
    e, s = L.signed_offset(pt, poly)
    return -e, s


def chk_classify(fn):
    cases = [  # (L_P, L_S, dpsi_P, dpsi_S, e*, s*_P) -> class
        ((10, 0.5, 0, 0, 3.0, 5), "S_STOP"),
        ((20, 20, 0, 80, -3.0, 10), "WRONG_MANOEUVRE"),      # S turns left 80, P straight
        ((20, 20, 70, 80, 1.0, 10), "CORNER_CUT"),           # left turn, P left of S
        ((20, 20, 70, 80, -1.0, 10), "WIDE_TURN"),
        ((20, 20, -70, -80, -1.0, 10), "CORNER_CUT"),        # right turn, P right of S
        ((20, 20, 5, 2, 0.8, 10), "STRAIGHT_DRIFT"),
        ((20, 15, 5, 2, 0.1, 18), "RUN_LONG"),
        ((20, 25, 5, 2, 0.1, 18), "ON_PATH"),
        ((3, 25, 70, 2, 0.1, 2), "ON_PATH"),                 # P too short for WRONG_MANOEUVRE
    ]
    return all(fn(*a) == c for a, c in cases) and all(fn(*a) in L.CLASSES for a, _ in cases)


def mut_classify(L_P, L_S, dP, dS, e, s):                    # inside/outside sign inverted
    c = L.classify(L_P, L_S, dP, dS, e, s)
    return {"CORNER_CUT": "WIDE_TURN", "WIDE_TURN": "CORNER_CUT"}.get(c, c)


def chk_route_diff(fn):
    S = [["a"], ["a", "b"], ["b"]] + [[]] * 38
    ok = fn([["a"], ["c"], ["c"]] + [[]] * 38, S, 3) is True
    ok &= fn([["a"], ["a", "c"], ["b"]] + [[]] * 38, S, 3) is False   # overlap region shares 'a'
    ok &= fn([["a"], ["a"], ["c"]] + [[]] * 38, S, 2) is False        # 'c' only at/after t*
    ok &= fn([[], [], []] + [[]] * 38, S, 3) is False                  # off every lane: not a route claim
    return bool(ok)


def mut_route_diff(P, S, t):                                 # uses <= t* (includes the excursion step)
    return L.route_diff(P, S, t + 1)


def chk_auc(fn):
    ok = abs(fn([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) - 1.0) < 1e-12
    ok &= abs(fn([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0])) < 1e-12
    ok &= abs(fn([0.5, 0.5, 0.5], [1, 0, 0]) - 0.5) < 1e-12
    ok &= np.isnan(fn([0.5, 0.4], [1, 1]))
    return bool(ok)


def mut_auc(p, y):
    return L.within_token_auc(p, 1 - np.asarray(y))


def chk_decomp(fn):
    rng = np.random.default_rng(0)
    ok = True
    for rule, C in (("navsim_v1_lane", 7), ("navsim_v1", 6)):
        lg = rng.normal(size=(2, C))
        d = fn(lg[0], lg[1], rule)
        s = L.selection_score(lg, rule)
        ok &= abs(sum(d.values()) - (np.log(s[0]) - np.log(s[1]))) < 1e-9
    return bool(ok)


def mut_decomp(a, b, rule):
    d = L.term_decomposition(a, b, rule)
    d["lane"] = 0.0
    return d


def chk_best_nonzero(fn):
    score = np.array([0.9, 0.8, 0.7, 0.1])
    return fn(score, np.array([0.0, 0.5, 0.6, 0.9])) == 1 and fn(score, np.zeros(4)) == -1


def mut_best_nonzero(score, pdms):
    return int(np.argmax(score))


def chk_side(fn):
    return (fn([1, 0, 0, 0]) == "LEFT" and fn([0, 0, 1, 1]) == "RIGHT" and fn([1, 0, 0, 1]) == "MIXED"
            and fn([0, 0, 0, 0]) == "NONE")


def mut_side(o):
    s = L.side_of(o)
    return {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(s, s)


def chk_rank(fn):
    return fn(np.array([0.1, 0.9, 0.5]), 1) == 1 and fn(np.array([0.1, 0.9, 0.5]), 0) == 3


def mut_rank(score, k):
    return L.rank_of(-np.asarray(score), k)


def selection_formula_identity():
    """selection_score == the rl_val_eval.select algebra written out by hand."""
    rng = np.random.default_rng(1)
    lg = rng.normal(size=(64, 7))
    p = 1 / (1 + np.exp(-lg))
    ref = p[:, 0] * p[:, 1] * (5 * p[:, 2] + 5 * p[:, 3] + 2 * p[:, 4]) / 12 * p[:, 6]
    return np.allclose(L.selection_score(lg, "navsim_v1_lane"), ref, rtol=1e-12, atol=1e-15)


def pipeline_controls():
    """Geometry pipeline zero controls in the NAVSIM venv on 2 tokens: empty polygon set -> every step of every
    trajectory non-drivable (sim and raw); 500 m lateral shift -> every trajectory's RAW states leave the map."""
    data = HERE.parent / "data"
    env = dict(os.environ, PYTHONPATH="D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896")
    py = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
    outd = data / "mutation"
    for m in ("empty_poly", "shift500"):
        subprocess.run([py, str(HERE / "zt3_geom.py"), "--jobs", str(data / "jobs.npz"), "--shard", "0/600", "--limit", "2",
                        "--mutate", m, "--out", str(outd)], check=True, env=env, capture_output=True)
    J = np.load(data / "jobs.npz")
    e = np.load(outd / "geom_0of600_empty_poly.npz")
    s = np.load(outd / "geom_0of600_shift500.npz")
    hv = J["have"][e["row"]]
    check("pipeline: empty_poly -> every step non-drivable (sim)", bool(e["nd_sim"][hv].all()))
    check("pipeline: empty_poly -> every step non-drivable (raw)", bool(e["nd_raw"][hv].all()))
    check("pipeline: shift500 -> every trajectory's raw states leave the map", bool(s["nd_raw"][hv].any(axis=-1).all()))
    real = np.load(data / "smoke/geom_0of400.npz") if (data / "smoke/geom_0of400.npz").exists() else None
    if real is not None:
        check("pipeline: the unmutated run is NOT all non-drivable (controls are not vacuous)",
              not bool(real["nd_sim"][J["have"][real["row"]]].all()))


def main():
    zt_sha = hashlib.sha256(open(HERE / "zero_tail.py", "rb").read()).hexdigest()
    check("zero_tail.py is the landed classifier byte-for-byte", zt_sha == ZERO_TAIL_SHA256)
    check("selection_score == rl_val_eval.select algebra", selection_formula_identity())
    real_vs_mutant("nondrivable_steps", chk_nondrivable, L.nondrivable_steps, mut_nondrivable)
    real_vs_mutant("first_true", chk_first_true, L.first_true, mut_first_true)
    real_vs_mutant("signed_offset", chk_signed_offset, L.signed_offset, mut_signed_offset)
    real_vs_mutant("classify", chk_classify, L.classify, mut_classify)
    real_vs_mutant("route_diff", chk_route_diff, L.route_diff, mut_route_diff)
    real_vs_mutant("within_token_auc", chk_auc, L.within_token_auc, mut_auc)
    real_vs_mutant("term_decomposition", chk_decomp, L.term_decomposition, mut_decomp)
    real_vs_mutant("best_nonzero", chk_best_nonzero, L.best_nonzero, mut_best_nonzero)
    real_vs_mutant("side_of", chk_side, L.side_of, mut_side)
    real_vs_mutant("rank_of", chk_rank, L.rank_of, mut_rank)
    if "--pipeline" in sys.argv:
        pipeline_controls()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"ZZTEST {'PASS' if n_fail == 0 else 'FAIL'} {len(RESULTS) - n_fail}/{len(RESULTS)}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
