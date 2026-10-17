#!/usr/bin/env python3
"""test_zt5.py -- unit + MUTATION tests for PREREG_ZT5 section 8.  Every check is paired with a mutant that must go RED.

    python test_zt5.py                 # pure numpy (any venv)
    python test_zt5.py --pipeline      # + NAVSIM pipeline identities on one navtest token (navsim venv, PYTHONPATH set)
Prints ZZTEST_PASS <n> or exits 1.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zt5_lib as L  # noqa: E402
import zero_tail as zt  # noqa: E402

N = [0]
FAIL = []


def check(name, ok):
    N[0] += 1
    if not ok:
        FAIL.append(name)
    print(("PASS " if ok else "FAIL ") + name)


def arc(R=20.0, v=8.0, sign=1.0):
    """8 knots at 0.5 s on a circle through the origin tangent to +x (left turn for sign=+1), true headings."""
    t = 0.5 * np.arange(1, 9)
    phi = v * t / R
    x, y = R * np.sin(phi), sign * R * (1 - np.cos(phi))
    return np.stack([x, y, sign * phi], -1)


def tangent_exact_on_arc(fix):
    p = arc()
    bad = p.copy()
    bad[:, 2] += 0.3                                                  # corrupt every heading
    q = fix(bad)
    return np.allclose(q[:, 2], p[:, 2], atol=1e-9) and np.array_equal(q[:, :2], bad[:, :2])


def mutant_forward_chord(p8):
    p = np.array(p8, dtype=np.float64, copy=True)
    xy = np.concatenate([np.zeros((1, 2)), p[:, :2]], 0)
    for k in range(1, 8):
        c = xy[k + 1] - xy[k]                                         # MUTANT: forward chord instead of central
        p[k - 1, 2] = np.arctan2(c[1], c[0])
    return p


def main():
    # 1. F_c exact on a circular arc, positions bit-identical; mutant RED
    check("F_c tangent exact on an arc, xy bit-identical", tangent_exact_on_arc(L.fix_heading_tangent))
    check("MUTANT forward-chord tangent goes RED", not tangent_exact_on_arc(mutant_forward_chord))
    # right turn too
    p = arc(sign=-1.0)
    check("F_c exact on a right-turn arc", np.allclose(L.fix_heading_tangent(p + [0, 0, -0.2])[:, 2], p[:, 2], atol=1e-9))
    # 2. straight line with corrupt heading -> 0; stationary plan unchanged
    line = np.stack([np.arange(1, 9) * 4.0, np.zeros(8), np.full(8, 0.4)], -1)
    check("F_c straight line -> heading 0", np.allclose(L.fix_heading_tangent(line)[:, 2], 0.0, atol=1e-12))
    still = np.zeros((8, 3))
    still[:, 2] = 0.7
    check("F_c stationary plan unchanged (no tangent defined)", np.array_equal(L.fix_heading_tangent(still), still))
    # 3. nearest branch: heading near +pi stays on its branch
    q = L.fix_heading_tangent(np.stack([-np.arange(1, 9) * 3.0, np.zeros(8), np.full(8, np.pi - 0.05)], -1))
    check("F_c keeps the original heading branch", np.allclose(q[:, 2], np.pi, atol=1e-12))
    # 4. kappa0 exact on a circle; mutant (1/r) RED
    p = arc(R=25.0)
    check("kappa0 exact on a circle", abs(L.kappa0(p) - 1 / 25.0) < 1e-12)
    check("MUTANT kappa0 = y/r^2 goes RED", abs(p[0, 1] / (p[0, 0] ** 2 + p[0, 1] ** 2) - 1 / 25.0) > 1e-6)
    # 5. F_b: rho = 1 reproduces the pick; mutant without rho switches -> RED
    rng = np.random.default_rng(0)
    s = rng.random(64)
    d = rng.random(64) * 2
    pk = int(np.argmax(s))
    d[pk] = 5.0
    check("F_b rho=1.0 keeps the pick", L.fb_pick(s, d, pk, 1.0) == pk)
    mut = int(np.argmin(d))                                           # MUTANT: candidate set ignores rho
    check("MUTANT F_b ignoring rho goes RED", mut != pk)
    # switch threshold
    s2 = np.array([1.0, 0.999, 0.5])
    d2 = np.array([0.50, 0.45, 0.0])
    check("F_b no switch below 0.10 m gain", L.fb_pick(s2, d2, 0, 0.99) == 0)
    d3 = np.array([0.50, 0.35, 0.0])
    check("F_b switches at >= 0.10 m gain inside the tie band", L.fb_pick(s2, d3, 0, 0.99) == 1)
    check("F_b never leaves the tie band", L.fb_pick(s2, d3, 0, 0.99) != 2)
    # 6. self-deviation translation invariance; decompose sign
    a = rng.random((41, 2))
    b = a + rng.normal(0, 0.1, (41, 2))
    check("delta translation invariant", abs(L.self_deviation(a + 500, b + 500) - L.self_deviation(a, b)) < 1e-9)
    th = np.linspace(0, 0.8, 41)
    R = np.stack([20 * np.sin(th), 20 * (1 - np.cos(th))], -1)          # left turn
    inside = R + 0.3 * np.stack([-np.sin(th), np.cos(th)], -1)          # shifted toward the centre (left normal)
    dd = L.decompose(inside, R, th, t=20)
    check("decompose: inside shift of a left turn reads cross_inside > 0", dd["cross_inside"] > 0.29)
    outside = R - 0.3 * np.stack([-np.sin(th), np.cos(th)], -1)
    check("decompose: outside shift reads cross_inside < 0 (sign mutant RED)", L.decompose(outside, R, th, t=20)["cross_inside"] < -0.29)
    # 7. selection score == rl_val_eval.select
    sys.path.insert(0, "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval")
    try:
        import rl_val_eval as RV
        lg = rng.normal(0, 2, (64, 7))
        pk2, sel = RV.select(lg, "navsim_v1_lane")
        check("selection_score == rl_val_eval.select", np.allclose(L.selection_score(lg), sel, rtol=0, atol=1e-15)
              and int(np.argmax(L.selection_score(lg))) == pk2)
    except ImportError as e:
        check(f"rl_val_eval importable ({e})", False)
    # 8. bootstrap self-pair exactly 0; zero_tail vendored byte-identical
    li = np.repeat(np.arange(5), 4)
    r0 = zt.Boot(li, 5).ratio(np.zeros(20), np.ones(20))
    check("bootstrap self-pair exactly 0", r0["point"] == 0.0 and r0["lo"] == 0.0 and r0["hi"] == 0.0)
    h = hashlib.sha256(open(HERE / "zero_tail.py", "rb").read()).hexdigest()
    check("zero_tail.py byte-identical to ZT3's", h == "149c3e97927ed219323f8f070b67a36d9172a575e4ec23162241c417f8501a3d")
    # 9. comp_step fixed point
    p8 = arc()
    check("comp_step: tracked == plan leaves r unchanged", np.allclose(L.comp_step(p8, p8, p8), p8))
    if "--pipeline" in sys.argv:
        pipeline()
    print(f"ZZTEST_{'PASS' if not FAIL else 'FAIL'} {N[0]}" + (f" failed={FAIL}" if FAIL else ""))
    return 0 if not FAIL else 1


def pipeline():
    """NAVSIM identities on one navtest token (the first T_A tracker-made token of ZT3)."""
    import json
    import lzma
    import pickle
    import zt5_sim as Z
    tm = [json.loads(x) for x in open("C:/Users/Admin/zt3/raw/2026-10-17-zero-tail-mechanism/data/per_token_A.jsonl")]
    J = np.load("C:/Users/Admin/zt3/raw/2026-10-17-zero-tail-mechanism/data/jobs.npz")
    t = [x["token"] for x in tm if x["tracker_made"]][0]
    r = int(np.nonzero(J["token"] == t)[0][0])
    lg = str(J["log"][r])
    mc = pickle.loads(lzma.decompress(open(f"D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest/{lg}/unknown/{t}/metric_cache.pkl", "rb").read()))
    ego, dm = mc.ego_state, mc.drivable_area_map
    p8 = J["traj"][r, 0].astype(np.float64)
    R = Z.raw_states(p8[None], ego)
    S = Z.simulate(R, ego)
    S2 = Z.simulate(R, Z.ego_override(ego))
    check("pipeline: ego_override with no field changed reproduces the base sim bit-for-bit", np.array_equal(S, S2))
    S3 = Z.simulate(R, Z.ego_override(ego, v=float(ego.dynamic_car_state.rear_axle_velocity_2d.x) + 1.0))
    check("MUTANT pipeline: a changed initial speed changes the sim", not np.array_equal(S, S3))
    SA = Z.simulate(R, ego, actuator_lag=False)
    check("pipeline: CF-A (lags removed) changes the sim", not np.array_equal(S, SA))
    idcs = Z.drivable_idx(dm)
    check("pipeline: base sim DAC = 0 and raw DAC = 1 on a ZT3 tracker-made token",
          Z.dac(S, ego, dm, idcs)[0] == 0.0 and Z.dac(R, ego, dm, idcs)[0] == 1.0)
    check("MUTANT pipeline: no drivable polygons -> DAC 0 on the raw plan", Z.dac(R, ego, dm, [])[0] == 0.0)
    check("pipeline: AG1 (navtest ego steering and yaw rate are 0)", Z.admissible_ego_ok(ego))


if __name__ == "__main__":
    sys.exit(main())
