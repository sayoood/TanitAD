"""D2 analytic controls: synthetic tracks with KNOWN geometry.

Two objects are put through the same tracks:
  (1) the independent classifier in d2_lib (must read the literal expected class), and
  (2) the BUILDER (s2_geom_emit_v7.tactical_actions) as an object under test -- its output is
      reported, not asserted: a builder that disagrees with road geometry on a track whose
      class is analytic is itself a finding.
"""
import importlib.util
import json
import math
import sys

import numpy as np

sys.path.insert(0, "C:/Users/Admin/ev7/stack")
import d2_lib as L

HZ = 10.0
TA = 8.0
T_END = 40.0


def integrate(v_fn, w_fn=None, y_fn=None):
    """Fine (1 ms) integration, subsampled to 10 Hz.  Either a yaw-rate fn (arc tracks) or a
    lateral-offset fn y(t) (heading-line tracks)."""
    dt = 0.001
    t = np.arange(0.0, T_END + 1e-9, dt)
    v = np.array([v_fn(a) for a in t])
    if w_fn is not None:
        w = np.array([w_fn(a) for a in t])
        psi = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * dt)])
        x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] * np.cos(psi[1:]) + v[:-1] * np.cos(psi[:-1])) * dt)])
        y = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] * np.sin(psi[1:]) + v[:-1] * np.sin(psi[:-1])) * dt)])
        spd = v
    else:
        x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
        y = np.array([y_fn(a) for a in t])
        dy = np.gradient(y, dt)
        dx = np.gradient(x, dt)
        psi = np.arctan2(dy, dx)
        spd = np.hypot(dx, dy)
    s = slice(0, None, 100)
    return np.stack([x[s], y[s], psi[s], spd[s]], axis=1)


def pulse(a, t0, dur, amp):          # half-cosine out-and-back, peak amp at the middle
    if t0 <= a <= t0 + dur:
        return amp * 0.5 * (1 - math.cos(2 * math.pi * (a - t0) / dur))
    return 0.0


def step(a, t0, dur, amp):           # smooth 0 -> amp, stays
    if a < t0:
        return 0.0
    if a > t0 + dur:
        return amp
    return amp * 0.5 * (1 - math.cos(math.pi * (a - t0) / dur))


def arc_w(R, v, t_on, t_off, sign):
    return lambda a: sign * v / R if t_on <= a <= t_off else 0.0


def tracks():
    out = []
    V = lambda c: (lambda a: c)
    # ---- lateral, known geometry ----------------------------------------------------
    # R = 20 m, 5 m/s, yaw rate 0.25 rad/s starting at ta+2 : in-band dyaw = 0.25*4 rad = 57.2958 deg
    out.append(("arc_L_R20_v5", integrate(V(5.0), arc_w(20, 5.0, TA + 2.0, TA + 8.3, +1)), "TURN_L", "CRUISE", 57.2958))
    out.append(("arc_R_R20_v5", integrate(V(5.0), arc_w(20, 5.0, TA + 2.0, TA + 8.3, -1)), "TURN_R", "CRUISE", -57.2958))
    out.append(("straight_v14", integrate(V(14.0), lambda a: 0.0), "LANE_KEEP", "CRUISE", 0.0))
    # gentle bend R=300, v=14: in-band dyaw = 14*4/300 rad = 10.69 deg (a road curve, not a turn)
    out.append(("bend_L_R300_v14", integrate(V(14.0), arc_w(300, 14.0, TA, TA + 10, +1)), "LANE_KEEP", "CRUISE", 10.693))
    # bend R=160, v=14: in-band dyaw = 14*4/160 rad = 20.05 deg -> BEND (monotone, below the turn bar)
    out.append(("bend_L_R160_v14", integrate(V(14.0), arc_w(160, 14.0, TA, TA + 10, +1)), "BEND_L", "CRUISE", 20.0535))
    # wide bend R=100, v=14: in-band dyaw = 32.09 deg (A: TURN by the brief's pure-yaw rule)
    out.append(("bend_L_R100_v14", integrate(V(14.0), arc_w(100, 14.0, TA, TA + 10, +1)), "TURN_L", "CRUISE", 32.0856))
    # lateral pulses on a straight at 14 m/s (heading returns)
    out.append(("nudge_L_1.0m", integrate(V(14.0), y_fn=lambda a: pulse(a, TA + 2.0, 4.0, +1.0)), "NUDGE_L", "CRUISE", 0.0))
    out.append(("nudge_R_1.0m", integrate(V(14.0), y_fn=lambda a: pulse(a, TA + 2.0, 4.0, -1.0)), "NUDGE_R", "CRUISE", 0.0))
    out.append(("pulse_L_3.5m", integrate(V(14.0), y_fn=lambda a: pulse(a, TA + 2.0, 4.0, +3.5)), "SHIFT_L", "CRUISE", 0.0))
    out.append(("lanechange_R_3.5m", integrate(V(14.0), y_fn=lambda a: step(a, TA + 2.0, 4.0, -3.5)), "SHIFT_R", "CRUISE", 0.0))
    out.append(("wobble_0.1m", integrate(V(14.0), y_fn=lambda a: pulse(a, TA + 2.0, 4.0, +0.1)), "LANE_KEEP", "CRUISE", 0.0))
    # ---- longitudinal, known profile -----------------------------------------------
    ramp = lambda v0, a_, t0, t1: (lambda t: v0 if t < t0 else (v0 + a_ * (t - t0) if t < t1 else v0 + a_ * (t1 - t0)))
    out.append(("accel_+0.5", integrate(ramp(10.0, 0.5, TA + 2.0, TA + 6.0), lambda a: 0.0), "LANE_KEEP", "ACCEL", 0.0))
    out.append(("decel_-0.5", integrate(ramp(10.0, -0.5, TA + 2.0, TA + 6.0), lambda a: 0.0), "LANE_KEEP", "BRAKE", 0.0))
    out.append(("decel_to_stop", integrate(ramp(4.0, -1.0, TA + 2.0, TA + 6.0), lambda a: 0.0), "LANE_KEEP", "BRAKE", 0.0))
    out.append(("cruise_dv_+1.0", integrate(ramp(10.0, 0.25, TA + 2.0, TA + 6.0), lambda a: 0.0), "LANE_KEEP", "CRUISE", 0.0))
    out.append(("stopped_v0", integrate(V(0.0), lambda a: 0.0), "LANE_KEEP", "HOLD", 0.0))
    out.append(("creep_v1.0", integrate(V(1.0), lambda a: 0.0), "LANE_KEEP", "CREEP", 0.0))
    return out


def mk_track(P):
    t = np.arange(P.shape[0]) / HZ
    return L.Track(0, "synthetic", t, P[:, 0], P[:, 1], np.unwrap(P[:, 2]), P[:, 3], "synthetic")


def load_builder():
    spec = importlib.util.spec_from_file_location("s2v7", "C:/Users/Admin/ev7/stack/scripts/s2_geom_emit_v7.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(outp):
    B = load_builder()
    rows = []
    n_ok_A = n_ok_B = 0
    for name, P, exp_lat, exp_lon, exp_dpsi in tracks():
        tr = mk_track(P)
        fa = L.lat_features(tr, TA, 2.0, 6.0)
        fl = L.lon_features(tr, TA, 2.0, 6.0)
        latA, latB = L.classify_lat(fa, "A"), L.classify_lat(fa, "B")
        lon = L.classify_lon(fl)
        # builder under test
        key = int(round(TA * HZ))
        seq = B.manoeuvre_sequence(P, key, HZ)
        at = B.tactical_actions(P, key, seq, HZ)
        r = {"track": name, "expected_lat": exp_lat, "expected_lon": exp_lon,
             "expected_band_dyaw_deg": exp_dpsi,
             "indep": {"lat_A": latA, "lat_B": latB, "lon": lon,
                       "dpsi_peak_deg": round(fa["dpsi_peak"], 4), "dpsi_fast3_deg": round(fa["dpsi_fast3"], 4),
                       "eps_m": round(fa["eps"], 4), "dv": round(fl["dv"], 4)},
             "indep_lat_A_ok": latA == exp_lat,
             "indep_lon_ok": lon == exp_lon,
             "dpsi_err_deg": round(abs(fa["dpsi_peak"] - exp_dpsi), 4),
             "builder": {"lat": at["lat"], "lon": at["lon"], "seq": [list(s) for s in seq]}}
        rows.append(r)
    res = {"rows": rows}
    json.dump(res, open(outp, "w"), indent=1)
    return res


if __name__ == "__main__":
    res = main(sys.argv[1])
    print(f"{'track':20s} {'exp_lat':10s} {'indepA':10s} {'indepB':10s} | {'exp_lon':8s} {'indep':8s} | dpsi_exp  dpsi_got  eps | BUILDER lat/lon")
    for r in res["rows"]:
        i = r["indep"]
        print(f"{r['track']:20s} {r['expected_lat']:10s} {i['lat_A']:10s} {i['lat_B']:10s} | {r['expected_lon']:8s} {i['lon']:8s} | "
              f"{r['expected_band_dyaw_deg']:8.3f} {i['dpsi_peak_deg']:9.3f} {i['eps_m']:5.3f} | {r['builder']['lat']}/{r['builder']['lon']}")
    nA = sum(r["indep_lat_A_ok"] for r in res["rows"])
    nL = sum(r["indep_lon_ok"] for r in res["rows"])
    print(f"indep lat A correct {nA}/{len(res['rows'])}; lon correct {nL}/{len(res['rows'])}; max dpsi err {max(r['dpsi_err_deg'] for r in res['rows'])}")
