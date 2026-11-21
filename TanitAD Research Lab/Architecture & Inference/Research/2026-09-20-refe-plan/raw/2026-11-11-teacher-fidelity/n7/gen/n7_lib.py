"""n7_lib.py - candidate generator levers for the DAC = 0 residual (TF-N7 development, navtrain ONLY, CPU only).
Every lever is a pure function of (cx, t20, label, ref, params): params live in the frozen dict LEVERS (one entry per lever).
Every score call of a lever is recorded (Rec) and every iterate is a candidate; the tolerance certificate is the TF-N6 one (n6_nt_build.CARD6 N6T)."""
import n7_env  # noqa: F401  (before numpy)
import numpy as np
from scipy.interpolate import CubicSpline

import n4_repair as RP
import n4_repair_l1 as L1

EPS = 1e-12
T20 = 0.2 * np.arange(1, 21)
# the TF-N6 PI-approved tolerance certificate (n6_nt_build.CARD6 N6T): DAC = 1; NC, DDC not worse than the base plan; builder PDMS better; TTC/C free
CERT_TOL = {"dac": 1.0, "nc": True, "ddc": True, "ttc": False, "c": False, "pdms_gt": True}
CERT_FULL = dict(CERT_TOL, ttc=True, c=True)


def cert_clauses(m, ref, spec=CERT_TOL):
    out = {"dac": m["dac"] == spec["dac"]}
    if spec["nc"]:
        out["nc"] = m["nc"] >= ref["nc"]
    if spec["ddc"]:
        out["ddc"] = m["ddc"] >= ref["ddc"] - EPS
    if spec["ttc"]:
        out["ttc"] = m["ttc"] >= ref["ttc"] - EPS
    if spec["c"]:
        out["c"] = m["c"] >= ref["c"] - EPS
    if spec["pdms_gt"]:
        out["pdms"] = m["pdms"] > ref["pdms"]
    return out


def cert_ok(m, ref, spec=CERT_TOL):
    return all(cert_clauses(m, ref, spec).values())


class Rec:
    """recording proxy of the scorer context: every cx.score(plans) call is logged (plan, metrics) in call order"""

    def __init__(self, cx):
        self._cx, self.log = cx, []

    def score(self, plans):
        res, sim = self._cx.score(plans)
        for p, r in zip(plans, res):
            self.log.append((np.array(p, float, copy=True), dict(r)))
        return res, sim

    def __getattr__(self, name):
        return getattr(self._cx, name)


# ------------------------------------------------------------------------------------------------ pure plan transforms
def time_stretch(plan20, s):
    """pose at time t := the plan's pose at time s*t (cubic through the origin pose (0,0,0) at t = 0; heading unwrapped). s = 1.0 returns the plan bit-identical;
    s < 1 follows the same path more slowly (speed profile scaled by s), s > 1 faster (extrapolated: regression arm only)."""
    p = np.array(plan20, float)
    if s == 1.0:
        return p
    t = np.concatenate([[0.0], T20])
    full = np.vstack([[0.0, 0.0, 0.0], p])
    full[:, 2] = np.unwrap(full[:, 2])
    cs = CubicSpline(t, full, bc_type="natural", extrapolate=True)
    out = cs(s * T20)
    out[:, 2] = (out[:, 2] + np.pi) % (2 * np.pi) - np.pi
    return out


def lateral_offset(plan20, dy, t_ramp):
    """lateral offset dy (m, ego-frame y) ramped over t_ramp s; dy = 0 returns the plan bit-identical"""
    return RP.apply_shift(plan20, [0.0, dy], t_ramp)


# ------------------------------------------------------------------------------------------------ the levers: params in ONE frozen dict
LEVERS = {
    "D_base": {"kind": "D", "cap": 5, "margin": 0.05},                                   # REPAIR-D at A1 settings (control)
    "L1_base": {"kind": "L1", "cap": 5, "margin": 0.05},                                  # L1 at the TF-N5 frozen parameters (control)
}
for _c in (5, 10, 20):
    for _m in (0.05, 0.10, 0.20):
        LEVERS[f"D_c{_c}_m{int(round(_m * 100)):02d}"] = {"kind": "D", "cap": _c, "margin": _m}
        LEVERS[f"L1_c{_c}_m{int(round(_m * 100)):02d}"] = {"kind": "L1", "cap": _c, "margin": _m}
S_GRID = (0.95, 0.90, 0.85, 0.80)
LEVERS["STR_label"] = {"kind": "stretch_label", "s": S_GRID}
LEVERS["STR_target"] = {"kind": "stretch_target", "s": S_GRID}
OFF_DY = (0.05, -0.05, 0.10, -0.10, 0.20, -0.20, 0.30, -0.30, 0.50, -0.50, 0.75, -0.75, 1.0, -1.0)
OFF_RAMP = (0.2, 1.0, 2.0)
LEVERS["OFF_label"] = {"kind": "offset_label", "dy": OFF_DY, "ramp": OFF_RAMP}
LEVERS["STR_deep"] = {"kind": "stretch_label", "s": (0.70, 0.60, 0.50, 0.40, 0.30, 0.20)}      # tier 4: a large slow-down edit of the label (disclosed separately)
LEVERS["STR_OFF"] = {"kind": "stretch_offset_label", "s": (0.90, 0.80), "dy": OFF_DY, "ramp": OFF_RAMP}


def run_lever(name, params, cx, t20, label, ref, sign=1.0):
    """-> list of candidate dicts {lever, k, plan, m} in generation order. Pure in (cx scene, t20, label, ref, params); deterministic.
    sign = -1 flips every shift (deliberate-regression arm); speed levers: sign = -1 means s -> 2 - s (faster instead of slower)."""
    rc, kind = Rec(cx), params["kind"]
    if kind == "D":
        RP.repair_target(rc, t20, ref, margin=params["margin"], iters=params["cap"], sign=sign)
    elif kind == "L1":
        L1.repair_l1(rc, t20, ref, margin=params["margin"], iters=params["cap"], sign=sign)
    elif kind in ("stretch_label", "stretch_target"):
        ss = [1.0 + sign * (s - 1.0) for s in params["s"]]
        if kind == "stretch_label":
            rc.score([time_stretch(label, s) for s in ss])
        else:
            for s in ss:
                rc.score([cx.labels(time_stretch(t20, s))["R"]])
    elif kind == "stretch_offset_label":                                      # mild stretch then a lateral offset, scored directly (ONE scorer call)
        ss = [1.0 + sign * (s - 1.0) for s in params["s"]]
        rc.score([lateral_offset(time_stretch(label, s), sign * dy, tr) for s in ss for dy in params["dy"] for tr in params["ramp"]])
    elif kind == "offset_label":
        rc.score([lateral_offset(label, sign * dy, tr) for dy in params["dy"] for tr in params["ramp"]])
    else:
        raise ValueError(kind)
    return [{"lever": name, "k": k, "plan": p, "m": m} for k, (p, m) in enumerate(rc.log)]
