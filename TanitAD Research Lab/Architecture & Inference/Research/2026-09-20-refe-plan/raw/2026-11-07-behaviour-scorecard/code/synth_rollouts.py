"""Synthetic rollout / human-reference generator for the behaviour-score tests (no sim needed). 20 Hz, straight
road along +x unless a curvature is given. Everything is analytic so every expected value is hand-checkable."""
import numpy as np

DT = 0.05
EGO_L, EGO_W = 4.8, 2.3
SHA = "0" * 64


def tgrid(duration):
    return np.arange(0.0, duration + 1e-9, DT)


def kf(points, t):
    """piecewise-linear speed profile from (time, speed) keyframes"""
    p = np.asarray(points, float)
    return np.interp(t, p[:, 0], p[:, 1])


def integrate(v, x0=0.0):
    return x0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * DT)])


def stop_profile(t, v0, x0, x_stop, decel=2.0):
    """cruise at v0 from x0 then brake at `decel` to stand exactly at x_stop (centre). returns v(t)"""
    D = v0 * v0 / (2 * decel)
    tb = (x_stop - D - x0) / v0
    return np.where(t < tb, v0, np.maximum(v0 - decel * (t - tb), 0.0))


def agent(aid, x, y, v, yaw=0.0, l=4.5, w=2.0, h=1.6, cls=0, present=None):
    T = len(x)
    return dict(id=aid, x=np.asarray(x, float), y=np.asarray(y, float), v=np.asarray(v, float),
                yaw=np.broadcast_to(np.asarray(yaw, float), (T,)).copy(), l=l, w=w, h=h, cls=cls,
                present=np.ones(T, bool) if present is None else np.asarray(present, bool))


def const_agent(aid, t, x0, y, v, **kw):
    return agent(aid, x0 + v * t, np.full_like(t, y), np.full_like(t, v), **kw)


def make(t, v, y=None, yaw=None, x0=0.0, agents=(), scenario_id="syn", category="cruise", synthetic=False,
         signal=None, governs=None, stop_line_s=np.nan, zone=None, conf_ids=(), epdms=None, curvature=0.0, steer=None):
    """build a rollout dict (npz keys). curvature != 0 -> the ego path is a circular arc of that curvature."""
    T = len(t)
    v = np.asarray(v, float)
    s = integrate(v, x0)                               # arc length travelled
    if curvature:
        k = curvature
        x, yy, hd = np.sin(k * s) / k, (1 - np.cos(k * s)) / k, k * s
        yy = yy if y is None else yy + np.asarray(y)
    else:
        x, yy, hd = s, (np.zeros(T) if y is None else np.broadcast_to(np.asarray(y, float), (T,)).copy()), np.zeros(T)
    if yaw is not None:
        hd = hd + np.asarray(yaw, float)
    d = dict(
        schema_version=np.array("refe-rollout-v1"), scenario_id=np.array(scenario_id), log_id=np.array(scenario_id),
        category=np.array(category), synthetic=np.array(synthetic), checkpoint_sha256=np.array(SHA), dtype=np.array("fp32"),
        seed=np.array(20261107, dtype=np.int64), suite_tokens_sha256=np.array(SHA),
        t=t, ego_x=x, ego_y=yy, ego_yaw=hd, ego_v=v, ego_a=np.gradient(v, DT), ego_steer=np.zeros(T) if steer is None else steer,
        ego_length=np.array(EGO_L), ego_width=np.array(EGO_W), stop_line_s=np.array(stop_line_s, dtype=float),
    )
    A = len(agents)
    d["agent_id"] = np.array([a["id"] for a in agents], dtype=np.int64)
    d["agent_present"] = np.array([a["present"] for a in agents], dtype=bool).reshape(A, T)
    for k in ("x", "y", "yaw", "v"):
        d["agent_" + k] = np.array([np.where(a["present"], a[k], np.nan) for a in agents], dtype=float).reshape(A, T)
    d["agent_l"] = np.array([a["l"] for a in agents], dtype=float)
    d["agent_w"] = np.array([a["w"] for a in agents], dtype=float)
    d["agent_h"] = np.array([a["h"] for a in agents], dtype=float)
    d["agent_cls"] = np.array([a["cls"] for a in agents], dtype=np.int8)
    if signal is not None:
        sig = np.asarray(signal, np.int8)
        d["signal_state"] = sig.reshape(T, -1)
        d["signal_governs_ego"] = np.ones(d["signal_state"].shape[1], bool) if governs is None else np.asarray(governs, bool)
    else:
        d["signal_state"] = np.zeros((T, 0), np.int8)
        d["signal_governs_ego"] = np.zeros(0, bool)
    d["conflict_zone_xy"] = np.zeros((0, 2)) if zone is None else np.asarray(zone, float)
    d["conflict_agent_ids"] = np.asarray(conf_ids, np.int64)
    if epdms is not None:
        d["epdms"] = np.array(epdms, dtype=float)
    return d


def human_of(d):
    """the human reference file: same arrays minus identity/metadata the reference does not carry"""
    h = {k: v for k, v in d.items()}
    return h


def box_zone(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
