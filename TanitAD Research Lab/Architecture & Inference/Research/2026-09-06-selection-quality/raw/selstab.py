"""selstab - TEMPORAL SELECTION STABILITY and VALID SELECTION REGRET.

A new instrument.  It answers the two questions the programme does not
currently measure at all:

  (6) does the selection JUMP between consecutive frames?  -- and is that jump
      a DECISION change or only a waypoint wobble?
  (2) does the selector pick a WORSE candidate than the fan contains?

Both are read against a FLOOR and a CEILING, because a jitter number or a
regret number without them is unreadable.

============================================================================
THE FAN IS RECONSTRUCTED MODEL-FREE, AND THAT IS WHAT MAKES THE CEILING VALID
============================================================================
`refc.py::RefCDecoder._anchor_bank` emits, for a window whose measured speed is
`v0`:
    kappa_n = clamp(a_lat_n / max(v0, alat_v_floor)^2, +-kappa_cap)
    path_n  = rollout_unicycle(state0=(0,0,0,v0), (a_lon_n, kappa_n) HELD
                               CONSTANT for `anchor_roll_steps` ticks, dt=0.1)
so the emitted fan is a DETERMINISTIC FUNCTION OF (anchor_controls, v0) and
needs no forward pass.  `anchor_controls` is read out of the checkpoint itself.

!! THIS IS NOT `a_star`, AND THE DIFFERENCE IS THE WHOLE POINT.
`taniteval/tools/refcv3_arm.py` binds `a_star` against
`model.core.decoder.anchors` -- the FIXED bank rolled once at `ref_speed_ms` --
which is exactly the binding the trainer forbids for a v0-conditioned
vocabulary, and which produced a "ceiling" that scored WORSE than the arm it
was supposed to bound.  Everything here binds against the PER-WINDOW
v0-conditioned roll instead.  `prove_not_astar()` measures how often the two
argmins disagree, so the distinction is demonstrated rather than asserted.

CONTROLS THAT MUST READ KNOWN VALUES
  C1  straight candidate (a_lat = 0): y == 0 and dyaw == 0 EXACTLY.
  C2  v0 round-trip: dump v0 == poses[ws, 3] to 1e-5 on every window, or the
      window->pose mapping is wrong and every transform below is void.
  C3  GT self-consistency: the GT plan at t and at t+1, brought into a common
      frame, must agree to ~0.  This validates the rigid transform AND is the
      CEILING for waypoint stability.
  C4  the shuffled-selection floor must be strictly below the arm.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"
EPS = r"C:\Users\Admin\refav1_eval_full\eps"
KAPPA_CAP = 0.12
ALAT_V_FLOOR = 4.0
DT_TICK = 0.1
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)     # ticks
SLOTS_2S = (0, 1, 2, 3)                        # the 2 s scored grid
DT_S = 0.5


# --------------------------------------------------------------------------
# the fan
# --------------------------------------------------------------------------
def anchor_controls(ckpt=CKPT):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    d = sd.get("model", sd)
    k = [x for x in d if x.endswith("anchor_controls")][0]
    return d[k].float(), k


def fan(ctrl, v0, slots=None, steps=None):
    """[B, N, len(slots), 2] emitted fan -- the programme's own integrator.

    `steps` defaults to exactly the ticks the requested slots need, never the
    full 60: a 2 s read does not have to pay for a 6 s rollout, and the
    intermediate state tensor is the memory wall here.
    """
    from tanitad.models.kinematic import rollout_unicycle
    if steps is None:
        steps = max(HORIZONS) if slots is None else max(HORIZONS[s]
                                                        for s in slots)
    v0 = np.asarray(v0, dtype=np.float64)
    B, N = len(v0), ctrl.shape[0]
    a_lon = ctrl[:, 0].numpy().astype(np.float64)
    a_lat = ctrl[:, 1].numpy().astype(np.float64)
    vv = np.maximum(v0, ALAT_V_FLOOR) ** 2
    kap = np.clip(a_lat[None, :] / vv[:, None], -KAPPA_CAP, KAPPA_CAP)
    c = np.stack([np.broadcast_to(a_lon[None, :], (B, N)), kap], -1)
    c = torch.as_tensor(c, dtype=torch.float32)[:, :, None, :]
    c = c.expand(B, N, steps, 2).reshape(-1, steps, 2)
    s0 = torch.zeros(B * N, 4, dtype=torch.float32)
    s0[:, 3] = torch.as_tensor(np.repeat(v0, N), dtype=torch.float32)
    st = rollout_unicycle(s0, c, dt=DT_TICK).reshape(B, N, steps, 4)
    idx = [h - 1 for h in HORIZONS] if slots is None else \
        [HORIZONS[s] - 1 for s in slots]
    return st[:, :, idx, :]


# --------------------------------------------------------------------------
# per-window costs -- the four families, never pooled into one score
# --------------------------------------------------------------------------
def path_geometry(p, v0, dt=DT_S):
    """p [..., K, 2] in the ego frame at t0 -> per-path scalars.

    Prepends the origin so the first segment is the real one, then reads
    heading and curvature from the polyline.  Segments shorter than 0.25 m are
    masked out of the curvature read (the same MIN_DS the four-families
    lateral block uses), because a heading off a sub-decimetre step is noise.
    """
    z = np.zeros(p.shape[:-2] + (1, 2))
    q = np.concatenate([z, p], axis=-2)                  # [..., K+1, 2]
    d = np.diff(q, axis=-2)                              # [..., K, 2]
    ds = np.linalg.norm(d, axis=-1)                      # [..., K]
    th = np.arctan2(d[..., 1], d[..., 0])
    dth = np.diff(th, axis=-1)
    dth = (dth + np.pi) % (2 * np.pi) - np.pi            # [..., K-1]
    seg = 0.5 * (ds[..., :-1] + ds[..., 1:])
    ok = seg >= 0.25
    kap = np.where(ok, dth / np.maximum(seg, 1e-6), 0.0)
    n_ok = ok.sum(-1)
    return {"ds": ds, "heading": th, "kappa": kap, "kappa_ok": ok,
            "n_kappa": n_ok, "speed": ds / dt, "arc": ds.sum(-1)}


def family_costs(p, g, v0):
    """Per-path four-family errors against the GT path g [..., K, 2].

    Returns a dict of arrays broadcast over the leading dims of `p`.
    ADE is reported BESIDE them, never instead of them.
    """
    P, G = path_geometry(p, v0), path_geometry(g, v0)
    err = np.linalg.norm(p - g, axis=-1)                     # [..., K]
    ade = err.mean(-1)
    # LONGITUDINAL: along-track (GT heading frame) + speed
    gh = np.arctan2(g[..., 1], g[..., 0])
    u = np.stack([np.cos(gh), np.sin(gh)], -1)
    n = np.stack([-np.sin(gh), np.cos(gh)], -1)
    dv = p - g
    along = np.abs((dv * u).sum(-1)).mean(-1)
    cross = np.abs((dv * n).sum(-1)).mean(-1)
    sp = np.abs(P["speed"] - G["speed"]).mean(-1)
    # LATERAL: curvature MAE on the jointly-valid segments + heading MAE
    both = P["kappa_ok"] & G["kappa_ok"]
    kd = np.abs(P["kappa"] - G["kappa"]) * both
    nk = both.sum(-1)
    kmae = np.where(nk > 0, kd.sum(-1) / np.maximum(nk, 1), np.nan)
    hd = np.abs((P["heading"] - G["heading"] + np.pi) % (2 * np.pi) - np.pi)
    hmae = np.degrees(hd.mean(-1))
    return {"ade_m": ade, "along_mae_m": along, "cross_mae_m": cross,
            "speed_mae_mps": sp, "curv_mae_1pm": kmae, "heading_mae_deg": hmae,
            "n_kappa": nk}


# --------------------------------------------------------------------------
# rigid transform between consecutive windows
# --------------------------------------------------------------------------
def to_frame(p_next, pose_t, pose_n):
    """Bring a plan expressed in the ego frame at `pose_n` into the ego frame
    at `pose_t`.  poses are (x, y, yaw, v) in the clip's world frame."""
    dx = pose_n[0] - pose_t[0]
    dy = pose_n[1] - pose_t[1]
    c, s = np.cos(-pose_t[2]), np.sin(-pose_t[2])
    ox = c * dx - s * dy
    oy = s * dx + c * dy
    dyaw = pose_n[2] - pose_t[2]
    cr, sr = np.cos(dyaw), np.sin(dyaw)
    x = cr * p_next[..., 0] - sr * p_next[..., 1] + ox
    y = sr * p_next[..., 0] + cr * p_next[..., 1] + oy
    return np.stack([x, y], -1)


def resample(q, ts, t_dst):
    """Linear resample of a knotted polyline ``q`` at times ``ts`` onto t_dst.

    ⛔ ``q`` must ALREADY carry its own origin knot, at its own time.  An
    earlier version prepended ``(0, 0)`` at ``t = 0`` unconditionally, which is
    wrong for a plan that has been brought into ANOTHER frame: after the rigid
    transform the next window's origin is the ego OFFSET, not (0, 0), and it
    sits at ``t = +stride*0.1 s``, not at 0.  Both errors inflate the very
    quantity this function exists to measure.
    """
    ts = np.asarray(ts, dtype=float)
    out = np.empty(q.shape[:-2] + (len(t_dst), 2))
    for j, t in enumerate(t_dst):
        i = int(np.searchsorted(ts, t)) - 1
        i = min(max(i, 0), len(ts) - 2)
        w = (t - ts[i]) / (ts[i + 1] - ts[i])
        out[..., j, :] = (1 - w) * q[..., i, :] + w * q[..., i + 1, :]
    return out


PLAN_T = np.array([0.5, 1.0, 1.5, 2.0])


def plan_in_frame(p_next, pose_t, pose_n, shift_s):
    """The NEXT window's plan, expressed in window t's ego frame, WITH its own
    origin knot at its own time.  Returns ``(knots [K+1, 2], times [K+1])``."""
    org = to_frame(np.zeros((1, 2)), pose_t, pose_n)
    body = to_frame(p_next, pose_t, pose_n)
    return (np.concatenate([org, body], axis=0),
            np.concatenate([[shift_s], PLAN_T + shift_s]))


def plan_own_frame(p):
    """A plan in its OWN frame, with the origin knot at t = 0."""
    return (np.concatenate([np.zeros((1, 2)), p], axis=0),
            np.concatenate([[0.0], PLAN_T]))
