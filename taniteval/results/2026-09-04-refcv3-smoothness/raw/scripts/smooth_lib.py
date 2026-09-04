"""WP-5/6 smoothness kernel. Pure numpy; no GPU, no checkpoint.

DEFINITIONS (stated so a reader can falsify them)
  A plan is P = [p0=(0,0), p1..pS] in the ego frame at t0, with slot times
  t = [0, t1..tS].  Segment i (1..S) runs p_{i-1}->p_i over dt_i = t_i - t_{i-1}.

  L_i      segment length (m)
  psi_i    segment heading (rad)
  turn_i   vertex EXTERIOR angle at p_i (i = 1..S-1), wrapped to (-pi, pi].
           *** THIS IS EXACTLY THE KINK THE RENDERER DRAWS *** — render_refcv3_video.py
           densify() joins the emitted slots with straight lines, so the visible
           corner at p_i IS turn_i.
  kappa_i  turn_i / (0.5*(L_i + L_{i+1}))   [1/m]  -- turn per unit ARC LENGTH.
           dt-INVARIANT for a constant-curvature path, so it is the quantity that
           is comparable ACROSS the 2 s seam where dt doubles.
  omega_i  turn_i / (0.5*(dt_i + dt_{i+1}))  [rad/s] -- yaw rate.
  v_i      L_i / dt_i (m/s); a_i, jerk_i  its first/second differences in time.
"""
import numpy as np

T8 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0])   # refcv3 grid
T4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0])                        # refc-base grid
SEAM_VERTEX = 4          # p4 = t 2.0 s: the slot where dt goes 0.5 -> 1.0


def with_origin(path):
    """[..., S, 2] -> [..., S+1, 2] with the ego origin prepended."""
    path = np.asarray(path, dtype=np.float64)
    z = np.zeros(path.shape[:-2] + (1, 2))
    return np.concatenate([z, path], axis=-2)


def geom(path_with_origin, t):
    """Return dict of per-segment / per-vertex geometry. path [..., S+1, 2]."""
    p = np.asarray(path_with_origin, dtype=np.float64)
    d = np.diff(p, axis=-2)                       # [..., S, 2]
    L = np.linalg.norm(d, axis=-1)                # [..., S]
    psi = np.arctan2(d[..., 1], d[..., 0])        # [..., S]
    dt = np.diff(t)                               # [S]
    turn = np.diff(psi, axis=-1)                  # [..., S-1]
    turn = (turn + np.pi) % (2 * np.pi) - np.pi   # wrap
    Lm = 0.5 * (L[..., :-1] + L[..., 1:])
    dtm = 0.5 * (dt[:-1] + dt[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(Lm > 1e-6, turn / np.maximum(Lm, 1e-9), np.nan)
        v = L / dt
    omega = turn / dtm
    a = np.diff(v, axis=-1) / dtm                                  # [..., S-1]
    jerk = (np.diff(a, axis=-1)
            / (0.5 * (dtm[:-1] + dtm[1:])))                        # [..., S-2]
    return dict(L=L, psi=psi, turn=turn, kappa=kappa, omega=omega,
                v=v, a=a, jerk=jerk, dt=dt)


# ---------------------------------------------------------------- controls --
def cv_path(v0, t):
    """CONSTANT VELOCITY straight line. MUST read turn == 0 EXACTLY."""
    return np.stack([np.asarray(v0)[..., None] * t[1:],
                     np.zeros(np.shape(v0) + (len(t) - 1,))], axis=-1)


def arc_path(v0, kappa, t, a=0.0):
    """CONSTANT-CURVATURE arc from (v0, kappa, a) — a KNOWN-SMOOTH path whose
    true curvature is EXACTLY constant. Any turn-angle step it shows at the seam
    is 100 % a property of the SAMPLING GRID, not of the path."""
    v0 = np.atleast_1d(np.asarray(v0, dtype=np.float64))
    kap = np.atleast_1d(np.asarray(kappa, dtype=np.float64))
    a = np.atleast_1d(np.asarray(a, dtype=np.float64)) * np.ones_like(v0)
    dt = 0.01
    n = int(round(t[-1] / dt))
    x = np.zeros_like(v0); y = np.zeros_like(v0); yaw = np.zeros_like(v0)
    v = v0.copy()
    ts = np.zeros((n + 1,)); xs = [x.copy()]; ys = [y.copy()]
    for k in range(n):
        x = x + v * np.cos(yaw) * dt
        y = y + v * np.sin(yaw) * dt
        yaw = yaw + v * kap * dt
        v = np.maximum(v + a * dt, 0.0)
        ts[k + 1] = (k + 1) * dt
        xs.append(x.copy()); ys.append(y.copy())
    xs = np.stack(xs, -1); ys = np.stack(ys, -1)                   # [B, n+1]
    idx = np.round(t[1:] / dt).astype(int)
    return np.stack([xs[..., idx], ys[..., idx]], axis=-1)         # [B, S, 2]


def ego_future(poses, k, t, dt_frame=0.1):
    """Logged world poses -> ego-frame waypoints at times ``t[1:]`` from frame k.

    poses: [N, 3] (x, y, yaw). Returns ([S,2], valid[S] bool)."""
    x0, y0, yaw0 = poses[k]
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    out = np.zeros((len(t) - 1, 2)); ok = np.zeros(len(t) - 1, dtype=bool)
    for j, tt in enumerate(t[1:]):
        kk = k + int(round(tt / dt_frame))
        if kk >= len(poses):
            continue
        dx, dy = poses[kk, 0] - x0, poses[kk, 1] - y0
        out[j] = (c * dx - s * dy, s * dx + c * dy)
        ok[j] = True
    return out, ok


def gt_curvature0(poses, k, dt_frame=0.1):
    """GT yaw-rate / speed at frame k -> initial curvature (1/m), for arc()."""
    k = max(1, min(k, len(poses) - 2))
    dyaw = (poses[k + 1, 2] - poses[k - 1, 2] + np.pi) % (2 * np.pi) - np.pi
    ds = np.linalg.norm(poses[k + 1, :2] - poses[k - 1, :2])
    return float(dyaw / ds) if ds > 1e-6 else 0.0


def q(a, name):
    a = np.asarray(a, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"metric": name, "n": 0}
    return {"metric": name, "n": int(a.size), "mean": float(a.mean()),
            "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


#: Vertex turns below this are numerically zero: a banked constant-velocity path
#: carries +-1e-16 rad of float noise, and an undeadbanded sign test read a
#: 0.3678 "zig-zag" rate on a path whose mean |turn| is 0.000 deg. MEASURED
#: 2026-09-04; the real signal is ~1.2 deg, so 0.05 deg is 24x below it.
TURN_DEADBAND_DEG = 0.05


def sign_flip(turn, deadband_deg=TURN_DEADBAND_DEG):
    """Fraction of adjacent interior-vertex pairs whose turn changes sign,
    counting only pairs where BOTH turns exceed the deadband. Grid-invariant:
    resampling a smooth path onto a coarser grid does not create sign changes."""
    import numpy as _np
    t = _np.asarray(turn)
    live = _np.abs(t) > _np.radians(deadband_deg)
    both = live[..., :-1] & live[..., 1:]
    flip = (_np.sign(t[..., :-1]) * _np.sign(t[..., 1:]) < 0) & both
    n = both.sum()
    return float(flip.sum() / n) if n else 0.0
