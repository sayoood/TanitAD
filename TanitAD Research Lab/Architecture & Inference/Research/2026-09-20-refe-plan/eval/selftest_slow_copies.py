#!/usr/bin/env python3
"""Validity tests for refe/slow_copies.py::slow_copy -- the ONE implementation of a slowed copy (a trajectory's
PATH traversed `factor` times as fast), shared by the eval probe, the training targets and the on-policy labels.

Every check runs on the REAL function (must PASS) and on DELIBERATELY BROKEN variants, each of which must FAIL the
checks that exist to catch it -- a check no broken variant can fail is not a check:
  identity     factor 1.0 returns the input bit-for-bit (numpy float32 / float64, torch float32; real ep015 poses)
  on_polyline  every output point lies on the input polyline origin -> p_1 -> ... -> p_T (curved synthetic paths:
               <= 1e-9 m in float64; real ep015 poses in float32: <= 1e-5 m)
  speed        every output step whose two query times fall in ONE input segment moves exactly factor x that
               segment's speed (relative 1e-9); on a constant-speed line EVERY step is factor x v
  heading      the heading follows the path: on a constant-yaw-rate arc (heading = tangent = w t, stored WRAPPED)
               the copy's heading is w * factor * t_k, including across +-pi (a U-turn), to 1e-12 rad
  batch        vectorised over leading dims: slow_copy(batch)[i] == slow_copy(batch[i]) bit-for-bit; dtype, shape
               and type (numpy / torch) preserved; numpy and torch agree bit-for-bit
  domain       factor 0 and factor > 1 raise ValueError
Broken variants: SCALE_POSITIONS (x, y scaled by factor instead of time -- off the path on curves), NO_WRAP
(heading interpolated without wrapping -- unwinds through 0 at +-pi), NO_ORIGIN (the t = 0 origin omitted -- the
first step jumps to p_1 at full speed), NEAREST_SAMPLE (time snapped to a sample -- on the polyline but the wrong
speed and heading: why on_polyline alone is not enough).

    python eval/selftest_slow_copies.py            # prints ZZSELFTEST_OK or ZZSELFTEST_FAIL <what>
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "refe"))
import slow_copies as SC                 # noqa: E402  refe/slow_copies.py

DT = 0.2
T = 20
FACTORS = (0.3, 0.5, 0.75, 0.9)
REAL_DUMP = "D:/Projects/TanitAD/data/refe_navtest/proptable/sub200_ep015/stop_candidate_dump.npz"


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------------------------- broken variants
def scale_positions(traj, f, dt=DT):
    out = np.array(traj, dtype=np.float64, copy=True)
    out[..., :2] *= f
    return out.astype(np.asarray(traj).dtype)


def no_wrap(traj, f, dt=DT):
    """the real construction, but the heading interpolated WITHOUT wrapping"""
    x = np.asarray(traj, dtype=np.float64)
    src = np.concatenate([np.zeros(x.shape[:-2] + (1, 3)), x], axis=-2)
    out = np.empty_like(x)
    for k, (j, w) in enumerate(SC.slow_copy_plan(x.shape[-2], f, dt)):
        if w is None:
            out[..., k, :] = src[..., j, :]
        else:
            out[..., k, :] = (1 - w) * src[..., j - 1, :] + w * src[..., j, :]
    return out.astype(np.asarray(traj).dtype)


def no_origin(traj, f, dt=DT):
    """the t = 0 origin omitted: a query before the first sample is clamped to it"""
    x = np.asarray(traj, dtype=np.float64)
    Tn = x.shape[-2]
    out = np.empty_like(x)
    for k in range(Tn):
        tq = max(f * dt * (k + 1), dt)
        s = tq / dt - 1.0                                   # fractional index into x
        j0 = int(math.floor(s + 1e-9))
        w = s - j0
        if w < 1e-9 or j0 + 1 >= Tn:
            out[..., k, :] = x[..., min(j0, Tn - 1), :]
        else:
            out[..., k, :2] = (1 - w) * x[..., j0, :2] + w * x[..., j0 + 1, :2]
            out[..., k, 2] = wrap(x[..., j0, 2] + w * wrap(x[..., j0 + 1, 2] - x[..., j0, 2]))
    return out.astype(np.asarray(traj).dtype)


def nearest_sample(traj, f, dt=DT):
    """time snapped to the nearest earlier sample (origin included)"""
    x = np.asarray(traj, dtype=np.float64)
    src = np.concatenate([np.zeros(x.shape[:-2] + (1, 3)), x], axis=-2)
    out = np.empty_like(x)
    for k in range(x.shape[-2]):
        out[..., k, :] = src[..., int(math.floor(f * (k + 1) + 1e-9)), :]
    return out.astype(np.asarray(traj).dtype)


VARIANTS = {"REAL": SC.slow_copy, "SCALE_POSITIONS": scale_positions, "NO_WRAP": no_wrap,
            "NO_ORIGIN": no_origin, "NEAREST_SAMPLE": nearest_sample}
# the checks each broken variant must fail (it may fail others too)
MUST_FAIL = {"SCALE_POSITIONS": {"on_polyline", "heading"}, "NO_WRAP": {"heading"}, "NO_ORIGIN": {"speed"},
             "NEAREST_SAMPLE": {"speed", "heading"}}


# ---------------------------------------------------------------------------------------------- synthetic paths
def arc(v, w, h0_wrap=True):
    """constant speed v, yaw rate w, from the origin with heading 0; heading = tangent, stored WRAPPED"""
    t = DT * np.arange(1, T + 1)
    if abs(w) < 1e-12:
        xy = np.stack([v * t, np.zeros_like(t)], -1)
    else:
        R = v / w
        xy = np.stack([R * np.sin(w * t), R * (1 - np.cos(w * t))], -1)
    return np.concatenate([xy, wrap(w * t)[:, None]], -1), t


def curvy(rng, n):
    """random smooth paths: speed and yaw rate vary with time; heading = integrated yaw rate (wrapped)"""
    out = np.zeros((n, T, 3))
    for i in range(n):
        dt_f = DT / 20.0
        tt = np.arange(0, T * DT + 1e-9, dt_f)
        v = np.clip(rng.uniform(2, 15) + rng.uniform(-3, 3) * np.sin(rng.uniform(0.3, 1.5) * tt), 0.5, None)
        w = rng.uniform(-0.6, 0.6) + rng.uniform(-0.3, 0.3) * np.sin(rng.uniform(0.3, 2.0) * tt)
        th = np.concatenate([[0.0], np.cumsum(w[:-1] * dt_f)])
        x = np.concatenate([[0.0], np.cumsum(v[:-1] * np.cos(th[:-1]) * dt_f)])
        y = np.concatenate([[0.0], np.cumsum(v[:-1] * np.sin(th[:-1]) * dt_f)])
        idx = np.round(DT * np.arange(1, T + 1) / dt_f).astype(int)
        out[i] = np.stack([x[idx], y[idx], wrap(th[idx])], -1)
    return out


def polyline_distance(traj, q):
    """[..., T, 3] polyline (origin prepended) and [..., K, 3] points -> [..., K] distance of each point to it"""
    p = np.concatenate([np.zeros(traj.shape[:-2] + (1, 2)), traj[..., :2]], axis=-2).astype(np.float64)
    a, b = p[..., :-1, :], p[..., 1:, :]                                   # [..., T, 2]
    qq = q[..., :2].astype(np.float64)[..., :, None, :]                    # [..., K, 1, 2]
    ab = (b - a)[..., None, :, :]
    s = np.clip(((qq - a[..., None, :, :]) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-300), 0.0, 1.0)
    proj = a[..., None, :, :] + s[..., None] * ab
    return np.sqrt(((qq - proj) ** 2).sum(-1)).min(-1)


# ---------------------------------------------------------------------------------------------- checks
def check_on_polyline(fn, paths, bar):
    worst = max(float(polyline_distance(paths, fn(paths, f)).max()) for f in FACTORS)
    return worst <= bar, f"max distance to the input polyline {worst:.3e} m (bar {bar:g})"


def check_speed(fn, paths):
    """steps whose two query times lie in one input segment must move factor x that segment's speed"""
    src = np.concatenate([np.zeros(paths.shape[:-2] + (1, 3)), paths], axis=-2).astype(np.float64)
    seg = np.linalg.norm(np.diff(src[..., :2], axis=-2), axis=-1)            # [..., T] segment lengths
    worst, n_checked = 0.0, 0
    for f in FACTORS:
        out = np.concatenate([np.zeros(paths.shape[:-2] + (1, 3)), fn(paths, f).astype(np.float64)], axis=-2)
        step = np.linalg.norm(np.diff(out[..., :2], axis=-2), axis=-1)       # [..., T]
        for k in range(paths.shape[-2]):
            t1, t2 = f * DT * k, f * DT * (k + 1)
            j1, j2 = int(math.floor(t1 / DT + 1e-9)), int(math.ceil(t2 / DT - 1e-9)) - 1
            if j1 != j2:                                                      # straddles an input vertex
                continue
            want = f * seg[..., j1]
            rel = np.abs(step[..., k] - want) / np.maximum(want, 1e-12)
            worst = max(worst, float(rel.max()))
            n_checked += int(np.size(rel))
    # and a constant-speed straight line: EVERY step is factor x v
    line, _ = arc(12.0, 0.0)
    for f in FACTORS:
        o = np.concatenate([np.zeros((1, 3)), fn(line, f).astype(np.float64)], 0)
        st = np.linalg.norm(np.diff(o[:, :2], axis=0), axis=-1) / DT
        worst = max(worst, float(np.abs(st - f * 12.0).max() / (f * 12.0)))
    ok = worst <= 1e-9 and n_checked > 0
    return ok, f"max relative speed error {worst:.3e} over {n_checked} single-segment steps + a 12 m/s line (bar 1e-9)"


def check_heading(fn):
    worst = 0.0
    for v, w in ((10.0, 0.5), (5.0, 1.0), (8.0, -0.9)):     # the 1.0 and -0.9 rad/s arcs cross +-pi within 4 s
        path, t = arc(v, w)
        for f in FACTORS:
            h = fn(path, f)[:, 2].astype(np.float64)
            want = w * f * t
            worst = max(worst, float(np.abs(wrap(h - want)).max()))
    return worst <= 1e-12, f"max heading error vs the arc tangent w*f*t {worst:.3e} rad (bar 1e-12; arcs cross +-pi)"


def check_identity(fn, rng, real):
    ok, notes = True, []
    for dtype in (np.float32, np.float64):
        x = curvy(rng, 8).astype(dtype)
        y = fn(x, 1.0)
        good = isinstance(y, np.ndarray) and y.dtype == dtype and np.array_equal(y, x)
        ok &= good
        notes.append(f"{np.dtype(dtype).name} {'bit-identical' if good else 'DIFFERS'}")
    if real is not None:
        good = np.array_equal(fn(real, 1.0), real)
        ok &= good
        notes.append(f"real ep015 {real.shape} {'bit-identical' if good else 'DIFFERS'}")
    return ok, "factor 1.0: " + ", ".join(notes)


def check_batch(rng):
    x = curvy(rng, 6).astype(np.float32).reshape(2, 3, T, 3)
    ok, notes = True, []
    for f in FACTORS:
        yb = SC.slow_copy(x, f)
        each = np.stack([np.stack([SC.slow_copy(x[i, j], f) for j in range(3)]) for i in range(2)])
        ok &= bool(np.array_equal(yb, each) and yb.dtype == x.dtype and yb.shape == x.shape)
    notes.append("batch == per-trajectory bit-for-bit, dtype/shape kept" if ok else "BATCH MISMATCH")
    try:
        import torch
        for f in FACTORS:
            yt = SC.slow_copy(torch.from_numpy(x), f)
            same = isinstance(yt, torch.Tensor) and yt.dtype == torch.float32 and np.array_equal(yt.numpy(),
                                                                                              SC.slow_copy(x, f))
            ok &= bool(same)
        notes.append("torch == numpy bit-for-bit, torch in -> torch out" if ok else "TORCH MISMATCH")
    except ImportError:
        notes.append("torch not importable here: torch path NOT tested")
    return ok, "; ".join(notes)


def check_domain():
    bad = 0
    for f in (0.0, -0.5, 1.01, 2.0):
        try:
            SC.slow_copy(np.zeros((T, 3)), f)
        except ValueError:
            bad += 1
    return bad == 4, f"{bad}/4 out-of-domain factors refused"


def main() -> int:
    rng = np.random.default_rng(20260927)
    paths = curvy(rng, 64)
    real = None
    if os.path.exists(REAL_DUMP):
        real = np.load(REAL_DUMP)["traj"].astype(np.float32)               # [200, 64, 20, 3] REFe's own ep015 poses
    results = {}
    for name, fn in VARIANTS.items():
        r = {"on_polyline": check_on_polyline(fn, paths, 1e-9),
             "speed": check_speed(fn, paths),
             "heading": check_heading(fn)}
        if name == "REAL":
            r["identity"] = check_identity(fn, np.random.default_rng(7), real)
            if real is not None:
                worst = max(float(polyline_distance(real, fn(real, f)).max()) for f in (0.5, 0.75))
                r["on_polyline_real_ep015"] = (worst <= 1e-5, f"real poses (float32) max distance {worst:.3e} m "
                                                              f"(bar 1e-5), factors 0.5 and 0.75")
            r["batch"] = check_batch(np.random.default_rng(11))
            r["domain"] = check_domain()
        results[name] = r
    bad = []
    for name, r in results.items():
        failed = {k for k, (ok, _) in r.items() if not ok}
        for k, (ok, note) in r.items():
            print(f"  [{name:15s}] {'PASS' if ok else 'FAIL'} {k:22s} {note}")
        if name == "REAL" and failed:
            bad.append(f"REAL:{','.join(sorted(failed))}")
        if name != "REAL" and not MUST_FAIL[name] <= failed:
            bad.append(f"{name}_escaped:{','.join(sorted(MUST_FAIL[name] - failed))}")
    if real is None:
        print("  [NOTE] real ep015 dump not found: the real-pose checks did not run")
    print("ZZSELFTEST_OK" if not bad else "ZZSELFTEST_FAIL " + " ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
