#!/usr/bin/env python3
"""Tests for refe/field_loss.py (the proposal-level drivable-area field loss, SPEC_AUX_DRIVABLE_HEAD.md section 8, D-AUX-4).
Run:  python refe/test_field_loss.py   ->   ZZTEST_PASS <n>      (CPU only; prints the measured numbers and the mutation table)

PROOF STANDARD (CLAUDE.md, "built, tested and unreachable"): a check that cannot go RED is not a check.
  * CONTROLS: a fan entirely on the road at >= margin gives EXACTLY 0.0 loss and EXACTLY 0.0 gradient; a constant field gives a
    HAND-COMPUTED value (margin - s), so the zero is not a dead probe.
  * INDEPENDENT REFERENCE: a float64 numpy implementation built from `navsim_dac.corners_global` / `to_navsim` / `at_10hz`
    and the bilinear formula of `proto_label.raster_inside`, none of which share code with field_loss.py.
  * MUTATION: every mutation edits the module SOURCE (the snippet must occur exactly once, else the test errors: a vacuous
    mutation proves nothing) and the suite must then report >= 1 failing check.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types
from functools import lru_cache

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import navsim_dac as ND  # noqa: E402  (numpy-only at import; shapely / nuplan are imported lazily inside its map functions)

SRC_PATH = os.path.join(HERE, "field_loss.py")
LABELS = os.path.join(HERE, "..", "raw", "2026-10-11-aux-drivable", "labels_proto.npz")
RESULTS = os.path.join(HERE, "..", "raw", "2026-10-11-aux-drivable", "proto_label_results.json")

NX, NY, X0, Y0, RES, UNIT = 160, 128, -8.0, -32.0, 0.5, 0.1
INFO: dict = {}


def load(src: str, name: str):
    m = types.ModuleType(name)
    m.__file__ = SRC_PATH
    sys.modules[name] = m
    exec(compile(src, SRC_PATH, "exec"), m.__dict__)
    return m


# ============================================================================ fixtures (numpy, deterministic)
def centres():
    x = X0 + (np.arange(NX) + 0.5) * RES
    y = Y0 + (np.arange(NY) + 0.5) * RES
    return np.meshgrid(x, y, indexing="ij")                                   # [NX, NY]


def to_i8(s_m):
    return np.round(np.clip(s_m, -12.7, 12.7) / UNIT).astype(np.int8)


def road_sdf(half=3.5):
    gx, gy = centres()
    return to_i8(half - np.abs(gy))                                          # straight road along x, positive inside


def wavy_sdf():
    gx, gy = centres()
    return to_i8(5.0 * np.sin(gx / 9.0) * np.cos(gy / 7.0) + 0.04 * (gx - 20.0) - 0.1 * gy)


def const_sdf(v):
    return np.full((NX, NY), v, dtype=np.int8)


def halfplane_left_sdf():
    gx, gy = centres()
    return to_i8(np.where(gy >= 0.0, 10.0, -10.0))                           # y >= 0 (left) is "road"


def arcs(rng, B, K, vmin, vmax, kmax, ymax):
    t = 0.2 * np.arange(1, 21)
    v = rng.uniform(vmin, vmax, (B, K, 1))
    kap = rng.uniform(-kmax, kmax, (B, K, 1))
    y0 = rng.uniform(-ymax, ymax, (B, K, 1))
    s = v * t
    kk = np.where(np.abs(kap) < 1e-6, 1e-6, kap)
    x = np.sin(kk * s) / kk
    y = (1 - np.cos(kk * s)) / kk + y0 * (s / s[..., -1:])
    return np.stack([x, y, kk * s], -1)


def straight(y0, v, psi=0.0):
    t = 0.2 * np.arange(1, 21)
    x = v * t * math.cos(psi)
    y = y0 + v * t * math.sin(psi)
    return np.stack([x, y, np.full_like(t, psi)], -1)


@lru_cache(maxsize=None)
def fixtures():
    rng = np.random.default_rng(20261012)
    F = {}
    F["wavy"] = (arcs(rng, 2, 64, 2.0, 18.0, 0.05, 2.5), np.stack([wavy_sdf(), wavy_sdf()[::-1].copy()]))
    # reversing fan with the heading wrapped to (-pi, pi]: consecutive poses straddle +-pi
    t = 0.2 * np.arange(1, 21)
    rev = np.zeros((1, 64, 20, 3))
    for k in range(64):
        yaw = math.pi + rng.uniform(-0.06, 0.06, 20)
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi                      # wrapped: some are -3.1, some +3.1
        rev[0, k, :, 0] = -1.7 * t * (0.4 + 0.6 * k / 63.0)
        rev[0, k, :, 1] = rng.uniform(-1.5, 1.5) * (t / 4.0)
        rev[0, k, :, 2] = yaw
    F["wrap"] = (rev, wavy_sdf()[None])
    F["navsim"] = (arcs(rng, 1, 64, 4.0, 18.0, 0.05, 2.0), wavy_sdf()[None])
    return F


@lru_cache(maxsize=None)
def real_labels():
    if not os.path.exists(LABELS):
        return None
    return np.load(LABELS)["sdf_int8"]


TARGETS = "E:/Projects/TanitAD/data/refe_navtrain/r0_shard0/targets_rank0.jsonl"          # rank-0 teacher rows (dev box); the check skips if absent


@lru_cache(maxsize=None)
def teacher_paths():
    """The 20 labelled samples' REAL teacher paths ([20, 1, 20, 3], rear-axle frame) with their real SDFs; None if either file is absent."""
    if not os.path.exists(TARGETS) or real_labels() is None:
        return None
    z = np.load(LABELS)
    want = {(str(a), str(b)): i for i, (a, b) in enumerate(zip(z["logs"], z["tokens"]))}
    found = {}
    with open(TARGETS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = (str(r.get("log_name")), str(r.get("token")))
            if k in want and k not in found and r.get("rank", 0) == 0:
                found[k] = r
            if len(found) == len(want):
                break
    if len(found) != len(want):
        return None
    idx = [want[k] for k in found]
    return np.array([found[k]["traj"] for k in found], dtype=np.float64)[:, None], z["sdf_int8"][idx]


# ============================================================================ independent float64 reference
def ref_poses(p20, interp):
    if interp == "navsim2hz":
        return ND.at_10hz(ND.to_navsim(p20))
    full = np.vstack([np.zeros((1, 3)), p20])
    hd = np.unwrap(full[:, 2])
    tk, tq = 0.2 * np.arange(21), 0.1 * np.arange(41)
    return np.stack([np.interp(tq, tk, full[:, 0]), np.interp(tq, tk, full[:, 1]), np.interp(tq, tk, hd)], -1)


def ref_sample(sdf_i8, pts):
    """proto_label.raster_inside's bilinear branch, float64; pts [...,2] -> (s_m, in_extent)."""
    fx = (pts[..., 0] - X0) / RES - 0.5
    fy = (pts[..., 1] - Y0) / RES - 0.5
    in_ext = (fx >= 0) & (fx <= NX - 1) & (fy >= 0) & (fy <= NY - 1)
    x0 = np.clip(np.floor(fx).astype(int), 0, NX - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, NY - 2)
    wx, wy = np.clip(fx - x0, 0, 1), np.clip(fy - y0, 0, 1)
    s = sdf_i8.astype(np.float64) * UNIT
    bil = ((1 - wx) * (1 - wy) * s[x0, y0] + wx * (1 - wy) * s[x0 + 1, y0] + (1 - wx) * wy * s[x0, y0 + 1] + wx * wy * s[x0 + 1, y0 + 1])
    return bil, in_ext


def ref_loss(traj, sdf, margin=0.25, valid=None, interp="5hz", per_prop=False):
    traj = np.asarray(traj, np.float64)
    B, K = traj.shape[:2]
    tot, nv = 0.0, 0
    pp = np.zeros((B, K))
    for b in range(B):
        if valid is not None and not valid[b]:
            continue
        nv += 1
        for k in range(K):
            cr = ND.corners_global(ref_poses(traj[b, k], interp)[1:], 0.0, 0.0, 0.0)      # t = 0.1 ... 4.0
            s, ext = ref_sample(sdf[b], cr)
            pp[b, k] = (np.maximum(margin - s, 0.0) * ext).mean()
    loss = pp.sum() / (max(nv, 1) * K)
    return (loss, pp) if per_prop else loss


@lru_cache(maxsize=None)
def ref_cached(name, interp="5hz"):
    traj, sdf = fixtures()[name]
    return ref_loss(traj, sdf, interp=interp)


def tt(a, dtype=torch.float64, grad=False):
    x = torch.from_numpy(np.ascontiguousarray(a)).to(dtype)
    return x.requires_grad_(True) if grad else x


def i8(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(torch.int8)


# ============================================================================ the suite
def suite(M) -> tuple:
    """The checks, against module `M`. Returns (names of the FAILING checks (empty = all green), number of checks run)."""
    bad: list = []
    count = [0]

    def t(name, fn):
        count[0] += 1
        try:
            ok = bool(fn())
        except Exception:                                                       # noqa: BLE001 -- an exception IS a failed check
            ok = False
        if not ok:
            bad.append(name)

    FX = fixtures()

    def fl(x):
        return float(x.detach()) if torch.is_tensor(x) else float(x)

    def close(a, b, rtol, atol=0.0):
        return abs(fl(a) - fl(b)) <= atol + rtol * abs(fl(b))

    # ---- (R) agreement with the independent reference ------------------------------------------------------------------
    def r_wavy_f64():
        traj, sdf = FX["wavy"]
        return close(M.field_loss(tt(traj), i8(sdf)), ref_cached("wavy"), 1e-9) and ref_cached("wavy") > 0.05

    def r_wavy_f32():
        traj, sdf = FX["wavy"]
        return close(M.field_loss(tt(traj, torch.float32), i8(sdf)), ref_cached("wavy"), 2e-4)

    def r_wrap():
        traj, sdf = FX["wrap"]
        return close(M.field_loss(tt(traj), i8(sdf)), ref_cached("wrap"), 1e-9) and ref_cached("wrap") > 0.05

    def r_navsim():
        traj, sdf = FX["navsim"]
        a = float(M.field_loss(tt(traj), i8(sdf), interp="navsim2hz"))
        return close(a, ref_cached("navsim", "navsim2hz"), 1e-9) and not close(a, ref_cached("navsim", "5hz"), 1e-4)

    def r_real():
        lab = real_labels()
        if lab is None:
            return True
        rng = np.random.default_rng(7)
        traj = arcs(rng, 6, 64, 2.0, 16.0, 0.04, 2.0)
        sd = lab[:6]
        return close(M.field_loss(tt(traj), i8(sd)), ref_loss(traj, sd), 1e-9)

    def r_parts():
        traj, sdf = FX["wavy"]
        loss, parts = M.field_loss(tt(traj), i8(sdf), return_parts=True)
        _, pp = ref_loss(traj, sdf, per_prop=True)
        return (tuple(parts["per_proposal"].shape) == (2, 64) and float(parts["per_proposal"].sub(torch.from_numpy(pp)).abs().max()) < 1e-9
                and close(parts["per_proposal"].mean(), loss, 1e-9) and 0.0 < float(parts["violating_frac"]) < 1.0)

    def r_teacher():
        """FRAME SANITY on REAL data: 20 real teacher paths against their own real labels. The teacher leaves the drivable area on
        ~5 % of paths (REVIEW_7), so >= 17 of 20 must have a non-negative minimum corner SDF; the y-MIRRORED paths must be clearly worse."""
        tp = teacher_paths()
        if tp is None:
            INFO["r8"] = "skipped (teacher rows or labels absent)"
            return True
        traj, sdf = tp
        loss, parts = M.field_loss(tt(traj, torch.float32), i8(sdf), return_parts=True)
        mir = traj.copy()
        mir[..., 1] *= -1
        mir[..., 2] *= -1
        loss2, parts2 = M.field_loss(tt(mir, torch.float32), i8(sdf), return_parts=True)
        n_off, n_off_mir = int((parts["min_sdf"] < 0).sum()), int((parts2["min_sdf"] < 0).sum())
        INFO["r8"] = f"teacher paths off-road {n_off}/20, mirrored {n_off_mir}/20, loss {float(loss):.4f} vs mirrored {float(loss2):.4f}"
        return (20 - n_off) >= 17 and n_off_mir >= n_off + 3 and float(loss2) > 20.0 * float(loss) and close(loss, ref_loss(traj, sdf), 2e-4)

    # ---- (H) hand-computed values (no reference code involved) ----------------------------------------------------------
    def fan_in_extent():
        rng = np.random.default_rng(3)
        return arcs(rng, 1, 64, 2.0, 10.0, 0.03, 3.0)                           # x <= ~40 m, |y| <= ~5 m: every corner in extent

    def h_constant():
        tr = fan_in_extent()
        ok = True
        for v, margin, want in ((-10, 0.25, 1.25), (2, 0.25, 0.05), (3, 0.25, 0.0), (127, 0.25, 0.0), (0, 0.25, 0.25), (2, 0.5, 0.3), (-10, 0.0, 1.0)):
            got = float(M.field_loss(tt(tr), i8(const_sdf(v)[None]), margin=margin))
            ok &= abs(got - want) < 1e-9
        return ok

    def h_half_extent():
        tr = fan_in_extent()
        far = tr.copy()
        far[..., 0] += 400.0        # x >= 400: out of extent for all 40 steps (the t = 0.1 step is the MIDPOINT of the origin and pose 0, so x >= 200)
        both = np.concatenate([tr[:, :32], far[:, 32:]], axis=1)
        got = float(M.field_loss(tt(both), i8(const_sdf(-10)[None])))
        return abs(got - 0.5 * 1.25) < 1e-9

    def h_orientation():
        left, right = straight(3.0, 8.0)[None, None], straight(-3.0, 8.0)[None, None]
        lf, rt = np.repeat(left, 64, 1), np.repeat(right, 64, 1)
        sd = i8(halfplane_left_sdf()[None])
        lg = float(M.field_loss(tt(lf), sd))
        rg = float(M.field_loss(tt(rt), sd))
        return lg == 0.0 and abs(rg - 10.25) < 1e-9

    def h_extent_edges():
        sd = i8(const_sdf(50)[None])
        pts = torch.tensor([[[-7.74, 0.0], [-7.76, 0.0], [71.74, 0.0], [71.76, 0.0], [0.0, -31.74], [0.0, -31.76], [0.0, 31.74], [0.0, 31.76]]], dtype=torch.float64)
        _, ext = M.sample_sdf(sd, pts)
        return ext.tolist()[0] == [True, False, True, False, True, False, True, False]

    def h_step_edges():
        # a step edge between cell centres is a linear ramp over one cell: x-edge at 20.0 m (centres 19.75 | 20.25), y-edge at 10.0 m
        # (centres 9.75 | 10.25), values -10 m | +10 m. HAND VALUES: s = -10 + 20 * (p - lo) / 0.5.
        gx, gy = centres()
        sx, sy = i8(to_i8(np.where(gx >= 20.0, 10.0, -10.0))[None]), i8(to_i8(np.where(gy >= 10.0, 10.0, -10.0))[None])
        px = torch.tensor([[[20.1, 5.0], [19.9, -7.0], [19.75, 0.0], [20.25, 0.0], [40.0, 3.0]]], dtype=torch.float64)
        py = torch.tensor([[[5.0, 10.1], [30.0, 9.9], [5.0, 9.75], [5.0, 10.25], [8.0, -20.0]]], dtype=torch.float64)
        vx, _ = M.sample_sdf(sx, px)
        vy, _ = M.sample_sdf(sy, py)
        return (np.allclose(vx.numpy(), [[4.0, -4.0, -10.0, 10.0, 10.0]], atol=1e-9)
                and np.allclose(vy.numpy(), [[4.0, -4.0, -10.0, 10.0, -10.0]], atol=1e-9))

    # ---- (C) controls: exactly 0.0 loss and exactly 0.0 gradient -----------------------------------------------------------
    def clean_fan():
        rng = np.random.default_rng(11)
        tr = np.stack([straight(rng.uniform(-0.8, 0.8), rng.uniform(3.0, 17.0), rng.uniform(-0.015, 0.015)) for _ in range(64)])[None]
        return tr

    def c_zero():
        tr = tt(clean_fan(), grad=True)
        loss, parts = M.field_loss(tr, i8(road_sdf()[None]), return_parts=True)
        ok = float(loss.detach()) == 0.0 and float(parts["min_sdf"].min()) >= 0.25 and float(parts["violating_frac"]) == 0.0
        loss.backward()
        return ok and tr.grad is not None and bool((tr.grad == 0).all())

    def c_probe_live():
        tr0 = clean_fan().copy()
        tr0[0, 5, :, 1] += np.linspace(0.0, 4.0, 20)                              # ONE proposal drifts across the left edge
        tr = tt(tr0, grad=True)
        loss = M.field_loss(tr, i8(road_sdf()[None]))
        loss.backward()
        return float(loss.detach()) > 0.0 and float(tr.grad.abs().max()) > 0.0 and float(tr.grad[0, 5].abs().sum()) > 0.0

    # ---- (D) gradient direction: toward the road, on ALL proposals ---------------------------------------------------------
    def crossing_fan():
        tt_ = 0.2 * np.arange(1, 21)
        tr = np.zeros((1, 64, 20, 3))
        side = np.zeros(64)
        for k in range(64):
            if k % 4 == 1:                                                         # left crossing, drifting to y = d
                d, sgn = 4.0 + 0.2 * (k % 7), 1.0
            elif k % 4 == 3:                                                       # right crossing
                d, sgn = 4.0 + 0.2 * (k % 5), -1.0
            else:
                d, sgn = 0.5 * math.sin(k), 0.0                                    # clean, centred
            side[k] = sgn
            y = sgn * d * (tt_ / 4.0) ** 2 if sgn else d * np.ones_like(tt_) * (tt_ / 4.0)
            dy = np.gradient(y, tt_)
            tr[0, k] = np.stack([10.0 * tt_, y, np.arctan2(dy, 10.0)], -1)
        return tr, side

    def d_direction():
        tr0, side = crossing_fan()
        tr = tt(tr0, grad=True)
        loss = M.field_loss(tr, i8(road_sdf()[None]))
        loss.backward()
        gy = tr.grad[0, :, :, 1].sum(-1).numpy()                                    # d loss / d (all y of proposal k)
        gabs = tr.grad[0].abs().sum((-1, -2)).numpy()
        ok = bool((gy[side > 0] > 0).all()) and bool((gy[side < 0] < 0).all())     # increasing y worsens a left crossing; right: the opposite
        ok &= bool((gabs[side != 0] > 0).all()) and bool((gabs[side == 0] == 0.0).all())      # EVERY crossing proposal gets gradient, every clean one exactly 0
        step = tr.grad / tr.grad.abs().max()
        tr2 = (tr.detach() - 0.05 * step)
        before = M.footprint_corners(M.poses_10hz(tr.detach())[..., 1:, :])[..., 1]  # [1,64,40,4] corner y
        after = M.footprint_corners(M.poses_10hz(tr2)[..., 1:, :])[..., 1]
        left, right = torch.from_numpy(side > 0), torch.from_numpy(side < 0)
        ok &= bool(after[0, left].amax() < before[0, left].amax()) and bool(after[0, right].amin() > before[0, right].amin())
        ok &= float(M.field_loss(tr2, i8(road_sdf()[None]))) < float(loss.detach())
        return ok

    # ---- (G) finite differences ---------------------------------------------------------------------------------------------
    def g_fd():
        rng = np.random.default_rng(5)
        traj = arcs(rng, 1, 3, 3.0, 12.0, 0.04, 2.0)
        sdf = i8(wavy_sdf()[None])
        tr = tt(traj, grad=True)
        M.field_loss(tr, sdf).backward()
        an = tr.grad.numpy().ravel()
        num = np.zeros_like(an)
        eps = 1e-6
        flat = traj.ravel().copy()
        for i in range(flat.size):
            hi, lo = flat.copy(), flat.copy()
            hi[i] += eps
            lo[i] -= eps
            num[i] = (float(M.field_loss(tt(hi.reshape(traj.shape)), sdf)) - float(M.field_loss(tt(lo.reshape(traj.shape)), sdf))) / (2 * eps)
        INFO["fd_max_abs_err"] = float(np.abs(an - num).max())
        INFO["fd_max_abs_grad"] = float(np.abs(an).max())
        return np.abs(an - num).max() < 1e-6 and np.abs(an).max() > 1e-4

    # ---- (E) out-of-extent corners carry exactly nothing --------------------------------------------------------------------
    def e_all_out():
        far = fan_in_extent().copy()
        far[..., 0] += 400.0
        tr = tt(far, grad=True)
        loss = M.field_loss(tr, i8(const_sdf(-127)[None]))
        loss.backward()
        return float(loss.detach()) == 0.0 and bool((tr.grad == 0).all())

    def partial_fan():
        part = fan_in_extent().copy()
        part[:, :, 11:, 0] = 100.0 + np.arange(9)                                  # poses 11..19 far out of extent; 0..9 in
        return part

    def r_partial():
        part = partial_fan()
        sdf = i8(wavy_sdf()[None])
        got = M.field_loss(tt(part), sdf)
        return close(got, ref_loss(part, np.asarray(sdf.numpy())), 1e-9) and float(got) > 0.0

    def e_partial():
        tr = tt(partial_fan(), grad=True)
        loss = M.field_loss(tr, i8(wavy_sdf()[None]))
        loss.backward()
        # poses n >= 12 touch only steps whose corners are all out of extent: exactly 0.0; poses 0..9 are in extent and carry gradient
        return float(loss.detach()) > 0.0 and bool((tr.grad[:, :, 12:] == 0).all()) and float(tr.grad[:, :, :10].abs().sum()) > 0.0

    # ---- (V) valid mask, batch, input guards ---------------------------------------------------------------------------------
    def v_valid():
        traj, sdf = FX["wavy"]
        tr = tt(traj, grad=True)
        valid = torch.tensor([True, False])
        loss = M.field_loss(tr, i8(sdf), valid=valid)
        loss.backward()
        one = M.field_loss(tt(traj[:1]), i8(sdf[:1]))
        none = M.field_loss(tt(traj), i8(sdf), valid=torch.tensor([False, False]))
        return close(loss, one, 1e-9) and bool((tr.grad[1] == 0).all()) and float(tr.grad[0].abs().sum()) > 0 and float(none) == 0.0

    def v_batch():
        traj, sdf = FX["wavy"]
        both = float(M.field_loss(tt(traj), i8(sdf)))
        each = 0.5 * (float(M.field_loss(tt(traj[:1]), i8(sdf[:1]))) + float(M.field_loss(tt(traj[1:]), i8(sdf[1:]))))
        return close(both, each, 1e-9)

    def v_guards():
        traj, sdf = FX["wavy"]
        raised = 0
        for bad_args in ((tt(traj), tt(sdf, torch.float32)), (tt(traj[:, :, :19]), i8(sdf)), (tt(traj), i8(sdf[:, :100]))):
            try:
                M.field_loss(*bad_args)
            except (TypeError, ValueError):
                raised += 1
        return raised == 3

    # ---- (F) footprint and time base vs navsim_dac --------------------------------------------------------------------------
    def f_corners():
        rng = np.random.default_rng(1)
        rel = np.stack([rng.uniform(-8, 70, (200, 41)), rng.uniform(-30, 30, (200, 41)), rng.uniform(-math.pi, math.pi, (200, 41))], -1)
        want = ND.corners_global(rel, 0.0, 0.0, 0.0)
        d64 = float(np.abs(M.footprint_corners(tt(rel)).numpy() - want).max())
        d32 = float(np.abs(M.footprint_corners(tt(rel, torch.float32)).numpy() - want).max())
        INFO["corner_dev_f64"], INFO["corner_dev_f32"] = d64, d32
        consts = (M.HALF_LENGTH == ND.HALF_LENGTH and M.HALF_WIDTH == ND.HALF_WIDTH and M.REAR_AXLE_TO_CENTER == ND.REAR_AXLE_TO_CENTER)
        return d64 < 1e-12 and d32 < 1e-4 and consts

    def f_poses_5hz():
        rng = np.random.default_rng(2)
        tr = np.concatenate([arcs(rng, 20, 1, 2.0, 18.0, 0.05, 2.0)[:, 0], FX["wrap"][0][0, :20]])
        got = M.poses_10hz(tt(tr)).numpy()
        want = np.stack([ref_poses(p, "5hz") for p in tr])
        d = np.abs(got[..., :2] - want[..., :2]).max()
        dh = np.abs(np.cos(got[..., 2]) - np.cos(want[..., 2])).max() + np.abs(np.sin(got[..., 2]) - np.sin(want[..., 2])).max()
        INFO["poses_5hz_dev"] = float(max(d, dh))
        return d < 1e-12 and dh < 1e-12 and got.shape[-2] == 41

    def f_poses_hand():
        # constant speed 7 m/s on a straight line: the 10 Hz x is 0.7 * q (q = 0..40) EXACTLY; a pose pair (yaw 3.0, -3.0) rad
        # straddles the +-pi cut, so its midpoint heading is pi (cos = -1), not 0 (the unwrapped mean)
        line = straight(0.0, 7.0)[None, None]
        x = M.poses_10hz(tt(line))[0, 0, :, 0].numpy()
        ok = np.allclose(x, 0.7 * np.arange(41), atol=1e-12)
        y2 = np.zeros((1, 1, 20, 3))
        y2[0, 0, :, 2] = 3.0
        y2[0, 0, 1:2, 2] = -3.0                                                  # poses 0 and 1 have yaw 3.0 and -3.0
        h = M.poses_10hz(tt(y2))[0, 0, :, 2]
        return ok and abs(math.cos(float(h[3])) + 1.0) < 1e-12                   # q = 3 is the midpoint of poses 0 and 1

    def f_poses_navsim():
        rng = np.random.default_rng(4)
        tr = np.concatenate([arcs(rng, 40, 1, 2.0, 18.0, 0.05, 2.0)[:, 0], FX["wrap"][0][0, :20]])
        got = M.poses_10hz_navsim(tt(tr)).numpy()
        want = np.stack([ND.at_10hz(ND.to_navsim(p)) for p in tr])
        d = float(np.abs(got[..., :2] - want[..., :2]).max())
        dh = float(np.abs(np.cos(got[..., 2]) - np.cos(want[..., 2])).max() + np.abs(np.sin(got[..., 2]) - np.sin(want[..., 2])).max())
        INFO["poses_navsim_dev"] = max(d, dh)
        return d < 1e-9 and dh < 1e-9

    def f_grid_constants():
        ok = (M.X0, M.X1, M.Y0, M.Y1, M.RES, M.NX, M.NY, M.SDF_UNIT_M, M.SDF_CLIP_M) == (-8.0, 72.0, -32.0, 32.0, 0.5, 160, 128, 0.1, 12.7)
        if os.path.exists(RESULTS):
            r = json.load(open(RESULTS, encoding="utf-8"))["summary"]["raster"]
            ok &= (r["x0"], r["x1"], r["y0"], r["y1"], r["res_m"], r["nx"], r["ny"], r["sdf_unit_m"], r["sdf_clip_m"]) == \
                  (M.X0, M.X1, M.Y0, M.Y1, M.RES, M.NX, M.NY, M.SDF_UNIT_M, M.SDF_CLIP_M)
        return ok and (M.N_SRC, M.SRC_DT, M.N_STEPS, M.K_PROPOSALS) == (20, 0.2, 40, 64)

    for name, fn in (("R1 matches independent ref (wavy, f64)", r_wavy_f64), ("R2 matches independent ref (wavy, f32)", r_wavy_f32),
                     ("R3 yaw-wrap crossing +-pi", r_wrap), ("R4 navsim2hz chain matches ref, differs from 5hz", r_navsim),
                     ("R5 real int8 labels vs ref", r_real), ("R6 per-proposal breakdown", r_parts),
                     ("H1 constant field: (margin - s) by hand", h_constant), ("H2 out-of-extent half: denominators", h_half_extent),
                     ("H3 axis orientation (half-plane)", h_orientation), ("H4 extent edges", h_extent_edges),
                     ("C1 clean fan: loss==0.0 AND grad==0.0", c_zero), ("C2 probe is live (one proposal crosses)", c_probe_live),
                     ("D1 gradient direction + all proposals + descent", d_direction), ("G1 finite differences", g_fd),
                     ("E1 all out-of-extent: 0.0 / 0.0", e_all_out), ("R7 partial out-of-extent value vs ref", r_partial), ("R8 real teacher paths vs real labels: frame sanity", r_teacher),
                     ("E2 partial out-of-extent: grad exactly 0 beyond", e_partial),
                     ("V1 valid mask", v_valid), ("V2 batch mean", v_batch), ("V3 input guards (int8 only)", v_guards),
                     ("F1 corners == navsim_dac.corners_global", f_corners), ("F2 5hz->10hz poses == np.interp ref", f_poses_5hz),
                     ("F3 navsim2hz poses == to_navsim+at_10hz", f_poses_navsim),
                     ("F5 up-sampling by hand (line x = 0.7 q; +-pi midpoint)", f_poses_hand), ("H5 step-edge ramp by hand (x and y)", h_step_edges), ("F4 grid constants == label manifest", f_grid_constants)):
        t(name, fn)
    return bad, count[0]


# (name, required-term, old snippet (must occur exactly once), new snippet)
MUTATIONS = [
    ("margin sign flipped", "margin", "hinge = F.relu(margin - s)", "hinge = F.relu(-margin - s)"),
    ("x/y swapped in the sampling grid", "grid", "grid = torch.stack([gx, gy], dim=-1)", "grid = torch.stack([gy, gx], dim=-1)"),
    ("align_corners flipped", "grid", "align_corners=False", "align_corners=True"),
    ("upsampling dropped (5 Hz poses repeated)", "upsample", "mid = torch.cat([mid_xy, mid_h], dim=-1)", "mid = b"),
    ("loss on the best-matching proposal only", "all-64", "per_prop = (hinge * vm[:, None, None, None]).mean(dim=(2, 3))",
     "per_prop = (hinge * vm[:, None, None, None]).mean(dim=(2, 3)) * F.one_hot(traj[..., 1].abs().mean(-1).argmin(1), K).to(hinge.dtype)"),
    ("int8 unit (0.1 m) forgotten", "unit", "img = (sdf_int8.to(pts.dtype) * SDF_UNIT_M)", "img = (sdf_int8.to(pts.dtype))"),
    ("hinge dropped (linear)", "hinge", "hinge = F.relu(margin - s)", "hinge = margin - s"),
    ("out-of-extent mask dropped", "extent", "hinge = hinge * in_ext.to(hinge.dtype)", "hinge = hinge"),
    ("yaw wrap dropped", "wrap", "return torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi", "return d"),
    ("poses detached (no gradient path)", "grad", "poses = poses[..., 1:, :]", "poses = poses[..., 1:, :].detach()"),
    ("valid mask ignored", "valid", "vm = torch.as_tensor(valid, device=hinge.device).reshape(B).to(hinge.dtype)", "vm = torch.ones(B, dtype=hinge.dtype, device=hinge.device)"),
    ("rear-axle offset sign (centre vs axle)", "footprint", "cx, cy = x + REAR_AXLE_TO_CENTER * ch, y + REAR_AXLE_TO_CENTER * sh",
     "cx, cy = x - REAR_AXLE_TO_CENTER * ch, y - REAR_AXLE_TO_CENTER * sh"),
    ("interp switch ignored", "interp", 'poses = poses_10hz(traj) if interp == "5hz" else poses_10hz_navsim(traj)', "poses = poses_10hz(traj)"),
    ("time base off by one (t = 0 ... 3.9)", "steps", "poses = poses[..., 1:, :]", "poses = poses[..., :-1, :]"),
]
REQUIRED = {"margin", "grid", "upsample", "all-64", "unit"}                     # the PI's list: sign, x/y swap, align_corners, upsampling, best-only, int8 unit


def main() -> int:
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    src = open(SRC_PATH, encoding="utf-8").read()
    base = load(src, "field_loss_ref")
    failing, n_checks = suite(base)
    assert not failing, f"unmutated suite has failing checks: {failing}"
    info = dict(INFO)                                                           # snapshot: the mutants overwrite INFO
    n = n_checks
    print(f"unmutated suite: {n_checks} checks GREEN   (INFO {json.dumps({k: (float(f'{v:.3g}') if not isinstance(v, str) else v) for k, v in info.items()})})")
    print(f"{'mutation':44s} {'term':10s} caught-by (failing checks)")
    for i, (name, term, old, new) in enumerate(MUTATIONS):
        cnt = src.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times (must be exactly 1) -- a vacuous mutation proves nothing"
        try:
            mod = load(src.replace(old, new), f"field_loss_mut{i}")
        except SyntaxError as e:
            raise AssertionError(f"mutation {name} does not compile: {e}")
        try:
            fails, _ = suite(mod)
        except Exception as e:                                                  # noqa: BLE001 -- a mutant that breaks the suite's own setup is RED, and said so
            fails = [f"suite setup raised {type(e).__name__}: {str(e)[:50]}"]
        assert fails, f"mutation {name} ({term}) stayed GREEN: the suite cannot see it"
        hand = [f.split(' ')[0] for f in fails if not f.startswith("R")]          # checks that do not use the end-to-end numpy reference
        print(f"{name:44s} {term:10s} {len(fails):2d} failing; outside the end-to-end reference (R*): {', '.join(hand) if hand else '-- none (reference only)'}")
        if term in REQUIRED:
            assert hand, f"required mutation {name}: only the end-to-end reference (R*) saw it; add a hand-computed check"
        n += 1
    assert {t for _, t, _, _ in MUTATIONS} >= REQUIRED
    n += 1
    # the same-breath report of the measured agreement and of how much the up-sampling route matters
    rng = np.random.default_rng(99)
    tr = torch.from_numpy(arcs(rng, 1, 256, 5.0, 18.0, 0.05, 0.0)).double()
    c5 = base.footprint_corners(base.poses_10hz(tr)[..., 1:, :])
    cn = base.footprint_corners(base.poses_10hz_navsim(tr)[..., 1:, :])
    dev = (c5 - cn).norm(dim=-1)
    print(f"corner agreement vs navsim_dac.corners_global: max dev f64 {info['corner_dev_f64']:.2e} m, f32 {info['corner_dev_f32']:.2e} m")
    print(f"up-sampling route (5hz vs navsim2hz) on 256 arcs v 5-18 m/s, |kappa| <= 0.05: corner deviation max {float(dev.max()):.3f} m, "
          f"p99 {float(dev.flatten().quantile(0.99)):.3f} m, mean {float(dev.mean()):.4f} m  (margin is 0.25 m)")
    print(f"ZZTEST_PASS {n}  ({n_checks} checks green on the unmutated module + {len(MUTATIONS)} RED mutations + 1 coverage of the PI's six)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
