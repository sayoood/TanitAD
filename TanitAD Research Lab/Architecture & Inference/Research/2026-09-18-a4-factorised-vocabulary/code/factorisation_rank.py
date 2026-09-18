"""A4 — is REF-C's shipped trajectory vocabulary secretly a PRODUCT SET?

WHAT THIS ANSWERS, AND WHY IT NEEDS NO GPU AND NO CORPUS
============================================================================
SparseDriveV2 (arXiv 2603.29163, the claim BACKLOG A4 calls "refuted 0-3")
proposes enumerating ``P`` geometric PATHS x ``V`` VELOCITY PROFILES instead of
``N`` whole trajectories, so that ``P + V`` stored objects span ``P * V``
effective trajectories.

Our shipped vocabulary (refcv4/refcv5, ``refc_anchors_6s_b1train_128.pt``) is a
JOINT set: 128 whole ego-frame trajectories at 8 time slots. If those 128
trajectories in fact contain only ``P`` distinct path SHAPES and ``V`` distinct
SPEED profiles, then the joint enumeration is storing 128 of the ``P * V``
combinations its own parts already describe, and the factorised form is a free
density multiplier on the SAME bytes.

That is a property OF THE BANKED TENSOR. It needs no ground truth, no episode
cache and no GPU, which is why it can be measured today.

THE DECOMPOSITION
----------------------------------------------------------------------------
For anchor ``i`` with waypoints ``p_i[0..S-1]`` in the ego frame, prepend the
origin to get a polyline of ``S + 1`` points. Then

  * ``s_i[k]``  = cumulative arc length to knot ``k``           (metres)
  * SPEED PROFILE  ``sigma_i = s_i[1..S]``  -- every metre of longitudinal
    content, in 8 numbers
  * PATH          ``gamma_i(s)`` -- the polyline sampled on a SHARED ABSOLUTE
    arc-length grid in METRES (:func:`path_abs`), observed only out to ``L_i``
  * composition   ``traj_i(t_k) = gamma_i(sigma_i[k])``

CONTROLS -- EACH MUST READ A KNOWN VALUE, AND THEY ARE REPORTED FIRST
----------------------------------------------------------------------------
  C1  round trip, normalised grid  -- a CONVERGENCE SWEEP, not an asserted zero
  C1b round trip, ABSOLUTE grid    -- the number that prices the identity bound
  C2  known rank    a bank built as a TRUE product of P0 paths x V0 speeds must
                    be recovered at exactly (P0, V0). This is the control that
                    decides whether anything else here may be read.
  C2b instrument floor  within one TRUE speed group, the recovered sigma must
                    have zero spread. Whatever it actually reads is the floor
                    below which no speed tolerance can merge.
  C3  degenerate    128 identical anchors must read (1, 1).
  C4  anti-collapse 128 anchors distinct on BOTH axes must read (128, 128).

⛔⛔ WHAT THE CONTROLS ACTUALLY DID -- READ THIS BEFORE QUOTING ANY RANK.

  * C2 REFUTED THE FIRST REPRESENTATION. Normalising each path by its own length
    makes an arc's shape depend on ``kappa * L``, so the path axis silently
    carried the speed axis and a bank of 8 KNOWN arcs read as 128 paths. The
    absolute-arc-length form in :func:`path_abs` is the repair, and it recovers
    (8, 16) exactly.
  * C4 FAILS AT C2's OPERATING POINT (reads 91 paths / 23 speeds where the truth
    is 128 / 128). No single tolerance passes both, because "distinct path" is
    not tolerance-free -- C4's bank is deliberately a near-continuum. ⇒ EVERY
    RANK READ ON THE REAL BANK IS A LOWER BOUND, and is reported as one.

⛔ NOT MEASURED HERE: whether the product set lowers oracle-in-vocabulary ADE by
more than the identity bound. That needs the 19,602 held-out GT windows, which
live on the pod. The identity bound itself is C1b + the diagonal: the product
contains every joint member to within the C1b resampling error, so oracle ADE
can only fall, and C1b prices the "within".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

# The refcv4/refcv5 slot grid, in ticks at 10 Hz (tanitad.refs.refc_v3.V3_HORIZONS).
V3_HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOT_T_S = tuple(h / 10.0 for h in V3_HORIZONS)


# ---------------------------------------------------------------- decomposition
def arc_lengths(traj: torch.Tensor) -> torch.Tensor:
    """[N, S, 2] -> [N, S] cumulative arc length from the ego origin."""
    n, s, _ = traj.shape
    pts = torch.cat([traj.new_zeros(n, 1, 2), traj], dim=1)      # [N, S+1, 2]
    seg = (pts[:, 1:] - pts[:, :-1]).norm(dim=-1)                 # [N, S]
    return seg.cumsum(dim=1)


def path_abs(traj: torch.Tensor, ds: float = 1.0, s_max: float = 220.0,
             eps: float = 1e-9) -> tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor]:
    """[N, S, 2] -> (samples [N, G, 2], valid [N, G] bool, length [N]).

    ⭐ THE REPRESENTATION THAT MAKES THE TWO AXES INDEPENDENT. The path is
    sampled on a SHARED ABSOLUTE arc-length grid ``s = 0, ds, 2ds, ... s_max``
    in METRES and is NOT normalised by its own length -- so "the same arc driven
    further" is the SAME path, which is exactly what a path x velocity
    factorisation asserts. ``valid`` marks the samples an anchor actually
    reaches (``s <= L_i``); everything beyond is UNOBSERVED, never extrapolated.

    ⛔ The self-normalising form (divide by ``L_i``) was tried first and control
    C2 REFUTED it: an arc of curvature k normalised by its own length has a
    shape that depends on ``k * L``, so the path axis silently carries the speed
    axis and a bank of 8 known arcs read as 128 distinct paths.
    """
    n, s, _ = traj.shape
    cum = arc_lengths(traj)
    length = cum[:, -1]
    pts = torch.cat([traj.new_zeros(n, 1, 2), traj], dim=1)
    knot = torch.cat([cum.new_zeros(n, 1), cum], dim=1)
    grid = torch.arange(0.0, s_max + ds, ds, dtype=traj.dtype)
    g = grid.numel()
    out = traj.new_zeros(n, g, 2)
    valid = torch.zeros(n, g, dtype=torch.bool)
    for i in range(n):
        li = float(length[i])
        k = torch.searchsorted(knot[i].contiguous(), grid.contiguous()).clamp(1, s)
        s0, s1 = knot[i][k - 1], knot[i][k]
        w = ((grid - s0) / (s1 - s0).clamp_min(eps)).unsqueeze(-1)
        out[i] = pts[i][k - 1] * (1 - w) + pts[i][k] * w
        valid[i] = grid <= li
    return out, valid, length


def path_shape(traj: torch.Tensor, n_grid: int = 64,
               eps: float = 1e-9) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, S, 2] -> (shape [N, n_grid, 2], length [N]).

    ``shape`` is the polyline resampled at ``n_grid`` equally spaced NORMALISED
    arc lengths and divided by its total length.

    ⛔ KEPT ONLY FOR THE C1 ROUND TRIP, which needs an invertible composition.
    ⛔ NOT ADMISSIBLE FOR RANK COUNTING -- see :func:`path_abs`; control C2
    refuted it for that use.
    """
    cum = arc_lengths(traj)                                       # [N, S]
    length = cum[:, -1]                                           # [N]
    n, s, _ = traj.shape
    pts = torch.cat([traj.new_zeros(n, 1, 2), traj], dim=1)       # [N, S+1, 2]
    knot = torch.cat([cum.new_zeros(n, 1), cum], dim=1)           # [N, S+1]
    u = torch.linspace(0.0, 1.0, n_grid, dtype=traj.dtype)        # [G]
    out = traj.new_zeros(n, n_grid, 2)
    for i in range(n):
        li = float(length[i])
        if li < eps:                     # degenerate: no path at all
            continue
        target = u * li                                           # [G] metres
        k = torch.searchsorted(knot[i].contiguous(), target.contiguous())
        k = k.clamp(1, s)
        s0, s1 = knot[i][k - 1], knot[i][k]
        w = ((target - s0) / (s1 - s0).clamp_min(eps)).unsqueeze(-1)
        out[i] = pts[i][k - 1] * (1 - w) + pts[i][k] * w
        out[i] = out[i] / li                                      # scale-free
    return out, length


def recompose(shape: torch.Tensor, sigma: torch.Tensor,
              eps: float = 1e-9) -> torch.Tensor:
    """(shape [N, G, 2] scale-free, sigma [N, S] metres) -> traj [N, S, 2].

    ``shape`` is read at normalised arc length ``sigma / sigma[-1]`` and scaled
    back up by the total length. This is the PRODUCT operator: pass shape ``j``
    with sigma ``i`` and you get the cross member (j, i).
    """
    n, g, _ = shape.shape
    s = sigma.shape[1]
    total = sigma[:, -1].clamp_min(eps)                           # [N]
    u = (sigma / total.unsqueeze(-1)).clamp(0.0, 1.0)             # [N, S]
    grid = torch.linspace(0.0, 1.0, g, dtype=shape.dtype)         # [G]
    out = shape.new_zeros(n, s, 2)
    for i in range(n):
        k = torch.searchsorted(grid.contiguous(), u[i].contiguous()).clamp(1, g - 1)
        u0, u1 = grid[k - 1], grid[k]
        w = ((u[i] - u0) / (u1 - u0).clamp_min(eps)).unsqueeze(-1)
        out[i] = (shape[i][k - 1] * (1 - w) + shape[i][k] * w) * total[i]
    return out


# ------------------------------------------------------------------ rank counting
def greedy_distinct(x: torch.Tensor, tol: float) -> tuple[int, list[int]]:
    """Count members that are pairwise further apart than ``tol``.

    Deterministic, order-dependent-by-index, single linkage free: walk the rows
    in order and keep a row only if it is further than ``tol`` (RMS over its
    feature vector) from every kept row. Returns (count, kept indices).
    """
    flat = x.reshape(x.shape[0], -1)
    kept: list[int] = []
    for i in range(flat.shape[0]):
        ok = True
        for j in kept:
            d = float((flat[i] - flat[j]).pow(2).mean().sqrt())
            if d <= tol:
                ok = False
                break
        if ok:
            kept.append(i)
    return len(kept), kept


def greedy_distinct_masked(x: torch.Tensor, valid: torch.Tensor, tol: float,
                           min_common_m: float, ds: float
                           ) -> tuple[int, int]:
    """Greedy distinct count on the COMMON OBSERVED domain of each pair.

    Two paths are compared only where BOTH are observed. A pair whose common
    domain is shorter than ``min_common_m`` is INCOMPARABLE and is counted as
    distinct (the conservative direction: it can only inflate the path rank,
    never deflate it). Returns (n_distinct, n_incomparable_pairs_seen).
    """
    n = x.shape[0]
    kept: list[int] = []
    incomparable = 0
    need = int(round(min_common_m / ds))
    for i in range(n):
        ok = True
        for j in kept:
            m = valid[i] & valid[j]
            if int(m.sum()) < need:
                incomparable += 1
                continue                     # cannot merge on no evidence
            d = float((x[i][m] - x[j][m]).pow(2).mean().sqrt())
            if d <= tol:
                ok = False
                break
        if ok:
            kept.append(i)
    return len(kept), incomparable


def rank_report(traj: torch.Tensor, shape_tols: list[float],
                speed_tols: list[float], ds: float, s_max: float,
                min_common_m: float) -> dict:
    samples, valid, length = path_abs(traj, ds=ds, s_max=s_max)
    sigma = arc_lengths(traj)
    rows = []
    for st in shape_tols:
        p, inc = greedy_distinct_masked(samples, valid, st, min_common_m, ds)
        rows.append({"axis": "path_abs_m", "tol_m": st, "distinct": p,
                     "incomparable_pairs": inc})
    for vt in speed_tols:
        v, _ = greedy_distinct(sigma, vt)
        rows.append({"axis": "speed_profile_m", "tol_m": vt, "distinct": v})
    return {"rows": rows,
            "length_m": {"min": float(length.min()), "max": float(length.max()),
                         "mean": float(length.mean())}}


def within_group_spread(traj: torch.Tensor, groups: list[list[int]], ds: float,
                        s_max: float) -> dict:
    """C2b -- the INSTRUMENT'S OWN FLOOR, on a bank whose factorisation is known.

    ``groups`` lists the index sets that share one TRUE speed profile. A perfect
    instrument recovers a spread of exactly 0 inside each group; whatever it
    actually reads is the floor below which no rank reading on the real bank can
    be trusted.
    """
    sigma = arc_lengths(traj)
    worst = 0.0
    tot = 0.0
    k = 0
    for gidx in groups:
        if len(gidx) < 2:
            continue
        sub = sigma[gidx]
        for a in range(len(gidx)):
            for b in range(a + 1, len(gidx)):
                d = float((sub[a] - sub[b]).pow(2).mean().sqrt())
                worst = max(worst, d)
                tot += d
                k += 1
    return {"n_pairs": k, "max_within_group_rms_m": worst,
            "mean_within_group_rms_m": (tot / k) if k else 0.0,
            "_ds": ds, "_s_max": s_max}


# ------------------------------------------------------------------ the controls
def make_product_bank(p0: int, v0: int, n: int, seed: int = 0) -> torch.Tensor:
    """C2: a bank that IS a product of ``p0`` paths and ``v0`` speed profiles.

    Constant-curvature arcs (the path axis) traversed with constant-acceleration
    speed profiles (the speed axis). ``n`` members are taken from the p0 x v0
    grid in row-major order, so the true rank is known exactly.
    """
    g = torch.Generator().manual_seed(seed)
    kappa = torch.linspace(-0.05, 0.05, p0)                       # 1/m
    v_end = torch.linspace(4.0, 22.0, v0)                         # m/s at 6 s
    t = torch.tensor(SLOT_T_S)                                    # [S]
    out = []
    for a in range(p0):
        for b in range(v0):
            if len(out) >= n:
                break
            # speed profile: constant acceleration from 8 m/s to v_end[b]
            v = 8.0 + (v_end[b] - 8.0) * (t / t[-1])
            s = 0.5 * (8.0 + v) * t                               # arc length
            k = float(kappa[a])
            if abs(k) < 1e-6:
                xy = torch.stack([s, torch.zeros_like(s)], dim=-1)
            else:
                r = 1.0 / k
                th = s * k
                xy = torch.stack([r * torch.sin(th), r * (1 - torch.cos(th))],
                                 dim=-1)
            out.append(xy)
    assert len(out) == n, f"{len(out)} != {n}; p0*v0 must be >= n"
    _ = g  # determinism is structural here; generator kept for signature parity
    return torch.stack(out)


def make_degenerate_bank(n: int) -> torch.Tensor:
    """C3: n identical straight anchors -> true rank (1, 1)."""
    t = torch.tensor(SLOT_T_S)
    s = 10.0 * t
    one = torch.stack([s, torch.zeros_like(s)], dim=-1)
    return one.unsqueeze(0).expand(n, -1, -1).contiguous().clone()


def make_allpairs_bank(n: int) -> torch.Tensor:
    """C4: n anchors distinct on BOTH axes -> true rank (n, n)."""
    kappa = torch.linspace(-0.05, 0.05, n)
    v_end = torch.linspace(4.0, 22.0, n)
    t = torch.tensor(SLOT_T_S)
    out = []
    for i in range(n):
        v = 8.0 + (v_end[i] - 8.0) * (t / t[-1])
        s = 0.5 * (8.0 + v) * t
        k = float(kappa[i])
        if abs(k) < 1e-6:
            xy = torch.stack([s, torch.zeros_like(s)], dim=-1)
        else:
            r = 1.0 / k
            th = s * k
            xy = torch.stack([r * torch.sin(th), r * (1 - torch.cos(th))], dim=-1)
        out.append(xy)
    return torch.stack(out)


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-grid", type=int, default=64)
    ap.add_argument("--ds", type=float, default=1.0,
                    help="absolute arc-length sample spacing, metres")
    ap.add_argument("--s-max", type=float, default=220.0)
    ap.add_argument("--op-tol-path", type=float, default=0.5,
                    help="path tolerance of the C2-validated operating point")
    ap.add_argument("--op-tol-speed", type=float, default=1.0,
                    help="speed tolerance of the C2-validated operating point")
    ap.add_argument("--min-common-m", type=float, default=10.0,
                    help="a pair with less common observed path than this is "
                         "INCOMPARABLE and counted distinct")
    args = ap.parse_args()

    p = Path(args.anchors)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    obj = torch.load(p, map_location="cpu", weights_only=False)
    traj = obj["anchors"] if isinstance(obj, dict) else obj
    traj = traj.to(torch.float64)
    n, s, c = traj.shape
    assert c == 2 and s == len(V3_HORIZONS), f"unexpected shape {tuple(traj.shape)}"

    res: dict = {"artifact": str(p).replace("\\", "/"), "sha256": sha,
                 "shape": list(traj.shape), "slot_t_s": list(SLOT_T_S)}

    # ---- C1: the round trip, and it is NOT free ------------------------------
    # ⚠️ The decomposition is exact at the KNOTS only if the path is stored as
    # the polyline itself. Storing it on a uniform normalised-arc-length grid --
    # which is what a factorised vocabulary would actually ship -- costs a
    # RESAMPLING error, because the knots do not land on grid points. So C1 is
    # reported as a CONVERGENCE SWEEP, never as a single number asserted to be
    # zero: a factorised bank's path resolution is a real design parameter and
    # this sweep prices it.
    sigma = arc_lengths(traj)
    sweep = []
    for g in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
        sh_g, _ = path_shape(traj, n_grid=g)
        err = (recompose(sh_g, sigma) - traj).abs()
        sweep.append({"n_grid": g,
                      "max_abs_err_m": float(err.max()),
                      "mean_abs_err_m": float(err.mean())})
    res["C1_roundtrip_grid_sweep"] = sweep

    # ---- C1b: the SHIPPED form's round trip, on the ABSOLUTE grid ------------
    # ⭐ This is the number that prices the identity bound. A factorised bank
    # stores paths on an absolute arc-length grid; composing path_i with its OWN
    # speed_i must reproduce anchor i, or the product set does NOT contain the
    # joint set and "ADE can only go down" is not an identity.
    sweep_abs = []
    for dsx in (4.0, 2.0, 1.0, 0.5, 0.25):
        sm, vm, _ = path_abs(traj, ds=dsx, s_max=args.s_max)
        gsz = sm.shape[1]
        gridx = torch.arange(0.0, args.s_max + dsx, dsx, dtype=traj.dtype)
        k = torch.searchsorted(gridx.contiguous(),
                               sigma.contiguous()).clamp(1, gsz - 1)
        w = ((sigma - gridx[k - 1]) / dsx).unsqueeze(-1)
        rec = torch.gather(sm, 1, (k - 1).unsqueeze(-1).expand(-1, -1, 2)) * (1 - w) \
            + torch.gather(sm, 1, k.unsqueeze(-1).expand(-1, -1, 2)) * w
        e = (rec - traj).abs()
        sweep_abs.append({"ds_m": dsx, "n_samples_per_path": gsz,
                          "max_abs_err_m": float(e.max()),
                          "mean_abs_err_m": float(e.mean())})
    res["C1b_absolute_grid_roundtrip"] = sweep_abs
    shape, length = path_shape(traj, n_grid=args.n_grid)
    c1 = float((recompose(shape, sigma) - traj).abs().max())
    res["C1_roundtrip_max_abs_err_m_at_n_grid"] = {"n_grid": args.n_grid,
                                                   "max_abs_err_m": c1}

    # ---- C2/C3/C4: the instrument must recover KNOWN ranks -------------------
    shape_tols = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]        # metres
    speed_tols = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]         # metres
    ds, s_max, min_common = args.ds, args.s_max, args.min_common_m
    res["grid"] = {"ds_m": ds, "s_max_m": s_max, "min_common_m": min_common}

    ctl: dict = {}
    p0, v0 = 8, 16
    prod = make_product_bank(p0, v0, n).to(torch.float64)
    ctl["C2_product_bank_true_rank"] = {"paths": p0, "speeds": v0, "n": n}
    ctl["C2_measured"] = rank_report(prod, shape_tols, speed_tols, ds, s_max,
                                     min_common)
    # the index sets that share ONE true speed profile (row-major build order)
    groups = [[a * v0 + b for a in range(p0)] for b in range(v0)]
    ctl["C2b_instrument_floor"] = within_group_spread(prod, groups, ds, s_max)
    deg = make_degenerate_bank(n).to(torch.float64)
    ctl["C3_degenerate_true_rank"] = {"paths": 1, "speeds": 1}
    ctl["C3_measured"] = rank_report(deg, shape_tols, speed_tols, ds, s_max,
                                     min_common)
    allp = make_allpairs_bank(n).to(torch.float64)
    ctl["C4_allpairs_true_rank"] = {"paths": n, "speeds": n}
    ctl["C4_measured"] = rank_report(allp, shape_tols, speed_tols, ds, s_max,
                                     min_common)
    res["controls"] = ctl

    # ---- the real bank -------------------------------------------------------
    res["bank"] = rank_report(traj, shape_tols, speed_tols, ds, s_max,
                              min_common)

    # ---- how much of the bank's spread is LONGITUDINAL -----------------------
    res["axis_spread"] = {
        "speed_profile_rms_m": float(sigma.std(dim=0).mean()),
        "arc_length_m_p5_p50_p95": [float(length.quantile(q))
                                    for q in (0.05, 0.5, 0.95)],
        "final_speed_ms_p5_p50_p95": [
            float(((sigma[:, -1] - sigma[:, -2]) / (SLOT_T_S[-1] - SLOT_T_S[-2]))
                  .quantile(q)) for q in (0.05, 0.5, 0.95)],
    }

    # ---- the arithmetic, at the C2-VALIDATED operating point -----------------
    # ⛔ The operating point is not chosen here: it is the tolerance pair at
    # which control C2 recovers its KNOWN rank exactly. Anything read at another
    # tolerance is not backed by a control.
    def _at(rep, axis, tol):
        for r in rep["rows"]:
            if r["axis"] == axis and abs(r["tol_m"] - tol) < 1e-12:
                return r["distinct"]
        raise KeyError((axis, tol))

    tol_path, tol_speed = args.op_tol_path, args.op_tol_speed
    c2_p = _at(ctl["C2_measured"], "path_abs_m", tol_path)
    c2_v = _at(ctl["C2_measured"], "speed_profile_m", tol_speed)
    p = _at(res["bank"], "path_abs_m", tol_path)
    v = _at(res["bank"], "speed_profile_m", tol_speed)
    n_samp = next(r["n_samples_per_path"] for r in sweep_abs
                  if abs(r["ds_m"] - ds) < 1e-12)
    res["product_arithmetic"] = {
        "operating_point": {"tol_path_m": tol_path, "tol_speed_m": tol_speed},
        "C2_recovered": {"paths": c2_p, "paths_true": p0,
                         "speeds": c2_v, "speeds_true": v0,
                         "PASS": bool(c2_p == p0 and c2_v == v0)},
        "C3_recovered": {"paths": _at(ctl["C3_measured"], "path_abs_m", tol_path),
                         "speeds": _at(ctl["C3_measured"], "speed_profile_m",
                                       tol_speed)},
        "C4_recovered": {"paths": _at(ctl["C4_measured"], "path_abs_m", tol_path),
                         "speeds": _at(ctl["C4_measured"], "speed_profile_m",
                                       tol_speed), "true": n},
        "bank_paths_LOWER_BOUND": p,
        "bank_speeds_LOWER_BOUND": v,
        "joint_candidates": n,
        "product_candidates": p * v,
        "density_multiplier": (p * v) / n,
        "logits_joint": n,
        "logits_factorised": p + v,
        "floats_joint": n * s * 2,
        "floats_factorised": p * n_samp * 2 + v * s,
        "floats_ratio": (p * n_samp * 2 + v * s) / (n * s * 2),
    }

    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("controls", "bank")}, indent=1))
    print("C1 roundtrip max abs err (m):", c1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
