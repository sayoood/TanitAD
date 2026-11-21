"""⛔ refcv4 LAUNCH BLOCKER 2 — RE-DERIVE the S2 reach clamp for the 6 s horizon.

`refc_v3.py:448` sets `sel_reach_clamp = True` citing "inert on ADE, deletes
72.08 %" — a statistic MEASURED at `horizon_s = 2.0`.  `horizon_s` is DERIVED
from `max(horizons)` (`refc.py:652`), so at 6 s the SAME `sel_accel_max = 2.5`
opens the band from +-5.0 m/s to +-15.0 m/s.  `refc_v3.py:79-82` warns in
writing not to inherit those statistics.  This re-derives them.

THE BAND (`flagship_v15.reachability_mask`, the one implementation):
    v_mean(i) = ||traj_i[-1]|| / T          # T = horizon_s
    keep(i)   <=>  max(0, v0 - a*T) <= v_mean(i) <= v0 + a*T

THE DERIVATION CRITERIA, in priority order:
  1. ⛔ THE BAND MUST NOT DELETE THE GROUND TRUTH.  A reachability clamp that
     removes the trajectory the ego actually flew is not conservative, it is
     wrong.  Measured directly as the GT-deletion rate at each `a`.
  2. INERT ON ADE — the original 2 s criterion: the oracle-in-vocabulary ADE
     computed over SURVIVORS ONLY must equal the unclamped oracle.
  3. NEVER EMPTY, and the survivor set must still SPAN THE MANOEUVRE SPACE
     (>30 deg turns must survive).
  4. Subject to 1-3, as selective as possible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, "/workspace/TanitAD/stack")

from tanitad.data.v2_dataset import build_v2_providers            # noqa: E402
from tanitad.models.flagship_v15 import (candidate_mean_speed,    # noqa: E402
                                         reachability_mask)
from tanitad.refs.refc import default_anchors                     # noqa: E402
from tanitad.refs.refc_v3 import V3_HORIZONS, SEAM_SLOT           # noqa: E402

DT = 0.1
TURN_DEG = 30.0


def ego_pool(poses, horizons):
    poses = poses.float()
    max_h = max(horizons)
    n = poses.shape[0] - max_h
    if n <= 0:
        return None, None
    p0 = poses[:n]
    yaw = p0[:, 2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    outs = []
    for k in horizons:
        d = poses[k:k + n, :2] - p0[:, :2]
        outs.append(torch.stack([c * d[:, 0] + s * d[:, 1],
                                 -s * d[:, 0] + c * d[:, 1]], dim=-1))
    return torch.stack(outs, dim=1), p0[:, 3]


def corpus_pool(cache_dir, horizons, limit=0, tag=""):
    eps = build_v2_providers([cache_dir], lru_size=1, verbose=False)
    if limit:
        eps = eps[:limit]
    P, V = [], []
    for ep in eps:
        p, v = ego_pool(ep.poses, horizons)
        if p is not None:
            P.append(p), V.append(v)
    P, V = torch.cat(P), torch.cat(V)
    print(f"[pool:{tag}] {len(eps)} clips -> {P.shape[0]} windows", flush=True)
    return P, V


def turn_deg(traj: torch.Tensor) -> torch.Tensor:
    """|bearing of the 6 s endpoint| in degrees, ego frame (heading +x)."""
    return torch.rad2deg(torch.atan2(traj[..., -1, 1],
                                     traj[..., -1, 0]).abs())


def term_turn_deg(traj: torch.Tensor) -> torch.Tensor:
    """|heading of the FINAL segment| in degrees — the manoeuvre being held."""
    d = traj[..., -1, :] - traj[..., -2, :]
    return torch.rad2deg(torch.atan2(d[..., 1], d[..., 0]).abs())


def oracle_ade(anchors, gt, k, keep=None):
    """Oracle-in-vocab ADE over the first k slots; `keep` [B,N] masks candidates."""
    d = (anchors[None, :, :k] - gt[:, None, :k]).norm(dim=-1).mean(-1)  # [B,N]
    if keep is not None:
        d = d.masked_fill(~keep, float("inf"))
    return d.min(1).values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--train", default="/root/data/train")
    ap.add_argument("--eval", dest="ev", default="/root/data/eval")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--out", default="/workspace/clamp6s.json")
    args = ap.parse_args()

    H = V3_HORIZONS
    T = max(H) * DT                       # 6.0 s — what refc.py:652 derives
    T2 = (SEAM_SLOT + 1) and H[SEAM_SLOT] * DT      # 2.0 s, the inherited one
    A = torch.load(args.anchors, map_location="cpu", weights_only=True)
    if isinstance(A, dict):
        A = A["anchors"]
    A = A.float()
    A_syn = default_anchors(H, A.shape[0], 4096, 0)
    print(f"[cfg] anchors {tuple(A.shape)}  horizon_s DERIVED = {T}  "
          f"(the inherited statistics were measured at {T2})", flush=True)

    Pev, Vev = corpus_pool(args.ev, H, args.limit, "eval")
    Ptr, Vtr = corpus_pool(args.train, H, args.limit, "train")

    # ---- 1. WHAT THE DATA ACTUALLY NEEDS ---------------------------------
    # |v_mean(GT over T) - v0| — the smallest band that never deletes the truth.
    res = {}
    for tag, P, V in (("train", Ptr, Vtr), ("eval", Pev, Vev)):
        dv = (P[:, -1].norm(dim=-1) / T - V).abs()
        q = {f"p{p}": float(dv.quantile(p / 100)) for p in (50, 90, 99, 99.9)}
        q["max"] = float(dv.max())
        q["a_for_p99.9"] = q["p99.9"] / T
        q["a_for_max"] = q["max"] / T
        res[f"gt_dv_{tag}"] = q
        print(f"[gt-band:{tag}] |v_mean(GT,{T}s) - v0|  p50 {q['p50']:.3f}  "
              f"p90 {q['p90']:.3f}  p99 {q['p99']:.3f}  p99.9 {q['p99.9']:.3f}  "
              f"max {q['max']:.3f} m/s   => a_max needed: "
              f"p99.9 {q['a_for_p99.9']:.4f}  max {q['a_for_max']:.4f} m/s^2",
              flush=True)

    # the anchors' own turn census (a property of the SET, not of any window)
    for nm, X in (("NEW  ", A), ("SYNTH", A_syn)):
        t_ep, t_tm = turn_deg(X), term_turn_deg(X)
        print(f"[vocab:{nm}] anchors with |endpoint bearing| > {TURN_DEG:.0f} deg: "
              f"{int((t_ep > TURN_DEG).sum())}/{X.shape[0]}   "
              f"|terminal heading| > {TURN_DEG:.0f} deg: "
              f"{int((t_tm > TURN_DEG).sum())}/{X.shape[0]}   "
              f"max endpoint bearing {float(t_ep.max()):.1f} deg", flush=True)

    # ---- 1b. FLYABILITY of the vocabulary itself --------------------------
    # k-means centroids are AVERAGES of real trajectories, so "is every anchor a
    # path a car could drive?" is a real question and is answered here rather
    # than assumed.  Slot times are irregular (0.5 s x4 then 1.0 s x4), so the
    # per-slot dt is used explicitly instead of a single constant.
    ts = torch.tensor([h * DT for h in H])
    for nm, X in (("NEW  ", A), ("SYNTH", A_syn)):
        p = torch.cat([torch.zeros(X.shape[0], 1, 2), X], dim=1)       # +origin
        tt = torch.cat([torch.zeros(1), ts])
        dt = (tt[1:] - tt[:-1])[None, :, None]
        vel = (p[:, 1:] - p[:, :-1]) / dt                              # [N,S,2]
        spd = vel.norm(dim=-1)
        acc = ((vel[:, 1:] - vel[:, :-1]) / dt[:, 1:]).norm(dim=-1)
        print(f"[flyable:{nm}] speed  max {float(spd.max()):6.2f} m/s  "
              f"p99 {float(spd.flatten().quantile(0.99)):6.2f} | "
              f"|accel| max {float(acc.max()):5.2f} m/s^2  "
              f"p99 {float(acc.flatten().quantile(0.99)):5.2f}  "
              f"(>8 m/s^2 on {float((acc > 8).float().mean()*100):.2f} % of "
              f"segments)", flush=True)

    # ---- 2/3/4. THE SWEEP -------------------------------------------------
    ade_free = oracle_ade(A, Pev, SEAM_SLOT + 1)
    v_mean = candidate_mean_speed(A[None], T)[0]              # [N]
    is_turn_ep = turn_deg(A) > TURN_DEG
    is_turn_tm = term_turn_deg(A) > TURN_DEG
    print(f"\n[unclamped] oracle-in-vocab ADE 0-2 s on EVAL = "
          f"{float(ade_free.mean()):.4f} m", flush=True)
    print(f"\n{'a_max':>6} {'band':>9} | {'kill%':>7} {'empty%':>7} "
          f"{'GTdel%':>7} | {'surv/win':>9} {'turnEP/win':>10} "
          f"{'turnTM/win':>10} | {'ADE(surv)':>9} {'dADE':>8}", flush=True)
    print("-" * 108, flush=True)
    rows = []
    gt_vm = Pev[:, -1].norm(dim=-1) / T
    gt_vm_tr = Ptr[:, -1].norm(dim=-1) / T
    for a in (0.25, 0.4, 0.5, 0.625, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        reach = a * T
        lo = (Vev - reach).clamp_min(0.0)
        hi = Vev + reach
        keep = (v_mean[None] >= lo[:, None]) & (v_mean[None] <= hi[:, None])
        gt_del = ((gt_vm < lo) | (gt_vm > hi)).float().mean() * 100
        lo_t, hi_t = (Vtr - reach).clamp_min(0.0), Vtr + reach
        gt_del_tr = ((gt_vm_tr < lo_t) | (gt_vm_tr > hi_t)).float().mean() * 100
        surv = keep.sum(1).float()
        ade_c = oracle_ade(A, Pev, SEAM_SLOT + 1, keep)
        finite = torch.isfinite(ade_c)
        rows.append({
            "a_max": a, "band_ms": reach,
            "kill_pct": float((~keep).float().mean() * 100),
            "empty_pct": float((surv == 0).float().mean() * 100),
            "gt_deleted_pct": float(gt_del),
            "gt_deleted_train_pct": float(gt_del_tr),
            "surv_per_window": float(surv.mean()),
            "turn_ep_per_window": float((keep & is_turn_ep[None]).sum(1).float().mean()),
            "turn_tm_per_window": float((keep & is_turn_tm[None]).sum(1).float().mean()),
            "windows_with_no_turn_ep": float(((keep & is_turn_ep[None]).sum(1) == 0)
                                             .float().mean() * 100),
            "ade_survivors": float(ade_c[finite].mean()),
            "d_ade": float(ade_c[finite].mean() - ade_free[finite].mean()),
        })
        r = rows[-1]
        print(f"{a:6.3f} {reach:8.2f}m | {r['kill_pct']:6.2f}% "
              f"{r['empty_pct']:6.2f}% {r['gt_deleted_pct']:6.2f}% | "
              f"{r['surv_per_window']:9.1f} {r['turn_ep_per_window']:10.1f} "
              f"{r['turn_tm_per_window']:10.1f} | {r['ade_survivors']:9.4f} "
              f"{r['d_ade']:+8.4f}", flush=True)

    # ---- the recommendation ----------------------------------------------
    # binding: GT deletion == 0, never empty, never a window without a turn,
    # ADE within 1 mm of unclamped.  Then the SMALLEST admissible a (= tightest).
    # ⛔ the band may not delete the truth: EXACTLY zero on the held-out eval
    # split, and <= 1 in 10,000 on the 635k train windows (where the selection
    # CE target is built).  Requiring the absolute train MAX would be a
    # one-window criterion on 635,331 samples.
    ok = [r for r in rows if r["gt_deleted_pct"] == 0.0
          and r["gt_deleted_train_pct"] <= 0.01
          and r["empty_pct"] == 0.0 and r["windows_with_no_turn_ep"] == 0.0
          and abs(r["d_ade"]) <= 1e-3]
    pick = min(ok, key=lambda r: r["a_max"]) if ok else None
    inherited = next(r for r in rows if r["a_max"] == 2.5)
    print(f"\n=== VERDICT ===")
    print(f"  INHERITED  a=2.500 (band +-{inherited['band_ms']:.1f} m/s at "
          f"T={T}s): kills {inherited['kill_pct']:.2f} %  "
          f"(the 2 s statistic it cites is 77.28 % / 72.08 %)")
    if pick is None:
        print("  ⛔ NO admissible a in the sweep — DO NOT LAUNCH with the clamp on")
        res["pick"] = None
    else:
        print(f"  RE-DERIVED a={pick['a_max']:.3f} (band +-{pick['band_ms']:.1f} m/s): "
              f"kills {pick['kill_pct']:.2f} %, empty {pick['empty_pct']:.2f} %, "
              f"GT deleted {pick['gt_deleted_pct']:.2f} %, "
              f"dADE {pick['d_ade']:+.5f} m")
        print(f"  survivor set spans the manoeuvre space: "
              f"{pick['surv_per_window']:.1f} survivors/window of which "
              f"{pick['turn_ep_per_window']:.1f} are >{TURN_DEG:.0f} deg turns; "
              f"{pick['windows_with_no_turn_ep']:.2f} % of windows have no turn")
        res["pick"] = pick
    res.update({"horizon_s": T, "inherited_horizon_s": T2, "sweep": rows,
                "anchors": os.path.basename(args.anchors),
                "n_anchors": int(A.shape[0]),
                "eval_windows": int(Pev.shape[0]),
                "train_windows": int(Ptr.shape[0]),
                "ade_unclamped": float(ade_free.mean()),
                "inherited": inherited})
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"\n[saved] {args.out}")
    return 0 if pick else 3


if __name__ == "__main__":
    raise SystemExit(main())
