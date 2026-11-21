"""⛔ refcv4 LAUNCH BLOCKER 1 — build + GATE the 6 s DATA-DRIVEN anchor vocabulary.

MEASURED root cause (GOALS_AND_CLAIMS D-REFCV3-SMOOTH1): `refcv3_b1_launch.sh`
passes no --anchors, so refcv3 carries `refc.default_anchors` = FPS over 4,096
SYNTHETIC constant-(yaw_rate, accel) unicycle rollouts.  Oracle-in-vocabulary
ADE 0-2 s: synthetic 0.9433 m vs data-FPS 0.4369 m vs a SINGLE STRAIGHT LINE
0.6780 m — refcv3's fan is worse than one straight line.

This builds the vocabulary the incumbent should have had, at the 6 s horizon
`V3_HORIZONS = (5,10,15,20,30,40,50,60)`, by FPS over REAL ego-frame GT
trajectories from the B1 v7.2 TRAIN split, and GATES it out-of-sample on the
EVAL split (clip-disjoint, asserted and printed).

TWO KNOWN MECHANISMS, both LONGITUDINAL (the deficit is 92.2 % along-track):
  (a) synth_anchor_pool gives every anchor ONE constant acceleration for 6 s, so
      its longitudinal half is structurally 1-D.  Fixed by using the real pool.
  (b) FPS flattens [M, S*2] and normalises in `n` but NEVER in `S`, so the 6 s
      coordinate carries ~9x the magnitude (~81x the squared-distance weight) of
      the 0.5 s one — all 128 anchors go to spreading the 6 s endpoint while the
      0-2 s band, which the gate scores, is left coarse.  Fixed by an explicit
      per-slot (and optionally per-axis) FPS METRIC; the metric is swept and the
      winner is chosen on the GATE, which is scored on held-out clips.

Every arm carries its controls: a constant-velocity straight line (the floor the
vocabulary must beat), the zero path (no-information value), and the SYNTHETIC
vocabulary itself (the incumbent).  Errors are reported split ALONG vs LATERAL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/workspace/TanitAD/stack")

from tanitad.data.v2_dataset import build_v2_providers          # noqa: E402
from tanitad.refs.refc import default_anchors                   # noqa: E402
from tanitad.refs.refc_v3 import V3_HORIZONS, SEAM_SLOT         # noqa: E402

DT = 0.1
GATE_SLOTS = SEAM_SLOT + 1          # slots 0..3 == 0.5/1.0/1.5/2.0 s
STRAIGHT_LINE_FLOOR = 0.6780        # m, MEASURED, the control we MUST beat
DATA_FPS_TARGET = 0.4369            # m, MEASURED, refc-base's 2 s construction


# ---------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------
def ego_pool(poses: torch.Tensor, horizons) -> tuple[torch.Tensor, torch.Tensor]:
    """Ego-frame GT waypoints for EVERY window of one clip.

    Returns ([n, S, 2] waypoints, [n] v0).  Identical convention to
    refb_labels.waypoint_targets / ego_frame (verified against that source):
    x = dx*cos(yaw) + dy*sin(yaw); y = -dx*sin(yaw) + dy*cos(yaw).
    """
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


def corpus_pool(cache_dir: str, horizons, limit: int = 0, tag: str = ""):
    eps = build_v2_providers([cache_dir], lru_size=1, verbose=False)
    if limit:
        eps = eps[:limit]
    P, V = [], []
    for ep in eps:
        p, v = ego_pool(ep.poses, horizons)
        if p is not None:
            P.append(p)
            V.append(v)
    print(f"[pool:{tag}] {len(eps)} clips -> {sum(x.shape[0] for x in P)} windows",
          flush=True)
    return torch.cat(P), torch.cat(V), len(eps)


# ---------------------------------------------------------------------------
# FPS with an explicit metric
# ---------------------------------------------------------------------------
def fps_weighted(pool: torch.Tensor, n: int, w: torch.Tensor,
                 seed: int = 0) -> torch.Tensor:
    """Greedy FPS in the space (pool * w); returns the UNWEIGHTED pool points.

    `w` [S, 2] is the per-slot/per-axis metric.  w = ones reproduces
    refc.furthest_point_sample byte-for-byte (asserted by --selftest).
    """
    m = pool.shape[0]
    if n > m:
        raise ValueError(f"cannot FPS {n} from a pool of {m}")
    flat = (pool * w[None]).reshape(m, -1)
    g = torch.Generator(device=flat.device).manual_seed(seed)
    first = int(torch.randint(m, (1,), generator=g, device=flat.device))
    chosen = [first]
    dist = ((flat - flat[first]) ** 2).sum(-1)
    for _ in range(n - 1):
        nxt = int(torch.argmax(dist))
        chosen.append(nxt)
        dist = torch.minimum(dist, ((flat - flat[nxt]) ** 2).sum(-1))
    return pool[torch.tensor(chosen)].contiguous()


def kmeans_weighted(pool: torch.Tensor, n: int, w: torch.Tensor, seed: int = 0,
                    iters: int = 40, medoid: bool = False) -> torch.Tensor:
    """k-means++ / Lloyd in the space (pool * w); returns anchors in RAW units.

    WHY THIS IS OFFERED BESIDE FPS.  The GATE is the mean min-distance from a
    held-out trajectory to the vocabulary, and that is EXACTLY the objective
    Lloyd minimises; FPS minimises the *worst-case* covering radius instead.
    ``build_refc_anchors.py``'s docstring rejects k-means because it "collapses
    centroids onto the straight mode and starves the turns" — a statement about
    MODE COVERAGE, not about the oracle ADE this gate scores.  Both are built and
    the out-of-sample gate decides, rather than the docstring.
    ``medoid=True`` snaps every centroid to its nearest REAL pool trajectory, so
    the vocabulary contains only physically-flown paths (a mean of two
    trajectories need not be flyable).
    """
    m = pool.shape[0]
    flat = (pool * w[None]).reshape(m, -1)
    g = torch.Generator().manual_seed(seed)
    # k-means++ seeding
    idx = [int(torch.randint(m, (1,), generator=g))]
    d2 = ((flat - flat[idx[0]]) ** 2).sum(-1)
    for _ in range(n - 1):
        p = (d2 / d2.sum().clamp_min(1e-12))
        nxt = int(torch.multinomial(p, 1, generator=g))
        idx.append(nxt)
        d2 = torch.minimum(d2, ((flat - flat[nxt]) ** 2).sum(-1))
    C = flat[torch.tensor(idx)].clone()
    for _ in range(iters):
        a = torch.cdist(flat, C).argmin(1)                     # [M]
        newC = C.clone()
        for k in range(n):
            sel = flat[a == k]
            if sel.numel():
                newC[k] = sel.mean(0)
        if torch.allclose(newC, C, atol=1e-5):
            C = newC
            break
        C = newC
    if medoid:                                # snap to the nearest real path
        near = torch.cdist(C, flat).argmin(1)
        return pool[near].contiguous()
    S = pool.shape[1]
    return (C.reshape(n, S, 2) / w[None]).contiguous()


def metrics_for(pool: torch.Tensor, horizons) -> dict:
    """The FPS metric variants.  All are [S, 2] positive weights."""
    S = pool.shape[1]
    rms = pool.pow(2).sum(-1).mean(0).sqrt().clamp_min(1e-6)          # [S]
    sd = pool.std(0).clamp_min(1e-6)                                  # [S, 2]
    ones = torch.ones(S, 2)
    out = {
        # what build_refc_anchors.py does today: raw metres, 6 s dominates
        "raw": ones,
        # equalise the SLOTS, keep the physical along/lateral ratio inside each
        "slotnorm": (1.0 / rms)[:, None].expand(S, 2).contiguous(),
        # equalise slots AND axes (whitening) — spends the MOST on lateral
        "slotaxis": 1.0 / sd,
    }
    # whitened, then LATERAL deliberately down-weighted so FPS spends its
    # resolution on the ALONG axis (the axis carrying 92.2 % of the deficit)
    for beta in (0.5, 0.25):
        w = (1.0 / sd).clone()
        w[:, 1] *= beta
        out[f"slotaxis_along{beta:g}"] = w
    return out


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------
def oracle(anchors: torch.Tensor, gt: torch.Tensor, k: int) -> dict:
    """Oracle-in-vocabulary over the FIRST k slots, split ALONG vs LATERAL.

    Selection is the argmin of the 2-D ADE over those k slots (exactly
    run_vocab.py's `best_in_vocab`); the split reports the components of the
    error of THAT selected anchor, so along/lateral are not two separate
    selections.
    """
    A, G = anchors[:, :k], gt[:, :k]
    d = (A[None] - G[:, None])                       # [N, K, k, 2]
    ade = d.norm(dim=-1).mean(-1)                    # [N, K]
    idx = ade.argmin(1)
    sel = d[torch.arange(d.shape[0]), idx]           # [N, k, 2]
    return {"ade": ade.min(1).values, "along": sel[..., 0].abs().mean(-1),
            "lat": sel[..., 1].abs().mean(-1), "idx": idx}


def path_err(path: torch.Tensor, gt: torch.Tensor, k: int) -> dict:
    d = path[:, :k] - gt[:, :k]
    return {"ade": d.norm(dim=-1).mean(-1), "along": d[..., 0].abs().mean(-1),
            "lat": d[..., 1].abs().mean(-1), "idx": None}


def cv_path(v0: torch.Tensor, horizons) -> torch.Tensor:
    """Constant-velocity straight line: x = v0*t, y = 0.  [N, S, 2]."""
    t = torch.tensor([h * DT for h in horizons])
    return torch.stack([v0[:, None] * t[None], torch.zeros(len(v0), len(t))], -1)


def row(name: str, r: dict) -> str:
    return (f"  {name:<44s} ADE {r['ade'].mean():7.4f}  "
            f"ALONG {r['along'].mean():7.4f}  LAT {r['lat'].mean():7.4f}  "
            f"p50 {r['ade'].median():7.4f}  p90 "
            f"{r['ade'].quantile(0.90):7.4f}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/root/data/train")
    ap.add_argument("--eval", dest="ev", default="/root/data/eval")
    ap.add_argument("--n-anchors", type=int, default=128)
    ap.add_argument("--max-pool", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="clips per split (0=all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    H = V3_HORIZONS
    t0 = time.time()
    print(f"[cfg] horizons {H} (={[h*DT for h in H]} s)  n_anchors "
          f"{args.n_anchors}  gate slots 0..{GATE_SLOTS-1} (0-2 s)", flush=True)

    # ---- ⛔ CLIP DISJOINTNESS, asserted on the FILENAMES (independent of the
    # loader) and PRINTED, per the brief.
    tr_ids = {f[:-len(".v2ep.pt")] for f in os.listdir(args.train)
              if f.endswith(".v2ep.pt")}
    ev_ids = {f[:-len(".v2ep.pt")] for f in os.listdir(args.ev)
              if f.endswith(".v2ep.pt")}
    inter = sorted(tr_ids & ev_ids)
    print(f"[split] train {len(tr_ids)} clips · eval {len(ev_ids)} clips · "
          f"INTERSECTION {len(inter)} -> {inter}", flush=True)
    if inter:
        raise SystemExit("⛔ train/eval clip overlap — the gate would be leaked")

    Ptr, Vtr, ntr = corpus_pool(args.train, H, args.limit, "train")
    Pev, Vev, nev = corpus_pool(args.ev, H, args.limit, "eval")

    if args.selftest:                       # w=ones must reproduce refc's FPS
        from tanitad.refs.refc import furthest_point_sample
        sub = Ptr[:5000]
        a = furthest_point_sample(sub, 16, seed=0)
        b = fps_weighted(sub, 16, torch.ones(len(H), 2), seed=0)
        assert torch.equal(a, b), "fps_weighted(w=1) != refc.furthest_point_sample"
        print("[selftest] fps_weighted(w=1) == refc.furthest_point_sample ✅")

    g = torch.Generator().manual_seed(args.seed)
    pool = Ptr
    if pool.shape[0] > args.max_pool:
        sel = torch.randperm(pool.shape[0], generator=g)[:args.max_pool]
        pool = pool[sel]
    print(f"[fps] pool {tuple(pool.shape)} (subsampled from {tuple(Ptr.shape)})",
          flush=True)

    # ---- controls -------------------------------------------------------
    CV = cv_path(Vev, H)
    ZERO = torch.zeros_like(Pev)
    A_syn = default_anchors(H, args.n_anchors, 4096, 0)

    N = args.n_anchors
    print(f"\n=== GATE: oracle-in-vocabulary over 0-2 s, HELD-OUT EVAL CLIPS "
          f"(n={len(Pev)} windows / {nev} clips) ===", flush=True)
    print(f"  {'':<44s} {'(metres, lower = better)'}")
    results = {}
    base = {
        "CONTROL: zero path (no information)": path_err(ZERO, Pev, GATE_SLOTS),
        "CONTROL: constant-velocity straight line": path_err(CV, Pev, GATE_SLOTS),
        f"INCUMBENT: SYNTHETIC {N} (refcv3's actual)": oracle(A_syn, Pev, GATE_SLOTS),
    }
    for k, v in base.items():
        print(row(k, v), flush=True)
    print()

    built = {}
    mets = metrics_for(pool, H)
    arms = [(f"FPS      metric={k}", k, "fps") for k in mets]
    for k in ("raw", "slotnorm"):
        arms.append((f"KMEANS   metric={k}", k, "kmeans"))
        arms.append((f"KMEDOID  metric={k}", k, "kmedoid"))
    for label, mk, kind in arms:
        w = mets[mk]
        name = f"{kind}:{mk}"
        if kind == "fps":
            A = fps_weighted(pool, N, w, seed=args.seed)
        else:
            A = kmeans_weighted(pool, N, w, seed=args.seed,
                                medoid=(kind == "kmedoid"))
        built[name] = A
        r = oracle(A, Pev, GATE_SLOTS)
        r6 = oracle(A, Pev, len(H))
        results[name] = {
            "ade_0_2s": float(r["ade"].mean()), "along_0_2s": float(r["along"].mean()),
            "lat_0_2s": float(r["lat"].mean()),
            "ade_0_6s": float(r6["ade"].mean()), "along_0_6s": float(r6["along"].mean()),
            "lat_0_6s": float(r6["lat"].mean()),
            "distinct_anchors_used": int(r["idx"].unique().numel()),
            "weight": w.tolist(),
        }
        print(row(f"DATA {N:3d}  {label}", r)
              + f"  | 0-6s ADE {r6['ade'].mean():7.4f}"
              + f"  | uses {r['idx'].unique().numel():3d}/{N}",
              flush=True)

    # ---- verdict --------------------------------------------------------
    best = min(results, key=lambda k: results[k]["ade_0_2s"])
    b = results[best]
    syn = float(base[f"INCUMBENT: SYNTHETIC {N} (refcv3's actual)"]["ade"].mean())
    cvm = float(base["CONTROL: constant-velocity straight line"]["ade"].mean())
    ok = b["ade_0_2s"] < cvm
    print(f"\n=== VERDICT ===")
    print(f"  winner                       : {best}")
    print(f"  ADE 0-2 s                    : {b['ade_0_2s']:.4f} m "
          f"(ALONG {b['along_0_2s']:.4f} / LAT {b['lat_0_2s']:.4f})")
    print(f"  vs straight-line control     : {cvm:.4f} m  "
          f"({'BEATS' if ok else '⛔ LOSES TO'} it, "
          f"{cvm - b['ade_0_2s']:+.4f} m)")
    print(f"  vs published floor {STRAIGHT_LINE_FLOOR:.4f}   : "
          f"{'BEATS' if b['ade_0_2s'] < STRAIGHT_LINE_FLOOR else '⛔ LOSES'}")
    print(f"  vs refc-base target {DATA_FPS_TARGET:.4f}  : "
          f"{b['ade_0_2s'] / DATA_FPS_TARGET:.2f}x")
    print(f"  vs SYNTHETIC incumbent       : {syn:.4f} m -> "
          f"{syn / b['ade_0_2s']:.2f}x better, {syn - b['ade_0_2s']:+.4f} m")
    print(f"  GATE                         : "
          f"{'✅ PASS' if ok else '⛔ FAIL — DO NOT LAUNCH'}")

    A = built[best]
    torch.save(A, args.out)                     # ⛔ BARE TENSOR: refc_v3_train.py
                                                # does `torch.load(...).to(device)`
    meta = {
        "method": "fps", "metric": best, "horizons": list(H),
        "n_anchors": args.n_anchors, "pool_windows": int(Ptr.shape[0]),
        "pool_fps": int(pool.shape[0]), "train_clips": ntr, "eval_clips": nev,
        "eval_windows": int(Pev.shape[0]), "seed": args.seed,
        "source": f"v2-cache {args.train} (TRAIN split, clip-disjoint from "
                  f"{args.ev}: intersection 0)",
        "gate": {"ade_0_2s": b["ade_0_2s"], "along_0_2s": b["along_0_2s"],
                 "lat_0_2s": b["lat_0_2s"], "ade_0_6s": b["ade_0_6s"],
                 "straight_line_control": cvm, "synthetic_incumbent": syn,
                 "zero_path": float(base["CONTROL: zero path (no information)"]["ade"].mean()),
                 "pass": bool(ok)},
        "all_metrics": results,
        "sha_note": "bare [N,S,2] tensor; metadata in the .json sidecar",
        "built_s": round(time.time() - t0, 1),
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"\n[saved] {args.out}  shape {tuple(A.shape)}  "
          f"({time.time()-t0:.0f} s)\n[saved] {args.out}.json", flush=True)
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
