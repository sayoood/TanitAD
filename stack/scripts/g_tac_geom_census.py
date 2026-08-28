"""Census `tanitad.data.g_tac_geom` over a real episode cache.

⭐ **WHY THIS SCRIPT IS PART OF THE DELIVERABLE.**
``g_tac_geom.LAT_OFFSET_MIN_M`` is **DECLARED, NOT CALIBRATED** — 0.75 m chosen
because a lane is ~3.5 m and a nudge ~0.5-1.0 m. *Plausible* is exactly what
killed ``LANE_TARGET``: the PI adjudicated 14 of 18 of its labels wrong because a
threshold that looked reasonable fired on ordinary road curvature. ⇒ the
threshold must be read off the **null distribution of real driving**, and this is
the measurement that reads it.

⛔ **THE CONTROLS ARE MANDATORY AND THEY PRINT.** Four estimator failures on
2026-08-22 each produced a confident wrong number and each was caught ONLY by a
control that had to read a known value. This script always emits:

* ``control_naive_gate`` — the REFUTED ``LANE_TARGET`` gate (raw ego-frame
  lateral displacement over the same band) run on the SAME windows. It must fire
  far more often than the curvature-relative deriver; if the two agree, the
  reference is not cancelling curvature and the whole module is void.
* ``control_constant_only`` — a constant emitter's rate. The deriver must beat
  the majority-class share, or it is a constant wearing a label's clothes.
* ``control_synthetic_bend`` — an analytic constant-radius bend, in-process. Must
  read ~0 offset. If this drifts, the corpus numbers below are unreadable.
* ``degenerate_lat`` / ``degenerate_lon`` from ``g_tac_geom.census``.

Every output carries ``n`` (windows) and ``d`` (episodes).

⚠️ **PARITY.** This script SELECTS NO EPISODES for training — it reads whatever
cache it is pointed at and reports its key. It cannot break parity because it
produces a distribution, not a corpus. The cache key is recorded in the output so
a threshold set from a non-parity cache is never mistaken for one set on parity.

Usage::

    python stack/scripts/g_tac_geom_census.py \
        --cache C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-train-14231cd29c74 \
        --stride 5 --out census.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO / "stack"), str(REPO / "stack" / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from refb_labels import ego_frame, path_curvature                  # noqa: E402

from tanitad.data import g_tac_geom as G                           # noqa: E402
from tanitad.models.v6 import DT, TAC_BAND_S                       # noqa: E402


def naive_lateral_gate(poses: torch.Tensor, t: int, thresh: float,
                       band_s=TAC_BAND_S, dt: float = DT) -> bool | None:
    """The REFUTED gate: raw ego-frame |dy| over the band, no curvature frame.

    This is what ``LANE_TARGET`` used and what the PI ruled out on 2026-08-16.
    Reproduced here ONLY as the control the new deriver must beat."""
    t_end = t + int(round(band_s[1] / dt))
    if t_end > poses.shape[0] - 1:
        return None
    dxy = (poses[t_end, :2] - poses[t, :2]).unsqueeze(0)
    e = ego_frame(dxy, poses[t, 2].unsqueeze(0))
    return abs(float(e[0, 1])) >= thresh


def oracle_circle_residual(poses: torch.Tensor, t: int,
                           band_s=TAC_BAND_S, dt: float = DT):
    """⭐ THE CEILING CONTROL, and the measurement that REFUTED the LAT axis.

    Fit the reference circle's curvature to **the very future path it is about
    to measure** — an oracle no deployable deriver can have. The residual is
    therefore the BEST any circular corridor can do. If the oracle residual
    already exceeds the lane-scale threshold, the failure is the MODEL CLASS and
    no amount of estimator work can rescue it.

    Returns ``(theta, residual_m)`` or ``None``.
    """
    t_end = t + int(round(band_s[1] / dt))
    if t_end > poses.shape[0] - 1:
        return None
    sub = poses[t:t_end + 1]
    kap = path_curvature(sub)
    seg = (sub[1:, :2] - sub[:-1, :2]).norm(dim=-1)
    tot = float(seg.sum())
    if tot < 1e-3:
        return None
    k_oracle = float((kap * seg).sum() / tot)
    rx, ry, rth = G.constant_curvature_point(k_oracle, tot)
    dxy = (poses[t_end, :2] - poses[t, :2]).unsqueeze(0)
    e = ego_frame(dxy, poses[t, 2].unsqueeze(0))
    ax, ay = float(e[0, 0]), float(e[0, 1])
    resid = abs((ax - rx) * (-math.sin(rth)) + (ay - ry) * math.cos(rth))
    return abs(k_oracle) * tot, resid


def _quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    q = {f"p{p}": s[min(int(p / 100.0 * len(s)), len(s) - 1)]
         for p in (50, 75, 90, 95, 99)}
    q["max"], q["mean"], q["n"] = s[-1], sum(s) / len(s), len(s)
    return q


def synthetic_bend_control(radius: float = 100.0, v: float = 20.0) -> dict:
    """In-process analytic control: a constant-radius bend must read ~0."""
    kap, T = 1.0 / radius, 200
    s = torch.arange(T, dtype=torch.float64) * v * DT
    th = kap * s
    r = 1.0 / kap
    poses = torch.stack([r * torch.sin(th), r * (1.0 - torch.cos(th)), th,
                         torch.full((T,), v, dtype=torch.float64)], dim=-1)
    lab = G.g_tac_geom(poses, 20, lat_arm="refuted-diagnostic")
    return {"radius_m": radius, "v_ms": v,
            "lat_offset_m": lab.audit["lat_offset_m"],
            "token": lab.lat.token,
            "naive_gate_would_fire": naive_lateral_gate(poses, 20, 0.75),
            "PASS": lab.lat.token == "LAT_UNCONSTRAINED"
                    and abs(lab.audit["lat_offset_m"]) < 0.10}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True,
                    help="episode cache dir holding ep_*.pt")
    ap.add_argument("--stride", type=int, default=5,
                    help="window stride in frames; consecutive windows overlap "
                         "5.9 of 6.0 s, so stride 1 over-counts correlated "
                         "samples. Reported in the output.")
    ap.add_argument("--limit-episodes", type=int, default=0)
    ap.add_argument("--glob", default="ep_*.pt",
                    help="episode filename pattern. The dev-box cache uses "
                         "'ep_*.pt'; the Thor w120 parity corpus uses "
                         "'*.v2ep.pt'. Both carry poses [T, 4].")
    ap.add_argument("--lat-arm", default="refuted-diagnostic",
                    choices=("abstain", "refuted-diagnostic"),
                    help="the census DEFAULTS to the refuted arm because its "
                         "job is to REPRODUCE the refutation; 'abstain' is what "
                         "label production uses.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    cache = Path(args.cache)
    eps = sorted(cache.glob(args.glob))
    if args.limit_episodes:
        eps = eps[:args.limit_episodes]
    if not eps:
        raise SystemExit(f"no {args.glob} under {cache}")

    band_end = int(round(TAC_BAND_S[1] / DT))
    labels, naive_fire, naive_n = [], 0, 0
    oracle: list[tuple[float, float]] = []     # (theta, residual) — the CEILING
    ep_ids, t0 = [], time.time()

    for i, f in enumerate(eps):
        d = torch.load(f, map_location="cpu", weights_only=False)
        poses = d["poses"].double()
        ep_ids.append(int(d.get("episode_id", -1)))
        T = poses.shape[0]
        for t in range(0, max(T - band_end, 0), args.stride):
            labels.append(G.g_tac_geom(poses, t, lat_arm=args.lat_arm))
            g = naive_lateral_gate(poses, t, G.LAT_OFFSET_MIN_M)
            if g is not None:
                naive_n += 1
                naive_fire += int(g)
            o = oracle_circle_residual(poses, t)
            if o is not None:
                oracle.append(o)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(eps)} episodes, {len(labels)} windows, "
                  f"{time.time()-t0:.0f}s", flush=True)

    cen = G.census(labels)
    cen["d_episodes"] = len(eps)
    cen["cache_key"] = cache.name
    cen["stride_frames"] = args.stride
    cen["band_s"] = list(TAC_BAND_S)
    cen["episode_ids_sample"] = ep_ids[:5]
    cen["glob"] = args.glob
    cen["elapsed_s"] = round(time.time() - t0, 1)

    # ---- CONTROLS ---------------------------------------------------------- #
    n = cen["n_windows"]
    lat_major = max(cen["lat"].values()) if cen["lat"] else 0
    lon_major = max(cen["lon"].values()) if cen["lon"] else 0
    cen["control_constant_only"] = {
        "lat_majority_token": max(cen["lat"], key=cen["lat"].get) if cen["lat"] else None,
        "lat_majority_share": round(lat_major / n, 4) if n else None,
        "lon_majority_token": max(cen["lon"], key=cen["lon"].get) if cen["lon"] else None,
        "lon_majority_share": round(lon_major / n, 4) if n else None,
        "NOTE": "a deriver whose majority share is ~1.0 is a constant"}
    cen["control_naive_gate"] = {
        "n": naive_n, "fired": naive_fire,
        "fire_rate": round(naive_fire / naive_n, 4) if naive_n else None,
        "curvature_relative_fire_rate":
            round(cen["lat"].get("CORRIDOR_OFFSET", 0) / n, 4) if n else None,
        "NOTE": "the REFUTED LANE_TARGET gate on the SAME windows; it must fire "
                "MUCH more often, or the curvature frame is not cancelling "
                "anything"}
    cen["control_synthetic_bend"] = [synthetic_bend_control(r, v)
                                     for r, v in ((100.0, 20.0), (300.0, 30.0))]
    cen["lat_arm"] = args.lat_arm

    # ---- ⭐ THE CEILING CONTROL — what refuted the LAT axis ----------------- #
    bands = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.50),
             (0.50, 1.00), (1.00, 1e9)]
    strat = []
    for lo, hi in bands:
        sel = [r for th, r in oracle if lo <= th < hi]
        q = _quantiles(sel)
        strat.append({"theta_lo": lo, "theta_hi": None if hi > 1e8 else hi,
                      "n": len(sel), "share": round(len(sel) / max(len(oracle), 1), 4),
                      "oracle_resid": q,
                      "ceiling_clears_lane_bar":
                          bool(q) and q["p50"] < G.LAT_OFFSET_MIN_M})
    cen["control_oracle_circle_ceiling"] = {
        "all": _quantiles([r for _, r in oracle]),
        "by_theta": strat,
        "lane_bar_m": G.LAT_OFFSET_MIN_M,
        "NOTE": "curvature fitted to the very future path measured — the BEST "
                "any circular corridor can do. A p50 above the lane bar means "
                "the MODEL CLASS fails, not the estimator."}

    Path(args.out).write_text(json.dumps(cen, indent=2), encoding="utf-8")

    # ---- report ------------------------------------------------------------ #
    print(f"\n=== g_tac geometry census — n={n} windows, d={len(eps)} episodes, "
          f"stride={args.stride}, cache={cache.name} ===")
    print("\nLAT:")
    for k, v in sorted(cen["lat"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v:7d}  {v/n*100:6.2f} %")
    print("LON:")
    for k, v in sorted(cen["lon"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:22s} {v:7d}  {v/n*100:6.2f} %")
    q = cen["abs_lat_offset_m"]
    if q:
        print(f"\n|lat_offset| (m), n={cen['n_with_offset']}:")
        print("  " + "  ".join(f"{k}={v:.3f}" for k, v in q.items()))
    c = cen["control_naive_gate"]
    print(f"\nCONTROL naive (refuted) gate: fired {c['fired']}/{c['n']} = "
          f"{(c['fire_rate'] or 0)*100:.2f} %   vs curvature-relative "
          f"{(c['curvature_relative_fire_rate'] or 0)*100:.2f} %")
    oc = cen["control_oracle_circle_ceiling"]
    print("\nCONTROL oracle circle = the CEILING any circular corridor can reach")
    print(f"  overall: p50={oc['all'].get('p50', 0):.3f} m  "
          f"p90={oc['all'].get('p90', 0):.3f} m  n={oc['all'].get('n', 0)}   "
          f"(lane bar {oc['lane_bar_m']} m)")
    print(f"  {'theta band':>16} {'share':>7} {'n':>7} {'oracle p50':>11}  clears lane bar?")
    for b in oc["by_theta"]:
        hi = "inf" if b["theta_hi"] is None else f"{b['theta_hi']:.2f}"
        print(f"  [{b['theta_lo']:.2f},{hi:>5}) {b['share']*100:6.1f}% {b['n']:7d} "
              f"{b['oracle_resid'].get('p50', float('nan')):11.3f}  "
              f"{'YES' if b['ceiling_clears_lane_bar'] else 'no'}")
    for b in cen["control_synthetic_bend"]:
        print(f"CONTROL synthetic bend R={b['radius_m']:.0f} v={b['v_ms']:.0f}: "
              f"offset={b['lat_offset_m']:.5f} m token={b['token']} "
              f"-> {'PASS' if b['PASS'] else 'FAIL'}")
    print(f"CONTROL constant-only: lat majority "
          f"{cen['control_constant_only']['lat_majority_share']}, lon majority "
          f"{cen['control_constant_only']['lon_majority_share']}")
    print(f"degenerate_lat={cen['degenerate_lat']} "
          f"degenerate_lon={cen['degenerate_lon']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
