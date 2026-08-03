#!/usr/bin/env python3
"""LAN probes — two measurements, both cheap, both pre-registered.

``--agreement`` (0 GPU, seconds, runs on the dev box)
    Does the MAP-FREE route supplier (S1: the ego's own future path, arc-length
    resampled) agree with the MAP-BASED one (S2: lane centrelines + the lane
    graph)? S1 is the only supplier available on the parity corpus
    ``physicalai-train-e438721ae894``, which has no map. It is admissible as a
    stand-in for a real lane-graph route ONLY if the two agree — so this is the
    measurement that licenses the whole LAN label, and it runs on artifacts
    already banked in the repo (the NuRec ``map.xodr`` probe output).

``--navcf`` (minutes of GPU, forward passes only, NO training)
    The cheapest discriminating experiment for the C6 confound: sweep the
    existing 4-way ``nav_cmd`` over the val windows on a DEPLOYED REF-C
    checkpoint and measure how far the decoded trajectory moves. See
    ``tanitad.eval.route_cf.nav_cmd_sensitivity`` for the two pre-registered
    readings.

Usage
-----
    python scripts/lan_probe.py --agreement \
        --centerlines <lane_centerlines.json> --edges <lane_graph_edges.json> \
        --track <ego_track_map_frame.json> --out agreement.json

    python scripts/lan_probe.py --navcf --ckpt <refc.pt> --cache <val cache> \
        --out navcf.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tanitad.data.lan import (LaneCorridor, LanConfig,  # noqa: E402
                              horizon_lead_m, lan_from_future_path,
                              lan_from_polyline, route_agreement, yaw_from_path)


def _load_track(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``ego_track_map_frame.json`` -> (xy [T, 2], yaw [T] rad).

    ⚠️ Read with ``[]``, not ``.get()`` — a missing stamp must raise here, not
    silently become a default that a downstream metric then averages.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["track"] if isinstance(d, dict) else d
    xy = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    if rows and "yaw_deg" in rows[0]:
        yaw = np.radians([float(r["yaw_deg"]) for r in rows])
    else:
        yaw = np.array([yaw_from_path(xy, i) for i in range(len(rows))])
    return xy, yaw


def agreement(centerlines: Path, edges: Path, track: Path,
              cfg: LanConfig, stride: int = 1,
              max_snap_m: float = 5.0,
              max_heading_dev_deg: float | None = 60.0,
              hysteresis_m: float = 1.0) -> dict:
    """S1 (ego future) vs S2 (lane graph) LAN corridors, window by window."""
    corr = LaneCorridor.from_json(centerlines, edges)
    xy, yaw = _load_track(track)
    poly, seq, stats = corr.route_polyline(
        xy, max_snap_m=max_snap_m, heading=yaw,
        max_heading_dev_deg=max_heading_dev_deg, hysteresis_m=hysteresis_m)
    rows, per = [], []
    for i in range(0, len(xy) - 1, stride):
        # Leak guard uses the ego's own 2 s path length, exactly as the trainer
        # would: the same guard must apply to both suppliers or the comparison
        # is between different arc-length sets.
        fut = xy[i:]
        speed = float(np.linalg.norm(xy[min(i + 1, len(xy) - 1)] - xy[i])) * 10.0
        lead = horizon_lead_m(v0=speed, t_pred_s=2.0, cfg=cfg)
        s1 = lan_from_future_path(fut, xy[i], float(yaw[i]), cfg, lead)
        s2 = lan_from_polyline(poly, xy[i], float(yaw[i]), cfg, lead)
        a = route_agreement(s1, s2)
        a["i"] = int(i)
        rows.append(a)
        if a["n_compared"]:
            per.append(a)

    def _agg(key: str) -> dict:
        v = np.array([r[key] for r in per], dtype=np.float64)
        v = v[np.isfinite(v)]
        if not v.size:
            return {"n": 0}
        return {"n": int(v.size), "median": round(float(np.median(v)), 4),
                "mean": round(float(v.mean()), 4),
                "p90": round(float(np.percentile(v, 90)), 4),
                "max": round(float(v.max()), 4)}

    return {"probe": "lan_s1_vs_s2_agreement",
            "n_samples": len(rows),
            "n_compared_samples": len(per),
            "arclengths_m": list(cfg.arclengths_m),
            "min_lead_m": cfg.min_lead_m,
            "snap": {"max_snap_m": max_snap_m,
                     "max_heading_dev_deg": max_heading_dev_deg,
                     "hysteresis_m": hysteresis_m},
            "corridor": {"n_lanes": corr.n_lanes, "n_edges": len(corr.edges)},
            "route": {"lane_sequence_len": len(seq), **stats},
            "pos_l2_m": _agg("pos_l2_m"),
            "lat_delta_m": _agg("lat_delta_m"),
            "bearing_deg": _agg("bearing_deg"),
            "side_agree": _agg("side_agree"),
            "s1_valid_frac": _agg("a_valid_frac"),
            "s2_valid_frac": _agg("b_valid_frac"),
            "both_valid_frac": _agg("both_valid_frac"),
            "per_sample": rows}


def _navcf(args) -> dict:                       # pragma: no cover - needs a ckpt
    """nav_cmd counterfactual sweep on a deployed REF-C checkpoint."""
    import torch

    from tanitad.eval.route_cf import nav_cmd_sensitivity
    from tanitad.refs.refc import NAV_COMMANDS, RefCModel, refc_config

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = refc_config()
    model = RefCModel(cfg)
    model.load_state_dict(ck["model"])
    dev = args.device
    model.to(dev).eval()
    frames = torch.load(args.cache, map_location="cpu", weights_only=False)
    if isinstance(frames, dict):
        frames = frames["frames"]
    v0 = None if args.no_v0 else torch.zeros(frames.shape[0])

    def predict(nav_idx: int):
        outs = []
        with torch.no_grad():
            for s in range(0, frames.shape[0], args.batch):
                fr = frames[s:s + args.batch].to(dev)
                nav = torch.full((fr.shape[0],), int(nav_idx),
                                 dtype=torch.long, device=dev)
                ev = None if v0 is None else v0[s:s + args.batch].to(dev)
                outs.append(model(fr, nav_cmd=nav, v0=ev,
                                  steps=args.steps)["traj"].cpu().numpy())
        return np.concatenate(outs, axis=0)

    res = nav_cmd_sensitivity(predict, frames.shape[0], len(NAV_COMMANDS))
    pw = res.pop("per_window_max_m")
    res["per_window_max_m_stats"] = {
        "median": round(float(np.median(pw)), 6),
        "p90": round(float(np.percentile(pw, 90)), 6),
        "max": round(float(pw.max()), 6)}
    res["ckpt"] = str(args.ckpt)
    res["nav_commands"] = list(NAV_COMMANDS)
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agreement", action="store_true")
    p.add_argument("--navcf", action="store_true")
    p.add_argument("--centerlines", type=Path)
    p.add_argument("--edges", type=Path)
    p.add_argument("--track", type=Path)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-snap-m", type=float, default=5.0)
    p.add_argument("--max-heading-dev-deg", type=float, default=60.0)
    p.add_argument("--hysteresis-m", type=float, default=1.0)
    p.add_argument("--arclengths", type=float, nargs="+", default=None)
    p.add_argument("--min-lead-m", type=float, default=5.0)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--no-v0", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    cfg = LanConfig(
        arclengths_m=tuple(a.arclengths) if a.arclengths else LanConfig().arclengths_m,
        min_lead_m=a.min_lead_m)

    if a.agreement:
        if not (a.centerlines and a.edges and a.track):
            p.error("--agreement needs --centerlines --edges --track")
        res = agreement(a.centerlines, a.edges, a.track, cfg,
                        stride=a.stride, max_snap_m=a.max_snap_m,
                        max_heading_dev_deg=a.max_heading_dev_deg,
                        hysteresis_m=a.hysteresis_m)
    elif a.navcf:
        if not (a.ckpt and a.cache):
            p.error("--navcf needs --ckpt and --cache")
        res = _navcf(a)
    else:
        p.error("pick a probe: --agreement or --navcf")

    slim = {k: v for k, v in res.items() if k != "per_sample"}
    print(json.dumps(slim, indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[lan_probe] wrote {a.out}")
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
