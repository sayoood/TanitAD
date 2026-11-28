#!/usr/bin/env python3
"""Build `win["lead"]` for the canonical 40-episode / 881-window val split — ON THOR.

This is the durability proof for `taniteval.lead_source`: it must reproduce the canonical
**881** windows and the published LEAD/NO_LEAD/NO_LABEL split on a host that is not the
Drive-backed checkout, from a 29 MB bundle rather than the 2.4 GB obstacle corpus.

Inputs (all local to the host):
  --episodes   the val40 episode dir (ep_00000.pt … ep_00039.pt)
  --manifest   manifest_EVALPOD_val40.json — supplies T and poses_sha256 per episode
  --bundle     val40_lead_bundle.zip — obstacle/<clip>.parquet + egomotion/<clip>.parquet
  --index      val40_lead_index.json — ep_XXXXX.pt -> clip_id

⚠️ INTEGRITY FIRST. Every episode's poses are sha256'd against the manifest BEFORE it is used.
A silently truncated transfer with exit 0 has bitten this programme repeatedly; a lead block built
on a truncated episode would still return perfectly plausible headway numbers.

⚠️ The eval grid is NOT 10 Hz. `register_poses_to_time` FITS the spacing (~0.1007 s); the realised
`b` is reported per episode so an assumed 0.1 would be visible rather than silent.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/nvidia/TanitAD/taniteval")
sys.path.insert(0, "/home/nvidia/TanitAD/stack/scripts")

from taniteval.lead_source import (LEAD, NO_LABEL, NO_LEAD,  # noqa: E402
                                   RegistrationError, lead_block,
                                   register_poses_to_time, window_last_indices)
from taniteval.lead_source import VEHICLE_CLASSES  # noqa: E402


def quaternion_yaw(qx, qy, qz, qw):
    """Same convention as `tanitad.data.physicalai.quaternion_yaw` / `lead_state_gate`."""
    return np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def ego_dict(ego: pd.DataFrame) -> dict:
    t = ego["timestamp"].to_numpy(np.float64) / 1e6
    o = np.argsort(t)
    g = lambda c: ego[c].to_numpy(np.float64)[o]  # noqa: E731
    return {"t": t[o], "x": g("x"), "y": g("y"),
            "yaw": np.unwrap(quaternion_yaw(g("qx"), g("qy"), g("qz"), g("qw"))),
            "v": np.hypot(g("vx"), g("vy"))}


def obs_dict(obs: pd.DataFrame) -> dict:
    return {"t": obs["timestamp_us"].to_numpy(np.float64) / 1e6,
            "track": obs["track_id"].astype(str).to_numpy(object),
            "center_x": obs["center_x"].to_numpy(np.float64),
            "center_y": obs["center_y"].to_numpy(np.float64),
            "size_x": obs["size_x"].to_numpy(np.float64),
            "is_vehicle": obs["label_class"].isin(VEHICLE_CLASSES).to_numpy(bool)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="/home/nvidia/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--manifest", default="/home/nvidia/leadwork/manifest_EVALPOD_val40.json")
    ap.add_argument("--bundle", default="/home/nvidia/leadwork/val40_lead_bundle.zip")
    ap.add_argument("--index", default="/home/nvidia/leadwork/val40_lead_index.json")
    ap.add_argument("--out", default="/home/nvidia/leadwork/val40_lead_blocks.pt")
    ap.add_argument("--report", default="/home/nvidia/leadwork/val40_lead_report.json")
    ap.add_argument("--horizons-s", default="0.5,1.0,1.5,2.0")
    args = ap.parse_args()

    ts_rel = np.array([float(x) for x in args.horizons_s.split(",")])
    man = {e["file"]: e for e in json.loads(Path(args.manifest).read_text())["episodes"]}
    idx = json.loads(Path(args.index).read_text())
    zf = zipfile.ZipFile(args.bundle)

    rows, blocks = [], {}
    tot = {LEAD: 0, NO_LEAD: 0, NO_LABEL: 0}
    n_win_total = 0
    for ep in sorted(man):
        m, info = man[ep], idx[ep]
        clip = info["clip_id"]
        d = torch.load(Path(args.episodes) / ep, map_location="cpu", weights_only=False)
        poses = d["poses"].numpy().astype(np.float64)
        T = int(poses.shape[0])
        # --- integrity: T and the poses hash the manifest committed -------------------
        sha = hashlib.sha256(np.ascontiguousarray(
            d["poses"].numpy().astype(np.float32)).tobytes()).hexdigest()
        ok_sha = (sha == m["poses_sha256"])
        ok_T = (T == int(m["T"]))
        widx = window_last_indices(T)
        n_win_total += len(widx)

        r = {"ep": ep, "clip_sha8": hashlib.sha256(clip.encode()).hexdigest()[:8],
             "T": T, "T_manifest": int(m["T"]), "T_match": ok_T,
             "poses_sha256_match": ok_sha, "n_windows": int(len(widx))}

        me = f"egomotion/{clip}.parquet"
        mo = f"obstacle/{clip}.parquet"
        if me not in zf.namelist():
            r.update(status="NO_EGOMOTION")
            rows.append(r)
            continue
        ego = ego_dict(pd.read_parquet(io.BytesIO(zf.read(me))))
        try:
            reg = register_poses_to_time(poses[:, :2], ego["t"], ego["x"], ego["y"])
        except RegistrationError as e:
            r.update(status="REGISTRATION_FAILED", error=str(e)[:220])
            rows.append(r)
            continue
        r.update(status="OK", grid_dt_s=reg["grid_dt_s"],
                 residual_median_m=reg["residual_m"]["median"],
                 residual_p95_m=reg["residual_m"]["p95"],
                 n_inlier=reg["n_inlier"], n_probe=reg["n_probe"])
        t0s = reg["t_s"][widx]

        obs = None
        if mo in zf.namelist():
            obs = obs_dict(pd.read_parquet(io.BytesIO(zf.read(mo))))
            r["obstacle"] = "present"
            r["n_cuboids"] = int(obs["t"].size)
        else:
            r["obstacle"] = "ABSENT"
            r["n_cuboids"] = 0

        blk = lead_block(t0s, ts_rel, obs, ego)
        for k in tot:
            tot[k] += blk["counts"][k]
        r["counts"] = blk["counts"]
        r["label_span_s"] = blk["label_span_s"]
        g0 = blk["gap0_m"][np.isfinite(blk["gap0_m"])]
        r["gap0_median_m"] = round(float(np.median(g0)), 4) if g0.size else None
        blocks[ep] = {k: v for k, v in blk.items() if k != "conventions"}
        rows.append(r)
        print(f"{ep} T={T} win={len(widx):3d} dt={r.get('grid_dt_s')} "
              f"res={r.get('residual_median_m')} {blk['counts']}", flush=True)

    report = {
        "host": "tanitad-thor (aarch64, NVIDIA Thor)",
        "episodes_dir": args.episodes,
        "n_episodes": len(rows),
        "TOTAL_WINDOWS": n_win_total,
        "canonical_881": n_win_total == 881,
        "counts": tot,
        "integrity": {
            "T_match_all": all(r["T_match"] for r in rows),
            "poses_sha256_match_all": all(r["poses_sha256_match"] for r in rows),
            "n_poses_sha_mismatch": sum(1 for r in rows if not r["poses_sha256_match"]),
        },
        "registration": {
            "n_ok": sum(1 for r in rows if r.get("status") == "OK"),
            "n_failed": sum(1 for r in rows if r.get("status") == "REGISTRATION_FAILED"),
            "grid_dt_s_min": min((r["grid_dt_s"] for r in rows if "grid_dt_s" in r), default=None),
            "grid_dt_s_max": max((r["grid_dt_s"] for r in rows if "grid_dt_s" in r), default=None),
            "residual_median_m_max": max((r["residual_median_m"] for r in rows
                                          if "residual_median_m" in r), default=None),
        },
        "episodes_without_obstacle_labels": [r["ep"] for r in rows
                                             if r.get("obstacle") == "ABSENT"],
        "per_episode": rows,
        "horizons_s": ts_rel.tolist(),
    }
    Path(args.report).write_text(json.dumps(report, indent=1))
    torch.save(blocks, args.out)
    print("\n" + json.dumps({k: v for k, v in report.items() if k != "per_episode"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
