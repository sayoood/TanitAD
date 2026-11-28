#!/usr/bin/env python3
"""Build a LOCAL stand-in for /workspace/val40cache so `score_val40_lead.py` can be executed.

It writes `ep_%05d.pt` with the SAME shape the real cache has (a ToyEpisode carrying poses and the
`episode_id` build_episode mints) for R0 clips whose chunk has a local obstacle.offline zip — so
the runner exercises every step it will take on the eval host except the pod paths themselves.

`frames` is a 1-pixel dummy: the runner reads only `poses` and `episode_id`. That is the point —
it makes the script's real dependency surface visible.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
from tanitad.data.mixing import save_episode                      # noqa: E402
from tanitad.data.physicalai import TARGET_HZ, signals_at         # noqa: E402
from tanitad.data.toy_driving import ToyEpisode                   # noqa: E402

ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
OUT = Path(__file__).with_name("standin_valcache")
N_STACK = 3
N_EPS = 8


def main():
    OUT.mkdir(exist_ok=True)
    sel = pd.read_parquet(ROOT / "r0" / "r0_selection.parquet")
    sel["clip_id"] = sel["clip_id"].astype(str)
    chunk_of = dict(zip(sel["clip_id"], sel["chunk"].astype(int)))
    obs_ch = {int(p.name.split("_")[-1].split(".")[0])
              for p in (ROOT / "labels" / "obstacle.offline").glob("*.zip")}
    cam = {p.name.split(".")[0]: p for p in (ROOT / "r0" / "camera_front_wide").rglob("*.mp4")}
    cands = sorted(c for c in cam if chunk_of.get(c) in obs_ch)
    print(f"{len(cands)} R0 clips whose chunk has a local obstacle zip")
    n = 0
    for cid in cands:
        if n >= N_EPS:
            break
        ch = chunk_of[cid]
        ezp = ROOT / "labels" / "egomotion" / f"egomotion.chunk_{ch:04d}.zip"
        tsp = cam[cid].with_name(cam[cid].name.replace(".mp4", ".timestamps.parquet"))
        if not (ezp.exists() and tsp.exists()):
            continue
        with zipfile.ZipFile(ezp) as ez:
            m = next((x for x in ez.namelist()
                      if x.endswith(".parquet") and x.split("/")[-1].startswith(cid)), None)
            if m is None:
                continue
            ego = pd.read_parquet(io.BytesIO(ez.read(m)))
        ts = pd.read_parquet(tsp)
        tcol = next(c for c in ts.columns if "time" in c.lower())
        tf = ts[tcol].to_numpy(np.float64)
        span = tf[-1] - tf[0]
        unit = next((u for u in (1e9, 1e6, 1e3) if span / u > 1.0), 1.0)
        n_target = max(int(span / unit * TARGET_HZ), N_STACK + 1)
        tq = np.linspace(tf[0], tf[-1], n_target)
        actions, poses = signals_at(ego, tq)
        k = N_STACK - 1
        p = torch.from_numpy(poses[k:])
        ep = ToyEpisode(
            frames=torch.zeros((p.shape[0], 9, 1, 1), dtype=torch.uint8),
            actions=torch.from_numpy(actions[k:]), poses=p,
            episode_id=int.from_bytes(cid.encode()[:4].ljust(4, b"\0"), "big"),
            maneuvers=torch.full((p.shape[0],), -1, dtype=torch.long))
        save_episode(ep, OUT / f"ep_{n:05d}.pt")
        print(f"  ep_{n:05d}.pt  chunk {ch:04d}  T={p.shape[0]}")
        n += 1
    print(f"wrote {n} stand-in episodes to {OUT}")


if __name__ == "__main__":
    main()
