"""PhysicalAI-AV R0 loader: contract, action derivation, clip splits —
synthetic fixtures only (no real clips in CI)."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tanitad.data._contract import assert_contract
from tanitad.data.physicalai import (WHEELBASE, build_episode,
                                     discover_r0_clips, signals_at,
                                     split_clips)

N = 200  # egomotion samples over a 20 s clip


def make_fake_r0(root: Path, clip_ids: list[str]) -> None:
    (root / "r0" / "camera_front_wide").mkdir(parents=True)
    (root / "labels" / "egomotion").mkdir(parents=True)
    sel = pd.DataFrame({"clip_id": clip_ids,
                        "chunk": [7] * len(clip_ids),
                        "urban_score": [2.0] * len(clip_ids)})
    sel.to_parquet(root / "r0" / "r0_selection.parquet")
    t = np.linspace(0.0, 20.0, N)
    ego = pd.DataFrame({
        "timestamp": t,
        "x": 8.0 * t, "y": np.zeros(N), "z": np.zeros(N),
        "vx": np.full(N, 8.0), "vy": np.zeros(N), "vz": np.zeros(N),
        "ax": np.zeros(N), "ay": np.zeros(N), "az": np.zeros(N),
        "qx": np.zeros(N), "qy": np.zeros(N), "qz": np.zeros(N),
        "qw": np.ones(N),
        "curvature": np.full(N, 0.01),        # gentle right-hand arc
    })
    with zipfile.ZipFile(root / "labels" / "egomotion"
                         / "egomotion.chunk_0007.zip", "w") as z:
        for cid in clip_ids:
            import io
            buf = io.BytesIO()
            ego.to_parquet(buf)
            z.writestr(f"{cid}.egomotion.parquet", buf.getvalue())
    for cid in clip_ids:
        base = root / "r0" / "camera_front_wide"
        (base / f"{cid}.camera_front_wide_120fov.mp4").write_bytes(b"")
        pd.DataFrame({"timestamp": np.linspace(0.0, 20.0, 600)}).to_parquet(
            base / f"{cid}.camera_front_wide_120fov.timestamps.parquet")


def fake_decode(mp4, size):
    return torch.randint(0, 255, (600, 3, size, size), dtype=torch.uint8)


def test_signals_derivation():
    t = np.linspace(0.0, 20.0, N)
    ego = pd.DataFrame({"timestamp": t, "vx": np.full(N, 8.0),
                        "vy": np.zeros(N), "x": 8.0 * t, "y": np.zeros(N),
                        "qx": np.zeros(N), "qy": np.zeros(N), "qz": np.zeros(N),
                        "qw": np.ones(N),               # identity quaternion -> yaw 0 (east)
                        "curvature": np.full(N, 0.01)})
    actions, poses = signals_at(ego, np.linspace(0.0, 20.0, 201))
    assert abs(actions[50, 0] - np.arctan(WHEELBASE * 0.01)) < 1e-6  # steer
    assert abs(actions[50, 1]) < 1e-4                                # const v
    assert abs(poses[50, 3] - 8.0) < 1e-6                            # v
    assert abs(poses[50, 2]) < 1e-6                                  # yaw east


def test_build_episode_contract_and_split(tmp_path):
    make_fake_r0(tmp_path, ["clipA", "clipB", "clipC", "clipD", "clipE"])
    clips = discover_r0_clips(tmp_path)
    assert len(clips) == 5
    ep = build_episode(clips[0], size=64, decode_fn=fake_decode)
    assert_contract(ep, channels=9)                                  # D-015
    assert ep.frames.shape[0] >= 150                                 # ~10 Hz * 20 s
    assert abs(float(ep.poses[:, 3].mean()) - 8.0) < 0.2
    tr, va = split_clips(clips, val_frac=0.2, seed=0)
    assert len(tr) == 4 and len(va) == 1                             # clip-level I3
